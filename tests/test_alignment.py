"""Trying to make sure the spike alignment is perfect at every step.

## Testing subtraction

The idea is to have a template like this
         -^-
        |   |
    ____|   |_______
There are 3 "on" samples and the rest are 0. The peak is in the center,
but it's a bit small relative to the neighbors, so that the noise can
easily cause the peak to slide around +/- 1 sample.

This can be done on multiple channels as well.

Another template can be the negative version.

We can have a perfect linear denoiser: take the template and versions
shifted left/right by 1 sample, and Gram-Schmidt them. If you have the
negative version in there then it'll be denoised fine by this basis.

Now that optimal denoiser can be used during subtraction, and denoiser
realignment should be perfect. No-denoiser alignment should fail.

## Testing templates

First off, the time_shifts property assigned by the previous subtractor
should lead to perfect alignment, so that templates computed with no
realignment from ^ spike train should be perfect.

They should not be perfect withouth that.

Need to also test per-template shifts.
"""

import tempfile
from typing import cast

import numpy as np
import pytest
import torch

import dartsort
from dartsort.util.spiketorch import taper
from dartsort.util.testing_util import matching_debug_util

# constants
fs = 1000
t = 7
trough = 3
snr = 20.0
bump = 0.2

waveform_cfg = dartsort.WaveformConfig(ms_before=trough, ms_after=t - trough)


@pytest.fixture(scope="module")
def align_templates():
    templates = np.zeros((2, t, 1), dtype=np.float32)
    templates[0, trough - 1 : trough + 2, :] = snr
    assert templates.sum() == 3 * snr
    templates[0, trough] += bump
    templates[1] = -templates[0]

    return dartsort.TemplateData(
        templates=templates,
        unit_ids=np.arange(2),
        spike_counts=np.ones(2),
        trough_offset_samples=trough,
        registered_geom=np.zeros((1, 2)),
    )


@pytest.fixture(scope="module")
def align_sim(tmp_path_factory, align_templates):
    res = dartsort.simkit.generate_simulation(
        n_units=2,
        folder=tmp_path_factory.mktemp("ds_align_sim"),
        noise_recording_folder=tmp_path_factory.mktemp("ds_align_sim_noise"),
        noise_kind="white",
        templates_kind="static",
        template_simulator_kwargs=dict(template_data=align_templates),
        temporal_jitter=1,
        amplitude_jitter=0.0,
        duration_seconds=100.0,
        min_fr_hz=50.0,
        featurization_cfg=dartsort.skip_featurization_cfg,
        max_fr_hz=55.0,
        noise_in_memory=True,
        recording_dtype="float32",
        sampling_frequency=fs,
        globally_refractory=True,
        refractory_ms=t,
        geom=align_templates.registered_geom,
        common_reference=False,
        save_noise_waveforms=True,
    )
    res = cast(dict, res)

    # for the analysis below to work with high probability, we need
    # standard error in both units small relative to bump. (that way
    # the sample means will have accurate extrema.)
    assert res["sorting"].labels is not None
    se = 1.0 / np.sqrt(np.unique(res["sorting"].labels, return_counts=True)[1])
    assert (se < 0.1 * bump).all()

    return res


def test_denoiser_alignment(align_sim, align_templates):
    t0 = align_templates.templates[0, :, 0]
    ci = dartsort.util.waveform_util.full_channel_index(1, to_torch=True)
    rec = align_sim["recording"]
    gt_st = align_sim["sorting"]
    noise_wfs = gt_st._load_dataset("noise_waveforms")

    # an optimal linear denoiser
    rolls = (-1, 0, 1)
    data = np.array([np.roll(t0, r, axis=0) for r in rolls])
    assert data.shape == (len(rolls), *t0.shape)
    data = torch.asarray(data, dtype=torch.float)
    targ_peak = torch.tensor(rolls) + trough
    assert torch.equal(data.argmax(dim=1), targ_peak)
    pca = dartsort.transform.DebugMatchingPursuitDenoiser(ci, basis=data)

    # run subtraction
    denoiser = dartsort.WaveformPipeline([pca])
    transformers = [
        dartsort.transform.Waveform(
            name_prefix="collisioncleaned",
            channel_index=ci,
            spike_length_samples=t0.shape[0],
        ),
        dartsort.transform.Voltage(),
    ]
    featurizer = dartsort.WaveformPipeline(transformers)
    peelers = [
        dartsort.SubtractionPeeler(
            recording=rec,
            channel_index=ci,
            subtraction_denoising_pipeline=denoiser,
            featurization_pipeline=featurizer,
            trough_offset_samples=trough,
            spike_length_samples=t,
            realign_to_denoiser=rtd,
            denoiser_realignment_shift=3,
            detection_threshold=15.0,
            relative_peak_radius_samples=1,
            temporal_dedup_radius_samples=3,
            positive_temporal_dedup_radius_samples=0,
        )
        for rtd in (False, True)
    ]
    with tempfile.TemporaryDirectory() as tdir:
        tdir = dartsort.resolve_path(tdir)
        st0, st1 = sts = [
            dartsort.DARTsortSorting.from_peeling_hdf5(
                p.peel(tdir / "hi.h5", overwrite=True)
            )
            for p in peelers
        ]
        wf0 = st0._load_dataset("collisioncleaned_waveforms")
        wf1 = st1._load_dataset("collisioncleaned_waveforms")

    # -- denoiser does alignment right
    # both have right count
    assert st0.times_samples.shape == gt_st.times_samples.shape
    assert st1.times_samples.shape == gt_st.times_samples.shape

    # both have voltage features
    assert all(getattr(st, "voltages", None) is not None for st in sts)

    # they have or don't have shifts
    assert not hasattr(st0, "time_shifts")
    assert hasattr(st1, "time_shifts")

    # st0 is bad
    assert not np.array_equal(st0.times_samples, gt_st.times_samples)

    # st1 is perfect
    assert np.array_equal(st0.times_samples, st1.times_samples - st1.time_shifts)  # type: ignore
    assert np.array_equal(st1.times_samples, gt_st.times_samples)

    # -- the template computation handles time_shifts correctly
    # to do this, we need to cluster the detections. we can just cheat and use the sign.
    # unit 0 had positive sign, hence the <
    sts = [
        st.ephemeral_replace(labels=(st.voltages < 0).astype(int))  # type: ignore
        for st in sts
    ]
    assert len(sts) == 2

    # get templates
    (pst0, tr0), (pst1, tr1) = [
        dartsort.postprocess(
            rec, st, template_cfg=dartsort.raw_template_cfg, waveform_cfg=waveform_cfg
        )
        for st in sts
    ]

    # postprocess realigns the spike trains
    assert np.array_equal(pst0.times_samples, st0.times_samples)
    assert np.array_equal(pst1.times_samples, gt_st.times_samples)

    # st1 templates are aligned
    assert np.all(np.abs(tr1.templates[:, :, 0]).argmax(1) == trough)

    # st0 templates are worse than st1. they could still be aligned
    # so wont test that but they should be noisier.
    d0 = np.sqrt(np.square(align_templates.templates - tr0.templates).mean())
    d1 = np.sqrt(np.square(align_templates.templates - tr1.templates).mean())
    assert d0 > d1
    # d1 should match based on standard error bound whp
    assert d1 < 3 * 0.1 * bump

    # actually, st0 templates will be smoothed out into triangles.
    # the noisier the more so. worth testing to make sure the noise
    # is aggro enough relative to bump. here 1.4 means the triangle
    # is quite steep
    peaks = np.abs(tr0.templates[:, trough, 0])
    assert np.all(peaks > 1.4 * np.abs(tr0.templates[:, :trough, 0]).max(1))
    assert np.all(peaks > 1.4 * np.abs(tr0.templates[:, trough + 1 :, 0]).max(1))

    # test alignment of cc waveforms w and w/o time shift
    assert wf0.shape == wf1.shape == (gt_st.n_spikes, t0.shape[0], 1)
    true_wfs = align_templates.templates[gt_st.labels, 1:-1]
    assert np.all(np.abs(true_wfs[:, :, 0]).argmax(1) == trough - 1)
    time_ixs = np.arange(1, t0.shape[0] - 1) + st1.time_shifts[:, None]  # type: ignore
    time_ixs = time_ixs[:, :, None]
    wf1_sliced = np.take_along_axis(wf1, indices=time_ixs, axis=1)
    np.testing.assert_allclose(wf1_sliced, true_wfs, atol=snr / 2)
    true_cc_wfs = true_wfs + noise_wfs[:, 1:-1]
    np.testing.assert_allclose(wf1_sliced, true_cc_wfs)
    assert not np.allclose(wf0[:, 1:-1], true_wfs, atol=snr / 2)


def template_makers(rec, st, align=True, align_max=0):
    t0 = dartsort.TemplateData.from_config(
        rec,
        st,
        template_cfg=dartsort.TemplateConfig(
            realign_peaks=align,
            denoising_method="none",
            realign_shift_ms=align_max,
            algorithm="by_chunk",
            spikes_per_unit=10000,
        ),
        waveform_cfg=waveform_cfg,
    )
    t1 = dartsort.TemplateData.from_config(
        rec,
        st,
        template_cfg=dartsort.TemplateConfig(
            realign_peaks=align,
            denoising_method="none",
            realign_shift_ms=align_max,
            algorithm="by_unit",
            spikes_per_unit=10000,
        ),
        waveform_cfg=waveform_cfg,
    )
    t2 = dartsort.TemplateData.from_config(
        rec,
        st,
        template_cfg=dartsort.TemplateConfig(
            realign_peaks=align,
            denoising_method="none",
            realign_shift_ms=align_max,
            reduction="median",
            spikes_per_unit=10000,
        ),
        waveform_cfg=waveform_cfg,
    )
    _, t3 = dartsort.postprocess(
        rec,
        st,
        template_cfg=dartsort.TemplateConfig(
            realign_peaks=align,
            denoising_method="none",
            realign_shift_ms=align_max,
            spikes_per_unit=10000,
        ),
        waveform_cfg=waveform_cfg,
    )
    return [t0, t1, t2, t3]


@pytest.mark.parametrize("with_spike_shifts", (False, True))
@pytest.mark.parametrize("with_unit_shifts", (False, True))
@pytest.mark.parametrize("seed", (0, 1, 64))
@pytest.mark.parametrize("align_max", (1, trough, t, 2 * t))
def test_template_shifts(
    align_sim, align_templates, with_spike_shifts, with_unit_shifts, seed, align_max
):
    rec = align_sim["recording"]
    gt_st = align_sim["sorting"]
    rg = np.random.default_rng(seed)

    # make fake spike trains with time_shifts and per-unit shifts
    times = gt_st.times_samples.copy()
    labels = gt_st.labels
    if with_spike_shifts and seed == 0:
        time_shifts = np.array([-align_max, align_max])[labels]
        times -= time_shifts
    elif with_spike_shifts:
        time_shifts = rg.integers(-align_max, align_max + 1, size=times.shape)
        times -= time_shifts
    else:
        time_shifts = np.zeros_like(times)
    if with_unit_shifts and seed == 0:
        ushifts = np.array([-align_max, align_max])
        times += ushifts[labels]
    elif with_unit_shifts:
        ushifts = rg.integers(-align_max, align_max + 1, size=2)
        times += ushifts[labels]
    else:
        ushifts = np.zeros(2, dtype=np.int64)

    st = dartsort.DARTsortSorting(
        times_samples=times + time_shifts,
        channels=gt_st.channels,
        labels=labels,
        sampling_frequency=gt_st.sampling_frequency,
        ephemeral_features=dict(time_shifts=time_shifts),
    )
    if align_max < t:
        temps0 = template_makers(rec, st, align=False, align_max=align_max)
    else:
        # they can scoot right out of the window and cause errors.
        temps0 = []
    temps1 = template_makers(rec, st, align=True, align_max=align_max)

    # which ones should or should not be aligned?
    if with_unit_shifts:
        tempsa = temps1
    else:
        tempsa = temps0 + temps1

    # test all of the template constructors including postprocess, using the auto
    # realignment. test that not using the auto realignment fails (if with_unit_shifts)
    # and that the resulting misalignment is as expected.

    for tt in tempsa:
        assert (np.abs(tt.templates[:, :, 0]).argmax(1) == trough).all()

        # test (in cases where result should be correct) that se bound is met.
        d = np.sqrt(np.square(align_templates.templates - tt.templates).mean())
        if d >= 3 * 0.1 * bump:
            import matplotlib.pyplot as plt

            for ttt in tt.templates[:, :, 0]:
                plt.plot(ttt)
            plt.show()
        assert d < 3 * 0.1 * bump

    # what is the expected misalignment? this one i have gotten wrong many times.
    # if the spike times for a unit are 1 larger than they should be (ushift=1),
    # then the spike-triggered average will be tuned too late, and the peak will
    # be a sample EARLY.

    for tt in temps0:
        assert (np.abs(tt.templates[:, :, 0]).argmax(1) == trough - ushifts).all()


@pytest.mark.parametrize("matchtype", ["individual_compressed_upsampled", "drifty"])
def test_matching_alignment_basic(align_sim, align_templates, matchtype):
    with tempfile.TemporaryDirectory() as tdir:
        st = dartsort.match(
            tdir,
            align_sim["recording"],
            template_data=align_templates,
            waveform_cfg=waveform_cfg,
            featurization_cfg=dartsort.FeaturizationConfig(skip=True),
            matching_cfg=dartsort.MatchingConfig(
                refractory_radius_frames=1,
                template_temporal_upsampling_factor=1,
                template_type=matchtype,
            ),
        )
    gt_st = align_sim["sorting"]
    assert np.array_equal(st.times_samples, gt_st.times_samples)
    assert st.labels is not None
    assert np.array_equal(st.labels, gt_st.labels)


@pytest.mark.parametrize("tempkind", ["exp", "parabola"])
@pytest.mark.parametrize(
    "matchtype", ["debug", "individual_compressed_upsampled", "drifty"]
)
@pytest.mark.parametrize("up_factor", (1, 2, 4, 8))
def test_matching_alignment_upsampled(up_factor, matchtype, tempkind):
    # we'll have to use a smoother template library here
    if tempkind == "exp":
        trough_offset = 42
        # here is a nice smooth basic shape
        tleft = np.linspace(1.0, 0.0, num=42, endpoint=False)
        tright = np.linspace(0.0, 1.0, num=79)
        tt = np.concatenate([tleft, tright])
        assert tt.argmin() == trough_offset
        assert tt.shape == (121,)
        tshape = np.exp(-np.square(tt / 0.3)).astype(np.float32)
        tshape = snr * taper(torch.tensor(tshape), dim=0).numpy()
    elif tempkind == "parabola":
        # very smooth, reconstructed perfectly by cubic interpolators
        # have to put trough in the center bc parabola.
        trough_offset = 60
        tt = np.linspace(-1.0, 1.0, num=121, endpoint=True)
        tshape = -(tt**2)
        tshape -= tshape.min()
        assert tt.shape == tshape.shape == (121,)
    else:
        assert False

    assert tshape.argmax() == trough_offset

    # again, we'll create positive and negative versions
    # I want to look at 4 different time shifts and I want the time shift
    # to be fixed for each unit, so we need 4 units. in that case let's
    # use 2 channels to make it all work out.
    templates = np.zeros((4, 121, 2), dtype=np.float32)
    templates[0, :, 0] = tshape
    templates[1, :, 0] = -tshape
    templates[2, :, 1] = tshape
    templates[3, :, 1] = -tshape

    gt_td = dartsort.TemplateData(
        templates=templates,
        unit_ids=np.arange(4),
        spike_counts=np.ones(4),
        trough_offset_samples=trough_offset,
        registered_geom=np.c_[np.zeros(2), np.arange(2.0)],
    )

    # simulate with no noise -- that's not relevant.
    res = dartsort.simkit.generate_simulation(
        n_units=4,
        folder=None,
        noise_recording_folder=None,
        noise_kind="zero",
        templates_kind="static",
        template_simulator_kwargs=dict(template_data=gt_td),
        temporal_jitter=up_factor,
        temporal_jitter_family="by_unit",
        amplitude_jitter=0.0,
        duration_seconds=5.0,
        min_fr_hz=500.0,
        max_fr_hz=505.0,
        no_save=True,
        noise_in_memory=True,
        recording_dtype="float32",
        sampling_frequency=30_000.0,
        globally_refractory=True,
        refractory_ms=121.0 / 30.0,
        geom=gt_td.registered_geom,
        common_reference=False,
    )
    sim_recording, template_simulator = res  # type: ignore
    sim_recording: dartsort.simkit.InjectSpikesPreprocessor
    gt_st: dartsort.DARTsortSorting = sim_recording.basic_sorting()

    # match with the gt templates
    with tempfile.TemporaryDirectory() as tdir:
        st = dartsort.match(
            tdir,
            sim_recording,
            template_data=gt_td,
            featurization_cfg=dartsort.FeaturizationConfig(skip=True),
            waveform_cfg=dartsort.WaveformConfig.from_samples(
                trough_offset, 121 - trough_offset
            ),
            matching_cfg=dartsort.MatchingConfig(
                threshold=1.0,
                amplitude_scaling_variance=0.0,
                template_temporal_upsampling_factor=up_factor,
                upsampling_compression_map="none",
                template_type=matchtype,
                up_method="keys4" if matchtype == "drifty" else "direct",
            ),
        )

    assert gt_st.labels is not None
    assert st.labels is not None
    assert np.array_equal(gt_st.labels, st.labels)
    gt_up = getattr(gt_st, "jitter_ix", None)
    match_up = getattr(st, "up_inds", None)
    assert gt_up is not None
    if up_factor > 1:
        assert match_up is not None
        assert np.array_equal(gt_up, match_up)
    assert np.array_equal(gt_st.times_samples, st.times_samples)
    mcs = np.abs(templates).sum(axis=1).argmax(axis=1)
    assert np.array_equal(st.channels, mcs[gt_st.labels])
