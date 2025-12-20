from itertools import product
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import spikeinterface.full as si
import torch
import torch.nn.functional as F
from dredge import motion_util
from test_util import dense_layout, no_overlap_recording_sorting

import dartsort
from dartsort.evaluate import simkit
from dartsort.localize.localize_torch import point_source_amplitude_at
from dartsort.peel.matching import ObjectiveUpdateTemplateMatchingPeeler
from dartsort.peel.matching_util import CompressedUpsampledMatchingTemplates
from dartsort.templates import TemplateData, template_util
from dartsort.util.internal_config import MatchingConfig
from dartsort.util.job_util import ensure_computation_config
from dartsort.util.logging_util import get_logger
from dartsort.util.waveform_util import upsample_multichan


logger = get_logger(__name__)

nofeatcfg = dartsort.FeaturizationConfig(
    do_nn_denoise=False,
    do_tpca_denoise=False,
    do_enforce_decrease=False,
    denoise_only=True,
    n_residual_snips=0,
)

spike_length_samples = 121
trough_offset_samples = 42

RES_ATOL = 1e-10
CONV_ATOL = 1e-4


@pytest.fixture
def refractory_sim(request, tmp_path_factory):
    """Globally refractory sims can be matched perfectly"""
    upsampling, scaling, nc = request.param

    p = tmp_path_factory.mktemp(f"refsim_{upsampling}_{scaling}_{nc}")
    p = dartsort.resolve_path(p)
    sim = simkit.generate_simulation(
        p / "sim",
        p / "noise",
        n_units=10,
        duration_seconds=3.0,
        noise_kind="zero",
        min_fr_hz=50.0,
        max_fr_hz=100.0,
        globally_refractory=True,
        probe_kwargs=dict(num_columns=1, num_contact_per_column=nc),
        template_simulator_kwargs=dict(snr_adjustment=10.0),
        refractory_ms=5.0,
        white_noise_scale=0.0,
        recording_dtype="float32",
        temporal_jitter=upsampling,
        amplitude_jitter=scaling,
        featurization_cfg=dartsort.skip_featurization_cfg,
    )
    yield (sim, upsampling, scaling, nc, p / "matchtmp")
    shutil.rmtree(p, ignore_errors=True)


crumbs_test_upsampling = [1, 2, 4]
crumbs_test_scaling = [0.0, 0.1]
crumbs_test_chans = [1, 5]


@pytest.mark.parametrize("cd_iter", [0, 3])
@pytest.mark.parametrize("method", ["drifty_keys4"])#, "upcomp"])
# @pytest.mark.parametrize("method", ["upcomp"])#, "upcomp"])
@pytest.mark.parametrize(
    "refractory_sim",
    product(crumbs_test_upsampling, crumbs_test_scaling, crumbs_test_chans),
    indirect=True,
)
def test_no_crumbs(subtests, refractory_sim, method, cd_iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim, upsampling, scaling, nc, tmpdir = refractory_sim
    logger.info("Crumbs test: upsampling=%s scaling=%s nc=%s", upsampling, scaling, nc)
    recording = sim["recording"]
    template_data = sim["templates"]
    gt_sorting = sim["sorting"]

    # threshold should be min template norm less epsilon, or a bit less
    # if svd compression is bad. but we use full rank here.
    threshold = np.linalg.norm(template_data.templates, axis=(1, 2)).min().item()
    if scaling:
        threshold = 0.9 * threshold
    else:
        threshold = 0.99 * threshold

    # instantiate matcher
    cfg_kw: dict[str, Any] = dict(
        template_temporal_upsampling_factor=upsampling,
        # free scaling here.
        amplitude_scaling_variance=100.0 if scaling else 0.0,
        template_svd_compression_rank=121,
        threshold=threshold,
        cd_iter=cd_iter,
    )
    if method == "upcomp":
        cfg_kw["template_type"] = "individual_compressed_upsampled"
    elif method == "drifty_keys4":
        cfg_kw["template_type"] = "drifty"
        cfg_kw["up_method"] = "keys4"
    else:
        assert False
    matching_cfg = MatchingConfig(**cfg_kw)
    matcher = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording=recording,
        template_data=template_data,
        matching_cfg=matching_cfg,
        waveform_cfg=dartsort.default_waveform_cfg,
        featurization_cfg=dartsort.skip_featurization_cfg,
    )
    matcher = matcher.to(device=device)
    matcher.precompute_peeling_data(tmpdir, overwrite=True)
    matcher = matcher.to(device=device)

    # grab recording chunk and run matching
    chunk = recording.get_traces(0, 30_000 - 242, 60_000 + 242)
    chunk = torch.asarray(chunk.copy(), device=device, dtype=torch.float)
    res = matcher.peel_chunk(
        traces=chunk,
        chunk_start_samples=30_000,
        left_margin=242,
        right_margin=242,
        return_residual=True,
        return_conv=True,
    )
    assert matcher.matching_templates is not None
    chunk_temp_data = matcher.matching_templates.data_at_time(
        1.5,
        scaling=matcher.is_scaling,
        inv_lambda=matcher.inv_lambda,
        scale_min=matcher.amp_scale_min,
        scale_max=matcher.amp_scale_max,
    )
    conv = res["conv"].numpy(force=True)
    residual = res["residual"].numpy(force=True)
    times_samples = res["times_samples"].numpy(force=True)
    labels = res["labels"].numpy(force=True)

    # testing tolerance depends on whether the method is "exact" and on how
    # well the (upsampled) templates were reconstructed in the first place
    # first, figure out the template reconstruction error...
    gt_up_templates = upsample_multichan(
        template_data.templates, temporal_jitter=upsampling
    )
    assert gt_up_templates.shape[1] == upsampling
    match_up_templates = chunk_temp_data.reconstruct_up_templates().numpy(force=True)
    np.testing.assert_allclose(gt_up_templates, match_up_templates, atol=1e-4)

    # difference between upsampling before going to multichan or after...
    true_temps_up = gt_sorting._load_dataset("templates_up")
    np.testing.assert_allclose(gt_up_templates, true_temps_up, atol=1e-4)
    up_err = np.abs(gt_up_templates - true_temps_up).max()

    abs_err = np.abs(gt_up_templates - match_up_templates).max().item()
    l2_err = np.linalg.norm(gt_up_templates - match_up_templates, axis=(1, 2)).max().item()
    conv_min_atol = l2_err ** 2 + abs_err

    if matching_cfg.up_method == "direct":
        conv_atol = conv_min_atol + 1e-5
        atol = abs_err + 1e-5
    else:
        conv_atol = conv_min_atol + 1e-5
        atol = abs_err + 1e-5
    gt_in_chunk = gt_sorting.times_samples == gt_sorting.times_samples.clip(
        30_000, 60_000 - 1
    )
    gt_n_spikes = np.sum(gt_in_chunk)
    gt_up = getattr(gt_sorting, "jitter_ix", None)
    assert gt_up is not None
    gt_up = gt_up[gt_in_chunk]
    gt_scale = getattr(gt_sorting, "scalings", None)
    assert gt_scale is not None
    gt_scale = gt_scale[gt_in_chunk]
    times_rel = gt_sorting.times_samples[gt_in_chunk] - 30_000 + 242
    gt_shift = getattr(gt_sorting, "time_shifts", None)
    assert gt_shift is not None
    gt_shift = gt_shift[gt_in_chunk]

    with subtests.test(msg="crumbs_sorting"):
        assert res["n_spikes"] == gt_n_spikes
        np.testing.assert_equal(gt_sorting.labels[gt_in_chunk], labels)
        np.testing.assert_equal(gt_sorting.times_samples[gt_in_chunk], times_samples)

        match_up = res["upsampling_indices"].numpy(force=True)
        np.testing.assert_array_equal(gt_up, match_up)

        match_scale = res["scalings"].numpy(force=True)
        np.testing.assert_allclose(gt_scale, match_scale, atol=1e-6)

        if "time_shifts" in res:
            match_shift = res["time_shifts"].numpy(force=True)
            np.testing.assert_array_equal(match_shift, gt_shift)
        else:
            assert (gt_shift == 0).all()

    with subtests.test(msg="crumbs_peaks"):
        logger.info("Crumbs test: upsampling=%s scaling=%s nc=%s", upsampling, scaling, nc)
        # check that they match the templates
        toff = template_data.trough_offset_samples
        slen = template_data.templates.shape[1]
        chk_labels = gt_sorting.labels[gt_in_chunk]
        for pkt, pkl, pku, pks, pksh in zip(times_rel, chk_labels, gt_up, gt_scale, gt_shift):
            pk = chunk[pkt - toff : pkt - toff + slen].numpy(force=True)
            if upsampling > 1:
                np.testing.assert_allclose(pk, pks * gt_up_templates[pkl, pku], atol=up_err)
            elif scaling:
                np.testing.assert_allclose(pk, pks * gt_up_templates[pkl, pku])
            else:
                np.testing.assert_equal(pk, gt_up_templates[pkl, pku])
            np.testing.assert_allclose(pk, pks * match_up_templates[pkl, pku], atol=1e-4)

        # check peaks are in the right places of the chunk
        chunk_time_proj = chunk.abs().amax(dim=1)
        assert chunk_time_proj.shape == (30_000 + 2 * 242,)
        assert torch.all(chunk_time_proj[times_rel] > 0)
        for j in range(1, 20):
            print(f"{j=}")
            assert torch.all(chunk_time_proj[times_rel] > chunk_time_proj[times_rel + j])
            assert torch.all(chunk_time_proj[times_rel] > chunk_time_proj[times_rel - j])

    # with subtests.test(msg="crumbs_conv"):
    #     conv_zero = np.zeros_like(conv)
    #     np.testing.assert_allclose(conv, conv_zero, atol=conv_atol)

    with subtests.test(msg="crumbs_residual"):
        resid_zero = np.zeros_like(residual)
        if scaling:
            residual_atol = atol + 1e-6 * np.abs(gt_up_templates).max().item()
        else:
            residual_atol = atol
        np.testing.assert_allclose(residual, resid_zero, atol=residual_atol)

    # # test alignment of cc waveforms w time shift
    # with subtests.test(msg="crumbs_ccwf"):
    #     assert False

@pytest.mark.parametrize("scaling", [0.0, 0.01])
@pytest.mark.parametrize("coarse_cd", [False, True])
@pytest.mark.parametrize("cd_iter", [0, 1])
def test_tiny(tmp_path, scaling, coarse_cd, cd_iter):
    recording_length_samples = 200
    n_channels = 2
    geom = np.c_[np.zeros(2), np.arange(2)]

    # template main channel traces
    trace0 = 50 * np.exp(
        -(((np.arange(spike_length_samples) - trough_offset_samples) / 10) ** 2)
    )

    # templates
    templates = np.zeros((2, spike_length_samples, n_channels), dtype="float32")
    templates[0, :, 0] = trace0
    templates[1, :, 1] = trace0

    # spike train
    # fmt: off
    tcl = [
        50, 0, 0,
        51, 1, 1,
    ]
    # fmt: on
    times, channels, labels = np.array(tcl).reshape(-1, 3).T
    rec0 = np.zeros((recording_length_samples, n_channels), dtype="float32")
    for t, l in zip(times, labels):
        rec0[
            t - trough_offset_samples : t - trough_offset_samples + spike_length_samples
        ] += templates[l]
    rec0 = si.NumpyRecording(rec0, 30_000)
    rec0.set_dummy_probe_from_locations(geom)

    comp_cfg = ensure_computation_config(None)

    rec1 = rec0.save_to_folder(str(tmp_path / "rec"))
    for rec in [rec0, rec1]:
        template_cfg = dartsort.TemplateConfig(
            denoising_method="none",
            realign_peaks=False,
            superres_bin_min_spikes=0,
        )
        rec_no_overlap, sorting_no_overlap = no_overlap_recording_sorting(templates)
        template_data = TemplateData.from_config(
            recording=rec_no_overlap,
            sorting=sorting_no_overlap,
            template_cfg=template_cfg,
            save_folder=tmp_path,
            overwrite=True,
        )

        matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
            rec,
            waveform_cfg=dartsort.default_waveform_cfg,
            matching_cfg=dartsort.MatchingConfig(
                amplitude_scaling_variance=scaling,
                threshold=0.01,
                template_temporal_upsampling_factor=1,
                cd_iter=cd_iter,
                coarse_cd=coarse_cd,
            ),
            featurization_cfg=nofeatcfg,
            template_data=template_data,
            motion_est=motion_util.IdentityMotionEstimate(),
        )
        matcher.precompute_peeling_data(tmp_path)
        matcher.to(comp_cfg.actual_device())
        res = matcher.peel_chunk(
            torch.asarray(rec.get_traces().copy(), device=comp_cfg.actual_device()),
            return_residual=True,
            return_conv=True,
        )
        assert matcher.matching_templates is not None

        print(f"{matcher.matching_templates.device=}")
        ixa, pconv, ixb = matcher.matching_templates.pconv_db.query(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            upsampling_indices_b=torch.tensor([0, 0]),
        )
        maxpc = pconv.max(dim=1).values
        for ia, ib, pc in zip(ixa, ixb, maxpc):
            assert np.isclose(pc, (templates[ia] * templates[ib]).sum())
        assert res["n_spikes"] == len(times)
        assert np.array_equal(res["times_samples"].numpy(force=True), times)
        assert np.array_equal(res["labels"].numpy(force=True), labels)
        resid_rms = torch.square(res["residual"]).mean().numpy(force=True)
        assert np.isclose(resid_rms, 0.0, atol=RES_ATOL)
        conv_rms = torch.square(res["conv"]).mean().numpy(force=True)
        assert np.isclose(conv_rms, 0.0, atol=CONV_ATOL)
        matcher.cpu()

        matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
            rec,
            waveform_cfg=dartsort.default_waveform_cfg,
            matching_cfg=dartsort.MatchingConfig(
                threshold=0.01,
                amplitude_scaling_variance=0.0,
                template_temporal_upsampling_factor=8,
            ),
            featurization_cfg=nofeatcfg,
            template_data=template_data,
            motion_est=motion_util.IdentityMotionEstimate(),
        )
        matcher.precompute_peeling_data(tmp_path)
        matcher.to(comp_cfg.actual_device())
        res = matcher.peel_chunk(
            torch.asarray(rec.get_traces().copy(), device=comp_cfg.actual_device()),
            return_residual=True,
            return_conv=True,
        )
        assert res["n_spikes"] == len(times)
        assert np.array_equal(res["times_samples"].numpy(force=True), times)
        assert np.array_equal(res["labels"].numpy(force=True), labels)
        assert np.array_equal(res["upsampling_indices"].numpy(force=True), [0, 0])
        resid_rms = torch.square(res["residual"]).mean().numpy(force=True)
        assert np.isclose(resid_rms, 0.0, atol=RES_ATOL)
        conv_rms = torch.square(res["conv"]).mean().numpy(force=True)
        assert np.isclose(conv_rms, 0.0, atol=CONV_ATOL)
        assert torch.all(res["scores"] > 0)
        matcher.cpu()


@pytest.mark.parametrize("up_offset", [0, 1, -1])
@pytest.mark.parametrize("up_factor", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("scaling", [0.0, 0.01])
@pytest.mark.parametrize("cd_iter", [0, 1])
def test_tiny_up(tmp_path, up_factor, scaling, cd_iter, up_offset):
    comp_cfg = ensure_computation_config(None)
    dev = comp_cfg.actual_device()
    recording_length_samples = 2000
    n_channels = 11
    geom = np.c_[np.zeros(n_channels), np.arange(n_channels)]
    # template main channel traces
    trace0 = 50 * np.exp(
        -(((np.arange(spike_length_samples) - trough_offset_samples) / 10) ** 2)
    )

    # templates
    templates = np.zeros((1, spike_length_samples, n_channels), dtype="float32")
    templates[0, :, 0] = trace0
    # templates[1, :, 1] = trace0
    print("-- cupts")
    cupts = template_util.compressed_upsampled_templates(
        templates,
        ptps=np.ptp(templates, 1).max(1),
        max_upsample=up_factor,
    )

    # spike train
    # fmt: off
    start = 50
    # tclu = []
    # for i in range(up_factor):
    #     tclu.extend((start + 200 * i, 0, 0, i))
    tclu = [50, 0, 0, min(up_factor - 1, up_offset if up_offset >= 0 else up_factor + up_offset)]
    # fmt: on
    times, channels, labels, upsampling_indices = np.array(tclu).reshape(-1, 4).T
    trough_shifts = []
    rec0 = np.zeros((recording_length_samples, n_channels), dtype="float32")
    for t, l, u, c in zip(times, labels, upsampling_indices, channels):
        temp = cupts.compressed_upsampled_templates[
            cupts.compressed_upsampling_map[l, u]
        ]
        trough_shifts.append(np.abs(temp[:, c]).argmax() - trough_offset_samples)
        rec0[
            t - trough_offset_samples : t - trough_offset_samples + spike_length_samples
        ] += temp
    rec0 = si.NumpyRecording(rec0, 30_000)
    rec0.set_dummy_probe_from_locations(geom)
    trough_shifts = np.array(trough_shifts)

    rec1 = rec0.save_to_folder(tmp_path / "rec")
    for rec in [rec0, rec1]:
        template_cfg = dartsort.TemplateConfig(
            realign_peaks=False,
            denoising_method="none",
            superres_bin_min_spikes=0,
        )
        template_data = TemplateData.from_config(
            *no_overlap_recording_sorting(templates),
            template_cfg,
            motion_est=motion_util.IdentityMotionEstimate(),
            save_folder=tmp_path,
            overwrite=True,
        )
        assert np.allclose(template_data.templates, templates)

        print("-- make matcher")
        matching_cfg = dartsort.MatchingConfig(
            threshold=0.01,
            amplitude_scaling_variance=scaling,
            template_temporal_upsampling_factor=up_factor,
            cd_iter=cd_iter,
        )
        matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
            recording=rec,
            waveform_cfg=dartsort.default_waveform_cfg,
            matching_cfg=matching_cfg,
            featurization_cfg=nofeatcfg,
            template_data=template_data,
            motion_est=motion_util.IdentityMotionEstimate(),
        )
        matcher.precompute_peeling_data(tmp_path)
        matcher.to(dev)

        lrt = template_util.svd_compress_templates(
            template_data.templates, rank=matching_cfg.template_svd_compression_rank
        )
        print("-- tempup")
        tempup = template_util.compressed_upsampled_templates(
            lrt.temporal_components,
            ptps=np.ptp(template_data.templates, 1).max(1),
            max_upsample=up_factor,
        )
        matchdata = matcher.matching_templates
        assert isinstance(matchdata, CompressedUpsampledMatchingTemplates)
        assert np.allclose(
            matchdata.b.cup_temporal.numpy(force=True),
            tempup.compressed_upsampled_templates,
        )
        assert np.allclose(
            matchdata.b.obj_spatial_sing.numpy(force=True),
            lrt.singular_values[:, :, None] * lrt.spatial_components,
        )
        for up in range(up_factor):
            ixa, pconv, ixb = matchdata.pconv_db.query(
                np.arange(1), np.arange(1), upsampling_indices_b=up
            )
            centerpc = pconv[:, spike_length_samples - 1]
            for ia, ib, pc, pcf in zip(ixa, ixb, centerpc, pconv):
                tempupb = tempup.compressed_upsampled_templates[
                    tempup.compressed_upsampling_map[ib, up]
                ]
                tupb = (tempupb * lrt.singular_values[ib]) @ lrt.spatial_components[ib]
                tc = (templates[ia] * tupb).sum()

                # test against a very direct way of computing this convolution
                # just directly create the actual templates
                temp_a = torch.as_tensor(templates[ia])
                ssb = lrt.singular_values[ib][:, None] * lrt.spatial_components[ib]
                temp_b = torch.as_tensor(tempupb) @ (torch.as_tensor(ssb))

                pconv2 = F.conv2d(
                    temp_a[None, None],
                    temp_b[None, None],
                    padding=(temp_a.shape[0] - 1, 0),
                )
                pconv2 = pconv2[0, 0, :, 0]
                assert np.allclose(pconv2.numpy()[::-1], pconv)
                assert np.isclose(pc, tc)

        res = matcher.peel_chunk(
            torch.asarray(
                rec.get_traces().copy(), device=matcher.b.channel_index.device
            ),
            return_residual=True,
            return_conv=True,
        )

        assert res["n_spikes"] == len(times)
        assert np.array_equal(
            res["times_samples"].numpy(force=True), times + trough_shifts
        )
        assert np.array_equal(res["labels"].numpy(force=True), labels)
        print(f"{res['n_spikes']=} {len(times)=}")
        print(f"{res['times_samples']=}")
        print(f"{res['upsampling_indices']=}")
        assert np.array_equal(
            res["upsampling_indices"].numpy(force=True), upsampling_indices
        )
        resid_rms = torch.square(res["residual"]).mean().numpy(force=True)
        assert np.isclose(resid_rms, 0.0, atol=RES_ATOL)
        conv_rms = torch.square(res["conv"]).mean().numpy(force=True)
        assert np.isclose(conv_rms, 0.0, atol=CONV_ATOL)
        assert torch.all(res["scores"] > 0)
        assert torch.all(res["scores"] > 0)


@pytest.mark.parametrize("up_factor", [1, 2, 4, 8])
@pytest.mark.parametrize("cd_iter", [0, 1])
def test_static(tmp_path, up_factor, cd_iter):
    comp_cfg = ensure_computation_config(None)
    dev = comp_cfg.actual_device()
    recording_length_samples = 40_011
    n_channels = 2
    geom = np.c_[np.zeros(2), np.arange(2)]

    # template main channel traces
    trace0 = 50 * np.exp(
        -(((np.arange(spike_length_samples) - trough_offset_samples) / 10) ** 2)
    )
    trace1 = 250 * np.exp(
        -(((np.arange(spike_length_samples) - trough_offset_samples) / 30) ** 2)
    )

    # templates
    templates = np.zeros((3, spike_length_samples, n_channels), dtype="float32")
    templates[0, :, 0] = trace0
    templates[1, :, 0] = trace1
    templates[2, :, 1] = trace0

    # spike train
    # fmt: off
    tcl = [
        100, 0, 0,
        150, 0, 0,
        151, 1, 2,
        500, 0, 1,
        2000, 0, 0,
        2001, 0, 1,
        35000, 1, 2,
        35001, 0, 1,
    ]
    # fmt: on
    times, channels, labels = np.array(tcl).reshape(-1, 3).T
    rec0 = np.zeros((recording_length_samples, n_channels), dtype="float32")
    for t, l in zip(times, labels):
        rec0[
            t - trough_offset_samples : t - trough_offset_samples + spike_length_samples
        ] += templates[l]
    rec0 = si.NumpyRecording(rec0, 30_000)
    rec0.set_dummy_probe_from_locations(geom)

    rec1 = rec0.save_to_folder(tmp_path / "rec")
    for rec in [rec0, rec1]:
        template_cfg = dartsort.TemplateConfig(
            denoising_method="none", superres_bin_min_spikes=0
        )
        template_data = TemplateData.from_config(
            *no_overlap_recording_sorting(templates),
            template_cfg,
            motion_est=motion_util.IdentityMotionEstimate(),
            save_folder=tmp_path,
            overwrite=True,
        )
        matching_cfg = dartsort.MatchingConfig(
            threshold=0.01,
            template_temporal_upsampling_factor=up_factor,
            amplitude_scaling_variance=0.0,
            coarse_approx_error_threshold=0.0,
            conv_ignore_threshold=0.0,
            template_svd_compression_rank=2,
            cd_iter=cd_iter,
        )

        matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
            rec,
            waveform_cfg=dartsort.default_waveform_cfg,
            matching_cfg=matching_cfg,
            featurization_cfg=nofeatcfg,
            template_data=template_data,
            motion_est=motion_util.IdentityMotionEstimate(),
        )
        matcher.precompute_peeling_data(tmp_path)
        matcher.to(dev)
        matchdata = matcher.matching_templates
        assert isinstance(matchdata, CompressedUpsampledMatchingTemplates)

        lrt = template_util.svd_compress_templates(
            template_data.templates, rank=matching_cfg.template_svd_compression_rank
        )
        tempup = template_util.compressed_upsampled_templates(
            lrt.temporal_components,
            ptps=np.ptp(template_data.templates, 1).max(1),
            max_upsample=up_factor,
        )
        assert np.allclose(
            matchdata.b.cup_temporal.numpy(force=True),
            tempup.compressed_upsampled_templates,
        )
        assert np.allclose(
            matchdata.b.obj_spatial_sing.numpy(force=True),
            lrt.singular_values[:, :, None] * lrt.spatial_components,
        )
        for up in range(up_factor):
            ixa, pconv, ixb = matchdata.pconv_db.query(
                np.arange(3),
                np.arange(3),
                upsampling_indices_b=up + np.zeros(3, dtype=np.int64),
            )
            centerpc = pconv[:, spike_length_samples - 1]
            for ia, ib, pc, pcf in zip(ixa, ixb, centerpc, pconv):
                tempupb = tempup.compressed_upsampled_templates[
                    tempup.compressed_upsampling_map[ib, up]
                ]
                tupb = (tempupb * lrt.singular_values[ib]) @ lrt.spatial_components[ib]
                tc = (templates[ia] * tupb).sum()

                template_a = torch.as_tensor(templates[ia][None])
                ssb = lrt.singular_values[ib][:, None] * lrt.spatial_components[ib]
                conv_filt = torch.bmm(torch.as_tensor(ssb[None]), template_a.mT)
                conv_filt = conv_filt[:, None]  # (nco, 1, rank, t)
                conv_in = torch.as_tensor(tempupb[None]).mT[None]
                pconv_ = F.conv2d(conv_in, conv_filt, padding=(0, 120), groups=1)
                pconv1 = pconv_.squeeze()[spike_length_samples - 1].numpy(force=True)
                assert torch.isclose(pcf.cpu(), pconv_).all()

                pconv2 = (
                    F.conv2d(
                        torch.as_tensor(templates[ia])[None, None],
                        torch.as_tensor(tupb)[None, None],
                    )
                    .squeeze()
                    .numpy(force=True)
                )
                assert np.isclose(pconv2, tc)
                assert np.isclose(pc.cpu(), tc)
                assert np.isclose(pconv1, pc.cpu())

        res = matcher.peel_chunk(
            torch.asarray(
                rec.get_traces().copy(), device=matcher.b.channel_index.device
            ),
            return_residual=True,
            return_conv=True,
        )

        assert res["n_spikes"] == len(times)
        assert np.array_equal(res["times_samples"].cpu(), times)
        assert np.array_equal(res["upsampling_indices"].cpu(), np.zeros(len(times)))
        assert np.array_equal(res["labels"].cpu(), labels)
        assert np.isclose(torch.square(res["residual"]).mean().cpu(), 0.0, atol=1e-5)
        assert np.isclose(torch.square(res["conv"]).mean().cpu(), 0.0, atol=1e-3)
        assert torch.all(res["scores"].cpu() > 0)


def test_fakedata_nonn(tmp_path, threshold=7.0):
    print("test_fakedata_nonn")
    # generate fake neuropixels data with artificial templates
    T_s = 9.5
    fs = 30000
    n_channels = 25
    T_samples = int(fs * T_s)
    rg = np.random.default_rng(0)

    # np1 geom
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]][:n_channels]

    # template main channel traces
    t0 = np.exp(-(((np.arange(121) - 42) / 10) ** 2))
    t1 = np.exp(-(((np.arange(121) - 42) / 30) ** 2))
    t2 = t0 - 0.5 * np.exp(-(((np.arange(121) - 46) / 10) ** 2))
    t3 = t0 - 0.5 * np.exp(-(((np.arange(121) - 46) / 30) ** 2))

    # fake main channels, positions, brightnesses
    chans = rg.integers(0, len(geom), size=4)
    chan_zs = geom[chans, 1]
    xs = rg.normal(loc=geom[:, 0].mean(), scale=10, size=4)
    ys = rg.uniform(1e-3, 100, size=4)
    z_rels = rg.normal(scale=10, size=4)
    z_abss = chan_zs + z_rels
    alphas = rg.uniform(5.0, 15, size=4)

    # fake amplitude distributions
    amps = [
        point_source_amplitude_at(x, y, z, alpha, torch.as_tensor(geom)).numpy()
        for x, y, z, alpha in torch.column_stack(
            list(map(torch.as_tensor, [xs, ys, z_abss, alphas]))
        )
    ]

    # combine to make templates
    templates = np.array(
        [t[:, None] * a[None, :] for t, a in zip((t0, t1, t2, t3), amps)]
    )
    norm = np.abs(templates[0]).max()
    templates[0] *= 100 / norm
    templates[1] *= 50 / norm
    templates[2] *= 100 / norm
    templates[3] *= 50 / norm

    # make fake spike trains
    spikes_per_unit = 51
    sts = []
    labels = []
    for i in range(len(templates)):
        while True:
            st = rg.choice(T_samples - 121, size=spikes_per_unit)
            st.sort()
            ref = np.diff(st).min()
            if ref > 15:
                sts.append(st)
                break
        labels.append(np.full((spikes_per_unit,), i))
    times = np.concatenate(sts)
    labels = np.concatenate(labels)
    order = np.argsort(times)
    times = times[order]
    labels = labels[order]
    gts = dartsort.DARTsortSorting(
        times_samples=times + 42, labels=labels, channels=np.zeros_like(labels)
    )

    # inject the spikes into a noise background
    rec0 = 0.1 * rg.normal(size=(T_samples, len(geom))).astype(np.float32)
    for t, l in zip(times, labels):
        rec0[t : t + 121] += templates[l]
    assert np.sum(np.abs(rec0) > 80) >= 50
    assert np.sum(np.abs(rec0) > 40) >= 100

    # make into spikeinterface
    rec0 = si.NumpyRecording(rec0, fs)
    rec0.set_dummy_probe_from_locations(geom)

    featconf = dartsort.FeaturizationConfig(
        do_nn_denoise=False, do_tpca_denoise=False, n_residual_snips=8
    )
    tempconf = dartsort.TemplateConfig(
        realign_peaks=False,
        denoising_method="none",
        superres_templates=False,
        registered_templates=False,
    )
    matchconf = dartsort.MatchingConfig(threshold=threshold)
    matchconf_fp = dartsort.MatchingConfig(threshold="fp_control")

    rec1 = rec0.save_to_folder(tmp_path / "rec")
    for rec in [rec1, rec0]:
        (tmp_path / "match").mkdir()
        st = dartsort.match(
            recording=rec,
            sorting=gts,
            output_dir=tmp_path / "match",
            motion_est=None,
            template_cfg=tempconf,
            featurization_cfg=featconf,
            matching_cfg=matchconf,
        )
        assert st.scores is not None  # type: ignore[reportAttributeAccessIssue]
        assert np.all(st.scores > 0)  # type: ignore[reportAttributeAccessIssue]

        (tmp_path / "match2").mkdir()
        st2 = dartsort.match(
            recording=rec,
            sorting=st,
            output_dir=tmp_path / "match2",
            motion_est=None,
            template_cfg=tempconf,
            featurization_cfg=featconf,
            matching_cfg=matchconf_fp,
        )
        assert np.all(st.scores > 0)  # type: ignore[reportAttributeAccessIssue]

        print(f"{st=}")
        print(f"{st2=}")

        shutil.rmtree(tmp_path / "match")
        shutil.rmtree(tmp_path / "match2")


@pytest.mark.parametrize("sim_name", ["driftn_szmini", "drifty_szmini"])
@pytest.mark.parametrize("threshold", ["check", "fp_control"])
def test_with_simkit(simulations, sim_name, threshold):
    sim = simulations[sim_name]
    rec = sim["recording"]
    template_data = sim["templates"]
    motion_est = sim["motion_est"]
    gt_st = sim["sorting"]

    with tempfile.TemporaryDirectory() as tdir:
        if threshold == "check":
            threshold = np.sqrt(
                0.5 * np.square(template_data.templates).sum((1, 2)).min()
            )
        st = dartsort.match(
            recording=rec,
            sorting=gt_st,
            output_dir=tdir,
            motion_est=motion_est,
            template_data=template_data,
            featurization_cfg=dartsort.FeaturizationConfig(skip=True),
            matching_cfg=dartsort.MatchingConfig(threshold=threshold),
        )
        print(f"{threshold=} {st=}")
        assert len(st) > 0.9 * len(gt_st)
        # assert abs(len(st) - len(gt_st)) / len(gt_st) < 0.3
