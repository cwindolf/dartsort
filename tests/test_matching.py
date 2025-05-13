import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import spikeinterface.full as si
import torch
import torch.nn.functional as F

import dartsort
from dartsort.localize.localize_torch import point_source_amplitude_at
from dartsort.templates import TemplateData, template_util
from dredge import motion_util
from test_util import dense_layout, no_overlap_recording_sorting

nofeatcfg = dartsort.FeaturizationConfig(
    do_nn_denoise=False,
    do_tpca_denoise=False,
    do_enforce_decrease=False,
    denoise_only=True,
    n_residual_snips=0,
)

spike_length_samples = 121
trough_offset_samples = 42

RES_ATOL = 1e-5
CONV_ATOL = 0.1


def _test_tiny(tmp_path, scaling=0.0):
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

    with tempfile.TemporaryDirectory(dir=tmp_path) as tdir:
        rec1 = rec0.save_to_folder(str(Path(tdir) / "rec"))
        for rec in [rec0, rec1]:
            template_config = dartsort.TemplateConfig(
                low_rank_denoising=False,
                superres_bin_min_spikes=0,
            )
            template_data = TemplateData.from_config(
                *no_overlap_recording_sorting(templates),
                template_config,
                motion_est=motion_util.IdentityMotionEstimate(),
                save_folder=tdir,
                overwrite=True,
            )

            matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
                rec,
                dartsort.default_waveform_config,
                dartsort.MatchingConfig(
                    amplitude_scaling_variance=scaling,
                    threshold=0.01,
                    template_temporal_upsampling_factor=1,
                ),
                nofeatcfg,
                template_data,
                motion_est=motion_util.IdentityMotionEstimate(),
            )
            matcher.precompute_peeling_data(tdir)
            res = matcher.peel_chunk(
                torch.from_numpy(rec.get_traces().copy()),
                return_residual=True,
                return_conv=True,
            )

            ixa, ixb, pconv = matcher.pairwise_conv_db.query(
                [0, 1], [0, 1], upsampling_indices_b=[0, 0], grid=True
            )
            maxpc = pconv.max(dim=1).values
            for ia, ib, pc in zip(ixa, ixb, maxpc):
                assert np.isclose(pc, (templates[ia] * templates[ib]).sum())
            assert res["n_spikes"] == len(times)
            assert np.array_equal(res["times_samples"], times)
            assert np.array_equal(res["labels"], labels)
            assert np.isclose(torch.square(res["residual"]).mean(), 0.0, atol=RES_ATOL)
            assert np.isclose(torch.square(res["conv"]).mean(), 0.0, atol=CONV_ATOL)

            matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
                rec,
                dartsort.default_waveform_config,
                dartsort.MatchingConfig(
                    threshold=0.01,
                    amplitude_scaling_variance=0.0,
                    template_temporal_upsampling_factor=8,
                ),
                nofeatcfg,
                template_data,
                motion_est=motion_util.IdentityMotionEstimate(),
            )
            matcher.precompute_peeling_data(tmp_path)
            res = matcher.peel_chunk(
                torch.from_numpy(rec.get_traces().copy()),
                return_residual=True,
                return_conv=True,
            )
            assert res["n_spikes"] == len(times)
            assert np.array_equal(res["times_samples"], times)
            assert np.array_equal(res["labels"], labels)
            assert np.array_equal(res["upsampling_indices"], [0, 0])
            print(f'{torch.square(res["residual"]).mean()=}')
            assert np.isclose(torch.square(res["residual"]).mean(), 0.0, atol=RES_ATOL)
            print(f'{torch.square(res["conv"]).mean()=}')
            assert np.isclose(torch.square(res["conv"]).mean(), 0.0, atol=CONV_ATOL)
            assert torch.all(res["scores"] > 0)

        del matcher, rec1, rec, pconv # trying to help windows runner?


def test_tiny_unscaled(tmp_path):
    _test_tiny(tmp_path)


def test_tiny_scaled(tmp_path):
    _test_tiny(tmp_path, scaling=0.01)


def _test_tiny_up(tmp_path, up_factor=1, scaling=0.0):
    recording_length_samples = 2000
    n_channels = 2
    geom = np.c_[np.zeros(n_channels), np.arange(n_channels)]
    # template main channel traces
    trace0 = 50 * np.exp(
        -(((np.arange(spike_length_samples) - trough_offset_samples) / 10) ** 2)
    )

    # templates
    templates = np.zeros((1, spike_length_samples, n_channels), dtype="float32")
    templates[0, :, 0] = trace0
    # templates[1, :, 1] = trace0
    cupts = template_util.compressed_upsampled_templates(
        templates, max_upsample=up_factor
    )

    # spike train
    # fmt: off
    start = 50
    # tclu = []
    # for i in range(up_factor):
    #     tclu.extend((start + 200 * i, 0, 0, i))
    tclu = [50, 0, 0, up_factor - 1]
    # fmt: on
    times, channels, labels, upsampling_indices = np.array(tclu).reshape(-1, 4).T
    rec0 = np.zeros((recording_length_samples, n_channels), dtype="float32")
    for t, l, u in zip(times, labels, upsampling_indices):
        temp = cupts.compressed_upsampled_templates[
            cupts.compressed_upsampling_map[l, u]
        ]
        rec0[
            t - trough_offset_samples : t - trough_offset_samples + spike_length_samples
        ] += temp
    rec0 = si.NumpyRecording(rec0, 30_000)
    rec0.set_dummy_probe_from_locations(geom)

    with tempfile.TemporaryDirectory(dir=tmp_path) as tdir:
        rec1 = rec0.save_to_folder(Path(tdir) / "rec")
        for rec in [rec0, rec1]:
            template_config = dartsort.TemplateConfig(
                low_rank_denoising=False,
                superres_bin_min_spikes=0,
            )
            template_data = TemplateData.from_config(
                *no_overlap_recording_sorting(templates),
                template_config,
                motion_est=motion_util.IdentityMotionEstimate(),
                save_folder=tdir,
                overwrite=True,
            )

            matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
                rec,
                dartsort.default_waveform_config,
                dartsort.MatchingConfig(
                    threshold=0.01,
                    amplitude_scaling_variance=scaling,
                    template_temporal_upsampling_factor=up_factor,
                ),
                nofeatcfg,
                template_data,
                motion_est=motion_util.IdentityMotionEstimate(),
            )
            matcher.precompute_peeling_data(tdir)

            lrt = template_util.svd_compress_templates(
                template_data.templates, rank=matcher.svd_compression_rank
            )
            tempup = template_util.compressed_upsampled_templates(
                lrt.temporal_components,
                ptps=np.ptp(template_data.templates, 1).max(1),
                max_upsample=up_factor,
            )
            assert np.array_equal(
                matcher.compressed_upsampled_temporal,
                tempup.compressed_upsampled_templates,
            )
            assert np.array_equal(
                matcher.objective_spatial_components, lrt.spatial_components
            )
            assert np.array_equal(
                matcher.objective_singular_values, lrt.singular_values
            )
            assert np.array_equal(matcher.spatial_components, lrt.spatial_components)
            assert np.array_equal(matcher.singular_values, lrt.singular_values)
            for up in range(up_factor):
                ixa, ixb, pconv = matcher.pairwise_conv_db.query(
                    np.arange(1),
                    np.arange(1),
                    upsampling_indices_b=up,
                    grid=True,
                )
                centerpc = pconv[:, spike_length_samples - 1]
                for ia, ib, pc, pcf in zip(ixa, ixb, centerpc, pconv):
                    tempupb = tempup.compressed_upsampled_templates[
                        tempup.compressed_upsampling_map[ib, up]
                    ]
                    tupb = (tempupb * lrt.singular_values[ib]) @ lrt.spatial_components[
                        ib
                    ]
                    tc = (templates[ia] * tupb).sum()

                    template_a = torch.as_tensor(templates[ia][None])
                    ssb = lrt.singular_values[ib][:, None] * lrt.spatial_components[ib]
                    conv_filt = torch.bmm(torch.as_tensor(ssb[None]), template_a.mT)
                    conv_filt = conv_filt[:, None]  # (nco, 1, rank, t)
                    conv_in = torch.as_tensor(tempupb[None]).mT[None]
                    pconv_ = F.conv2d(conv_in, conv_filt, padding=(0, 120), groups=1)
                    pconv1 = pconv_.squeeze()[spike_length_samples - 1].numpy(
                        force=True
                    )
                    assert torch.isclose(pcf, pconv_).all()

                    pconv2 = (
                        F.conv2d(
                            torch.as_tensor(templates[ia])[None, None],
                            torch.as_tensor(tupb)[None, None],
                        )
                        .squeeze()
                        .numpy(force=True)
                    )
                    assert np.isclose(pconv2, tc)
                    assert np.isclose(pc, tc)
                    assert np.isclose(pconv1, pc)
                    print(f" - {ia=} {ib=} {up=} {pc=} {tc=}")

            res = matcher.peel_chunk(
                torch.from_numpy(rec.get_traces().copy()),
                return_residual=True,
                return_conv=True,
            )

            assert res["n_spikes"] == len(times)
            assert np.array_equal(res["times_samples"], times)
            assert np.array_equal(res["labels"], labels)
            print(f'{torch.square(res["residual"]).mean()=}')
            assert np.isclose(torch.square(res["residual"]).mean(), 0.0, atol=RES_ATOL)
            print(f'{torch.square(res["conv"]).mean()=}')
            assert np.isclose(torch.square(res["conv"]).mean(), 0.0, atol=CONV_ATOL)
            assert torch.all(res["scores"] > 0)

        del matcher, rec1, rec, pconv, pconv2 # trying to help windows runner?


def test_tiny_up_1_0(tmp_path):
    _test_tiny_up(tmp_path, up_factor=1, scaling=0.0)


def test_tiny_up_8_0(tmp_path):
    _test_tiny_up(tmp_path, up_factor=8, scaling=0.0)


def test_tiny_up_1_001(tmp_path):
    _test_tiny_up(tmp_path, up_factor=1, scaling=0.01)


def test_tiny_up_8_001(tmp_path):
    _test_tiny_up(tmp_path, up_factor=8, scaling=0.01)


def static_tester(tmp_path, up_factor=1):
    recording_length_samples = 40_011
    n_channels = 2
    geom = np.c_[np.zeros(2), np.arange(2)]
    geom

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

    with tempfile.TemporaryDirectory() as tdir:
        rec1 = rec0.save_to_folder(Path(tdir) / "rec")
        for rec in [rec0, rec1]:
            template_config = dartsort.TemplateConfig(
                low_rank_denoising=False, superres_bin_min_spikes=0
            )
            template_data = TemplateData.from_config(
                *no_overlap_recording_sorting(templates),
                template_config,
                motion_est=motion_util.IdentityMotionEstimate(),
                save_folder=tmp_path,
                overwrite=True,
            )

            matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
                rec,
                dartsort.default_waveform_config,
                dartsort.MatchingConfig(
                    threshold=0.01,
                    template_temporal_upsampling_factor=up_factor,
                    amplitude_scaling_variance=0.0,
                    coarse_approx_error_threshold=0.0,
                    conv_ignore_threshold=0.0,
                    template_svd_compression_rank=2,
                ),
                nofeatcfg,
                template_data,
                motion_est=motion_util.IdentityMotionEstimate(),
            )
            matcher.precompute_peeling_data(tmp_path)

            lrt = template_util.svd_compress_templates(
                template_data.templates, rank=matcher.svd_compression_rank
            )
            tempup = template_util.compressed_upsampled_templates(
                lrt.temporal_components,
                ptps=np.ptp(template_data.templates, 1).max(1),
                max_upsample=up_factor,
            )
            assert np.array_equal(
                matcher.compressed_upsampled_temporal,
                tempup.compressed_upsampled_templates,
            )
            assert np.array_equal(
                matcher.objective_spatial_components, lrt.spatial_components
            )
            assert np.array_equal(
                matcher.objective_singular_values, lrt.singular_values
            )
            assert np.array_equal(matcher.spatial_components, lrt.spatial_components)
            assert np.array_equal(matcher.singular_values, lrt.singular_values)
            for up in range(up_factor):
                ixa, ixb, pconv = matcher.pairwise_conv_db.query(
                    np.arange(3),
                    np.arange(3),
                    upsampling_indices_b=up + np.zeros(3, dtype=int),
                    grid=True,
                )
                centerpc = pconv[:, spike_length_samples - 1]
                for ia, ib, pc, pcf in zip(ixa, ixb, centerpc, pconv):
                    tempupb = tempup.compressed_upsampled_templates[
                        tempup.compressed_upsampling_map[ib, up]
                    ]
                    tupb = (tempupb * lrt.singular_values[ib]) @ lrt.spatial_components[
                        ib
                    ]
                    tc = (templates[ia] * tupb).sum()

                    template_a = torch.as_tensor(templates[ia][None])
                    ssb = lrt.singular_values[ib][:, None] * lrt.spatial_components[ib]
                    conv_filt = torch.bmm(torch.as_tensor(ssb[None]), template_a.mT)
                    conv_filt = conv_filt[:, None]  # (nco, 1, rank, t)
                    conv_in = torch.as_tensor(tempupb[None]).mT[None]
                    pconv_ = F.conv2d(conv_in, conv_filt, padding=(0, 120), groups=1)
                    pconv1 = pconv_.squeeze()[spike_length_samples - 1].numpy(
                        force=True
                    )
                    assert torch.isclose(pcf, pconv_).all()

                    pconv2 = (
                        F.conv2d(
                            torch.as_tensor(templates[ia])[None, None],
                            torch.as_tensor(tupb)[None, None],
                        )
                        .squeeze()
                        .numpy(force=True)
                    )
                    assert np.isclose(pconv2, tc)
                    assert np.isclose(pc, tc)
                    assert np.isclose(pconv1, pc)

            res = matcher.peel_chunk(
                torch.from_numpy(rec.get_traces().copy()),
                return_residual=True,
                return_conv=True,
            )

            assert res["n_spikes"] == len(times)
            assert np.array_equal(res["times_samples"], times)
            assert np.array_equal(res["labels"], labels)
            assert np.isclose(torch.square(res["residual"]).mean(), 0.0, atol=1e-5)
            print(f"D {torch.square(res['conv']).mean()=}")
            assert np.isclose(torch.square(res["conv"]).mean(), 0.0, atol=1e-3)
            assert torch.all(res["scores"] > 0)


def test_static_noup(tmp_path):
    static_tester(tmp_path)


def test_static_up(tmp_path):
    static_tester(tmp_path, up_factor=8)


def _test_fakedata_nonn(threshold):
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
        low_rank_denoising=True,
        superres_templates=False,
        registered_templates=False,
    )
    matchconf = dartsort.MatchingConfig(threshold=threshold)
    matchconf_fp = dartsort.MatchingConfig(threshold="fp_control")

    with tempfile.TemporaryDirectory() as tdir:
        rec1 = rec0.save_to_folder(Path(tdir) / "rec")
        for rec in [rec1, rec0]:
            (Path(tdir) / "match").mkdir()
            st = dartsort.match(
                recording=rec,
                sorting=gts,
                output_dir=Path(tdir) / "match",
                motion_est=None,
                template_config=tempconf,
                featurization_config=featconf,
                matching_config=matchconf,
            )
            assert np.all(st.scores > 0)

            (Path(tdir) / "match2").mkdir()
            st2 = dartsort.match(
                recording=rec,
                sorting=st,
                output_dir=Path(tdir) / "match2",
                motion_est=None,
                template_config=tempconf,
                featurization_config=featconf,
                matching_config=matchconf_fp,
            )
            assert np.all(st.scores > 0)

            print(f"{st=}")
            print(f"{st2=}")

            shutil.rmtree(Path(tdir) / "match")
            shutil.rmtree(Path(tdir) / "match2")


def test_fakedata_nonn():
    _test_fakedata_nonn(7.0)


@pytest.mark.parametrize("rec_type", ["static", "drifting"])
@pytest.mark.parametrize("threshold", ["check", "fp_control"])
def test_with_simkit(sim_recordings, rec_type, threshold):
    sim = sim_recordings[rec_type]
    rec = sim["rec"]
    template_data = sim["template_data"]
    motion_est = sim["motion_est"]
    gt_st = sim["sorting"]

    with tempfile.TemporaryDirectory() as tdir:
        if threshold == "check":
            threshold = 0.5 * np.sqrt(np.square(template_data.templates).sum((1, 2)).min())
        st = dartsort.match(
            recording=rec,
            sorting=gt_st,
            output_dir=tdir,
            motion_est=motion_est,
            template_data=template_data,
            featurization_config=dartsort.FeaturizationConfig(skip=True),
            matching_config=dartsort.MatchingConfig(threshold=threshold),
        )
        print(f"{threshold=} {st=}")
        assert abs(len(st) - len(gt_st)) / len(gt_st) < 0.1


if __name__ == "__main__":
    test_fakedata_nonn()

    print("\n\ntest tiny unscaled")
    with tempfile.TemporaryDirectory() as tdir:
        test_tiny_unscaled(Path(tdir))

    print("\n\ntest tiny scaled")
    with tempfile.TemporaryDirectory() as tdir:
        test_tiny_scaled(Path(tdir))

    for up in (1, 8):
        for sc in (0.0, 0.01):
            print(f"\n\ntest tiny_up {up=} {sc=}")
            with tempfile.TemporaryDirectory() as tdir:
                _test_tiny_up(Path(tdir), up_factor=up, scaling=sc)

    print("\n\ntest test_static_noup")
    with tempfile.TemporaryDirectory() as tdir:
        test_static_noup(Path(tdir))

    print("\n\ntest test_static_up")
    with tempfile.TemporaryDirectory() as tdir:
        test_static_up(Path(tdir))
