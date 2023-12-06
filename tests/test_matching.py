import numpy as np
import spikeinterface.full as si
import torch
import torch.nn.functional as F
from dartsort import config, main
from dartsort.templates import TemplateData, template_util
from dredge import motion_util
from test_util import no_overlap_recording_sorting

nofeatcfg = config.FeaturizationConfig(
    do_nn_denoise=False,
    do_tpca_denoise=False,
    do_enforce_decrease=False,
    denoise_only=True,
)

spike_length_samples = 121
trough_offset_samples = 42


def test_tiny(tmp_path):
    recording_length_samples = 200
    n_channels = 2
    geom = np.c_[np.zeros(2), np.arange(2)]
    geom

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
    rec = np.zeros((recording_length_samples, n_channels), dtype="float32")
    for t, l in zip(times, labels):
        rec[
            t - trough_offset_samples : t - trough_offset_samples + spike_length_samples
        ] += templates[l]
    rec = si.NumpyRecording(rec, 30_000)
    rec.set_dummy_probe_from_locations(geom)

    template_config = config.TemplateConfig(
        low_rank_denoising=False,
        superres_bin_min_spikes=0,
    )
    template_data = TemplateData.from_config(
        *no_overlap_recording_sorting(templates),
        template_config,
        motion_est=motion_util.IdentityMotionEstimate(),
        n_jobs=0,
        save_folder=tmp_path,
        overwrite=True,
    )

    matcher = main.ObjectiveUpdateTemplateMatchingPeeler.from_config(
        rec,
        main.MatchingConfig(
            threshold=0.01,
            template_temporal_upsampling_factor=1,
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

    ixa, ixb, pconv = matcher.pairwise_conv_db.query(
        [0, 1], [0, 1], upsampling_indices_b=[0, 0], grid=True
    )
    maxpc = pconv.max(dim=1).values
    for ia, ib, pc in zip(ixa, ixb, maxpc):
        assert np.isclose(pc, (templates[ia] * templates[ib]).sum())
    assert res["n_spikes"] == len(times)
    assert np.array_equal(res["times_samples"], times)
    assert np.array_equal(res["labels"], labels)
    assert np.isclose(
        torch.square(res["residual"]).mean(),
        0.0,
    )
    assert np.isclose(
        torch.square(res["conv"]).mean(),
        0.0,
        atol=1e-5,
    )

    matcher = main.ObjectiveUpdateTemplateMatchingPeeler.from_config(
        rec,
        main.MatchingConfig(
            threshold=0.01,
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
    assert np.isclose(
        torch.square(res["residual"]).mean(),
        0.0,
    )
    assert np.isclose(
        torch.square(res["conv"]).mean(),
        0.0,
        atol=1e-5,
    )


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
    rec = np.zeros((recording_length_samples, n_channels), dtype="float32")
    for t, l in zip(times, labels):
        rec[
            t - trough_offset_samples : t - trough_offset_samples + spike_length_samples
        ] += templates[l]
    rec = si.NumpyRecording(rec, 30_000)
    rec.set_dummy_probe_from_locations(geom)

    template_config = config.TemplateConfig(
        low_rank_denoising=False, superres_bin_min_spikes=0
    )
    template_data = TemplateData.from_config(
        *no_overlap_recording_sorting(templates),
        template_config,
        motion_est=motion_util.IdentityMotionEstimate(),
        n_jobs=0,
        save_folder=tmp_path,
        overwrite=True,
    )

    matcher = main.ObjectiveUpdateTemplateMatchingPeeler.from_config(
        rec,
        main.MatchingConfig(
            threshold=0.01,
            template_temporal_upsampling_factor=up_factor,
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
        ptps=template_data.templates.ptp(1).max(1),
        max_upsample=up_factor,
    )
    assert np.array_equal(matcher.compressed_upsampled_temporal, tempup.compressed_upsampled_templates)
    assert np.array_equal(matcher.objective_spatial_components, lrt.spatial_components)
    assert np.array_equal(matcher.objective_singular_values, lrt.singular_values)
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
            tupb = (tempupb * lrt.singular_values[ib]) @ lrt.spatial_components[ib]
            tc = (templates[ia] * tupb).sum()

            template_a = torch.as_tensor(templates[ia][None])
            ssb = lrt.singular_values[ib][:, None] * lrt.spatial_components[ib]
            conv_filt = torch.bmm(torch.as_tensor(ssb[None]), template_a.mT)
            conv_filt = conv_filt[:, None]  # (nco, 1, rank, t)
            conv_in = torch.as_tensor(tempupb[None]).mT[None]
            pconv_ = F.conv2d(
                conv_in, conv_filt, padding=(0, 120), groups=1
            )
            pconv1 = pconv_.squeeze()[spike_length_samples - 1].numpy(force=True)
            assert torch.isclose(pcf, pconv_).all()

            pconv2 = F.conv2d(
                torch.as_tensor(templates[ia])[None, None],
                torch.as_tensor(tupb)[None, None],
            ).squeeze().numpy(force=True)
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
    assert np.isclose(
        torch.square(res["residual"]).mean(),
        0.0,
    )
    assert np.isclose(
        torch.square(res["conv"]).mean(),
        0.0,
        atol=1e-4,
    )


def test_static_noup(tmp_path):
    static_tester(tmp_path)


def test_static_up(tmp_path):
    static_tester(tmp_path, up_factor=4)


def drifting_tester(tmp_path, up_factor=1):
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
        25000, 1, 2,
        25001, 0, 1,
    ]
    # fmt: on
    times, channels, labels = np.array(tcl).reshape(-1, 3).T
    rec = np.zeros((recording_length_samples, n_channels), dtype="float32")
    for t, l in zip(times, labels):
        rec[
            t - trough_offset_samples : t - trough_offset_samples + spike_length_samples
        ] += templates[l]
    rec = si.NumpyRecording(rec, 30_000)
    rec.set_dummy_probe_from_locations(geom)

    template_config = config.TemplateConfig(
        low_rank_denoising=False, superres_bin_min_spikes=0
    )
    template_data = TemplateData.from_config(
        *no_overlap_recording_sorting(templates),
        template_config,
        motion_est=motion_util.IdentityMotionEstimate(),
        n_jobs=0,
        save_folder=tmp_path,
        overwrite=True,
    )

    matcher = main.ObjectiveUpdateTemplateMatchingPeeler.from_config(
        rec,
        main.MatchingConfig(
            threshold=0.01,
            template_temporal_upsampling_factor=up_factor,
            coarse_approx_error_threshold=0.0,
            conv_ignore_threshold=0.0,
            template_svd_compression_rank=2,
        ),
        nofeatcfg,
        template_data,
        motion_est=motion_util.IdentityMotionEstimate(),
    )
    matcher.precompute_peeling_data(tmp_path)

    # tup = template_util.compressed_upsampled_templates(
    #     template_data.templates, max_upsample=up_factor
    # )
    lrt = template_util.svd_compress_templates(
        template_data.templates, rank=matcher.svd_compression_rank
    )
    tempup = template_util.compressed_upsampled_templates(
        lrt.temporal_components,
        ptps=template_data.templates.ptp(1).max(1),
        max_upsample=up_factor,
    )
    print(f"{lrt.temporal_components.shape=}")
    print(f"{lrt.singular_values.shape=}")
    print(f"{lrt.spatial_components.shape=}")
    assert np.array_equal(matcher.compressed_upsampled_temporal, tempup.compressed_upsampled_templates)
    assert np.array_equal(matcher.objective_spatial_components, lrt.spatial_components)
    assert np.array_equal(matcher.objective_singular_values, lrt.singular_values)
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
            # tupb = tup.compressed_upsampled_templates[
            #     tup.compressed_upsampling_map[ib, up]
            # ]
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
            pconv_ = F.conv2d(
                conv_in, conv_filt, padding=(0, 120), groups=1
            )
            # print(f"{torch.abs(pcf - pconv_).max()=}")
            pconv1 = pconv_.squeeze()[spike_length_samples - 1].numpy(force=True)
            assert torch.isclose(pcf, pconv_).all()

            pconv2 = F.conv2d(
                torch.as_tensor(templates[ia])[None, None],
                torch.as_tensor(tupb)[None, None],
            ).squeeze().numpy(force=True)
            assert np.isclose(pconv2, tc)

            # print(f" - {ia=} {ib=}")
            # print(f"   {pc=} {tc=} {pconv1=} {pconv2=}")
            # print(f"   {pcf[120]=} {pcf[121]=} {pcf[122]=}")
            # print(f" ~ {np.isclose(pc, tc)=}")
            # print(f"   {np.isclose(pconv1, pc)=} {np.isclose(tc, pconv2)=}")
            assert np.isclose(pc, tc)
            assert np.isclose(pconv1, pc)

    res = matcher.peel_chunk(
        torch.from_numpy(rec.get_traces().copy()),
        return_residual=True,
        return_conv=True,
    )

    print()
    print()
    print(f"{len(times)=}")
    print(f"{res['n_spikes']=}")
    print()
    print(f'{torch.square(res["residual"]).mean()=}')
    print(f'{torch.abs(res["residual"]).max()=}')
    print(f'{torch.square(res["conv"]).mean()=}')
    print(f'{torch.abs(res["conv"]).max()=}')
    print(f'{res["conv"].min()=} {res["conv"].max()=}')
    tnsq = np.linalg.norm(templates, axis=(1, 2)) ** 2
    print(f"{res['conv'].shape=} {tnsq.shape=}")
    print(f'{(2*res["conv"] - tnsq[:,None]).max()=}')
    print()
    print(f'{res["times_samples"]=}')
    print(f"{times=}")
    print()
    print(f'{res["labels"]=}')
    print(f"{labels=}")
    print()
    print(f'{np.c_[res["times_samples"], res["labels"], res["upsampling_indices"]]=}')
    print(f"{np.c_[times, labels]=}")
    print()
    print(f'{res["upsampling_indices"]=}')

    assert res["n_spikes"] == len(times)
    assert np.array_equal(res["times_samples"], times)
    assert np.array_equal(res["labels"], labels)
    print(f"{torch.square(res['residual']).mean()=}")
    print(f"{torch.square(res['conv']).mean()=}")
    assert np.isclose(
        torch.square(res["residual"]).mean(),
        0.0,
    )
    assert np.isclose(
        torch.square(res["conv"]).mean(),
        0.0,
        atol=1e-4,
    )


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("test tiny")
    with tempfile.TemporaryDirectory() as tdir:
        test_tiny(Path(tdir))

    print()
    print("test test_static_noup")
    with tempfile.TemporaryDirectory() as tdir:
        test_static_noup(Path(tdir))

    print()
    print("test test_static_up")
    with tempfile.TemporaryDirectory() as tdir:
        test_static_up(Path(tdir))
