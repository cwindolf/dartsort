import dataclasses
import tempfile
from typing import cast
import pytest

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dartsort.util.internal_config import (
    FeaturizationConfig,
    SubtractionConfig,
    ComputationConfig,
)
from dartsort.localize.localize_torch import point_source_amplitude_at
from dartsort.main import subtract
from dartsort.util import waveform_util
from test_util import dense_layout

fixedlenkeys = (
    "subtract_channel_index",
    "channel_index",
    "geom",
    "residual",
    "residual_times_seconds",
)

two_jobs_cfg_cpu = ComputationConfig(n_jobs_cpu=2, n_jobs_gpu=2, device="cpu")
two_jobs_cfg = ComputationConfig(n_jobs_cpu=2, n_jobs_gpu=2)
two_jobs_cfg_spawn = ComputationConfig(
    n_jobs_cpu=2, n_jobs_gpu=2, executor="ProcessPoolExecutor"
)


def test_fakedata_nonn(tmp_path):
    print("test_fakedata_nonn")
    # generate fake neuropixels data with artificial templates
    T_s = 15.5
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
    templates[0] *= 100 / np.abs(templates[0]).max()
    templates[1] *= 50 / np.abs(templates[1]).max()
    templates[2] *= 100 / np.abs(templates[2]).max()
    templates[3] *= 50 / np.abs(templates[3]).max()

    # make fake spike trains
    spikes_per_unit = 51
    refrac_t = 100
    t_remaining = T_samples - spikes_per_unit * refrac_t - 2 * 121 - 1
    mnp = np.ones(spikes_per_unit) / spikes_per_unit
    assert t_remaining > spikes_per_unit
    sts = []
    labels = []
    for i in range(len(templates)):
        dt = refrac_t + rg.multinomial(t_remaining, mnp)
        assert dt.shape == (spikes_per_unit,)
        st = 121 + np.cumsum(dt)
        assert st.max() < T_samples - 121
        sts.append(st)
        labels.append(np.full((spikes_per_unit,), i))
    times = np.concatenate(sts)
    labels = np.concatenate(labels)
    gt_times = np.sort(times) + 42

    # inject the spikes into a noise background
    rec = 0.1 * rg.normal(size=(T_samples, len(geom))).astype(np.float32)
    for t, l in zip(times, labels):
        rec[t : t + 121] += templates[l]
    assert np.sum(np.abs(rec) > 80) >= 100
    assert np.sum(np.abs(rec) > 40) >= 50

    # make into spikeinterface
    rec = sc.NumpyRecording(rec, fs)
    rec.set_dummy_probe_from_locations(geom)

    subconf = SubtractionConfig(
        detection_threshold=20.0,
        peak_sign="both",
        subtraction_denoising_cfg=FeaturizationConfig(
            do_nn_denoise=False, denoise_only=True
        ),
        first_denoiser_thinning=0.0,
        first_denoiser_spatial_jitter=0,
        first_denoiser_temporal_jitter=0,
        convexity_threshold=-100.0,
    )
    featconf = FeaturizationConfig(do_nn_denoise=False, n_residual_snips=8)
    channel_index = waveform_util.make_channel_index(geom, featconf.extract_radius)
    assert channel_index.shape[0] == len(geom)
    assert channel_index.max() == len(geom)
    assert channel_index.min() == 0

    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        print("first one")
        torch.manual_seed(0)
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            overwrite=True,
        )
        assert st is not None
        out_h5 = st.parent_h5_path
        ns0 = len(st)
        print(ns0)
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
            assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]  # type: ignore[reportAttributeAccessIssue]
            assert np.array_equal(h5["channel_index"][:], channel_index)  # type: ignore[reportAttributeAccessIssue]
            assert h5["collisioncleaned_tpca_features"].shape == (  # type: ignore[reportAttributeAccessIssue]
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )
            assert np.array_equal(h5["geom"][()], geom)  # type: ignore[reportAttributeAccessIssue]
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)  # type: ignore[reportAttributeAccessIssue]

        # test that resuming works
        print("resume")
        torch.manual_seed(0)
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            overwrite=False,
        )
        assert st is not None
        ns1 = len(st)
        out_h5 = st.parent_h5_path
        assert ns0 == ns1
        print(ns1)
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
            assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]  # type: ignore[reportAttributeAccessIssue]
            assert np.array_equal(h5["channel_index"][:], channel_index)  # type: ignore[reportAttributeAccessIssue]
            assert np.array_equal(h5["geom"][()], geom)  # type: ignore[reportAttributeAccessIssue]
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)  # type: ignore[reportAttributeAccessIssue]
            assert h5["collisioncleaned_tpca_features"].shape == (  # type: ignore[reportAttributeAccessIssue]
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )

        # test overwrite
        print("overwrite")
        torch.manual_seed(0)
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            overwrite=True,
        )
        assert st is not None
        out_h5 = st.parent_h5_path
        ns2 = len(st)
        print(ns2)
        assert ns0 == ns2
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]´
            assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]´
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]  # type: ignore[reportAttributeAccessIssue]´
            assert np.array_equal(h5["channel_index"][:], channel_index)  # type: ignore[reportAttributeAccessIssue]´
            assert np.array_equal(h5["geom"][()], geom)  # type: ignore[reportAttributeAccessIssue]´
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)  # type: ignore[reportAttributeAccessIssue]´
            assert h5["collisioncleaned_tpca_features"].shape == (  # type: ignore[reportAttributeAccessIssue]´
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )

    for ccfg in (two_jobs_cfg, two_jobs_cfg_spawn):
        print(f"---- {ccfg=}")
        with tempfile.TemporaryDirectory(
            dir=tmp_path, ignore_cleanup_errors=True
        ) as tempdir:
            print("parallel first one")
            torch.manual_seed(0)
            st = subtract(
                recording=rec,
                output_dir=tempdir,
                featurization_cfg=featconf,
                subtraction_cfg=subconf,
                overwrite=True,
                computation_cfg=ccfg,
            )
            assert st is not None
            out_h5 = st.parent_h5_path
            ns0 = len(st)
            print(ns0)
            assert ns2 == ns0
            with h5py.File(out_h5, locking=False) as h5:
                assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]  # type: ignore[reportAttributeAccessIssue]
                assert np.array_equal(h5["channel_index"][:], channel_index)  # type: ignore[reportAttributeAccessIssue]
                assert h5["collisioncleaned_tpca_features"].shape == (  # type: ignore[reportAttributeAccessIssue]
                    ns0,
                    featconf.tpca_rank,
                    channel_index.shape[1],
                )
                assert np.array_equal(h5["geom"][()], geom)  # type: ignore[reportAttributeAccessIssue]
                assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)  # type: ignore[reportAttributeAccessIssue]

            # test that resuming works
            print("parallel resume")
            torch.manual_seed(0)
            st = subtract(
                recording=rec,
                output_dir=tempdir,
                featurization_cfg=featconf,
                subtraction_cfg=subconf,
                overwrite=False,
                computation_cfg=ccfg,
            )
            assert st is not None
            out_h5 = st.parent_h5_path
            ns1 = len(st)
            print(ns1)
            assert ns0 == ns1
            with h5py.File(out_h5, locking=False) as h5:
                assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]  # type: ignore[reportAttributeAccessIssue]
                assert np.array_equal(h5["channel_index"][:], channel_index)  # type: ignore[reportAttributeAccessIssue]
                assert np.array_equal(h5["geom"][()], geom)  # type: ignore[reportAttributeAccessIssue]
                assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)  # type: ignore[reportAttributeAccessIssue]
                assert h5["collisioncleaned_tpca_features"].shape == (  # type: ignore[reportAttributeAccessIssue]
                    ns0,
                    featconf.tpca_rank,
                    channel_index.shape[1],
                )

            # test overwrite
            print("parallel overwrite")
            torch.manual_seed(0)
            st = subtract(
                recording=rec,
                output_dir=tempdir,
                featurization_cfg=featconf,
                subtraction_cfg=subconf,
                overwrite=True,
                computation_cfg=ccfg,
            )
            assert st is not None
            out_h5 = st.parent_h5_path
            ns2 = len(st)
            print(ns2)
            assert ns0 == ns2
            with h5py.File(out_h5, locking=False) as h5:
                assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]  # type: ignore[reportAttributeAccessIssue]
                assert np.array_equal(h5["channel_index"][:], channel_index)  # type: ignore[reportAttributeAccessIssue]
                assert np.array_equal(h5["geom"][()], geom)  # type: ignore[reportAttributeAccessIssue]
                assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)  # type: ignore[reportAttributeAccessIssue]
                assert h5["collisioncleaned_tpca_features"].shape == (  # type: ignore[reportAttributeAccessIssue]
                    ns0,
                    featconf.tpca_rank,
                    channel_index.shape[1],
                )

    # simulate resuming a job that got cancelled in the middle
    print("test resume1")
    subconf = dataclasses.replace(
        subconf,
        subtraction_denoising_cfg=dataclasses.replace(
            subconf.subtraction_denoising_cfg,
            do_nn_denoise=True,
            do_tpca_denoise=False,
        ),
    )
    nolocfeatconf = dataclasses.replace(featconf, do_localization=False)
    print(subconf)
    print(nolocfeatconf)
    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        st0 = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=nolocfeatconf,
            subtraction_cfg=subconf,
        )
        assert st0 is not None
        ns0 = len(st0)
        print(ns0)

    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        sta = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=nolocfeatconf,
            subtraction_cfg=subconf,
            chunk_starts_samples=np.arange(3) * int(fs),
        )
        assert sta is not None
        with h5py.File(sta.parent_h5_path, locking=False) as h5:
            assert h5["last_chunk_start"][()] == int(2 * fs)   # type: ignore[reportAttributeAccessIssue]
        stb = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=nolocfeatconf,
            subtraction_cfg=subconf,
        )
        assert stb is not None
        print(f"{len(sta)=} {len(stb)=}")
        print(f"{np.setdiff1d(st0.times_samples, stb.times_samples)=}")
        assert len(sta) < ns0
        assert len(stb) == ns0
        np.testing.assert_array_equal(st0.times_samples, stb.times_samples)
        np.testing.assert_array_equal(st0.channels, stb.channels)


@pytest.mark.parametrize("nn_localization", [False, True])
def test_small_nonn(tmp_path, nn_localization):
    # noise recording
    T_samples = 50_100
    n_channels = 50
    rg = np.random.default_rng(0)
    noise = rg.normal(size=(T_samples, n_channels)).astype(np.float32)

    # add a spike every so_often samples
    so_often = 501
    template = 50 * np.exp(-(((np.arange(121) - 42) / 10) ** 2))
    for t in range(0, T_samples - 121, so_often):
        random_channel = rg.integers(n_channels)
        noise[t : t + 121, random_channel] += template

    h = dense_layout()
    geom = np.c_[h["x"], h["y"]][:n_channels]
    rec = sc.NumpyRecording(noise, 30_000)
    rec.set_dummy_probe_from_locations(geom)

    subconf = SubtractionConfig(
        detection_threshold=40.0,
        peak_sign="both",
        subtraction_denoising_cfg=FeaturizationConfig(
            do_nn_denoise=False, denoise_only=True
        ),
    )
    featconf = FeaturizationConfig(
        do_nn_denoise=False, n_residual_snips=8, nn_localization=nn_localization
    )

    print("No parallel")
    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            overwrite=True,
        )
        assert st is not None
        with h5py.File(st.parent_h5_path, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:   # type: ignore[reportAttributeAccessIssue]
                    lens.append(h5[k].shape[0])   # type: ignore[reportAttributeAccessIssue]
            assert np.unique(lens).size == 1

    print("CPU parallel")
    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        # test default config
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            overwrite=True,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            computation_cfg=two_jobs_cfg_cpu,
        )
        assert st is not None
        with h5py.File(st.parent_h5_path, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:   # type: ignore[reportAttributeAccessIssue]
                    lens.append(h5[k].shape[0])   # type: ignore[reportAttributeAccessIssue]
            assert np.unique(lens).size == 1

    print("Yes parallel")
    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        # test default config
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            overwrite=True,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            computation_cfg=two_jobs_cfg,
        )
        assert st is not None
        out_h5 = st.parent_h5_path
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:   # type: ignore[reportAttributeAccessIssue]
                    lens.append(h5[k].shape[0])   # type: ignore[reportAttributeAccessIssue]
            assert np.unique(lens).size == 1


def test_small_default_config(tmp_path, extract_radius=100):
    # noise recording
    T_samples = 50_100
    n_channels = 50
    rg = np.random.default_rng(0)
    noise = rg.normal(size=(T_samples, n_channels)).astype(np.float32)

    # add a spike every so often samples
    so_often = 501
    template = -20 * np.exp(-(((np.arange(121) - 42) / 5) ** 2))
    gt_times = []
    gt_channels = []
    for t in range(0, T_samples - 121, so_often):
        random_channel = rg.integers(n_channels)
        noise[t : t + 121, random_channel] += template
        gt_times.append(t + 42)
        gt_channels.append(random_channel)
    gt_times = np.array(gt_times)
    gt_channels = np.array(gt_channels)

    h = dense_layout()
    geom = np.c_[h["x"], h["y"]][:n_channels]
    rec = sc.NumpyRecording(noise, 30_000)
    rec.set_dummy_probe_from_locations(geom)

    cfg = SubtractionConfig(detection_threshold=15.0, convexity_threshold=-100)
    fcfg = FeaturizationConfig(extract_radius=extract_radius, n_residual_snips=8)

    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        # test default config
        print("test_small_default_config first")
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            overwrite=True,
            subtraction_cfg=cfg,
            featurization_cfg=fcfg,
        )
        assert st is not None
        out_h5 = st.parent_h5_path
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:  # type: ignore[reportAttributeAccessIssue]
                    lens.append(h5[k].shape[0])  # type: ignore[reportAttributeAccessIssue]
            h5_times = cast(h5py.Dataset, h5["times_samples"])[:]
            h5_channels = cast(h5py.Dataset, h5["channels"])[:]
            np.testing.assert_allclose(h5_times, gt_times, atol=3)
            np.testing.assert_array_equal(h5_channels, gt_channels)
            assert np.unique(lens).size == 1
            assert lens[0] == len(gt_times)

        # test default config
        print("test_small_default_config second")
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            overwrite=True,
            computation_cfg=two_jobs_cfg,
            subtraction_cfg=cfg,
            featurization_cfg=fcfg,
        )
        assert st is not None
        out_h5 = st.parent_h5_path
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:  # type: ignore[reportAttributeAccessIssue]
                    lens.append(h5[k].shape[0])  # type: ignore[reportAttributeAccessIssue]
            assert (np.abs(h5["times_samples"][:] - gt_times) <= 3).all()  # type: ignore
            assert np.unique(lens).size == 1
            assert np.array_equal(h5["channels"][:], gt_channels)  # type: ignore[reportAttributeAccessIssue]
            assert lens[0] == len(gt_times)
