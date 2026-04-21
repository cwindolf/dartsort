import dataclasses
import tempfile
from typing import cast
from pathlib import Path

import h5py
import numpy as np
import pytest
import spikeinterface.core as sc
import torch
from test_util import dense_layout

from dartsort.localize.localize_torch import point_source_amplitude_at
from dartsort.main import subtract
from dartsort.util import waveform_util
from dartsort.util.internal_config import (
    ComputationConfig,
    FeaturizationConfig,
    FitSamplingConfig,
    SubtractionConfig,
)

fixedlenkeys = (
    "subtract_channel_index",
    "sub_channel_index",
    "channel_index",
    "geom",
    "residual",
    "residual_times_seconds",
    "chunk_starts_samples",
)

two_jobs_cfg_cpu = ComputationConfig(n_jobs_cpu=2, n_jobs_gpu=2, device="cpu")
two_jobs_cfg = ComputationConfig(n_jobs_cpu=2, n_jobs_gpu=2)
two_jobs_cfg_spawn = ComputationConfig(
    n_jobs_cpu=2, n_jobs_gpu=2, executor="ProcessPoolExecutor"
)


@pytest.fixture
def fakedata():
    T_s = 3.5
    fs = 30000
    n_channels = 25
    T_samples = int(fs * T_s)
    rg = np.random.default_rng(0)

    # np1 geom
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]][:n_channels]

    # template main channel traces
    t0 = np.exp(-(((np.arange(121) - 42) / 4) ** 2))
    t1 = np.exp(-(((np.arange(121) - 42) / 8) ** 2))
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
    refrac_t = 100
    times = np.arange(121, T_samples - 121, refrac_t)
    labels = rg.integers(len(templates), size=times.shape)

    # inject the spikes into a noise background
    rec = 0.1 * rg.normal(size=(T_samples, len(geom))).astype(np.float32)
    for t, ll in zip(times, labels):
        rec[t : t + 121] += templates[ll]
    assert np.sum(np.abs(rec) > 80) >= 100
    assert np.sum(np.abs(rec) > 40) >= 50

    # make into spikeinterface
    rec = sc.NumpyRecording(rec, fs)
    rec.set_dummy_probe_from_locations(geom)

    return rec, geom, T_s, fs


def test_fakedata_nonn(fakedata, tmp_path):
    rec, geom, T_s, fs = fakedata
    print("test_fakedata_nonn")
    # generate fake neuropixels data with artificial templates

    subconf = SubtractionConfig(
        detection_threshold=20.0,
        peak_sign="both",
        subtraction_denoising_cfg=FeaturizationConfig(
            do_nn_denoise=False, denoise_only=True
        ),
        first_denoiser_thinning=0.0,
        first_denoiser_spatial_jitter=0,
        first_denoiser_temporal_jitter=0,
    )
    featconf = FeaturizationConfig(do_nn_denoise=False)
    sampconf = FitSamplingConfig(n_residual_snips=8)
    nolocfeatconf = dataclasses.replace(featconf, do_localization=False)
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
            sampling_cfg=sampconf,
            overwrite=True,
        )
        assert st is not None
        out_h5 = st.parent_h5_path
        ns0 = len(st)
        subtraction_full_spike_count = len(st)
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
            sampling_cfg=sampconf,
            subtraction_cfg=subconf,
            overwrite=False,
        )
        assert st is not None
        ns1 = len(st)
        out_h5 = st.parent_h5_path
        assert subtraction_full_spike_count == ns1
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
            assert h5["residual"].shape[0] == 8
            assert h5["residual"].shape[1] == 121
            assert h5["residual"].shape[2] == geom.shape[0]

        # test overwrite
        print("overwrite")
        torch.manual_seed(0)
        st = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=featconf,
            sampling_cfg=FitSamplingConfig(n_residual_snips=0),
            subtraction_cfg=subconf,
            overwrite=True,
        )
        assert st is not None
        out_h5 = st.parent_h5_path
        ns2 = len(st)
        print(ns2)
        assert subtraction_full_spike_count == ns2
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
        with tempfile.TemporaryDirectory(
            dir=tmp_path, ignore_cleanup_errors=True
        ) as tempdir:
            print("parallel first one")
            torch.manual_seed(0)
            st = subtract(
                recording=rec,
                output_dir=tempdir,
                featurization_cfg=nolocfeatconf,
                sampling_cfg=FitSamplingConfig(n_residual_snips=0),
                subtraction_cfg=subconf,
                overwrite=True,
                computation_cfg=ccfg,
            )
            assert st is not None
            out_h5 = st.parent_h5_path
            ns0 = len(st)
            print(ns0)
            assert subtraction_full_spike_count == ns0
            with h5py.File(out_h5, locking=False) as h5:
                assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
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
                featurization_cfg=nolocfeatconf,
                sampling_cfg=FitSamplingConfig(n_residual_snips=0),
                subtraction_cfg=subconf,
                overwrite=False,
                computation_cfg=ccfg,
            )
            assert st is not None
            out_h5 = st.parent_h5_path
            ns1 = len(st)
            print(ns1)
            assert subtraction_full_spike_count == ns1
            with h5py.File(out_h5, locking=False) as h5:
                assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
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
                featurization_cfg=nolocfeatconf,
                sampling_cfg=FitSamplingConfig(n_residual_snips=0),
                subtraction_cfg=subconf,
                overwrite=True,
                computation_cfg=ccfg,
            )
            assert st is not None
            out_h5 = st.parent_h5_path
            ns2 = len(st)
            print(ns2)
            assert subtraction_full_spike_count == ns2
            with h5py.File(out_h5, locking=False) as h5:
                assert h5["times_samples"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert h5["channels"].shape == (ns0,)  # type: ignore[reportAttributeAccessIssue]
                assert np.array_equal(h5["channel_index"][:], channel_index)  # type: ignore[reportAttributeAccessIssue]
                assert np.array_equal(h5["geom"][()], geom)  # type: ignore[reportAttributeAccessIssue]
                assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)  # type: ignore[reportAttributeAccessIssue]
                assert h5["collisioncleaned_tpca_features"].shape == (  # type: ignore[reportAttributeAccessIssue]
                    ns0,
                    featconf.tpca_rank,
                    channel_index.shape[1],
                )


def test_resume(fakedata, tmp_path):
    rec, geom, T_s, fs = fakedata

    subconf = SubtractionConfig(
        detection_threshold=20.0,
        peak_sign="both",
        subtraction_denoising_cfg=FeaturizationConfig(
            do_nn_denoise=False, denoise_only=True
        ),
        first_denoiser_thinning=0.0,
        first_denoiser_spatial_jitter=0,
        first_denoiser_temporal_jitter=0,
    )
    featconf = FeaturizationConfig(skip=True)
    sampconf = FitSamplingConfig(n_residual_snips=0)
    channel_index = waveform_util.make_channel_index(geom, featconf.extract_radius)
    assert channel_index.shape[0] == len(geom)
    assert channel_index.max() == len(geom)
    assert channel_index.min() == 0

    with tempfile.TemporaryDirectory(
        dir=tmp_path, ignore_cleanup_errors=True
    ) as tempdir:
        # run 30% of recording, which rounds to 2 chunks, so that
        # the last chunk starts at int(fs)
        print(f"--- {rec=}")
        torch.manual_seed(0)
        st_orig = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            sampling_cfg=sampconf,
        )
        assert st_orig is not None
        subtraction_full_spike_count = len(st_orig)

        # remove output, but retain models so that same basis etc are used
        (Path(tempdir) / "subtraction.h5").unlink()

        # run 30% of recording, which rounds to 2 chunks, so that
        # the last chunk starts at int(fs)
        print(f"--- {rec=}")
        torch.manual_seed(0)
        sta = subtract(
            recording=rec,
            output_dir=tempdir,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            sampling_cfg=sampconf,
            ensure_coverage=0.3,
            stop_after_n_spikes=0,
        )
        print("i stop", flush=True)
        assert sta is not None
        with h5py.File(sta.parent_h5_path, locking=False) as h5:
            assert h5["last_chunk_index"][()] == 1
            assert h5["last_chunk_start"][()] >= 0
        # run the rest
        stb = subtract(
            recording=rec,
            output_dir=tempdir,
            sampling_cfg=sampconf,
            featurization_cfg=featconf,
            subtraction_cfg=subconf,
            shuffle=True,
        )
        assert stb is not None
        with h5py.File(stb.parent_h5_path, locking=False) as h5:
            assert h5["last_chunk_index"][()] == 3
            assert h5["last_chunk_start"][()] >= 0
        assert len(sta) < subtraction_full_spike_count
        assert len(stb) == subtraction_full_spike_count
        order = np.argsort(stb.times_samples, stable=True)
        assert st_orig.times_samples.shape == stb.times_samples.shape
        np.testing.assert_array_equal(st_orig.times_samples, stb.times_samples[order])
        np.testing.assert_array_equal(st_orig.channels, stb.channels[order])


@pytest.mark.parametrize("nn_localization", [True])
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
    featconf = FeaturizationConfig(do_nn_denoise=False, nn_localization=nn_localization)

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
            ll = None
            for k in h5.keys():
                if k in fixedlenkeys:
                    continue
                ds = cast(h5py.Dataset, h5[k])
                if ds.ndim < 1:
                    continue
                if ll is None:
                    ll = ds.shape[0]
                assert ll == ds.shape[0], f"{k} {ll}, {ds}"

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
                if k not in fixedlenkeys and h5[k].ndim >= 1:  # type: ignore[reportAttributeAccessIssue]
                    lens.append(h5[k].shape[0])  # type: ignore[reportAttributeAccessIssue]
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
                if k not in fixedlenkeys and h5[k].ndim >= 1:  # type: ignore[reportAttributeAccessIssue]
                    lens.append(h5[k].shape[0])  # type: ignore[reportAttributeAccessIssue]
            assert np.unique(lens).size == 1
