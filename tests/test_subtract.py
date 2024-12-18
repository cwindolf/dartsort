import dataclasses
import tempfile

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dartsort.config import FeaturizationConfig, SubtractionConfig, ComputationConfig
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


def test_fakedata_nonn():
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
    templates[0] *= 100 / np.abs(templates[0]).max()
    templates[1] *= 50 / np.abs(templates[1]).max()
    templates[2] *= 100 / np.abs(templates[2]).max()
    templates[3] *= 50 / np.abs(templates[3]).max()

    # make fake spike trains
    spikes_per_unit = 51
    sts = []
    labels = []
    for i in range(len(templates)):
        while True:
            st = rg.choice(T_samples - 121, size=spikes_per_unit)
            st.sort()
            if np.diff(st).min() > 15:
                sts.append(st)
                break
        labels.append(np.full((spikes_per_unit,), i))
    times = np.concatenate(sts)
    labels = np.concatenate(labels)

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
        subtraction_denoising_config=FeaturizationConfig(
            do_nn_denoise=False, denoise_only=True
        ),
    )
    featconf = FeaturizationConfig(do_nn_denoise=False, n_residual_snips=8)
    channel_index = waveform_util.make_channel_index(geom, subconf.extract_radius)
    assert channel_index.shape[0] == len(geom)
    assert channel_index.max() == len(geom)
    assert channel_index.min() == 0

    with tempfile.TemporaryDirectory() as tempdir:
        print("first one")
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=True,
        )
        ns0 = len(st)
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)
            assert h5["channels"].shape == (ns0,)
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]
            assert np.array_equal(h5["channel_index"][:], channel_index)
            assert h5["collisioncleaned_tpca_features"].shape == (
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )
            assert np.array_equal(h5["geom"][()], geom)
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)

        # test that resuming works
        print("resume")
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=False,
        )
        ns1 = len(st)
        assert ns0 == ns1
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)
            assert h5["channels"].shape == (ns0,)
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]
            assert np.array_equal(h5["channel_index"][:], channel_index)
            assert np.array_equal(h5["geom"][()], geom)
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)
            assert h5["collisioncleaned_tpca_features"].shape == (
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )

        # test overwrite
        print("overwrite")
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=True,
        )
        ns2 = len(st)
        assert ns0 == ns2
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)
            assert h5["channels"].shape == (ns0,)
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]
            assert np.array_equal(h5["channel_index"][:], channel_index)
            assert np.array_equal(h5["geom"][()], geom)
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)
            assert h5["collisioncleaned_tpca_features"].shape == (
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )

    with tempfile.TemporaryDirectory() as tempdir:
        print("parallel first one")
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=True,
            computation_config=two_jobs_cfg,
        )
        ns0 = len(st)
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)
            assert h5["channels"].shape == (ns0,)
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]
            assert np.array_equal(h5["channel_index"][:], channel_index)
            assert h5["collisioncleaned_tpca_features"].shape == (
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )
            assert np.array_equal(h5["geom"][()], geom)
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)

        # test that resuming works
        print("parallel resume")
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=False,
            computation_config=two_jobs_cfg,
        )
        ns1 = len(st)
        assert ns0 == ns1
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)
            assert h5["channels"].shape == (ns0,)
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]
            assert np.array_equal(h5["channel_index"][:], channel_index)
            assert np.array_equal(h5["geom"][()], geom)
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)
            assert h5["collisioncleaned_tpca_features"].shape == (
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )

        # test overwrite
        print("parallel overwrite")
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=True,
            computation_config=two_jobs_cfg,
        )
        ns2 = len(st)
        assert ns0 == ns2
        with h5py.File(out_h5, locking=False) as h5:
            assert h5["times_samples"].shape == (ns0,)
            assert h5["channels"].shape == (ns0,)
            assert h5["point_source_localizations"].shape in [(ns0, 4), (ns0, 3)]
            assert np.array_equal(h5["channel_index"][:], channel_index)
            assert np.array_equal(h5["geom"][()], geom)
            assert h5["last_chunk_start"][()] == int(np.floor(T_s) * fs)
            assert h5["collisioncleaned_tpca_features"].shape == (
                ns0,
                featconf.tpca_rank,
                channel_index.shape[1],
            )

    # simulate resuming a job that got cancelled in the middle
    print("test resume1")
    subconf = dataclasses.replace(
        subconf,
        subtraction_denoising_config=dataclasses.replace(
            subconf.subtraction_denoising_config,
            do_nn_denoise=True,
            do_tpca_denoise=False,
        ),
    )
    nolocfeatconf = dataclasses.replace(featconf, do_localization=False)
    print(subconf)
    print(nolocfeatconf)
    with tempfile.TemporaryDirectory() as tempdir:
        st0, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=nolocfeatconf,
            subtraction_config=subconf,
        )
        ns0 = len(st0)

    with tempfile.TemporaryDirectory() as tempdir:
        sta, out_h5 = subtract(
            rec.frame_slice(start_frame=0, end_frame=int((T_s // 3) * fs)),
            tempdir,
            featurization_config=nolocfeatconf,
            subtraction_config=subconf,
        )
        stb, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=nolocfeatconf,
            subtraction_config=subconf,
        )
        assert len(sta) < ns0
        assert len(stb) == ns0


def test_small_nonn():
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
        subtraction_denoising_config=FeaturizationConfig(
            do_nn_denoise=False, denoise_only=True
        ),
    )
    featconf = FeaturizationConfig(do_nn_denoise=False, n_residual_snips=8)

    print("No parallel")
    with tempfile.TemporaryDirectory() as tempdir:
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=True,
        )
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:
                    lens.append(h5[k].shape[0])
            assert np.unique(lens).size == 1

    print("CPU parallel")
    with tempfile.TemporaryDirectory() as tempdir:
        # test default config
        st, out_h5 = subtract(
            rec,
            tempdir,
            overwrite=True,
            featurization_config=featconf,
            subtraction_config=subconf,
            computation_config=two_jobs_cfg_cpu,
        )
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:
                    lens.append(h5[k].shape[0])
            assert np.unique(lens).size == 1

    print("Yes parallel")
    with tempfile.TemporaryDirectory() as tempdir:
        # test default config
        st, out_h5 = subtract(
            rec,
            tempdir,
            overwrite=True,
            featurization_config=featconf,
            subtraction_config=subconf,
            computation_config=two_jobs_cfg,
        )
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:
                    lens.append(h5[k].shape[0])
            assert np.unique(lens).size == 1


def small_default_config(extract_radius=200):
    # noise recording
    T_samples = 50_100
    n_channels = 50
    rg = np.random.default_rng(0)
    noise = rg.normal(size=(T_samples, n_channels)).astype(np.float32)

    # add a spike every so often samples
    so_often = 501
    template = 20 * np.exp(-(((np.arange(121) - 42) / 10) ** 2))
    for t in range(0, T_samples - 121, so_often):
        random_channel = rg.integers(n_channels)
        noise[t : t + 121, random_channel] += template

    h = dense_layout()
    geom = np.c_[h["x"], h["y"]][:n_channels]
    rec = sc.NumpyRecording(noise, 30_000)
    rec.set_dummy_probe_from_locations(geom)

    cfg = SubtractionConfig(detection_threshold=10.0, extract_radius=extract_radius)

    with tempfile.TemporaryDirectory() as tempdir:
        # test default config
        print("test_small_default_config first")
        st, out_h5 = subtract(
            rec,
            tempdir,
            overwrite=True,
            subtraction_config=cfg,
        )
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:
                    lens.append(h5[k].shape[0])
            assert np.unique(lens).size == 1

        # test default config
        print("test_small_default_config second")
        st, out_h5 = subtract(
            rec,
            tempdir,
            overwrite=True,
            computation_config=two_jobs_cfg,
            subtraction_config=cfg,
        )
        with h5py.File(out_h5, locking=False) as h5:
            lens = []
            for k in h5.keys():
                if k not in fixedlenkeys and h5[k].ndim >= 1:
                    lens.append(h5[k].shape[0])
            assert np.unique(lens).size == 1


def test_small_default_config():
    small_default_config()


def test_small_default_config_subex():
    small_default_config(extract_radius=100.0)


if __name__ == "__main__":
    test_fakedata_nonn()
    test_small_nonn()
    test_small_default_config()
    test_small_default_config_subex()
