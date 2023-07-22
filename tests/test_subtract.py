import tempfile

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dartsort.config import FeaturizationConfig, SubtractionConfig
from dartsort.localize.localize_torch import point_source_amplitude_at
from dartsort.main import subtract
from neuropixel import dense_layout


def test_fakedata_nonn():
    # generate fake neuropixels data with artificial templates
    T_s = 49.5
    fs = 30000
    n_channels = 50
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
        point_source_amplitude_at(
            x, y, z, alpha, torch.as_tensor(geom)
        ).numpy()
        for x, y, z, alpha in torch.column_stack(
            list(map(torch.as_tensor, [xs, ys, z_abss, alphas]))
        )
    ]

    # combine to make templates
    templates = np.array(
        [t[:, None] * a[None, :] for t, a in zip((t0, t1, t2, t3), amps)]
    )
    templates[0] *= 100 / np.abs(templates[0]).max()
    templates[1] *= 50 / np.abs(templates[0]).max()
    templates[2] *= 100 / np.abs(templates[0]).max()
    templates[3] *= 50 / np.abs(templates[0]).max()

    # make fake spike trains
    spikes_per_unit = 500
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
    assert np.sum(np.abs(rec) > 80) >= 1000
    assert np.sum(np.abs(rec) > 40) >= 2000

    # make into spikeinterface
    rec = sc.NumpyRecording(rec, fs)
    rec.set_dummy_probe_from_locations(geom)

    subconf = SubtractionConfig(
        detection_thresholds=(80, 40),
        peak_sign="both",
        subtraction_denoising_config=FeaturizationConfig(
            do_nn_denoise=False, do_featurization=False
        ),
    )
    featconf = FeaturizationConfig(do_nn_denoise=False)

    with tempfile.TemporaryDirectory() as tempdir:
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=True,
        )
        ns0 = len(st)
        # add asserts to test that datasets are there

        # test that resuming works
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=False,
        )
        ns1 = len(st)
        assert ns0 == ns1

        # test parallel
        st, out_h5 = subtract(
            rec,
            tempdir,
            featurization_config=featconf,
            subtraction_config=subconf,
            overwrite=True,
            n_jobs=2,
        )
        ns2 = len(st)
        assert ns0 == ns2


def test_small_default_config():
    # noise recording
    T_samples = 100_100
    n_channels = 50
    rg = np.random.default_rng(0)
    noise = rg.normal(size=(T_samples, n_channels)).astype(np.float32)

    # add a spike every 50 samples
    template = 20 * np.exp(-(((np.arange(121) - 42) / 10) ** 2))
    for t in range(0, 100_100 - 121, 50):
        random_channel = rg.integers(n_channels)
        noise[t : t + 121, random_channel] += template

    h = dense_layout()
    geom = np.c_[h["x"], h["y"]][:n_channels]
    rec = sc.NumpyRecording(noise, 10_000)
    rec.set_dummy_probe_from_locations(geom)

    with tempfile.TemporaryDirectory() as tempdir:
        # test default config
        st, out_h5 = subtract(
            rec,
            tempdir,
            overwrite=True,
            n_jobs=2,
        )
        with h5py.File(out_h5) as h5:
            lens = []
            for k in h5.keys():
                if k not in ("channel_index", "geom") and h5[k].ndim >= 1:
                    lens.append(h5[k].shape[0])
            assert np.unique(lens).size == 1


if __name__ == "__main__":
    test_small_default_config()
    # test_fakedata_nonn()
