import numpy as np
import spikeinterface.core as sc
import torch
from dartsort.config import FeaturizationConfig, SubtractionConfig
from dartsort.localize.localize_torch import point_source_amplitude_at
from dartsort.main import subtract
from neuropixel import dense_layout


def test_fakedata_nonn():
    # generate fake neuropixels data with sin/cos templates
    T_s = 49.5
    fs = 30000
    T_samples = int(fs * T_s)
    rg = np.random.default_rng(0)

    # np1 geom
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]]

    # template main channel traces
    t0 = np.sin(np.arange(121) / np.pi)
    t1 = np.sin(np.arange(121) / np.pi)
    t2 = np.sin(0.5 * np.arange(121) / np.pi)
    t3 = np.cos(0.5 * np.arange(121) / np.pi)

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
    templates = 50 * np.array(
        [t[:, None] * a[None, :] for t, a in zip((t0, t1, t2, t3), amps)]
    )

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
    rec = rg.normal(size=(T_samples, len(geom))).astype(np.float32)
    for t, l in zip(times, labels):
        rec[t : t + 121] += templates[l]

    # make into spikeinterface
    rec = sc.NumpyRecording(rec, fs)
    rec.set_dummy_probe_from_locations(geom)

    subconf = SubtractionConfig(
        subtraction_denoising_config=FeaturizationConfig(
            do_nn_denoise=False, do_featurization=False
        )
    )
    featconf = FeaturizationConfig(do_nn_denoise=False)
    print(subconf)
    print(featconf)

    st = subtract(
        rec,
        "/tmp/test.h5",
        featurization_config=featconf,
        subtraction_config=subconf,
    )
    print(st)
