from test_util import dense_layout
import dartsort
import numpy as np
import spikeinterface.core as sc
import tempfile
import torch
from dartsort.localize.localize_torch import point_source_amplitude_at


def test_fakedata():
    print("test the full pipeline")
    # generate fake neuropixels data with artificial templates
    T_s = 3.5
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
    spikes_per_unit = 101
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
    order = np.argsort(times)
    times = times[order]
    labels = labels[order]

    # inject the spikes into a noise background
    rec = 0.1 * rg.normal(size=(T_samples, len(geom))).astype(np.float32)
    for t, l in zip(times, labels):
        rec[t : t + 121] += templates[l]
    assert np.sum(np.abs(rec) > 80) >= 100
    assert np.sum(np.abs(rec) > 40) >= 50

    # make into spikeinterface
    rec = sc.NumpyRecording(rec, fs)
    rec.set_dummy_probe_from_locations(geom)

    with tempfile.TemporaryDirectory() as tempdir:
        cfg = dartsort.DARTsortInternalConfig(
            subtraction_config=dartsort.SubtractionConfig(
                subtraction_denoising_config=dartsort.FeaturizationConfig(
                    denoise_only=True, do_nn_denoise=False
                )
            ),
            featurization_config=dartsort.FeaturizationConfig(n_residual_snips=128),
            motion_estimation_config=dartsort.MotionEstimationConfig(
                do_motion_estimation=False
            ),
        )
        st = dartsort.dartsort(rec, output_directory=tempdir, cfg=cfg)
