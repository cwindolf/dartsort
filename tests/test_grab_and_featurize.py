"""A good integration test of a few pieces
"""
import tempfile
from pathlib import Path

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dartsort import transform
from dartsort.peel.grab import GrabAndFeaturize
from dartsort.util.waveform_util import make_channel_index


def test_grab_and_featurize():
    # noise recording
    T_samples = 100_100
    n_channels = 50
    rg = np.random.default_rng(0)
    noise = rg.normal(size=(T_samples, n_channels)).astype(np.float32)
    geom = rg.uniform(low=0, high=100, size=(n_channels, 2))
    rec = sc.NumpyRecording(noise, 10_000)
    rec.set_dummy_probe_from_locations(geom)

    # random spike times
    n_spikes = 50203
    times = rg.integers(100, T_samples - 100, size=n_spikes)
    channels = rg.integers(0, n_channels, size=n_spikes)

    # grab the wfs
    channel_index = make_channel_index(geom, 20)
    pipeline = transform.WaveformPipeline([transform.Waveform(channel_index)])
    grab = GrabAndFeaturize(
        rec,
        torch.as_tensor(channel_index),
        pipeline,
        torch.as_tensor(times),
        torch.as_tensor(channels),
    )

    with tempfile.TemporaryDirectory() as tempdir:
        grab.peel(Path(tempdir) / "grab.h5")

        with h5py.File(Path(tempdir) / "grab.h5") as h5:
            assert h5["times"].shape == (n_spikes,)
            assert h5["channels"].shape == (n_spikes,)
            assert h5["waveforms"].shape == (
                n_spikes,
                121,
                channel_index.shape[1],
            )
            assert np.array_equal(h5["geom"][()], geom)
            assert np.array_equal(h5["channel_index"][()], channel_index)
            assert h5["last_chunk_start"][()] == 90_000

    # try one with TPCA
    channel_index = make_channel_index(geom, 20)
    pipeline = transform.WaveformPipeline(
        [
            transform.Waveform(channel_index),
            transform.TemporalPCADenoiser(
                channel_index=torch.tensor(channel_index),
                geom=torch.tensor(geom),
                fit_radius=10,
            ),
            transform.Waveform(channel_index, name="tpca_waveforms"),
        ]
    )
    grab = GrabAndFeaturize(
        rec,
        torch.as_tensor(channel_index),
        pipeline,
        torch.as_tensor(times),
        torch.as_tensor(channels),
    )

    with tempfile.TemporaryDirectory() as tempdir:
        grab.fit_models(tempdir)
        grab.peel(Path(tempdir) / "grab.h5")

        with h5py.File(Path(tempdir) / "grab.h5") as h5:
            assert h5["times"].shape == (n_spikes,)
            assert h5["channels"].shape == (n_spikes,)
            assert h5["waveforms"].shape == (
                n_spikes,
                121,
                channel_index.shape[1],
            )
            assert h5["tpca_waveforms"].shape == (
                n_spikes,
                121,
                channel_index.shape[1],
            )
            assert np.array_equal(h5["geom"][()], geom)
            assert np.array_equal(h5["channel_index"][()], channel_index)
            assert h5["last_chunk_start"][()] == 90_000


if __name__ == "__main__":
    test_grab_and_featurize()
