"""A good integration test of a few pieces
"""
import tempfile
from pathlib import Path

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dartsort import transform
from dartsort.localize.localize_util import localize_hdf5
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

    # random spike times_samples
    n_spikes = 50203
    times_samples = rg.integers(100, T_samples - 100, size=n_spikes)
    channels = rg.integers(0, n_channels, size=n_spikes)

    # grab the wfs
    channel_index = make_channel_index(geom, 20)
    pipeline = transform.WaveformPipeline([transform.Waveform(channel_index)])
    grab = GrabAndFeaturize(
        rec,
        torch.as_tensor(channel_index),
        pipeline,
        torch.as_tensor(times_samples),
        torch.as_tensor(channels),
    )

    with tempfile.TemporaryDirectory() as tempdir:
        grab.peel(Path(tempdir) / "grab.h5")

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert h5["times_samples"].shape == (n_spikes,)
            assert h5["channels"].shape == (n_spikes,)
            assert h5["waveforms"].shape == (
                n_spikes,
                121,
                channel_index.shape[1],
            )
            assert np.array_equal(h5["geom"][()], geom)
            assert np.array_equal(h5["channel_index"][()], channel_index)
            assert h5["last_chunk_start"][()] == 90_000

        grab.peel(Path(tempdir) / "grab.h5", overwrite=True, n_jobs=2)

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert h5["times_samples"].shape == (n_spikes,)
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
        torch.as_tensor(times_samples),
        torch.as_tensor(channels),
    )

    with tempfile.TemporaryDirectory() as tempdir:
        grab.fit_models(tempdir)
        grab.peel(Path(tempdir) / "grab.h5")

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert h5["times_samples"].shape == (n_spikes,)
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

    with tempfile.TemporaryDirectory() as tempdir:
        grab.fit_models(tempdir, n_jobs=2)
        grab.peel(Path(tempdir) / "grab.h5", n_jobs=2)

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert h5["times_samples"].shape == (n_spikes,)
            assert h5["channels"].shape == (n_spikes,)
            assert h5["waveforms"].shape == (
                n_spikes,
                121,
                channel_index.shape[1],
            )
            assert np.array_equal(h5["geom"][()], geom)
            assert np.array_equal(h5["channel_index"][()], channel_index)
            assert h5["last_chunk_start"][()] == 90_000


def test_grab_locations():
    # noise recording
    T_samples = 100_100
    n_channels = 50
    rg = np.random.default_rng(0)
    noise = rg.normal(size=(T_samples, n_channels)).astype(np.float32)
    geom = rg.uniform(low=0, high=100, size=(n_channels, 2))
    rec = sc.NumpyRecording(noise, 10_000)
    rec.set_dummy_probe_from_locations(geom)

    # random spike times_samples
    n_spikes = 50203
    times_samples = rg.integers(100, T_samples - 100, size=n_spikes)
    channels = rg.integers(0, n_channels, size=n_spikes)

    # try one with TPCA F/D, NN, localization
    channel_index = make_channel_index(geom, 20)
    pipeline = transform.WaveformPipeline(
        [
            transform.TemporalPCAFeaturizer(
                channel_index=torch.tensor(channel_index),
                geom=torch.tensor(geom),
                fit_radius=10,
            ),
            transform.Waveform(channel_index),
            transform.SingleChannelWaveformDenoiser(channel_index),
            transform.TemporalPCADenoiser(
                channel_index=torch.tensor(channel_index),
                geom=torch.tensor(geom),
                fit_radius=10,
            ),
            transform.Waveform(channel_index, name="tpca_waveforms"),
            transform.Localization(channel_index=channel_index, geom=geom, radius=50.0),
        ]
    )
    torch.manual_seed(0)
    grab = GrabAndFeaturize(
        rec,
        torch.as_tensor(channel_index),
        pipeline,
        torch.as_tensor(times_samples),
        torch.as_tensor(channels),
    )

    with tempfile.TemporaryDirectory() as tempdir:
        grab.fit_models(tempdir)
        grab.peel(Path(tempdir) / "grab.h5", device="cpu")

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert h5["times_samples"].shape == (n_spikes,)
            assert h5["channels"].shape == (n_spikes,)
            assert h5["point_source_localizations"].shape == (n_spikes, 4)
            locs0 = h5["point_source_localizations"][:]
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

    # grab the wfs
    channel_index = make_channel_index(geom, 20)
    pipeline = transform.WaveformPipeline(
        [
            transform.TemporalPCAFeaturizer(
                channel_index=torch.tensor(channel_index),
                geom=torch.tensor(geom),
                fit_radius=10,
            ),
            transform.Waveform(channel_index),
            transform.SingleChannelWaveformDenoiser(channel_index),
            transform.TemporalPCADenoiser(
                channel_index=torch.tensor(channel_index),
                geom=torch.tensor(geom),
                fit_radius=10,
            ),
            transform.Waveform(channel_index, name="tpca_waveforms"),
            transform.AmplitudeFeatures(channel_index),
        ]
    )
    torch.manual_seed(0)
    grab = GrabAndFeaturize(
        rec,
        torch.as_tensor(channel_index),
        pipeline,
        torch.as_tensor(times_samples),
        torch.as_tensor(channels),
    )
    with tempfile.TemporaryDirectory() as tempdir:
        grab.fit_models(tempdir)
        grab.peel(Path(tempdir) / "grab.h5", device="cpu")

        localize_hdf5(
            Path(tempdir) / "grab.h5",
            radius=50.0,
            amplitude_vectors_dataset_name="peak_amplitude_vectors",
        )

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert h5["times_samples"].shape == (n_spikes,)
            assert h5["channels"].shape == (n_spikes,)
            assert h5["point_source_localizations"].shape == (n_spikes, 4)
            locs1 = h5["point_source_localizations"][:]
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

    # this is kind of a good test of reproducibility
    valid = np.logical_and(
        locs0[:, 2] == locs0[:, 2].clip(geom[:, 1].min(), geom[:, 1].max()),
        locs1[:, 2] == locs1[:, 2].clip(geom[:, 1].min(), geom[:, 1].max()),
    )
    assert np.allclose(locs0[valid], locs1[valid], rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    # test_grab_and_featurize()
    test_grab_locations()
