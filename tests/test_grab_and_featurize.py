"""A good integration test of a few pieces"""

from typing import cast

import tempfile
from pathlib import Path

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dartsort import transform
from dartsort.peel.grab import GrabAndFeaturize
from dartsort.util.waveform_util import make_channel_index
from dartsort.util.internal_config import ComputationConfig, WaveformConfig, FitSamplingConfig


two_jobs_cfg = ComputationConfig(n_jobs_cpu=2, n_jobs_gpu=2)


def test_grab_and_featurize():
    # noise recording
    T_samples = 50_100
    n_channels = 50
    rg = np.random.default_rng(0)
    noise = rg.normal(size=(T_samples, n_channels)).astype(np.float32)
    geom = rg.uniform(low=0, high=100, size=(n_channels, 2))
    fs = 10_000.0
    rec = sc.NumpyRecording(noise, fs)
    rec.set_dummy_probe_from_locations(geom)
    wf_cfg = WaveformConfig()
    sls = wf_cfg.spike_length_samples(fs)

    # random spike times_samples
    n_spikes = 5203
    times_samples = rg.integers(100, T_samples - 100, size=n_spikes)
    channels = rg.integers(0, n_channels, size=n_spikes)

    # grab the wfs
    channel_index = make_channel_index(geom, 20)
    pipeline = transform.WaveformPipeline(
        [transform.Waveform(channel_index, waveform_cfg=wf_cfg, sampling_frequency=fs)]
    )
    grab = GrabAndFeaturize(
        rec,
        torch.as_tensor(channel_index),
        pipeline,
        waveform_cfg=wf_cfg,
        times_samples=torch.as_tensor(times_samples),
        fixed_properties=dict(channels=torch.as_tensor(channels)),
    )

    with tempfile.TemporaryDirectory() as tempdir:
        grab.peel(Path(tempdir) / "grab.h5")

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert cast(h5py.Dataset, h5["times_samples"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["channels"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["waveforms"]).shape == (
                n_spikes,
                sls,
                channel_index.shape[1],
            )
            assert np.array_equal(cast(h5py.Dataset, h5["geom"])[()], geom)
            assert np.array_equal(
                cast(h5py.Dataset, h5["channel_index"])[()], channel_index
            )
            assert cast(h5py.Dataset, h5["last_chunk_start"])[()] == 30_000

        grab.peel(Path(tempdir) / "grab.h5", overwrite=True)

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert cast(h5py.Dataset, h5["times_samples"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["channels"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["waveforms"]).shape == (
                n_spikes,
                sls,
                channel_index.shape[1],
            )
            assert np.array_equal(cast(h5py.Dataset, h5["geom"])[()], geom)
            assert np.array_equal(
                cast(h5py.Dataset, h5["channel_index"])[()], channel_index
            )
            assert cast(h5py.Dataset, h5["last_chunk_start"])[()] == 30_000

    # try one with TPCA
    channel_index = make_channel_index(geom, 20)
    pipeline = transform.WaveformPipeline(
        [
            transform.Waveform(
                channel_index, waveform_cfg=wf_cfg, sampling_frequency=fs
            ),
            transform.TemporalPCADenoiser(
                channel_index=torch.tensor(channel_index),
                geom=torch.tensor(geom),
                fit_radius=10,
                waveform_cfg=wf_cfg,
                sampling_frequency=fs,
            ),
            transform.Waveform(
                channel_index,
                name="tpca_waveforms",
                waveform_cfg=wf_cfg,
                sampling_frequency=fs,
            ),
        ]
    )
    grab = GrabAndFeaturize(
        rec,
        torch.as_tensor(channel_index),
        pipeline,
        times_samples=torch.as_tensor(times_samples),
        fixed_properties=dict(channels=torch.as_tensor(channels)),
        waveform_cfg=wf_cfg,
    )

    with tempfile.TemporaryDirectory() as tempdir:
        grab.fit_models(tempdir)
        grab.peel(Path(tempdir) / "grab.h5")

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert cast(h5py.Dataset, h5["times_samples"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["channels"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["waveforms"]).shape == (
                n_spikes,
                sls,
                channel_index.shape[1],
            )
            assert cast(h5py.Dataset, h5["tpca_waveforms"]).shape == (
                n_spikes,
                sls,
                channel_index.shape[1],
            )
            assert np.array_equal(cast(h5py.Dataset, h5["geom"])[()], geom)
            assert np.array_equal(
                cast(h5py.Dataset, h5["channel_index"])[()], channel_index
            )
            assert cast(h5py.Dataset, h5["last_chunk_start"])[()] == 30_000

    with tempfile.TemporaryDirectory() as tempdir:
        grab.fit_models(tempdir, computation_cfg=two_jobs_cfg)
        grab.peel(Path(tempdir) / "grab.h5", computation_cfg=two_jobs_cfg)

        with h5py.File(Path(tempdir) / "grab.h5", locking=False) as h5:
            assert cast(h5py.Dataset, h5["times_samples"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["channels"]).shape == (n_spikes,)
            assert cast(h5py.Dataset, h5["waveforms"]).shape == (
                n_spikes,
                sls,
                channel_index.shape[1],
            )
            assert np.array_equal(cast(h5py.Dataset, h5["geom"])[()], geom)
            assert np.array_equal(
                cast(h5py.Dataset, h5["channel_index"])[()], channel_index
            )
            assert cast(h5py.Dataset, h5["last_chunk_start"])[()] == 30_000
