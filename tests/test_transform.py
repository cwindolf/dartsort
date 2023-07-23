import neuropixel
import numpy as np
import torch
from dartsort.transform import WaveformPipeline, transformers_by_class_name
from dartsort.util.waveform_util import make_channel_index


def test_all_transformers():
    # make a bunch of fake waveforms, put all of the transformers into
    # one long pipeline, and try running its fit and forward
    h = neuropixel.dense_layout()
    geom = np.c_[h["x"], h["y"]]
    channel_index = make_channel_index(geom, 100)
    rg = np.random.default_rng(0)
    n_spikes = 1001
    spike_length_samples = 11
    waveforms = rg.normal(
        (n_spikes, spike_length_samples, channel_index.shape[1])
    )
    waveforms = waveforms.astype(np.float32)
    channels = rg.integers(0, len(geom), size=n_spikes)
    # set channels to nan as they would be in a real context
    for i in range(n_spikes):
        waveforms[i][:, channel_index[channels[i]] == len(geom)] = np.nan
    assert np.isnan(waveforms).any()
    assert not np.isnan(waveforms).all(axis=(1, 2)).any()

    pipeline = WaveformPipeline.from_class_names_and_kwargs(
        geom,
        channel_index,
        [(name, {}) for name in transformers_by_class_name],
    )

    # check shapes, dtype, etc match as they should
    # check no nans coming through
