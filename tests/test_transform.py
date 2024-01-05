import numpy as np
from dartsort.transform import WaveformPipeline, transformers_by_class_name
from dartsort.util.waveform_util import make_channel_index
from test_util import dense_layout


def test_all_transformers():
    # make a bunch of fake waveforms, put all of the transformers into
    # one long pipeline, and try running its fit and forward
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]]
    channel_index = make_channel_index(geom, 100)
    rg = np.random.default_rng(0)
    n_spikes = 1001
    spike_length_samples = 11
    waveforms = rg.normal(
        size=(n_spikes, spike_length_samples, channel_index.shape[1])
    )
    waveforms = waveforms.astype(np.float32)
    channels = rg.integers(0, len(geom), size=n_spikes)
    # set channels to nan as they would be in a real context
    for i in range(n_spikes):
        rel_chans = channel_index[channels[i]] + 0
        rel_chans[rel_chans < len(geom)] -= channel_index[channels[i]][0]
        waveforms[i, :, rel_chans == len(geom)] = np.nan
    assert np.isnan(waveforms).any()
    assert not np.isnan(waveforms).all(axis=(1, 2)).any()

    pipeline = WaveformPipeline.from_class_names_and_kwargs(
        geom,
        channel_index,
        [(name, {}) for name in transformers_by_class_name],
    )

    # TODO
    # check shapes, dtype, etc match as they should
    # check no nans coming through


if __name__ == "__main__":
    test_all_transformers()
