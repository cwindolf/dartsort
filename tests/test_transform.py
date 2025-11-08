import tempfile
from pathlib import Path
import pickle

import numpy as np
import torch
import spikeinterface.core as sc

from dartsort.transform import WaveformPipeline, transformers_by_class_name
from dartsort.util.waveform_util import make_channel_index
from test_util import dense_layout


def _check_state_equal(d1, d2):
    for k1, v1 in d1.items():
        assert k1 in d2
        if isinstance(v1, dict):
            _check_state_equal(v1, d2[k1])
        elif torch.is_tensor(v1):
            assert torch.equal(v1, d2[k1])
        else:
            assert v1 == d2[k1]


def _check_saveload(geom, channel_index, pipeline):
    # check saving and loading before fit
    with tempfile.TemporaryDirectory() as tdir:
        pt = Path(tdir) / "pt.pt"

        # try them individually first with save/load
        for f in pipeline.transformers:
            nf = f.needs_fit()
            try:
                torch.save(f, pt)
                ff = torch.load(pt)
                assert ff.needs_fit() == nf
            except pickle.UnpicklingError as e:
                raise ValueError(f"Save/load failed for {f=}") from e
            pt.unlink()

        # individually with state dict
        for f in pipeline.transformers:
            nf = f.needs_fit()
            try:
                torch.save(f.state_dict(), pt)
                f.load_state_dict(torch.load(pt))
                assert f.needs_fit() == nf
            except pickle.UnpicklingError as e:
                raise ValueError(f"Save/load failed for {f=}") from e
            pt.unlink()

        # now the full thing, save/load...
        nf = pipeline.needs_fit()
        orig_state_dict = pipeline.state_dict()
        torch.save(pipeline, pt)
        pipeline2 = torch.load(pt)
        _check_state_equal(orig_state_dict, pipeline2.state_dict())
        assert pipeline2.needs_fit() == nf

        # now with state dict
        torch.save(pipeline.state_dict(), pt)
        pipeline.load_state_dict(torch.load(pt))
        assert pipeline.needs_fit() == nf
        _check_state_equal(orig_state_dict, pipeline.state_dict())

        # now tith from_state_dict_pt
        pipeline2 = WaveformPipeline.from_state_dict_pt(geom, channel_index, pt)
        assert pipeline2.needs_fit() == nf
        _check_state_equal(orig_state_dict, pipeline2.state_dict())


def test_all_transformers():
    # make a bunch of fake waveforms, put all of the transformers into
    # one long pipeline, and try running its fit and forward
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]]
    channel_index = make_channel_index(geom, 100)
    rg = np.random.default_rng(0)
    n_spikes = 1001
    spike_length_samples = 121
    waveforms = rg.normal(size=(n_spikes, spike_length_samples, channel_index.shape[1]))
    waveforms = waveforms.astype(np.float32)
    channels = rg.integers(0, len(geom), size=n_spikes)
    # set channels to nan as they would be in a real context
    for i in range(n_spikes):
        rel_chans = channel_index[channels[i]] + 0
        rel_chans[rel_chans < len(geom)] -= channel_index[channels[i]][0]  # type: ignore
        waveforms[i, :, rel_chans == len(geom)] = np.nan
    assert np.isnan(waveforms).any()
    assert not np.isnan(waveforms).all(axis=(1, 2)).any()

    # helper noise recording
    noise_rec = sc.NumpyRecording(
        rg.normal(size=(30_000, len(geom))).astype("float32"),
        sampling_frequency=30_000,
    )
    noise_rec.set_dummy_probe_from_locations(geom)

    smoke_test_kwargs = {
        "Decollider": dict(n_epochs=1, epoch_size=n_spikes),
        "AmortizedLocalization": dict(n_epochs=1, epoch_size=n_spikes),
    }
    skip_me = {"SupervisedDenoiser"}

    pipeline = WaveformPipeline.from_class_names_and_kwargs(
        geom,
        channel_index,
        [
            (name, {"name_prefix": j, **smoke_test_kwargs.get(name, {})})
            for j, name in enumerate(transformers_by_class_name)
            if name not in skip_me
        ],
    )

    waveforms = torch.from_numpy(waveforms)
    channels = torch.from_numpy(channels)
    print(f"{waveforms.requires_grad=}")
    print("-- Precompute")
    pipeline.precompute()
    print("-- Pre-fit check")
    _check_saveload(geom, channel_index, pipeline)
    print("-- Fit")
    pipeline.fit(recording=noise_rec, waveforms=waveforms, channels=channels)
    assert not pipeline.needs_fit()
    print("-- Forward")
    twaveforms, features = pipeline(waveforms, channels=channels)
    assert twaveforms.dtype == waveforms.dtype
    assert twaveforms.shape == waveforms.shape
    assert torch.equal(torch.isnan(twaveforms), torch.isnan(waveforms))
    assert torch.equal(
        torch.isnan(twaveforms[:, 0]),
        torch.from_numpy(channel_index[channels] == len(geom)),
    )
    for k, v in features.items():
        assert len(v) == n_spikes

        if v.ndim == 3 and v.shape[2] == waveforms.shape[2]:
            # TPCAs can have nans
            assert "pca" in k.lower() or "waveform" in k.lower()
            assert torch.equal(v[:, 0].isnan(), waveforms[:, 0].isnan())
        elif "amplitude_vector" in k.lower():
            assert v.ndim == 2 and v.shape[1] == waveforms.shape[2]
            assert torch.equal(v.isnan(), waveforms[:, 0].isnan())
        else:
            assert v.isfinite().all()
    print("-- Final check")
    _check_saveload(geom, channel_index, pipeline)


if __name__ == "__main__":
    test_all_transformers()
