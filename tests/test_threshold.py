import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from spikeinterface import NumpyRecording

import dartsort
from dartsort.peel.threshold import Threshold
from dartsort.util.internal_config import (
    FeaturizationConfig,
    FitSamplingConfig,
    ThresholdingConfig,
    WaveformConfig,
)


def test_sim(tmp_path, simulations):
    rec = simulations["driftn_szmini"]["recording"]
    gt_st = simulations["driftn_szmini"]["sorting"]

    st = dartsort.threshold(recording=rec, output_dir=tmp_path)
    assert abs(len(st) - len(gt_st)) / len(gt_st) < 0.2


def test_residual_snips_from_silent_chunks(tmp_path, mini_simulations):
    sim_rec = mini_simulations["driftn_szmini"]["recording"]
    traces = np.array(sim_rec.get_traces())
    traces[: traces.shape[0] // 2] = 0.0
    rec = NumpyRecording(traces, sampling_frequency=sim_rec.sampling_frequency)
    rec.set_dummy_probe_from_locations(sim_rec.get_channel_locations())

    thresholder = Threshold.from_config(
        recording=rec,
        waveform_cfg=WaveformConfig(),
        thresholding_cfg=ThresholdingConfig(),
        featurization_cfg=FeaturizationConfig(
            do_localization=False, do_tpca_denoise=False, save_amplitudes=False
        ),
        sampling_cfg=FitSamplingConfig(),
    )
    thresholder.load_or_fit_and_save_models(tmp_path / "models")
    n_chunks = len(thresholder.get_chunk_starts())
    total_residual_snips = 10 * n_chunks
    thresholder.peel(tmp_path / "peel.h5", total_residual_snips=total_residual_snips)

    with h5py.File(tmp_path / "peel.h5", "r") as h5:
        assert h5["n_residuals"][()] == total_residual_snips
        assert h5["times_seconds"][:].min() >= rec.get_duration() / 2 - 1.0


def proposal_thresholder(rec, proposal):
    return Threshold.from_config(
        recording=rec,
        waveform_cfg=WaveformConfig(),
        thresholding_cfg=ThresholdingConfig(
            detection_proposal=proposal,
            proposal_filters=1 if proposal == "tpca" else 4,
            proposal_threshold=5.0,
            time_jitter=3,
            spatial_jitter_radius=35.0,
        ),
        featurization_cfg=FeaturizationConfig(
            do_localization=False, do_tpca_denoise=False, save_amplitudes=False
        ),
        sampling_cfg=FitSamplingConfig(),
    )


@pytest.mark.parametrize("proposal", ["tpca", "vq"])
def test_proposal_thresholding(tmp_path, mini_simulations, proposal):
    rec = mini_simulations["driftn_szmini"]["recording"]
    thresholder = proposal_thresholder(rec, proposal)
    assert thresholder.needs_fit() and thresholder.peeling_needs_fit()

    chunk, _, left_margin, right_margin = thresholder.get_chunk(0)
    with pytest.raises(ValueError, match="needs to be fitted"):
        thresholder.peel(tmp_path / "unfit.h5")
    with pytest.raises(ValueError, match="not fit"):
        thresholder.peel_chunk(
            chunk, left_margin=left_margin, right_margin=right_margin
        )

    thresholder.load_or_fit_and_save_models(tmp_path / "models")
    assert not thresholder.needs_fit()
    filters = thresholder.b.proposal_filters
    assert filters is not None
    assert 1 <= filters.shape[0] <= thresholder.p.proposal_filters
    assert filters.shape[1] == 61
    assert torch.allclose(filters.norm(dim=1), torch.ones(len(filters)), atol=1e-5)

    with torch.no_grad():
        res = thresholder.peel_chunk(
            chunk, left_margin=left_margin, right_margin=right_margin
        )
        field = thresholder._build_proposer().full_proposal_score_field(
            F.pad(chunk, (0, 1), value=torch.nan)
        )
    assert res["n_spikes"] > 0
    orig_times = res["orig_times_samples"] + left_margin
    assert (field[orig_times, res["orig_channels"]] > 5.0**2).all()
    assert (res["times_samples"] - res["orig_times_samples"]).abs().max() <= 3

    reloaded = proposal_thresholder(rec, proposal)
    reloaded.load_models(tmp_path / "models")
    assert not reloaded.needs_fit()
    assert torch.equal(reloaded.b.proposal_filters, filters)
