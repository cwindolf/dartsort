import numpy as np
import spikeinterface.full as si
from spikeinterface.core import BaseRecording

from .internal_config import PreprocessingStrategy

preprocessing_strategies = {}


def none(rec: BaseRecording, dtype: str) -> BaseRecording:
    del dtype
    return rec


preprocessing_strategies["none"] = none


def ibllike(rec: BaseRecording, dtype: str) -> BaseRecording:
    rec = rec.astype(np.float32)
    rec = si.highpass_filter(rec)
    if "inter_sample_shift" in rec._properties:
        rec = si.phase_shift(rec)
    if rec.has_scaleable_traces():
        bcids = si.detect_bad_channels(rec, seed=0)
        rec = rec.remove_channels(bcids[0])
    rec = si.common_reference(rec)

    nl = si.get_noise_levels(
        rec,
        return_in_uV=False,
        random_slices_kwargs=dict(seed=0, num_chunks_per_segment=100),
    )
    rec = si.scale(rec, gain=1.0 / nl)
    rec = si.highpass_spatial_filter(rec)

    rec = rec.astype(dtype)

    return rec


preprocessing_strategies["ibllike"] = ibllike


def ibllikecmr(rec: BaseRecording, dtype: str) -> BaseRecording:
    rec = rec.astype(np.float32)
    rec = si.highpass_filter(rec)
    if "inter_sample_shift" in rec._properties:
        rec = si.phase_shift(rec)
    if rec.has_scaleable_traces():
        bcids = si.detect_bad_channels(rec, seed=0)
        rec = rec.remove_channels(bcids[0])
    rec = si.common_reference(rec)

    nl = si.get_noise_levels(
        rec,
        return_in_uV=False,
        random_slices_kwargs=dict(seed=0, num_chunks_per_segment=100),
    )
    rec = si.scale(rec, gain=1.0 / nl)
    rec = si.common_reference(rec)

    rec = rec.astype(dtype)

    return rec


preprocessing_strategies["ibllikecmr"] = ibllikecmr


def standardize(rec: BaseRecording, dtype: str) -> BaseRecording:
    rec = rec.astype(np.float32)
    nl = si.get_noise_levels(
        rec,
        return_in_uV=False,
        random_slices_kwargs=dict(seed=0, num_chunks_per_segment=100),
    )
    rec = si.scale(rec, gain=1.0 / nl)
    rec = rec.astype(dtype)
    return rec


preprocessing_strategies["standardize"] = standardize


def preprocess(
    rec: BaseRecording,
    strategy: PreprocessingStrategy = "none",
    dtype: str = "float32",
) -> BaseRecording:
    return preprocessing_strategies[strategy](rec, dtype)
