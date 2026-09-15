import warnings
from typing import Literal

import numpy as np
import spikeinterface.full as si
from spikeinterface.core import BaseRecording

from .internal_config import PreprocessingStrategy
from .logging_util import get_logger
from .py_util import panic

logger = get_logger(__name__)

preprocessing_strategies = {}


def none(rec: BaseRecording, dtype: str) -> BaseRecording:
    del dtype
    return rec


preprocessing_strategies["none"] = none


def ibllike(rec: BaseRecording, dtype: str) -> BaseRecording:
    rec = rec.astype(np.float32)
    rec = si.highpass_filter(rec)
    if "inter_sample_shift" in rec.get_property_keys():
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
    if "inter_sample_shift" in rec.get_property_keys():
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


class DSPreprocessingWarning(UserWarning):
    pass


class DSPreprocessingError(ValueError):
    pass


def warn_about_preprocessing(rec: BaseRecording, will_preprocess: bool):
    from .data_util import check_recording

    if (not will_preprocess) and rec.dtype.kind != "f":
        raise DSPreprocessingError(
            f"The input recording had data type {rec.dtype.name}, but "
            "dartsort's preprocessing flag was set to 'none'. Please "
            "set preprocessing to another strategy."
        )

    not_in_range, _, max_abs, std = check_recording(
        rec=rec, log=False, count_spikes=False
    )
    in_range = not not_in_range

    if will_preprocess and in_range:
        warnings.warn(
            f"preprocessing was configured to be skipped, but recording values "
            f"reach |{max_abs:0.2f}| with std dev {std:0.2f}, so the recording "
            "looks like it may already have some preprocessing applied. Just a "
            "heads up to give a chance to double check.",
            DSPreprocessingWarning,
            stacklevel=2,
        )
    if (not will_preprocess) and not_in_range:
        warnings.warn(
            f"preprocessing was configured to be skipped, but recording values "
            f"reach |{max_abs:0.2f}| with std dev {std:0.2f}, so the recording "
            "looks like it was not preprocessed. Did you want to set the "
            "preprocessing flag?",
            DSPreprocessingWarning,
            stacklevel=2,
        )


def preprocess(
    rec: BaseRecording,
    strategy: PreprocessingStrategy = "none",
    already_preprocessed: Literal[
        "yes", "no", "assume_yes_if_float"
    ] = "assume_yes_if_float",
    dtype: str = "float32",
) -> BaseRecording:
    if already_preprocessed == "yes":
        logger.info("skipping preprocessing since already_preprocessed=yes.")
        warn_about_preprocessing(rec=rec, will_preprocess=False)
        return rec
    elif already_preprocessed == "assume_yes_if_float":
        if rec.dtype.kind == "f":
            logger.info(
                "skipping preprocessing since already_preprocessed=assume_yes_if_float and "
                "the data is already floating point."
            )
            warn_about_preprocessing(rec=rec, will_preprocess=False)
            return rec
    elif already_preprocessed == "no":
        pass
    else:
        panic(already_preprocessed)

    logger.info("applying preprocessing: %s", strategy)
    warn_about_preprocessing(rec=rec, will_preprocess=strategy != "none")
    return preprocessing_strategies[strategy](rec, dtype)
