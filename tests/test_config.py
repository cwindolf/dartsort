import dataclasses
import inspect
import re

import pytest

import dartsort
from dartsort.util import internal_config


def test_cfg_consistency():
    """Ensure config.py and internal_config.py don't diverge."""
    cfg0 = dartsort.to_internal_config(dartsort.DeveloperConfig())
    cfg1 = dartsort.DARTsortInternalConfig()

    # can just do assert cfg0 == cfg1, but pytest gives a better
    # error message if you do...
    for field in dataclasses.fields(dartsort.DARTsortInternalConfig):
        assert getattr(cfg0, field.name) == getattr(cfg1, field.name), field.name


def test_all_developer_flags_used():
    """Every DeveloperConfig field is read by to_internal_config or its builders."""
    module_src = inspect.getsource(internal_config)
    builders_src = module_src.split("# -- step config builders")[1]
    src = inspect.getsource(internal_config.to_internal_config) + builders_src
    read = set(re.findall(r"\bcfg\.(\w+)", src))
    # _motion_estimation_cfg copies over any field whose name matches
    read |= {f.name for f in dataclasses.fields(internal_config.MotionEstimationConfig)}

    unread = [
        f.name for f in dataclasses.fields(dartsort.DeveloperConfig) if f.name not in read
    ]
    assert not unread


@pytest.mark.parametrize("detection_type", ["subtract", "threshold", "match"])
def test_shared_flags_agree(detection_type):
    """Flags which feed several steps reach every one of them."""
    dev = dartsort.DeveloperConfig(
        detection_type=detection_type,
        precomputed_templates_npz="templates.npz",
        chunk_length_samples=12345,
        peak_sign="neg",
        temporal_pca_rank=6,
        subtraction_radius_um=123.0,
        whiten_estimator="sparsechol",
        whiten_temporal_length=7,
        template_interp_kind="clampna",
    )
    cfg = dartsort.to_internal_config(dev)
    detection_cfg = cfg.initial_detection_cfg
    motion_threshold_cfg = cfg.motion_estimation_cfg.threshold_cfg

    assert detection_cfg.chunk_length_samples == 12345
    assert cfg.matching_cfg.chunk_length_samples == 12345
    assert motion_threshold_cfg.chunk_length_samples == 12345

    assert motion_threshold_cfg.peak_sign == "neg"
    if detection_type != "match":
        assert detection_cfg.peak_sign == "neg"  # ty: ignore[unresolved-attribute]

    tpca_ranks = {
        "featurization": cfg.featurization_cfg.tpca_rank,
        "clustering features": cfg.clustering_features_cfg.feature_rank,
        "motion": cfg.motion_estimation_cfg.tpca_rank,
    }
    if detection_type == "subtract":
        tpca_ranks["subtraction"] = detection_cfg.subtraction_denoising_cfg.tpca_rank  # ty: ignore[unresolved-attribute]
    for name, rank in tpca_ranks.items():
        assert rank == 6, name

    (agg_cfg,) = [
        r for r in cfg.final_refinement_cfgs if r.refinement_strategy == "agglomerate"
    ]
    whitening_cfgs = {
        "template": cfg.template_cfg.whitening,
        "matching": cfg.matching_cfg.whitening,
        "agglomerate": agg_cfg.template_merge_cfg.whitening,  # ty: ignore[unresolved-attribute]
        "agglomerate template": agg_cfg.template_merge_cfg.template_cfg.whitening,  # ty: ignore[unresolved-attribute]
    }
    if detection_type == "subtract":
        whitening_cfgs["subtraction"] = detection_cfg.whiten_cfg  # ty: ignore[unresolved-attribute]
    for name, whitening_cfg in whitening_cfgs.items():
        assert whitening_cfg is not None
        assert whitening_cfg.estimator == "sparsechol", name
        assert whitening_cfg.radius == 123.0, name
        assert whitening_cfg.temporal_length == 7, name
        assert whitening_cfg.interp_params == internal_config.clampna_interp_params, name


def test_waveform_config():
    cfg0 = dartsort.WaveformConfig()
    cfg1 = dartsort.WaveformConfig.from_samples(42, 79)

    assert cfg0 == cfg1
    assert cfg0.trough_offset_samples() == 42
    assert cfg0.spike_length_samples() == 121
    assert cfg1.trough_offset_samples() == 42
    assert cfg1.spike_length_samples() == 121
    assert cfg0.trough_offset_samples(30_000.1) == 42
    assert cfg0.spike_length_samples(30_000.1) == 121
