"""Agglomeration of clusters to fix up GMM oversplits."""

from typing import cast

import numpy as np
import torch
from spikeinterface.core import BaseRecording

from ..templates.template_util import shared_basis_compress_templates
from ..templates.templates import TemplateData
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    RefinementConfig,
    TemplateMergeConfig,
    WaveformConfig,
)
from ..util.job_util import ensure_computation_config
from ..util.motion import MotionInfo
from ..util.py_util import databag
from ..util.spiketorch import (
    best_shared_pconv,
    scaled_normeuc_from_dots,
    shared_temporal_pconv,
)


@databag
class Agglomeration:
    agglomerated_sorting: DARTsortSorting
    merge_mapping: np.ndarray
    distances: np.ndarray
    shifts: np.ndarray
    bimodalities: np.ndarray | None


def agglomerate(
    *,
    sorting: DARTsortSorting,
    recording: BaseRecording | None,
    template_merge_cfg: TemplateMergeConfig | None,
    refinement_cfg: RefinementConfig | None,
    motion: MotionInfo,
    template_data: TemplateData | None = None,
    computation_cfg: ComputationConfig | None = None,
    waveform_cfg: WaveformConfig,
) -> Agglomeration:
    computation_cfg = ensure_computation_config(computation_cfg)

    if template_merge_cfg is None:
        assert refinement_cfg is not None
        template_merge_cfg = refinement_cfg.template_merge_cfg

    if template_data is None:
        assert sorting is not None
        assert recording is not None
        template_data = TemplateData.from_config(
            recording=recording,
            sorting=sorting,
            template_cfg=template_merge_cfg.to_template_config(),
            motion=motion,
            waveform_cfg=waveform_cfg,
        )
    assert template_data is not None

    tdist_res = template_distances(
        template_data=template_data,
        template_merge_cfg=template_merge_cfg,
        computation_cfg=computation_cfg,
    )

    # get mask
    # reconstruct scores from sorting attached data (exclude train_ix?)
    # get bimodality in mask
    # update mask, extract merge
    # apply with shifts
    return NotImplemented


def apply_agglomeration(
    sorting: DARTsortSorting, merge_mapping: np.ndarray, shifts: np.ndarray | None
) -> DARTsortSorting: ...


@databag
class TemplateDistanceResult:
    distances: np.ndarray
    shifts: np.ndarray
    r2: np.ndarray


def template_distances(
    *,
    template_data: TemplateData,
    template_merge_cfg: TemplateMergeConfig,
    computation_cfg: ComputationConfig | None = None,
) -> TemplateDistanceResult:
    if template_data.tsvd is not None:
        basis = template_data.tsvd.components_
    else:
        basis = None
    computation_cfg = ensure_computation_config(computation_cfg)
    device = computation_cfg.actual_device()

    sbt = shared_basis_compress_templates(
        template_data,
        rank=template_merge_cfg.svd_compression_rank,
        precomputed_basis=basis,
        computation_cfg=computation_cfg,
        with_r2=True,
    )
    tcomp = torch.asarray(sbt.temporal_components, device=device)
    spatial_sing = torch.asarray(sbt.spatial_singular, device=device)

    tconv = shared_temporal_pconv(
        temporal_comps=tcomp,
        up_temporal_comps=tcomp[:, None],
    )
    tconv = tconv[:, :, 0, :]

    # trim tconv to shift range
    max_shift = WaveformConfig.ms_to_samples(
        ms=template_merge_cfg.max_shift_ms,
        sampling_frequency=template_data.sampling_frequency,
    )
    conv_len = tconv.shape[2]
    center = conv_len // 2
    assert conv_len == 2 * center + 1
    assert center >= max_shift
    tconv = tconv[:, :, center - max_shift : center + 1 + max_shift]
    tconv = tconv.contiguous()

    best_conv, best_lag = best_shared_pconv(tconv, spatial_sing)

    # convert conv to distance
    if template_merge_cfg.distance_kind == "scaled_normeuc":
        dist = scaled_normeuc_from_dots(best_conv)
    else:
        raise ValueError(f"{template_merge_cfg.distance_kind=} not implemented.")

    # okay then
    return TemplateDistanceResult(
        distances=dist.numpy(force=True),
        shifts=best_lag.numpy(force=True),
        r2=cast(np.ndarray, sbt.r2),
    )
