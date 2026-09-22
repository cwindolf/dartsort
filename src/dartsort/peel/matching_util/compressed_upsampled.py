from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from torch import Tensor

from ...templates import (
    CompressedUpsampledTemplates,
    LowRankTemplates,
    TemplateData,
    compressed_upsampled_templates,
    svd_compress_templates,
    templates_at_time,
)
from ...util.internal_config import ComputationConfig, MatchingConfig
from ...util.job_util import ensure_computation_config
from ...util.logging_util import DARTSORTVERBOSE, get_logger
from ...util.motion import MotionInfo
from ...util.spiketorch import add_at_, convolve_lowrank, grab_spikes_full
from .matching_base import (
    ChunkTemplateData,
    MatchingPeaks,
    MatchingTemplates,
    PconvBase,
)
from .pairwise import CompressedPairwiseConv

logger = get_logger(__name__)
_extra_checks = logger.isEnabledFor(DARTSORTVERBOSE)


class CompressedUpsampledMatchingTemplates(MatchingTemplates):
    """YASS-style individually-compressed temporally upsampled templates

    Prioritizes good reconstruction over fast computation by using individual
    SVD basis for each template. Tries to be smart with temporal upsampling,
    using more or fewer upsamples per component, which ends up requiring a lot
    of bookkeeping logic.
    """

    template_type = "individual_compressed_upsampled"

    def __init__(
        self,
        *,
        low_rank_templates: LowRankTemplates,
        pconv_db: PconvBase,
        compressed_upsampled_temporal: CompressedUpsampledTemplates,
        trough_offset_samples: int,
        geom: np.ndarray | None = None,
        registered_geom: np.ndarray | None = None,
        registered_template_depths_um: np.ndarray | None = None,
        motion: MotionInfo,
        dtype=torch.float,
    ):
        super().__init__()
        global _extra_checks
        _extra_checks = logger.isEnabledFor(DARTSORTVERBOSE)

        lrt = low_rank_templates
        del low_rank_templates

        # in this case there is bookkeeping to manage correspondence
        self.n_templates = lrt.unit_ids.size
        self.spike_length_samples = lrt.temporal_components.shape[1]
        n_cupt = compressed_upsampled_temporal.n_compressed_upsampled_templates
        self.upsampling = n_cupt > self.n_templates
        self.comp_up_max = n_cupt
        self.registered_template_depths_um = registered_template_depths_um
        self.pconv_db = pconv_db

        # -- store relevant arrays from LRTs and obj LRTs
        self.svd_rank = lrt.singular_values.shape[1]
        uids = torch.asarray(lrt.unit_ids, dtype=torch.int32)
        tc = torch.asarray(lrt.temporal_components, dtype=dtype)
        sv = torch.asarray(lrt.singular_values, dtype=dtype)
        sc = torch.asarray(lrt.spatial_components, dtype=dtype)
        if _extra_checks:
            assert tc.isfinite().all()
            assert sv.isfinite().all()
            assert sc.isfinite().all()
        self.register_buffer("unit_ids", uids)
        self.register_buffer("temporal_comps", tc)
        self.register_buffer("spatial_sing", sv[:, :, None] * sc)
        self.register_buffer("padded_spatial_sing", F.pad(self.b.spatial_sing, (0, 1)))
        self.obj_lrts = lrt
        self.register_buffer("obj_unit_ids", self.b.unit_ids)
        self.register_buffer("obj_temporal_comps", self.b.temporal_comps)
        self.register_buffer("obj_spatial_sing", self.b.spatial_sing)
        self.obj_n_templates = len(self.b.obj_unit_ids)

        # -- geometry, as needed
        self.drifting = motion.drifting
        self.motion = motion
        self.n_channels_full = len(motion.rgeom)
        self.n_channels = len(motion.geom)
        self.match_rad = motion.min_dist / 1.5
        self.check_shapes()

        # -- upsampled temporal bases
        cupt = compressed_upsampled_temporal
        cup_map = torch.asarray(cupt.compressed_upsampling_map)
        cup_index = torch.asarray(cupt.compressed_upsampling_index)
        cup_ix_to_up_ix = torch.asarray(cupt.compressed_index_to_upsampling_index)
        cup_temporal = torch.asarray(cupt.compressed_upsampled_templates)
        self.register_buffer("cup_map", cup_map)
        self.register_buffer("cup_index", cup_index)
        self.register_buffer("cup_ix_to_up_ix", cup_ix_to_up_ix)
        self.register_buffer("cup_temporal", cup_temporal)

        # aux bufs
        self.register_buffer("rank_ix", torch.arange(self.svd_rank))
        sls = self.spike_length_samples
        self.register_buffer("time_ix", torch.arange(sls))
        self.register_buffer("chan_ix", torch.arange(self.n_channels))
        self.register_buffer("conv_lags", torch.arange(-sls + 1, sls))

    @property
    def device(self) -> torch.device:
        return self.b.rank_ix.device

    def check_shapes(self):
        assert self.b.temporal_comps.shape == (
            self.n_templates,
            self.spike_length_samples,
            self.svd_rank,
        )
        assert self.b.spatial_sing.shape == (
            self.n_templates,
            self.svd_rank,
            self.n_channels_full,
        )
        assert self.unit_ids.shape == (self.n_templates,)

    @classmethod
    def _from_config(
        cls,
        *,
        save_folder: Path | None,
        recording: BaseRecording,
        template_data: TemplateData,
        matching_cfg: MatchingConfig,
        computation_cfg: ComputationConfig | None,
        motion: MotionInfo,
        overwrite: bool,
        dtype: torch.dtype,
    ) -> Self:
        assert matching_cfg.up_method == "direct"
        assert save_folder is not None
        computation_cfg = ensure_computation_config(computation_cfg)

        lrt = svd_compress_templates(
            template_data,
            min_channel_amplitude=matching_cfg.template_min_channel_amplitude,
            rank=matching_cfg.template_svd_compression_rank,
            computation_cfg=computation_cfg,
        )
        cupt = compressed_upsampled_templates(
            lrt.temporal_components,
            ptps=np.ptp(template_data.templates, axis=1).max(1),
            max_upsample=matching_cfg.up_factor,
            n_upsamples_map=matching_cfg.upsampling_compression_map,
        )

        T_samples = recording.get_num_samples()
        dt = matching_cfg.chunk_length_samples
        chunk_starts = np.arange(0, T_samples, dt)
        chunk_ends = np.minimum(chunk_starts + dt, T_samples)
        chunk_centers_samples = (chunk_starts + chunk_ends) / 2
        chunk_centers_samples = chunk_centers_samples.clip(0, T_samples - 1)
        chunk_centers_s = recording.sample_index_to_time(chunk_centers_samples)
        geom = recording.get_channel_locations()
        pconv_td = template_data
        pconv_lrt = lrt
        assert pconv_td is not None
        assert pconv_lrt is not None
        pairwise_conv_db = CompressedPairwiseConv.from_template_data(
            hdf5_filename=save_folder / "pconv.h5",
            template_data=pconv_td,
            low_rank_templates=pconv_lrt,
            compressed_upsampled_temporal=cupt,
            chunk_time_centers_s=chunk_centers_s,
            motion=motion,
            geom=geom,
            computation_cfg=computation_cfg,
            overwrite=overwrite,
        )
        return cls(
            low_rank_templates=lrt,
            compressed_upsampled_temporal=cupt,
            trough_offset_samples=template_data.trough_offset_samples,
            geom=geom,
            registered_geom=template_data.registered_geom,
            registered_template_depths_um=template_data.registered_depths_um(),
            pconv_db=pairwise_conv_db,
            motion=motion,
            dtype=dtype,
        )

    def data_at_time(
        self,
        t_s: float,
        *,
        scaling: bool,
        free_scaling: bool,
        inv_lambda: float,
        scale_min: float,
        scale_max: float,
        resid_offset: int,
    ) -> "CompressedUpsampledChunkTemplateData":
        assert not resid_offset
        if self.drifting:
            shifts, padded_spatial_sing = templates_at_time(
                t_s=t_s,
                registered_templates=self.b.spatial_sing,
                registered_template_depths_um=self.registered_template_depths_um,
                motion=self.motion,
                return_pitch_shifts=True,
                return_padded=True,
                fill_value=0.0,
            )
            shifts = torch.asarray(shifts, device=self.device)
            padded_spatial_sing = torch.asarray(padded_spatial_sing, device=self.device)
        else:
            shifts = None
            padded_spatial_sing = self.b.padded_spatial_sing

        normsq_chan = padded_spatial_sing.square().sum(dim=1)
        main_channels = normsq_chan.argmax(dim=1)
        normsq = normsq_chan.sum(dim=1)
        obj_spatial_sing = padded_spatial_sing[..., :-1]
        obj_normsq_plus_inv_lambda = normsq[:, None] + inv_lambda
        inv_obj_normsq_plus_inv_lambda = obj_normsq_plus_inv_lambda.reciprocal()

        return CompressedUpsampledChunkTemplateData(
            resid_offset=resid_offset,
            upsampling=self.upsampling,
            scaling=scaling,
            free_scaling=free_scaling,
            needs_fine_pass=self.upsampling,
            comp_up_max=self.comp_up_max,
            n_templates=self.n_templates,
            obj_n_templates=self.obj_n_templates,
            spike_length_samples=self.spike_length_samples,
            filter_length_samples=self.spike_length_samples,
            up_factor=self.b.cup_index.shape[1],
            inv_lambda=torch.tensor(inv_lambda, device=normsq.device),
            scale_min=torch.tensor(scale_min, device=normsq.device),
            scale_max=torch.tensor(scale_max, device=normsq.device),
            obj_normsq=normsq,
            obj_normsq_plus_inv_lambda=obj_normsq_plus_inv_lambda,
            inv_obj_normsq_plus_inv_lambda=inv_obj_normsq_plus_inv_lambda,
            obj_temporal_comps=self.b.obj_temporal_comps,
            obj_spatial_sing=obj_spatial_sing,
            temporal_comps=self.b.temporal_comps,
            spatial_sing=padded_spatial_sing[..., :-1],
            padded_spatial_sing=padded_spatial_sing,
            cup_temporal=self.b.cup_temporal,
            normsq=normsq,
            cup_index=self.b.cup_index,
            cup_map=self.b.cup_map,
            cup_ix_to_up_ix=self.b.cup_ix_to_up_ix,
            unit_ids=self.b.unit_ids,
            main_channels=main_channels,
            conv_lags=self.b.conv_lags,
            rank_ix=self.b.rank_ix,
            time_ix=self.b.time_ix,
            chan_ix=self.b.chan_ix,
            pconv_db=self.pconv_db,
            shifts_a=shifts,
            shifts_b=shifts,
        )


@dataclass(kw_only=True, slots=True, frozen=True, repr=False, eq=False)
class CompressedUpsampledChunkTemplateData(ChunkTemplateData):
    upsampling: bool
    scaling: bool
    free_scaling: bool
    needs_fine_pass: bool
    comp_up_max: int
    n_templates: int
    obj_n_templates: int
    spike_length_samples: int
    up_factor: int
    inv_lambda: Tensor
    scale_min: Tensor
    scale_max: Tensor
    resid_offset: int
    filter_length_samples: int

    # objective props
    obj_normsq: Tensor
    obj_normsq_plus_inv_lambda: Tensor
    inv_obj_normsq_plus_inv_lambda: Tensor
    obj_temporal_comps: Tensor
    obj_spatial_sing: Tensor
    temporal_comps: Tensor
    spatial_sing: Tensor
    padded_spatial_sing: Tensor
    cup_temporal: Tensor
    normsq: Tensor

    # indexing
    cup_index: Tensor
    cup_map: Tensor
    cup_ix_to_up_ix: Tensor
    unit_ids: Tensor
    main_channels: Tensor
    conv_lags: Tensor
    rank_ix: Tensor
    time_ix: Tensor
    chan_ix: Tensor

    # pconv
    pconv_db: PconvBase
    shifts_a: Tensor | None
    shifts_b: Tensor | None
    prewhiten: bool = False
    needs_residual: bool = True

    def convolve(self, traces, padding=0, out=None):
        """Convolve the objective templates with traces."""
        return convolve_lowrank(
            traces,
            self.obj_spatial_sing,
            self.obj_temporal_comps,
            padding=padding,
            out=out,
        )

    def subtract_conv(self, conv, peaks, padding=0, batch_size=256, sign=-1):
        if not peaks.n_spikes:
            return
        assert peaks.times is not None
        assert peaks.template_inds is not None
        for batch_start in range(0, peaks.n_spikes, batch_size):
            batch_end = min(batch_start + batch_size, peaks.n_spikes)
            temp_inds_b = peaks.template_inds[batch_start:batch_end]
            if peaks.up_inds is None:
                up_inds = torch.zeros_like(temp_inds_b)
            else:
                up_inds = peaks.up_inds[batch_start:batch_end]
            if peaks.scalings is None:
                scalings = None
            else:
                scalings = peaks.scalings[batch_start:batch_end]
            template_indices_a, pconvs, which_b = self.pconv_db.query(
                template_indices_a=None,
                template_indices_b=temp_inds_b,
                upsampling_indices_b=up_inds,
                shifts_a=self.shifts_a,
                shifts_b=self.shifts_b[temp_inds_b]
                if self.shifts_b is not None
                else None,
                scalings_b=scalings,
            )
            pconvs = pconvs.to(conv.device)
            times_sub = peaks.times[batch_start:batch_end][which_b]
            ix_template = template_indices_a[:, None]
            ix_time = times_sub[:, None] + (padding + self.conv_lags)[None, :]
            add_at_(conv, (ix_template, ix_time), pconvs, sign=sign)

    def subtract(self, traces, peaks, sign=-1):
        """Subtract templates from traces."""
        if not peaks.n_spikes:
            return
        assert peaks.times is not None
        assert peaks.template_inds is not None
        if peaks.up_inds is None:
            assert self.cup_map.shape[1] == 1
            compressed_up_inds = self.cup_map[peaks.template_inds][:, 0]
        else:
            compressed_up_inds = self.cup_map[peaks.template_inds, peaks.up_inds]
        if peaks.scalings is not None:
            batch_templates = torch.einsum(
                "n,nrc,ntr->ntc",
                peaks.scalings,
                self.spatial_sing[peaks.template_inds],
                self.cup_temporal[compressed_up_inds],
            )
        else:
            batch_templates = torch.bmm(
                self.cup_temporal[compressed_up_inds],
                self.spatial_sing[peaks.template_inds],
            )
        time_ix = peaks.times[:, None, None] + self.time_ix[None, :, None]
        add_at_(
            traces, (time_ix, self.chan_ix[None, None, :]), batch_templates, sign=sign
        )

    def fine_match(
        self,
        *,
        peaks: MatchingPeaks,
        residual: Tensor | None,
        conv: Tensor,
        padding: int = 0,
    ):
        """Determine temporal upsampling and scaling

        Returns
        -------
        time_shifts : Optional[array]
        up_inds : Optional[array]
        scalings : Optional[array]
        template_inds : array
        objs : array
        """
        assert residual is not None
        if not self.needs_fine_pass:
            return peaks
        if not peaks.n_spikes:
            return peaks
        del conv, padding  # unused

        if self.upsampling:
            residual_snips = grab_spikes_full(
                residual,
                peaks.times,
                trough_offset=0,
                spike_length_samples=self.spike_length_samples + 1,
            )
        else:
            residual_snips = None

        template_inds = peaks.template_inds
        norms = self.normsq[template_inds]
        objs = peaks.scores

        if not self.upsampling:
            return MatchingPeaks(
                times=peaks.times,
                template_inds=template_inds,
                scalings=peaks.scalings,
                scores=objs,
            )
        assert residual_snips is not None
        assert template_inds is not None

        # get the objective for snips now and one step back
        # TODO: jagged tensor? no-compression mode?
        comp_up_ix = self.cup_index[template_inds]
        dup_ix, column_ix = (comp_up_ix < self.comp_up_max).nonzero(as_tuple=True)
        comp_up_indices = comp_up_ix[dup_ix, column_ix]
        temps_t = self.cup_temporal[comp_up_indices]
        temps_s = self.spatial_sing[template_inds[dup_ix]]
        snips_dup_dt = residual_snips[dup_ix].unfold(1, self.spike_length_samples, 1)
        convs = torch.einsum("ndct,ntr,nrc->nd", snips_dup_dt, temps_t, temps_s)
        norms = norms[dup_ix]
        if self.scaling:
            b = convs + self.inv_lambda
            a = norms[:, None] + self.inv_lambda
            scalings = b.div(a).clip_(self.scale_min, self.scale_max)
            # 2sb - s^2a - 1/l
            scalingsqa = scalings.square().mul_(-a)
            objs = scalingsqa.addcmul_(scalings, b, value=2.0).sub_(self.inv_lambda)
            del convs, scalingsqa
        else:
            objs = torch.add(-norms[:, None], convs, alpha=2.0)
            scalings = None
            del convs
        # this is just for numerical duplicates encountered in testing.
        objs += (column_ix == 0).float()[:, None] * 1e-5
        objs_, better_dt = objs.max(dim=1)
        objs = objs.new_full(comp_up_ix.shape, -torch.inf)
        objs[dup_ix, column_ix] = objs_
        objs, best_column_ix = objs.max(dim=1)

        comp_up_indices = comp_up_ix.take_along_dim(
            dim=1, indices=best_column_ix[:, None]
        )
        comp_up_indices = comp_up_indices[:, 0]
        up_inds = self.cup_ix_to_up_ix[comp_up_indices]

        # prev convs were one step earlier
        time_shifts = comp_up_ix.new_full(comp_up_ix.shape, 0)
        time_shifts[dup_ix, column_ix] += better_dt.long()
        time_shifts = time_shifts.take_along_dim(dim=1, indices=best_column_ix[:, None])
        time_shifts = time_shifts[:, 0]
        if self.scaling:
            assert scalings is not None
            scalings_ = scalings.take_along_dim(indices=better_dt[:, None], dim=1)[:, 0]
            scalings = scalings_.new_zeros(comp_up_ix.shape)
            scalings[dup_ix, column_ix] = scalings_
            scalings = scalings.take_along_dim(dim=1, indices=best_column_ix[:, None])
            scalings = scalings[:, 0]

        assert peaks.times is not None
        times = peaks.times + time_shifts
        up_half = self.up_factor // 2
        time_shifts = (up_inds > up_half).long().neg_()
        return MatchingPeaks(
            times=times,
            template_inds=template_inds,
            up_inds=up_inds,
            scalings=scalings,
            scores=objs,
            time_shifts=time_shifts,
        )

    def get_clean_waveforms(
        self,
        peaks: MatchingPeaks,
        channels: Tensor,
        channel_index: Tensor,
        add_into: Tensor | None = None,
    ):
        if not peaks.n_spikes:
            return add_into
        assert peaks.template_inds is not None
        spatial = self.padded_spatial_sing[
            peaks.template_inds[:, None, None],
            self.rank_ix[None, :, None],
            channel_index[channels][:, None, :],
        ]
        if peaks.scalings is not None:
            spatial.mul_(peaks.scalings[:, None, None])
        if peaks.up_inds is None:
            assert self.cup_map.shape[1] == 1
            comp_up_ix = self.cup_map[peaks.template_inds][:, 0]
        else:
            comp_up_ix = self.cup_map[peaks.template_inds, peaks.up_inds]
        temporal = self.cup_temporal[comp_up_ix]
        if add_into is None:
            return temporal.bmm(spatial)
        else:
            return add_into.baddbmm_(temporal, spatial)

    def reconstruct_up_templates(self):
        up_comps = self.cup_temporal[self.cup_map].cpu()
        return torch.einsum("nrc,nutr->nutc", self.spatial_sing.cpu(), up_comps)
