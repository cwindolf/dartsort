from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
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
from ...util.logging_util import get_logger, DARTSORTVERBOSE
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

    Also supports this concept of groups, where some templates are conceptually
    identified as mutually co-exclusive / co-refractory. This is applied when
    the input template_data has duplicates in its unit_ids in from_config, and
    there are cases in the code to try to avoid its cost when not used.
    """

    template_type = "individual_compressed_upsampled"

    def __init__(
        self,
        low_rank_templates: LowRankTemplates,
        pconv_db: PconvBase,
        compressed_upsampled_temporal: CompressedUpsampledTemplates,
        trough_offset_samples: int,
        obj_low_rank_templates: LowRankTemplates | None = None,
        geom: np.ndarray | None = None,
        registered_geom: np.ndarray | None = None,
        registered_template_depths_um: np.ndarray | None = None,
        refractory_radius_frames: int = 10,
        motion_est=None,
        dtype=torch.float,
    ):
        super().__init__()
        global _extra_checks
        _extra_checks = logger.isEnabledFor(DARTSORTVERBOSE)

        lrt = low_rank_templates
        del low_rank_templates

        # in this case there is bookkeeping to manage correspondence
        # between coarse and fine templates
        self.coarse_objective = obj_low_rank_templates is not None
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
        if obj_low_rank_templates is None:
            self.obj_lrts = lrt
            self.register_buffer("obj_unit_ids", self.b.unit_ids)
            self.register_buffer("obj_temporal_comps", self.b.temporal_comps)
            self.register_buffer("obj_spatial_sing", self.b.spatial_sing)
        else:
            olrt = obj_low_rank_templates
            self.obj_lrts = obj_low_rank_templates
            uids = torch.asarray(olrt.temporal_components, dtype=torch.int32)
            tc = torch.asarray(olrt.temporal_components, dtype=dtype)
            sv = torch.asarray(olrt.singular_values, dtype=dtype)
            sc = torch.asarray(olrt.spatial_components, dtype=dtype)
            self.register_buffer("obj_spike_counts", uids)
            self.register_buffer("obj_unit_ids", uids)
            self.register_buffer("obj_temporal_comps", tc)
            self.register_buffer("obj_spatial_sing", sv[:, :, None] * sc)
        self.obj_n_templates = len(self.b.obj_unit_ids)

        # -- geometry, as needed
        self.drifting = motion_est is not None and np.any(motion_est.displacement)
        if self.drifting:
            assert geom is not None
            assert registered_geom is not None
            self.geom = geom
            self.geom_kdtree = KDTree(geom)
            self.rgeom = registered_geom
            self.motion_est = motion_est
            self.match_rad = pdist(geom).min() / 1.5
            self.n_channels_full = len(self.rgeom)
            self.n_channels = len(geom)
        else:
            self.geom_kdtree = self.geom = self.motion_est = self.rgeom = None
            self.n_channels = self.b.spatial_sing.shape[2]
            self.n_channels_full = self.n_channels
            self.match_rad = None
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

        # -- template grouping and coarse objective indexing
        gres = handle_template_groups(
            self.obj_unit_ids, self.unit_ids, self.coarse_objective
        )
        self.have_groups, group_index, fine_to_coarse, coarse_index = gres
        self.register_buffer_or_none("group_index", group_index)
        self.register_buffer_or_none("fine_to_coarse", fine_to_coarse)
        self.register_buffer_or_none("coarse_index", coarse_index)

        # aux bufs
        rr = refractory_radius_frames
        self.register_buffer("refrac_ix", torch.arange(-rr, rr + 1))
        self.register_buffer("rank_ix", torch.arange(self.svd_rank))
        sls = self.spike_length_samples
        self.register_buffer("time_ix", torch.arange(sls))
        self.register_buffer("chan_ix", torch.arange(self.n_channels))
        self.register_buffer("conv_lags", torch.arange(-sls + 1, sls))

    @property
    def device(self) -> torch.device:
        return self.b.refrac_ix.device

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
        save_folder: Path,
        recording: BaseRecording,
        template_data: TemplateData,
        matching_cfg: MatchingConfig,
        computation_cfg: ComputationConfig | None = None,
        motion_est=None,
        overwrite: bool = False,
        dtype: torch.dtype = torch.float,
    ) -> Self:
        assert matching_cfg.up_method == "direct"
        computation_cfg = ensure_computation_config(computation_cfg)

        unit_ids, id_counts = np.unique(template_data.unit_ids, return_counts=True)
        have_groups = np.any(id_counts > 1)
        coarse_objective = matching_cfg.coarse_objective and have_groups

        lrt = svd_compress_templates(
            template_data,
            min_channel_amplitude=matching_cfg.template_min_channel_amplitude,
            rank=matching_cfg.template_svd_compression_rank,
            computation_cfg=computation_cfg,
        )
        cupt = compressed_upsampled_templates(
            lrt.temporal_components,
            ptps=np.ptp(template_data.templates, axis=1).max(1),
            max_upsample=matching_cfg.template_temporal_upsampling_factor,
            n_upsamples_map=matching_cfg.upsampling_compression_map,
        )

        if coarse_objective:
            obj_td = template_data.coarsen()
            obj_lrt = svd_compress_templates(
                obj_td,
                min_channel_amplitude=matching_cfg.template_min_channel_amplitude,
                rank=matching_cfg.template_svd_compression_rank,
            )
        else:
            obj_td = obj_lrt = None

        T_samples = recording.get_num_samples()
        dt = matching_cfg.chunk_length_samples
        chunk_starts = np.arange(0, T_samples, dt)
        chunk_ends = np.minimum(chunk_starts + dt, T_samples)
        chunk_centers_samples = (chunk_starts + chunk_ends) / 2
        chunk_centers_s = recording.sample_index_to_time(chunk_centers_samples)
        geom = recording.get_channel_locations()
        if coarse_objective:
            pconv_td = obj_td
            pconv_lrt = obj_lrt
        else:
            pconv_td = template_data
            pconv_lrt = lrt
        assert pconv_td is not None
        assert pconv_lrt is not None
        drifting = motion_est is not None and np.any(motion_est.displacement)
        pairwise_conv_db = CompressedPairwiseConv.from_template_data(
            save_folder / "pconv.h5",
            template_data=pconv_td,
            low_rank_templates=pconv_lrt,
            template_data_b=template_data if coarse_objective else None,
            low_rank_templates_b=lrt if coarse_objective else None,
            compressed_upsampled_temporal=cupt,
            chunk_time_centers_s=chunk_centers_s,
            motion_est=motion_est if drifting else None,
            geom=geom,
            computation_cfg=computation_cfg,
            overwrite=overwrite,
        )
        return cls(
            low_rank_templates=lrt,
            compressed_upsampled_temporal=cupt,
            refractory_radius_frames=matching_cfg.refractory_radius_frames,
            trough_offset_samples=template_data.trough_offset_samples,
            geom=geom,
            registered_geom=template_data.registered_geom,
            registered_template_depths_um=template_data.registered_depths_um(),
            pconv_db=pairwise_conv_db,
            motion_est=motion_est,
            dtype=dtype,
        )

    def data_at_time(
        self,
        t_s: float,
        scaling: bool,
        inv_lambda: float,
        scale_min: float,
        scale_max: float,
    ) -> "CompressedUpsampledChunkTemplateData":
        if self.drifting:
            shifts, padded_spatial_sing = templates_at_time(
                t_s,
                self.b.spatial_sing,
                self.geom,
                registered_template_depths_um=self.registered_template_depths_um,
                registered_geom=self.rgeom,
                motion_est=self.motion_est,
                return_pitch_shifts=True,
                geom_kdtree=self.geom_kdtree,
                match_distance=self.match_rad,
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

        if self.drifting and self.coarse_objective:
            obj_shifts, obj_spatial_sing = templates_at_time(
                t_s,
                self.b.spatial_sing,
                self.geom,
                registered_template_depths_um=self.registered_template_depths_um,
                registered_geom=self.rgeom,
                motion_est=self.motion_est,
                return_pitch_shifts=True,
                geom_kdtree=self.geom_kdtree,
                match_distance=self.match_rad,
                return_padded=False,
            )
            obj_shifts = torch.asarray(shifts, device=self.device)
            obj_spatial_sing = torch.asarray(obj_spatial_sing, device=self.device)
            obj_normsq = obj_spatial_sing.square().sum(dim=(1, 2))
        else:
            obj_shifts = shifts
            obj_spatial_sing = padded_spatial_sing[..., :-1]
            obj_normsq = normsq

        return CompressedUpsampledChunkTemplateData(
            coarse_objective=self.coarse_objective,
            grouping=self.have_groups,
            upsampling=self.upsampling,
            scaling=scaling,
            needs_fine_pass=self.have_groups or self.upsampling,
            comp_up_max=self.comp_up_max,
            n_templates=self.n_templates,
            obj_n_templates=self.obj_n_templates,
            spike_length_samples=self.spike_length_samples,
            up_factor=self.b.cup_index.shape[1],
            inv_lambda=torch.tensor(inv_lambda, device=normsq.device),
            scale_min=torch.tensor(scale_min, device=normsq.device),
            scale_max=torch.tensor(scale_max, device=normsq.device),
            obj_normsq=obj_normsq,
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
            coarse_index=self.b.coarse_index,
            group_index=self.b.group_index,
            unit_ids=self.b.unit_ids,
            fine_to_coarse=self.b.fine_to_coarse,
            main_channels=main_channels,
            conv_lags=self.b.conv_lags,
            refrac_ix=self.b.refrac_ix,
            rank_ix=self.b.rank_ix,
            time_ix=self.b.time_ix,
            chan_ix=self.b.chan_ix,
            pconv_db=self.pconv_db,
            shifts_a=shifts,
            shifts_b=obj_shifts,
        )


@dataclass(kw_only=True, slots=True, frozen=True, repr=False, eq=False)
class CompressedUpsampledChunkTemplateData(ChunkTemplateData):
    coarse_objective: bool
    grouping: bool
    upsampling: bool
    scaling: bool
    needs_fine_pass: bool
    comp_up_max: int
    n_templates: int
    obj_n_templates: int
    spike_length_samples: int
    up_factor: int
    inv_lambda: Tensor
    scale_min: Tensor
    scale_max: Tensor

    # objective props
    obj_normsq: Tensor
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
    coarse_index: Tensor
    group_index: Tensor | None
    unit_ids: Tensor
    fine_to_coarse: Tensor
    main_channels: Tensor
    conv_lags: Tensor
    refrac_ix: Tensor
    rank_ix: Tensor
    time_ix: Tensor
    chan_ix: Tensor

    # pconv
    pconv_db: PconvBase
    shifts_a: Tensor | None
    shifts_b: Tensor | None

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
        self, *, peaks: MatchingPeaks, residual: Tensor, conv: Tensor, padding: int = 0
    ):
        """Determine superres ids, temporal upsampling, and scaling

        Given coarse matches (unit ids at times) and the current residual,
        pick the best superres template, the best temporal offset, and the
        best amplitude scaling.

        We used to upsample the objective to figure out the temporal upsampling,
        but with superres in the picture we are now not computing the objective
        using the same templates that we temporally upsample. So, instead
        we use a greedy strategy: first pick the best (non-temporally upsampled)
        superres template, then pick the upsampling and scaling at the same time.
        These are all done by dotting everything and computing the objective,
        which is probably more expensive than what we had before.

        Returns
        -------
        time_shifts : Optional[array]
        up_inds : Optional[array]
        scalings : Optional[array]
        template_inds : array
        objs : array
        """
        if not self.needs_fine_pass:
            return peaks
        if not peaks.n_spikes:
            return peaks
        del conv, padding  # unused

        if self.coarse_objective or self.upsampling:
            residual_snips = grab_spikes_full(
                residual,
                peaks.times,
                trough_offset=0,
                spike_length_samples=self.spike_length_samples + 1,
            )
        else:
            residual_snips = None

        if self.coarse_objective:
            assert residual_snips is not None
            # TODO best I came up with, but it still syncs
            # can refactor with a jagged/nested tensor. not prioritizing since
            # grouped is rare for now.
            superres_ix = self.coarse_index[peaks.template_inds]
            dup_ix, column_ix = (superres_ix < self.n_templates).nonzero(as_tuple=True)
            template_inds = superres_ix[dup_ix, column_ix]
            convs = torch.einsum(
                "ntr,ntc,nrc->n",
                self.temporal_comps[template_inds],
                residual_snips[dup_ix],
                self.spatial_sing[template_inds],
            )
            norms = self.normsq[template_inds]
            objs = convs.new_full(superres_ix.shape, -torch.inf)
            objs[dup_ix, column_ix] = torch.add(norms._neg_view(), convs, alpha=2.0)
            objs, best_column_ix = objs.max(dim=1)
            row_ix = torch.arange(best_column_ix.numel(), device=best_column_ix.device)
            template_inds = superres_ix[row_ix, best_column_ix]
        else:
            template_inds = peaks.template_inds
            norms = self.normsq[template_inds]
            objs = peaks.scores

        if not self.upsampling:
            return MatchingPeaks(
                times=peaks.times,
                obj_template_inds=peaks.obj_template_inds,
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
            obj_template_inds=peaks.obj_template_inds,
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

    def _enforce_refractory(self, mask, peaks, offset=0, value=-torch.inf):
        if not peaks.n_spikes:
            return
        assert peaks.times is not None
        time_ix = peaks.times[:, None] + (self.refrac_ix[None, :] + offset)
        if self.coarse_objective:
            assert peaks.obj_template_inds is not None
            row_ix = peaks.obj_template_inds[:, None]
        elif self.grouping:
            assert peaks.template_inds is not None
            assert self.group_index is not None
            row_ix = self.group_index[peaks.template_inds]
            row_ix = row_ix[:, :, None]
            time_ix = time_ix[:, None, :]
        else:
            assert peaks.template_inds is not None
            row_ix = peaks.template_inds[:, None]
        mask[row_ix, time_ix] = value

    def reconstruct_up_templates(self):
        up_comps = self.cup_temporal[self.cup_map].cpu()
        return torch.einsum("nrc,nutr->nutc", self.spatial_sing.cpu(), up_comps)


def handle_template_groups(obj_unit_ids, unit_ids, coarse_objective: bool):
    """Grouped templates in objective

    If not coarse_objective, then several rows of the objective may
    belong to the same unit. They must be handled together when imposing
    refractory conditions.
    """
    _u, fine_to_coarse, counts = unit_ids.unique(
        return_inverse=True, return_counts=True
    )
    max_group_size = counts.max()
    have_groups = max_group_size > 1

    if not have_groups:
        assert not coarse_objective
        return have_groups, None, None, None

    if have_groups and not coarse_objective:
        # refractory enforcement helper array for grouped non-coarse objective
        # like a channel index, sort of
        # this is a n_templates x group_size array that maps each
        # template index to the set of other template indices that
        # are part of its group. so that the array is not ragged,
        # we pad rows with -1s when their group is smaller than the
        # largest group.
        _u, counts = unit_ids.unique(return_counts=True)
        assert torch.equal(_u, unit_ids)
        assert (_u >= 0).all()
        group_index = torch.full((len(unit_ids), counts.max()), -1)
        for j, u in enumerate(unit_ids):
            (row,) = torch.nonzero(unit_ids == u, as_tuple=True)
            group_index[j, : len(row)] = row

        return have_groups, group_index, None, None

    # coarse objective with groups
    assert obj_unit_ids is not None
    assert have_groups  # otherwise coarse_objective would be false.
    _u, fine_to_coarse, counts = torch.unique(
        unit_ids, return_counts=True, return_inverse=True
    )
    assert torch.equal(_u, obj_unit_ids)
    assert (_u >= 0).all()

    coarse_index = torch.full((len(obj_unit_ids), counts.max()), len(unit_ids))
    for j, u in enumerate(obj_unit_ids):
        (group,) = torch.nonzero(unit_ids == u, as_tuple=True)
        coarse_index[j, : len(group)] = group

    return have_groups, None, fine_to_coarse, coarse_index
