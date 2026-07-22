"""Fit and apply the GMM of clustering/mixture.py as a transform node.

TODO:
 - I don't love that this depends on the neighborhoods in advance in this way.
   If there's a new neighborhood encountered in transform(), we should handle
   that. Could do this by checking all neighborhoods in fit()? Or dropping the
   precomputation neighb_cov/lut stuff entirely if possible to do that efficiently?
 - Trim down the TMM. This class should probably just take only what's needed from
   the TMM (only some bufs used for inference) and toss the rest.
 - Figure out the memory story... seems like this is either taking up a lot of space
   or preventing torch from freeing stuff due to the fitting process??
"""

import sys
from dataclasses import replace
from math import fabs
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    RefinementConfig,
    WaveformConfig,
)
from ..util.interpolation_util import StableFeaturesInterpolator, pad_geom
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import handle_negative_jobs
from ..util.py_util import panic
from ..util.torch_util import cleanup_and_log_gpu_usage, torch_compile
from .transform_base import BaseWaveformFeaturizer

if TYPE_CHECKING:
    from ..clustering.mixture import MixtureModelAndDatasets, TruncatedMixtureModel


class TruncatedMixtureModelTransformer(BaseWaveformFeaturizer):
    """Gaussian mixture clustering and classification as a featurization node."""

    is_multi = True
    needs_residual = True
    fits_from_disk = True
    needs_more_features = True

    def __init__(
        self,
        *,
        channel_index=None,
        geom=None,
        name=None,
        name_prefix=None,
        motion: MotionInfo | None = None,
        clustering_cfg: ClusteringConfig | None,
        clustering_features_cfg: ClusteringFeaturesConfig,
        pre_gmm_refinement_cfgs: Sequence[RefinementConfig | None] | None,
        gmm_refinement_cfg: RefinementConfig,
        waveform_cfg: WaveformConfig,
        sampling_frequency: float = 30_000.0,
        save_neighborhood_ids: bool = False,
    ):
        self.motion = motion
        self.clustering_cfg = clustering_cfg
        self.clustering_features_cfg = clustering_features_cfg
        self.pre_gmm_refinement_cfgs = pre_gmm_refinement_cfgs
        self.gmm_refinement_cfg = gmm_refinement_cfg

        amp_ds: str = self.clustering_features_cfg.amplitudes_dataset_name
        volt_ds: str = self.clustering_features_cfg.voltages_dataset_name
        pca_ds: str = self.clustering_features_cfg.pca_dataset_name
        loc_ds: str = self.clustering_features_cfg.localizations_dataset_name
        self.motion_depth_mode = clustering_features_cfg.motion_depth_mode
        need_loc = (
            self.motion_depth_mode == "localization" or self.clustering_cfg is not None
        )
        self.load_feat_names: list[str] = [amp_ds, volt_ds]
        if need_loc:
            self.load_feat_names.append(loc_ds)
        else:
            # can skip locations
            self.clustering_features_cfg = replace(
                self.clustering_features_cfg, use_x=False, use_z=False
            )
        self.pca_ds: str = pca_ds
        self.loc_ds: str = loc_ds

        assert name is None
        name = ("labels", "gmm_candidates", "gmm_log_liks", "gmm_responsibilities")
        if save_neighborhood_ids:
            name = (*name, "neighborhood_ids")
        super().__init__(
            channel_index=channel_index,
            geom=geom,
            name=name,
            name_prefix=name_prefix,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
        )

        ncand = gmm_refinement_cfg.n_candidates
        assert gmm_refinement_cfg.robust_strategy == "none"  # not implemented atm
        self.n_candidates = ncand
        self.feature_rank = self.clustering_features_cfg.feature_rank
        self.save_neighborhood_ids = save_neighborhood_ids
        self.shape = [(), (ncand,), (ncand + 1,), (ncand + 1,)]
        self.dtype = [torch.int32, torch.int32, torch.float32, torch.float32]
        if save_neighborhood_ids:
            self.shape += [()]
            self.dtype += [torch.int32]
        self.save_neighborhood_ids = save_neighborhood_ids
        self.motion: MotionInfo | None = None
        self.channel_index_np = self.b.channel_index.numpy(force=True)
        self.workers = 1

    def get_extra_state(self):
        es = super().get_extra_state()
        es["feature_rank"] = self.feature_rank
        return es

    def needs_fit(self):
        return not hasattr(self, "tmm")

    def attach_motion(self, motion: MotionInfo):
        self.motion = motion

    def register_cpu_workers(self, workers: int):
        self.workers = workers

    def needs_precompute(self):
        if not hasattr(self, "tmm"):
            return False
        else:
            return (
                self.tmm.lut_params is None
                or not hasattr(self, "neighb_candidates")
                or self.tmm.neighb_cov.b.Linv.shape[0]
                < self.b.neighb_candidates.shape[0]
            )

    def precompute(self):
        if not hasattr(self, "tmm"):
            return
        self.tmm.update_lut(self.tmm.lut, puff=0.0)
        self.update_proposals()
        # special handling of unmatched neighborhoods
        self.tmm.neighb_cov.pad_for_noise_score_(self.b.neighb_candidates.shape[0])
        assert self.motion is not None
        geomp = np.pad(self.motion.geom, [(0, 1), (0, 0)], constant_values=np.nan)
        self.static_neighbs = geomp[self.channel_index_np]
        cii, cjj = np.nonzero(self.channel_index_np < len(self.channel_index_np))
        self.channel_index_valid_inds = cii, cjj

        # precompute drifting neighborhood lookup tables
        if self.motion.drifting:
            # in this case, prebake a lookup table
            # TODO just consider relevant time bins here?
            # or chunk this somehow?
            tbs = self.motion.time_bins_s
            tcm_t, nid_t = neighborhood_mapping_at_time(
                motion=self.motion,
                t_s=torch.from_numpy(tbs),
                neighborhoods=self.b.neighborhoods,
                channel_index=self.channel_index_np,
                workers=self.workers,
                static_neighbs=self.static_neighbs,
                channel_index_valid_inds=self.channel_index_valid_inds,
            )
            nid_map = None
        else:
            tcm_t = nid_t = tbs = None
            # in this case, channel index is a superset of neighborhoods and
            # we prebake the mapping from chans to neighb ids
            ci_eq_neighb = _outer_all_equal(
                self.b.channel_index, self.tmm.neighb_cov.obs_ix
            )
            # some neighborhood entries are identical, but they all get covered
            assert torch.all(ci_eq_neighb.sum(0) >= 1)
            # each channel index maps to at most one neighb id
            assert torch.all(ci_eq_neighb.sum(1) <= 1)
            _chans, _chan_nids = ci_eq_neighb.nonzero(as_tuple=True)
            nid_map = _chans.new_full(
                (self.b.channel_index.shape[0],),
                self.tmm.neighb_cov.b.obs_ix.shape[0] + 1,
            )
            nid_map[_chans] = _chan_nids
        self.register_buffer_or_none("neighborhood_ids_map", nid_map)
        self.drifting_time_bins_s = tbs
        self.register_buffer_or_none(
            "drifting_target_channels_map_t", tcm_t, persistent=False
        )
        self.register_buffer_or_none(
            "drifting_neighborhood_ids_map_t", nid_t, persistent=False
        )

    def fit(
        self,
        recording,
        waveforms,
        *,
        hdf5_filename=None,
        pipeline=None,
        computation_cfg,
        **spike_data,
    ):
        from ..clustering import (
            SimpleMatrixFeatures,
            StableWaveformFeatures,
            TMMRefinement,
            get_clusterer,
        )

        del waveforms  # I load everything from h5
        assert self.motion is not None
        assert hdf5_filename is not None

        assert pipeline is not None
        tpca = pipeline.get_transformer(self.pca_ds)

        # build sorting object
        load_features = ["times_seconds", *self.load_feat_names]
        sorting = DARTsortSorting.from_peeling_hdf5(
            h5_path=hdf5_filename, load_feature_names=load_features
        )
        if sorting.labels is None:
            assert self.clustering_cfg is not None

        # build clusterer
        ref_cfgs = self.pre_gmm_refinement_cfgs or []
        ref_cfgs = [*ref_cfgs, self.gmm_refinement_cfg]
        clus = get_clusterer(
            clustering_cfg=self.clustering_cfg,
            refinement_cfgs=ref_cfgs,
            computation_cfg=computation_cfg,
        )
        assert isinstance(clus, TMMRefinement)

        # build features, init mixture model
        simple_features = SimpleMatrixFeatures.from_config(
            sorting=sorting,
            motion=self.motion,
            clustering_features_cfg=self.clustering_features_cfg,
            computation_cfg=computation_cfg,
        )
        stable_features = StableWaveformFeatures.from_config(
            sorting=sorting,
            motion=self.motion,
            clustering_features_cfg=self.clustering_features_cfg,
            computation_cfg=computation_cfg,
        )
        mix_data: "MixtureModelAndDatasets" = clus.get_tmm(
            features=simple_features,
            stable_features=stable_features,
            sorting=sorting,
            motion=self.motion,
            tpca=tpca,
        )
        assert mix_data.tmm.noise is not None
        self.feature_rank = mix_data.tmm.noise.rank

        self.erp: StableFeaturesInterpolator = stable_features.erp

        # de-puff the lut for space reasons
        mix_data.tmm.lut_params = None
        mix_data.tmm.update_lut(mix_data.tmm.lut, puff=0.0)

        # these guys are bad for torch.save() with weights only
        # and not needed for inference
        # one could implement serialization logic for them, I just didn't
        mix_data.tmm.erp = None
        self.tmm: "TruncatedMixtureModel" = mix_data.tmm
        self.register_buffer("neighborhoods", mix_data.tmm.neighb_cov.b.obs_ix.clone())
        _, self.workers = handle_negative_jobs(
            computation_cfg.actual_n_jobs(small=True)
        )
        assert sys.getrefcount(mix_data) <= 2, sys.getrefcount(mix_data)
        del mix_data
        cleanup_and_log_gpu_usage(
            computation_cfg=computation_cfg, message=f"{self.__class__.__name__}: Free"
        )
        self.precompute()

    def update_proposals(self):
        # pad both these to allow unknown neighborhoods
        # TODO: if this is common, consider reworking candidate logic for inference.
        # could, for instance, precompute all possible neighborhoods and allow candidates
        # when neighborhoods are subsets of existing LUT neighborhoods. or, could expand
        # the search at inference time (like explore steps).
        neighb_candidates = self.tmm.lut.full_proposal_candidates()
        neighb_candidates = F.pad(neighb_candidates, (0, 0, 0, 1), value=-1)
        self.register_buffer("neighb_candidates", neighb_candidates)
        self.register_buffer("neighb_candidate_counts", (neighb_candidates >= 0).sum(1))

    def _other_pre_load_state(self, state_dict, prefix):
        from ..clustering.mixture import TruncatedMixtureModel
        from ..util.interpolation_util import SpikeNeighborhoods

        assert hasattr(self, "motion")
        assert self.motion is not None
        extra_state = state_dict[f"{prefix}_extra_state"]
        self.feature_rank = extra_state["feature_rank"]

        # reconstruct non-__init__ modules...
        stripped_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        for k in (
            "neighb_candidates",
            "neighb_candidate_counts",
            "neighborhood_ids_map",
            "neighborhoods",
        ):
            if k in stripped_dict:
                self.register_buffer(k, stripped_dict[k])
        neighborhoods = SpikeNeighborhoods(
            self.motion.rgeom.shape[0],
            neighborhood_ids=None,
            neighborhoods=self.b.neighborhoods,
        )
        self.erp = StableFeaturesInterpolator(
            source_geom=pad_geom(self.motion.geom),
            target_geom=pad_geom(self.motion.rgeom),
            channel_index=self.b.channel_index.clone(),
            params=self.clustering_features_cfg.interp_params,
        )

        tmm_dict = {
            k.removeprefix("tmm."): v
            for k, v in stripped_dict.items()
            if k.startswith("tmm.")
        }
        self.tmm = TruncatedMixtureModel.from_state_dict(
            self.motion,
            tmm_dict,
            neighborhoods=neighborhoods,
            feature_rank=self.feature_rank,
            refinement_cfg=self.gmm_refinement_cfg,
        )
        self.tmm.lut_params = None

    def transform(self, waveforms, *, channels, **spike_data):
        t_s = spike_data["times_seconds"]
        chunk_center_s = spike_data["chunk_center_s"]
        if self.motion_depth_mode == "localization":
            z = spike_data[self.loc_ds][:, 2].numpy(force=True)
        elif self.motion_depth_mode == "channel":
            z = self.b.geom[channels, 1].numpy(force=True)
        else:
            panic(self.motion_depth_mode)

        # drifting channel mapping
        assert self.motion is not None
        if self.motion.drifting:
            assert self.drifting_time_bins_s is not None
            t_idx = find_nearest(self.drifting_time_bins_s, chunk_center_s)
            target_channels_map = self.b.drifting_target_channels_map_t[t_idx]
            neighborhood_ids_map = self.b.drifting_neighborhood_ids_map_t[t_idx]
        else:
            target_channels_map = self.b.channel_index
            neighborhood_ids_map = self.b.neighborhood_ids_map

        # interpolate features
        features = spike_data[self.pca_ds]
        neighborhood_ids = neighborhood_ids_map[channels]
        if self.motion.drifting:
            # TODO motion on gpu
            source_shifts = self.motion.disp_at_s(
                times_s=t_s.numpy(force=True), depths_um=z
            )
            source_shifts = -torch.asarray(source_shifts, device=waveforms.device)
            target_channels = target_channels_map[channels]
            features = self.erp.interp(
                features=features,
                source_main_channels=channels,
                target_channels=target_channels,
                source_shifts=source_shifts,
            )
            features = features.nan_to_num_()
        else:
            features = features.nan_to_num()

        # soft assignment
        cand_count = self.b.neighb_candidate_counts[neighborhood_ids].sum()
        candidates = self.b.neighb_candidates[neighborhood_ids]
        labels, scores = self.tmm.score_features(
            features=features,
            candidates=candidates,
            neighborhood_ids=neighborhood_ids,
            n_candidates=self.n_candidates,
            candidate_count=int(cand_count.item()),
            duties=None,
        )

        res = {
            "labels": labels.to(dtype=torch.int32),
            "gmm_candidates": scores.candidates.to(dtype=torch.int32),
            "gmm_log_liks": scores.log_liks,
            "gmm_responsibilities": scores.responsibilities,
        }
        if self.save_neighborhood_ids:
            res["neighborhood_ids"] = neighborhood_ids.to(dtype=torch.int32)
        return res


def neighborhood_mapping_at_time(
    motion: MotionInfo,
    *,
    t_s: torch.Tensor | np.ndarray,
    channel_index: np.ndarray,
    neighborhoods: torch.Tensor,
    workers: int = 4,
    shift_mode="round",
    static_neighbs: np.ndarray,
    channel_index_valid_inds: tuple[np.ndarray, np.ndarray],
):
    t_s = t_s.numpy(force=True) if isinstance(t_s, torch.Tensor) else np.asarray(t_s)
    t_s = np.atleast_1d(t_s)

    index_at_time = torch.full((len(t_s), *channel_index.shape), -1)
    nids_at_time = torch.full(
        (len(t_s), channel_index.shape[0]),
        neighborhoods.shape[0],
        device=neighborhoods.device,
    )

    for j, t in enumerate(t_s):
        probe_disp = -motion.disp_at_s(
            times_s=t, depths_um=motion.geom[:, 1], grid=True
        )
        if shift_mode == "floor":
            n_pitches_shift = (probe_disp / motion.pitch).astype(np.int32)
        elif shift_mode == "round":
            n_pitches_shift = np.round(probe_disp / motion.pitch).astype(np.int32)
        else:
            panic(shift_mode)
        shift = n_pitches_shift * motion.pitch
        shifted_neighbs = static_neighbs.copy()
        shifted_neighbs[:, :, 1] += shift
        _, umatch = motion.rgeom_kdt.query(
            shifted_neighbs[channel_index_valid_inds],
            distance_upper_bound=motion.min_dist,
            workers=workers,
        )
        index = np.full(channel_index.shape, motion.rgeom.shape[0])
        index[channel_index_valid_inds] = umatch
        index = torch.asarray(index, device=neighborhoods.device)
        assert index.shape[1] == neighborhoods.shape[1]

        index_at_time[j] = index

        mapping = _outer_all_equal(index, neighborhoods)
        _chans, _chan_nids = mapping.nonzero(as_tuple=True)
        nids_at_time[j, _chans] = _chan_nids

    assert (index_at_time >= 0).all()

    return index_at_time, nids_at_time


@torch_compile
def _outer_all_equal(x: torch.Tensor, y: torch.Tensor):
    dim = x.shape[1]
    x = x[:, None, :]
    y = y[None, :, :]

    out = x[:, :, 0] == y[:, :, 0]
    if dim == 1:
        return out

    msk = x[:, :, 1] == y[:, :, 1]
    out.logical_and_(msk)

    for j in range(2, dim):
        out.logical_and_(torch.eq(x[:, :, j], y[:, :, j], out=msk))

    return out


def find_nearest(x, v):
    idx = np.searchsorted(x, v, side="left")
    if idx > 0 and (idx == len(x) or fabs(v - x[idx - 1]) < fabs(v - x[idx])):
        return idx - 1
    else:
        return idx
