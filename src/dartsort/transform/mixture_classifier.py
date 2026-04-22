"""Fit and apply the GMM of clustering/mixture.py as a transform node."""

from dataclasses import replace
from typing import Sequence, TYPE_CHECKING

import torch
import numpy as np

from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    RefinementConfig,
    WaveformConfig,
)
from ..util.motion import MotionInfo
from .transform_base import BaseWaveformFeaturizer

if TYPE_CHECKING:
    from ..clustering.mixture import MixtureModelAndDatasets, TruncatedMixtureModel
    from ..util.interpolation_util import StableFeaturesInterpolator


class TruncatedMixtureModelTransformer(BaseWaveformFeaturizer):
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
        self.shape = [(), (ncand,), (ncand + 1,), (ncand + 1,)]
        self.dtype = [torch.int32, torch.int32, torch.float32, torch.float32]
        self.motion: MotionInfo | None = None
        self.channel_index_np = self.b.channel_index.numpy(force=True)
        self.workers = 1

    def needs_fit(self):
        return not hasattr(self, "tmm")

    def attach_motion(self, motion: MotionInfo):
        self.motion = motion

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
        load_features = ["times_seconds"] + self.load_feat_names
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

        if not self.motion.drifting:
            # in this case, channel index is a superset of neighborhoods and
            # we prebake the mapping from chans to neighb ids
            ci_eq_neighb = _outer_all_equal(
                self.b.channel_index, mix_data.tmm.neighb_cov.obs_ix
            )
            # some neighborhood entries are identical, but they all get covered
            assert torch.all(ci_eq_neighb.sum(0) >= 1)
            # each channel index maps to at most one neighb id
            assert torch.all(ci_eq_neighb.sum(1) <= 1)
            _chans, _chan_nids = ci_eq_neighb.nonzero(as_tuple=True)
            nid_map = _chans.new_full(
                (self.b.channel_index.shape[0],),
                mix_data.tmm.neighb_cov.obs_ix.shape[0] + 1,
            )
            nid_map[_chans] = _chan_nids
            self.register_buffer("neighborhood_ids_map", nid_map)

        self.erp: StableFeaturesInterpolator = stable_features.erp
        self.tmm: "TruncatedMixtureModel" = mix_data.tmm
        self.register_buffer("neighborhoods", mix_data.tmm.neighb_cov.obs_ix.clone())
        self.workers = computation_cfg.actual_n_jobs(small=True)
        neighb_candidates = mix_data.tmm.lut.full_proposal_candidates()
        self.register_buffer("neighb_candidates", neighb_candidates)
        self.register_buffer("neighb_candidate_counts", (neighb_candidates >= 0).sum(1))

    def transform(self, waveforms, *, channels, **spike_data):
        t_s = spike_data["times_seconds"]
        chunk_center_s = spike_data["chunk_center_s"]
        if self.motion_depth_mode == "localization":
            z = spike_data[self.loc_ds][:, 2].numpy(force=True)
        elif self.motion_depth_mode == "channel":
            z = self.b.geom[channels, 1].numpy(force=True)
        else:
            assert False

        # drifting channel mapping
        assert self.motion is not None
        if self.motion.drifting:
            target_channels_map, neighborhood_ids_map = neighborhood_mapping_at_time(
                motion=self.motion,
                t_s=chunk_center_s,
                neighborhoods=self.b.neighborhoods,
                channel_index=self.channel_index_np,
                workers=self.workers,
            )
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
            source_shifts = torch.asarray(source_shifts, device=waveforms.device)
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
        scores = self.tmm.score_features(
            features=features,
            candidates=candidates,
            neighborhood_ids=neighborhood_ids,
            n_candidates=self.n_candidates,
            candidate_count=int(cand_count.item()),
            duties=None,
        )

        return {
            "labels": scores.candidates[:, 0].to(dtype=torch.int32),
            "gmm_candidates": scores.candidates.to(dtype=torch.int32),
            "gmm_log_liks": scores.log_liks,
            "gmm_responsibilities": scores.responsibilities,
        }


def neighborhood_mapping_at_time(
    motion: MotionInfo,
    t_s: torch.Tensor,
    channel_index: np.ndarray,
    neighborhoods: torch.Tensor,
    workers: int = 4,
):
    # TODO... what if a new neighborhood is observed during rest
    # of the matching? it will crash atm, which is good

    # shifted geom neighborhoods at time
    disp = motion.disp_at_s(
        times_s=t_s.numpy(force=True), depths_um=motion.geom[:, 1], grid=True
    )
    assert disp.shape == (motion.geom.shape[0], 1)
    shifted_geom = motion.geom.copy()
    shifted_geom[:, 1] -= disp[:, 0]

    # match shifted geom channels to rgeom channels
    _, channels = motion.rgeom_kdt.query(
        shifted_geom, distance_upper_bound=motion.min_dist, workers=workers
    )

    # get shifted channel neighborhoods
    index = np.full(channel_index.shape, motion.rgeom.shape[0])
    ii, jj = np.nonzero(channel_index < len(channel_index))
    index[ii, jj] = channels[channel_index[ii, jj]]

    # mapping between these and `neighborhoods`
    index = torch.asarray(index, device=neighborhoods.device)
    assert index.shape[1] == neighborhoods.shape[1]
    mapping = _outer_all_equal(index, neighborhoods)
    _chans, _chan_nids = mapping.nonzero(as_tuple=True)
    nid_map = _chans.new_full((channel_index.shape[0],), neighborhoods.shape[0] + 1)
    nid_map[_chans] = _chan_nids

    return index, nid_map


@torch.jit.script
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
