"""Fit and apply the GMM of clustering/mixture.py as a transform node."""

from typing import Sequence

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


class TruncatedMixtureModelTransformer(BaseWaveformFeaturizer):
    is_multi = True

    def __init__(
        self,
        *,
        channel_index=None,
        geom=None,
        name=None,
        name_prefix=None,
        motion: MotionInfo,
        clustering_cfg: ClusteringConfig | None,
        clustering_features_cfg: ClusteringFeaturesConfig,
        pre_gmm_refinement_cfgs: Sequence[RefinementConfig] | None,
        gmm_refinement_cfg: RefinementConfig,
        waveform_cfg: WaveformConfig,
        sampling_frequency: float = 30_000.0,
    ):
        self.motion = motion
        self.clustering_cfg = clustering_cfg
        self.clustering_features_cfg = clustering_features_cfg
        self.pre_gmm_refinement_cfgs = pre_gmm_refinement_cfgs
        self.gmm_refinement_cfg = gmm_refinement_cfg

        assert name is None
        name = ("gmm_candidates", "gmm_log_liks", "gmm_responsibilities")
        super().__init__(
            channel_index=channel_index,
            geom=geom,
            name=name,
            name_prefix=name_prefix,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
        )

        ncand = gmm_refinement_cfg.n_candidates
        self.motion_depth_mode = clustering_features_cfg.motion_depth_mode
        assert gmm_refinement_cfg.robust_strategy == "none"  # not implemented atm
        self.n_candidates = ncand
        self.shape = [(ncand,), (ncand + 1,), (ncand + 1,)]
        self.dtype = [torch.int16, torch.float32, torch.float32]
        self.motion: MotionInfo | None = None
        self.channel_index_np = self.b.channel_index.numpy(force=True)
        self.workers = 1

    def attach_motion(self, motion: MotionInfo):
        self.motion = motion

    def fit(self, recording, waveforms, *, computation_cfg, channels, **spike_data):
        from ..clustering import (
            SimpleMatrixFeatures,
            StableWaveformFeatures,
            TMMRefinement,
            get_clusterer,
        )

        assert self.motion is not None

        # build sorting object
        labels = spike_data.get("labels")
        if labels is None:
            assert self.clustering_cfg is not None
        self.pca_ds = self.clustering_features_cfg.pca_dataset_name
        amp_ds = self.clustering_features_cfg.amplitudes_dataset_name
        self.loc_ds = self.clustering_features_cfg.localizations_dataset_name
        volt_ds = self.clustering_features_cfg.voltages_dataset_name
        sorting = DARTsortSorting(
            times_samples=spike_data["times_samples"],
            channels=channels,
            labels=labels,
            ephemeral_features={
                "channel_index": self.b.channel_index.numpy(force=True),
                "geom": self.b.geom.numpy(force=True),
                "times_seconds": spike_data["times_seconds"],
                self.pca_ds: spike_data[self.pca_ds],
                volt_ds: spike_data[volt_ds],
                amp_ds: spike_data[amp_ds],
                self.loc_ds: spike_data[self.loc_ds],
            },
            sampling_frequency=recording.sampling_frequency,
        )

        # build clusterer
        ref_cfgs = self.pre_gmm_refinement_cfgs or []
        ref_cfgs = [*ref_cfgs, self.gmm_refinement_cfg]
        clus = get_clusterer(
            clustering_cfg=self.clustering_cfg,
            refinement_cfgs=ref_cfgs,
            computation_cfg=computation_cfg,
        )
        assert isinstance(clus, TMMRefinement)

        # build features
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

        mix_data = clus.get_tmm(
            features=simple_features,
            stable_features=stable_features,
            sorting=sorting,
            motion=self.motion,
        )

        self.erp = stable_features.erp
        self.tmm = mix_data.tmm
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
        target_channels_map, neighborhood_ids_map = neighborhood_mapping_at_time(
            motion=self.motion,
            t_s=chunk_center_s,
            neighborhoods=self.b.neighborhoods,
            channel_index=self.channel_index_np,
            workers=self.workers,
        )

        # per-spike channels and drifts
        # TODO motion on gpu
        source_shifts = self.motion.disp_at_s(times_s=t_s, depths_um=z)
        source_shifts = torch.asarray(source_shifts, device=waveforms.device)
        target_channels = target_channels_map[channels]
        neighborhood_ids = neighborhood_ids_map[channels]

        # interpolate features
        features = spike_data[self.pca_ds]
        if self.motion.drifting:
            features = self.erp.interp(
                features=features,
                source_main_channels=channels,
                target_channels=target_channels,
                source_shifts=source_shifts,
            )

        # soft assignment
        cand_count = self.b.neighb_candidate_counts[neighborhood_ids].sum()
        candidates = self.b.neighb_candidates[neighborhood_ids]
        scores = self.tmm.score_features(
            features=features,
            candidates=candidates,
            neighborhood_ids=neighborhood_ids,
            n_candidates=self.n_candidates,
            candidate_count=cand_count.item(),
            duties=None,
        )

        return {
            "gmm_candidates": scores.candidates,
            "gmm_log_liks": scores.log_liks,
            "gmm_responsibilities": scores.responsibilities,
        }


def neighborhood_mapping_at_time(
    motion: MotionInfo,
    t_s: float,
    channel_index: np.ndarray,
    neighborhoods: torch.Tensor,
    workers: int = 4,
):
    if not motion.drifting:
        return channel_index

    # shifted geom neighborhoods at time
    disp = motion.disp_at_s(
        times_s=np.atleast_1d(t_s), depths_um=motion.geom[:, 1], grid=True
    )
    assert disp.shape == (1, motion.geom.shape[0])
    neighb0 = motion.geom[channel_index]
    source_shifts = disp[channel_index]
    neighb0[:, :, 1] -= source_shifts

    # match to rgeom channels
    _, channels = motion.rgeom_kdt.query(
        neighb0, distance_upper_bound=motion.min_dist, workers=workers
    )
    assert channels.shape == channel_index.shape

    # mapping between these and `neighborhoods`
    channels = torch.asarray(channels, device=neighborhoods.device)
    assert channels.shape[1] == neighborhoods.shape[1]
    mapping = _outer_all_equal(channels, neighborhoods)
    _, chan_to_neighb_id = mapping.nonzero(as_tuple=True)
    # check everyone got a match
    assert chan_to_neighb_id.shape == (motion.geom.shape[0],)

    return channels, chan_to_neighb_id


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
