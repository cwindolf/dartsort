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
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording

from ..util.data_util import DARTsortSorting
from ..util.drift_util import get_stable_channels
from ..util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    ComputationConfig,
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
    from .pipeline import WaveformPipeline


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
        self.motion: MotionInfo | None = motion
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
                self.clustering_features_cfg,
                use_x=False,
                use_z=False,
                n_main_channel_pcs=0,
                n_multi_channel_pcs=0,
                amplitude_kind="stored",
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
        assert gmm_refinement_cfg.robust_strategy in ("none", "huber")
        self.n_candidates = ncand
        self.feature_rank = self.clustering_features_cfg.feature_rank
        self.save_neighborhood_ids = save_neighborhood_ids
        self.shape = [(), (ncand,), (ncand + 1,), (ncand + 1,)]
        self.dtype = [torch.int32, torch.int32, torch.float32, torch.float32]
        if save_neighborhood_ids:
            self.shape += [()]
            self.dtype += [torch.int32]
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
                or not hasattr(self, "neighborhood_ids_by_shift")
                or self.tmm.neighb_cov.b.spatial_whitener.shape[0]
                < self.b.neighb_candidates.shape[0]
            )

    def precompute(self):
        if not hasattr(self, "tmm"):
            return
        self.tmm.update_lut(self.tmm.lut, puff=0.0)
        self.update_proposals()
        # special handling of unmatched neighborhoods
        self.tmm.neighb_cov.pad_for_noise_score_(self.b.neighb_candidates.shape[0])
        self._precompute_neighborhoods_by_shift()

    def _precompute_neighborhoods_by_shift(self):
        assert self.motion is not None
        n_channels = self.b.channel_index.shape[0]
        if self.motion.drifting:
            time_bins_s = self.motion.time_bins_s
            _, n_pitches_shift = self.motion.pitch_shifts(
                times_s=np.repeat(time_bins_s, n_channels),
                depths_um=np.tile(self.motion.geom[:, 1], len(time_bins_s)),
            )
            # margin for localization depths
            min_shift = int(n_pitches_shift.min()) - 1
            max_shift = int(n_pitches_shift.max()) + 1
        else:
            min_shift = max_shift = 0
        n_shifts = max_shift - min_shift + 1

        # channel sets like in StableWaveformFeatures for each chan, shift
        target_channels, *_ = get_stable_channels(
            motion=self.motion,
            channels=np.tile(np.arange(n_channels), n_shifts),
            channel_index=self.b.channel_index.numpy(force=True),
            n_pitches_shift=np.repeat(np.arange(min_shift, max_shift + 1), n_channels),
            workers=self.workers,
        )
        neighborhoods = self.b.neighborhoods
        target_channels = torch.asarray(target_channels, device=neighborhoods.device)
        rows, nids = _outer_all_equal(target_channels, neighborhoods).nonzero(
            as_tuple=True
        )
        neighborhood_ids = rows.new_full((len(target_channels),), len(neighborhoods))
        neighborhood_ids[rows] = nids

        self.min_n_pitches_shift = min_shift
        self.register_buffer(
            "target_channels_by_shift",
            target_channels.view(n_shifts, n_channels, -1).transpose(0, 1),
            persistent=False,
        )
        self.register_buffer(
            "neighborhood_ids_by_shift",
            neighborhood_ids.view(n_shifts, n_channels).T,
            persistent=False,
        )

    def fit_from_disk(
        self,
        *,
        recording: BaseRecording,
        hdf5_filename: Path,
        pipeline: "WaveformPipeline",
        computation_cfg: ComputationConfig,
        save_fit_data: bool = False,
    ) -> None:
        from ..clustering import (
            SimpleMatrixFeatures,
            StableWaveformFeatures,
            TMMRefinement,
            get_clusterer,
        )

        del recording
        assert self.motion is not None
        tpca = pipeline.get_transformer(self.pca_ds)

        # build sorting object
        load_features = ["times_seconds", *self.load_feat_names]
        sorting = DARTsortSorting.from_peeling_hdf5(
            h5_path=hdf5_filename,
            load_feature_names=load_features,
            load_all_features=False,
            load_simple_features=False,
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
            save_labels_dir=hdf5_filename.parent if save_fit_data else None,
            initial_name="classifier_initial",
            refine_labels_fmt="classifier{stepname}",
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
        mix_data: MixtureModelAndDatasets = clus.get_tmm(
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
        self.tmm: TruncatedMixtureModel = mix_data.tmm
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
        for k in ("neighb_candidates", "neighb_candidate_counts", "neighborhoods"):
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
        assert self.motion is not None
        features = spike_data[self.pca_ds]
        if self.motion.drifting:
            if self.motion_depth_mode == "localization":
                z = spike_data[self.loc_ds][:, 2].numpy(force=True)
            elif self.motion_depth_mode == "channel":
                z = self.b.geom[channels, 1].numpy(force=True)
            else:
                panic(self.motion_depth_mode)
            # TODO motion on gpu
            shifts, n_pitches_shift = self.motion.pitch_shifts(
                times_s=spike_data["times_seconds"].numpy(force=True), depths_um=z
            )
            shift_ix = n_pitches_shift - self.min_n_pitches_shift
            assert shift_ix.min() >= 0
            assert shift_ix.max() < self.b.neighborhood_ids_by_shift.shape[1]
            shift_ix = torch.asarray(shift_ix, device=channels.device)
            features = self.erp.interp(
                features=features,
                source_main_channels=channels,
                target_channels=self.b.target_channels_by_shift[channels, shift_ix],
                source_shifts=torch.asarray(shifts, device=waveforms.device),
            )
            features = features.nan_to_num_()
        else:
            shift_ix = 0
            features = features.nan_to_num()
        neighborhood_ids = self.b.neighborhood_ids_by_shift[channels, shift_ix]

        # soft assignment
        cand_count = self.b.neighb_candidate_counts[neighborhood_ids].sum()
        candidates = self.b.neighb_candidates[neighborhood_ids]
        labels, scores = self.tmm.score_features(
            features=features,
            candidates=candidates,
            neighborhood_ids=neighborhood_ids,
            n_candidates=self.n_candidates,
            candidate_count=int(cand_count.item()),
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
