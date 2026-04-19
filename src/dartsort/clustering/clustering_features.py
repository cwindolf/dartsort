from typing import Self, cast

import h5py
import numpy as np
import torch
from torch import Tensor

from ..util.data_util import DARTsortSorting
from ..util.drift_util import get_stable_channels
from ..util.internal_config import ClusteringFeaturesConfig, ComputationConfig
from ..util.interpolation_util import SpikeNeighborhoods, interpolate_by_chunk
from ..util.job_util import ensure_computation_config
from ..util.motion import MotionInfo
from ..util.py_util import databag
from ..util.waveform_util import single_channel_index
from . import cluster_util

default_clustering_features_cfg = ClusteringFeaturesConfig()
minimal_features_cfg = ClusteringFeaturesConfig(
    n_main_channel_pcs=0,
    use_amplitude=False,
    use_signed_amplitude=False,
    use_x=False,
    use_z=False,
)


@databag
class SimpleMatrixFeatures:
    n: int
    features: np.ndarray
    x: np.ndarray
    z: np.ndarray
    z_reg: np.ndarray
    xyza: np.ndarray
    signed_amplitudes: np.ndarray
    amplitudes: np.ndarray

    def mask(self, mask) -> Self:
        x = self.x[mask]
        return self.__class__(
            n=x.shape[0],
            features=self.features[mask],
            x=x,
            z=self.z[mask],
            z_reg=self.z_reg[mask],
            xyza=self.xyza[mask],
            signed_amplitudes=self.signed_amplitudes[mask],
            amplitudes=self.amplitudes[mask],
        )

    @classmethod
    def from_config(
        cls,
        *,
        sorting: DARTsortSorting,
        motion: MotionInfo,
        clustering_features_cfg: ClusteringFeaturesConfig,
        computation_cfg: ComputationConfig | None,
    ) -> Self:
        computation_cfg = ensure_computation_config(computation_cfg)
        xyza = getattr(sorting, clustering_features_cfg.localizations_dataset_name)
        x = xyza[:, 0]
        z = xyza[:, 2]
        t_s = sorting.times_seconds
        z_reg = motion.correct_s(t_s, z)

        features = []

        if clustering_features_cfg.use_z:
            if clustering_features_cfg.motion_aware:
                features.append(z_reg[:, None])
            else:
                features.append(z[:, None])

        if clustering_features_cfg.use_x:
            features.append(x[:, None] * clustering_features_cfg.x_scale)

        amp = getattr(sorting, clustering_features_cfg.amplitudes_dataset_name)
        if clustering_features_cfg.use_amplitude:
            assert amp is not None
            ampft = amp.copy()
            if clustering_features_cfg.log_transform_amplitude:
                ampft = np.log(clustering_features_cfg.amp_log_c + ampft)
                ampft *= clustering_features_cfg.amp_scale
            features.append(ampft[:, None])

        v = getattr(sorting, clustering_features_cfg.voltages_dataset_name, None)
        if v is None:
            samp = amp.copy()
        else:
            samp = amp * np.sign(v)

        if clustering_features_cfg.use_signed_amplitude:
            samp *= clustering_features_cfg.amp_scale
            features.append(clustering_features_cfg.amp_scale * samp[:, None])

        do_pcs = bool(clustering_features_cfg.n_main_channel_pcs)
        pcs = None
        if do_pcs and not clustering_features_cfg.motion_aware:
            pcs = cluster_util.get_main_channel_pcs(
                sorting,
                rank=clustering_features_cfg.n_main_channel_pcs,
                dataset_name=clustering_features_cfg.pca_dataset_name,
            )
        elif do_pcs and clustering_features_cfg.motion_aware:
            shifts, n_pitches_shift = motion.pitch_shifts(
                sorting=sorting,
                motion_depth_mode=clustering_features_cfg.motion_depth_mode,
            )
            mainchan_ci = single_channel_index(len(motion.geom))
            schan, *_ = get_stable_channels(
                motion=motion,
                channels=sorting.channels,
                channel_index=mainchan_ci,
                n_pitches_shift=n_pitches_shift,
                workers=computation_cfg.n_jobs_small,
            )
            mask = np.ones((1,), dtype=bool)
            mask = np.broadcast_to(mask, len(schan))
            if hasattr(sorting, clustering_features_cfg.pca_dataset_name):
                pcs = getattr(sorting, clustering_features_cfg.pca_dataset_name)
                pcs = interpolate_by_chunk(
                    mask=mask,
                    dataset=pcs,
                    geom=motion.geom,
                    channel_index=cast(h5py.Dataset, sorting.channel_index)[:],
                    channels=sorting.channels,
                    shifts=shifts,
                    registered_geom=motion.rgeom,
                    target_channels=schan,
                    params=clustering_features_cfg.interp_params,
                )
                pcs = pcs[:, : clustering_features_cfg.n_main_channel_pcs, 0]
            else:
                assert sorting.parent_h5_path is not None
                with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
                    pcs = interpolate_by_chunk(
                        mask=mask,
                        dataset=h5[clustering_features_cfg.pca_dataset_name],
                        geom=motion.geom,
                        channel_index=cast(h5py.Dataset, h5["channel_index"])[:],
                        channels=sorting.channels,
                        shifts=shifts,
                        registered_geom=motion.rgeom,
                        target_channels=schan,
                        params=clustering_features_cfg.interp_params,
                    )
                    pcs = pcs[:, : clustering_features_cfg.n_main_channel_pcs, 0]

        if do_pcs:
            assert pcs is not None
            if clustering_features_cfg.pc_transform == "log":
                pcs = signed_log1p(
                    pcs, pre_scale=clustering_features_cfg.pc_pre_transform_scale
                )
            elif clustering_features_cfg.pc_transform == "sqrt":
                pcs = signed_sqrt_transform(
                    pcs, pre_scale=clustering_features_cfg.pc_pre_transform_scale
                )
            else:
                assert clustering_features_cfg.pc_transform in ("none", None)
            pcs *= clustering_features_cfg.pc_scale
            if torch.is_tensor(pcs):
                pcs = pcs.numpy(force=True)
            features.append(pcs)

        n = x.shape[0]
        if len(features):
            features = np.concatenate(features, axis=1)
        else:
            features = np.empty((n, 0))
        return cls(
            n=n,
            features=features,
            x=x,
            z=z,
            z_reg=z_reg,
            xyza=xyza,
            amplitudes=amp,
            signed_amplitudes=samp,
        )


@databag
class StableWaveformFeatures:
    # n, nc_extract
    channels: Tensor
    # n, rank, nc_extract
    features: Tensor
    neighborhoods: SpikeNeighborhoods

    @classmethod
    def from_config(
        cls,
        *,
        sorting: DARTsortSorting,
        motion: MotionInfo,
        clustering_features_cfg: ClusteringFeaturesConfig,
        computation_cfg: ComputationConfig | None,
    ) -> Self:
        computation_cfg = ensure_computation_config(computation_cfg)
        shifts, n_pitches_shift = motion.pitch_shifts(
            sorting=sorting, motion_depth_mode=clustering_features_cfg.motion_depth_mode
        )
        res = get_stable_channels(
            motion=motion,
            channels=sorting.channels,
            channel_index=sorting.channel_index,
            n_pitches_shift=n_pitches_shift,
            core_radius=None,
            workers=computation_cfg.n_jobs_small,
            device=computation_cfg.actual_device(),
        )
        channels, neighborhoods, neighborhood_ids = res[:3]
        channels = torch.asarray(channels)
        spike_neighborhoods = SpikeNeighborhoods(
            n_channels=motion.rgeom.shape[0],
            neighborhood_ids=neighborhood_ids,
            neighborhoods=neighborhoods,
        )

        if hasattr(sorting, clustering_features_cfg.pca_dataset_name):
            features = getattr(sorting, clustering_features_cfg.pca_dataset_name)
            features = interpolate_by_chunk(
                mask=np.ones(len(sorting), dtype=np.bool_),
                dataset=features,
                geom=motion.geom,
                channel_index=sorting.channel_index,
                channels=sorting.channels,
                shifts=shifts,
                registered_geom=motion.rgeom,
                target_channels=channels,
                trim_to_rank=clustering_features_cfg.feature_rank,
                params=clustering_features_cfg.interp_params.normalize(),
                device=computation_cfg.actual_device(),
            )
        else:
            assert sorting.parent_h5_path is not None
            with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
                features = interpolate_by_chunk(
                    mask=np.ones(len(sorting), dtype=np.bool_),
                    dataset=h5[clustering_features_cfg.pca_dataset_name],
                    geom=motion.geom,
                    channel_index=sorting.channel_index,
                    channels=sorting.channels,
                    shifts=shifts,
                    registered_geom=motion.rgeom,
                    target_channels=channels,
                    trim_to_rank=clustering_features_cfg.feature_rank,
                    params=clustering_features_cfg.interp_params.normalize(),
                    device=computation_cfg.actual_device(),
                )

        return cls(
            channels=channels,
            features=torch.asarray(features),
            neighborhoods=spike_neighborhoods,
        )


# -- helpers


def signed_log1p(x, pre_scale=1.0):
    """sgn(x) * log(1+|x|*pre_scale)"""
    x = torch.asarray(x)
    xx = x.abs()
    if pre_scale != 1.0:
        xx.mul_(pre_scale)
    torch.log1p(xx, out=xx)
    xx.mul_(torch.sign(x))
    return xx


def signed_sqrt_transform(x, pre_scale=1.0):
    """sgn(x) * (sqrt(1 + |x|*pre_scale) - 1)"""
    x = torch.asarray(x)
    xx = x.abs()
    if pre_scale != 1.0:
        xx.mul_(pre_scale)
    xx.add_(1.0)
    xx.sqrt_()
    xx.sub_(1.0)
    xx.mul_(torch.sign(x))
    return xx
