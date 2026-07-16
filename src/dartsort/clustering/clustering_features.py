from typing import Self, cast
from logging import WARN

import h5py
import numpy as np
import torch
from torch import Tensor

from ..util.data_util import DARTsortSorting
from ..util.drift_util import get_stable_channels
from ..util.internal_config import (
    ClusteringFeaturesConfig,
    ComputationConfig,
    default_clustering_features_cfg,
)
from ..util.interpolation_util import (
    SpikeNeighborhoods,
    StableFeaturesInterpolator,
    interpolate_by_chunk,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import handle_negative_jobs
from ..util.py_util import databag
from ..util.waveform_util import single_channel_index
from . import cluster_util

logger = get_logger(__name__)


@databag
class SimpleMatrixFeatures:
    n: int
    features: np.ndarray
    x: np.ndarray | None
    z: np.ndarray | None
    z_reg: np.ndarray | None
    xyza: np.ndarray | None
    signed_amplitudes: np.ndarray
    amplitudes: np.ndarray
    keep_mask: np.ndarray | None

    def mask(self, mask) -> Self:
        a = self.amplitudes[mask]
        if self.keep_mask is not None:
            k = self.keep_mask[mask]
        else:
            k = None
        return self.__class__(
            n=a.shape[0],
            features=self.features[mask],
            x=None if self.x is None else self.x[mask],
            z=None if self.z is None else self.z[mask],
            z_reg=None if self.z_reg is None else self.z_reg[mask],
            xyza=None if self.xyza is None else self.xyza[mask],
            signed_amplitudes=self.signed_amplitudes[mask],
            amplitudes=a,
            keep_mask=k,
        )

    @classmethod
    def from_config(
        cls,
        *,
        sorting: DARTsortSorting,
        motion: MotionInfo,
        clustering_features_cfg: ClusteringFeaturesConfig = default_clustering_features_cfg,
        computation_cfg: ComputationConfig | None = None,
    ) -> Self:
        computation_cfg = ensure_computation_config(computation_cfg)
        t_s = sorting.times_seconds
        xyza = getattr(
            sorting, clustering_features_cfg.localizations_dataset_name, None
        )
        if xyza is not None:
            x = xyza[:, 0]
            _check_numbers(
                "x", x, raise_for_numerics=clustering_features_cfg.raise_for_numerics
            )
            z = xyza[:, 2]
            _check_numbers(
                "z", z, raise_for_numerics=clustering_features_cfg.raise_for_numerics
            )
            z_reg = motion.correct_s(t_s, z)
            _check_numbers(
                "z_reg",
                z_reg,
                raise_for_numerics=clustering_features_cfg.raise_for_numerics,
            )
        else:
            x = z = z_reg = None

        features = []

        if clustering_features_cfg.use_z:
            assert z is not None
            assert z_reg is not None
            if clustering_features_cfg.motion_aware:
                features.append(z_reg[:, None])
            else:
                features.append(z[:, None])

        if clustering_features_cfg.use_x:
            assert x is not None
            features.append(x[:, None] * clustering_features_cfg.x_scale)

        amp = getattr(sorting, clustering_features_cfg.amplitudes_dataset_name)
        if clustering_features_cfg.use_amplitude:
            assert amp is not None
            _check_numbers(
                "amp",
                amp,
                raise_for_numerics=clustering_features_cfg.raise_for_numerics,
            )
            ampft = amp.copy()
            if clustering_features_cfg.log_transform_amplitude:
                ampft = np.log(clustering_features_cfg.amp_log_c + ampft)
                ampft *= clustering_features_cfg.amp_scale
            _check_numbers(
                "ampft",
                ampft,
                raise_for_numerics=clustering_features_cfg.raise_for_numerics,
            )
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
            _check_numbers(
                "No motion pcs",
                pcs,
                raise_for_numerics=clustering_features_cfg.raise_for_numerics,
            )
        elif do_pcs and clustering_features_cfg.motion_aware:
            shifts, n_pitches_shift = motion.pitch_shifts(
                sorting=sorting,
                motion_depth_mode=clustering_features_cfg.motion_depth_mode,
            )
            mainchan_ci = single_channel_index(len(motion.geom))
            _, workers = handle_negative_jobs(computation_cfg.n_jobs_small)
            schan, *_ = get_stable_channels(
                motion=motion,
                channels=sorting.channels,
                channel_index=mainchan_ci,
                n_pitches_shift=n_pitches_shift,
                workers=workers,
            )
            mask = np.ones((1,), dtype=bool)
            mask = np.broadcast_to(mask, len(schan))
            if hasattr(sorting, clustering_features_cfg.pca_dataset_name):
                pcs = getattr(sorting, clustering_features_cfg.pca_dataset_name)
                _check_numbers(
                    "sorting pcs",
                    pcs,
                    raise_for_numerics=clustering_features_cfg.raise_for_numerics,
                )
                erp, pcs = interpolate_by_chunk(
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
                _check_numbers(
                    "sorting interp pcs",
                    pcs,
                    raise_for_numerics=clustering_features_cfg.raise_for_numerics,
                )
            else:
                assert sorting.parent_h5_path is not None
                with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
                    erp, pcs = interpolate_by_chunk(
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
                _check_numbers(
                    "h5 interp pcs",
                    pcs,
                    raise_for_numerics=clustering_features_cfg.raise_for_numerics,
                )

        if do_pcs:
            assert pcs is not None
            pctf = clustering_features_cfg.pc_transform
            if pctf == "log":
                pcs = signed_log1p(
                    pcs, pre_scale=clustering_features_cfg.pc_pre_transform_scale
                )
            elif pctf == "sqrt":
                pcs = signed_sqrt_transform(
                    pcs, pre_scale=clustering_features_cfg.pc_pre_transform_scale
                )
            else:
                assert pctf in ("none", None)
            pcs *= clustering_features_cfg.pc_scale
            _check_numbers(
                f"{pctf} pcs",
                pcs,
                raise_for_numerics=clustering_features_cfg.raise_for_numerics,
            )
            if torch.is_tensor(pcs):
                pcs = pcs.numpy(force=True)
            features.append(pcs)

        n = t_s.shape[0]
        if len(features):
            features = np.concatenate(features, axis=1)
        else:
            features = np.empty((n, 0))
        keep = np.atleast_1d(np.isfinite(features).all(axis=1))
        if keep.all():
            keep = None
        return cls(
            n=n,
            features=features,
            x=x,
            z=z,
            z_reg=z_reg,
            xyza=xyza,
            amplitudes=amp,
            signed_amplitudes=samp,
            keep_mask=keep,
        )


@databag
class StableWaveformFeatures:
    # n, nc_extract
    channels: Tensor
    # n, rank, nc_extract
    features: Tensor
    neighborhoods: SpikeNeighborhoods
    erp: StableFeaturesInterpolator

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
        _, workers = handle_negative_jobs(computation_cfg.n_jobs_small)
        res = get_stable_channels(
            motion=motion,
            channels=sorting.channels,
            channel_index=sorting.channel_index,
            n_pitches_shift=n_pitches_shift,
            core_radius=None,
            workers=workers,
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
            erp, features = interpolate_by_chunk(
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
                erp, features = interpolate_by_chunk(
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

        logger.dartsortdebug(f"StableWaveformFeatures {features.shape=}.")

        return cls(
            channels=channels,
            features=torch.asarray(features),
            neighborhoods=spike_neighborhoods,
            erp=erp,
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


def _allfinite(x):
    if isinstance(x, torch.Tensor):
        return x.isfinite().all()
    else:
        return np.isfinite(x).all()


def _check_numbers(name: str, x: np.ndarray, raise_for_numerics=False):
    if isinstance(x, torch.Tensor):
        x = x.numpy(force=True)

    npinf = np.isposinf(x).sum()
    nninf = np.isneginf(x).sum()
    nna = np.isnan(x).sum()
    if not (npinf + nninf + nna):
        return

    err = f"{name}: {npinf} +inf, {nninf} -inf, {nna} nan."
    if raise_for_numerics:
        raise ValueError(err)
    else:
        logger.warn(err)
