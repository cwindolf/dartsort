from typing import Self, cast

import h5py
import numba
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
from ..util.spiketorch import svd_lowrank_helper
from ..util.waveform_util import make_channel_index, single_channel_index
from . import cluster_util

logger = get_logger(__name__)


@databag
class SimpleMatrixFeatures:
    n: int
    features: np.ndarray
    x: np.ndarray | None = None
    z: np.ndarray | None = None
    z_reg: np.ndarray | None = None
    xyza: np.ndarray | None = None
    signed_amplitudes: np.ndarray | None = None
    amplitudes: np.ndarray | None = None
    keep_mask: np.ndarray | None = None

    def mask(self, mask: np.ndarray) -> Self:
        if mask.dtype.kind == "b":
            mask = np.flatnonzero(mask)
        if self.keep_mask is not None:
            k = self.keep_mask[mask]
        else:
            k = None
        return self.__class__(
            n=mask.shape[0],
            features=self.features[mask],
            x=None if self.x is None else self.x[mask],
            z=None if self.z is None else self.z[mask],
            z_reg=None if self.z_reg is None else self.z_reg[mask],
            xyza=None if self.xyza is None else self.xyza[mask],
            signed_amplitudes=None
            if self.signed_amplitudes is None
            else self.signed_amplitudes[mask],
            amplitudes=None if self.amplitudes is None else self.amplitudes[mask],
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
        if clustering_features_cfg.skip:
            return cls(n=len(sorting), features=np.zeros((len(sorting), 0)))

        computation_cfg = ensure_computation_config(computation_cfg)
        t_s = sorting.times_seconds
        raise_on_na = clustering_features_cfg.raise_for_numerics
        if sorting.has_dataset(clustering_features_cfg.localizations_dataset_name):
            x = sorting.load_dataset(
                clustering_features_cfg.localizations_dataset_name,
                sl=(slice(None), 0),
            )
            check_numbers("x", x, raise_for_numerics=raise_on_na)
            z = sorting.load_dataset(
                clustering_features_cfg.localizations_dataset_name,
                sl=(slice(None), 2),
            )
            check_numbers("z", z, raise_for_numerics=raise_on_na)
            z_reg = motion.correct_s(t_s, z)
            check_numbers("z_reg", z_reg, raise_for_numerics=raise_on_na)
            if clustering_features_cfg.need_xyza:
                xyza = getattr(
                    sorting, clustering_features_cfg.localizations_dataset_name
                )
            else:
                xyza = None
        else:
            x = z = z_reg = xyza = None

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
        v = getattr(sorting, clustering_features_cfg.voltages_dataset_name, None)
        if (
            clustering_features_cfg.use_amplitude
            or clustering_features_cfg.use_signed_amplitude
        ):
            assert amp is not None
            check_numbers("amp", amp, raise_for_numerics=raise_on_na)
            if clustering_features_cfg.log_transform_amplitude:
                ampft = cast(np.ndarray, clustering_features_cfg.amp_log_c + amp)
                np.log(ampft, out=ampft)
            else:
                ampft = amp.copy()

            if clustering_features_cfg.use_signed_amplitude:
                if v is not None:
                    ampft *= np.sign(v)

            ampft *= clustering_features_cfg.amp_scale
            check_numbers("ampft", ampft, raise_for_numerics=raise_on_na)
            features.append(ampft[:, None])

        if clustering_features_cfg.n_main_channel_pcs:
            pcs = _compute_main_channel_pcs(
                sorting=sorting,
                motion=motion,
                clustering_features_cfg=clustering_features_cfg,
                computation_cfg=computation_cfg,
                raise_on_na=raise_on_na,
            )
            assert pcs is not None
            features.append(
                _transform_pcs(
                    pcs,
                    clustering_features_cfg=clustering_features_cfg,
                    raise_on_na=raise_on_na,
                )
            )

        if clustering_features_cfg.n_multi_channel_pcs:
            pcs = _compute_multi_channel_pcs(
                sorting=sorting,
                motion=motion,
                clustering_features_cfg=clustering_features_cfg,
                computation_cfg=computation_cfg,
                raise_on_na=raise_on_na,
            )
            assert pcs is not None
            features.append(
                _transform_pcs(
                    pcs,
                    clustering_features_cfg=clustering_features_cfg,
                    raise_on_na=raise_on_na,
                    name="multi-channel pcs",
                )
            )

        n = t_s.shape[0]
        if len(features):
            features = np.concatenate(features, axis=1)
        else:
            features = np.empty((n, 0))
        keep = np.atleast_1d(np.isfinite(features).all(axis=1))
        if keep.all():
            keep = None
        if v is not None:
            samp = amp * v
        else:
            samp = amp
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


def _compute_main_channel_pcs(
    *,
    sorting: DARTsortSorting,
    clustering_features_cfg: ClusteringFeaturesConfig,
    motion: MotionInfo,
    computation_cfg: ComputationConfig,
    raise_on_na: bool,
):
    do_pcs = bool(clustering_features_cfg.n_main_channel_pcs)
    if not do_pcs:
        return None

    if not clustering_features_cfg.motion_aware:
        pcs = cluster_util.get_main_channel_pcs(
            sorting,
            rank=clustering_features_cfg.n_main_channel_pcs,
            dataset_name=clustering_features_cfg.pca_dataset_name,
        )
        check_numbers("No motion pcs", pcs, raise_for_numerics=raise_on_na)
    elif clustering_features_cfg.motion_aware:
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
        assert sorting.parent_h5_path is not None
        with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
            _erp, pcs = interpolate_by_chunk(
                mask=None,
                dataset=h5[clustering_features_cfg.pca_dataset_name],
                geom=motion.geom,
                channel_index=cast(h5py.Dataset, h5["channel_index"])[:],
                channels=sorting.channels,
                shifts=shifts,
                registered_geom=motion.rgeom,
                target_channels=schan,
                params=clustering_features_cfg.interp_params,
                trim_to_rank=clustering_features_cfg.n_main_channel_pcs,
                show_progress=False,
            )
            assert pcs.shape[2] == 1  # just one channel here
            pcs = pcs[:, : clustering_features_cfg.n_main_channel_pcs, 0]
        check_numbers("h5 interp pcs", pcs, raise_for_numerics=raise_on_na)
    else:
        pcs = None
    return pcs


def _compute_multi_channel_pcs(
    *,
    sorting: DARTsortSorting,
    clustering_features_cfg: ClusteringFeaturesConfig,
    motion: MotionInfo,
    computation_cfg: ComputationConfig,
    raise_on_na: bool,
):
    n_pcs = clustering_features_cfg.n_multi_channel_pcs
    if not n_pcs:
        return None

    feats, neighborhoods, neighborhood_ids, n_target_channels = _multi_channel_features(
        sorting=sorting,
        clustering_features_cfg=clustering_features_cfg,
        motion=motion,
        computation_cfg=computation_cfg,
    )
    pcs = _multi_channel_embedding(
        feats=feats,
        neighborhoods=neighborhoods,
        neighborhood_ids=neighborhood_ids,
        n_target_channels=n_target_channels,
        n_pcs=n_pcs,
    )
    check_numbers("multi channel pcs", pcs, raise_for_numerics=raise_on_na)
    return pcs


def _multi_channel_features(
    *,
    sorting: DARTsortSorting,
    clustering_features_cfg: ClusteringFeaturesConfig,
    motion: MotionInfo,
    computation_cfg: ComputationConfig,
) -> tuple[Tensor, np.ndarray, np.ndarray, int]:
    geom = motion.geom
    multi_ci = make_channel_index(geom, clustering_features_cfg.multi_channel_pc_radius)
    n_spikes = len(sorting)

    if clustering_features_cfg.motion_aware:
        shifts, n_pitches_shift = motion.pitch_shifts(
            sorting=sorting,
            motion_depth_mode=clustering_features_cfg.motion_depth_mode,
        )
        target_geom = motion.rgeom
        _, workers = handle_negative_jobs(computation_cfg.n_jobs_small)
        target_channels, neighborhoods, neighborhood_ids, *_ = get_stable_channels(
            motion=motion,
            channels=sorting.channels,
            channel_index=multi_ci,
            n_pitches_shift=n_pitches_shift,
            workers=workers,
        )
    else:
        shifts = np.zeros(n_spikes, dtype=np.float32)
        target_geom = geom
        neighborhoods, chan_to_neighborhood_id = np.unique(
            multi_ci, axis=0, return_inverse=True
        )
        neighborhood_ids = chan_to_neighborhood_id[sorting.channels]
        target_channels = neighborhoods[neighborhood_ids]

    assert sorting.parent_h5_path is not None
    with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
        _erp, feats = interpolate_by_chunk(
            mask=None,
            dataset=h5[clustering_features_cfg.pca_dataset_name],
            geom=geom,
            channel_index=cast(h5py.Dataset, h5["channel_index"])[:],
            channels=sorting.channels,
            shifts=shifts,
            registered_geom=target_geom,
            target_channels=target_channels,
            params=clustering_features_cfg.interp_params,
            trim_to_rank=clustering_features_cfg.feature_rank,
            show_progress=False,
        )

    feats = torch.asarray(feats)
    return feats, neighborhoods, neighborhood_ids, len(target_geom)


def _multi_channel_embedding(
    *,
    feats: Tensor,
    neighborhoods: np.ndarray,
    neighborhood_ids: np.ndarray | Tensor,
    n_target_channels: int,
    n_pcs: int,
) -> Tensor:
    n_spikes, rank, n_chans = feats.shape
    dim = rank * n_chans
    assert dim >= n_pcs, f"{n_pcs=} {dim=}."
    feats_flat = feats.view(n_spikes, dim)

    neighborhood_ids = torch.asarray(neighborhood_ids)
    obs_mask = torch.asarray(neighborhoods) < n_target_channels
    neighb_is_full = obs_mask.all(dim=1)
    spike_is_full = neighb_is_full[neighborhood_ids]

    if spike_is_full.all():
        fit_feats = feats_flat
    elif neighb_is_full.any():
        fit_feats = feats_flat[spike_is_full]
    else:
        logger.warning("No full neighborhoods??")
        fit_feats = torch.nan_to_num(feats_flat)
    logger.dartsortdebug(
        f"mcPCA: rank {n_pcs} fit to {fit_feats.shape=} ({feats.shape=})."
    )
    _, components, *_ = svd_lowrank_helper(fit_feats, rank=n_pcs)
    del fit_feats

    pcs = feats_flat @ components.T
    basis = components.T.reshape(rank, n_chans, n_pcs).double()
    order = torch.argsort(neighborhood_ids)
    starts = torch.zeros(len(neighborhoods) + 1, dtype=torch.long)
    torch.cumsum(
        torch.bincount(neighborhood_ids, minlength=len(neighborhoods)),
        dim=0,
        out=starts[1:],
    )
    for j in neighb_is_full.logical_not().nonzero(as_tuple=True)[0]:
        in_neighb = order[starts[j] : starts[j + 1]]
        if not in_neighb.numel():
            continue
        obs = obs_mask[j]
        # (rank * n_obs, n_pcs)
        w = basis[:, obs].reshape(-1, n_pcs)
        # (n_in_neighb, rank * n_obs)
        x = feats[in_neighb][:, :, obs].reshape(in_neighb.numel(), -1).double()
        # inv(w'w) w' x' for each spike
        pcs[in_neighb] = torch.linalg.solve(w.T @ w, (x @ w).T).T.to(pcs)

    return pcs


def _transform_pcs(
    pcs,
    *,
    clustering_features_cfg: ClusteringFeaturesConfig,
    raise_on_na: bool,
    name: str = "pcs",
) -> np.ndarray:
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
    check_numbers(f"{pctf} {name}", pcs, raise_for_numerics=raise_on_na)
    if torch.is_tensor(pcs):
        pcs = pcs.numpy(force=True)
    return pcs


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


def check_numbers(name: str, x: np.ndarray, raise_for_numerics=False):
    if isinstance(x, torch.Tensor):
        x = x.numpy(force=True)
    if x.ndim > 1:
        x = x.ravel()

    npinf, nninf, nna = count_non_finite(x)

    if not (npinf + nninf + nna):
        return

    err = f"{name}: {npinf} +inf, {nninf} -inf, {nna} nan."
    if raise_for_numerics:
        raise ValueError(err)
    else:
        logger.warning(err)


@numba.njit(nogil=True, parallel=True)
def count_non_finite(x: np.ndarray):
    npinf = 0
    nninf = 0
    nna = 0
    for i in numba.prange(x.shape[0]):  # ty: ignore[not-iterable]
        xi = x[i]
        if np.isposinf(xi):
            npinf += 1
        if np.isneginf(xi):
            nninf += 1
        if np.isnan(xi):
            nna += 1
    return npinf, nninf, nna
