from dataclasses import dataclass

import h5py
import numpy as np
import torch

from ..util.internal_config import ClusteringFeaturesConfig
from ..util import drift_util, interpolation_util
from ..util.waveform_util import single_channel_index
from . import cluster_util

default_clustering_features_config = ClusteringFeaturesConfig()


def get_clustering_features(
    recording,
    sorting,
    motion_est=None,
    clustering_features_cfg: (
        ClusteringFeaturesConfig | None
    ) = default_clustering_features_config,
):
    if clustering_features_cfg is None:
        return None
    if clustering_features_cfg.features_type == "simple_matrix":
        return SimpleMatrixFeatures.from_config(
            recording, sorting, motion_est, clustering_features_cfg
        )
    assert False


@dataclass
class SimpleMatrixFeatures:
    features: np.ndarray
    x: np.ndarray
    z: np.ndarray
    z_reg: np.ndarray
    xyza: np.ndarray
    amplitudes: np.ndarray | None

    def mask(self, mask):
        return self.__class__(
            features=self.features[mask],
            x=self.x[mask],
            z=self.z[mask],
            z_reg=self.z_reg[mask],
            xyza=self.xyza[mask],
            amplitudes=self.amplitudes[mask] if self.amplitudes is not None else None,
        )

    @classmethod
    def from_config(cls, recording, sorting, motion_est, clustering_features_cfg):
        assert clustering_features_cfg.features_type == "simple_matrix"

        xyza = getattr(sorting, clustering_features_cfg.localizations_dataset_name)
        x = xyza[:, 0]
        z_reg = z = xyza[:, 2]
        if motion_est is not None:
            z_reg = motion_est.correct_s(sorting.times_seconds, z)

        features = []

        if clustering_features_cfg.use_x:
            features.append(x[:, None])

        if clustering_features_cfg.use_z:
            if clustering_features_cfg.motion_aware:
                features.append(z_reg[:, None])
            else:
                features.append(z[:, None])

        amp = None
        if clustering_features_cfg.use_amplitude:
            amp = getattr(sorting, clustering_features_cfg.amplitudes_dataset_name)
            if clustering_features_cfg.log_transform_amplitude:
                amp = np.log(clustering_features_cfg.amp_log_c + amp)
                amp *= clustering_features_cfg.amp_scale
            features.append(amp[:, None])

        do_pcs = bool(clustering_features_cfg.n_main_channel_pcs)
        pcs = None
        if do_pcs and not clustering_features_cfg.motion_aware:
            pcs = cluster_util.get_main_channel_pcs(
                sorting,
                rank=clustering_features_cfg.n_main_channel_pcs,
                dataset_name=clustering_features_cfg.pca_dataset_name,
            )
        elif do_pcs and clustering_features_cfg.motion_aware:
            geom = sorting.geom
            if motion_est is None:
                registered_geom = geom
            else:
                registered_geom = drift_util.registered_geometry(
                    geom, motion_est=motion_est
                )
            res = drift_util.get_shift_info(sorting, motion_est, geom)
            channels, shifts, n_pitches_shift = res
            mainchan_ci = single_channel_index(len(geom))
            schan, *_ = drift_util.get_stable_channels(
                geom,
                channels,
                mainchan_ci,
                registered_geom,
                n_pitches_shift,
                workers=clustering_features_cfg.workers,
            )
            mask = np.ones((1,), dtype=bool)
            mask = np.broadcast_to(mask, len(schan))
            with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
                pcs = interpolation_util.interpolate_by_chunk(
                    mask,
                    h5[clustering_features_cfg.pca_dataset_name],
                    geom,
                    h5["channel_index"][:],
                    sorting.channels,
                    shifts,
                    registered_geom,
                    schan,
                    method=clustering_features_cfg.interpolation_method,
                    extrap_method=None,
                    kernel_name=clustering_features_cfg.kernel_name,
                    sigma=clustering_features_cfg.interpolation_sigma,
                    rq_alpha=clustering_features_cfg.rq_alpha,
                    kriging_poly_degree=clustering_features_cfg.kriging_poly_degree,
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

        features = np.concatenate(features, axis=1)
        return cls(features=features, x=x, z=z, z_reg=z_reg, xyza=xyza, amplitudes=amp)


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
