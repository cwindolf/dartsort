from dataclasses import dataclass

import numpy as np

from ..util.internal_config import ClusteringFeaturesConfig
from . import cluster_util

default_clustering_features_config = ClusteringFeaturesConfig()


def get_clustering_features(
    recording,
    sorting,
    motion_est=None,
    clustering_features_cfg: ClusteringFeaturesConfig
    | None = default_clustering_features_config,
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
            if clustering_features_cfg.register_z:
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

        if clustering_features_cfg.n_main_channel_pcs:
            pcs = cluster_util.get_main_channel_pcs(
                sorting, rank=clustering_features_cfg.n_main_channel_pcs
            )
            pcs *= clustering_features_cfg.pc_scale
            features.append(pcs)

        features = np.concatenate(features, axis=1)
        return cls(features=features, x=x, z=z, z_reg=z_reg, xyza=xyza, amplitudes=amp)
