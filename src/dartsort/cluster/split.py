from pathlib import Path

import h5py
import numpy as np
import torch
from hdbscan.hdbscan import HDBSCAN


def split_clusters(
    sorting,
    split_strategy="HDBSCANSplit",
    split_step_kwargs=None,
):
    """Parallel split step runner function

    Arguments
    ---------
    sorting : DARTsortSorting
    split_step_class_name : str
        One of split_steps_by_class_name.keys()
    split_step_kwargs : dictionary

    Returns
    -------
    split_sorting : DARTsortSorting
    """


# -- split steps

# the purpose of using classes here is that the class manages expensive
# resources that we don't want to set up every time we split a cluster.
# this way, these objects are created once on each worker process.
# we only have one split step so it is a bit silly to have the superclass,
# but it's just to illustrate the interface that's actually used by the
# main function split_clusters, in case someone wants to write another


class SplitStrategy:
    """Split steps subclass this and implement split_cluster"""

    def split_cluster(self, in_unit):
        """
        Arguments
        ---------
        in_unit : indices of spikes in a putative unit

        Returns
        -------
        is_split : bool
        labels : array of labels, like in_unit
        """
        raise NotImplementedError


class HDBSCANSplit(SplitStrategy):
    def __init__(
        self,
        peeling_hdf5_filename,
        peeling_featurization_pt=None,
        motion_est=None,
        use_localization_features=True,
        n_pca_features=2,
        relocated=True,
        localization_feature_scales=(1.0, 1.0, 50.0),
        channel_selection_radius=75.0,
        min_cluster_size=25,
        min_samples=25,
        cluster_selection_epsilon=25,
        tpca_features_dataset_name="collisioncleaned_tpca_features",
        localizations_dataset_name="point_source_localizations",
        amplitudes_dataset_name="denoised_amplitudes",
    ):
        """Split clusters based on per-cluster PCA and localization features

        Fits PCA to relocated waveforms within a cluster, combines these with
        localization features (x, registered z, and max amplitude), and runs
        HDBSCAN to refine the labeling.

        Arguments
        ---------
        peeling_hdf5_filename : str or Path
            Path to the HDF5 file output by a peeling step
        peeling_featurization_pt : str or Path
            Path to the .pt file where a featurization WaveformPipeline was
            saved during peeling. Required if relocated=True.
        motion_est : dredge.motion_util.MotionEstimate, optional
            If supplied, registered depth positions will be used for clustering
            and relocation
        use_localization_features : bool
            If false, only PCA features are used for clustering
        n_pca_features : int
            If 0, no PCA is used
        relocated : bool
            Whether to relocate waveforms according to the motion estimate.
            Only used if motion_est is not None
        localization_feature_scales : 3-tuple of float
            Scales to apply to x, registered z, and log-scaled amplitude during
            clustering to approximately sphere the clusters
        channel_selection_radius : float
            Spatial radius around the main channel used when selecting channels
            used when extracting local PCA features
        min_cluster_size, min_samples, cluster_selection_epsilon
            HDBSCAN parameters. See their documentation for lots of info.
        """
        # method parameters
        self.channel_selection_radius = channel_selection_radius
        self.use_localization_features = use_localization_features
        self.n_pca_features = n_pca_features
        self.relocated = relocated and motion_est is not None

        # hdbscan parameters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

        # load up the required h5 datasets
        h5 = h5py.File(peeling_hdf5_filename, "r")
        self.geom = h5["geom"][:]
        self.channel_index = h5["channel_index"][:]
        if self.use_localization_features or self.relocated:
            self.xyza = h5[localizations_dataset_name][:]
            self.amplitudes = h5[amplitudes_dataset_name][:]
            t_s = h5["times_seconds"][:]
            # registered spike positions (=originals if not relocated)
            self.z_reg = self.xyza[:, 2]
            if self.relocated:
                self.z_reg = motion_est.correct_s(t_s, self.z_reg)
        if self.use_localization_features:
            self.localization_features = np.c[
                self.xyza[:, 0], self.z_reg, self.amplitudes
            ]
            self.localization_features *= localization_feature_scales
        if self.n_pca_features > 0:
            # don't load this one into memory, since it's a bit heavier
            self.tpca_features = h5[tpca_features_dataset_name]
        if self.n_pca_features > 0 and self.relocated:
            # load up featurization pipeline for tpca inversion
            assert peeling_featurization_pt is not None
            feature_pipeline = torch.load(peeling_featurization_pt)
            tpca_feature = [
                f
                for f in feature_pipeline.transformers
                if f.name == tpca_features_dataset_name
            ]
            assert len(tpca_feature) == 1
            self.tpca = tpca_feature.to_sklearn()


# this is to help split_clusters take a string argument
all_split_strategies = [HDBSCANSplit]
split_strategies_by_class_name = {
    cls.__name__: cls for cls in all_split_strategies
}


# -- parallelism widgets


def _split_job_init():
    pass


def _split_job():
    pass
