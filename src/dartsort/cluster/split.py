from dataclasses import dataclass, replace
from typing import Optional

import h5py
import numpy as np
import torch
from dartsort.util import drift_util, waveform_util
from dartsort.util.multiprocessing_util import get_pool
from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from . import cluster_util, relocate


def split_clusters(
    sorting,
    split_strategy="FeatureSplit",
    split_strategy_kwargs=None,
    recursive=False,
    split_big=False,
    split_big_kw=dict(dz=40, dx=48, min_size_split=50),
    show_progress=True,
    n_jobs=0,
    max_size_wfs=None,
):
    """Parallel split step runner function

    Arguments
    ---------
    sorting : DARTsortSorting
    split_step_class_name : str
        One of split_steps_by_class_name.keys()
    split_step_kwargs : dictionary
    recursive : bool
        If True, attempt to split newly created clusters again
    show_progress : bool
    n_jobs : int

    Returns
    -------
    split_sorting : DARTsortSorting
    """
    # initialize split state
    labels = sorting.labels.copy()
    labels_to_process = np.unique(labels)
    labels_to_process = list(labels_to_process[labels_to_process > 0])
    cur_max_label = max(labels_to_process)

    n_jobs, Executor, context = get_pool(n_jobs)
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_split_job_init,
        initargs=(split_strategy, split_strategy_kwargs),
    ) as pool:
        iterator = jobs = [
            pool.submit(_split_job, np.flatnonzero(labels == i), max_size_wfs)
            for i in labels_to_process
        ]
        if show_progress:
            iterator = tqdm(
                jobs,
                desc=split_strategy,
                total=len(labels_to_process),
                smoothing=0,
            )
        for future in iterator:
            split_result = future.result()
            if not split_result.is_split:
                continue

            # assign new labels. -1 -> -1, 0 keeps current label,
            # others start at cur_max_label + 1
            in_unit = split_result.in_unit
            new_labels = split_result.new_labels
            triaged = split_result.new_labels < 0
            labels[in_unit[triaged]] = new_labels[triaged]
            labels[in_unit[new_labels > 0]] = cur_max_label + new_labels[new_labels > 0]
            new_untriaged_labels = labels[in_unit[new_labels >= 0]]
            cur_max_label = new_untriaged_labels.max()

            # submit recursive jobs to the pool, if any
            if recursive:
                new_units = np.unique(new_untriaged_labels)
                for i in new_units:
                    jobs.append(
                        pool.submit(
                            _split_job, np.flatnonzero(labels == i), max_size_wfs
                        )
                    )
                if show_progress:
                    iterator.total += len(new_units)
            elif split_big:
                new_units = np.unique(new_untriaged_labels)
                for i in new_units:
                    idx = np.flatnonzero(new_untriaged_labels == i)
                    tall = split_result.x[idx].ptp() > split_big_kw["dx"]
                    wide = split_result.z_reg[idx].ptp() > split_big_kw["dx"]
                    if (tall or wide) and len(idx) > split_big_kw["min_size_split"]:
                        jobs.append(
                            pool.submit(
                                _split_job, np.flatnonzero(labels == i), max_size_wfs
                            )
                        )
                        if show_progress:
                            iterator.total += 1

    new_sorting = replace(sorting, labels=labels)
    return new_sorting


# -- split steps

# the purpose of using classes here is that the class manages expensive
# resources that we don't want to set up every time we split a cluster.
# this way, these objects are created once on each worker process.
# we only have one split step so it is a bit silly to have the superclass,
# but it's just to illustrate the interface that's actually used by the
# main function split_clusters, in case someone wants to write another


@dataclass
class SplitResult:
    """If not is_split, it is fine to leave other fields as None"""

    is_split: bool = False
    in_unit: Optional[np.ndarray] = None
    new_labels: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    z_reg: Optional[np.ndarray] = None

    def __post_init__(self):
        """Runs after a SplitResult is constructed, check valid"""
        if self.is_split:
            assert self.in_unit is not None
            assert self.new_labels is not None
            assert self.in_unit.ndim == 1
            assert self.in_unit.shape == self.new_labels.shape


class SplitStrategy:
    """Split steps subclass this and implement split_cluster"""

    def split_cluster(self, in_unit):
        """
        Arguments
        ---------
        in_unit : indices of spikes in a putative unit

        Returns
        -------
        split_result : a SplitResult object
        """
        raise NotImplementedError


class FeatureSplit(SplitStrategy):
    def __init__(
        self,
        peeling_hdf5_filename,
        peeling_featurization_pt=None,
        motion_est=None,
        use_localization_features=True,
        n_pca_features=2,
        relocated=True,
        localization_feature_scales=(1.0, 1.0, 50.0),
        log_c=5,
        channel_selection_radius=75.0,
        min_cluster_size=25,
        min_samples=25,
        cluster_selection_epsilon=25,
        reassign_outliers=False,
        max_size_wfs=None,
        random_state=0,
        **dataset_name_kwargs,
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
        self.motion_est = motion_est
        self.localization_feature_scales = localization_feature_scales
        self.log_c = log_c
        self.rg = np.random.default_rng(random_state)
        self.reassign_outliers = reassign_outliers

        # hdbscan parameters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

        self.max_size_wfs = max_size_wfs

        # load up the required h5 datasets
        self.initialize_from_h5(
            peeling_hdf5_filename,
            peeling_featurization_pt,
            **dataset_name_kwargs,
        )

    def split_cluster(self, in_unit_all, max_size_wfs):
        n_spikes = in_unit_all.size
        if max_size_wfs is not None and n_spikes > max_size_wfs:
            # TODO: max_size_wfs could be chosen automatically based on available memory and number of spikes
            idx_subsample = np.random.choice(n_spikes, max_size_wfs, replace=False)
            idx_subsample.sort()
            in_unit = in_unit_all[idx_subsample]
            subsampling = True
        else:
            in_unit = in_unit_all
            subsampling = False

        if n_spikes < self.min_cluster_size:
            return SplitResult()

        (
            max_registered_channel,
            n_pitches_shift,
            reloc_amplitudes,
            kept,
        ) = self.get_registered_channels(in_unit)
        if not kept.size:
            return SplitResult()

        features = []
        if self.use_localization_features:
            loc_features = self.localization_features[in_unit]
            if self.relocated:
                loc_features[kept, 2] = reloc_amplitudes
            features.append(loc_features)

        if self.n_pca_features > 0:
            enough_good_spikes, kept, pca_embeds = self.pca_features(
                in_unit, max_registered_channel, n_pitches_shift
            )
            if not enough_good_spikes:
                return SplitResult()
            # scale pc features to match localization features
            if self.use_localization_features:
                pca_embeds *= loc_features.std(axis=0).mean()
            features.append(pca_embeds)
        features = np.column_stack([f[kept] for f in features])

        clust = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            core_dist_n_jobs=1,  # let's just use our parallelism
            prediction_data=self.reassign_outliers,
        )
        hdb_labels = clust.fit_predict(features)

        is_split = np.setdiff1d(np.unique(hdb_labels), [-1]).size > 1

        if is_split and self.reassign_outliers:
            hdb_labels = cluster_util.knn_reassign_outliers(hdb_labels, features)

        new_labels = None
        if is_split:
            new_labels = np.full(n_spikes, -1)
            if not subsampling:
                new_labels[kept] = hdb_labels
            else:
                new_labels[idx_subsample[kept]] = hdb_labels

        if self.use_localization_features:
            return SplitResult(
                is_split=is_split,
                in_unit=in_unit_all,
                new_labels=new_labels,
                x=loc_features[:, 0],
                z_reg=loc_features[:, 1],
            )
        else:
            return SplitResult(
                is_split=is_split, in_unit=in_unit_all, new_labels=new_labels
            )

    def get_registered_channels(self, in_unit):
        n_pitches_shift = drift_util.get_spike_pitch_shifts(
            self.z[in_unit],
            geom=self.geom,
            registered_depths_um=self.z_reg[in_unit],
        )
        amp_vecs = batched_h5_read(self.amplitude_vectors, in_unit)
        amplitude_template = drift_util.registered_average(
            amp_vecs,
            n_pitches_shift,
            self.geom,
            self.registered_geom,
            main_channels=self.channels[in_unit],
            channel_index=self.channel_index,
        )
        max_registered_channel = amplitude_template.argmax()

        if self.relocated:
            targ_chans = self.registered_channel_index[max_registered_channel]
            targ_chans = targ_chans[targ_chans < len(self.registered_geom)]
            reloc_amp_vecs = relocate.relocated_waveforms_on_static_channels(
                amp_vecs,
                main_channels=self.channels[in_unit],
                channel_index=self.channel_index,
                xyza_from=self.xyza[in_unit],
                target_channels=targ_chans,
                z_to=self.z_reg[in_unit],
                geom=self.geom,
                registered_geom=self.registered_geom,
            )
            kept = np.flatnonzero(~np.isnan(reloc_amp_vecs).any(axis=1))
            reloc_amplitudes = np.nanmax(reloc_amp_vecs[kept], axis=1)
            reloc_amplitudes = np.log(self.log_c + reloc_amplitudes)
        else:
            reloc_amplitudes = None
            kept = np.arange(in_unit.size)

        return max_registered_channel, n_pitches_shift, reloc_amplitudes, kept

    def pca_features(
        self,
        in_unit,
        max_registered_channel,
        n_pitches_shift,
        batch_size=1_000,
        max_samples_pca=50_000,
    ):
        """Compute relocated PCA features on a drift-invariant channel set"""
        # figure out which set of channels to use
        # we use the stored amplitudes to do this rather than computing a
        # template, which can be expensive
        # max_batch_size set to avoid memory errors
        pca_channels = self.registered_channel_index[max_registered_channel]
        pca_channels = pca_channels[pca_channels < len(self.registered_geom)]

        # load waveform embeddings and invert TPCA if we are relocating
        waveforms = None  # will allocate output array when we know it's size
        for bs in range(0, in_unit.size, batch_size):
            be = min(in_unit.size, bs + batch_size)

            batch = batched_h5_read(self.tpca_features, in_unit[bs:be])
            n_batch, rank, c = batch.shape

            # invert TPCA for relocation
            if self.relocated:
                batch = batch.transpose(0, 2, 1).reshape(n_batch * c, rank)
                batch = self.tpca.inverse_transform(batch)
                t = batch.shape[1]
                batch = batch.reshape(n_batch, c, t).transpose(0, 2, 1)

            # relocate or just restrict to channel subset
            if self.relocated:
                batch = relocate.relocated_waveforms_on_static_channels(
                    batch,
                    main_channels=self.channels[in_unit][bs:be],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    xyza_from=self.xyza[in_unit][bs:be],
                    z_to=self.z_reg[in_unit][bs:be],
                    geom=self.geom,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            else:
                batch = drift_util.get_waveforms_on_static_channels(
                    batch,
                    self.geom,
                    main_channels=self.channels[in_unit][bs:be],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    n_pitches_shift=n_pitches_shift,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )

            if waveforms is None:
                waveforms = np.empty(
                    (in_unit.size, t * pca_channels.size), dtype=batch.dtype
                )  # POTENTIALLY WAY TOO BIG
            waveforms[bs:be] = batch.reshape(n_batch, -1)

        # figure out which waveforms actually overlap with the requested channels
        no_nan = np.flatnonzero(~np.isnan(waveforms).any(axis=1))
        if no_nan.size < max(self.min_cluster_size, self.n_pca_features):
            return False, no_nan, None

        # fit the per-cluster small rank PCA
        pca = PCA(
            self.n_pca_features,
            random_state=np.random.RandomState(seed=self.rg.bit_generator),
            whiten=True,
        )
        fit_indices = no_nan
        if fit_indices.size > max_samples_pca:
            fit_indices = self.rg.choice(
                fit_indices, size=max_samples_pca, replace=False
            )
        pca.fit(waveforms[fit_indices])

        # embed into the cluster's PCA space
        pca_projs = np.full(
            (waveforms.shape[0], self.n_pca_features), np.nan, dtype=waveforms.dtype
        )
        for bs in range(0, no_nan.size, batch_size):
            be = min(no_nan.size, bs + batch_size)
            pca_projs[no_nan[bs:be]] = pca.transform(waveforms[no_nan[bs:be]])

        return True, no_nan, pca_projs

    def initialize_from_h5(
        self,
        peeling_hdf5_filename,
        peeling_featurization_pt,
        tpca_features_dataset_name="collisioncleaned_tpca_features",
        localizations_dataset_name="point_source_localizations",
        amplitudes_dataset_name="denoised_amplitudes",
        amplitude_vectors_dataset_name="denoised_amplitude_vectors",
    ):
        h5 = h5py.File(peeling_hdf5_filename, "r")
        self.geom = h5["geom"][:]
        self.channel_index = h5["channel_index"][:]
        self.channels = h5["channels"][:]

        if self.use_localization_features or self.relocated:
            self.xyza = h5[localizations_dataset_name][:]
            self.t_s = h5["times_seconds"][:]
            # registered spike positions (=originals if not relocated)
            self.z_reg = self.z = self.xyza[:, 2]

            self.amplitudes = h5[amplitudes_dataset_name][:]
            self.registered_geom = self.geom
            self.registered_channel_index = self.channel_index
            if self.relocated:
                self.z_reg = self.motion_est.correct_s(self.t_s, self.z)
                self.registered_geom = drift_util.registered_geometry(
                    self.geom, self.motion_est
                )
                self.registered_channel_index = waveform_util.make_channel_index(
                    self.registered_geom, self.channel_selection_radius
                )

        if (self.use_localization_features and self.relocated) or self.n_pca_features:
            self.amplitude_vectors = h5[amplitude_vectors_dataset_name]

        if self.use_localization_features:
            self.localization_features = np.c_[
                self.xyza[:, 0], self.z_reg, np.log(self.log_c + self.amplitudes)
            ]
            self.localization_features *= self.localization_feature_scales

        if self.n_pca_features:
            # don't load these one into memory, since it's a bit heavier
            self.tpca_features = h5[tpca_features_dataset_name]
            self.match_distance = pdist(self.geom).min() / 2

        if self.n_pca_features and self.relocated:
            # load up featurization pipeline for tpca inversion
            assert peeling_featurization_pt is not None
            feature_pipeline = torch.load(peeling_featurization_pt)
            tpca_feature = [
                f
                for f in feature_pipeline.transformers
                if f.name == tpca_features_dataset_name
            ]
            assert len(tpca_feature) == 1
            self.tpca = tpca_feature[0].to_sklearn()


# this is to help split_clusters take a string argument
all_split_strategies = [FeatureSplit]
split_strategies_by_class_name = {cls.__name__: cls for cls in all_split_strategies}

# -- parallelism widgets


class SplitJobContext:
    def __init__(self, split_strategy):
        self.split_strategy = split_strategy


_split_job_context = None


def _split_job_init(split_strategy_class_name, split_strategy_kwargs):
    global _split_job_context
    split_strategy = split_strategies_by_class_name[split_strategy_class_name]
    _split_job_context = SplitJobContext(split_strategy(**split_strategy_kwargs))


def _split_job(in_unit, max_size_wfs):
    return _split_job_context.split_strategy.split_cluster(in_unit, max_size_wfs)


# -- h5 helper... slow reading...


def batched_h5_read(dataset, indices, batch_size=1000):
    if indices.size < batch_size:
        return dataset[indices]
    else:
        out = np.empty((indices.size, *dataset.shape[1:]), dtype=dataset.dtype)
        for bs in range(0, indices.size, batch_size):
            be = min(indices.size, bs + batch_size)
            out[bs:be] = dataset[indices[bs:be]]
        return out
