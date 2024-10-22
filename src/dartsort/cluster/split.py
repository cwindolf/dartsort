from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from dartsort.util import drift_util, waveform_util
from dartsort.util.data_util import DARTsortSorting, batched_h5_read
from dartsort.util.multiprocessing_util import get_pool
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import nan_euclidean_distances
from tqdm.auto import tqdm

from . import cluster_util, density, relocate
from .forward_backward import forward_backward


def split_clusters(
    sorting,
    split_strategy="FeatureSplit",
    split_strategy_kwargs=None,
    recursive=False,
    split_big=False,
    split_big_kw=dict(dz=40, dx=48, min_size_split=50),
    motion_est=None,
    show_progress=True,
    depth_order=True,
    n_jobs=0,
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

    if split_strategy_kwargs is None:
        split_strategy_kwargs = {}
    if motion_est is not None:
        split_strategy_kwargs["motion_est"] = motion_est

    n_jobs, Executor, context = get_pool(n_jobs)
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_split_job_init,
        initargs=(
            split_strategy,
            sorting.parent_h5_path,
            split_strategy_kwargs,
        ),
    ) as pool:
        iterator = jobs = [
            pool.submit(_split_job, np.flatnonzero(labels == i))
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
                    jobs.append(pool.submit(_split_job, np.flatnonzero(labels == i)))
                if show_progress:
                    iterator.total += len(new_units)
            elif split_big:
                new_units = np.unique(new_untriaged_labels)
                for i in new_units:
                    idx = np.flatnonzero(new_untriaged_labels == i)
                    tall = np.ptp(split_result.x[idx]) > split_big_kw["dx"]
                    wide = np.ptp(split_result.z_reg[idx]) > split_big_kw["dx"]
                    if (tall or wide) and len(idx) > split_big_kw["min_size_split"]:
                        jobs.append(
                            pool.submit(_split_job, np.flatnonzero(labels == i))
                        )
                        if show_progress:
                            iterator.total += 1

    new_sorting = replace(sorting, labels=labels)
    if depth_order:
        new_sorting = cluster_util.reorder_by_depth(new_sorting, motion_est=motion_est)
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
        recording=None,
        chunk_size_s=300,
        peeling_featurization_pt=None,
        motion_est=None,
        use_localization_features=True,
        return_localization_features=False,
        n_pca_features=2,
        pca_imputation=None,
        pca_reassign=False,
        relocated=True,
        use_wfs_L2_norm=False,
        localization_feature_scales=(1.0, 1.0, 50.0),
        time_scale=3e-3,
        use_time_feature=False,
        log_c=5,
        channel_selection_radius=75.0,
        min_cluster_size=25,
        min_samples=25,
        cluster_selection_epsilon=1,
        sigma_local=5,
        sigma_local_low=None,
        sigma_regional=None,
        noise_density=0.1,
        n_neighbors_search=20,
        radius_search=5.0,
        triage_quantile_per_cluster=0.2,
        remove_clusters_smaller_than=25,
        knn_reassign_outliers=False,
        random_state=0,
        rescale_all_features=False,
        use_ptp=True,
        amplitude_normalized=False,
        use_spread=False,
        max_spikes=None,
        cluster_alg="hdbscan",
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
        self.pca_imputation = pca_imputation
        self.pca_reassign = pca_reassign
        self.relocated = relocated and motion_est is not None
        self.motion_est = motion_est
        self.localization_feature_scales = localization_feature_scales
        self.log_c = log_c
        self.rg = np.random.default_rng(random_state)
        self.knn_reassign_outliers = knn_reassign_outliers
        self.max_spikes = max_spikes
        self.use_wfs_L2_norm = use_wfs_L2_norm
        self.rescale_all_features = rescale_all_features
        self.use_ptp = use_ptp
        self.amplitude_normalized = amplitude_normalized
        self.use_spread = use_spread
        self.use_time_feature = use_time_feature
        self.time_scale = time_scale
        self.return_localization_features = return_localization_features

        # hdbscan parameters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

        # DPC parameters
        self.sigma_local = sigma_local
        self.sigma_local_low = sigma_local_low
        self.sigma_regional = sigma_regional
        self.noise_density = noise_density
        self.n_neighbors_search = n_neighbors_search
        self.radius_search = radius_search
        self.triage_quantile_per_cluster = triage_quantile_per_cluster
        self.remove_clusters_smaller_than = remove_clusters_smaller_than

        assert cluster_alg in ("hdbscan", "dpc")
        self.cluster_alg = cluster_alg

        # load up the required h5 datasets
        self.initialize_from_h5(
            peeling_hdf5_filename,
            peeling_featurization_pt,
            **dataset_name_kwargs,
        )

    def split_cluster(self, in_unit_all):
        n_spikes = in_unit_all.size
        subsampling = self.max_spikes and n_spikes > self.max_spikes
        if subsampling:
            # TODO: max_spikes could be chosen automatically
            # based on available memory and number of spikes
            idx_subsample = self.rg.choice(n_spikes, self.max_spikes, replace=False)
            idx_subsample.sort()
            in_unit = in_unit_all[idx_subsample]
        else:
            in_unit = in_unit_all

        if n_spikes < self.min_cluster_size:
            return SplitResult()

        do_pca = self.n_pca_features > 0

        if do_pca or self.relocated or self.use_wfs_L2_norm or self.pca_reassign:
            (
                max_registered_channel,
                n_pitches_shift,
                reloc_amplitudes,
                kept,
            ) = self.get_registered_channels(in_unit)
            if not kept.size:
                return SplitResult()
        else:
            kept = np.arange(in_unit.shape[0])

        features = []

        if self.use_localization_features:
            loc_features = self.localization_features[in_unit]
            if self.relocated:
                loc_features[kept, 2] = self.localization_feature_scales[2] * np.log(
                    self.log_c + reloc_amplitudes
                )
            if self.rescale_all_features:
                mad0 = mad_sigma(loc_features[kept, 0])
                loc_features[kept] *= mad0 / mad_sigma(loc_features[kept])
            if not self.use_ptp:
                loc_features = loc_features[:, :2]
            features.append(loc_features)

        if self.use_time_feature:
            features.append(self.t_s[in_unit] * self.time_scale)

        if do_pca:
            enough_good_spikes, pca_kept, pca_embeds = self.pca_features(
                in_unit[kept],
                max_registered_channel,
                n_pitches_shift,
                amplitude_normalized=self.amplitude_normalized,
            )
            do_pca = enough_good_spikes

        if do_pca:
            pca_embeds_ = np.full(
                (in_unit.size, pca_embeds.shape[1]),
                np.nan,
                dtype=pca_embeds.dtype,
            )
            pca_embeds_[kept[pca_kept]] = pca_embeds
            pca_embeds = pca_embeds_
            kept = kept[pca_kept]
            # scale pc features to match localization features
            if self.rescale_all_features:
                mad0 = mad_sigma(loc_features[kept, 0])
                for k in range(self.n_pca_features):
                    pca_embeds[:, k] *= mad0 / mad_sigma(pca_embeds[:, k])
            elif self.use_localization_features:
                pca_embeds *= mad_sigma(loc_features[kept]).mean()
            features.append(pca_embeds)

        l2_norm = None
        if self.use_wfs_L2_norm:
            enough_good_spikes, wf2_norm_kept, l2_norm = self.waveforms_L2norm(
                in_unit,
                max_registered_channel,
                n_pitches_shift,
            )
            kept = kept[wf2_norm_kept]

        if self.use_spread:
            spread = self.spread_feature(in_unit)
            if self.rescale_all_features:
                mad0 = mad_sigma(loc_features[kept, 0])
                spread *= mad0 / mad_sigma(spread[kept])
            elif self.use_localization_features:
                spread *= mad_sigma(loc_features[kept]).mean()
            features.append(spread)

        if not features:
            return SplitResult()
        features = np.column_stack([f[kept] for f in features])

        if self.cluster_alg == "hdbscan" and features.shape[0] > self.min_cluster_size:
            clust = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                core_dist_n_jobs=1,  # let's just use our parallelism
                prediction_data=self.knn_reassign_outliers,
            )
            clust_labels = clust.fit_predict(features)
            new_ids = np.unique(clust_labels)
            new_ids = new_ids[new_ids >= 0]
            is_split = new_ids.size > 1
        elif (
            self.cluster_alg == "dpc"
            and features.shape[0] > self.remove_clusters_smaller_than
        ):
            clust_labels = density.density_peaks_clustering(
                features,
                l2_norm=l2_norm,
                sigma_local=self.sigma_local,
                sigma_local_low=self.sigma_local_low,
                sigma_regional=self.sigma_regional,
                noise_density=self.noise_density,
                n_neighbors_search=self.n_neighbors_search,
                radius_search=self.radius_search,
                triage_quantile_per_cluster=self.triage_quantile_per_cluster,
                remove_clusters_smaller_than=self.remove_clusters_smaller_than,
            )
            new_ids = np.unique(clust_labels)
            new_ids = new_ids[new_ids >= 0]
            is_split = new_ids.size > 1
        else:
            is_split = False

        if is_split and self.knn_reassign_outliers:
            clust_labels = cluster_util.knn_reassign_outliers(clust_labels, features)

        new_labels = None
        if is_split:
            new_labels = np.full(n_spikes, -1)
            if subsampling:
                new_labels[idx_subsample[kept]] = clust_labels
            else:
                new_labels[kept] = clust_labels
        
        if is_split and self.pca_reassign:
            if subsampling:
                new_labels[idx_subsample] = self.tpca_reassignment(
                    new_labels[idx_subsample], in_unit, n_pitches_shift
                )
            else:
                new_labels = self.tpca_reassignment(
                    new_labels, in_unit, n_pitches_shift
                )
        
        if self.return_localization_features and self.use_localization_features:
            return SplitResult(
                is_split=is_split,
                in_unit=in_unit_all,
                new_labels=new_labels,
                x=loc_features[:, 0],
                z_reg=loc_features[:, 1],
            )
        return SplitResult(
            is_split=is_split, in_unit=in_unit_all, new_labels=new_labels
        )

    def get_registered_channels(self, in_unit, n_samples=1000):
        n_pitches_shift = drift_util.get_spike_pitch_shifts(
            self.z[in_unit],
            geom=self.geom,
            registered_depths_um=self.z_reg[in_unit],
        )

        amp_samples = slice(None)
        if in_unit.size > n_samples:
            amp_samples = self.rg.choice(in_unit.size, size=n_samples, replace=False)
            amp_samples.sort()
        amp_vecs = batched_h5_read(self.amplitude_vectors, in_unit)
        amplitude_template = drift_util.registered_average(
            amp_vecs[amp_samples],
            n_pitches_shift[amp_samples],
            self.geom,
            self.registered_geom,
            main_channels=self.channels[in_unit[amp_samples]],
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
                match_distance=self.match_distance,
            )
            kept = np.flatnonzero(~np.isnan(reloc_amp_vecs).all(axis=1))
            reloc_amplitudes = np.nanmax(reloc_amp_vecs[kept], axis=1)
        else:
            reloc_amplitudes = None
            kept = np.arange(in_unit.size)

        return max_registered_channel, n_pitches_shift, reloc_amplitudes, kept

    def spread_feature(
        self,
        in_unit,
        max_value_dist=70,
    ):
        spread = np.zeros(in_unit.shape[0])
        amp_vecs = batched_h5_read(self.amplitude_vectors, in_unit)
        main_channels = self.channels[in_unit]

        channel_distances_index = np.sqrt(
            (
                (
                    np.pad(
                        self.geom,
                        [[0, 1], [0, 0]],
                        mode="constant",
                        constant_values=np.nan,
                    )[self.channel_index]
                    - self.geom[:, None]
                )
                ** 2
            ).sum(2)
        )

        for k in range(in_unit.shape[0]):
            max_chan = main_channels[k]
            channels_effective = np.flatnonzero(self.channel_index[max_chan] < 384)
            channels_effective = channels_effective[
                channel_distances_index[max_chan][channels_effective] < max_value_dist
            ]
            spread[k] = np.nansum(
                amp_vecs[k, channels_effective]
                * channel_distances_index[max_chan][channels_effective]
            ) / np.nansum(amp_vecs[k, channels_effective])

        return spread

    def waveforms_L2norm(
        self,
        in_unit,
        max_registered_channel,
        n_pitches_shift,
        batch_size=1_000,
    ):
        """Compute relocated waveforms on a drift-invariant channel set, for computing L2 distance"""
        # figure out which set of channels to use
        # we use the stored amplitudes to do this rather than computing a
        # template, which can be expensive
        # max_batch_size set to avoid memory errors
        pca_channels = self.registered_channel_index[max_registered_channel]
        pca_channels = pca_channels[pca_channels < len(self.registered_geom)]

        # load waveform embeddings and invert TPCA if we are relocating
        waveforms = None  # will allocate output array when we know its size
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
                    main_channels=self.channels[in_unit[bs:be]],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    xyza_from=self.xyza[in_unit[bs:be]],
                    z_to=self.z_reg[in_unit[bs:be]],
                    geom=self.geom,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            else:
                batch = drift_util.get_waveforms_on_static_channels(
                    batch,
                    self.geom,
                    main_channels=self.channels[in_unit[bs:be]],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    n_pitches_shift=n_pitches_shift[bs:be],
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
                t = batch.shape[1]

            if waveforms is None:
                waveforms = np.empty(
                    (in_unit.size, t * pca_channels.size), dtype=batch.dtype
                )
            waveforms[bs:be] = batch.reshape(n_batch, -1)

        # remove channels which are entirely nan
        not_entirely_nan_channels = np.flatnonzero(np.isfinite(waveforms).any(axis=0))
        if not_entirely_nan_channels.size < waveforms.shape[1]:
            waveforms = waveforms[:, not_entirely_nan_channels]

        # figure out which waveforms overlap completely with the remaining channels
        no_nan = np.flatnonzero(np.isfinite(waveforms).all(axis=1))
        if no_nan.size < max(self.min_cluster_size, self.n_pca_features):
            return False, no_nan, None

        else:
            size_wf = waveforms.shape[1]
            waveforms_l2_norm = cdist(waveforms[no_nan], waveforms[no_nan]) / size_wf
        return True, no_nan, waveforms_l2_norm

    def pca_features(
        self,
        in_unit,
        max_registered_channel,
        n_pitches_shift,
        batch_size=128,
        max_samples_pca=5_000,
        amplitude_normalized=False,
    ):
        """Compute relocated PCA features on a drift-invariant channel set

        Returns
        -------
        enough_good_spikes : bool
        kept : integer index array, size < in_unit.size, indices into in_unit
        pca_embeds : array of shape (kept.size, rank)

        """
        # figure out which set of channels to use
        # we use the stored amplitudes to do this rather than computing a
        # template, which can be expensive
        # max_batch_size set to avoid memory errors
        pca_channels = self.registered_channel_index[max_registered_channel]
        pca_channels = pca_channels[pca_channels < len(self.registered_geom)]

        # fast path for small data case
        if in_unit.size < max_samples_pca:
            waveforms = batched_h5_read(self.tpca_features, in_unit)
            n, rank, c = waveforms.shape

            # invert TPCA for relocation
            if self.relocated:
                waveforms = waveforms.transpose(0, 2, 1).reshape(n * c, rank)
                waveforms = self.tpca.inverse_transform(waveforms)
                t = waveforms.shape[1]
                waveforms = waveforms.reshape(n, c, t).transpose(0, 2, 1)

            # relocate or just restrict to channel subset
            if self.relocated:
                waveforms = relocate.relocated_waveforms_on_static_channels(
                    waveforms,
                    main_channels=self.channels[in_unit],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    xyza_from=self.xyza[in_unit],
                    z_to=self.z_reg[in_unit],
                    geom=self.geom,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            else:
                waveforms = drift_util.get_waveforms_on_static_channels(
                    waveforms,
                    self.geom,
                    main_channels=self.channels[in_unit],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    n_pitches_shift=n_pitches_shift,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
                t = waveforms.shape[1]

            # remove channels which are entirely nan
            not_entirely_nan_channels = np.flatnonzero(
                np.isfinite(waveforms[:, 0, :]).any(axis=0)
            )
            if not not_entirely_nan_channels.size:
                return False, None, None
            if not_entirely_nan_channels.size < waveforms.shape[2]:
                waveforms = waveforms[:, :, not_entirely_nan_channels]

            # figure out which waveforms overlap completely with the remaining channels
            no_nan = np.flatnonzero(np.isfinite(waveforms[:, 0, :]).all(axis=1))
            if no_nan.size < max(self.min_cluster_size, self.n_pca_features):
                return False, no_nan, None

            # fit the per-cluster small rank PCA
            pca = PCA(
                self.n_pca_features,
                random_state=np.random.RandomState(seed=self.rg.bit_generator),
                whiten=True,
            )
            if self.pca_imputation is None:
                waveforms = waveforms[no_nan]
                waveforms = waveforms.reshape(waveforms.shape[0], -1)
                if self.amplitude_normalized:
                    waveforms /= self.amplitudes[in_unit[no_nan]][:, None]
                pca_projs = pca.fit_transform(waveforms)
                return True, no_nan, pca_projs
            elif self.pca_imputation in ("mean", "iterative"):
                wfs = waveforms.reshape(waveforms.shape[0], -1)
                if self.amplitude_normalized:
                    wfs /= self.amplitudes[in_unit][:, None]
                pca.fit(wfs[no_nan])
                if self.pca_imputation == "iterative":
                    wfs = np.where(np.isfinite(wfs), wfs, pca.mean_[None])
                    isnan = np.setdiff1d(np.arange(wfs.shape[0]), no_nan)
                    for i in range(10):
                        emb = pca.fit_transform(wfs)
                        wfs[isnan] = pca.inverse_transform(emb[isnan])
                    pca_projs = pca.transform(wfs)
                else:
                    pca_projs = pca.transform(
                        np.where(np.isfinite(wfs), wfs, pca.mean_[None])
                    )
                return True, slice(None), pca_projs
            else:
                assert False

        # otherwise, we have bigger data. in that case, first fit the PCA
        fit_choices = self.rg.choice(in_unit.size, size=max_samples_pca, replace=False)
        fit_choices.sort()
        fit_waveforms = batched_h5_read(self.tpca_features, in_unit[fit_choices])
        n, rank, c = fit_waveforms.shape

        # invert TPCA for relocation
        if self.relocated:
            fit_waveforms = fit_waveforms.transpose(0, 2, 1).reshape(n * c, rank)
            fit_waveforms = self.tpca.inverse_transform(fit_waveforms)
            t = fit_waveforms.shape[1]
            fit_waveforms = fit_waveforms.reshape(n, c, t).transpose(0, 2, 1)

        # relocate or just restrict to channel subset
        if self.relocated:
            fit_waveforms = relocate.relocated_waveforms_on_static_channels(
                fit_waveforms,
                main_channels=self.channels[in_unit[fit_choices]],
                channel_index=self.channel_index,
                target_channels=pca_channels,
                xyza_from=self.xyza[in_unit[fit_choices]],
                z_to=self.z_reg[in_unit[fit_choices]],
                geom=self.geom,
                registered_geom=self.registered_geom,
                match_distance=self.match_distance,
            )
        else:
            fit_waveforms = drift_util.get_waveforms_on_static_channels(
                fit_waveforms,
                self.geom,
                main_channels=self.channels[in_unit[fit_choices]],
                channel_index=self.channel_index,
                target_channels=pca_channels,
                n_pitches_shift=n_pitches_shift[fit_choices],
                registered_geom=self.registered_geom,
                match_distance=self.match_distance,
            )
            t = fit_waveforms.shape[1]

        # remove channels which are entirely nan
        not_entirely_nan_channels = np.flatnonzero(
            np.isfinite(fit_waveforms[:, 0, :]).any(axis=0)
        )
        if not not_entirely_nan_channels.size:
            print("All empty chans")
            return False, None, None

        subset_chans = not_entirely_nan_channels.size < fit_waveforms.shape[2]
        if subset_chans:
            fit_waveforms = fit_waveforms[:, :, not_entirely_nan_channels]

        # figure out which fit_waveforms overlap completely with the remaining channels
        no_nan = np.flatnonzero(np.isfinite(fit_waveforms[:, 0, :]).all(axis=1))
        if no_nan.size < max(self.min_cluster_size, self.n_pca_features):
            return False, no_nan, None
        fit_waveforms = fit_waveforms.reshape(fit_waveforms.shape[0], -1)
        if self.amplitude_normalized:
            amp_ix = in_unit[fit_choices]
            fit_waveforms /= self.amplitudes[amp_ix][:, None]

        # fit the per-cluster small rank PCA
        pca = PCA(
            self.n_pca_features,
            random_state=np.random.RandomState(seed=self.rg.bit_generator),
            whiten=True,
        )
        pca.fit(fit_waveforms[no_nan])
        if self.pca_imputation == "iterative":
            fit_waveforms = np.where(
                np.isfinite(fit_waveforms), fit_waveforms, pca.mean_[None]
            )
            isnan = np.setdiff1d(np.arange(fit_waveforms.shape[0]), no_nan)
            for i in range(10):
                emb = pca.fit_transform(fit_waveforms)
                fit_waveforms[isnan] = pca.inverse_transform(emb[isnan])

        # now, we have the PCA. let's embed our data in batches.
        pca_projs = np.full(
            (in_unit.size, self.n_pca_features),
            np.nan,
            dtype=fit_waveforms.dtype,
        )
        all_kept = []

        # load waveform embeddings and invert TPCA if we are relocating
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
                    main_channels=self.channels[in_unit[bs:be]],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    xyza_from=self.xyza[in_unit[bs:be]],
                    z_to=self.z_reg[in_unit[bs:be]],
                    geom=self.geom,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            else:
                batch = drift_util.get_waveforms_on_static_channels(
                    batch,
                    self.geom,
                    main_channels=self.channels[in_unit[bs:be]],
                    channel_index=self.channel_index,
                    target_channels=pca_channels,
                    n_pitches_shift=n_pitches_shift[bs:be],
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
                t = batch.shape[1]

            if subset_chans:
                batch = batch[:, :, not_entirely_nan_channels]

            if self.pca_imputation is None:
                no_nan = np.flatnonzero(np.isfinite(batch[:, 0, :]).all(axis=1))
                if not no_nan.size:
                    continue
                all_kept.append(bs + no_nan)
                batch = batch[no_nan].reshape(no_nan.size, -1)
                if self.amplitude_normalized:
                    amp_ix = in_unit[bs:be][no_nan]
                    batch /= self.amplitudes[amp_ix][:, None]
                pca_projs[bs + no_nan] = pca.transform(batch)
            elif self.pca_imputation in ("mean", "iterative"):
                batch = batch.reshape(batch.shape[0], -1)
                batch /= self.amplitudes[in_unit[bs:be]][:, None]
                pca_projs[bs:be] = pca.transform(
                    np.where(np.isfinite(batch), batch, pca.mean_[None])
                )
            else:
                assert False
        if self.pca_imputation is None:
            kept = np.concatenate(all_kept)
        elif self.pca_imputation in ("mean", "iterative"):
            kept = slice(None)
        else:
            assert False
        return True, kept, pca_projs[kept]

    def tpca_reassignment(
        self,
        clust_labels,
        in_unit,
        n_pitches_shift,
        centroid_samples=500,
        batch_size=128,
    ):
        unit_ids = np.unique(clust_labels)
        unit_ids = unit_ids[unit_ids >= 0]
        n_units = unit_ids.size

        # compute centroids in TPCA space
        centroids = []
        for split_label in unit_ids:
            in_split = np.flatnonzero(clust_labels == split_label)
            if in_split.size > centroid_samples:
                in_split = self.rg.choice(
                    in_split, size=centroid_samples, replace=False
                )
                in_split.sort()

            ix = in_unit[in_split]
            split_feats = batched_h5_read(self.tpca_features, ix)
            n_batch, rank, c = split_feats.shape

            # invert TPCA for relocation
            if self.relocated:
                split_feats = split_feats.transpose(0, 2, 1).reshape(n_batch * c, rank)
                split_feats = self.tpca.inverse_transform(split_feats)
                t = split_feats.shape[1]
                split_feats = split_feats.reshape(n_batch, c, t).transpose(0, 2, 1)

            # relocate or just restrict to channel subset
            if self.relocated:
                split_feats = relocate.relocated_waveforms_on_static_channels(
                    split_feats,
                    target_channels=slice(None),
                    main_channels=self.channels[ix],
                    channel_index=self.channel_index,
                    xyza_from=self.xyza[ix],
                    z_to=self.z_reg[ix],
                    geom=self.geom,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            else:
                split_feats = drift_util.get_waveforms_on_static_channels(
                    split_feats,
                    self.geom,
                    main_channels=self.channels[ix],
                    channel_index=self.channel_index,
                    n_pitches_shift=n_pitches_shift[in_split],
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            centroids.append(np.nan_to_num(np.nanmean(split_feats, axis=0)))
        centroids = np.array(centroids)
        chans = np.flatnonzero(np.isfinite(centroids[:, 0, :]).any(0))
        centroids = centroids[:, :, chans]

        # load waveform embeddings and invert TPCA if we are relocating
        new_labels = np.full_like(clust_labels, -1)
        assert clust_labels.shape == in_unit.shape
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
                    main_channels=self.channels[in_unit[bs:be]],
                    channel_index=self.channel_index,
                    target_channels=chans,
                    xyza_from=self.xyza[in_unit[bs:be]],
                    z_to=self.z_reg[in_unit[bs:be]],
                    geom=self.geom,
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            else:
                batch = drift_util.get_waveforms_on_static_channels(
                    batch,
                    self.geom,
                    main_channels=self.channels[in_unit[bs:be]],
                    channel_index=self.channel_index,
                    target_channels=chans,
                    n_pitches_shift=n_pitches_shift[bs:be],
                    registered_geom=self.registered_geom,
                    match_distance=self.match_distance,
                )
            
            d = nan_sqeuclidean_distances(batch.reshape(batch.shape[0], -1), centroids.reshape(n_units, -1))
            # d = batch.reshape(batch.shape[0], -1)[:, None] - centroids.reshape(n_units, -1)[None, :]
            # np.nan_to_num(d, copy=False)
            # d *= d
            # d = d.sum(2)
            new_labels[bs:be] = d.argmin(1)
            # for j, i in enumerate(range(bs, be)):
            #     wf = batch[j]
            #     mask = np.isfinite(wf[0])
            #     wf = wf[:, mask].ravel()[None]
            #     c = centroids[:, :, mask].reshape(n_units, -1)
            #     new_labels[i] = cdist(wf, c, metric="sqeuclidean").ravel().argmin()

        return new_labels

    def initialize_from_h5(
        self,
        peeling_hdf5_filename,
        peeling_featurization_pt,
        tpca_features_dataset_name="collisioncleaned_tpca_features",
        localizations_dataset_name="point_source_localizations",
        amplitudes_dataset_name="denoised_ptp_amplitudes",
        amplitude_vectors_dataset_name="denoised_ptp_amplitude_vectors",
    ):
        peeling_hdf5_filename = Path(peeling_hdf5_filename)
        h5 = h5py.File(peeling_hdf5_filename, "r", libver="latest", locking=False)
        self.geom = h5["geom"][:]
        self.channel_index = h5["channel_index"][:]
        self.channels = h5["channels"][:]
        self.match_distance = pdist(self.geom).min() / 2

        if self.use_localization_features or self.relocated or self.use_time_feature:
            self.t_s = h5["times_seconds"][:]

        if self.use_localization_features or self.relocated:
            self.xyza = h5[localizations_dataset_name][:]
            # registered spike positions (=originals if not relocated)
            self.z_reg = self.z = self.xyza[:, 2]

            self.amplitudes = h5[amplitudes_dataset_name][:]
            self.registered_geom = self.geom
            self.registered_channel_index = self.channel_index

            if self.motion_est is not None:
                self.z_reg = self.motion_est.correct_s(self.t_s, self.z)
            if self.relocated or self.n_pca_features or self.use_wfs_L2_norm:
                self.registered_geom = drift_util.registered_geometry(
                    self.geom, self.motion_est
                )
                self.registered_channel_index = waveform_util.make_channel_index(
                    self.registered_geom, self.channel_selection_radius
                )

        if (
            (self.use_localization_features and self.relocated)
            or self.n_pca_features
            or self.use_ptp
            or self.use_wfs_L2_norm
        ):
            self.amplitude_vectors = h5[amplitude_vectors_dataset_name]

        if self.use_localization_features:
            self.localization_features = np.c_[
                self.xyza[:, 0],
                self.z_reg,
                np.log(self.log_c + self.amplitudes),
            ]
            self.localization_features *= self.localization_feature_scales

        if self.n_pca_features or self.use_wfs_L2_norm:
            # don't load these one into memory, since it's a bit heavier
            self.tpca_features = h5[tpca_features_dataset_name]

        if peeling_featurization_pt is None:
            mdir = peeling_hdf5_filename.parent / f"{peeling_hdf5_filename.stem}_models"
            peeling_featurization_pt = mdir / "featurization_pipeline.pt"
            assert peeling_featurization_pt.exists()

        if (self.n_pca_features or self.use_wfs_L2_norm) and self.relocated:
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


class ChunkForwardBackwardFeatureSplit(FeatureSplit):
    def __init__(
        self,
        peeling_hdf5_filename,
        recording=None,
        chunk_size_s=300.0,
        **split_strategy_kwargs,
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
        split_strategy_kwargs["return_localization_features"] = True
        super().__init__(
            peeling_hdf5_filename, recording=recording, **split_strategy_kwargs
        )

        # Check for ensembling
        self.chunk_size_s = chunk_size_s
        assert (
            self.use_localization_features
        ), "Need to use loc features for ensembling over chunks"
        assert (
            self.recording is not None
        ), "Need to input recording for ensembling over chunks"
        assert (
            self.chunk_size_s is not None
        ), "Need to input chunk size for ensembling over chunks"

    def split_cluster(self, in_unit):
        chunk_samples = self.recording.sampling_frequency * self.chunk_size_s

        n_chunks = self.recording.get_num_samples() / chunk_samples
        # we'll count the remainder as a chunk if it's at least 2/3 of one
        n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
        n_chunks = int(max(1, n_chunks))

        # evenly divide the recording into chunks
        assert self.recording.get_num_segments() == 1
        start_time_s, end_time_s = self.recording._recording_segments[
            0
        ].sample_index_to_time(np.array([0, self.recording.get_num_samples() - 1]))
        chunk_times_s = np.linspace(start_time_s, end_time_s, num=n_chunks + 1)
        chunk_time_ranges_s = list(zip(chunk_times_s[:-1], chunk_times_s[1:]))

        chunks_indices_unit = [
            np.flatnonzero(
                np.logical_and(
                    self.t_s[in_unit] >= chunk_range[0],
                    self.t_s[in_unit] < chunk_range[1],
                )
            )
            for chunk_range in chunk_time_ranges_s
        ]

        # cluster each chunk. can be parallelized in the future.
        splits = [
            super().split_cluster(in_unit[idx_unit_chunk])
            for idx_unit_chunk in chunks_indices_unit
        ]

        new_labels_chunks = []
        for k in range(len(splits)):
            new_labels = np.full(len(in_unit), -1)
            if len(chunks_indices_unit[k]) and splits[k].new_labels is not None:
                new_labels[chunks_indices_unit[k]] = splits[k].new_labels
            elif len(chunks_indices_unit[k]) and splits[k].new_labels is None:
                # moved the 0 logic from split_cluster to here
                new_labels[chunks_indices_unit[k]] = 0
            new_labels_chunks.append(new_labels)

        split_sortings = [
            DARTsortSorting(
                times_samples=self.t_s[
                    in_unit
                ],  # Needed for initialization but not for anything else
                channels=self.channels[in_unit],
                labels=new_labels,
                extra_features={
                    "point_source_localizations": self.xyza[in_unit],
                    "denoised_ptp_amplitudes": self.amplitudes[in_unit],
                    "times_seconds": self.t_s[in_unit],
                },
            )
            for new_labels in new_labels_chunks
        ]

        labels_ensembled = forward_backward(
            self.recording,
            chunk_time_ranges_s,
            split_sortings,
            log_c=self.log_c,
            feature_scales=self.localization_feature_scales,
            adaptive_feature_scales=self.rescale_all_features,
            motion_est=self.motion_est,
            verbose=False,
        )

        is_split = np.setdiff1d(np.unique(labels_ensembled), [-1]).size > 1

        return SplitResult(
            is_split=is_split,
            in_unit=in_unit,
            new_labels=labels_ensembled,
            x=self.localization_features[in_unit, 0],
            z_reg=self.localization_features[in_unit, 2],
        )


class NullSplit(SplitStrategy):
    def __init__(self, *args, **kwargs):
        pass

    def split_cluster(self, in_unit):
        return SplitResult(in_unit=in_unit)


# this is to help split_clusters take a string argument
all_split_strategies = [
    FeatureSplit,
    ChunkForwardBackwardFeatureSplit,
    NullSplit,
]
split_strategies_by_class_name = {cls.__name__: cls for cls in all_split_strategies}

# -- parallelism widgets


class SplitJobContext:
    def __init__(self, split_strategy):
        self.split_strategy = split_strategy


_split_job_context = None


def _split_job_init(
    split_strategy_class_name, peeling_hdf5_filename, split_strategy_kwargs
):
    global _split_job_context
    SplitStrategy = split_strategies_by_class_name[split_strategy_class_name]
    if split_strategy_kwargs is None:
        split_strategy_kwargs = {}
    split_strategy_kwargs["peeling_hdf5_filename"] = peeling_hdf5_filename
    _split_job_context = SplitJobContext(SplitStrategy(**split_strategy_kwargs))


def _split_job(in_unit):
    return _split_job_context.split_strategy.split_cluster(in_unit)


# other helpers


def mad_sigma(x, axis=0, keepdims=False, gauss_correct=True):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=keepdims)
    if gauss_correct:
        mad *= 1.4826
    return mad


def nan_sqeuclidean_distances(X, Y, copy_X=False, copy_Y=True):
    missing_X = np.isnan(X)
    missing_Y = np.isnan(Y)
    if copy_X:
        X = X.copy()
    if copy_Y:
        Y = Y.copy()
    X[missing_X] = 0
    Y[missing_Y] = 0
    distances = cdist(X, Y, metric="sqeuclidean")
    
    # Adjust distances for missing values
    X *= X
    Y *= Y
    distances -= np.dot(X, missing_Y.T)
    distances -= np.dot(missing_X, Y.T)

    return distances
        