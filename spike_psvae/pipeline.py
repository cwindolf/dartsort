import numpy as np
import h5py
from sklearn.decomposition import PCA

from . import (
    denoise,
    cluster_utils,
    pre_deconv_merge_split,
    after_deconv_merge_split,
    spike_train_utils,
    extractors,
    subtract,
)


def initial_clustering(
    sub_h5,
    raw_data_bin,
    remove_pair_duplicates=True,
    remove_self_duplicates=True,
    use_registered=True,
    frames_dedup=12,
    reducer=np.median,
):
    # load features
    with h5py.File(sub_h5, "r") as h5:
        spike_index = h5["spike_index"][:]
        x, y, z, alpha, z_rel = h5["localizations"][:].T
        maxptps = h5["maxptps"][:]
        geom = h5["geom"][:]
        z_reg = h5["z_reg"][:]

    z = z_reg if use_registered else z

    print("Initial clustering...")
    (
        clusterer,
        cluster_centers,
        tspike_index,
        tx,
        tz,
        tmaxptps,
        idx_keep_full,
    ) = cluster_utils.cluster_spikes(
        x,
        z,
        maxptps,
        spike_index,
        split_big=True,
        do_remove_dups=False,
    )
    
    # remove cross dups after remove self dups
    if remove_pair_duplicates:
        print("dups", flush=True)
        (
            clusterer,
            duplicate_indices,
            duplicate_spikes,
        ) = cluster_utils.remove_duplicate_spikes(
            clusterer, tspike_index[:, 0], tmaxptps, frames_dedup=frames_dedup
        )
        clusterer.labels_ = spike_train_utils.make_labels_contiguous(clusterer.labels_)

    # remove self-duplicate spikes
    if remove_self_duplicates:
        print("Self-duplicates...")
        kept_ix, removed_ix = cluster_utils.remove_self_duplicates(
            tspike_index[:, 0],
            clusterer.labels_,
            raw_data_bin,
            geom.shape[0],
            frame_dedup=20,
        )
        clusterer.labels_[removed_ix] = -1
        clusterer.labels_ = spike_train_utils.make_labels_contiguous(clusterer.labels_)
    
    # labels in full index space (not triaged)
    spike_train = spike_index.copy()
    spike_train[:, 1] = -1
    spike_train[idx_keep_full, 1] = clusterer.labels_

    (
        spike_train,
        _,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        geom.shape[0],
        raw_data_bin,
        sort_by_time=False,
        reducer=reducer,
        min_n_spikes=0,
        pbar=True,
    )
    aligned_spike_index = np.c_[spike_train[:, 0], spike_index[:, 1]]
    clusterer.labels_ = spike_train[idx_keep_full, 1]

    return spike_train, aligned_spike_index, templates, template_shifts, clusterer, idx_keep_full


def pre_deconv_split_step(
    sub_h5,
    raw_data_bin,
    residual_data_bin,
    spike_train,
    aligned_spike_index,
    templates,
    template_shifts,
    clusterer,
    idx_keep_full,
    use_registered=True,
    reducer=np.median,
    device=None,
):
    with h5py.File(sub_h5, "r") as h5:
        sub_wf = h5["subtracted_waveforms"]
        firstchans = h5["first_channels"][:]
        x, y, z, alpha, z_rel = h5["localizations"][:].T
        geom = h5["geom"][:]
        z_reg = h5["z_reg"][:]
        channel_index = h5["channel_index"][:]
        tpca = subtract.tpca_from_h5(h5)
        orig_spike_index = h5["spike_index"][:]

        z = z_reg if use_registered else z

        spike_train[:, 1] = pre_deconv_merge_split.split_clusters(
            residual_data_bin,
            sub_wf,
            firstchans,
            aligned_spike_index,
            templates.ptp(1).argmax(1),
            template_shifts,
            spike_train[:, 1],
            x,
            z_reg,
            geom,
            denoise.SingleChanDenoiser().load(),
            device,
            tpca,
        )

        # ks split
        print("before ks split", spike_train[:, 1].max() + 1)
        spike_train[:, 1], split_map = pre_deconv_merge_split.ks_maxchan_tpca_split(
            h5["subtracted_tpca_projs"],
            channel_index,
            aligned_spike_index[:, 1],
            spike_train[:, 1],
            tpca,
            recursive=True,
            top_pc_init=True,
            aucsplit=0.85,
            min_size_split=50,
            max_split_corr=0.9,
            min_amp_sim=0.2,
            min_split_prop=0.05,
        )
        print("after ks split", spike_train[:, 1].max() + 1)

    # re-order again
    clusterer.labels_ = spike_train[:, 1][idx_keep_full]
    cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
    clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
    cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
    spike_train[idx_keep_full, 1] = clusterer.labels_

    # note, we use the original spike index times here so that the template
    # shifts are correct relative to the stored subtracted waveforms
    (
        spike_train,
        _,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        np.c_[orig_spike_index[:, 0], spike_train[:, 1]],
        geom.shape[0],
        raw_data_bin,
        sort_by_time=False,
        reducer=reducer,
        pbar=True,
    )
    aligned_spike_index = np.c_[spike_train[:, 0], spike_index[:, 1]]
    clusterer.labels_ = spike_train[idx_keep_full, 1]

    return spike_train, aligned_spike_index, templates, template_shifts, clusterer


def pre_deconv_merge_step(
    sub_h5,
    raw_data_bin,
    residual_data_bin,
    spike_train,
    aligned_spike_index,
    templates,
    template_shifts,
    clusterer,
    idx_keep_full,
    final_align_max_shift=25,
    final_clean_min_spikes=5,
    device=None,
    merge_dipscore=0.5,
    reducer=np.median,
):
    with h5py.File(sub_h5, "r") as h5:
        sub_wf = h5["subtracted_waveforms"]
        firstchans = h5["first_channels"][:]
        x, y, z, alpha, z_rel = h5["localizations"][:].T
        geom = h5["geom"][:]
        z_reg = h5["z_reg"][:]
        tpca = subtract.tpca_from_h5(h5)

        K_pre = spike_train[:, 1].max() + 1
        spike_train[:, 1] = pre_deconv_merge_split.get_merged(
            residual_data_bin,
            sub_wf,
            firstchans,
            geom,
            templates,
            template_shifts,
            len(templates),
            aligned_spike_index,
            spike_train[:, 1],
            x,
            z_reg,
            denoise.SingleChanDenoiser().load(),
            device,
            tpca,
            threshold_diptest=merge_dipscore,
        )
    print("pre->post merge", K_pre, spike_train[:, 1].max() + 1)

    # re-order again
    clusterer.labels_ = spike_train[idx_keep_full, 1]
    cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
    clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
    spike_train[idx_keep_full, 1] = clusterer.labels_

    # final templates
    # here, we don't need to use the original spike index, since we
    # won't be touching the subtracted waveforms any more.
    (
        spike_train,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        geom.shape[0],
        raw_data_bin,
        sort_by_time=False,
        reducer=reducer,
        max_shift=final_align_max_shift,
        min_n_spikes=final_clean_min_spikes,
        pbar=True,
    )
    spike_index = np.c_[spike_train[:, 0], spike_index[:, 1]]
    clusterer.labels_ = spike_train[idx_keep_full, 1]

    return spike_train, spike_index, templates, template_shifts


def post_deconv_split_step(
    deconv_dir,
    deconv_results_h5,
    bin_file,
    geom,
    clean_min_spikes=0,
    reducer=np.median,
):
    spike_train = np.load(deconv_dir / "spike_train.npy")
    templates = np.load(deconv_dir / "templates.npy")
    assert templates.shape[0] == spike_train[:, 1].max() + 1
    print("original")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())
    n_channels = templates.shape[2]
    assert n_channels == geom.shape[0]

    deconv_extractor = extractors.DeconvH5Extractor(
        deconv_results_h5, bin_file
    )
    assert deconv_extractor.spike_train_up.shape == spike_train.shape

    # deconv can produce an un-aligned spike train.
    (
        spike_train,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )

    # the initial split
    spike_train[:, 1] = after_deconv_merge_split.split(
        spike_train[:, 1],
        deconv_extractor,
        order=order,
        pc_split_rank=6,
    )
    print("A")
    print("sorted?", (spike_train[1:, 0] >= spike_train[:-1, 0]).all())
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())

    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]

    # clean big
    after_deconv_merge_split.clean_big_clusters(
        templates,
        spike_train,
        deconv_extractor.ptp[order],
        bin_file,
        geom,
        reducer=reducer,
    )
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]

    print("B")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())

    # remove oversplits
    (
        spike_train,
        templates,
    ) = after_deconv_merge_split.remove_oversplits(templates, spike_train)

    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]

    return spike_train, order, templates


def post_deconv_merge_step(
    spike_train,
    order,
    templates,
    deconv_dir,
    deconv_results_h5,
    bin_file,
    geom,
    clean_min_spikes=25,
    reducer=np.median,
):
    spike_train = spike_train.copy()

    deconv_extractor = extractors.DeconvH5Extractor(
        deconv_results_h5, bin_file
    )
    assert deconv_extractor.spike_train_up.shape == spike_train.shape
    n_channels = geom.shape[0]

    print("Just before merge...")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    spike_train[:, 1] = after_deconv_merge_split.merge(
        spike_train[:, 1],
        templates,
        deconv_extractor,
        geom,
        order=order,
        spike_times=spike_train[:, 0],
        n_chan_merge=10,
        tpca=PCA(6),
        isi_veto=False,
    )
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]
    print("Just after merge...")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    print("Clean big ")
    n_cleaned = after_deconv_merge_split.clean_big_clusters(
        templates,
        spike_train,
        deconv_extractor.ptp[order],
        bin_file,
        geom,
        reducer=reducer,
    )
    print(f"{n_cleaned=}")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=0,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]

    print("clean/align")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    print("Remove oversplit")
    spike_train, templates = after_deconv_merge_split.remove_oversplits(
        templates, spike_train
    )
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    return spike_train, order, templates


def post_deconv2_clean_step(
    deconv2_dir,
    deconv2_results_h5,
    bin_file,
    geom,
    clean_min_spikes=25,
):
    n_channels = geom.shape[0]

    # load results
    with h5py.File(deconv2_results_h5) as f:
        # for k in f:
        #     print(k, f[k].shape)
        maxptps = f["maxptps"][:]
        firstchans = f["first_channels"][:]
        x, y, z_rel, z_abs, alpha = f["localizations"][:].T
        print(
            f"deconv result shapes {maxptps.shape=} {x.shape=} {z_abs.shape=}"
        )
    spike_train = np.load(deconv2_dir / "spike_train.npy")
    print(deconv2_dir / "spike_train.npy", f"{spike_train.shape}")
    assert (spike_train.shape[0],) == maxptps.shape == firstchans.shape

    (
        spike_train,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )

    print("original")
    print((spike_train[1:, 0] >= spike_train[:-1, 0]).all())
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())

    print("Before clean big ")
    n_cleaned = after_deconv_merge_split.clean_big_clusters(
        templates, spike_train, maxptps[order], bin_file, geom
    )
    print(f"n_cleaned={n_cleaned}")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )
    order = order[reorder]

    print("After clean big")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    print("Remove oversplits")
    (
        spike_train,
        split_templates,
    ) = after_deconv_merge_split.remove_oversplits(templates, spike_train)
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )
    order = order[reorder]

    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    return spike_train, order, templates
