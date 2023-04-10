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
    deconv_resid_merge,
    waveform_utils,
    spike_reassignment,
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
        clusterer.labels_ = spike_train_utils.make_labels_contiguous(
            clusterer.labels_
        )

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
        clusterer.labels_ = spike_train_utils.make_labels_contiguous(
            clusterer.labels_
        )

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

    return (
        spike_train,
        aligned_spike_index,
        templates,
        template_shifts,
        clusterer,
        idx_keep_full,
    )


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
        orig_spike_index = h5["spike_index"][:]
        channel_index = h5["channel_index"][:]
        firstchans = channel_index[:, 0][orig_spike_index[:,1]]
        # firstchans = h5["first_channels"][:]
        x, y, z, alpha, z_rel = h5["localizations"][:].T
        geom = h5["geom"][:]
        z_reg = h5["z_reg"][:]
        channel_index = h5["channel_index"][:]
        # tpca = subtract.tpca_from_h5(h5)
        # tpca = 
        # tpca_mean = h5["tpca_mean"][:]
        # tpca_components = h5["tpca_components"][:]
        # print("Loading TPCA from h5")
        # tpca = PCA(tpca_components.shape[0])
        # tpca.mean_ = tpca_mean
        # tpca.components_ = tpca_components

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
        (
            spike_train[:, 1],
            split_map,
        ) = pre_deconv_merge_split.ks_maxchan_tpca_split(
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
    aligned_spike_index = np.c_[spike_train[:, 0], aligned_spike_index[:, 1]]
    clusterer.labels_ = spike_train[idx_keep_full, 1]

    return (
        spike_train,
        aligned_spike_index,
        templates,
        template_shifts,
        clusterer,
    )


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
    merge_resid_threshold=1.1,
    reducer=np.median,
):
    with h5py.File(sub_h5, "r") as h5:
        geom = h5["geom"][:]

    # start with new merge step before LDA with smaller threshold
    # K_pre = spike_train[:, 1].max() + 1
    # print(f"Before resid merge {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})")
    # labels_updated = deconv_resid_merge.run_deconv_merge(
    #     spike_train[spike_train[:, 1] >= 0],
    #     geom,
    #     raw_data_bin,
    #     templates.ptp(1).argmax(1),
    #     merge_resid_threshold=merge_resid_threshold,
    # )
    # spike_train[spike_train[:, 1] >= 0, 1] = labels_updated
    # print(f"Resid merge: {K_pre} -> {np.unique(labels_updated).size} {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})")

    # re-order again
    # clusterer.labels_ = spike_train[idx_keep_full, 1]
    # cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
    # clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
    # spike_train[idx_keep_full, 1] = clusterer.labels_

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
        pbar=True,
    )
    aligned_spike_index = np.c_[spike_train[:, 0], aligned_spike_index[:, 1]]
    clusterer.labels_ = spike_train[idx_keep_full, 1]

    with h5py.File(sub_h5, "r") as h5:
        sub_wf = h5["subtracted_waveforms"]
        orig_spike_index = h5["spike_index"][:]
        channel_index = h5["channel_index"][:]
        firstchans = channel_index[:, 0][orig_spike_index[:,1]]
        # firstchans = h5["first_channels"][:]
        x, y, z, alpha, z_rel = h5["localizations"][:].T
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
        pbar=True,
    )
    aligned_spike_index = np.c_[spike_train[:, 0], aligned_spike_index[:, 1]]
    clusterer.labels_ = spike_train[idx_keep_full, 1]

    # remove oversplits -- important to do this after the big align
    K_pre = spike_train[:, 1].max() + 1
    print(
        f"Before resid merge {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )
    times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
        spike_train[spike_train[:, 1] >= 0],
        geom,
        raw_data_bin,
        templates.ptp(1).argmax(1),
        merge_resid_threshold=merge_resid_threshold,
    )
    spike_train[spike_train[:, 1] >= 0, 0] = times_updated
    spike_train[spike_train[:, 1] >= 0, 1] = labels_updated
    order = np.argsort(spike_train[:, 0])
    spike_train = spike_train[order]
    print(
        f"Resid merge: {K_pre} -> {np.unique(labels_updated).size} {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )

    # and clean up the spike train to finish.
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        geom.shape[0],
        raw_data_bin,
        sort_by_time=False,
        reducer=reducer,
        min_n_spikes=final_clean_min_spikes,
        pbar=True,
    )
    aligned_spike_index = np.c_[spike_train[:, 0], aligned_spike_index[:, 1]]
    clusterer.labels_ = spike_train[idx_keep_full, 1]

    # we don't use this, but just for bookkeeping, in case another step is added...
    order = order[reorder]

    return spike_train, aligned_spike_index, templates, order


def post_deconv_split_step(
    deconv_dir,
    deconv_results_h5,
    raw_data_bin,
    geom,
    merge_resid_threshold=2.0,
    clean_min_spikes=0,
    reducer=np.median,
):
    spike_train = np.load(deconv_dir / "spike_train.npy")
    templates = np.load(deconv_dir / "templates.npy")
    assert templates.shape[0] >= spike_train[:, 1].max() + 1
    print("original")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())
    n_channels = templates.shape[2]
    assert n_channels == geom.shape[0]

    deconv_extractor = extractors.DeconvH5Extractor(
        deconv_results_h5, raw_data_bin
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
        raw_data_bin,
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
        raw_data_bin,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]

    # # clean big
    # after_deconv_merge_split.clean_big_clusters(
    #     templates,
    #     spike_train,
    #     deconv_extractor.ptp[order],
    #     raw_data_bin,
    #     geom,
    #     reducer=reducer,
    # )
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        raw_data_bin,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]

    print("B")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())

    # remove oversplits
    # (
    #     spike_train,
    #     templates,
    # ) = after_deconv_merge_split.remove_oversplits(templates, spike_train)
    print(
        f"Before resid merge {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )
    times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
        spike_train[spike_train[:, 1] >= 0],
        geom,
        raw_data_bin,
        templates.ptp(1).argmax(1),
        merge_resid_threshold=merge_resid_threshold,
    )
    spike_train[spike_train[:, 1] >= 0, 0] = times_updated
    spike_train[spike_train[:, 1] >= 0, 1] = labels_updated
    reorder = np.argsort(spike_train[:, 0])
    spike_train = spike_train[reorder]
    order = order[reorder]

    print(
        f"Resid merge: {np.unique(labels_updated).size} {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )

    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        raw_data_bin,
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
    raw_data_bin,
    geom,
    merge_resid_threshold=2.5,
    clean_min_spikes=25,
    reducer=np.median,
):
    spike_train = spike_train.copy()

    deconv_extractor = extractors.DeconvH5Extractor(
        deconv_results_h5, raw_data_bin
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
        distance_threshold=5.0,
    )
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        raw_data_bin,
        min_n_spikes=clean_min_spikes,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]
    print("Just after merge...")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    # print("Clean big ")
    # n_cleaned = after_deconv_merge_split.clean_big_clusters(
    #     templates,
    #     spike_train,
    #     deconv_extractor.ptp[order],
    #     raw_data_bin,
    #     geom,
    #     reducer=reducer,
    # )
    # print(f"{n_cleaned=}")
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
        raw_data_bin,
        min_n_spikes=0,
        pbar=True,
        reducer=reducer,
    )
    order = order[reorder]

    print("clean/align")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    print("Remove oversplit")
    # spike_train, templates = after_deconv_merge_split.remove_oversplits(
    #     templates, spike_train
    # )
    print(
        f"Before resid merge {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )
    times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
        spike_train[spike_train[:, 1] >= 0],
        geom,
        raw_data_bin,
        templates.ptp(1).argmax(1),
        merge_resid_threshold=merge_resid_threshold,
    )
    spike_train[spike_train[:, 1] >= 0, 0] = times_updated
    spike_train[spike_train[:, 1] >= 0, 1] = labels_updated
    reorder = np.argsort(spike_train[:, 0])
    spike_train = spike_train[reorder]
    order = order[reorder]
    print(
        f"Resid merge: {np.unique(labels_updated).size} {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        raw_data_bin,
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
    raw_data_bin,
    geom,
    merge_resid_threshold=3.0,
    clean_min_spikes=25,
    do_clean_big=False,
):
    n_channels = geom.shape[0]

    # load results
    with h5py.File(deconv2_results_h5) as f:
        # for k in f:
        #     print(k, f[k].shape)
        maxptps = f["maxptps"][:]
        spike_index = f["spike_index_up"][:] #changed to spike_index_up
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
        raw_data_bin,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )

    print("original")
    print((spike_train[1:, 0] >= spike_train[:-1, 0]).all())
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())

#     if do_clean_big:
#         print("Before clean big ")
#         n_cleaned = after_deconv_merge_split.clean_big_clusters(
#             templates, spike_train, maxptps[order], raw_data_bin, geom
#         )
#         print(f"n_cleaned={n_cleaned}")
#         u, c = np.unique(spike_train[:, 1], return_counts=True)
#         print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

#         (
#             spike_train,
#             reorder,
#             templates,
#             template_shifts,
#         ) = spike_train_utils.clean_align_and_get_templates(
#             spike_train,
#             n_channels,
#             raw_data_bin,
#             min_n_spikes=clean_min_spikes,
#             pbar=True,
#         )
#         order = order[reorder]

#         print("After clean big")
#         u, c = np.unique(spike_train[:, 1], return_counts=True)
#         print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())
#         clean_st = spike_train.copy()
#         clean_temp = templates.copy()
#         clean_ord = order.copy()

    print("Remove oversplits")
    # (
    #     spike_train,
    #     split_templates,
    # ) = after_deconv_merge_split.remove_oversplits(templates, spike_train)
    print(
        f"Before resid merge {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )
    times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
        spike_train[spike_train[:, 1] >= 0],
        geom,
        raw_data_bin,
        templates.ptp(1).argmax(1),
        merge_resid_threshold=merge_resid_threshold,
    )
    spike_train[spike_train[:, 1] >= 0, 0] = times_updated
    spike_train[spike_train[:, 1] >= 0, 1] = labels_updated
    reorder = np.argsort(spike_train[:, 0])
    spike_train = spike_train[reorder]
    order = order[reorder]
    print(
        f"After resid merge: {np.unique(labels_updated).size} {spike_train.shape=} {(spike_train[:, 1] >= 0).sum()=} {(spike_train[:, 1].max() + 1)=} {np.unique(spike_train[:,1]).size=})"
    )
    (
        spike_train,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        raw_data_bin,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )
    order = order[reorder]

    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    return spike_train, order, templates #, clean_st, clean_ord, clean_temp


# data_path = '/media/cat/data/'
# data_name = 'CSH_ZAD_026_5min'
# data_dir = data_path + data_name + '/'
# raw_bin = data_dir + 'CSH_ZAD_026_snip.ap.bin'

# deconv_dir = '/media/cat/julien/5min_full_pipeline_output/second_deconv_output/'
# data_dir = '/media/cat/data/CSH_ZAD_026_5min/'
# residual_bin_path = deconv_dir + 'residual.bin'
# templates_path = deconv_dir + 'templates.npy'
# spike_train_path = deconv_dir + 'spike_train.npy'
# spike_index_path = deconv_dir + 'spike_index.npy'
# spike_index = np.load(spike_index_path)
# tpca =  fit_tpca_bin(spike_index, geom, raw_bin)

# geom_path = data_dir + 'np1_channel_map.npy'
# # tpca_components = np.load(data_dir + 'tpca_components.npy')
# # tpca_mean = np.load(data_dir + 'tpca_mean.npy')

# output_path = '/media/cat/data/outlier_detection'


def reassign_and_triage_spikes(
    deconv_dir,
    raw_data_bin,
    geom,
    n_chans=8,
    n_sim_units=2,
    num_sigma_outlier=4,
    batch_size=4096,
):

    residual_bin_path = deconv_dir + "residual.bin"
    templates_path = deconv_dir + "templates.npy"
    spike_train_path = deconv_dir + "spike_train.npy"
    spike_index_path = deconv_dir + "spike_index.npy"
    spike_index = np.load(spike_index_path)
    # compute tpca on raw waveforms
    tpca = waveform_utils.fit_tpca_bin(spike_index, geom, raw_data_bin)
    # run spike reassignment and outlier triaging
    (
        soft_assignment_scores,
        reassignment,
        reassigned_scores,
    ) = spike_reassignment.run(
        residual_bin_path,
        templates_path,
        spike_train_path,
        geom,
        tpca,
        n_chans=n_chans,
        n_sim_units=n_sim_units,
        num_sigma_outlier=num_sigma_outlier,
        batch_size=batch_size,
    )
    return soft_assignment_scores, reassignment, reassigned_scores
