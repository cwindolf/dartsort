import numpy as np
import h5py
from spike_psvae import (
    spike_train_utils,
    before_deconv_merge_split,
    deconv_resid_merge,
    cluster_utils,
)
from scipy.spatial.distance import cdist


def registered_maxchan(
    spike_index, p, geom, pfs=30000, offset=None, depth_domain=None, ymin=0
):
    pos = geom[spike_index[:, 1]]
    if p.ndim == 1 or p.shape[1] == 1:
        if offset is None:
            p = p - np.median(p)
        pos[:, 1] -= p[spike_index[:, 0] // pfs]
    elif p.ndim == 2:
        from spike_psvae.ibme import warp_nonrigid

        pos[:, 1] -= ymin
        print(p.shape, depth_domain.shape)
        pos[:, 1] = warp_nonrigid(
            pos[:, 1], spike_index[:, 0] / pfs, p, depth_domain=depth_domain
        )
    regmc = cdist(pos, geom).argmin(1)
    return regmc


def augment_geom(geom, p):
    sorted_geom = np.array(sorted(geom, key=lambda x: (x[1], x[0])))
    z_pitch = np.max(np.diff(sorted_geom[:, 1]))
    total_drift = np.max(p) - np.min(p)
    orig_z_max = sorted_geom[-1][1]
    orig_z_min = sorted_geom[0][1]
    L = orig_z_max - orig_z_min
    uniq_x_location = sorted_geom[0:len(np.unique(geom[:,0])), 0]
    uniq_x_location = uniq_x_location.tolist()
    
    aug_geom = geom
    del geom
    z_augment = orig_z_max
    CH_N = np.shape(aug_geom)[0]
    x_augment = aug_geom[-1][0]
    x_idx = uniq_x_location.index(x_augment)
    z_max = orig_z_max + total_drift
    while z_augment < orig_z_max + total_drift:
        if CH_N % 2 == 0:
            z_augment += z_pitch
        x_idx = x_idx % 4
        x_augment = uniq_x_location[x_idx]
        x_idx += 1    
        aug_geom = np.append(aug_geom, [[x_augment, z_augment]], axis = 0)
        CH_N += 1 
       
    return aug_geom, CH_N, L, z_max

# relocate max_chan on an augmented version of probe (NP1), for Eric's insertion data
def reloc_maxchan_augmented_geom(spike_index, p, geom, pfs=30000, offset=None, depth_domain=None, ymin=0
):
    aug_geom, CH_N, L, z_max = augment_geom(geom, p)
    pos = geom[spike_index[:, 1]]
    if p.ndim == 1 or p.shape[1] == 1:
        p = p - p[0]
        p = p + L - z_max
        pos[:, 1] -= p[spike_index[:, 0] // pfs]
        
    aug_which = np.intersect1d(np.argwhere(aug_geom[:, 1]>np.min(pos[:,1])), np.argwhere(aug_geom[:, 1]<np.max(pos[:,1])))

    batch_size = 10000
    n_batch = int(np.ceil(len(pos)/batch_size))
    regmc = []
    for i in range(n_batch):
        if i < n_batch - 1:
            regmc_batch = cdist(pos[(batch_size * i):(batch_size * (i + 1))], aug_geom[aug_which]).argmin(1)
        else:
            regmc_batch = cdist(pos[(batch_size * i):None], aug_geom[aug_which]).argmin(1)

        regmc = np.int32(np.append(regmc, regmc_batch))
    regmc = aug_which[regmc]
    return regmc

def new_merge_split(
    spike_train,
    n_channels,
    raw_bin,
    sub_h5,
    geom,
    outdir,
    n_workers=1,
    herding_npcs=2,
    herding_clust="hdbscan",
    merge_resid_threshold=2.0,
    relocated=False,
    trough_offset=42,
    spike_length_samples=121,
    maxchan_split=True,
    max_shift=20,
):
    (
        aligned_spike_train,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        raw_bin,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

    if maxchan_split:
        new_labels = before_deconv_merge_split.split_clusters(
            aligned_spike_train[:, 1],
            raw_bin,
            sub_h5,
            n_workers=n_workers,
            relocated=relocated,
        )
    else:
        new_labels = before_deconv_merge_split.split_clusters(
            aligned_spike_train[:, 1],
            raw_bin,
            sub_h5,
            n_workers=n_workers,
            relocated=relocated,
            split_steps=(before_deconv_merge_split.herding_split,),
            recursive_steps=(False,),
        )

    (
        aligned_spike_train2,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        np.c_[aligned_spike_train[:, 0], new_labels],
        n_channels,
        raw_bin,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

    assert (order == np.arange(len(order))).all()

    print("Save split...")
    np.save(outdir + "/" + "split_st.npy", aligned_spike_train2)
    np.save(outdir + "/" + "split_templates.npy", templates)
    np.save(outdir + "/" + "split_order.npy", order)

    aligned_times, new_labels = before_deconv_merge_split.merge_clusters(
        sub_h5,
        raw_bin,
        aligned_spike_train2[:, 1],
        templates,
        relocated=relocated,
        trough_offset=trough_offset,
        n_jobs=n_workers,
    )

    (
        aligned_spike_train3,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        np.c_[aligned_spike_train2[:, 0], new_labels],
        n_channels,
        raw_bin,
        max_shift=max_shift,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

    kept = aligned_spike_train3[:, 1] >= 0
    times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
        aligned_spike_train3[kept],
        geom,
        raw_bin,
        templates.ptp(1).argmax(1),
        merge_resid_threshold=merge_resid_threshold,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    aligned_spike_train3[kept, 0] = times_updated
    aligned_spike_train3[kept, 1] = labels_updated

    (
        aligned_spike_train4,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        aligned_spike_train3,
        n_channels,
        raw_bin,
        max_shift=max_shift,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    order = order[reorder]

    print("Save merge...")
    np.save(outdir + "/" + "merge_st.npy", aligned_spike_train4)
    np.save(outdir + "/" + "merge_templates.npy", templates)
    np.save(outdir + "/" + "merge_order.npy", order)

    return aligned_spike_train4, templates, order


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
        do_copy_spikes=False,
        ptp_low_threshold=5,
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
