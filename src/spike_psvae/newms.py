import numpy as np
from . import (
    spike_train_utils,
    before_deconv_merge_split,
    deconv_resid_merge,
    uhd_split_merge,
)
from scipy.spatial.distance import cdist
import h5py


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


def new_merge_split(
    spike_train,
    n_channels,
    raw_bin,
    sub_h5,
    geom,
    outdir,
    n_workers=1,
    merge_resid_threshold=2.0,
    threshold_diptest=0.5,
    relocated=False,
    trough_offset=42,
    spike_length_samples=121,
    # extra_pc_split=True,
    # pc_only=False,
    # exp_split=False,
    load_split=False,
    split_kwargs=None,
    drift_merge=False,
    zero_radius_um=70,
    threshold_resid=.25,
    bin_size_um=None,
    sampling_rate=32000,
):
    orig_labels = spike_train[:, 1].copy()
    split_kwargs = {} if split_kwargs is None else split_kwargs
    outdir.mkdir(exist_ok=True, parents=True)
    if not load_split:
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

        new_labels = before_deconv_merge_split.split_clusters(
            aligned_spike_train[:, 1],
            raw_bin,
            sub_h5,
            n_workers=n_workers,
            relocated=relocated,
            **split_kwargs,
        )

        untriaged_orig = orig_labels >= 0
        # untriaged_orig_a = orig_labels_a >= 0
        untriaged_now = new_labels >= 0

        print(f"Split total kept: {(untriaged_now.sum() / untriaged_orig.sum())=}")
        # print(f"{untriaged_orig.sum()=} {untriaged_orig_a.sum()=} {untriaged_now.sum()=}")

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
            order_units_by_z=True,
            geom=geom,
        )

        assert (order == np.arange(len(order))).all()

        print("Save split...")
        np.save(outdir / "split_st.npy", aligned_spike_train2)
        np.save(outdir / "split_templates.npy", templates)
        np.save(outdir / "split_order.npy", order)
    else:
        aligned_spike_train2 = np.load(outdir / "split_st.npy")
        templates = np.load(outdir / "split_templates.npy")
        order = np.load(outdir / "split_order.npy")
    
    #disable diptest
    # aligned_times, new_labels = before_deconv_merge_split.merge_clusters(
    #     sub_h5,
    #     raw_bin,
    #     aligned_spike_train2[:, 1],
    #     templates,
    #     relocated=relocated,
    #     trough_offset=trough_offset,
    #     n_jobs=n_workers,
    #     threshold_diptest=threshold_diptest,
    # )

    (
        aligned_spike_train3,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        np.c_[aligned_spike_train2[:, 0], new_labels],
        n_channels,
        raw_bin,
        max_shift=20,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        sort_by_time=False,
    )
    assert np.array_equal(order, np.arange(len(order)))

    kept = aligned_spike_train3[:, 1] >= 0
    
    if not drift_merge:
        times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
            aligned_spike_train3[kept],
            geom,
            raw_bin,
            templates.ptp(1).argmax(1),
            merge_resid_threshold=merge_resid_threshold,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
        )
    else:
        spt = aligned_spike_train3[kept]
        labels_split = aligned_spike_train3[:,1][kept]
        h5 = h5py.File(sub_h5, "r")
        x, y, z_abs, alpha = h5["localizations"][:, :4].T
        z_reg = h5["z_reg"][:]
        raw_data_bin = raw_bin
        print("running drift-aware merge...")
        labels_updated = uhd_split_merge.template_deconv_merge(spt, labels_split, z_abs[kept], z_reg[kept],
                                                               x[kept], geom, raw_data_bin,
                                                               threshold_resid=threshold_resid, su_chan_vis=1.5, 
                                                               bin_size_um=bin_size_um,
                                                               zero_radius_um=zero_radius_um, n_jobs=n_workers,
                                                               sampling_rate=sampling_rate)
        print("done with drift-aware merge")
        times_updated = aligned_spike_train3[:,0][kept]
        
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
        max_shift=20,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        order_units_by_z=True,
        geom=geom,
        sort_by_time=False,
    )
    order = order[reorder]
    assert np.array_equal(order, np.arange(len(order)))

    print("Save merge...")
    np.save(outdir / "merge_st.npy", aligned_spike_train4)
    np.save(outdir / "merge_templates.npy", templates)
    np.save(outdir / "merge_order.npy", order)

    return aligned_spike_train4, templates, order

def new_merge_split_ensemble(
    spike_train,
    n_channels,
    raw_bin,
    sub_h5,
    geom,
    outdir,
    num_ensemble=5,
    ensemble_percent=.8,
    n_workers=1,
    merge_resid_threshold=2.0,
    threshold_diptest=0.5,
    relocated=False,
    trough_offset=42,
    spike_length_samples=121,
    # extra_pc_split=True,
    # pc_only=False,
    # exp_split=False,
    load_split=False,
    split_kwargs=None,
):
    assert ensemble_percent <= 1.0
    outdir.mkdir(exist_ok=True, parents=True)
    split_kwargs = {} if split_kwargs is None else split_kwargs
    if not load_split:
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
        indices = np.arange(len(aligned_spike_train))
        random_seeds = list(range(num_ensemble))
        label_list = []
        spike_time_subset_list = []
        prev_last_label = 0
        for i in range(num_ensemble):
            np.random.seed(random_seeds[i])
            indices_subset = np.random.choice(indices, size=int(len(indices)*(1-ensemble_percent)), replace=False)
            indices_subset = np.sort(indices_subset)
            spike_train_subset = aligned_spike_train.copy()
            spike_train_subset[indices_subset,1] = -1
            orig_labels = spike_train_subset[:, 1].copy()

            new_labels = before_deconv_merge_split.split_clusters(
                spike_train_subset[:, 1],
                raw_bin,
                sub_h5,
                n_workers=n_workers,
                relocated=relocated,
                **split_kwargs,
            )
            
            (
                aligned_spike_train2,
                order,
                templates,
                template_shifts,
            ) = spike_train_utils.clean_align_and_get_templates(
                np.vstack((spike_train_subset[:, 0], new_labels)).T,
                n_channels,
                raw_bin,
                trough_offset=trough_offset,
                spike_length_samples=spike_length_samples,
                order_units_by_z=True,
                geom=geom,
            )
            assert (order == np.arange(len(order))).all()
            
            kept = aligned_spike_train2[:, 1] >= 0
            times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
                aligned_spike_train2[kept],
                geom,
                raw_bin,
                templates.ptp(1).argmax(1),
                merge_resid_threshold=merge_resid_threshold,
                trough_offset=trough_offset,
                spike_length_samples=spike_length_samples,
            )
        
            ####MERGE HERE
            aligned_spike_train2[kept, 0] = times_updated
            aligned_spike_train2[kept, 1] = labels_updated
            
            aligned_spike_train2[kept, 1] = aligned_spike_train2[kept, 1] + prev_last_label
            prev_last_label = np.unique(aligned_spike_train2[kept, 1])[-1] + 1
            
            label_list.append(aligned_spike_train2[kept, 1])
            spike_time_subset_list.append(aligned_spike_train2[kept, 0])
            
            untriaged_orig = orig_labels >= 0
            # untriaged_orig_a = orig_labels_a >= 0
            untriaged_now = aligned_spike_train2[kept, 1] >= 0
            
            print(f"Split total kept: {(untriaged_now.sum() / untriaged_orig.sum())=}")
            
            
        labels_cat = np.concatenate(label_list)
        spike_time_cat = np.concatenate(spike_time_subset_list)
        time_order = np.argsort(spike_time_cat)
        labels_cat = labels_cat[time_order]
        spike_time_cat = spike_time_cat[time_order]
        (
            aligned_spike_train3,
            order,
            templates,
            template_shifts,
        ) = spike_train_utils.clean_align_and_get_templates(
            np.vstack((spike_time_cat, labels_cat)).T,
            n_channels,
            raw_bin,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            order_units_by_z=True,
            geom=geom,
        )
        assert (order == np.arange(len(order))).all()

        print("Save split...")
        np.save(outdir / "split_st.npy", aligned_spike_train3)
        np.save(outdir / "split_templates.npy", templates)
        np.save(outdir / "split_order.npy", order)
    else:
        aligned_spike_train3 = np.load(outdir / "split_st.npy")
        templates = np.load(outdir / "split_templates.npy")
        order = np.load(outdir / "split_order.npy")
    
#     aligned_times, new_labels = before_deconv_merge_split.merge_clusters(
#         sub_h5,
#         raw_bin,
#         aligned_spike_train2[:, 1],
#         templates,
#         relocated=relocated,
#         trough_offset=trough_offset,
#         n_jobs=n_workers,
#         threshold_diptest=threshold_diptest,
#     )

#     (
#         aligned_spike_train3,
#         order,
#         templates,
#         template_shifts,
#     ) = spike_train_utils.clean_align_and_get_templates(
#         np.c_[aligned_spike_train2[:, 0], new_labels],
#         n_channels,
#         raw_bin,
#         max_shift=20,
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )

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
        max_shift=20,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        order_units_by_z=True,
        remove_double_counted=True,
        geom=geom,
    )
    order = order[reorder]

    print("Save merge...")
    np.save(outdir / "merge_st.npy", aligned_spike_train4)
    np.save(outdir / "merge_templates.npy", templates)
    np.save(outdir / "merge_order.npy", order)

    return aligned_spike_train4, templates, order
