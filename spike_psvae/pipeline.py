import numpy as np
import h5py
from sklearn.decomposition import PCA

from . import after_deconv_merge_split, spike_train_utils


def post_deconv_split_step(
    deconv_dir,
    deconv_results_h5,
    bin_file,
    geom,
    clean_min_spikes=25,
):
    # -- load up the deconv result
    with h5py.File(deconv_results_h5) as f:
        maxptps = f["maxptps"][:]
        firstchans = f["first_channels"][:]

    spike_train = np.load(deconv_dir / "spike_train.npy")
    assert spike_train.shape == (*maxptps.shape, 2)
    templates = np.load(deconv_dir / "templates.npy")
    assert templates.shape[0] == spike_train[:, 1].max() + 1
    print("original")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.size, (c > 25).sum(), c[c > 25].sum())
    n_channels = templates.shape[2]
    assert n_channels == geom.shape[0]

    # deconv can produce an un-aligned spike train.
    # let's also get rid of units with few spikes.
    (
        spike_train,
        order,
        templates,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )

    # the initial split
    spike_train[:, 1] = after_deconv_merge_split.split(
        spike_train[:, 1],
        templates,
        firstchans,
        deconv_results_h5,
        order=order,
        wfs_key="denoised_waveforms",
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
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )
    order = order[reorder]

    # clean big
    after_deconv_merge_split.clean_big_clusters(
        templates, spike_train, maxptps[order], bin_file, geom
    )
    (
        spike_train,
        reorder,
        templates,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )
    order = order[reorder]

    print("B")
    print("sorted?", (spike_train[1:, 0] >= spike_train[:-1, 0]).all())
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
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
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
):
    n_channels = geom.shape[0]
    with h5py.File(deconv_results_h5) as f:
        maxptps = f["maxptps"][:]
        firstchans = f["first_channels"][:]

    spike_train = spike_train.copy()

    print("Just before merge...")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    spike_train[:, 1] = after_deconv_merge_split.merge(
        spike_train[:, 1],
        templates,
        deconv_results_h5,
        firstchans,
        geom,
        order=order,
        spike_times=spike_train[:, 0],
        n_chan_merge=10,
        wfs_key="cleaned_waveforms",
        tpca=PCA(6),
        # isi_veto=True,
    )
    (
        spike_train,
        reorder,
        templates,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
    )
    order = order[reorder]
    print("Just after merge...")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    print("Clean big ")
    n_cleaned = after_deconv_merge_split.clean_big_clusters(
        templates, spike_train, maxptps[order], bin_file, geom
    )
    print(f"{n_cleaned=}")
    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    (
        spike_train,
        reorder,
        templates,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        bin_file,
        min_n_spikes=clean_min_spikes,
        pbar=True,
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
        print(f"deconv result shapes {maxptps.shape=} {x.shape=} {z_abs.shape=}")
    spike_train = np.load(deconv2_dir / "spike_train.npy")
    print(deconv2_dir / "spike_train.npy", f"{spike_train.shape}")
    assert (spike_train.shape[0],) == maxptps.shape == firstchans.shape

    (
        spike_train,
        order,
        templates,
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
    ) = after_deconv_merge_split.remove_oversplits(
        templates, spike_train
    )
    (
        spike_train,
        reorder,
        templates,
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
