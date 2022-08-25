import numpy as np
import h5py
from sklearn.decomposition import PCA

from . import after_deconv_merge_split, spike_train_utils, extractors

from .hybrid_analysis import Sorting
import matplotlib.pyplot as plt

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
        templates, spike_train, deconv_extractor.ptp[order], bin_file, geom, reducer=reducer,
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
        # isi_veto=True,
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
        templates, spike_train, deconv_extractor.ptp[order], bin_file, geom, reducer=reducer
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
        print(f"deconv result shapes {maxptps.shape=} {x.shape=} {z_abs.shape=}")
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
    ) = after_deconv_merge_split.remove_oversplits(
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
    )
    order = order[reorder]

    u, c = np.unique(spike_train[:, 1], return_counts=True)
    print(u.max() + 1, u.size, (c > 25).sum(), c[c > 25].sum())

    return spike_train, order, templates
