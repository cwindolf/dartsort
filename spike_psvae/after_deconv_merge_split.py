import hdbscan
import numpy as np

from spike_psvae.isocut5 import isocut5 as isocut
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from spike_psvae import pre_deconv_merge_split, cluster_utils
from spike_psvae.spikeio import read_waveforms
from spike_psvae.subtract import make_contiguous_channel_index
from spike_psvae.pyks_ccg import ccg_metrics
from tqdm.auto import tqdm, trange


def split(
    labels_deconv,
    deconv_extractor,
    order=None,
    batch_size=1000,
    n_chans_split=10,
    min_cluster_size=25,
    min_samples=25,
    pc_split_rank=5,
    ptp_threshold=4,
    wfs_kind="denoised",
):
    next_label = labels_deconv.max() + 1
    n_channels = deconv_extractor.channel_index.shape[0]
    split_channel_index = make_contiguous_channel_index(
        n_channels, n_neighbors=n_chans_split
    )

    if order is None:
        order = np.arange(len(labels_deconv))

    for cluster_id in tqdm(np.unique(labels_deconv), desc="Split"):
        which = np.flatnonzero(labels_deconv == cluster_id)
        # np.flatnonzero(np.logical_and(
        #     labels_deconv == cluster_id, maxptps > ptp_threshold
        # ))

        # too small to split?
        if len(which) <= min_cluster_size:
            continue

        # indexing logic
        which_load = order[which]
        sort = np.argsort(which_load)
        which_load = which_load[sort]
        which = which[sort]

        # load denoised waveforms on n_chans_split channels
        wfs_split = deconv_extractor.get_waveforms(
            which_load, channel_index=split_channel_index, kind=wfs_kind
        )

        pcs_cluster = PCA(pc_split_rank).fit_transform(
            wfs_split.reshape(wfs_split.shape[0], -1)
        )
        clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
        clusterer.fit(pcs_cluster)

        if len(np.unique(clusterer.labels_)) > 1:
            labels_deconv[which[clusterer.labels_ == -1]] = -1
            for i in np.setdiff1d(np.unique(clusterer.labels_), [-1, 0]):
                labels_deconv[which[clusterer.labels_ == i]] = next_label
                next_label += 1

    return labels_deconv


def get_templates_com(templates, geom, n_channels=12):
    x_z_templates = np.zeros((templates.shape[0], 2))
    n_chan_half = n_channels // 2
    n_chan_total = geom.shape[0]
    for i in range(templates.shape[0]):
        mc = templates[i].ptp(0).argmax()
        mc = mc - mc % 2
        mc = max(min(n_chan_total - n_chan_half, mc), n_chan_half)
        x_z_templates[i, 0] = (
            templates[i].ptp(0)[mc - n_chan_half : mc + n_chan_half]
            * geom[mc - n_chan_half : mc + n_chan_half, 0]
        ).sum() / templates[i].ptp(0)[
            mc - n_chan_half : mc + n_chan_half
        ].sum()
        x_z_templates[i, 1] = (
            templates[i].ptp(0)[mc - n_chan_half : mc + n_chan_half]
            * geom[mc - n_chan_half : mc + n_chan_half, 1]
        ).sum() / templates[i].ptp(0)[
            mc - n_chan_half : mc + n_chan_half
        ].sum()
    return x_z_templates


def check_merge(
    unit_reference,
    unit_bis_reference,
    template_pair_shift,
    dist_argsort,
    reference_units,
    templates,
    n_spikes_templates,
    labels_updated,
    deconv_extractor,
    tpca,
    merge_channel_index,
    order=None,
    max_spikes=500,
    threshold_diptest=1.0,
    ptp_threshold=4,
    mc_diff_max=3,
    wfs_kind="cleaned",
    rg=None,
):
    if unit_reference == unit_bis_reference:
        return False, unit_bis_reference, 0

    unit_mc = templates[unit_reference].ptp(0).argmax()
    unit_bis_mc = templates[unit_bis_reference].ptp(0).argmax()
    mc_diff = np.abs(unit_mc - unit_bis_mc)
    unit_ptp = templates[unit_reference].ptp(0).max()
    unit_bis_ptp = templates[unit_bis_reference].ptp(0).max()
    ptp_diff = np.abs(
        templates[unit_bis_reference].ptp(0).max()
        - templates[unit_reference].ptp(0).max()
    )
    if mc_diff >= mc_diff_max:
        return False, unit_bis_reference, 0

    if order is None:
        order = np.arange(len(labels_updated))

    # ALIGN BASED ON MAX PTP TEMPLATE MC
    # the template with the larger MC is not shifted, so
    # we set unit_shifted to be the unit with smaller ptp
    unit_shifted = (
        unit_reference if unit_ptp <= unit_bis_ptp else unit_bis_reference
    )
    # we will load wfs on the same subset of channels, using this
    # as the maxchan
    mc = unit_mc if unit_ptp <= unit_bis_ptp else unit_bis_mc
    # template_pair_shift is unit argmin - unit_bis argmin
    # this ensures it has the same sign as before
    two_units_shift = (
        1 if unit_shifted == unit_reference else -1
    ) * template_pair_shift
    # print(mc, unit_mc, unit_bis_mc)
    # print(merge_channel_index[mc], deconv_extractor.channel_index[unit_mc], deconv_extractor.channel_index[unit_bis_mc])

    n_wfs_max = int(
        min(
            max_spikes,
            n_spikes_templates[unit_reference],
            n_spikes_templates[unit_bis_reference],
        )
    )
    which = order[np.flatnonzero(labels_updated == unit_reference)]
    which.sort()
    which_bis = order[np.flatnonzero(labels_updated == unit_bis_reference)]
    which_bis.sort()
    #     np.logical_and(
    #         maxptps > ptp_threshold,
    #         labels_updated == unit_reference,
    #     )
    # )

    if len(which) < 2 or len(which_bis) < 2:
        return False, unit_bis_reference, 0

    if len(which) > n_wfs_max:
        idx = rg.choice(np.arange(len(which)), n_wfs_max, replace=False)
        idx.sort()
    else:
        idx = np.arange(len(which))

    # load waveforms on n_chan_merge channels
    # print("load")
    wfs_merge_ref = deconv_extractor.get_waveforms(
        which[idx],
        kind=wfs_kind,
        channels=merge_channel_index[mc],
    )
    if np.isnan(wfs_merge_ref).any():
        return False, unit_bis_reference, 0

    if len(which_bis) > n_wfs_max:
        idx = rg.choice(np.arange(len(which_bis)), n_wfs_max, replace=False)
        idx.sort()
    else:
        idx = np.arange(len(which_bis))

    # load waveforms on n_chan_merge channels
    # print("load bis")
    wfs_merge_ref_bis = deconv_extractor.get_waveforms(
        which_bis[idx],
        kind=wfs_kind,
        channels=merge_channel_index[mc],
    )
    if np.isnan(wfs_merge_ref_bis).any():
        return False, unit_bis_reference, 0

    # shift according to template trough difference
    if unit_shifted == unit_reference and two_units_shift > 0:
        wfs_merge_ref = wfs_merge_ref[:, two_units_shift:, :]
        wfs_merge_ref_bis = wfs_merge_ref_bis[:, :-two_units_shift, :]
    elif unit_shifted == unit_reference and two_units_shift < 0:
        wfs_merge_ref = wfs_merge_ref[:, :two_units_shift, :]
        wfs_merge_ref_bis = wfs_merge_ref_bis[:, -two_units_shift:, :]
    elif unit_shifted == unit_bis_reference and two_units_shift > 0:
        wfs_merge_ref = wfs_merge_ref[:, :-two_units_shift, :]
        wfs_merge_ref_bis = wfs_merge_ref_bis[:, two_units_shift:, :]
    elif unit_shifted == unit_bis_reference and two_units_shift < 0:
        wfs_merge_ref = wfs_merge_ref[:, -two_units_shift:, :]
        wfs_merge_ref_bis = wfs_merge_ref_bis[:, :two_units_shift, :]

    # it's possible that a waveform could have been skipped,
    n_wfs_max = int(
        min(max_spikes, wfs_merge_ref.shape[0], wfs_merge_ref_bis.shape[0])
    )
    idx_ref = rg.choice(wfs_merge_ref.shape[0], n_wfs_max, replace=False)
    idx_ref_bis = rg.choice(
        wfs_merge_ref_bis.shape[0],
        n_wfs_max,
        replace=False,
    )

    wfs_diptest = np.concatenate(
        (
            wfs_merge_ref[idx_ref],
            wfs_merge_ref_bis[idx_ref_bis],
        )
    )
    N, T, C = wfs_diptest.shape
    wfs_diptest = wfs_diptest.transpose(0, 2, 1).reshape(N * C, T)

    wfs_diptest = tpca.fit_transform(wfs_diptest)
    wfs_diptest = (
        wfs_diptest.reshape(N, C, tpca.n_components)
        .transpose(0, 2, 1)
        .reshape((N, C * tpca.n_components))
    )

    labels_diptest = np.zeros(2 * n_wfs_max)
    labels_diptest[:n_wfs_max] = 1

    lda_model = LDA(n_components=1)
    lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
    value_dpt, cut_calue = isocut(lda_comps[:, 0])

    # do we ever reach the second criterion?
    # can we remove that or not clause?
    # or, just throw away small
    if n_wfs_max >= 20 or not (mc_diff > 0 or ptp_diff > 1):
        if value_dpt < threshold_diptest and np.abs(two_units_shift) < 2:
            shift = (
                -two_units_shift
                if unit_shifted == unit_bis_reference
                else two_units_shift
            )
            return True, unit_bis_reference, shift

    return False, unit_bis_reference, 0


def merge(
    labels,
    templates,
    deconv_extractor,
    geom,
    order=None,
    n_chan_merge=10,
    tpca=PCA(5),
    n_temp=10,
    distance_threshold=3.0,
    always_merge_threshold=0.0,  # haven't picked a good default yet
    threshold_diptest=1.0,
    ptp_threshold=4.0,
    max_spikes=500,
    wfs_kind="cleaned",
    isi_veto=False,
    spike_times=None,
    contam_ratio_threshold=0.2,
    contam_alpha=0.05,
    isi_nbins=500,
    isi_bin_nsamples=30,
    seed=0,
):
    """
    merge is applied on spikes with ptp > ptp_threshold only
    """
    rg = np.random.default_rng(seed)

    merge_channel_index = make_contiguous_channel_index(
        deconv_extractor.channel_index.shape[0], n_neighbors=n_chan_merge
    )

    labels_updated = labels.copy()
    n_templates = templates.shape[0]
    n_spikes_templates = pre_deconv_merge_split.get_n_spikes_templates(
        n_templates, labels
    )
    x_z_templates = get_templates_com(templates, geom)
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = pre_deconv_merge_split.get_proposed_pairs(
        n_templates, templates, x_z_templates, n_temp, shifts=[-2, -1, 0, 1, 2]
    )
    reference_units = np.setdiff1d(np.unique(labels), [-1])
    print(n_templates, reference_units.shape)

    # get template shifts
    template_mcs = templates.ptp(1).argmax(1)
    template_mctraces = templates[np.arange(len(templates)), :, template_mcs]
    template_troughs = template_mctraces.argmin(1)
    # pair_shifts[i, j] = template i trough time - template j trough time
    pair_shifts = template_troughs[:, None] - template_troughs[None, :]

    for unit in trange(n_templates, desc="merge"):
        unit_reference = reference_units[unit]
        to_be_merged = [unit_reference]
        merge_shifts = [0]
        is_merged = False

        for j in range(n_temp):
            unit_bis = dist_argsort[unit, j]
            unit_bis_reference = reference_units[unit_bis]
            if dist_template[unit, j] < distance_threshold:
                is_merged_bis, unit_bis_reference, shift = check_merge(
                    unit_reference,
                    unit_bis_reference,
                    pair_shifts[unit_reference, unit_bis_reference],
                    dist_argsort,
                    reference_units,
                    templates,
                    n_spikes_templates,
                    labels_updated,
                    deconv_extractor,
                    tpca,
                    merge_channel_index,
                    order=order,
                    max_spikes=max_spikes,
                    threshold_diptest=threshold_diptest,
                    ptp_threshold=ptp_threshold,
                    wfs_kind=wfs_kind,
                    rg=rg,
                )

                # check isi violation
                # if is_merged:
                #     print("will try", unit_reference, unit_bis_reference)
                # else:
                #     print("nope for", unit_reference, unit_bis_reference)

                if isi_veto and is_merged_bis:
                    st1 = spike_times[labels == unit_reference]
                    st2 = spike_times[labels == unit_bis_reference]
                    contam_ratio, p_value = ccg_metrics(
                        st1, st2, isi_nbins, isi_bin_nsamples
                    )
                    contam_ok = contam_ratio < contam_ratio_threshold
                    contam_sig = p_value < contam_alpha
                    is_merged_bis = contam_ok and contam_sig
                    if not is_merged_bis:
                        print(
                            "ISI prevented merge with",
                            contam_ratio,
                            p_value,
                            st1.shape,
                            st2.shape,
                        )
                    else:
                        print(
                            "ISI allowed merge with",
                            contam_ratio,
                            p_value,
                            st1.shape,
                            st2.shape,
                        )

                    # ccg1 = ccg(st1, st1, 500, 30)
                    # contam_ratio1, p_value1 = ccg_metrics(
                    #     st1, st1, isi_nbins, isi_bin_nsamples
                    # )
                    # ccg2 = ccg(st2, st2, 500, 30)
                    # contam_ratio2, p_value2 = ccg_metrics(
                    #     st2, st2, isi_nbins, isi_bin_nsamples
                    # )
                    # ccg12 = ccg(st1, st2, 500, 30)
                    # ccg1[isi_nbins] = ccg2[isi_nbins] = ccg12[isi_nbins] = 0
                    # import matplotlib.pyplot as plt
                    # fig, axes = plt.subplots(3, 1, figsize=(8, 12))
                    # axes[0].step(np.arange(-isi_nbins, isi_nbins + 1), ccg1)
                    # axes[0].set_title(f"{unit_reference} self-ccg. {contam_ratio1=}, {p_value1=}")
                    # axes[1].step(np.arange(-isi_nbins, isi_nbins + 1), ccg2)
                    # axes[1].set_title(f"{unit_bis_reference} self-ccg. {contam_ratio2=}, {p_value2=}")
                    # axes[2].step(np.arange(-isi_nbins, isi_nbins + 1), ccg12)
                    # axes[2].set_title(f"ccg {contam_ratio=}, {p_value=}")
                    # axes[2].set_xlabel("1ms ccg bins")
                    # plt.show()
                    # plt.close(fig)

                is_merged |= is_merged_bis
                if is_merged_bis:
                    to_be_merged.append(unit_bis_reference)
                    merge_shifts.append(shift)

        if is_merged:
            n_total_spikes = 0
            for unit_merged in np.unique(np.asarray(to_be_merged)):
                n_total_spikes += n_spikes_templates[unit_merged]

            new_reference_unit = to_be_merged[0]

            templates[new_reference_unit] = (
                n_spikes_templates[new_reference_unit]
                * templates[new_reference_unit]
                / n_total_spikes
            )
            cmp = 1
            for unit_merged in to_be_merged[1:]:
                shift_ = merge_shifts[cmp]
                templates[new_reference_unit] += (
                    n_spikes_templates[unit_merged]
                    * np.roll(templates[unit_merged], shift_, axis=0)
                    / n_total_spikes
                )
                n_spikes_templates[new_reference_unit] += n_spikes_templates[
                    unit_merged
                ]
                n_spikes_templates[unit_merged] = 0
                labels_updated[
                    labels_updated == unit_merged
                ] = new_reference_unit
                reference_units[
                    reference_units == unit_merged
                ] = new_reference_unit
                cmp += 1

    return labels_updated


def clean_big_clusters(
    templates,
    spike_train,
    ptps,
    raw_bin,
    geom,
    min_ptp=6.0,
    split_diff=2.0,
    seed=0,
    reducer=np.median,
    min_size_split=25,
):
    """This operates on spike_train in place."""
    # TODO:
    # it's not possible to load all waveforms as is currently done
    # rather, we should do something like, sort the ptps,
    # then load N wfs above/below
    # or, uniformly subsample e.g. 1000 spikes according to PTP,
    # and use those...
    # and, what should happen when there aren't many spikes?
    n_temp_cleaned = 0
    next_label = templates.shape[0]
    # rg = np.random.default_rng(seed)
    # orig_ids = {}

    for unit in trange(templates.shape[0], desc="clean big"):
        mc = templates[unit].ptp(0).argmax()
        template_mc_trace = templates[unit, :, mc]
        if template_mc_trace.ptp() < min_ptp:
            continue

        spikes_in_unit = np.flatnonzero(spike_train[:, 1] == unit)
        spike_times_unit = spike_train[spikes_in_unit, 0]
        wfs_unit = read_waveforms(
            spike_times_unit, raw_bin, geom.shape[0], channels=[mc]
        )[0][:, :, 0]

        ptp_sort_idx = wfs_unit.ptp(1).argsort()
        wfs_unit = wfs_unit[ptp_sort_idx]
        lower = max(min_size_split, int(wfs_unit.shape[0] * 0.05))
        upper = min(
            spikes_in_unit.size - min_size_split, int(wfs_unit.shape[0] * 0.95)
        )

        if lower >= upper:
            continue

        max_diff = 0
        max_diff_N = 0
        for n in range(lower, upper):
            # Denoise templates?
            temp_1 = reducer(wfs_unit[:n], axis=0)
            temp_2 = reducer(wfs_unit[n:], axis=0)
            diff = np.abs(temp_1 - temp_2).max()
            if diff > max_diff:
                max_diff = diff
                max_diff_N = n

        if max_diff < split_diff:
            continue

        temp_1 = reducer(wfs_unit[:max_diff_N], axis=0)
        temp_2 = reducer(wfs_unit[max_diff_N:], axis=0)

        if (
            np.abs(temp_1 - template_mc_trace).max()
            > np.abs(temp_2 - template_mc_trace).max()
        ):
            which = spikes_in_unit[ptp_sort_idx[:max_diff_N]]
            spike_train[which, 1] = next_label
        else:
            which = spikes_in_unit[ptp_sort_idx[max_diff_N:]]
            spike_train[which, 1] = next_label
        # orig_ids[next_label] = unit

        n_temp_cleaned += 1
        next_label += 1

    # new_id_to_old_id = np.zeros(next_label, dtype=int)
    # for new, old in orig_ids.items():
    #     new_id_to_old_id[new] = old

    # new_id_to_old_id = np.zeros(next_label, dtype=int)
    # for new, old in orig_ids.items():
    #     new_id_to_old_id[new] = old

    return n_temp_cleaned


def remove_oversplits(templates, spike_train, min_ptp=4.0, max_diff=2.0):
    """This will modify spike_train"""
    # remove oversplits according to max abs norm
    for unit in trange(templates.shape[0] - 1, desc="max abs merge"):
        if templates[unit].ptp(0).max(0) >= min_ptp:
            max_vec = (
                np.abs(templates[unit, :, :] - templates[unit + 1 :])
                .max(1)
                .max(1)
            )
            if max_vec.min() < max_diff:
                idx_units_to_change = (
                    unit + 1 + np.where(max_vec < max_diff)[0]
                )
                in_change = np.isin(spike_train[:, 1], idx_units_to_change)
                assert in_change.shape == spike_train[:, 0].shape
                spike_train[in_change, 1] = unit
                templates[idx_units_to_change] = templates[unit]

    # make labels contiguous and get corresponding templates
    spike_train[:, 1], orig_uniq = cluster_utils.make_labels_contiguous(
        spike_train[:, 1], return_orig_unit_labels=True
    )
    templates = templates[orig_uniq]

    return spike_train, templates
