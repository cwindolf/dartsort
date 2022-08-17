import h5py
import hdbscan
import numpy as np

from spike_psvae.isocut5 import isocut5 as isocut
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from spike_psvae import pre_deconv_merge_split, cluster_utils
from spike_psvae.spikeio import read_waveforms
from spike_psvae.pyks_ccg import ccg_metrics
from tqdm.auto import tqdm, trange


def split(
    labels_deconv,
    templates,
    firstchans,
    path_denoised_wfs_h5,
    order=None,
    batch_size=1000,
    n_chans_split=10,
    min_cluster_size=25,
    min_samples=25,
    pc_split_rank=5,
    ptp_threshold=4,
    wfs_key="denoised_waveforms",
):
    cmp = labels_deconv.max() + 1

    if order is None:
        order = np.arange(len(labels_deconv))

    for cluster_id in tqdm(np.unique(labels_deconv)):
        which = np.flatnonzero(
            labels_deconv
            == cluster_id
            #             np.logical_and(
            #                 labels_deconv == cluster_id, maxptps > ptp_threshold
            #             )
        )
        which_load = order[which]
        if len(which) > min_cluster_size:
            with h5py.File(path_denoised_wfs_h5, "r") as h5:
                batch_wfs = np.empty(
                    (len(which), *h5[wfs_key].shape[1:]),
                    dtype=h5[wfs_key].dtype,
                )
                h5_wfs = h5[wfs_key]
                for batch_start in range(0, len(which), 1000):
                    batch_wfs[batch_start : batch_start + batch_size] = h5_wfs[
                        which_load[batch_start : batch_start + batch_size]
                    ]

            C = batch_wfs.shape[2]
            if C < n_chans_split:
                n_chans_split = C

            maxchan = templates[cluster_id].ptp(0).argmax()
            firstchan_maxchan = maxchan - firstchans[which_load]
            firstchan_maxchan = np.maximum(
                firstchan_maxchan, n_chans_split // 2
            )
            firstchan_maxchan = np.minimum(
                firstchan_maxchan, C - n_chans_split // 2
            )
            firstchan_maxchan = firstchan_maxchan.astype("int")

            if len(np.unique(firstchan_maxchan)) <= 1:
                wfs_split = batch_wfs[
                    :,
                    :,
                    firstchan_maxchan[0]
                    - n_chans_split // 2 : firstchan_maxchan[0]
                    + n_chans_split // 2,
                ]
            else:
                wfs_split = np.zeros(
                    (batch_wfs.shape[0], batch_wfs.shape[1], n_chans_split)
                )
                for j in range(batch_wfs.shape[0]):
                    wfs_split[j] = batch_wfs[
                        j,
                        :,
                        firstchan_maxchan[j]
                        - n_chans_split // 2 : firstchan_maxchan[j]
                        + n_chans_split // 2,
                    ]

            pcs_cluster = PCA(pc_split_rank).fit_transform(
                wfs_split.reshape(wfs_split.shape[0], -1)
            )
            clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
            clusterer.fit(pcs_cluster)

            if len(np.unique(clusterer.labels_)) > 1:
                labels_deconv[which[clusterer.labels_ == -1]] = -1
                for i in np.setdiff1d(np.unique(clusterer.labels_), [-1, 0]):
                    labels_deconv[which[clusterer.labels_ == i]] = cmp
                    cmp += 1

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
    path_cleaned_wfs_h5,
    labels_updated,
    firstchans,
    tpca,
    order=None,
    n_chan_merge=10,
    max_spikes=500,
    threshold_diptest=1.0,
    ptp_threshold=4,
    wfs_key="cleaned_waveforms",
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
    if mc_diff >= 3:
        return False, unit_bis_reference, 0

    if order is None:
        order = np.arange(len(labels_updated))

    # ALIGN BASED ON MAX PTP TEMPLATE MC
    # the template with the larger MC is not shifted, so
    # we set unit_shifted to be the unit with smaller ptp
    unit_shifted = (
        unit_reference if unit_ptp <= unit_bis_ptp else unit_bis_reference
    )
    mc = unit_mc if unit_ptp <= unit_bis_ptp else unit_bis_mc
    # template_pair_shift is unit argmin - unit_bis argmin
    # this ensures it has the same sign as before
    two_units_shift = (
        1 if unit_shifted == unit_reference else -1
    ) * template_pair_shift

    n_wfs_max = int(
        min(
            max_spikes,
            min(
                n_spikes_templates[unit_reference],
                n_spikes_templates[unit_bis_reference],
            ),
        )
    )
    which = order[np.flatnonzero(labels_updated == unit_reference)]
    #         np.logical_and(
    #             maxptps > ptp_threshold,
    #             labels_updated == unit_reference,
    #         )
    #     )

    if len(which) < 2:
        return False, unit_bis_reference, 0

    if len(which) > n_wfs_max:
        idx = np.random.choice(np.arange(len(which)), n_wfs_max, replace=False)
        idx.sort()
    else:
        idx = np.arange(len(which))

    with h5py.File(path_cleaned_wfs_h5, "r") as h5:
        waveforms_ref = np.empty(
            (n_wfs_max, *h5[wfs_key].shape[1:]),
            dtype=h5[wfs_key].dtype,
        )
        waveforms_ref = h5[wfs_key][which[idx]]

    C = waveforms_ref.shape[2]
    if C < n_chan_merge:
        n_chan_merge = C
    firstchan_maxchan = mc - firstchans[which[idx]]

    firstchan_maxchan = np.maximum(firstchan_maxchan, n_chan_merge // 2)
    firstchan_maxchan = np.minimum(firstchan_maxchan, C - n_chan_merge // 2)

    firstchan_maxchan = firstchan_maxchan.astype("int")

    if len(np.unique(firstchan_maxchan)) <= 1:
        wfs_merge_ref = waveforms_ref[
            :,
            :,
            firstchan_maxchan[0]
            - n_chan_merge // 2 : firstchan_maxchan[0]
            + n_chan_merge // 2,
        ]
    else:
        wfs_merge_ref = np.zeros(
            (
                waveforms_ref.shape[0],
                waveforms_ref.shape[1],
                n_chan_merge,
            )
        )
        for j in range(waveforms_ref.shape[0]):
            wfs_merge_ref[j] = waveforms_ref[
                j,
                :,
                firstchan_maxchan[j]
                - n_chan_merge // 2 : firstchan_maxchan[j]
                + n_chan_merge // 2,
            ]

    which = order[np.flatnonzero(labels_updated == unit_bis_reference)]
    #         np.logical_and(
    #             maxptps > ptp_threshold,
    #             labels_updated == unit_bis_reference,
    #         )
    #     )

    if len(which) < 2:
        return False, unit_bis_reference, 0

    if len(which) > n_wfs_max:
        idx = np.random.choice(np.arange(len(which)), n_wfs_max, replace=False)
        idx.sort()
    else:
        idx = np.arange(len(which))

    firstchan_maxchan = mc - firstchans[which[idx]]

    firstchan_maxchan = np.maximum(firstchan_maxchan, n_chan_merge // 2)
    firstchan_maxchan = np.minimum(firstchan_maxchan, C - n_chan_merge // 2)
    firstchan_maxchan = firstchan_maxchan.astype("int")

    with h5py.File(path_cleaned_wfs_h5, "r") as h5:
        waveforms_ref_bis = np.empty(
            (n_wfs_max, *h5[wfs_key].shape[1:]),
            dtype=h5[wfs_key].dtype,
        )
        waveforms_ref_bis = h5[wfs_key][which[idx]]
    # print(f"{firstchan_maxchan=}, {mc=}, {len(which)=}, {firstchans.shape=}, {firstchan_maxchan.shape=}")
    if len(np.unique(firstchan_maxchan)) <= 1:
        wfs_merge_ref_bis = waveforms_ref_bis[
            :,
            :,
            firstchan_maxchan[0]
            - n_chan_merge // 2 : firstchan_maxchan[0]
            + n_chan_merge // 2,
        ]
    else:
        wfs_merge_ref_bis = np.zeros(
            (
                waveforms_ref_bis.shape[0],
                waveforms_ref_bis.shape[1],
                n_chan_merge,
            )
        )
        for j in range(waveforms_ref_bis.shape[0]):
            wfs_merge_ref_bis[j] = waveforms_ref_bis[
                j,
                :,
                firstchan_maxchan[j]
                - n_chan_merge // 2 : firstchan_maxchan[j]
                + n_chan_merge // 2,
            ]

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

    n_wfs_max = int(
        min(
            max_spikes,
            min(
                wfs_merge_ref.shape[0],
                wfs_merge_ref_bis.shape[0],
            ),
        )
    )

    idx_ref = np.random.choice(
        wfs_merge_ref.shape[0], n_wfs_max, replace=False
    )
    idx_ref_bis = np.random.choice(
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
    path_cleaned_wfs_h5,
    firstchans,
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
    wfs_key="cleaned_waveforms",
    isi_veto=False,
    spike_times=None,
    contam_ratio_threshold=0.2,
    contam_alpha=0.05,
    isi_nbins=500,
    isi_bin_nsamples=30,
):
    """
    merge is applied on spikes with ptp > ptp_threshold only
    """

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
                    path_cleaned_wfs_h5,
                    labels_updated,
                    firstchans,
                    tpca,
                    order=order,
                    n_chan_merge=n_chan_merge,
                    max_spikes=max_spikes,
                    threshold_diptest=threshold_diptest,
                    ptp_threshold=ptp_threshold,
                    wfs_key=wfs_key,
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
    max_samples=500,
    min_size_split=25,
    seed=0,
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
    rg = np.random.default_rng(seed)
    for unit in trange(templates.shape[0], desc="clean big"):
        mc = templates[unit].ptp(0).argmax()
        template_mc_trace = templates[unit, :, mc]

        if template_mc_trace.ptp() < min_ptp:
            continue

        in_unit = np.flatnonzero(spike_train[:, 1] == unit)
        if in_unit.size <= 2 * min_size_split:
            # we won't split if smaller than this
            continue
        n_samples = min(max_samples, in_unit.size)

        # pick random wfs
        choices = rg.choice(in_unit.size, size=n_samples, replace=False)
        spike_times_unit = spike_train[in_unit[choices], 0]
        wfs_unit, skipped_idx = read_waveforms(
            spike_times_unit, raw_bin, geom.shape[0], channels=[mc]
        )
        assert wfs_unit.shape[-1] == 1
        assert not skipped_idx.size
        wfs_unit = wfs_unit[:, :, 0]

        # ptp order
        ptps_unit = ptps[in_unit]
        ptps_choice = ptps_unit[choices]
        ptps_sort = np.argsort(ptps_choice)
        wfs_sort = wfs_unit[ptps_sort]

        lower = int(max(np.ceil(in_unit.size * 0.05), min_size_split))
        upper = int(
            min(np.floor(in_unit.size * 0.95), in_unit.size - min_size_split)
        )
        if lower >= upper:
            continue

        max_diff = 0
        max_diff_ix = 0
        for n in range(lower, upper):
            # Denoise templates?
            temp_1 = np.median(wfs_sort[:n], axis=0)
            temp_2 = np.median(wfs_sort[n:], axis=0)
            diff = np.abs(temp_1 - temp_2).max()
            if diff > max_diff:
                max_diff = diff
                max_diff_ix = n
        max_diff_ptp = 0.5 * (
            ptps_sort[max_diff_ix] + ptps_sort[max_diff_ix - 1]
        )

        if max_diff < split_diff:
            continue

        which_a = in_unit[ptps_unit <= max_diff_ptp]
        which_b = in_unit[ptps_unit > max_diff_ptp]

        temp_a = np.median(wfs_unit[:max_diff_ix], axis=0)
        temp_b = np.median(wfs_unit[max_diff_ix:], axis=0)
        temp_diff_a = np.abs(temp_a - template_mc_trace).max()
        temp_diff_b = np.abs(temp_b - template_mc_trace).max()

        if temp_diff_a < temp_diff_b:
            spike_train[which_b] = next_label
        else:
            spike_train[which_a] = next_label

        n_temp_cleaned += 1
        next_label += 1

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
        spike_train[:, 1], return_unique=True
    )
    templates = templates[orig_uniq]

    return spike_train, templates
