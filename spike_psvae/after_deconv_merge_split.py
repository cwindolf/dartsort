import h5py
import hdbscan
import numpy as np

from spike_psvae.isocut5 import isocut5 as isocut
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from spike_psvae import pre_deconv_merge_split, cluster_utils
from spike_psvae.deconvolve import read_waveforms
from tqdm.auto import tqdm, trange


def split(
    labels_deconv,
    templates,
    maxptps,
    firstchans,
    path_denoised_wfs_h5,
    batch_size=1000,
    n_chans_split=10,
    min_cluster_size=25,
    min_samples=25,
    pc_split_rank=5,
    ptp_threshold=4,
    wfs_key="denoised_waveforms",
):
    cmp = labels_deconv.max() + 1

    for cluster_id in tqdm(np.unique(labels_deconv)):
        which = np.flatnonzero(
            np.logical_and(
                maxptps > ptp_threshold, labels_deconv == cluster_id
            )
        )
        if len(which) > min_cluster_size:
            with h5py.File(path_denoised_wfs_h5, "r") as h5:
                batch_wfs = np.empty(
                    (len(which), *h5[wfs_key].shape[1:]),
                    dtype=h5[wfs_key].dtype,
                )
                h5_wfs = h5[wfs_key]
                for batch_start in range(0, len(which), 1000):
                    batch_wfs[batch_start : batch_start + batch_size] = h5_wfs[
                        which[batch_start : batch_start + batch_size]
                    ]

            C = batch_wfs.shape[2]
            if C < n_chans_split:
                n_chans_split = C

            maxchan = templates[cluster_id].ptp(0).argmax()
            firstchan_maxchan = maxchan - firstchans[which]
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
                for i in np.unique(clusterer.labels_)[2:]:
                    labels_deconv[which[clusterer.labels_ == i]] = cmp
                    cmp += 1

    return labels_deconv


def get_templates_com(templates, geom, n_channels=12):
    x_z_templates = np.zeros((n_templates, 2))
    n_chan_half = n_channels // 2
    n_chan_total = geom.shape[0]
    for i in range(n_templates):
        mc = templates[i].ptp(0).argmax()
        mc = mc - mc % 2
        mc = max(min(n_chan_total - n_chan_half, mc), n_chan_half)
        x_z_templates[i] = (
            templates[i].ptp(0)[mc - n_chan_half : mc + n_chan_half]
            * geom[mc - n_chan_half : mc + n_chan_half]
        ) / templates[i].ptp(0)[mc - n_chan_half : mc + n_chan_half].sum()
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
    maxptps,
    labels_updated,
    firstchans,
    tpca,
    n_chan_merge=10,
    max_spikes=500,
    threshold_diptest=0.1,
    threshold_ptp=4,
    wfs_key="cleaned_waveforms",
):
    if unit_reference == unit_bis_reference:
        return False, -1, 0

    unit_mc = templates[unit_reference].ptp(0).argmax()
    unit_bis_mc = templates[unit_bis_reference].ptp(0).argmax()
    mc_diff = np.abs(unit_mc - unit_bis_mc)
    unit_ptp = templates[unit_reference].ptp(0).max()
    unit_bis_ptp = templates[unit_bis_reference].ptp(0).max()
    ptp_diff = np.abs(
        templates[unit_bis_reference].ptp(0).max()
        - templates[unit_reference].ptp(0).max()
    )
    if mc_diff < 3:
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
        which = np.flatnonzero(
            np.logical_and(
                maxptps > threshold_ptp,
                labels_updated == unit_reference,
            )
        )
        if len(which) > n_wfs_max:
            idx = np.random.choice(
                np.arange(len(which)), n_wfs_max, replace=False
            )
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
        firstchan_maxchan = np.minimum(
            firstchan_maxchan, C - n_chan_merge // 2
        )

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

        which = np.flatnonzero(
            np.logical_and(
                maxptps > threshold_ptp,
                labels_updated == unit_bis_reference,
            )
        )

        if len(which) > n_wfs_max:
            idx = np.random.choice(
                np.arange(len(which)), n_wfs_max, replace=False
            )
            idx.sort()
        else:
            idx = np.arange(len(which))

        firstchan_maxchan = mc - firstchans[which[idx]]

        firstchan_maxchan = np.maximum(firstchan_maxchan, n_chan_merge // 2)
        firstchan_maxchan = np.minimum(
            firstchan_maxchan, C - n_chan_merge // 2
        )
        firstchan_maxchan = firstchan_maxchan.astype("int")

        with h5py.File(path_cleaned_wfs_h5, "r") as h5:
            waveforms_ref_bis = np.empty(
                (n_wfs_max, *h5[wfs_key].shape[1:]),
                dtype=h5[wfs_key].dtype,
            )
            waveforms_ref_bis = h5[wfs_key][which[idx]]

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

        if ~(n_wfs_max < 20 and (mc_diff > 0 or ptp_diff > 1)):
            if value_dpt < threshold_diptest and np.abs(two_units_shift) < 2:
                shift = (
                    -two_units_shift
                    if unit_shifted == unit_bis_reference
                    else two_units_shift
                )
                return True, unit_bis_reference, shift

    return False, -1, 0


def merge(
    labels,
    templates,
    path_cleaned_wfs_h5,
    xs,
    z_reg,
    maxptps,
    firstchans,
    n_chan_merge=10,
    tpca=PCA(8),
    n_temp=10,
    distance_threshold=3.0,
    always_merge_threshold=0.0,  # haven't picked a good default yet
    threshold_diptest=0.1,
    max_spikes=500,
    threshold_ptp=4,
    wfs_key="cleaned_waveforms",
):
    """
    merge is applied on spikes with ptp > threshold_ptp only
    """

    labels_updated = labels.copy()
    n_templates = templates.shape[0]
    n_spikes_templates = pre_deconv_merge_split.get_n_spikes_templates(
        n_templates, labels
    )
    x_z_templates = pre_deconv_merge_split.get_x_z_templates(
        n_templates, labels, xs, z_reg
    )
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = pre_deconv_merge_split.get_proposed_pairs(
        n_templates, templates, x_z_templates, n_temp
    )
    reference_units = np.setdiff1d(np.unique(labels), [-1])

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
                    maxptps,
                    labels_updated,
                    firstchans,
                    tpca,
                    n_chan_merge=n_chan_merge,
                    max_spikes=max_spikes,
                    threshold_diptest=threshold_diptest,
                    threshold_ptp=threshold_ptp,
                    wfs_key=wfs_key,
                )
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
    templates, spike_train, raw_bin, geom, min_ptp=6.0, split_diff=2.0
):
    """This operates on spike_train in place."""
    n_temp_cleaned = 0
    cmp = templates.shape[0]
    for unit in trange(templates.shape[0], desc="clean big"):
        mc = templates[unit].ptp(0).argmax()
        template_mc_trace = templates[unit, :, mc]
        if template_mc_trace.ptp() > min_ptp:
            spikes_in_unit = np.flatnonzero(spike_train[:, 1] == unit)
            spike_times_unit = spike_train[spikes_in_unit, 0]
            wfs_unit = read_waveforms(
                spike_times_unit, raw_bin, geom, channels=[mc]
            )[0][:, :, 0]

            ptp_sort_idx = wfs_unit.ptp(1).argsort()
            wfs_unit = wfs_unit[ptp_sort_idx]
            lower = int(wfs_unit.shape[0] * 0.05)
            upper = int(wfs_unit.shape[0] * 0.95)

            max_diff = 0
            max_diff_N = 0
            for n in np.arange(lower, upper):
                # Denoise templates?
                temp_1 = np.mean(wfs_unit[:n], axis=0)
                temp_2 = np.mean(wfs_unit[n:], axis=0)
                diff = np.abs(temp_1 - temp_2).max()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_N = n

            if max_diff > split_diff:
                temp_1 = np.mean(wfs_unit[:max_diff_N], axis=0)
                temp_2 = np.mean(wfs_unit[max_diff_N:], axis=0)
                n_temp_cleaned += 1
                if (
                    np.abs(temp_1 - template_mc_trace).max()
                    > np.abs(temp_2 - template_mc_trace).max()
                ):
                    which = spikes_in_unit[ptp_sort_idx[:max_diff_N]]
                    spike_train[which] = cmp
                else:
                    which = spikes_in_unit[ptp_sort_idx[max_diff_N:]]
                    spike_train[which] = cmp
            cmp += 1

    return n_temp_cleaned


def remove_oversplits(templates, spike_train, min_ptp=4.0, max_diff=2.0):
    """This will modify spike_train"""
    # remove oversplits according to max abs norm
    for unit in trange(templates.shape[0] - 1, desc="max abs merge"):
        if templates[unit].ptp(0).max(0) >= min_ptp:
            max_vec = np.abs(
                templates[unit, :, :] - templates[unit + 1 :]
            ).max(1).max(1)
            if max_vec.min() < max_diff:
                idx_units_to_change = unit + 1 + np.where(max_vec < max_diff)[0]
                spike_train[
                    np.isin(spike_train[:, 1], idx_units_to_change)
                ] = unit
                templates[idx_units_to_change] = templates[unit]

    # make labels contiguous and get corresponding templates
    spike_train[:, 1], orig_uniq = cluster_utils.make_labels_contiguous(
        spike_train[:, 1], return_unique=True
    )
    templates = templates[orig_uniq]

    return spike_train, templates
