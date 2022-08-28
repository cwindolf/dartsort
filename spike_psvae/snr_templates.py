import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

from . import denoise, spikeio, waveform_utils


def get_templates(
    spike_train,
    geom,
    raw_binary_file,
    unit_max_channels,
    max_spikes_per_unit=500,
    do_tpca=True,
    do_temporal_decrease=True,
    zero_radius_um=200,
    reducer=np.median,
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    sampling_frequency=30_000,
    return_raw_cleaned=False,
    tpca_rank=5,
    tpca_radius=75,
    radial_parents=None,
    pbar=True,
    tpca_n_wfs=50_000,
    seed=0,
):
    """Get denoised templates

    This computes a weighted average of the raw template
    and the TPCA'd collision-cleaned template, based on
    the SNR = maxptp * sqrt(n). If snr > snr_threshold,
    the raw template is used. Otherwise, a convex combination
    with weight snr / snr_threshold is used.

    Enforce decrease is applied to waveforms before reducing.

    Arguments
    ---------
    spike_train : np.array, (n_spikes, 2)
        First column is spike trough time (samples), second column
        is cluster ID.
    geom : np.array (n_channels, 2)
    raw_binary_file, residual_binary_file : string or Path
    subtracted_waveforms : array, memmap, or h5 dataset
    n_templates : None or int
        If None, it will be set to max unit id + 1
    max_spikes_per_unit : int
        If a unit spikes more than this, this many will be sampled
        uniformly (separately for raw + cleaned wfs)
    reducer : e.g. np.mean or np.median
    snr_threshold : float
        Below this number, a weighted combo of raw and cleaned
        template will be computed (weight based on template snr).

    Returns
    -------
    templates : np.array (n_templates, spike_length_samples, geom.shape[0])
    snrs : np.array (n_templates,)
        The snrs of the original raw templates.
    if return_raw_cleaned, also returns raw_templates and denoised_templates,
    both arrays like templates.
    """
    # -- initialize output
    n_templates = spike_train[:, 1].max() + 1
    templates = np.zeros((n_templates, spike_length_samples, len(geom)))

    snr_by_channel = np.zeros((n_templates, len(geom)))
    rg = np.random.default_rng(seed)

    raw_templates = np.zeros_like(templates)
    orig_raw_templates = np.zeros_like(templates)
    denoised_templates = np.zeros_like(templates)
    extra = dict(
        orig_raw_templates=orig_raw_templates,
        raw_templates=raw_templates,
        denoised_templates=denoised_templates,
        snr_by_channel=snr_by_channel,
    )

    # -- fit TPCA to randomly sampled waveforms
    if do_tpca:
        print("pca wfs...")
        tpca_channel_index = waveform_utils.make_channel_index(
            geom, tpca_radius, steps=1, distance_order=False, p=1
        )
        choices = rg.choice(len(spike_train), size=tpca_n_wfs, replace=False)
        choices.sort()
        tpca_waveforms, skipped_idx = spikeio.read_waveforms(
            spike_train[choices, 0],
            raw_binary_file,
            geom.shape[0],
            channel_index=tpca_channel_index,
            spike_length_samples=spike_length_samples,
            max_channels=unit_max_channels[spike_train[choices, 1]],
        )
        print(tpca_waveforms.shape)
        # NTC -> NCT
        denoise.enforce_temporal_decrease(tpca_waveforms, in_place=True)
        tpca_waveforms = tpca_waveforms.transpose(0, 2, 1).reshape(
            -1, spike_length_samples
        )
        which = np.isfinite(tpca_waveforms[:, 0])
        tpca_waveforms = tpca_waveforms[which]
        tpca = PCA(tpca_rank).fit(tpca_waveforms)
        extra["tpca"] = tpca

    # -- main loop to make templates
    units = np.unique(spike_train[:, 1])
    units = units[units >= 0]
    for unit in tqdm(units, desc="Cleaned templates") if pbar else units:
        # get raw template
        in_unit = np.flatnonzero(spike_train[:, 1] == unit)
        choices = slice(None)
        if in_unit.size > max_spikes_per_unit:
            choices = rg.choice(
                in_unit.size, max_spikes_per_unit, replace=False
            )
            choices.sort()
        waveforms = spikeio.get_waveforms(
            spike_train[choices, 0],
            raw_binary_file,
            len(geom),
            max_spikes_per_unit=max_spikes_per_unit,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
        )
        orig_raw_templates[unit] = reducer(waveforms, axis=0)

        if do_temporal_decrease:
            denoise.enforce_temporal_decrease(waveforms, in_place=True)
        raw_templates[unit] = reducer(waveforms, axis=0)
        raw_ptp = raw_templates[unit].ptp(0)
        snr_by_channel[unit] = raw_ptp * np.sqrt(len(waveforms))

        # denoise the waveforms
        if do_tpca:
            nn, tt, cc = waveforms.shape
            waveforms = waveforms.transpose(0, 2, 1).reshape(nn * cc, tt)
            waveforms = tpca.inverse_transform(tpca.transform(waveforms))
            waveforms = waveforms.reshape(nn, cc, tt).transpose(0, 2, 1)

        # enforce decrease for both, using raw maxchan
        if do_temporal_decrease:
            denoise.enforce_temporal_decrease(waveforms, in_place=True)
        denoised_templates[unit] = reducer(waveforms, axis=0)

    # SNR-weighted combination to create the template
    weights = denoised_weights(
        snr_by_channel, spike_length_samples, trough_offset, snr_threshold
    )
    templates = weights * denoised_templates + (1 - weights) * raw_templates
    extra["weights"] = weights

    # zero out far away channels
    if zero_radius_um is not None:
        zero_ci = waveform_utils.make_channel_index(
            geom, zero_radius_um, steps=1, distance_order=False, p=2
        )
        for i in range(len(templates)):
            mc = templates[i].ptp(0).argmax()
            far = ~np.isin(np.arange(len(geom)), zero_ci[mc])
            templates[i, :, far] = 0

    return templates, extra


def denoised_weights(
    snrs,
    spike_length_samples,
    trough_offset,
    snr_threshold,
    a=12.0,
    b=12.0,
    d=6.0,
):
    # v shaped function for time weighting
    vt = np.abs(np.arange(spike_length_samples) - trough_offset, dtype=float)[
        :, None
    ]
    vt[trough_offset:] = vt[trough_offset:] / vt[trough_offset:].max()
    vt[:trough_offset] = vt[:trough_offset] / vt[:trough_offset].max()

    # snr weighting per channel
    sc = np.minimum(snrs, snr_threshold) / snr_threshold

    # pass it through a hand picked squashing function
    wtc = 1.0 / (1.0 + np.exp(d + a * vt - b * sc))

    return wtc
