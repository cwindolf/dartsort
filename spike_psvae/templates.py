import numpy as np
from pathlib import Path
import spikeinterface.full as si

from . import denoise


def get_templates(
    spike_train,
    geom,
    cache_dir,
    raw_binary_file,
    residual_binary_file,
    subtracted_waveforms,
    n_templates=None,
    max_spikes_per_unit=500,
    tpca=None,
    reducer=np.mean,
    snr_threshold=10 * np.sqrt(200),
    n_jobs=20,
    spike_length_samples=121,
    trough_offset=42,
    sampling_frequency=30_000,
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
    cache_dir : string or Path
        Waveforms will be cached by spikeinterface here.
    raw_binary_file, residual_binary_file : string or Path
    subtracted_waveforms : array, memmap, or h5 dataset
    n_templates : None or int
        If None, it will be set to max unit id + 1
    max_spikes_per_unit : int
        If a unit spikes more than this, this many will be sampled
        uniformly (separately for raw + cleaned wfs)
    tpca : None or sklearn transformer
        If None, no temporal PCA is applied to collision-cleaned
        waveforms before computing templates. If supplied, that's
        what it's used for.
    reducer : e.g. np.mean or np.median
    snr_threshold : float
        Below this number, a weighted combo of raw and cleaned
        template will be computed (weight based on template snr).

    Returns
    -------
    templates : np.array (n_templates, spike_length_samples, geom.shape[0])
    snrs : np.array (n_templates,)
        The snrs of the original raw templates.
    """
    # -- initialize output
    if n_templates is None:
        n_templates = spike_train[:, 1].max() + 1
    templates = np.zeros((n_templates, spike_length_samples, len(geom)))
    snrs = np.zeros(n_templates)

    # -- enforce decrease helpers
    full_channel_index = np.array([np.arange(len(geom))] * len(geom))
    radial_parents = denoise.make_radial_order_parents(
        geom, full_channel_index
    )

    # -- get waveform extractor
    # this will sample random waveforms from the raw/residual for us
    raw_we = get_waveform_extractor(
        raw_binary_file,
        spike_train,
        geom,
        cache_dir,
        max_spikes_per_unit=max_spikes_per_unit,
        n_jobs=n_jobs,
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset,
        sampling_frequency=sampling_frequency,
    )
    res_we = get_waveform_extractor(
        residual_binary_file,
        spike_train,
        geom,
        cache_dir,
        max_spikes_per_unit=max_spikes_per_unit,
        n_jobs=n_jobs,
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset,
        sampling_frequency=sampling_frequency,
    )

    # -- main loop to make templates
    units = np.unique(spike_train[:, 1])
    for unit in units[units >= 0]:
        # get raw template
        raw_wfs = get_unit_waveforms(unit, spike_train, raw_we)
        denoise.enforce_decrease_shells(
            raw_wfs,
            raw_wfs.ptp(1).argmax(1),
            radial_parents,
            in_place=True,
        )
        raw_template = reducer(raw_wfs, axis=0)
        raw_ptp = raw_template.ptp(0).max()
        snr = raw_ptp * len(raw_wfs)
        snrs[unit] = snr

        if snr > snr_threshold:
            templates[unit] = raw_template
            continue

        # load cleaned waveforms
        # NOTE: we can't control the random seed for spikeinterface
        #       WEs, so these will not be the same waveforms, unless
        #       this unit has fewer than `max_spikes_per_unit` spikes
        cleaned_wfs = get_unit_waveforms(
            unit, spike_train, res_we, subtracted_waveforms
        )

        # enforce decrease for both, using raw maxchan
        denoise.enforce_decrease_shells(
            cleaned_wfs,
            raw_wfs.ptp(1).argmax(1),
            radial_parents,
            in_place=True,
        )

        # apply TPCA to cleaned ones
        if tpca is not None:
            cleaned_wfs = tpca.fit_transform(
                cleaned_wfs.reshape(len(cleaned_wfs), -1)
            )
            cleaned_wfs = tpca.inverse_transform(cleaned_wfs).reshape(
                raw_wfs.shape
            )
        cleaned_template = reducer(cleaned_wfs, axis=0)

        # SNR-weighted combination to create the template
        lerp = snr / snr_threshold
        templates[unit] = (
            (lerp) * raw_template
            + (1 - lerp) * cleaned_template
        )

    return templates, snrs


def get_unit_waveforms(
    unit, spike_train, waveform_extractor, subtracted_waveforms=None
):
    """Handle loading raw vs. collision-cleaned waveforms"""
    waveforms, indices = waveform_extractor.get_waveforms(
        unit, with_index=True
    )

    if subtracted_waveforms is not None:
        # which waveforms were sampled by the waveform extractor?
        unit_which = np.flatnonzero(spike_train[:, 1] == unit)
        unit_selected = unit_which[indices["spike_index"]]

        # add in the subtracted wfs to make collision-cleaned wfs
        waveforms += subtracted_waveforms[unit_selected]

    return waveforms


def get_waveform_extractor(
    binary_file,
    spike_train,
    geom,
    cache_dir,
    max_spikes_per_unit=500,
    n_jobs=20,
    spike_length_samples=121,
    trough_offset=42,
    sampling_frequency=30_000,
    standardized_dtype=np.float32,
):
    # cache directory logic, delete if empty for waveform extractor
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    wf_cache_dir = cache_dir / f"{binary_file.stem}_template_waveforms"
    if wf_cache_dir.exists():
        assert wf_cache_dir.is_dir()
        if not len(list(wf_cache_dir.glob("*"))):
            wf_cache_dir.unlink()

    # get a recording extractor for our destriped data
    recording = si.read_binary(
        binary_file,
        num_chan=len(geom),
        sampling_frequency=sampling_frequency,
        dtype=standardized_dtype,
        time_axis=0,
        is_filtered=True,
    )
    recording.set_channel_locations(geom)

    # make dumpable NpzSortingExtractor
    sorting = si.NumpySorting.from_times_labels(
        times_list=spike_train[spike_train[:, 1] >= 0, 0],
        labels_list=spike_train[spike_train[:, 1] >= 0, 1],
        sampling_frequency=sampling_frequency,
    )
    si.NpzSortingExtractor.write_sorting(
        sorting, cache_dir / "npz_sorting.npz"
    )
    sorting = si.NpzSortingExtractor(cache_dir / "npz_sorting.npz")

    # spikeinterface uses trough offset etc in units of ms
    ms_before = trough_offset * 1000 / sampling_frequency
    ms_after = (
        (spike_length_samples - trough_offset) * 1000 / sampling_frequency
    )
    # check no issues with rounding
    assert int(ms_before * sampling_frequency / 1000) == trough_offset
    assert (
        int(ms_after * sampling_frequency / 1000)
        == spike_length_samples - trough_offset
    )

    # make waveform extractor
    we = si.extract_waveforms(
        recording,
        sorting,
        wf_cache_dir,
        overwrite=True,
        ms_before=ms_before,
        ms_after=ms_after,
        max_spikes_per_unit=max_spikes_per_unit,
        n_jobs=n_jobs,
        chunk_size=sampling_frequency,
        progress_bar=True,
        return_scaled=False,
    )

    # check assumptions on what this thing does
    test_wf, ix = we.get_waveforms(spike_train[0, 1], with_index=True)
    assert test_wf.shape[0] == len(ix)
    assert test_wf.shape[1] == spike_length_samples
    assert test_wf.shape[2] == len(geom)

    return we
