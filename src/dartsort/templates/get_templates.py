"""Compute template waveforms, including accounting for drift and denoising

The class TemplateData in templates.py provides a friendlier interface,
where you can get templates using the TemplateConfig in config.py.
"""
from dataclasses import replace

import numpy as np
from dartsort.util import spikeio
from dartsort.util.drift_util import registered_average
from dartsort.util.multiprocessing_util import get_pool
from dartsort.util.waveform_util import make_channel_index
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm


def get_templates(
    recording,
    sorting,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    pitch_shifts=None,
    registered_geom=None,
    realign_peaks=False,
    realign_max_sample_shift=20,
    low_rank_denoising=True,
    denoising_tsvd=None,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    zero_radius_um=None,
    reducer=np.nanmedian,
    random_seed=0,
    n_jobs=0,
    show_progress=True,
):
    """Raw, denoised, and shifted templates

    Low-level helper function which does the work of template computation for
    the template classes elsewhere in this folder

    Arguments
    ---------
    times, channels, labels : arrays of shape (n_spikes,)
        The trough (or peak) times, main channels, and unit labels
    geom : array of shape (n_channels, 2)
        Probe channel geometry, needed to subsample channels when fitting
        the low-rank denoising model, and also needed if the shifting
        arguments are specified
    pitch_shifts : int array of shape (n_spikes,)
        When computing extended templates, these shifts are applied
        before averaging
    registered_geom : array of shape (n_channels_extended, 2)
        Required if pitch_shifts is supplied. See drift_util.registered_geometry.
    realign_peaks : bool
        If True, a first round of raw templates are computed and used to shift
        the spike times such that their peaks/troughs land on trough_offset_samples
    trough_offset_samples, spike_length_samples : int
        Waveform snippets will be loaded from times[i] - trough_offset_samples
        to times[i] - trough_offset_samples + spike_length_samples
    spikes_per_unit : int
        Load at most this many randomly selected spikes per unit
    low_rank_denoising : bool
        Should we compute denoised templates? If not, raw averages.
    denoising_model : sklearn Transformer
        Pre-fit denoising model, in which case the next args are ignored
    denoising_rank, denoising_fit_radius, denoising_spikes_fit
        Parameters for the low rank model fit for denoising
    denoising_snr_threshold : int
        The SNR (=amplitude*sqrt(n_spikes)) threshold at which the
        denoising is ignored and we just use the usual template
    output_hdf5_filename : str or Path
        Denoised and/or raw templates will be saved here under the dataset
        names "raw_templates" and "denoised_templates"
    keep_waveforms_in_hdf5 : bool
        If True and output_hdf5_filename is supplied, waveforms extracted
        for template computation are retained in the output hdf5. Else,
        deleted to save disk space.
    scratch_dir : str or Path
        This is where a temporary directory will be made for intermediate
        computations, if output_hdf5_filename is None. If it's left blank,
        the tempfile default directory is used. If output_hdf5_file is not
        None, that hdf5 file is used and this argument is ignored.

    Returns
    -------
    dict whose keys vary based on the above arguments

    """
    # validate arguments
    raw_only = not low_rank_denoising
    geom = recording.get_channel_locations()

    # estimate peak sample times and realign spike train
    if realign_peaks:
        # pad the trough_offset_samples and spike_length_samples so that
        # if the user did not request denoising we can just return the
        # raw templates right away
        trough_offset_load = trough_offset_samples + realign_max_sample_shift
        spike_length_load = spike_length_samples + 2 * realign_max_sample_shift
        templates = get_raw_templates(
            recording,
            sorting,
            geom=geom,
            pitch_shifts=pitch_shifts,
            registered_geom=registered_geom,
            realign_peaks=False,
            trough_offset_samples=trough_offset_load,
            spike_length_samples=spike_length_load,
            spikes_per_unit=spikes_per_unit,
            zero_radius_um=zero_radius_um,
            reducer=reducer,
            random_seed=random_seed,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        sorting, templates = realign_sorting(
            sorting,
            templates,
            max_shift=realign_max_sample_shift,
            trough_offset_samples=trough_offset_samples,
        )

        if raw_only:
            # overwrite template dataset with aligned ones
            # handle keep_waveforms_in_hdf5
            return dict(sorting=sorting, templates=templates, raw_templates=templates)

    # fit tsvd
    if low_rank_denoising and denoising_tsvd is None:
        denoising_tsvd = fit_tsvd(
            recording,
            sorting,
            geom=geom,
            denoising_rank=denoising_rank,
            denoising_fit_radius=denoising_fit_radius,
            denoising_spikes_fit=denoising_spikes_fit,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            random_seed=random_seed,
        )

    # template logic
    # for each unit, get shifted raw and denoised averages and channel SNRs
    res = get_all_shifted_raw_and_low_rank_templates(
        recording,
        sorting,
        geom=geom,
        registered_geom=registered_geom,
        denoising_tsvd=denoising_tsvd,
        pitch_shifts=pitch_shifts,
        spikes_per_unit=spikes_per_unit,
        reducer=reducer,
        n_jobs=n_jobs,
        random_seed=random_seed,
        show_progress=show_progress,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
    )
    raw_templates, low_rank_templates, snrs_by_channel = res

    if raw_only:
        return dict(
            sorting=sorting,
            templates=raw_templates,
            raw_templates=raw_templates,
            snrs_by_channel=snrs_by_channel,
        )

    weights = denoising_weights(
        snrs_by_channel,
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset_samples,
        snr_threshold=denoising_snr_threshold,
    )
    templates = weights * raw_templates + (1 - weights) * low_rank_templates

    return dict(
        sorting=sorting,
        templates=templates,
        raw_templates=raw_templates,
        low_rank_templates=low_rank_templates,
        snrs_by_channel=snrs_by_channel,
        weights=weights,
    )


def get_raw_templates(
    recording,
    sorting,
    geom=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    pitch_shifts=None,
    registered_geom=None,
    realign_peaks=False,
    realign_max_sample_shift=20,
    reducer=np.nanmedian,
    random_seed=0,
    n_jobs=0,
    show_progress=True,
):
    return get_templates(
        recording,
        sorting,
        geom=geom,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        spikes_per_unit=spikes_per_unit,
        pitch_shifts=pitch_shifts,
        registered_geom=registered_geom,
        realign_peaks=realign_peaks,
        realign_max_sample_shift=realign_max_sample_shift,
        low_rank_denoising=False,
        reducer=reducer,
        random_seed=random_seed,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


# -- helpers


def realign_sorting(
    sorting,
    templates,
    max_shift=20,
    trough_offset_samples=42,
):
    n, t, c = templates.shape

    if max_shift == 0:
        return sorting, templates

    # find template peak time
    template_maxchans = templates.ptp(1).argmax(1)
    template_maxchan_traces = templates[np.arange(n), :, template_maxchans]
    template_peak_times = np.abs(template_maxchan_traces).argmax(1)

    # find unit sample time shifts
    template_shifts = template_peak_times - (trough_offset_samples + max_shift)
    template_shifts[np.abs(template_shifts) > max_shift] = 0

    # create aligned spike train
    new_times = sorting.times_samples + template_shifts[sorting.labels]
    aligned_sorting = replace(sorting, times_samples=new_times)

    # trim templates
    aligned_spike_len = t - 2 * max_shift
    aligned_templates = np.empty((n, aligned_spike_len, c))
    for i, dt in enumerate(template_shifts):
        aligned_templates[i] = templates[
            i, max_shift + dt : max_shift + dt + aligned_spike_len
        ]

    return aligned_sorting, aligned_templates


def fit_tsvd(
    recording,
    sorting,
    geom,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    trough_offset_samples=42,
    spike_length_samples=121,
    random_seed=0,
):
    # read spikes on channel neighborhood
    tsvd_channel_index = make_channel_index(geom, denoising_fit_radius)

    # subset spikes used to fit tsvd
    rg = np.random.default_rng(random_seed)
    choices = slice(None)
    if sorting.n_spikes > denoising_spikes_fit:
        choices = rg.choice(sorting.n_spikes, denoising_spikes_fit, replace=False)
        choices.sort()
    times = sorting.times_samples[choices]
    channels = sorting.channels[choices]

    # grab waveforms
    waveforms = spikeio.read_waveforms_channel_index(
        recording,
        times,
        tsvd_channel_index,
        channels,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
    )

    # reshape, fit tsvd, and done
    tsvd = TruncatedSVD(n_components=denoising_rank, random_seed=random_seed)
    tsvd.fit(waveforms.transpose(0, 2, 1).reshape(len(waveforms), -1))

    return tsvd


def denoising_weights(
    snrs,
    spike_length_samples,
    trough_offset,
    snr_threshold,
    a=12.0,
    b=12.0,
    d=6.0,
    edge_behavior="saturate",
):
    # v shaped function for time weighting
    vt = np.abs(np.arange(spike_length_samples) - trough_offset, dtype=float)
    if trough_offset < spike_length_samples:
        vt[trough_offset:] = vt[trough_offset:] / vt[trough_offset:].max()
    if trough_offset > 0:
        vt[:trough_offset] = vt[:trough_offset] / vt[:trough_offset].max()

    # snr weighting per channel
    if edge_behavior == "saturate":
        snc = np.minimum(snrs, snr_threshold) / snr_threshold
    if edge_behavior == "inf":
        snc = np.minimum(snrs, snr_threshold) / snr_threshold
        snc[snc >= 1.0] = np.inf
    elif edge_behavior == "raw":
        snc = snrs

    # pass it through a hand picked squashing function
    wntc = 1.0 / (1.0 + np.exp(d + a * vt[None, :, None] - b * snc[:, None, :]))

    return wntc


# -- main routine which does all the spike loading and computation


def get_all_shifted_raw_and_low_rank_templates(
    recording,
    sorting,
    geom=None,
    registered_geom=None,
    denoising_tsvd=None,
    pitch_shifts=None,
    spikes_per_unit=500,
    reducer=np.nanmedian,
    n_jobs=0,
    random_seed=0,
    show_progress=True,
    trough_offset_samples=42,
    spike_length_samples=121,
):
    n_jobs, Executor, context, rank_queue = get_pool(n_jobs, with_rank_queue=True)
    unit_ids = np.unique(sorting.labels)
    unit_ids = unit_ids[unit_ids >= 0]
    raw = denoising_tsvd is None
    prefix = "Raw" if raw else "Denoised"

    n_template_channels = recording.get_num_channels()
    if registered_geom is not None:
        n_template_channels = len(registered_geom)

    n_units = sorting.labels.max() + 1
    raw_templates = np.zeros((n_units, spike_length_samples, n_template_channels))
    low_rank_templates = None
    if not raw:
        low_rank_templates = np.zeros(
            (n_units, spike_length_samples, n_template_channels)
        )
    snrs_by_channel = np.zeros((n_units, n_template_channels))

    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_template_process_init,
        initargs=(
            rank_queue,
            random_seed,
            recording,
            sorting,
            geom,
            registered_geom,
            denoising_tsvd,
            pitch_shifts,
            spikes_per_unit,
            reducer,
            trough_offset_samples,
            spike_length_samples,
        ),
    ) as pool:
        # launch the jobs and wrap in a progress bar
        results = pool.map(_template_job, unit_ids)
        if show_progress:
            results = tqdm(
                results,
                smoothing=0.01,
                desc=f"{prefix} templates",
            )

        for unit_id, raw_template, low_rank_template, snr_by_chan in results:
            raw_templates[unit_id] = raw_template
            if not raw:
                low_rank_templates[unit_id] = low_rank_template
            snrs_by_channel[unit_id] = snr_by_chan

    return raw_templates, low_rank_templates, snrs_by_channel


class TemplateProcessContext:
    def __init__(
        self,
        rg,
        recording,
        sorting,
        geom,
        registered_geom,
        denoising_tsvd,
        pitch_shifts,
        spikes_per_unit,
        reducer,
        trough_offset_samples,
        spike_length_samples,
    ):
        self.rg = rg
        self.recording = recording
        self.sorting = sorting
        self.geom = geom
        self.registered_geom = registered_geom
        self.denoising_tsvd = denoising_tsvd
        self.pitch_shifts = pitch_shifts
        self.spikes_per_unit = spikes_per_unit
        self.reducer = reducer
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples

        self.n_channels = recording.get_num_channels()
        self.registered = registered_geom is not None


_template_process_context = None


def _template_process_init(
    rank_queue,
    random_seed,
    recording,
    sorting,
    geom,
    registered_geom,
    denoising_tsvd,
    pitch_shifts,
    spikes_per_unit,
    reducer,
    trough_offset_samples,
    spike_length_samples,
):
    global _template_process_context

    rank = rank_queue.get()
    rg = np.random.default_rng(random_seed + rank)
    _template_process_context = TemplateProcessContext(
        rg,
        recording,
        sorting,
        geom,
        registered_geom,
        denoising_tsvd,
        pitch_shifts,
        spikes_per_unit,
        reducer,
        trough_offset_samples,
        spike_length_samples,
    )


def _template_job(unit_id):
    p = _template_process_context

    in_unit = np.flatnonzero(p.sorting.labels == unit_id)
    if in_unit.size > p.spikes_per_unit:
        in_unit = p.rg.choice(in_unit, p.spikes_per_unit, replace=False)
        in_unit.sort()
    times = p.sorting.times_samples[in_unit]

    waveforms = spikeio.read_full_waveforms(
        p.recording,
        times,
        trough_offset_samples=p.trough_offset_samples,
        spike_length_samples=p.spike_length_samples,
    )
    n, t, c = waveforms.shape

    if p.registered:
        raw_template = registered_average(
            waveforms,
            p.pitch_shifts[in_unit],
            p.geom,
            p.registered_geom,
            reducer=p.reducer,
        )
        counts = registered_average(
            np.ones((n, p.n_channels)),
            p.pitch_shifts[in_unit],
            p.geom,
            p.registered_geom,
            reducer=np.nansum,
        )
    else:
        raw_template = p.reducer(waveforms, axis=0)
        counts = np.full(p.n_channels, float(n))
    snr_by_chan = raw_template.ptp(0) * np.sqrt(counts)

    if p.denoising_tsvd is None:
        return unit_id, raw_template, None, snr_by_chan

    waveforms = waveforms.transpose(0, 2, 1).reshape(n, t * c)
    waveforms = p.denoising_tsvd.transform(waveforms)
    waveforms = waveforms.reshape(n, c, t).transpose(0, 2, 1)
    if p.registered:
        low_rank_template = registered_average(
            waveforms,
            p.pitch_shifts[in_unit],
            p.geom,
            p.registered_geom,
            reducer=p.reducer,
        )
    else:
        low_rank_template = p.reducer(waveforms, axis=0)

    return unit_id, raw_template, low_rank_template, snr_by_chan
