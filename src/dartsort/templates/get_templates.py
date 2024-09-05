"""Compute template waveforms, including accounting for drift and denoising

The class TemplateData in templates.py provides a friendlier interface,
where you can get templates using the TemplateConfig in config.py.
"""
from dataclasses import replace

import numpy as np
import torch
from dartsort.util.data_util import batched_h5_read, chunked_h5_read
from dartsort.util import spikeio
from dartsort.util.data_util import keep_all_most_recent_spikes_per_chunk
from dartsort.util.drift_util import registered_template, get_waveforms_on_static_channels, get_spike_pitch_shifts
from dartsort.util.multiprocessing_util import get_pool
from dartsort.util.spiketorch import fast_nanmedian, ptp, fast_nanweightedmean
from dartsort.util.waveform_util import make_channel_index, full_channel_index, channel_subset_by_radius
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm
import h5py

def get_templates_with_h5(
    recording,
    h5_file,
    sorting,
    wfs_name,
    indices=None,
    pitch_shifts=None,
    registered_geom=None,
    spikes_per_unit=500,
    low_rank_denoising=True,
    denoising_rank=5,
    denoising_tsvd=None,
    realign_peaks=False,
    realign_max_sample_shift=20,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    min_fraction_at_shift=0.25,
    min_count_at_shift=25,
    reducer=fast_nanmedian,
    spatial_svdsmoothing=False,
    max_ptp_chans_to_spatialsmooth=3,
    spike_length_samples=121,
    weight_wfs=None,
    n_jobs=0,
    show_progress=True,
    device=None,
    units_per_job=8,
    random_seed=0,
):

    raw_only = not low_rank_denoising

    if realign_peaks:
        # pad the trough_offset_samples and spike_length_samples so that
        # if the user did not request denoising we can just return the
        # raw templates right away
        raw_results = get_templates_with_h5(
            recording,
            h5_file,
            sorting,
            wfs_name, 
            indices=indices,
            pitch_shifts=pitch_shifts,
            registered_geom=registered_geom,
            realign_peaks=False,
            spikes_per_unit=spikes_per_unit,
            min_fraction_at_shift=min_fraction_at_shift,
            min_count_at_shift=min_count_at_shift,
            reducer=reducer,
            weight_wfs=weight_wfs,
            random_seed=random_seed,
            n_jobs=n_jobs,
            show_progress=show_progress,
            device=device,
        )
        sorting, templates = realign_sorting(
            sorting,
            raw_results["raw_templates"],
            raw_results["snrs_by_channel"],
            max_shift=realign_max_sample_shift,
            trough_offset_samples=trough_offset_samples,
            recording_length_samples=recording.get_num_samples(),
        )
        if raw_only:
            # overwrite template dataset with aligned ones
            # handle keep_waveforms_in_hdf5
            raw_results["sorting"] = sorting
            raw_results["templates"] = raw_results["raw_templates"] = templates
            return raw_results

    res = get_all_shifted_raw_and_low_rank_templates_with_h5(
        recording,
        h5_file,
        sorting,
        wfs_name,
        indices=indices,
        registered_geom=registered_geom,
        # denoising_tsvd=denoising_tsvd, Not ready yet
        pitch_shifts=pitch_shifts, #pitch_shifts have same structure as labels_all
        spikes_per_unit=spikes_per_unit,
        reducer=reducer,
        n_jobs=n_jobs,
        units_per_job=units_per_job,
        random_seed=random_seed,
        show_progress=show_progress,
        min_fraction_at_shift=min_fraction_at_shift,
        min_count_at_shift=min_count_at_shift,
        weight_wfs=weight_wfs,
        device=device,
        spike_length_samples=spike_length_samples,
    )
    raw_templates, _, snrs_by_channel, counts = res

    return dict(
        sorting=sorting,
        templates=raw_templates,
        raw_templates=raw_templates,
        snrs_by_channel=snrs_by_channel,
        per_chan_counts=counts,
    )

def get_templates_multiple_chunks_linear(
    recording,
    sorting,
    chunk_time_ranges_s,
    geom,
    template_config,
    motion_est=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    registered_geom=None,
    realign_peaks=False,
    realign_max_sample_shift=20,
    low_rank_denoising=True,
    denoising_tsvd=None,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    min_fraction_at_shift=0.25,
    min_count_at_shift=25,
    reducer=fast_nanmedian,
    pad_value=0.0,
    spatial_svdsmoothing=False,
    max_ptp_chans_to_spatialsmooth=3,
    random_seed=0,
    # units_per_job=8,
    n_jobs=0,
    show_progress=True,
    device=None,
):
    """Raw, denoised, and shifted templates

    Low-level helper function which does the work of template computation for
    the template classes elsewhere in this folder

    Arguments
    ---------
    sorting : sorting object containing times, channels, labels and spike positions 
    chunk_time_ranges_s : all chunks of the data on which to compute templates. 
    geom : array of shape (n_channels, 2)
        Probe channel geometry, needed to subsample channels when fitting
        the low-rank denoising model, and also needed if the shifting
        arguments are specified
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

    if realign_peaks: 
        trough_offset_samples_original = trough_offset_samples
        trough_offset_samples = trough_offset_samples + realign_max_sample_shift
        spike_length_samples = spike_length_samples + 2 * realign_max_sample_shift
        
    # fit tsvd
    if low_rank_denoising and denoising_tsvd is None:
        print("fitting tsvd")
        denoising_tsvd = fit_tsvd(
            recording,
            sorting,
            denoising_rank=denoising_rank,
            denoising_fit_radius=denoising_fit_radius,
            denoising_spikes_fit=denoising_spikes_fit,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            random_seed=random_seed,
        )
        denoising_tsvd = TorchSVDProjector(
            torch.from_numpy(
                denoising_tsvd.components_.astype(recording.dtype)
            )
        )

    n_chunks = len(chunk_time_ranges_s)
    print("keeping all necessary spikes")
    times_samples_unique, times_seconds_unique, depths_um_unique, ind_arr, inv_arr, all_chunk_ids, all_labels = keep_all_most_recent_spikes_per_chunk(
        sorting,
        chunk_time_ranges_s,
        template_config,
        recording,
    )

    print("computing pitch shifts")
    pitch_shifts = get_spike_pitch_shifts(
                depths_um_unique,
                geom,
                times_s=times_seconds_unique,
                motion_est=motion_est,
            )

    # template logic
    # for each unit, get shifted raw and denoised averages and channel SNRs
    # This is not ready yet 
    # res = get_all_shifted_raw_and_low_rank_templates_linear_median_stochastic_approx(
    res = get_all_shifted_raw_and_low_rank_templates_linear(
        recording,
        sorting,
        times_samples_unique, 
        ind_arr,
        inv_arr, 
        all_chunk_ids, 
        all_labels,
        pad_value=pad_value,
        n_chunks=len(chunk_time_ranges_s),
        registered_geom=registered_geom,
        denoising_tsvd=denoising_tsvd,
        pitch_shifts=pitch_shifts,
        spikes_per_unit=spikes_per_unit,
        reducer=reducer, #This one will have to be the mean
        n_jobs=n_jobs,
        random_seed=random_seed,
        show_progress=show_progress,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        min_fraction_at_shift=min_fraction_at_shift,
        min_count_at_shift=min_count_at_shift,
        device=device,
    )
    raw_templates, low_rank_templates, snrs_by_channel, spike_counts = res

    if raw_only:
        if realign_peaks: 
            unit_ids = np.unique(sorting.labels)
            unit_ids = unit_ids[unit_ids>-1]
            sorting, raw_templates = realign_sorting_per_chunk(
                sorting,
                raw_templates,
                snrs_by_channel,
                unit_ids,
                chunk_time_ranges_s=chunk_time_ranges_s,
                max_shift=realign_max_sample_shift,
                trough_offset_samples=trough_offset_samples_original,
                recording_length_samples=recording.get_num_samples(),
            )
        
        return dict(
            sorting=sorting,
            templates=raw_templates,
            raw_templates=raw_templates,
            snrs_by_channel=snrs_by_channel,
            spike_counts=spike_counts,
            denoising_tsvd=denoising_tsvd,
        )

    templates = raw_templates.copy()
    for j in range(n_chunks):
        weights = denoising_weights(
            snrs_by_channel[j],
            spike_length_samples=spike_length_samples,
            trough_offset=trough_offset_samples,
            snr_threshold=denoising_snr_threshold,
        )
        templates[j] = weights * raw_templates[j] + (1 - weights) * low_rank_templates[j]
    templates = templates.astype(recording.dtype)

    if spatial_svdsmoothing:
        for j in range(n_chunks):
            for k in range(templates.shape[1]):
                chans_low_ptp = np.flatnonzero(templates[j, k].ptp(0)<max_ptp_chans_to_spatialsmooth)
                U, s, Vh = svd(templates[j, k][:, chans_low_ptp].T, full_matrices=False)
                s[denoising_rank:]=0
                templates[j, k, :, chans_low_ptp] = np.dot(U, np.dot(np.diag(s), Vh))

    if realign_peaks: 
        unit_ids = np.unique(sorting.labels)
        unit_ids = unit_ids[unit_ids>-1]
        sorting, templates = realign_sorting_per_chunk(
            sorting,
            templates,
            snrs_by_channel,
            unit_ids,
            chunk_time_ranges_s=chunk_time_ranges_s,
            max_shift=realign_max_sample_shift,
            trough_offset_samples=trough_offset_samples_original,
            recording_length_samples=recording.get_num_samples(),
        )


    return dict(
        sorting=sorting,
        templates=templates,
        raw_templates=raw_templates,
        low_rank_templates=low_rank_templates,
        snrs_by_channel=snrs_by_channel,
        weights=weights,
        spike_counts=spike_counts,
        denoising_tsvd=denoising_tsvd,
    )



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
    denoising_spikes_fit=20_000,
    denoising_snr_threshold=50.0,
    min_fraction_at_shift=0.25,
    min_count_at_shift=25,
    reducer=fast_nanmedian,
    spatial_svdsmoothing=False,
    max_ptp_chans_to_spatialsmooth=3,
    random_seed=0,
    units_per_job=8,
    n_jobs=0,
    dtype=np.float32,
    show_progress=True,
    device=None,
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

    # estimate peak sample times and realign spike train
    if realign_peaks:
        # pad the trough_offset_samples and spike_length_samples so that
        # if the user did not request denoising we can just return the
        # raw templates right away
        trough_offset_load = trough_offset_samples + realign_max_sample_shift
        spike_length_load = spike_length_samples + 2 * realign_max_sample_shift
        raw_results = get_raw_templates(
            recording,
            sorting,
            pitch_shifts=pitch_shifts,
            registered_geom=registered_geom,
            realign_peaks=False,
            trough_offset_samples=trough_offset_load,
            spike_length_samples=spike_length_load,
            spikes_per_unit=spikes_per_unit,
            min_fraction_at_shift=min_fraction_at_shift,
            min_count_at_shift=min_count_at_shift,
            reducer=reducer,
            random_seed=random_seed,
            n_jobs=n_jobs,
            show_progress=show_progress,
            dtype=dtype,
            device=device,
        )
        sorting, templates = realign_sorting(
            sorting,
            raw_results["raw_templates"],
            raw_results["snrs_by_channel"],
            raw_results["unit_ids"],
            max_shift=realign_max_sample_shift,
            trough_offset_samples=trough_offset_samples,
            recording_length_samples=recording.get_num_samples(),
        )
        if raw_only:
            # overwrite template dataset with aligned ones
            # handle keep_waveforms_in_hdf5
            raw_results["sorting"] = sorting
            raw_results["templates"] = raw_results["raw_templates"] = templates
            return raw_results

    # fit tsvd
    if low_rank_denoising and denoising_tsvd is None:
        print("Fitting tsvd")
        denoising_tsvd = fit_tsvd(
            recording,
            sorting,
            dtype=dtype,
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
        registered_geom=registered_geom,
        denoising_tsvd=denoising_tsvd,
        pitch_shifts=pitch_shifts,
        spikes_per_unit=spikes_per_unit,
        reducer=reducer,
        n_jobs=n_jobs,
        units_per_job=units_per_job,
        random_seed=random_seed,
        show_progress=show_progress,
        dtype=dtype,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        min_fraction_at_shift=min_fraction_at_shift,
        min_count_at_shift=min_count_at_shift,
        device=device,
    )
    unit_ids, spike_counts, raw_templates, low_rank_templates, snrs_by_channel = res

    if low_rank_denoising:
        denoising_tsvd = TorchSVDProjector(
            torch.from_numpy(
                denoising_tsvd.components_.astype(recording.dtype)
            )
        )


    if raw_only:
        return dict(
            sorting=sorting,
            unit_ids=unit_ids,
            spike_counts=spike_counts,
            templates=raw_templates,
            raw_templates=raw_templates,
            snrs_by_channel=snrs_by_channel,
            denoising_tsvd=denoising_tsvd,
        )

    weights = denoising_weights(
        snrs_by_channel,
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset_samples,
        snr_threshold=denoising_snr_threshold,
    )
    templates = weights * raw_templates + (1 - weights) * low_rank_templates
    templates = templates.astype(dtype)

    if spatial_svdsmoothing:
        for k in range(templates.shape[0]):
            chans_low_ptp = np.flatnonzero(templates[k].ptp(0)<max_ptp_chans_to_spatialsmooth)
            U, s, Vh = svd(templates[k][:, chans_low_ptp].T, full_matrices=False)
            s[denoising_rank:]=0
            templates[k, :, chans_low_ptp] = np.dot(U, np.dot(np.diag(s), Vh))

    return dict(
        sorting=sorting,
        unit_ids=unit_ids,
        spike_counts=spike_counts,
        templates=templates,
        raw_templates=raw_templates,
        low_rank_templates=low_rank_templates,
        snrs_by_channel=snrs_by_channel,
        weights=weights,
        denoising_tsvd=denoising_tsvd,
    )


def get_raw_templates(
    recording,
    sorting,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    pitch_shifts=None,
    registered_geom=None,
    realign_peaks=False,
    realign_max_sample_shift=20,
    min_fraction_at_shift=0.1,
    min_count_at_shift=5,
    reducer=fast_nanmedian,
    random_seed=0,
    n_jobs=0,
    dtype=np.float32,
    show_progress=True,
    device=None,
):
    return get_templates(
        recording,
        sorting,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        spikes_per_unit=spikes_per_unit,
        pitch_shifts=pitch_shifts,
        registered_geom=registered_geom,
        realign_peaks=realign_peaks,
        realign_max_sample_shift=realign_max_sample_shift,
        min_fraction_at_shift=min_fraction_at_shift,
        min_count_at_shift=min_count_at_shift,
        low_rank_denoising=False,
        reducer=reducer,
        random_seed=random_seed,
        n_jobs=n_jobs,
        dtype=dtype,
        show_progress=show_progress,
        device=device,
    )

# -- helpers

def realign_sorting_per_chunk(
    sorting,
    templates_all_chunks,
    snrs_by_channel_all_chunks,
    unit_ids,
    chunk_time_ranges_s,
    max_shift=20,
    trough_offset_samples=42,
    recording_length_samples=None,
):

    new_times = sorting.times_samples.copy()
    new_labels = sorting.labels.copy()
    n_chunks, n, t, c = templates_all_chunks.shape

    aligned_spike_len = t - 2 * max_shift
    templates_new = templates_all_chunks[:, :, max_shift:max_shift+aligned_spike_len]#.copy()

    if max_shift == 0:
        return sorting, templates_new

    for k, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), desc = "Realigning chunks"):

        idx_chunk = np.flatnonzero(
            np.logical_and(sorting.times_seconds>=chunk_time_range[0], sorting.times_seconds<chunk_time_range[1]) 
        )

        labels = sorting.labels[idx_chunk]

        snrs_by_channel = snrs_by_channel_all_chunks[k]
        templates = templates_all_chunks[k]
        # sorting = 
        
        # find template peak time
        template_maxchans = snrs_by_channel.argmax(1)
        template_maxchan_traces = templates[np.arange(n), :, template_maxchans]
        template_peak_times = np.abs(template_maxchan_traces).argmax(1)
    
        # find unit sample time shifts
        template_shifts_ = template_peak_times - (trough_offset_samples + max_shift)
        template_shifts_[np.abs(template_shifts_) > max_shift] = 0
        template_shifts = np.zeros(unit_ids.max() + 1, dtype=int)
        template_shifts[unit_ids] = template_shifts_
    
        # create aligned spike train
        new_times[idx_chunk] = sorting.times_samples[idx_chunk] + template_shifts[labels]
        if recording_length_samples is not None:
            highlim = recording_length_samples - (t - trough_offset_samples)
            new_labels[idx_chunk[(new_times[idx_chunk] < trough_offset_samples) & (new_times[idx_chunk] > highlim)]] = -1
    
        # trim templates
        aligned_templates = np.empty((n, aligned_spike_len, c))
        for i, dt in enumerate(template_shifts_):
            aligned_templates[i] = templates[
                i, max_shift + dt : max_shift + dt + aligned_spike_len
            ]

        templates_new[k] = aligned_templates
    
    aligned_sorting = replace(sorting, labels=new_labels, times_samples=new_times)
    
    return aligned_sorting, templates_new


def realign_sorting(
    sorting,
    templates,
    snrs_by_channel,
    unit_ids,
    max_shift=20,
    trough_offset_samples=42,
    recording_length_samples=None,
):
    n, t, c = templates.shape

    if max_shift == 0:
        return sorting, templates

    # find template peak time
    template_maxchans = snrs_by_channel.argmax(1)
    template_maxchan_traces = templates[np.arange(n), :, template_maxchans]
    template_peak_times = np.abs(template_maxchan_traces).argmax(1)

    # find unit sample time shifts
    template_shifts_ = template_peak_times - (trough_offset_samples + max_shift)
    template_shifts_[np.abs(template_shifts_) > max_shift] = 0
    template_shifts = np.zeros(sorting.labels.max() + 1, dtype=int)
    template_shifts[unit_ids] = template_shifts_

    # create aligned spike train
    new_times = sorting.times_samples + template_shifts[sorting.labels]
    labels = sorting.labels.copy()
    if recording_length_samples is not None:
        highlim = recording_length_samples - (t - trough_offset_samples)
        labels[(new_times < trough_offset_samples) & (new_times > highlim)] = -1
    aligned_sorting = replace(sorting, labels=labels, times_samples=new_times)

    # trim templates
    aligned_spike_len = t - 2 * max_shift
    aligned_templates = np.empty((n, aligned_spike_len, c))
    for i, dt in enumerate(template_shifts_):
        aligned_templates[i] = templates[
            i, max_shift + dt : max_shift + dt + aligned_spike_len
        ]

    return aligned_sorting, aligned_templates


def fit_tsvd(
    recording,
    sorting,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=25_000,
    trough_offset_samples=42,
    spike_length_samples=121,
    dtype=np.float32,
    random_seed=0,
):
    # read spikes on channel neighborhood
    geom = recording.get_channel_locations()
    tsvd_channel_index = make_channel_index(geom, denoising_fit_radius)

    # subset spikes used to fit tsvd
    rg = np.random.default_rng(random_seed)
    choices = np.flatnonzero(sorting.labels >= 0)
    if choices.size > denoising_spikes_fit:
        choices = rg.choice(choices, denoising_spikes_fit, replace=False)
        choices.sort()
    times = sorting.times_samples[choices]
    channels = sorting.channels[choices]

    # grab waveforms
    # waveforms = spikeio.read_waveforms_channel_index_chunked(
    waveforms = spikeio.read_waveforms_channel_index(
        recording,
        times,
        tsvd_channel_index,
        channels,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        fill_value=0.0,  # all-0 rows don't change SVD basis
    )
    waveforms = waveforms.astype(dtype)
    waveforms = waveforms.transpose(0, 2, 1)
    waveforms = waveforms.reshape(len(times) * tsvd_channel_index.shape[1], -1)

    # reshape, fit tsvd, and done
    tsvd = TruncatedSVD(n_components=denoising_rank, random_state=random_seed)
    tsvd.fit(waveforms)

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
    """Weights are applied to raw template, 1-weights to low rank
    """
    # v shaped function for time weighting
    vt = np.abs(np.arange(spike_length_samples) - trough_offset, dtype=float)
    if trough_offset < spike_length_samples:
        vt[trough_offset:] /= vt[trough_offset:].max()
    if trough_offset > 0:
        vt[:trough_offset] /= vt[:trough_offset].max()

    # snr weighting per channel
    if edge_behavior == "saturate":
        snc = np.minimum(snrs, snr_threshold) / snr_threshold
    elif edge_behavior == "inf":
        snc = np.minimum(snrs, snr_threshold) / snr_threshold
        snc[snc >= 1.0] = np.inf
    elif edge_behavior == "raw":
        snc = snrs

    # pass it through a hand picked squashing function
    wntc = 1.0 / (1.0 + np.exp(d + a * vt[None, :, None] - b * snc[:, None, :]))

    return wntc


# -- main routine which does all the spike loading and computation

def get_all_shifted_raw_and_low_rank_templates_linear(
    recording,
    sorting,
    times_samples_unique, 
    ind_arr,
    inv_arr, 
    all_chunk_ids, 
    all_labels, # This is unit ids of all spikes over time 
    pad_value=0.0,
    n_chunks=1,
    registered_geom=None,
    denoising_tsvd=None,
    pitch_shifts=None,
    spikes_per_unit=500,
    reducer=fast_nanmedian,
    n_jobs=0,
    batch_size=1024, #try 2048
    random_seed=0,
    show_progress=True,
    trough_offset_samples=42,
    spike_length_samples=121,
    min_fraction_at_shift=0.1,
    min_count_at_shift=5,
    device=None,
):

    """
    No parallelism yet
    """

    geom = recording.get_channel_locations()
    
    unit_ids = np.unique(sorting.labels) #CHANGE THIS WITH LABELS + UIDS
    unit_ids = unit_ids[unit_ids >= 0]
    raw = denoising_tsvd is None
    prefix = "Raw" if raw else "Denoised"

    n_template_channels = recording.get_num_channels()
    registered_kdtree = None
    registered=False
    if registered_geom is not None:
        n_template_channels = len(registered_geom)
        registered_kdtree = KDTree(registered_geom)
        registered=True

    n_units = len(unit_ids)
    raw_templates = np.zeros(
        (n_chunks, n_units, spike_length_samples, n_template_channels),
        dtype=recording.dtype,
    )
    # check how spike counts are used for denoising of templates 
    spike_counts = np.zeros(
        (n_chunks, n_units, n_template_channels),
        dtype=recording.dtype,
    )
    
    # low_rank_templates = None
    if not raw:
        low_rank_templates = np.zeros((n_chunks, n_units, spike_length_samples, n_template_channels),dtype=recording.dtype)

    # can parallelize here, since we send wfs_all_loaded[in_unit] to each job 
    batch_arr = np.arange(0, len(times_samples_unique)+batch_size, batch_size)
    batch_arr[-1]=len(times_samples_unique)

    for k in tqdm(range(len(batch_arr)-1), desc="Computing templates for all chunks"):
        batch_idx = np.arange(batch_arr[k], batch_arr[k+1])
        waveforms = spikeio.read_full_waveforms(
            recording,
            times_samples_unique[batch_idx],
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )
        # which_times = inv_arr[batch_idx_unique]
        
        # print("ind_arr")
        # print(ind_arr)
        # assert np.all(ind_arr.argsort() == np.arange(len(ind_arr)))
        # print(ind_arr[batch_idx])
        
        batch_idx_unique = np.arange(ind_arr[batch_idx][0], ind_arr[batch_idx][-1])
        which_times, which_chunks, which_units = inv_arr[batch_idx_unique], all_chunk_ids[batch_idx_unique], all_labels[batch_idx_unique]
        _, which_times = np.unique(which_times, return_inverse=True)
        # which_times, which_chunks, which_units = np.where(chunk_unit_ids[batch_idx]) #batch_size * n_chunks

        # print("LENS")
        # print(len(which_times))
        # print(len(which_chunks))
        # print(len(which_units))
        
        if not raw:
            n, t, c = waveforms.shape
            waveforms_denoised = waveforms.transpose(0, 2, 1).reshape(n * c, t)
            waveforms_denoised = denoising_tsvd(torch.tensor(waveforms_denoised), in_place=True)
            waveforms_denoised = waveforms_denoised.reshape(n, c, t).permute(0, 2, 1) #.cpu().detach().numpy()
        if registered:
            waveforms = get_waveforms_on_static_channels(
                waveforms,
                geom,
                registered_geom=registered_geom,
                n_pitches_shift=pitch_shifts[batch_idx],
                fill_value=0, 
            )
            
            nonan = ~(waveforms[which_times].ptp(1) == 0)
            # Here more complicated
            np.add.at(spike_counts, (which_chunks, which_units), nonan)
            # spike_counts[which_chunks, which_units] = spike_counts[which_chunks, which_units] + nonan
            if not raw:
                waveforms_denoised = get_waveforms_on_static_channels(
                    waveforms_denoised,
                    geom,
                    registered_geom=registered_geom,
                    n_pitches_shift=pitch_shifts[batch_idx],
                    fill_value=0, 
                )
        else:
            np.add.at(spike_counts, (which_chunks, which_units), 1)
            # spike_counts[which_chunks, which_units] += 1
        np.add.at(raw_templates, (which_chunks, which_units), waveforms[which_times])
        # raw_templates[which_chunks, which_units] = raw_templates[which_chunks, which_units] + waveforms[which_times] #reduce later
        if not raw:
            # low_rank_templates[which_chunks, which_units] = low_rank_templates[which_chunks, which_units] + waveforms_denoised[which_times]
            np.add.at(low_rank_templates, (which_chunks, which_units), waveforms_denoised[which_times].cpu().detach().numpy())

    # spike_counts = 0 --> nan
    # spike_counts shape: chunk*unit*chan
    valid = spike_counts > min_count_at_shift #> instead of >= to make sure to turn off 0 channels 
    valid &= spike_counts / spike_counts.max(2)[:, :, None] > min_fraction_at_shift
    spike_counts[~valid] = np.nan
        # spike_counts[spike_counts==0] = np.nan
    
    raw_templates /= spike_counts[:, :, None]
    if not raw:
        low_rank_templates = low_rank_templates/spike_counts[:, :, None]
    snrs_by_chan = ptp(raw_templates, 2) * spike_counts
    
    if not np.isnan(pad_value):
        raw_templates = np.nan_to_num(raw_templates, copy=False, nan=pad_value)
        snrs_by_chan = np.nan_to_num(snrs_by_chan, copy=False, nan=pad_value)
        spike_counts = np.nan_to_num(spike_counts, copy=False, nan=pad_value)
        if not raw:
            low_rank_templates = np.nan_to_num(low_rank_templates, copy=False, nan=pad_value)

    if raw:
        return raw_templates, raw_templates, snrs_by_chan, spike_counts

    return raw_templates, low_rank_templates, snrs_by_chan, spike_counts

def get_all_shifted_raw_and_low_rank_templates_linear_median_stochastic_approx(
    recording,
    sorting,
    times_samples_unique, 
    ind_arr,
    inv_arr, 
    all_chunk_ids, 
    all_labels, # This is unit ids of all spikes over time 
    pad_value=0.0,
    n_chunks=1,
    registered_geom=None,
    denoising_tsvd=None,
    pitch_shifts=None,
    spikes_per_unit=500,
    reducer=fast_nanmedian,
    
    n_jobs=0,
    batch_size=10_000, # Important to have a large size here fo rinitialization
    random_seed=0,
    show_progress=True,
    trough_offset_samples=42,
    spike_length_samples=121,
    min_fraction_at_shift=0.1,
    min_count_at_shift=5,
    device=None,
):

    """
    This function computes a stochastic approximation of the median by reading spikes by batches
    It follows this paper paragraph 2.2 (The SA Estimate)https://dl.acm.org/doi/pdf/10.1145/347090.347195 and https://epubs.siam.org/doi/pdf/10.1137/0904048

    The difference is that instead of tracking the density estimate at each batch (and updating accordingly i.e. if density estimate small, we update less as we are more confident), we use a factor M/sum(M) i.e. number new obs / total number of obs, to give more or less weight to the current median estimate if we have seen more or less spikes

    Possible improvements: 
     - still quite slow as we do this per channel so we need to track everything per channel AND per timestep
       (different than the mean computation as we can just add everything and just track spike counts there)
    - maybe no need for the density estimate (can fix this, and maybe give more weights to additional spikes)
         or use a value based on the number of spikes already seen i.e. if a lot then more confident in previous median 
         i.e. use M/(sum(M)*sqrt(n)) instead
         --> OK to use M/(sum(M)*sqrt(n)) instead, and twice as fast? not if we use small batches ....
     - np.where() maybe can track indices differently there 
     - for loop "for j, (c,u) in enumerate(chunks_units_unique)" - try to ged rid of it in a smart way
     - do a better initialization and then use smaller batch size (better in speed/memory) + would regularize a bit
       or can make sure each unit has at least >10 spikes in each batch
    """

    geom = recording.get_channel_locations()
    
    unit_ids = np.unique(sorting.labels) #CHANGE THIS WITH LABELS + UIDS
    unit_ids = unit_ids[unit_ids >= 0]
    raw = denoising_tsvd is None
    prefix = "Raw" if raw else "Denoised"

    n_template_channels = recording.get_num_channels()
    registered_kdtree = None
    registered=False
    if registered_geom is not None:
        n_template_channels = len(registered_geom)
        registered_kdtree = KDTree(registered_geom)
        registered=True

    n_units = len(unit_ids)
    current_median = np.zeros(
        (n_chunks, n_units, spike_length_samples, n_template_channels),
        dtype=recording.dtype,
    ) # this is not the mean of the templates!! it's the mean of templates inside of [mu-sigma, mu+sigma] 
    current_n_batch = np.zeros(
        (n_chunks, n_units, n_template_channels),
        dtype=recording.dtype,
    )

    """
    # Is this really needed? i.e. do we need to take the max of fn and f0/sqrt(n)
    current_density_estimate = np.zeros(
        (n_chunks, n_units, spike_length_samples, n_template_channels),
        dtype=recording.dtype,
    )
    initial_density_estimate = np.zeros(
        (n_chunks, n_units, spike_length_samples, n_template_channels),
        dtype=recording.dtype,
    )
    """
    
    # check how spike counts are used for denoising of templates 
    spike_counts = np.zeros(
        (n_chunks, n_units, n_template_channels),
        dtype=recording.dtype, #No int because then cannot have nans?
    ) # this is current spike counts i.e. before getting the new batch
    
    # low_rank_templates = None
    if not raw:
        low_rank_current_median = np.zeros((n_chunks, n_units, spike_length_samples, n_template_channels),dtype=recording.dtype) # this is not the mean of the templates!! it's the mean of templates inside of [mu-sigma, mu+sigma] for denoised spikes

        """
        current_density_estimate_low_rank = np.zeros(
            (n_chunks, n_units, spike_length_samples, n_template_channels),
            dtype=recording.dtype,
        )
        initial_density_estimate_low_rank = np.zeros(
            (n_chunks, n_units, spike_length_samples, n_template_channels),
            dtype=recording.dtype,
        )
        """


    # can parallelize here, since we send wfs_all_loaded[in_unit] to each job 
    batch_arr = np.arange(0, len(times_samples_unique)+batch_size, batch_size)
    batch_arr[-1]=len(times_samples_unique)

    for k in tqdm(range(len(batch_arr)-1), desc="Computing templates for all chunks"):
        
        batch_idx = np.arange(batch_arr[k], batch_arr[k+1])
        waveforms = spikeio.read_full_waveforms(
            recording,
            times_samples_unique[batch_idx],
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )
        # which_times = inv_arr[batch_idx_unique]
        
        # print("ind_arr")
        # print(ind_arr)
        # assert np.all(ind_arr.argsort() == np.arange(len(ind_arr)))
        # print(ind_arr[batch_idx])
        
        batch_idx_unique = np.arange(ind_arr[batch_idx][0], ind_arr[batch_idx][-1])
        which_times, which_chunks, which_units = inv_arr[batch_idx_unique], all_chunk_ids[batch_idx_unique], all_labels[batch_idx_unique]
        _, which_times = np.unique(which_times, return_inverse=True)
        
        if not raw:
            n, t, c = waveforms.shape
            waveforms_denoised = waveforms.transpose(0, 2, 1).reshape(n * c, t)
            waveforms_denoised = denoising_tsvd(torch.tensor(waveforms_denoised), in_place=True)
            waveforms_denoised = waveforms_denoised.reshape(n, c, t).permute(0, 2, 1).cpu().detach().numpy()
        
        if registered:
            waveforms = get_waveforms_on_static_channels(
                waveforms,
                geom,
                registered_geom=registered_geom,
                n_pitches_shift=pitch_shifts[batch_idx],
                fill_value=0, 
            )

            if not raw:
                waveforms_denoised = get_waveforms_on_static_channels(
                    waveforms_denoised,
                    geom,
                    registered_geom=registered_geom,
                    n_pitches_shift=pitch_shifts[batch_idx],
                    fill_value=np.nan, 
                )

        # These np.where() can be changed by just keeping indices where it is 0 or not 0 in a sparse matrix?
        
        # Check where batch is not 0 for later 
        idx_previous_chunkunits, idx_previous_channels = np.where(current_n_batch[which_chunks, which_units]>0)
        # Check where current_n_batch is 0 
        # Update median + estimate + batch count there 
        # Since this is per chunk / unit + channel, many indices. Is this slow? 
        idx_no_previous_chunkunits, idx_no_previous_channels = np.where(current_n_batch[which_chunks, which_units]==0)
        idx_no_previous_chunkunits_median = np.flatnonzero(np.any(current_n_batch[which_chunks, which_units]==0, axis=1))
        
        if len(idx_no_previous_chunkunits):
            # Get initial median

            # Change this -- compute the median per chunk and per unit 
            # Can this be done without for loop? If 10 chunks 1000 units - 10_000 for loop, ok?
            # Suffices to read in chunks + units where there is at least one channel where it is 0
            # can take nan median on all channels since current_n batch is only updated on relevant channels! 
            
            units_chunks = np.c_[which_chunks[idx_no_previous_chunkunits_median], which_units[idx_no_previous_chunkunits_median]]
            chunks_units_unique, inverse_arr = np.unique(units_chunks, axis=0, return_inverse=True)
            # for j, (c,u) in tqdm(enumerate(chunks_units_unique), desc = "initial median computation...", total=len(chunks_units_unique)):
            for j, (c,u) in enumerate(chunks_units_unique):
                idx_c_u = np.flatnonzero(inverse_arr == j)
                # If other can is not nana then it's a problem --- we only want to update channels that are ok 
                current_median[c, u] = np.nanmedian(waveforms[which_times[idx_no_previous_chunkunits_median[idx_c_u]]], axis=0) #[:, idx_no_previous_channels]
                if not raw:
                    low_rank_current_median[c, u] = np.nanmedian(waveforms_denoised[which_times[idx_no_previous_chunkunits_median[idx_c_u]]], axis=0)

            idx_set = (which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], idx_no_previous_channels)
            
            #Set number of batch = 1 - same as low rank
            current_n_batch[idx_set] = 1
            
            # Get M number of spikes in batch + spike counts - same as low rank
            number_new_obs = np.zeros(
                (n_chunks, n_units, n_template_channels),
                dtype=recording.dtype,
            )
            np.add.at(number_new_obs, idx_set, 1)
            np.add.at(spike_counts, idx_set, 1) 


            """
            # Can we replace the density estimate by something constant? 
            # Get number of spikes in the density estimation interval - different if low rank
            number_in_interval = np.zeros(
                (n_chunks, n_units, spike_length_samples, n_template_channels),
                dtype=recording.dtype,
            )
            idx_in_interval, idx_times_in_interval = np.where(np.abs(waveforms[which_times[idx_no_previous_chunkunits], :, idx_no_previous_channels] - current_median[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels])<1)
            np.add.at(number_in_interval, (which_chunks[idx_no_previous_chunkunits][idx_in_interval], which_units[idx_no_previous_chunkunits][idx_in_interval], idx_times_in_interval, idx_no_previous_channels[idx_in_interval]), 1)
                
            # Get current and initial density estimate (both are equal when n=1) - different if low rank 
            current_density_estimate[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] = number_in_interval[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] / (2*number_new_obs[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], idx_no_previous_channels] + 1e-6)[:, None]
            initial_density_estimate[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] = number_in_interval[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] / (2*number_new_obs[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], idx_no_previous_channels] + 1e-6)[:, None]
            if not raw:
                number_in_interval_low_rank = np.zeros(
                    (n_chunks, n_units, spike_length_samples, n_template_channels),
                    dtype=recording.dtype,
                )
                idx_in_interval, idx_times_in_interval = np.where(np.abs(waveforms_denoised[which_times[idx_no_previous_chunkunits], :, idx_no_previous_channels] - low_rank_current_median[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels])<1)
                np.add.at(number_in_interval_low_rank, (which_chunks[idx_no_previous_chunkunits][idx_in_interval], which_units[idx_no_previous_chunkunits][idx_in_interval], idx_times_in_interval, idx_no_previous_channels[idx_in_interval]), 1)
                current_density_estimate_low_rank[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] = number_in_interval_low_rank[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] / (2*number_new_obs[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], idx_no_previous_channels] + 1e-6)[:, None]
                initial_density_estimate_low_rank[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] = number_in_interval_low_rank[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], :, idx_no_previous_channels] / (2*number_new_obs[which_chunks[idx_no_previous_chunkunits], which_units[idx_no_previous_chunkunits], idx_no_previous_channels] + 1e-6)[:, None]
            """
            
        # Get new counts and update estimates where batch is not 0  
        # Update median + estimate + batch count where batch is not 0  
        if len(idx_previous_chunkunits):

            idx_set = (which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], idx_previous_channels)
            # update number of seen batches -same as low rank
            current_n_batch[idx_set] += 1
            n_batch = current_n_batch[idx_set]

            # Get M - number of spikes in batch -same as low rank
            number_new_obs = np.zeros(
                (n_chunks, n_units, n_template_channels),
                dtype=recording.dtype,
            )
            np.add.at(number_new_obs, idx_set, 1) 
            np.add.at(spike_counts, idx_set, 1) 

            """
            # compute max(f0/sqrt(n), fn) -different if low rank
            max_initial_current_density = np.maximum(initial_density_estimate[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels]/np.sqrt(n_batch)[:, None], current_density_estimate[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels])
            """
            
            max_initial_current_density = (number_new_obs[idx_set]/(spike_counts[idx_set] * np.sqrt(n_batch)))[:, None]

            # Get number of spikes smaller than median -different if low rank
            number_in_interval = np.zeros(
                (n_chunks, n_units, spike_length_samples, n_template_channels),
                dtype=recording.dtype,
            )
            idx_in_interval, idx_times_in_interval = np.where(waveforms[which_times[idx_previous_chunkunits], :, idx_previous_channels] <= current_median[idx_set[0], idx_set[1], :, idx_set[2]])
            np.add.at(number_in_interval, (idx_set[0][idx_in_interval], idx_set[1][idx_in_interval], idx_times_in_interval, idx_set[2][idx_in_interval]), 1)

            # compute median update -different if low rank
            current_median[idx_set[0], idx_set[1], :, idx_set[2]] += (1/2 - number_in_interval[idx_set[0], idx_set[1], :, idx_set[2]]/number_new_obs[idx_set][:, None]) * max_initial_current_density #/ (n*max_initial_current_density)

            """
            # Get number of spikes in the density estimation interval -different if low rank
            number_in_interval = np.zeros(
                (n_chunks, n_units, spike_length_samples, n_template_channels),
                dtype=recording.dtype,
            )
            idx_in_interval, idx_times_in_interval = np.where(np.abs(waveforms[which_times[idx_previous_chunkunits], :, idx_previous_channels] - current_median[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels])<1/np.sqrt(n_batch)[:, None])
            np.add.at(number_in_interval, (which_chunks[idx_previous_chunkunits][idx_in_interval], which_units[idx_previous_chunkunits][idx_in_interval], idx_times_in_interval, idx_previous_channels[idx_in_interval]), 1)

            # Update current density estimate - is it ok to multiply / add instead of add.at here? Should be -different if low rank
            current_density_estimate[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels] *= (1 - 1/n_batch)[:, None]
            current_density_estimate[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels] += number_in_interval[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels] / (2 * number_new_obs[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], idx_previous_channels] * np.sqrt(n_batch))[:, None]
            """

            if not raw:
                """
                # compute max(f0/sqrt(n), fn) -different if low rank
                max_initial_current_density = np.maximum(initial_density_estimate_low_rank[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels]/np.sqrt(n_batch)[:, None], current_density_estimate_low_rank[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels])
                """
    
                # Get number of spikes smaller than median -different if low rank
                number_in_interval = np.zeros(
                    (n_chunks, n_units, spike_length_samples, n_template_channels),
                    dtype=recording.dtype,
                )
                idx_in_interval, idx_times_in_interval = np.where(waveforms_denoised[which_times[idx_previous_chunkunits], :, idx_previous_channels] <= low_rank_current_median[idx_set[0], idx_set[1], :, idx_set[2]])
                np.add.at(number_in_interval, (idx_set[0][idx_in_interval], idx_set[1][idx_in_interval], idx_times_in_interval, idx_set[2][idx_in_interval]), 1)
    
                # compute median update -different if low rank
                low_rank_current_median[idx_set[0], idx_set[1], :, idx_set[2]] += (1/2 - number_in_interval[idx_set[0], idx_set[1], :, idx_set[2]]/number_new_obs[idx_set][:, None]) * max_initial_current_density  #/(n*max_initial_current_density)

                """
                number_in_interval = np.zeros(
                    (n_chunks, n_units, spike_length_samples, n_template_channels),
                    dtype=recording.dtype,
                )
                idx_in_interval, idx_times_in_interval = np.where(np.abs(waveforms_denoised[which_times[idx_previous_chunkunits], :, idx_previous_channels] - low_rank_current_median[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels])<1/np.sqrt(n_batch)[:, None])
                np.add.at(number_in_interval, (which_chunks[idx_previous_chunkunits][idx_in_interval], which_units[idx_previous_chunkunits][idx_in_interval], idx_times_in_interval, idx_previous_channels[idx_in_interval]), 1)
    
                # Update current density estimate - is it ok to multiply / add instead of add.at here? Should be -different if low rank
                current_density_estimate_low_rank[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels] *= (1 - 1/n_batch)[:, None]
                current_density_estimate_low_rank[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels] += number_in_interval[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], :, idx_previous_channels] / (2 * number_new_obs[which_chunks[idx_previous_chunkunits], which_units[idx_previous_chunkunits], idx_previous_channels] * np.sqrt(n_batch))[:, None]

                """
    # spike_counts = 0 --> nan
    # spike_counts shape: chunk*unit*chan
    valid = spike_counts > min_count_at_shift #> instead of >= to make sure to turn off 0 channels 
    valid &= spike_counts / spike_counts.max(2)[:, :, None] > min_fraction_at_shift
    spike_counts[~valid] = np.nan
        # spike_counts[spike_counts==0] = np.nan

    # raw_templates is current median
    snrs_by_chan = ptp(current_median, 2) * spike_counts #max, min or median here? make sure it's above 0? if far away from peak, ideally denoising would tke into account the per timestep snr
    
    if not np.isnan(pad_value):
        current_median = np.nan_to_num(current_median, copy=False, nan=pad_value)
        snrs_by_chan = np.nan_to_num(snrs_by_chan, copy=False, nan=pad_value)
        spike_counts = np.nan_to_num(spike_counts, copy=False, nan=pad_value)
        if not raw:
            low_rank_current_median = np.nan_to_num(low_rank_current_median, copy=False, nan=pad_value)

    if raw:
        return current_median, current_median, snrs_by_chan, spike_counts

    return current_median, low_rank_current_median, snrs_by_chan, spike_counts

def get_all_shifted_raw_and_low_rank_templates_with_h5(
    recording,
    h5_file,
    sorting,
    wfs_name,
    indices=None,
    registered_geom=None,
    # denoising_tsvd=None, Not ready yet, maybe not useful 
    pitch_shifts=None,
    spikes_per_unit=500,
    reducer=fast_nanmedian,
    n_jobs=0,
    units_per_job=8,
    random_seed=0,
    show_progress=True,
    min_fraction_at_shift=0.1,
    min_count_at_shift=5,
    spike_length_samples=121,
    weight_wfs=None,
    device=None,
):
    """
    No parallelism yet
    """
    geom = recording.get_channel_locations()

    if weight_wfs is not None:
        reducer = fast_nanweightedmean
    
    unit_ids = np.unique(sorting.labels) #CHANGE THIS WITH LABELS + UIDS
    unit_ids = unit_ids[unit_ids >= 0]
    # raw = denoising_tsvd is None
    raw = True
    prefix = "Raw" if raw else "Denoised"

    n_template_channels = recording.get_num_channels()
    registered_kdtree = None
    registered=False
    if registered_geom is not None:
        n_template_channels = len(registered_geom)
        registered_kdtree = KDTree(registered_geom)
        registered=True

    n_units = len(unit_ids)
    raw_templates = np.zeros(
        (n_units, spike_length_samples, n_template_channels),
        dtype=recording.dtype,
    )
    low_rank_templates = None
    # if not raw:
    #     low_rank_templates = np.zeros(
    #         (n_units, spike_length_samples, n_template_channels),
    #         dtype=recording.dtype,
    #     )
    # snrs_by_channel = np.zeros(
    #     (n_units, n_template_channels), dtype=recording.dtype
    # )

    raw_templates = []
    counts = []
    units_chunk = []

    # can parallelize here, snce we send wfs_all_loaded[in_unit] to each job 
    with h5py.File(h5_file, "r+") as h5:
        for u in unit_ids:
            if indices is None:
                in_unit = np.flatnonzero(sorting.labels == u)
            else:
                in_unit = np.flatnonzero(sorting.labels[indices] == u)
            pitch_shifts_unit = pitch_shifts[in_unit]
            if not in_unit.size:
                continue
            units_chunk.append(u)
            if registered:
            # with h5py.File(h5_file, "r+") as h5:
                wfs_all_loaded = chunked_h5_read(h5[wfs_name], in_unit) #HERE in unit + chunk
                channels = chunked_h5_read(h5["channels"], in_unit)
                # wfs_all_loaded = h5[wfs_name][:][in_unit]
                # channels = h5["channels"][:][in_unit]
                channel_index = h5["channel_index"][:]
                wfs_all_loaded = get_waveforms_on_static_channels(
                    wfs_all_loaded,
                    geom,
                    channels, 
                    channel_index, 
                    registered_geom=registered_geom,
                    n_pitches_shift=pitch_shifts_unit,
                )
                if weight_wfs is not None:
                    raw_templates.append(
                        reducer(wfs_all_loaded, weight_wfs[indices][in_unit], axis=0) #.numpy(force=True)
                    )
                else:
                    raw_templates.append(
                        reducer(wfs_all_loaded, axis=0) #.numpy(force=True)
                    )
                counts.append(
                    registered_template(
                        np.ones((in_unit.size, recording.get_num_channels())),
                        pitch_shifts_unit,
                        geom,
                        registered_geom,
                        min_fraction_at_shift=min_fraction_at_shift,
                        min_count_at_shift=min_count_at_shift,
                        registered_kdtree=registered_kdtree,
                        match_distance=pdist(geom).min() / 2,
                        reducer=np.nansum,
                    )
                )
            else:
                wfs_all_loaded = h5[wfs_name][:][in_unit]
                if weight_wfs is not None:
                    raw_templates.append(
                        reducer(wfs_all_loaded, weight_wfs[indices][in_unit], axis=0).numpy(force=True)
                    )
                else:
                    raw_templates.append(
                        reducer(wfs_all_loaded, axis=0).numpy(force=True)
                    )
    
                counts.append(in_unit.size)
    snrs_by_channel = np.array([ptp(rt, 0) * np.sqrt(c) for rt, c in zip(raw_templates, counts)]) # Here, need to return counts for smoothing
    raw_templates = np.array(raw_templates)

    # if denoising_tsvd is None:
    return raw_templates, None, snrs_by_channel, np.array(counts)


def get_all_shifted_raw_and_low_rank_templates_with_spikes_loaded(
    recording,
    wfs_all_loaded,
    indices_unique_inverse,
    indices_unique_index,
    labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
    channels_all, 
    registered_geom=None,
    denoising_tsvd=None,
    pitch_shifts=None,
    spikes_per_unit=500,
    reducer=fast_nanmedian,
    n_jobs=0,
    units_per_job=8,
    random_seed=0,
    show_progress=True,
    trough_offset_samples=42,
    spike_length_samples=121,
    min_fraction_at_shift=0.1,
    min_count_at_shift=5,
    device=None,
):

    """
    No parallelism yet
    """

    geom = recording.get_channel_locations()
    
    unit_ids = np.unique(labels_all) #CHANGE THIS WITH LABELS + UIDS
    unit_ids = unit_ids[unit_ids >= 0]
    raw = denoising_tsvd is None
    prefix = "Raw" if raw else "Denoised"

    n_template_channels = recording.get_num_channels()
    registered_kdtree = None
    registered=False
    if registered_geom is not None:
        n_template_channels = len(registered_geom)
        registered_kdtree = KDTree(registered_geom)
        registered=True

    n_units = len(unit_ids)
    raw_templates = np.zeros(
        (n_units, spike_length_samples, n_template_channels),
        dtype=recording.dtype,
    )
    low_rank_templates = None
    if not raw:
        low_rank_templates = np.zeros(
            (n_units, spike_length_samples, n_template_channels),
            dtype=recording.dtype,
        )
    snrs_by_channel = np.zeros(
        (n_units, n_template_channels), dtype=recording.dtype
    )

    raw_templates = []
    counts = []
    units_chunk = []

    # can parallelize here, snce we send wfs_all_loaded[in_unit] to each job 
    for u in unit_ids:
        in_unit = indices_unique_inverse[labels_all == u]
        pitch_shifts_unit = pitch_shifts[labels_all == u]
        if not in_unit.size:
            continue
        units_chunk.append(u)
        if registered:
            raw_templates.append(
                registered_template(
                    wfs_all_loaded[in_unit],
                    pitch_shifts_unit, #compute pitch_shifts
                    geom, #get geom from recording
                    registered_geom,
                    #pass all these arguments through other function
                    min_fraction_at_shift=min_fraction_at_shift,
                    min_count_at_shift=min_count_at_shift,
                    registered_kdtree=registered_kdtree,
                    match_distance=pdist(geom).min() / 2,
                    reducer=reducer,
                )
            )
            counts.append(
                registered_template(
                    np.ones((in_unit.size, recording.get_num_channels())),
                    pitch_shifts_unit,
                    geom,
                    registered_geom,
                    min_fraction_at_shift=min_fraction_at_shift,
                    min_count_at_shift=min_count_at_shift,
                    registered_kdtree=registered_kdtree,
                    match_distance=pdist(geom).min() / 2,
                    reducer=np.nansum,
                )
            )
        else:
            raw_templates.append(
                reducer(wfs_all_loaded[in_unit], axis=0).numpy(force=True)
            )
            counts.append(in_unit.size)
    snrs_by_channel = np.array([ptp(rt, 0) * np.sqrt(c) for rt, c in zip(raw_templates, counts)])
    raw_templates = np.array(raw_templates)

    if denoising_tsvd is None:
        return raw_templates, None, snrs_by_channel

    # nt, t, ct = raw_templates.shape
    # low_rank_templates = torch.tensor(raw_templates.transpose(0, 2, 1), device=p.device)
    # low_rank_templates = low_rank_templates.reshape(nt * ct, t)
    # low_rank_templates = p.denoising_tsvd(low_rank_templates, in_place=True)
    # low_rank_templates = low_rank_templates.view(nt, ct, t).permute(0, 2, 1)
    # low_rank_templates = low_rank_templates.numpy(force=True)
    
    # get low rank templates
    low_rank_templates = []
    for u in units_chunk:
        in_unit = indices_unique_inverse[labels_all == u]
        # apply denoising per unit (avoid huge matrix multiplication)
        n, t, c = wfs_all_loaded[in_unit].shape
        waveforms_unit = wfs_all_loaded[in_unit].transpose(0, 2, 1).reshape(n * c, t)
        waveforms_unit = denoising_tsvd(torch.tensor(waveforms_unit), in_place=True)
        waveforms_unit = waveforms_unit.reshape(n, c, t).permute(0, 2, 1)

        pitch_shifts_unit = pitch_shifts[labels_all == u]

        if registered:
            low_rank_templates.append(
                registered_template(
                    waveforms_unit,
                    pitch_shifts_unit,
                    geom,
                    registered_geom,
                    min_fraction_at_shift=min_fraction_at_shift,
                    min_count_at_shift=min_count_at_shift,
                    registered_kdtree=registered_kdtree,
                    match_distance=pdist(geom).min() / 2,
                    reducer=reducer,
                )
            )
        else:
            low_rank_templates.append(
                reducer(waveforms_unit, axis=0).numpy(force=True)
            )
    low_rank_templates = np.array(low_rank_templates)

    return raw_templates, low_rank_templates, snrs_by_channel
    

def get_all_shifted_raw_and_low_rank_templates(
    recording,
    sorting,
    registered_geom=None,
    denoising_tsvd=None,
    pitch_shifts=None,
    spikes_per_unit=500,
    reducer=fast_nanmedian,
    n_jobs=0,
    units_per_job=8,
    random_seed=0,
    show_progress=True,
    trough_offset_samples=42,
    spike_length_samples=121,
    min_fraction_at_shift=0.1,
    min_count_at_shift=5,
    dtype=np.float32,
    device=None,
):
    n_jobs, Executor, context, rank_queue = get_pool(
        n_jobs, with_rank_queue=True
    )
    unit_ids, spike_counts = np.unique(sorting.labels, return_counts=True)
    spike_counts = spike_counts[unit_ids >= 0]
    unit_ids = unit_ids[unit_ids >= 0]
    raw = denoising_tsvd is None
    prefix = "Raw" if raw else "Denoised"

    n_template_channels = recording.get_num_channels()
    registered_kdtree = None
    if registered_geom is not None:
        n_template_channels = len(registered_geom)
        registered_kdtree = KDTree(registered_geom)

    n_units = unit_ids.size
    raw_templates = np.zeros(
        (n_units, spike_length_samples, n_template_channels),
        dtype=dtype,
    )
    low_rank_templates = None
    if not raw:
        low_rank_templates = np.zeros(
            (n_units, spike_length_samples, n_template_channels),
            dtype=dtype,
        )
    snrs_by_channel = np.zeros(
        (n_units, n_template_channels), dtype=dtype
    )

    unit_id_chunks = [
        unit_ids[i : i + units_per_job]
        for i in range(0, n_units, units_per_job)
    ]

    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_template_process_init,
        initargs=(
            rank_queue,
            random_seed,
            recording,
            sorting,
            registered_kdtree,
            denoising_tsvd,
            pitch_shifts,
            spikes_per_unit,
            min_fraction_at_shift,
            min_count_at_shift,
            reducer,
            trough_offset_samples,
            spike_length_samples,
            device,
            units_per_job,
            dtype,
        ),
    ) as pool:
        # launch the jobs and wrap in a progress bar
        results = pool.map(_template_job, unit_id_chunks)
        if show_progress:
            pbar = tqdm(
                smoothing=0.01,
                desc=f"{prefix} templates",
                total=unit_ids.size,
                unit="template",
            )
        for res in results:
            if res is None:
                continue
            units_chunk, raw_temps_chunk, low_rank_temps_chunk, snrs_chunk = res
            ix_chunk = np.isin(unit_ids, units_chunk)
            raw_templates[ix_chunk] = raw_temps_chunk
            if not raw:
                low_rank_templates[ix_chunk] = low_rank_temps_chunk
            snrs_by_channel[ix_chunk] = snrs_chunk
            if show_progress:
                pbar.update(len(units_chunk))
        if show_progress:
            pbar.close()

    return unit_ids, spike_counts, raw_templates, low_rank_templates, snrs_by_channel


class TemplateProcessContext:
    def __init__(
        self,
        rg,
        recording,
        sorting,
        registered_kdtree,
        denoising_tsvd,
        pitch_shifts,
        spikes_per_unit,
        min_fraction_at_shift,
        min_count_at_shift,
        reducer,
        trough_offset_samples,
        spike_length_samples,
        device,
        units_per_job,
        dtype,
    ):
        self.n_channels = recording.get_num_channels()
        self.registered = registered_kdtree is not None

        self.rg = rg
        self.device = device
        self.dtype = dtype
        self.recording = recording
        self.sorting = sorting
        self.denoising_tsvd = denoising_tsvd
        
        if denoising_tsvd is not None:
            # self.denoising_tsvd = TorchSVDProjector(
            #     torch.from_numpy(
            #         denoising_tsvd.components_.astype(dtype)
            #     )
            # )
            self.denoising_tsvd.to(self.device)
            
        self.spikes_per_unit = spikes_per_unit
        self.reducer = reducer
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.max_spike_time = recording.get_num_samples() - (
            spike_length_samples - trough_offset_samples
        )
        self.min_fraction_at_shift = min_fraction_at_shift
        self.min_count_at_shift = min_count_at_shift

        self.spike_buffer = torch.zeros(
            (
                spikes_per_unit * units_per_job,
                spike_length_samples,
                self.n_channels,
            ),
            device=device,
            dtype=torch.from_numpy(np.zeros(1, dtype=dtype)).dtype,
        )

        self.n_template_channels = self.n_channels
        if self.registered:
            self.geom = recording.get_channel_locations()
            self.match_distance = pdist(self.geom).min() / 2
            self.registered_geom = registered_kdtree.data
            self.registered_kdtree = registered_kdtree
            self.pitch_shifts = pitch_shifts
            self.n_template_channels = len(self.registered_geom)


_template_process_context = None


def _template_process_init(
    rank_queue,
    random_seed,
    recording,
    sorting,
    registered_kdtree,
    denoising_tsvd,
    pitch_shifts,
    spikes_per_unit,
    min_fraction_at_shift,
    min_count_at_shift,
    reducer,
    trough_offset_samples,
    spike_length_samples,
    device,
    units_per_job,
    dtype,
):
    global _template_process_context

    rank = rank_queue.get()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        if torch.cuda.device_count() > 1:
            device = torch.device(
                "cuda", index=rank % torch.cuda.device_count()
            )
    torch.set_grad_enabled(False)

    rg = np.random.default_rng(random_seed + rank)
    _template_process_context = TemplateProcessContext(
        rg,
        recording,
        sorting,
        registered_kdtree,
        denoising_tsvd,
        pitch_shifts,
        spikes_per_unit,
        min_fraction_at_shift,
        min_count_at_shift,
        reducer,
        trough_offset_samples,
        spike_length_samples,
        device,
        units_per_job,
        dtype,
    )


def _template_job(unit_ids):
    p = _template_process_context

    in_units_full = np.flatnonzero(np.isin(p.sorting.labels, unit_ids))
    if not in_units_full.size:
        return
    labels_full = p.sorting.labels[in_units_full]

    # only so many spikes per unit
    uids, counts = np.unique(labels_full, return_counts=True)
    n_spikes_grab = np.minimum(counts, p.spikes_per_unit).sum()
    in_units = np.empty(n_spikes_grab, dtype=in_units_full.dtype)
    labels = np.empty(n_spikes_grab, dtype=labels_full.dtype)
    offset = 0
    for u, c in zip(uids, counts):
        if c > p.spikes_per_unit:
            in_unit = p.rg.choice(
                in_units_full[labels_full == u],
                p.spikes_per_unit,
                replace=False,
            )
            in_units[offset : offset + min(c, p.spikes_per_unit)] = in_unit
            labels[offset : offset + min(c, p.spikes_per_unit)] = u
        else:
            in_units[offset : offset + c] = in_units_full[labels_full == u]
            labels[offset : offset + c] = u
        offset += min(c, p.spikes_per_unit)
    order = np.argsort(in_units)
    in_units = in_units[order]
    labels = labels[order]

    # read waveforms for all units
    times = p.sorting.times_samples[in_units]
    valid = np.flatnonzero(
        (times >= p.trough_offset_samples) & (times <= p.max_spike_time)
    )
    if not valid.size:
        return
    in_units = in_units[valid]
    labels = labels[valid]
    times = times[valid]
    waveforms = spikeio.read_full_waveforms(
        p.recording,
        times,
        trough_offset_samples=p.trough_offset_samples,
        spike_length_samples=p.spike_length_samples,
    )
    p.spike_buffer[: times.size] = torch.from_numpy(waveforms.astype(p.dtype))
    waveforms = p.spike_buffer[: times.size]
    n, t, c = waveforms.shape

    # compute raw templates and spike counts per channel
    raw_templates = []
    counts = []
    units_chunk = []
    for u in uids:
        in_unit = np.flatnonzero(labels == u)
        if not in_unit.size:
            continue
        units_chunk.append(u)
        in_unit_orig = in_units[labels == u]
        if p.registered:
            raw_templates.append(
                registered_template(
                    waveforms[in_unit],
                    p.pitch_shifts[in_unit_orig],
                    p.geom,
                    p.registered_geom,
                    min_fraction_at_shift=p.min_fraction_at_shift,
                    min_count_at_shift=p.min_count_at_shift,
                    registered_kdtree=p.registered_kdtree,
                    match_distance=p.match_distance,
                    reducer=p.reducer,
                )
            )
            counts.append(
                registered_template(
                    np.ones((in_unit.size, p.n_channels)),
                    p.pitch_shifts[in_unit_orig],
                    p.geom,
                    p.registered_geom,
                    min_fraction_at_shift=p.min_fraction_at_shift,
                    min_count_at_shift=p.min_count_at_shift,
                    registered_kdtree=p.registered_kdtree,
                    match_distance=p.match_distance,
                    reducer=np.nansum,
                )
            )
        else:
            raw_templates.append(
                p.reducer(waveforms[in_unit], axis=0).numpy(force=True)
            )
            counts.append(in_unit.size)
    snrs_by_chan = [ptp(rt, 0) * c for rt, c in zip(raw_templates, counts)]
    raw_templates = np.array(raw_templates)

    if p.denoising_tsvd is None:
        return units_chunk, raw_templates, None, snrs_by_chan

    # nt, t, ct = raw_templates.shape
    # low_rank_templates = torch.tensor(raw_templates.transpose(0, 2, 1), device=p.device)
    # low_rank_templates = low_rank_templates.reshape(nt * ct, t)
    # low_rank_templates = p.denoising_tsvd(low_rank_templates, in_place=True)
    # low_rank_templates = low_rank_templates.view(nt, ct, t).permute(0, 2, 1)
    # low_rank_templates = low_rank_templates.numpy(force=True)

    # apply denoising
    waveforms = waveforms.permute(0, 2, 1).reshape(n * c, t)
    waveforms = p.denoising_tsvd(waveforms, in_place=True)
    waveforms = waveforms.reshape(n, c, t).permute(0, 2, 1)

    # get low rank templates
    low_rank_templates = []
    for u in units_chunk:
        in_unit = np.flatnonzero(labels == u)
        in_unit_orig = in_units[labels == u]
        if p.registered:
            low_rank_templates.append(
                registered_template(
                    waveforms[in_unit],
                    p.pitch_shifts[in_unit_orig],
                    p.geom,
                    p.registered_geom,
                    min_fraction_at_shift=p.min_fraction_at_shift,
                    min_count_at_shift=p.min_count_at_shift,
                    registered_kdtree=p.registered_kdtree,
                    match_distance=p.match_distance,
                    reducer=p.reducer,
                )
            )
        else:
            low_rank_templates.append(
                p.reducer(waveforms[in_unit], axis=0).numpy(force=True)
            )
    low_rank_templates = np.array(low_rank_templates)

    return units_chunk, raw_templates, low_rank_templates, snrs_by_chan


class TorchSVDProjector(torch.nn.Module):
    def __init__(self, components):
        super().__init__()
        self.register_buffer("components", components)

    def forward(self, x, in_place=False):
        embed = x @ self.components.T
        out = x if in_place else None
        return torch.matmul(embed, self.components, out=out)
