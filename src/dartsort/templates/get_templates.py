"""Compute template waveforms, including accounting for drift and denoising

The class TemplateData in templates.py provides a friendlier interface,
where you can get templates using the TemplateConfig in config.py.
"""
from dataclasses import replace

import numpy as np
import torch
from dartsort.util import spikeio
from dartsort.util.drift_util import registered_template, get_waveforms_on_static_channels
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

    # NO DENOISING YET!
    # if low_rank_denoising and denoising_tsvd is None:
    #     denoising_tsvd = fit_tsvd_with_h5(
    #         # TODO: WRITE THIS FUNCTION
    #         h5_file,
    #         sorting,
    #         wfs_name, 
    #         denoising_rank=denoising_rank,
    #         denoising_fit_radius=denoising_fit_radius,
    #         denoising_spikes_fit=denoising_spikes_fit,
    #         trough_offset_samples=trough_offset_samples,
    #         spike_length_samples=spike_length_samples,
    #         random_seed=random_seed,
    #     )
    # if denoising_tsvd is not None:
    #     denoising_tsvd = TorchSVDProjector(
    #         torch.from_numpy(
    #             denoising_tsvd.components_.astype(recording.dtype)
    #         )
    #     )

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



def get_templates_with_spikes_loaded(
    recording,
    wfs_all_loaded,
    indices_unique_inverse,
    indices_unique_index,
    sorting, #change all params to sorting.
    labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
    channels_all, 
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
    min_fraction_at_shift=0.25,
    min_count_at_shift=25,
    reducer=fast_nanmedian,
    spatial_svdsmoothing=False,
    max_ptp_chans_to_spatialsmooth=3,
    rank_spatial_svd=5,
    random_seed=0,
    units_per_job=8,
    n_jobs=0,
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
        raw_results = get_raw_templates_with_spikes_loaded(
            recording,
            wfs_all_loaded,
            indices_unique_inverse,
            indices_unique_index,
            labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
            channels_all, 
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

    # fit tsvd
    if low_rank_denoising and denoising_tsvd is None:
        denoising_tsvd = fit_tsvd_with_spikes_loaded(
            recording,
            wfs_all_loaded,
            indices_unique_inverse,
            indices_unique_index,
            labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
            channels_all, 
            denoising_rank=denoising_rank,
            denoising_fit_radius=denoising_fit_radius,
            denoising_spikes_fit=denoising_spikes_fit,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            random_seed=random_seed,
        )
    if denoising_tsvd is not None:
        denoising_tsvd = TorchSVDProjector(
            torch.from_numpy(
                denoising_tsvd.components_.astype(recording.dtype)
            )
        )

    # template logic
    # for each unit, get shifted raw and denoised averages and channel SNRs
    res = get_all_shifted_raw_and_low_rank_templates_with_spikes_loaded(
        recording,
        wfs_all_loaded,
        indices_unique_inverse,
        indices_unique_index,
        labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
        channels_all, 
        registered_geom=registered_geom,
        denoising_tsvd=denoising_tsvd,
        pitch_shifts=pitch_shifts, #pitch_shifts have same structure as labels_all
        spikes_per_unit=spikes_per_unit,
        reducer=reducer,
        n_jobs=n_jobs,
        units_per_job=units_per_job,
        random_seed=random_seed,
        show_progress=show_progress,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        min_fraction_at_shift=min_fraction_at_shift,
        min_count_at_shift=min_count_at_shift,
        device=device,
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
    templates = templates.astype(recording.dtype)

    if spatial_svdsmoothing:
        for k in range(templates.shape[0]):
            # chans_low_ptp = np.flatnonzero(templates[k].ptp(0)<max_ptp_chans_to_spatialsmooth)
            chans_low_ptp = np.flatnonzero(np.logical_or(
                templates[k].ptp(0)<max_ptp_chans_to_spatialsmooth,
                np.sqrt(((registered_geom - registered_geom[templates[k].ptp(0).argmax()][None])**2).sum(1))>50
            ))
            # chans_low_ptp = np.flatnonzero(snrs_by_channel[k]<denoising_snr_threshold)
            # print(chans_low_ptp.shape)
            # temp_pre = templates[k].copy()
            U, s, Vh = svd(templates[k][:, chans_low_ptp].T, full_matrices=False)
            #make sure not to set everything to 0
            # if s[0]<s.sum()*percent_svd_retain:
            #     # print("in percentage")
            #     s[np.cumsum(s)>s.sum()*percent_svd_retain]=0
            # else:
            #     # print("in first component")
            #     s[1:]=0
            s[rank_spatial_svd:]=0
            templates[k, :, chans_low_ptp] = np.dot(U, np.dot(np.diag(s), Vh))
            # print((temp_pre - templates[k]).ptp(0).max())

    return dict(
        sorting=sorting,
        templates=templates,
        raw_templates=raw_templates,
        low_rank_templates=low_rank_templates,
        snrs_by_channel=snrs_by_channel,
        weights=weights,
    )


def get_templates_linear(
    recording,
    sorting,
    n_chunks=1,
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
    min_fraction_at_shift=0.25,
    min_count_at_shift=25,
    reducer=fast_nanmedian,
    spatial_svdsmoothing=False,
    max_ptp_chans_to_spatialsmooth=3,
    random_seed=0,
    units_per_job=8,
    n_jobs=0,
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
    # if realign_peaks:
    #     # pad the trough_offset_samples and spike_length_samples so that
    #     # if the user did not request denoising we can just return the
    #     # raw templates right away
    #     trough_offset_load = trough_offset_samples + realign_max_sample_shift
    #     spike_length_load = spike_length_samples + 2 * realign_max_sample_shift
    #     raw_results = get_raw_templates(
    #         recording,
    #         sorting,
    #         pitch_shifts=pitch_shifts,
    #         registered_geom=registered_geom,
    #         realign_peaks=False,
    #         trough_offset_samples=trough_offset_load,
    #         spike_length_samples=spike_length_load,
    #         spikes_per_unit=spikes_per_unit,
    #         min_fraction_at_shift=min_fraction_at_shift,
    #         min_count_at_shift=min_count_at_shift,
    #         reducer=reducer,
    #         random_seed=random_seed,
    #         n_jobs=n_jobs,
    #         show_progress=show_progress,
    #         device=device,
    #     )
    #     sorting, templates = realign_sorting(
    #         sorting,
    #         raw_results["raw_templates"],
    #         raw_results["snrs_by_channel"],
    #         max_shift=realign_max_sample_shift,
    #         trough_offset_samples=trough_offset_samples,
    #         recording_length_samples=recording.get_num_samples(),
    #     )
    #     if raw_only:
    #         # overwrite template dataset with aligned ones
    #         # handle keep_waveforms_in_hdf5
    #         raw_results["sorting"] = sorting
    #         raw_results["templates"] = raw_results["raw_templates"] = templates
    #         return raw_results

    # fit tsvd
    if low_rank_denoising and denoising_tsvd is None:
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

    # template logic
    # for each unit, get shifted raw and denoised averages and channel SNRs
    res = get_all_shifted_raw_and_low_rank_templates_linear(
        recording,
        sorting,
        n_chunks=n_chunks,
        registered_geom=registered_geom,
        denoising_tsvd=denoising_tsvd,
        pitch_shifts=pitch_shifts,
        spikes_per_unit=spikes_per_unit,
        reducer=reducer,
        n_jobs=n_jobs,
        units_per_job=units_per_job,
        random_seed=random_seed,
        show_progress=show_progress,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        min_fraction_at_shift=min_fraction_at_shift,
        min_count_at_shift=min_count_at_shift,
        device=device,
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
    templates = templates.astype(recording.dtype)

    if spatial_svdsmoothing:
        for k in range(templates.shape[0]):
            chans_low_ptp = np.flatnonzero(templates[k].ptp(0)<max_ptp_chans_to_spatialsmooth)
            U, s, Vh = svd(templates[k][:, chans_low_ptp].T, full_matrices=False)
            s[np.cumsum(s)>s.sum()*0.5]=0
            templates[k, :, chans_low_ptp] = np.dot(U, np.dot(np.diag(s), Vh))

    return dict(
        sorting=sorting,
        templates=templates,
        raw_templates=raw_templates,
        low_rank_templates=low_rank_templates,
        snrs_by_channel=snrs_by_channel,
        weights=weights,
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
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    min_fraction_at_shift=0.25,
    min_count_at_shift=25,
    reducer=fast_nanmedian,
    spatial_svdsmoothing=False,
    max_ptp_chans_to_spatialsmooth=3,
    random_seed=0,
    units_per_job=8,
    n_jobs=0,
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

    # fit tsvd
    if low_rank_denoising and denoising_tsvd is None:
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
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        min_fraction_at_shift=min_fraction_at_shift,
        min_count_at_shift=min_count_at_shift,
        device=device,
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
    templates = templates.astype(recording.dtype)

    if spatial_svdsmoothing:
        for k in range(templates.shape[0]):
            chans_low_ptp = np.flatnonzero(templates[k].ptp(0)<max_ptp_chans_to_spatialsmooth)
            U, s, Vh = svd(templates[k][:, chans_low_ptp].T, full_matrices=False)
            s[np.cumsum(s)>s.sum()*0.5]=0
            templates[k, :, chans_low_ptp] = np.dot(U, np.dot(np.diag(s), Vh))

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
        show_progress=show_progress,
        device=device,
    )

def get_raw_templates_with_spikes_loaded(
    recording,
    wfs_all_loaded,
    indices_unique_inverse,
    indices_unique_index,
    labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
    channels_all, 
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    pitch_shifts=None,
    registered_geom=None,
    realign_peaks=False,
    realign_max_sample_shift=20,
    min_fraction_at_shift=0.25,
    min_count_at_shift=25,
    reducer=fast_nanmedian,
    random_seed=0,
    n_jobs=0,
    show_progress=True,
    device=None,
):
    return get_templates_with_spikes_loaded(
        recording,
        wfs_all_loaded,
        indices_unique_inverse,
        indices_unique_index,
        labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
        channels_all, 
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
        show_progress=show_progress,
        device=device,
    )


# -- helpers


def realign_sorting(
    sorting,
    templates,
    snrs_by_channel,
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
    template_shifts = template_peak_times - (trough_offset_samples + max_shift)
    template_shifts[np.abs(template_shifts) > max_shift] = 0

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
    for i, dt in enumerate(template_shifts):
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
    waveforms = waveforms.transpose(0, 2, 1)
    waveforms = waveforms.reshape(len(times) * tsvd_channel_index.shape[1], -1)

    # reshape, fit tsvd, and done
    tsvd = TruncatedSVD(n_components=denoising_rank, random_state=random_seed)
    tsvd.fit(waveforms)

    return tsvd

def fit_tsvd_with_spikes_loaded(
    recording,
    wfs_all_loaded,
    indices_unique_inverse,
    indices_unique_index,
    labels_all, #wfs_all_loaded[indices_unique_inverse[labels_all == k]] gives all wfs for unit k 
    channels_all, #channels_all[indices_unique_index] gives a set of max channel usable for fitting tsvd
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=10_000,
    trough_offset_samples=42,
    spike_length_samples=121,
    random_seed=0,
):
    geom = recording.get_channel_locations()
    channel_index = full_channel_index(geom.shape[0])

    wfs_all_loaded = wfs_all_loaded[indices_unique_inverse]

    if denoising_spikes_fit<wfs_all_loaded.shape[0]:
        rg = np.random.default_rng(random_seed)
    
        choices = rg.choice(wfs_all_loaded.shape[0], denoising_spikes_fit, replace=False)
        wfs_all_loaded = wfs_all_loaded[choices]
        channels_all = channels_all[choices]

    # grab waveforms
    wfs_all_loaded = channel_subset_by_radius(
        wfs_all_loaded,
        channels_all,
        channel_index, #all
        recording.get_channel_locations(),
        radius=denoising_fit_radius,
        fill_value=0.0, # all-0 rows don't change SVD basis
        return_new_channel_index=False,
    )
    wfs_all_loaded = wfs_all_loaded.transpose(0, 2, 1)
    wfs_all_loaded = wfs_all_loaded.reshape(wfs_all_loaded.shape[0] * wfs_all_loaded.shape[1], -1)

    # reshape, fit tsvd, and done
    tsvd = TruncatedSVD(n_components=denoising_rank, random_state=random_seed)
    tsvd.fit(wfs_all_loaded)
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
                wfs_all_loaded = h5[wfs_name][:][in_unit]
                channels = h5["channels"][:][in_unit]
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
    device=None,
):
    n_jobs, Executor, context, rank_queue = get_pool(
        n_jobs, with_rank_queue=True
    )
    unit_ids = np.unique(sorting.labels)
    unit_ids = unit_ids[unit_ids >= 0]
    raw = denoising_tsvd is None
    prefix = "Raw" if raw else "Denoised"

    n_template_channels = recording.get_num_channels()
    registered_kdtree = None
    if registered_geom is not None:
        n_template_channels = len(registered_geom)
        registered_kdtree = KDTree(registered_geom)

    n_units = sorting.labels.max() + 1
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
            raw_templates[units_chunk] = raw_temps_chunk
            if not raw:
                low_rank_templates[units_chunk] = low_rank_temps_chunk
            snrs_by_channel[units_chunk] = snrs_chunk
            if show_progress:
                pbar.update(len(units_chunk))
        if show_progress:
            pbar.close()

    return raw_templates, low_rank_templates, snrs_by_channel


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
    ):
        self.n_channels = recording.get_num_channels()
        self.registered = registered_kdtree is not None

        self.rg = rg
        self.device = device
        self.recording = recording
        self.sorting = sorting
        self.denoising_tsvd = denoising_tsvd
        if denoising_tsvd is not None:
            self.denoising_tsvd = TorchSVDProjector(
                torch.from_numpy(
                    denoising_tsvd.components_.astype(recording.dtype)
                )
            )
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
            dtype=torch.from_numpy(np.zeros(1, dtype=recording.dtype)).dtype,
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
    p.spike_buffer[: times.size] = torch.from_numpy(waveforms)
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
