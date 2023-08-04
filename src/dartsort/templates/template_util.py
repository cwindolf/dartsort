"""Compute template waveforms, including accounting for drift and denoising

This file is low-level: functions return arrays. The classes in templates.py

"""
import tempfile
from pathlib import Path

import numpy as np
from dartsort import transform
from dartsort.peel.grab import GrabAndFeaturize


def get_templates(
    sorting,
    geom=None,
    pitch_shifts=None,
    extended_geom=None,
    realign_peaks=True,
    realign_max_shift=20,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    low_rank_denoising=True,
    denoising_model=None,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    zero_radius_um=None,
    reducer=np.nanmedian,
    random_state=0,
    output_hdf5_filename=None,
    keep_waveforms_in_hdf5=False,
    scratch_dir=None,
    n_jobs=0,
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
    extended_geom : array of shape (n_channels_extended, 2)
        Required if pitch_shifts is supplied. See drift_util.extended_geometry.
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
        None, that hdf5 file is used.

    Returns
    -------
    dict whose keys vary based on the above arguments
     - "raw_templates" (always)
     - "denoised_templates" if low_rank_denoising
     - "sorting", which is the realigned sorting if realign_peaks is True
       and otherwise just the one you passed in
     - "output_hdf5_filename" if output_hdf5_filename is not None
       This hdf5 file will contain datasets raw_templates and optionally
       denoised_templates, and also "waveforms", "channel_index", etc
       if keep_waveforms_in_hdf5
    """
    # validate arguments
    if low_rank_denoising and denoising_fit_radius is not None:
        assert geom is not None

    return_hdf5 = output_hdf5_filename is not None
    if return_hdf5:
        output_hdf5_filename = Path(output_hdf5_filename)
        scratch_dir = output_hdf5_filename.parent

    # estimate peak sample times and realign spike train
    if realign_peaks:
        # pad the trough_offset_samples and spike_length_samples so that
        # if the user did not request denoising we can just return the
        # raw templates right away
        trough_offset_load = trough_offset_samples + realign_max_shift
        spike_length_load = spike_length_samples + 2 * realign_max_shift
        templates = get_raw_templates(
            sorting,
            geom=geom,
            pitch_shifts=pitch_shifts,
            extended_geom=extended_geom,
            realign_peaks=False,
            trough_offset_samples=trough_offset_load,
            spike_length_samples=spike_length_load,
            spikes_per_unit=spikes_per_unit,
            zero_radius_um=zero_radius_um,
            reducer=reducer,
            random_state=random_state,
            output_hdf5_filename=None,
            scratch_dir=scratch_dir,
            n_jobs=n_jobs,
        )
        sorting, templates = realign_sorting(
            sorting,
            templates,
            max_shift=realign_max_shift,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )

        if not low_rank_denoising:
            return


def get_raw_templates(
    sorting,
    geom=None,
    pitch_shifts=None,
    extended_geom=None,
    realign_peaks=False,
    realign_max_shift=20,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    zero_radius_um=None,
    reducer=np.nanmedian,
    random_state=0,
    output_hdf5_filename=None,
    keep_waveforms_in_hdf5=False,
    scratch_dir=None,
    n_jobs=0,
):
    pass


def realign_sorting(
    sorting,
    templates,
    max_shift=20,
    trough_offset_samples=42,
    spike_length_samples=121,
):
    pass
