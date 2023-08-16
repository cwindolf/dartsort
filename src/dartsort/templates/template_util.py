"""Compute template waveforms, including accounting for drift and denoising

This file is low-level: functions return arrays. The classes in templates.py
provide a more convenient interface.


 // About get_templates

get_templates has been written in such a way that it can realign and denoise
templates in a single pass through the data, making use of the GrabAndFeaturize
peeler to store temporary waveforms in an HDF5 file.

This makes the logic a bit complicated to follow just right off the bat reading
the code. So here's what is going on and why.

The steps of the algorithm are:

 - Grab raw waveforms corresponding to the input spike trains in `sorting`
    - The sorting is subsampled to at most spikes_per_unit spikes
        - But, if low_rank_denoising=True, we need to make sure that we
          extract at least denoising_spikes_fit waveforms. So we keep around
          some extra waveforms with label -1.
    - If realign_peaks is True, these waveforms are padded according to
      realign_max_shift. After extracting all the waveforms for a unit,
      we search for its new peak/trough (see next bullet), and this padding
      ensures that we can still extract a full-length (spike_length_samples)
      waveform after realigning to the new peak
 - Realignment
    - The padded raw waveforms are averaged, and we search for the template's
      max abs value sample in a radius of realign_max_shift around the current
      putative trough/peak time (trough_offset_samples)
    - The templates are cropped such that the new peak lands at trough_offset_samples
    - The sorting is also realigned according to these shifts
        > Note, the sorting might not be sorted by time after this step.
          Our pipeline should never assume that sortings are ordered by time,
          but we will output time-ordered sortings from peeling operations
          and at the end of the pipeline
 - If low_rank_denoising=False, return
    - We return the aligned spike train and templates
    - Also, if the user passed in output_hdf5_filename and
      keep_waveforms_in_hdf5=True, we align the waveforms in the hdf5
      file for the user. This means cropping each waveform according
      to its unit's shift
 - Otherwise, low_rank_denoising=True, so we apply the following denoising:
    - First, fit a temporal PCA to the extracted raw waveform traces.
      We need these to be aligned and of length spike_length_samples,
      so we run the individual waveform cropping routine of the previous
      bullet
    - Next, apply the temporal PCA to all the waveforms
    - Finally, weighted averaging is applied
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
    denoising_tpca=None,
    denoising_centered=False,
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

    The file this function lives in has an explanation of the steps taken to
    compute the templates in __doc__

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
        None, that hdf5 file is used and this argument is ignored.

    Returns
    -------
    dict whose keys vary based on the above arguments
     - "raw_templates" (always)
     - "denoised_templates" if low_rank_denoising
     - "tpca" (sklearn PCA or TruncatedSVD object) if low_rank_denoising
     - "sorting", which is the realigned sorting if realign_peaks is True
       and otherwise just the one you passed in
     - "output_hdf5_filename" if output_hdf5_filename is not None
       This hdf5 file will contain datasets raw_templates and optionally
       denoised_templates, and also "waveforms", "channel_index", "labels",
       "channels", etc if keep_waveforms_in_hdf5 (corresponding to the
       subsampled spike train)
    """
    # See this file's __doc__str at the very top of the file for
    # details about why this function performs the steps below

    # validate arguments
    if low_rank_denoising and denoising_fit_radius is not None:
        assert geom is not None

    raw_only = not low_rank_denoising

    # figure out if hdf5 output is requested, and where to store
    # a temporary hdf5 file for extracting waveforms if not
    return_hdf5 = output_hdf5_filename is not None
    if return_hdf5:
        output_hdf5_filename = Path(output_hdf5_filename)
    else:
        tempdir = tempfile.TemporaryDirectory(
            dir=scratch_dir, prefix="dartsort_templates"
        )
        output_hdf5_filename = tempdir / "dartsort_templates.h5"

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

        if raw_only:
            # overwrite template dataset with aligned ones
            # handle keep_waveforms_in_hdf5
            return dict(sorting=sorting, raw_templates=templates)


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
