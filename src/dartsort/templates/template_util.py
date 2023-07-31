import numpy as np
from dartsort import transform
from dartsort.peel.grab import GrabAndFeaturize


def get_templates(
    times,
    channels,
    labels,
    geom=None,
    pitch_shifts=None,
    extended_geom=None,
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
    reducer=np.median,
    random_state=0,
    n_jobs=-1,
):
    """Raw, denoised, and shifted templates

    Low-level helper function which does the work of template computation for
    the template classes (StaticTemplates and SuperresTemplates) in other files
    in this folder.

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
    """
    # validate arguments
    assert (labels >= 0).all()
    assert times.shape == channels.shape == labels.shape
    assert times.dtype.kind == "i"
    assert channels.dtype.kind == "i"
    assert labels.dtype.kind == "i"
    if low_rank_denoising and denoising_fit_radius is not None:
        assert geom is not None
