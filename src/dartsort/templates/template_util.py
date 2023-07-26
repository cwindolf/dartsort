import numpy as np
from dartsort import transform
from dartsort.peel.grab import GrabAndFeaturize

# virtual geometry is per unit or global?
# the 3P-1 thing has to be per unit... global is easier
# but it would be nice to do some channel index in the global
# virtual geometry per unit


def get_templates(
    times,
    channels,
    labels,
    geom=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    low_rank_denoising=True,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    pitch_shifts=None,
    zero_radius_um=None,
    reducer=np.median,
    random_state=0,
    n_jobs=-1,
):
    """Raw and denoised templates

    Low-level helper function which does the work of template
    computation for the template classes (StaticTemplates and
    SuperresTemplates) in other files in this folder.

    Arguments
    ---------
    times, channels, labels : arrays of shape (n_spikes,)
        The trough (or peak) times, main channels, and unit labels.
        The superres template class will replace unit labels with
        the unique values of (original unit label, superres bin).
    geom : array of shape (n_channels, 2)
        Probe channel geometry, needed to subsample channels when
        fitting the low-rank denoising model, and also needed
        if the shifting arguments are specified
    trough_offset_samples, spike_length_samples : int
        Waveform snippets will be loaded from times[i] - trough_offset_samples
        to times[i] - trough_offset_samples + spike_length_samples
    spikes_per_unit : int
        Load at most this many randomly selected spikes per unit
    low_rank_denoising : bool
        Should we compute denoised templates? If not, raw averages.
    denoising_rank, denoising_fit_radius, denoising_spikes_fit
        Parameters for the low rank model fit for denoising
    denoising_snr_threshold : int
        The SNR (=amplitude*sqrt(n_spikes)) threshold at which the
        denoising is ignored and we just use the usual template
    pitch_shifts : int
        When computing superres templates, these values will be used
        to shift the templates to the correct channel neighborhoods
        before averaging

    """
    # validate arguments
    assert (labels >= 0).all()
    assert times.shape == channels.shape == labels.shape
    assert times.dtype.kind == "i"
    assert channels.dtype.kind == "i"
    assert labels.dtype.kind == "i"
    if low_rank_denoising and denoising_fit_radius is not None:
        assert geom is not None
