import numpy as np
from spikeinterface.core import BaseRecording
from sklearn.decomposition import PCA, TruncatedSVD

from ..util.internal_config import (
    TemplateSVDMethod,
    WaveformConfig,
    TemplateConfig,
    ComputationConfig,
    default_waveform_cfg,
)
from ..util.data_util import DARTsortSorting, load_stored_tsvd
from ..util.waveform_util import make_channel_index
from ..util.spikeio import read_waveforms_channel_index
from .templates import TemplateData


def fit_tsvd(
    *,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    motion_est,
    denoising_rank=5,
    denoising_fit_radius=75.0,
    denoising_spikes_fit=25_000,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    computation_cfg: ComputationConfig | None = None,
    svd_method: TemplateSVDMethod = "spike_sklearn",
    svd_input_templates: TemplateData | None = None,
    dtype=np.float32,
    random_seed=0,
    n_iter=15,
) -> PCA | TruncatedSVD:
    if svd_method == "collisioncleaned":
        tsvd = load_stored_tsvd(sorting, trim_rank_to=denoising_rank)
        assert isinstance(tsvd, (TruncatedSVD, PCA))
        return tsvd

    if svd_method == "raw_template":
        from .template_util import shared_basis_compress_templates

        if svd_input_templates is None:
            td = quick_mean_templates(
                recording=recording,
                sorting=sorting,
                waveform_cfg=waveform_cfg,
                computation_cfg=computation_cfg,
                motion_est=motion_est,
            )
        else:
            td = svd_input_templates
        tdc = shared_basis_compress_templates(
            td, rank=denoising_rank, computation_cfg=computation_cfg
        )
        basis = tdc.temporal_components
        pca = PCA(n_components=denoising_rank, random_state=random_seed, whiten=False)
        pca.mean_ = np.zeros_like(td.templates[0, :, 0])
        pca.components_ = basis
        pca.explained_variance_ = np.full_like(basis[:, 0], np.nan)
        return pca

    assert svd_method == "spike_sklearn"
    assert sorting.labels is not None

    trough_offset_samples = waveform_cfg.trough_offset_samples(
        recording.sampling_frequency
    )
    spike_length_samples = waveform_cfg.spike_length_samples(
        recording.sampling_frequency
    )

    # read spikes on channel neighborhood
    geom = recording.get_channel_locations()
    tsvd_channel_index = make_channel_index(geom, denoising_fit_radius)

    # subset spikes used to fit tsvd
    rg = np.random.default_rng(random_seed)
    max_time = recording.get_num_samples() - (
        spike_length_samples - trough_offset_samples
    )
    t_clip = sorting.times_samples.clip(trough_offset_samples, max_time)
    valid = np.logical_and(sorting.labels >= 0, sorting.times_samples == t_clip)
    choices = np.flatnonzero(valid)
    if choices.size > denoising_spikes_fit:
        choices = rg.choice(choices, denoising_spikes_fit, replace=False)
        choices.sort()
    times = sorting.times_samples[choices]
    channels = sorting.channels[choices]

    # grab waveforms
    waveforms = read_waveforms_channel_index(
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
    tsvd = TruncatedSVD(
        n_components=denoising_rank, random_state=random_seed, n_iter=n_iter
    )
    tsvd.fit(waveforms)

    return tsvd


def denoising_weights(
    *,
    snrs: np.ndarray,
    spike_length_samples: int,
    trough_offset: int,
    snr_threshold: float,
    a=12.0,
    b=12.0,
    d=6.0,
    edge_behavior="saturate",
) -> np.ndarray:
    """Weights are applied to raw template, 1-weights to low rank"""
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
    else:
        assert False

    # pass it through a hand picked squashing function
    wntc = 1.0 / (1.0 + np.exp(d + a * vt[None, :, None] - b * snc[:, None, :]))

    return wntc


def quick_mean_templates(
    *,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    waveform_cfg: WaveformConfig,
    computation_cfg: ComputationConfig | None,
    motion_est,
):
    return TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        motion_est=motion_est,
        waveform_cfg=waveform_cfg,
        template_cfg=TemplateConfig(denoising_method="none"),
        computation_cfg=computation_cfg,
    )
