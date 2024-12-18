import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

from dartsort.util import spiketorch, waveform_util


def get_singlechan_centroids(
    singlechan_waveforms=None,
    rec=None,
    detection_threshold=6.0,
    deduplication_radius=150.0,
    trough_offset_samples=42,
    spike_length_samples=121,
    alignment_padding=20,
    n_centroids=10,
    pca_rank=8,
    n_waveforms_fit=20_000,
    taper=True,
    taper_start=20,
    taper_end=30,
    kmeanspp_initial="random",
    random_seed=0,
):
    """Kmeanspp in pca space"""
    from dartsort import kmeans

    full_length = spike_length_samples + 2 * alignment_padding
    full_trough = trough_offset_samples + alignment_padding
    trim_end = spike_length_samples + alignment_padding

    # a user may have better waveforms than we can provide naively
    # with thresholding detection below
    if singlechan_waveforms is None:
        singlechan_waveforms = get_singlechan_waveforms(
            rec,
            detection_threshold=detection_threshold,
            deduplication_radius=deduplication_radius,
            n_waveforms_fit=n_waveforms_fit,
            trough_offset_samples=full_trough,
            spike_length_samples=full_length,
        )
    else:
        assert singlechan_waveforms.shape[1] == full_length

    # PCA embed
    trim_waveforms = singlechan_waveforms[:, alignment_padding:trim_end]
    u, s, v = torch.pca_lowrank(trim_waveforms, q=pca_rank + 10, center=False, niter=7)
    u = u[:, :pca_rank]
    s = s[:pca_rank]
    v = v[:, :pca_rank]
    eigs = s.sqrt() / np.sqrt(len(u) - 1)
    embeds = u * eigs

    # kmeans++ (iter=0). seems better than kmeans for this application.
    labels, _, centroids = kmeans.kmeans(
        embeds,
        n_components=n_centroids,
        return_centroids=True,
        n_iter=0,
        kmeanspp_initial=kmeanspp_initial,
        random_state=random_seed,
    )

    # get templates by averaging and realigning single chan wfs
    temps = []
    for j in range(n_centroids):
        temp = singlechan_waveforms[labels == j].mean(0)
        peak = temp.abs().argmax()
        shift = peak - full_trough
        if abs(shift) > alignment_padding:
            wning = f"Odd {shift=} observed in singlechan templates with {alignment_padding=}."
            warnings.warn(wning)
        shift = min(max(shift, -alignment_padding), alignment_padding)
        temp = temp[shift + alignment_padding : shift + trim_end]
        temps.append(torch.asarray(temp))
    temps = torch.stack(temps, dim=0)

    # taper and normalize
    temps = spiketorch.taper(temps, t_start=taper_start, t_end=taper_end)
    temps /= torch.linalg.norm(temps, dim=1, keepdim=True)

    return temps


def get_singlechan_waveforms(
    rec,
    detection_threshold=6.0,
    deduplication_radius=150.0,
    n_waveforms_fit=20_000,
    trough_offset_samples=42,
    spike_length_samples=121,
    show_progress=False,
):
    from dartsort.peel.threshold import ThresholdAndFeaturize
    from dartsort.transform import WaveformPipeline, Waveform

    channel_index = waveform_util.single_channel_index(
        rec.get_num_channels(), to_torch=True
    )
    deduplication_index = waveform_util.make_channel_index(
        rec.get_channel_locations(), radius=deduplication_radius, to_torch=True
    )
    wfeat = Waveform(
        channel_index, name="hi", spike_length_samples=spike_length_samples
    )
    thresh = ThresholdAndFeaturize(
        rec,
        channel_index,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        featurization_pipeline=WaveformPipeline([wfeat]),
        spatial_dedup_channel_index=deduplication_index,
        detection_threshold=detection_threshold,
        n_waveforms_fit=n_waveforms_fit,
    )

    with tempfile.TemporaryDirectory(prefix="dartsorttmp") as tdir:
        tmp_h5 = Path(tdir) / "singlechan_wfs.h5"
        thresh.run_subsampled_peeling(tmp_h5, show_progress=show_progress)
        with h5py.File(tmp_h5) as h5:
            waveforms = h5["hi"][:]

    assert waveforms.shape[2] == 1
    waveforms = torch.asarray(waveforms[:, :, 0])

    return waveforms


def singlechan_to_library(singlechan_templates):
    from dartsort.templates import TemplateData


def universal_templates_from_data(
    rec,
):
    singlechan_centroids = get_singlechan_centroids(
        singlechan_waveforms=None,
        rec=None,
        detection_threshold=6.0,
        deduplication_radius=150.0,
        trough_offset_samples=42,
        spike_length_samples=121,
        alignment_padding=20,
        n_centroids=10,
        pca_rank=8,
        n_waveforms_fit=20_000,
        taper=True,
        taper_start=20,
        taper_end=30,
        kmeanspp_initial="random",
        random_seed=0,
    )
