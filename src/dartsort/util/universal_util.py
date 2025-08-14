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
    dedup_temporal_radius_samples=11,
    n_kmeanspp_tries=10,
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
    from dartsort.cluster import kmeans

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
            dedup_temporal_radius_samples=dedup_temporal_radius_samples,
            n_waveforms_fit=n_waveforms_fit,
            trough_offset_samples=full_trough,
            spike_length_samples=full_length,
        )
    else:
        assert singlechan_waveforms.shape[1] == full_length

    # PCA embed
    trim_waveforms = singlechan_waveforms[:, alignment_padding:trim_end]
    q = min(pca_rank + 10, *trim_waveforms.shape)
    u, s, v = torch.pca_lowrank(trim_waveforms, q=pca_rank + 10, center=False, niter=7)
    u = u[:, :pca_rank]
    s = s[:pca_rank]
    v = v[:, :pca_rank]
    eigs = s.sqrt() / np.sqrt(len(u) - 1)
    embeds = u * eigs

    # kmeans++ (iter=0). seems better than kmeans for this application.
    kmeans_res = kmeans.kmeans(
        embeds,
        n_components=n_centroids,
        n_iter=0,
        n_kmeanspp_tries=n_kmeanspp_tries,
        kmeanspp_initial=kmeanspp_initial,
        random_state=random_seed,
    )
    labels = kmeans_res["labels"]

    # get templates by averaging and realigning single chan wfs
    temps = []
    for j in range(n_centroids):
        temp = singlechan_waveforms[labels == j].mean(0)
        peak = temp.abs().argmax().cpu().item()
        shift = peak - full_trough
        if abs(shift) > alignment_padding:
            wning = f"Odd {shift=} observed in singlechan templates with {alignment_padding=}."
            warnings.warn(wning)
        shift = min(max(shift, -alignment_padding), alignment_padding)
        temp = temp[shift + alignment_padding : shift + trim_end]
        temps.append(torch.asarray(temp))
    temps = torch.stack(temps, dim=0)

    # taper and normalize
    if taper:
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
    dedup_temporal_radius_samples=11,
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
    baz = "hi"
    wfeat = Waveform(
        channel_index, name=baz, spike_length_samples=spike_length_samples
    )
    thresh = ThresholdAndFeaturize(
        rec,
        channel_index,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        featurization_pipeline=WaveformPipeline([wfeat]),
        spatial_dedup_channel_index=deduplication_index,
        detection_threshold=detection_threshold,
        dedup_temporal_radius_samples=dedup_temporal_radius_samples,
        n_waveforms_fit=n_waveforms_fit,
    )

    with tempfile.TemporaryDirectory(prefix="dartsorttmp") as tdir:
        tmp_h5 = Path(tdir) / "singlechan_wfs.h5"
        thresh.run_subsampled_peeling(tmp_h5, show_progress=show_progress)
        with h5py.File(tmp_h5) as h5:
            waveforms = h5[baz][:]

    assert waveforms.shape[2] == 1
    waveforms = torch.asarray(waveforms[:, :, 0])

    return waveforms


def spatial_footprint_bank(
    geom, n_sigmas=3, min_template_size=10.0, dsigma=2.0, max_distance=32.0, dx=32.0, eps=0.025
):
    z_unique = np.unique(geom[:, 1])
    dz = 1.0 if z_unique.size == 1 else np.median(np.diff(z_unique))

    z_min, z_max = z_unique.min(), z_unique.max()
    z_centers = np.arange(z_min, z_max + 1e-2, dz / 2)

    x_min, x_max = geom[:, 0].min(), geom[:, 0].max()
    nx = np.round((x_max - x_min) / (dx / 2)) + 1
    x_centers = np.linspace(x_min, x_max, num=int(nx))

    # get all centers in 2d and exclude ones which are too far from the probe
    # which won't be any i guess without missing chans etc
    xx, zz = np.meshgrid(x_centers, z_centers, indexing="ij")
    xz = np.stack((xx, zz), axis=-1).reshape(-1, 2)
    distsq = np.square(xz[:, None, :] - geom).sum(2)
    valid = distsq < max_distance**2
    valid = valid.any(1)
    xz = xz[valid]
    distsq = distsq[valid]
    n_centers = len(distsq)
    assert distsq.shape == (n_centers, len(geom))

    # puff them up
    sigmas = min_template_size * (1 + dsigma * np.arange(n_sigmas))
    spatial_profiles = np.exp(-distsq[:, None, :] / sigmas[:, None] ** 2)
    spatial_profiles = spatial_profiles.reshape(n_centers * n_sigmas, len(geom))
    spatial_profiles /= np.linalg.norm(spatial_profiles, axis=1, keepdims=True)
    spatial_profiles[spatial_profiles < eps] = 0.0

    return spatial_profiles


def singlechan_to_library(
    singlechan_templates,
    geom,
    trough_offset_samples=42,
    n_sigmas=5,
    min_template_size=10.0,
    max_distance=32.0,
    dx=32.0,
):
    from dartsort.templates import TemplateData

    footprints = spatial_footprint_bank(
        geom,
        n_sigmas=n_sigmas,
        min_template_size=min_template_size,
        max_distance=max_distance,
        dx=dx,
    )
    footprints = footprints.astype(str(singlechan_templates.dtype).split(".")[1])
    if torch.is_tensor(singlechan_templates):
        singlechan_templates = singlechan_templates.numpy(force=True)
    nf, nc = footprints.shape
    nsct, nt = singlechan_templates.shape
    templates = footprints[:, None, None, :] * singlechan_templates[None, :, :, None]
    assert templates.shape == (nf, nsct, nt, nc)

    # note: this is footprint-major
    templates = templates.reshape(nf * nsct, nt, nc)
    templates /= np.linalg.norm(templates, axis=(1, 2), keepdims=True)

    template_data = TemplateData(
        templates,
        unit_ids=np.arange(nsct * nf),
        spike_counts=np.ones(nsct * nf, dtype=int),
        trough_offset_samples=trough_offset_samples,
    )
    return footprints, template_data


def universal_templates_from_data(
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
    n_sigmas=5,
    min_template_size=10.0,
    max_distance=32.0,
    dx=32.0,
):
    singlechan_centroids = get_singlechan_centroids(
        singlechan_waveforms=singlechan_waveforms,
        rec=rec,
        detection_threshold=detection_threshold,
        deduplication_radius=deduplication_radius,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        alignment_padding=alignment_padding,
        n_centroids=n_centroids,
        pca_rank=pca_rank,
        n_waveforms_fit=n_waveforms_fit,
        taper=taper,
        taper_start=taper_start,
        taper_end=taper_end,
        kmeanspp_initial=kmeanspp_initial,
        random_seed=random_seed,
    )
    footprints, template_data = singlechan_to_library(
        singlechan_centroids,
        rec.get_channel_locations() if rec is not None else None,
        trough_offset_samples=trough_offset_samples,
        n_sigmas=n_sigmas,
        min_template_size=min_template_size,
        max_distance=max_distance,
        dx=dx,
    )
    return singlechan_centroids, footprints, template_data
