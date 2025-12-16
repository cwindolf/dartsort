import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F

from dartsort.cluster.cluster_util import get_main_channel_pcs

from ..util import drift_util, waveform_util
from ..util.job_util import ensure_computation_config
from ..util.spiketorch import fast_nanmedian, ptp
from .get_templates import get_templates

# -- alternate template constructors


def get_registered_templates(
    recording,
    sorting,
    spike_times_s,
    spike_depths_um,
    geom,
    motion_est,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    realign_peaks=False,
    realign_max_sample_shift=20,
    low_rank_denoising=True,
    denoising_tsvd=None,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    reducer=fast_nanmedian,
    random_seed=0,
    n_jobs=0,
    show_progress=True,
):
    # use geometry and motion estimate to get pitch shifts and reg geom
    registered_geom = drift_util.registered_geometry(geom, motion_est=motion_est)
    pitch_shifts = drift_util.get_spike_pitch_shifts(
        spike_depths_um, geom, times_s=spike_times_s, motion_est=motion_est
    )

    # now compute templates
    results = get_templates(
        recording,
        sorting,
        registered_geom=registered_geom,
        pitch_shifts=pitch_shifts,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        spikes_per_unit=spikes_per_unit,
        realign_peaks=realign_peaks,
        realign_max_sample_shift=realign_max_sample_shift,
        low_rank_denoising=low_rank_denoising,
        denoising_tsvd=denoising_tsvd,
        denoising_rank=denoising_rank,
        denoising_fit_radius=denoising_fit_radius,
        denoising_spikes_fit=denoising_spikes_fit,
        denoising_snr_threshold=denoising_snr_threshold,
        reducer=reducer,
        random_seed=random_seed,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    results["registered_geom"] = registered_geom
    results["registered_templates"] = results["templates"]

    return results


def get_realigned_sorting(
    recording,
    sorting,
    realign_peaks=True,
    low_rank_denoising=False,
    reassign_channels=False,
    **kwargs,
):
    results = get_templates(
        recording,
        sorting,
        realign_peaks=realign_peaks,
        low_rank_denoising=low_rank_denoising,
        **kwargs,
    )
    sorting = results["sorting"]
    if reassign_channels:
        assert "templates" in results
        templates = results["templates"]
        assert isinstance(templates, np.ndarray)
        max_chans = np.abs(templates).max(1).argmax(1)
        new_channels = max_chans[sorting.labels]
        sorting = sorting.ephemeral_replace(channels=new_channels)
    return sorting


def weighted_average(unit_ids, templates, weights):
    n_out = unit_ids.max() + 1
    n_in, t, c = templates.shape
    out = np.zeros((n_out, t, c), dtype=templates.dtype)
    weights = weights.astype(float)
    for i in range(n_out):
        which_in = np.flatnonzero(unit_ids == i)
        if not which_in.size:
            continue

        w = weights[which_in][:, None, None]
        if w.sum() == 0:
            continue
        w /= w.sum()
        out[i] = (w * templates[which_in]).sum(0)

    return out


# -- template drift handling


def templates_at_time(
    t_s,
    registered_templates,
    geom,
    registered_template_depths_um=None,
    registered_geom=None,
    motion_est=None,
    return_pitch_shifts=False,
    geom_kdtree=None,
    match_distance=None,
    return_padded=False,
    fill_value=torch.nan,
):
    if registered_geom is None and not return_padded:
        return registered_templates
    if registered_geom is None and return_padded:
        assert torch.is_tensor(registered_templates)
        return F.pad(registered_templates, (0, 1), value=fill_value)
    assert motion_est is not None
    assert registered_template_depths_um is not None

    # for each unit, extract relevant channels at time t_s
    # how many pitches to shift each unit relative to registered pos at time t_s?
    unregistered_depths_um = drift_util.invert_motion_estimate(
        motion_est, t_s, registered_template_depths_um
    )
    # reverse arguments to pitch shifts since we are going the other direction
    pitch_shifts = drift_util.get_spike_pitch_shifts(
        depths_um=registered_template_depths_um,
        geom=geom,
        registered_depths_um=unregistered_depths_um,
    )
    # extract relevant channel neighborhoods, also by reversing args to a drift helper
    unregistered_templates = drift_util.get_waveforms_on_static_channels(
        registered_templates,
        registered_geom,
        n_pitches_shift=pitch_shifts,
        registered_geom=geom,
        target_kdtree=geom_kdtree,
        match_distance=match_distance,
        fill_value=fill_value,
        return_padded=return_padded,
    )
    if return_pitch_shifts:
        return pitch_shifts, unregistered_templates
    return unregistered_templates


def spatially_mask_templates(template_data, radius_um=0.0):
    if not radius_um:
        return template_data

    tt = template_data.templates.copy()
    ci = waveform_util.make_channel_index(template_data.registered_geom, radius_um)
    chans = np.arange(ci.shape[0])
    for j, t in enumerate(tt):
        mask = ~np.isin(chans, ci[np.ptp(t, 0).argmax()])
        tt[j, :, mask] = 0.0

    return replace(template_data, templates=tt)


# -- template numerical processing


@dataclass
class LowRankTemplates:
    unit_ids: np.ndarray
    temporal_components: np.ndarray
    singular_values: np.ndarray
    spatial_components: np.ndarray
    spike_counts_by_channel: np.ndarray | None

    def shift_to_best_channels(self, geom, registered_geom=None) -> "LowRankTemplates":
        if registered_geom is None or registered_geom.shape == geom.shape:
            return self

        pitch = waveform_util.get_pitch(geom)
        rgkdt = KDTree(registered_geom)
        low = registered_geom[:, 1].min() - geom[:, 1].min()
        high = registered_geom[:, 1].max() - geom[:, 1].max()
        assert low <= 0 <= high
        nlow = int(-low // pitch)
        nhigh = int(high // pitch)

        spatial_components = self.spatial_components
        if torch.is_tensor(spatial_components):
            spatial_components = spatial_components.numpy(force=True)
        scores = np.full(len(self.spatial_components), -np.inf)
        best_spatial = spatial_components[:, :, : len(geom)] + 0.0

        for j in range(-nlow, nhigh + 1):
            gshift = geom + [0, j * pitch]
            d, regchans = rgkdt.query(gshift)
            assert np.all(regchans < rgkdt.n)

            spatial = spatial_components[:, :, regchans]
            new_scores = np.square(spatial).sum((1, 2))
            better = np.flatnonzero(new_scores > scores)
            best_spatial[better] = spatial[better]
            scores[better] = new_scores[better]

        return replace(self, spatial_components=best_spatial)


def svd_compress_templates(
    template_data,
    min_channel_amplitude=0.0,
    rank=5,
    channel_sparse=True,
    allow_na=False,
    computation_cfg=None,
    batch_size=32,
):
    """
    Returns:
    temporal_components: n_units, spike_length_samples, rank
    singular_values: n_units, rank
    spatial_components: n_units, rank, n_channels
    """
    computation_cfg = ensure_computation_config(computation_cfg)
    dev = computation_cfg.actual_device()
    if hasattr(template_data, "templates"):
        unit_ids = template_data.unit_ids
        templates = template_data.templates
        counts = template_data.spike_counts_by_channel
    else:
        templates = template_data
        counts = None
        unit_ids = np.arange(len(templates))

    rank = min(rank, *templates.shape[1:])

    amp_vecs = ptp(templates, dim=1, keepdims=True)
    isna = np.isnan(amp_vecs)
    if not allow_na:
        assert not isna.any()
    else:
        amp_vecs = np.nan_to_num(amp_vecs, copy=False)
        templates = np.nan_to_num(templates)
    vis_mask = amp_vecs > min_channel_amplitude
    dtype = templates.dtype

    if not channel_sparse:
        vis_templates = torch.as_tensor(templates * vis_mask)
        U, S, Vh = _svd_helper(vis_templates)
        # s is descending.
        temporal_components = U[:, :, :rank].to(dtype).numpy(force=True)
        singular_values = S[:, :rank].to(dtype).numpy(force=True)
        spatial_components = Vh[:, :rank, :].to(dtype).numpy(force=True)
    else:
        # channel sparse: only SVD the nonzero channels
        # this encodes the same exact subspace as above, and the reconstruction
        # error is the same as above as a function of rank. it's just that
        # we can zero out some spatial components, which is a useful property
        # (used in pairwise convolutions for instance)

        vis_mask = torch.as_tensor(vis_mask)
        n, t, c = templates.shape
        assert amp_vecs.shape == (n, 1, c)
        templates = torch.as_tensor(templates)
        uvec, groups = vis_mask[:, 0].unique(dim=0, return_inverse=True)
        assert uvec.shape[1] == c

        temporal_components = np.zeros((n, t, rank), dtype=dtype)
        singular_values = np.zeros((n, rank), dtype=dtype)
        spatial_components = np.zeros((n, rank, c), dtype=dtype)

        for j in range(len(uvec)):
            (in_j,) = (groups == j).nonzero(as_tuple=True)
            nj = in_j.numel()
            (mask,) = uvec[j].nonzero(as_tuple=True)
            rankj = min(rank, mask.numel())
            if not rankj:
                continue

            for bs in range(0, nj, batch_size):
                be = min(bs + batch_size, nj)
                binds = in_j[bs:be]

                batch_x = templates[binds[:, None], :, mask[None, :]].to(dev)
                # fancy ix comes to the front
                assert batch_x.shape == (be - bs, mask.numel(), t)
                U, S, Vh = _svd_helper(batch_x)
                spatial_components[binds[:, None], :rankj, mask[None, :]] = U[
                    :, :, :rankj
                ].numpy(force=True)
                singular_values[binds, :rankj] = S[:, :rankj].numpy(force=True)
                temporal_components[binds, :, :rankj] = Vh[:, :rankj, :].mT.numpy(
                    force=True
                )

    if allow_na:
        isna = np.broadcast_to(isna, spatial_components.shape)
        spatial_components[isna] = np.nan

    return LowRankTemplates(
        unit_ids=unit_ids,
        temporal_components=temporal_components,
        singular_values=singular_values,
        spatial_components=spatial_components,
        spike_counts_by_channel=counts,
    )


@dataclass
class SharedBasisTemplates:
    unit_ids: np.ndarray
    # rank, time
    temporal_components: np.ndarray
    # n, rank, chans
    spatial_singular: np.ndarray
    spike_counts_by_channel: np.ndarray | None


def shared_basis_compress_templates(
    template_data, min_channel_amplitude=1.0, rank=5, computation_cfg=None
):
    computation_cfg = ensure_computation_config(computation_cfg)
    dev = computation_cfg.actual_device()
    if hasattr(template_data, "templates"):
        unit_ids = template_data.unit_ids
        templates = template_data.templates
        counts = template_data.spike_counts_by_channel
    else:
        templates = template_data
        counts = None
        unit_ids = np.arange(len(templates))
    n, t, c = templates.shape
    rank = min(rank, *templates.shape[1:])
    amp_vecs = ptp(templates, dim=1, keepdims=True)
    assert np.isfinite(amp_vecs).all()
    visible = amp_vecs > min_channel_amplitude
    uu, cc = np.nonzero(visible)
    nvis = uu.shape[0]

    # put time on the first axis
    if nvis == visible.size:
        to_compress = templates.transpose(1, 0, 2)
        to_compress = to_compress.reshape(t, n * c)
    else:
        to_compress = templates[uu, :, cc]
        assert to_compress.shape == (nvis, t)
        to_compress = to_compress.T

    # get the temporal basis
    to_compress = torch.asarray(to_compress, device=dev)
    U, S, Vh = _svd_helper(to_compress)
    del S, Vh
    assert U.shape == (t, nvis)
    temporal_comps = U[:, :rank]
    # to rank-major
    temporal_comps = temporal_comps.T.contiguous()
    temporal_comps = temporal_comps.numpy(force=True)

    # project templates onto temporal comps (no sparsity here.)
    spatial_sing = np.einsum("ntc,rt->nrc", templates, temporal_comps)

    return SharedBasisTemplates(
        unit_ids=unit_ids,
        temporal_components=temporal_comps,
        spatial_singular=spatial_sing,
        spike_counts_by_channel=counts,
    )


def temporally_upsample_templates(
    templates, temporal_upsampling_factor=8, kind="cubic"
):
    """Note, also works on temporal components thanks to compatible shape."""
    n, t, c = templates.shape
    tp = np.arange(t).astype(float)
    erp = interp1d(tp, templates, axis=1, bounds_error=True, kind=kind)
    tup = np.arange(
        t, step=1.0 / temporal_upsampling_factor
    )  # pyright: ignore[reportCallIssue]
    tup.clip(0, t - 1, out=tup)
    upsampled_templates = erp(tup)
    upsampled_templates = upsampled_templates.reshape(
        n, t, temporal_upsampling_factor, c
    )
    upsampled_templates = upsampled_templates.astype(templates.dtype)
    return upsampled_templates


@dataclass
class CompressedUpsampledTemplates:
    n_compressed_upsampled_templates: int
    compressed_upsampled_templates: np.ndarray
    compressed_upsampling_map: np.ndarray
    compressed_upsampling_index: np.ndarray
    compressed_index_to_template_index: np.ndarray
    compressed_index_to_upsampling_index: np.ndarray
    trough_shifts: np.ndarray


def default_n_upsamples_map(ptps, max_upsample=8):
    # avoid overflow in 4 ** by trimming ptp range in advance
    max_ptp = 1 + 2 * math.log(max_upsample, 4)
    ptps = np.minimum(ptps, max_ptp)
    upsamples = 4 ** (ptps // 2)
    upsamples = upsamples.astype(int)
    np.clip(upsamples, 1, max_upsample, out=upsamples)
    return upsamples


def compressed_upsampled_templates(
    templates,
    *,
    trough_offset_samples,
    ptps=None,
    max_upsample=8,
    n_upsamples_map=default_n_upsamples_map,
    kind="cubic",
):
    """compressedly store fewer temporally upsampled copies of lower amplitude templates

    Returns
    -------
    A CompressedUpsampledTemplates object with fields:
        compressed_upsampled_templates : array (n_compressed_upsampled_templates, spike_length_samples, n_channels)
        compressed_upsampling_map : array (n_templates, max_upsample)
            compressed_upsampled_templates[compressed_upsampling_map[unit, j]] is an approximation
            of the jth upsampled template for this unit. for low-amplitude units,
            compressed_upsampling_map[unit] will have fewer unique entries, corresponding
            to fewer saved upsampled copies for that unit.
        compressed_upsampling_index : array (n_templates, max_upsample)
            A n_compressed_upsampled_templates-padded ragged array mapping each
            template index to its compressed upsampled indices
        compressed_index_to_template_index
        compressed_index_to_upsampling_index
    """
    n_templates = templates.shape[0]
    assert templates.ndim == 3
    if max_upsample == 1:
        return CompressedUpsampledTemplates(
            n_templates,
            templates,
            np.arange(n_templates)[:, None],
            np.arange(n_templates)[:, None],
            np.arange(n_templates),
            np.zeros(n_templates, dtype=np.int64),
            np.zeros(n_templates, dtype=np.int32),
        )

    # how many copies should each unit get?
    # sometimes users may pass temporal SVD components in instead of templates,
    # so we allow them to pass in the amplitudes of the actual templates
    if ptps is None:
        ptps = np.ptp(templates, 1).max(1)
    assert ptps.shape == (n_templates,)
    if n_upsamples_map is None:
        n_upsamples = np.full(n_templates, max_upsample)
    else:
        n_upsamples = n_upsamples_map(ptps, max_upsample=max_upsample)

    # build the compressed upsampling map
    compressed_upsampling_map = np.full((n_templates, max_upsample), -1, dtype=np.int64)
    compressed_upsampling_index = np.full(
        (n_templates, max_upsample), -1, dtype=np.int64
    )
    template_indices = []
    upsampling_indices = []
    current_compressed_index = 0
    for i, nup in enumerate(n_upsamples):
        compression = max_upsample // nup
        nup = max_upsample // compression  # handle divisibility failure

        # new compressed indices
        compressed_upsampling_map[i] = current_compressed_index + np.arange(nup).repeat(
            compression
        )
        compressed_upsampling_index[i, :nup] = current_compressed_index + np.arange(nup)
        current_compressed_index += nup

        # indices of the templates to keep in the full array of upsampled templates
        template_indices.extend([i] * nup)
        upsampling_indices.extend(compression * np.arange(nup))
    assert (compressed_upsampling_map >= 0).all()
    assert (
        np.unique(compressed_upsampling_map).size
        == (compressed_upsampling_index >= 0).sum()
        == compressed_upsampling_map.max() + 1
        == compressed_upsampling_index.max() + 1
        == current_compressed_index
    )
    template_indices = np.array(template_indices)
    upsampling_indices = np.array(upsampling_indices)
    compressed_upsampling_index[
        compressed_upsampling_index < 0
    ] = current_compressed_index

    # get the upsampled templates
    all_upsampled_templates = temporally_upsample_templates(
        templates, temporal_upsampling_factor=max_upsample, kind=kind
    )
    # n, up, t, c
    all_upsampled_templates = all_upsampled_templates.transpose(0, 2, 1, 3)
    rix = np.ravel_multi_index(
        (template_indices, upsampling_indices), all_upsampled_templates.shape[:2]
    )
    all_upsampled_templates = all_upsampled_templates.reshape(
        n_templates * max_upsample, templates.shape[1], templates.shape[2]
    )
    compressed_upsampled_templates = all_upsampled_templates[rix]
    _, _, trough_shifts = get_main_channels_and_alignments(templates=compressed_upsampled_templates)
    trough_shifts = trough_shifts - trough_offset_samples

    return CompressedUpsampledTemplates(
        current_compressed_index,
        compressed_upsampled_templates,
        compressed_upsampling_map,
        compressed_upsampling_index,
        template_indices,
        upsampling_indices,
        trough_shifts,
    )


def _svd_helper(x):
    """This matches numpy's behavior in tests/test_matching.py."""
    if x.device.type == "cuda":
        # torch's cuda results need a certain driver to agree with numpy
        try:
            return torch.linalg.svd(x, full_matrices=False, driver="gesvda")
        except torch.linalg.LinAlgError:  # type: ignore[reportPrivateImportUsage]
            return torch.linalg.svd(x, full_matrices=False, driver="gesvd")
    else:
        # torch's CPU results are disagreeing with numpy here.
        # return torch.linalg.svd(x, full_matrices=False)
        U, S, Vh = np.linalg.svd(x.numpy(), full_matrices=False)
        U = torch.from_numpy(U)
        S = torch.from_numpy(S)
        Vh = torch.from_numpy(Vh)
        return U, S, Vh


def get_main_channels_and_alignments(
    template_data=None, trough_factor=3.0, templates=None
):
    if templates is None:
        assert template_data is not None
        templates = template_data.templates
    main_chans = np.ptp(templates, axis=1).argmax(1)
    mc_traces = np.take_along_axis(templates, main_chans[:, None, None], axis=2)[
        :, :, 0
    ]
    aligner_traces = np.where(mc_traces < 0, trough_factor * mc_traces, -mc_traces)
    offsets = np.abs(aligner_traces).argmax(1)
    return main_chans, mc_traces, offsets


def estimate_offset(
    templates,
    strategy="mainchan_trough_factor",
    trough_factor=3.0,
    min_weight=0.75,
):
    if strategy == "main_chan_trough_factor":
        _, _, offsets = get_main_channels_and_alignments(
            None, trough_factor=trough_factor, templates=templates
        )
        return offsets

    if strategy == "normsq_weighted_trough_factor":
        tmp = np.square(templates)
        weights = tmp.sum(axis=1)
        weights /= weights.max(axis=1, keepdims=True)
        weights[weights < min_weight] = 0.0
        weights /= weights.sum(axis=1, keepdims=True)
        print(f"{weights[[25, 19, 32, 29]][:, 20:30]=}")
        tmp[:] = templates
        tmp[tmp < 0] *= trough_factor
        np.abs(tmp, out=tmp)
        offsets = tmp.argmax(axis=1).astype(np.float64)
        print(
            f"{offsets[[25, 19, 32, 29]][:, 20:30]*(weights[[25, 19, 32, 29]][:, 20:30]>0)=}"
        )
        offsets = np.sum(offsets * weights, axis=1)
        offsets = np.rint(offsets).astype(np.int64)
        return offsets

    if strategy == "ampsq_weighted_trough_factor":
        weights = np.square(np.ptp(templates, axis=1))
        weights /= weights.max(axis=1, keepdims=True)
        weights[weights < min_weight] = 0.0
        weights /= weights.sum(axis=1, keepdims=True)
        print(f"{weights[[25, 19, 32, 29]][:, 20:30]=}")
        tmp = templates.copy()
        tmp[tmp < 0] *= trough_factor
        np.abs(tmp, out=tmp)
        offsets = tmp.argmax(axis=1).astype(np.float64)
        print(
            f"{offsets[[25, 19, 32, 29]][:, 20:30]*(weights[[25, 19, 32, 29]][:, 20:30]>0)=}"
        )
        offsets = np.sum(offsets * weights, axis=1)
        offsets = np.rint(offsets).astype(np.int64)
        return offsets
