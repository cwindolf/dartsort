from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch
from dartsort.transform.temporal_pca import TemporalPCAFeaturizer
from dartsort.util import drift_util, interpolation_util, waveform_util
from dartsort.util.data_util import DARTsortSorting, get_tpca

# -- main class


class StableSpikeDataset(torch.nn.Module):

    def __init__(
        self,
        kept_indices: np.array,
        prgeom: torch.Tensor,
        tpca: TemporalPCAFeaturizer,
        extract_channels: torch.LongTensor,
        core_channels: torch.LongTensor,
        original_sorting: DARTsortSorting,
        core_features: torch.Tensor,
        extract_features: torch.Tensor,
        core_neighborhoods: "SpikeNeighborhoods",
        extract_neighborhoods: "SpikeNeighborhoods",
        features_on_device: bool = False,
        interpolation_method: str = "kriging",
        interpolation_sigma: float = 20.0,
        core_radius: float = 35.0,
    ):
        """Motion-corrected spike data on the registered probe"""
        super().__init__()

        self.features_on_device = features_on_device

        # data shapes
        self.rank = tpca.rank
        self.n_channels = prgeom.shape[0] - 1
        self.n_channels_extract = extract_channels.shape[1]
        self.n_channels_core = core_channels.shape[1]
        self.n_spikes = kept_indices.size
        self.interpolation_method = interpolation_method
        self.interpolation_sigma = interpolation_sigma
        self.core_radius = core_radius

        self.kept_indices = kept_indices
        self.original_sorting = original_sorting

        # pca module, to reconstructing wfs for vis
        self.tpca = tpca

        # neighborhoods module, for querying spikes by channel group
        self.core_neighborhoods = core_neighborhoods
        self.extract_neighborhoods = extract_neighborhoods

        extract_amp_vecs = torch.linalg.vector_norm(extract_features, dim=1)
        amps = extract_amp_vecs.nan_to_num().max(1).values

        # channel neighborhoods and features
        # if not self.features_on_device, .spike_data() will .to(self.device)
        times_s = torch.asarray(self.original_sorting.times_seconds[kept_indices])
        if self.features_on_device:
            self.register_buffer("core_channels", core_channels)
            self.register_buffer("extract_channels", extract_channels)
            self.register_buffer("core_features", core_features)
            self.register_buffer("extract_features", extract_features)
            # self.register_buffer("extract_amp_vecs", extract_amp_vecs)
            self.register_buffer("amps", amps)
            self.register_buffer("times_seconds", times_s)
        else:
            self.core_channels = core_channels
            self.extract_channels = extract_channels
            self.core_features = core_features
            self.extract_features = extract_features
            # self.extract_amp_vecs = extract_amp_vecs
            self.amps = amps
            self.times_seconds = times_s

        # always on device
        self.register_buffer("prgeom", prgeom)

    @property
    def device(self):
        return self.prgeom.device

    @property
    def dtype(self):
        return self.core_features.dtype

    @classmethod
    def from_sorting(
        cls,
        sorting,
        motion_est,
        core_radius=35.0,
        subsampling_rg=0,
        max_n_spikes=np.inf,
        interpolation_sigma=20.0,
        interpolation_method="kriging",
        motion_depth_mode="channel",
        features_dataset_name="collisioncleaned_tpca_features",
        show_progress=False,
        store_on_device=False,
        workers=-1,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # which spikes to keep?
        if len(sorting) > max_n_spikes:
            rg = np.random.default_rng(subsampling_rg)
            kept_indices = rg.choice(len(sorting), size=max_n_spikes, replace=False)
            kept_indices.sort()
            keep = np.zeros(len(sorting), dtype=bool)
            keep[kept_indices] = 1
            keep_select = kept_indices
        else:
            keep = np.ones(len(sorting), dtype=bool)
            kept_indices = np.arange(len(sorting))
            keep_select = slice(None)

        with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
            geom = h5["geom"][:]
            extract_channel_index = h5["channel_index"][:]
            registered_geom = drift_util.registered_geometry(
                geom, motion_est=motion_est
            )
            prgeom = interpolation_util.pad_geom(registered_geom)

            times_s, channels, depths, rdepths, shifts = get_shift_info(
                sorting, motion_est, geom, keep_select, motion_depth_mode
            )

            # determine channel occupancy of core/extract stable features
            extract_channels, core_channels = get_stable_channels(
                geom,
                depths,
                rdepths,
                times_s,
                channels,
                keep_select,
                extract_channel_index,
                registered_geom,
                motion_est,
                core_radius,
                workers=workers,
                motion_depth_mode=motion_depth_mode,
            )
            extract_channels = torch.from_numpy(extract_channels)
            core_channels = torch.from_numpy(core_channels)
            if store_on_device:
                extract_channels = extract_channels.to(device)
                core_channels = core_channels.to(device)

            # stabilize features with interpolation
            extract_features = interpolation_util.interpolate_by_chunk(
                keep,
                h5[features_dataset_name],
                geom,
                extract_channel_index,
                sorting.channels[keep_select],
                shifts,
                registered_geom,
                extract_channels,
                sigma=interpolation_sigma,
                interpolation_method=interpolation_method,
                device=device,
                store_on_device=store_on_device,
                show_progress=show_progress,
            )
            core_features = interpolation_util.interpolate_by_chunk(
                keep,
                h5[features_dataset_name],
                geom,
                extract_channel_index,
                sorting.channels[keep_select],
                shifts,
                registered_geom,
                core_channels,
                sigma=interpolation_sigma,
                interpolation_method=interpolation_method,
                device=device,
                store_on_device=store_on_device,
                show_progress=show_progress,
            )

        # load temporal PCA
        tpca = get_tpca(sorting)

        # determine channel neighborhoods
        core_neighborhoods = SpikeNeighborhoods.from_channels(
            core_channels, len(prgeom) - 1
        )
        extract_neighborhoods = SpikeNeighborhoods.from_channels(
            extract_channels, len(prgeom) - 1
        )

        self = cls(
            kept_indices=kept_indices,
            prgeom=prgeom,
            tpca=tpca,
            extract_channels=extract_channels,
            core_channels=core_channels,
            original_sorting=sorting,
            core_features=core_features,
            extract_features=extract_features,
            extract_neighborhoods=extract_neighborhoods,
            core_neighborhoods=core_neighborhoods,
            features_on_device=store_on_device,
            interpolation_method=interpolation_method,
            interpolation_sigma=interpolation_sigma,
            core_radius=core_radius,
        )
        self.to(device)
        return self

    def spike_data(
        self,
        indices: torch.LongTensor,
        neighborhood: str = "extract",
        with_channels=True,
        with_reconstructions: bool = False,
        with_neighborhood_ids: bool = False,
    ) -> "SpikeFeatures":
        channels = neighborhood_ids = None
        if neighborhood == "extract":
            features = self.extract_features[indices]
            if with_channels:
                channels = self.extract_channels[indices]
            if with_neighborhood_ids:
                neighborhood_ids = self.extract_neighborhoods.neighborhood_ids[indices]
        elif neighborhood == "core":
            features = self.core_features[indices]
            if with_channels:
                channels = self.core_channels[indices]
            if with_neighborhood_ids:
                neighborhood_ids = self.core_neighborhoods.neighborhood_ids[indices]
        else:
            assert False

        if not self.features_on_device:
            features = features.to(self.device)
            if with_channels:
                channels = channels.to(self.device)

        waveforms = None
        if with_reconstructions:
            n = len(features)
            c = channels.shape[1]
            waveforms = features.permute(0, 2, 1).reshape(n * c, -1)
            waveforms = self.tpca._inverse_transform_in_probe(waveforms)
            waveforms = waveforms.reshape(n, c, -1).permute(0, 2, 1)

        return SpikeFeatures(
            indices=indices,
            features=features,
            channels=channels,
            waveforms=waveforms,
            neighborhood_ids=neighborhood_ids,
        )

    def interp_to_chans(self, spike_data, channels):
        return interp_to_chans(
            spike_data,
            channels,
            self.prgeom,
            interpolation_method=self.interpolation_method,
            interpolation_sigma=self.interpolation_sigma,
        )


# -- utility classes


@dataclass(frozen=True, slots=True, kw_only=True)
class SpikeFeatures:
    """A 'messenger class' to hold onto batches of spike features

    Return type of StableSpikeDataset's spike_data method.
    """

    # these are relative to the StableFeatures instance's subsampling (keepers)
    indices: torch.LongTensor
    # n, r, c
    features: torch.Tensor
    # n, c
    channels: Optional[torch.LongTensor] = None
    # n, t, c
    waveforms: Optional[torch.Tensor] = None
    # n
    neighborhood_ids: Optional[torch.LongTensor] = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, ix):
        """Subset the spikes in this collection with [subset]."""
        waveforms = channels = neighborhood_ids = None
        if self.channels is not None:
            channels = self.channels[ix]
        if self.waveforms is not None:
            waveforms = self.waveforms[ix]
        if self.neighborhood_ids is not None:
            neighborhood_ids = self.neighborhood_ids[ix]
        return self.__class__(
            indices=self.indices[ix],
            features=self.features[ix],
            channels=channels,
            waveforms=waveforms,
            neighborhood_ids=neighborhood_ids,
        )

    def __repr__(self):
        indstr = f"indices.shape={self.indices.shape},"
        featstr = f"features.shape={self.features.shape},"
        chanstr = wfstr = ""
        if self.channels is not None:
            chanstr = f"channels.shape={self.channels.shape},"
        if self.waveforms is not None:
            wfstr = f"waveforms.shape={self.waveforms.shape},"
        if self.neighborhood_ids is not None:
            idstr = f"neighborhood_ids.shape={self.neighborhood_ids.shape},"
        pstr = f"{indstr}{featstr}{chanstr}{wfstr}{idstr}"
        return f"{self.__class__.__name__}({pstr.rstrip(',')})"


class SpikeNeighborhoods(torch.nn.Module):
    def __init__(
        self,
        n_channels,
        neighborhood_ids,
        neighborhoods,
        neighborhood_members=None,
        store_on_device: bool = False,
    ):
        """SpikeNeighborhoods

        Sparsely keep track of which channels each spike lives on. Used to query
        which core sets are overlapped completely by unit channel neighborhoods.

        Arguments
        ---------
        neighborhood_ids : torch.LongTensor
            Size (n_spikes,), the neighborhood id for each spike
        neighborhoods : list[torch.LongTensor]
            The channels in each neighborhood
        neighborhood_members : list[torch.LongTensor]
            The indices of spikes in each neighborhood
        """
        super().__init__()
        self.n_channels = n_channels
        if store_on_device:
            self.register_buffer("neighborhood_ids", neighborhood_ids)
            self.register_buffer("neighborhoods", neighborhoods)
        else:
            self.neighborhood_ids = neighborhood_ids
            self.neighborhoods = neighborhoods
        self.n_neighborhoods = len(neighborhoods)

        # store neighborhoods as a matrix
        indicators = torch.zeros((n_channels, len(neighborhoods)))
        for j, nhood in enumerate(neighborhoods):
            indicators[nhood[nhood < n_channels], j] = 1.0
        self.register_buffer("indicators", indicators)
        self.register_buffer("channel_counts", indicators.sum(0))

        if neighborhood_members is None:
            # cache lookups
            neighborhood_members = []
            for j in range(len(neighborhoods)):
                (in_nhood,) = torch.nonzero(neighborhood_ids == j, as_tuple=True)
                neighborhood_members.append(in_nhood.cpu())
        assert len(neighborhood_members) == self.n_neighborhoods

        # it's a pain to store dicts with register_buffer, so store offsets
        _neighborhood_members = torch.empty(
            sum(v.numel() for v in neighborhood_members), dtype=int
        )
        self.neighborhood_members_slices = []
        neighborhood_member_offset = 0
        neighborhood_popcounts = []
        for j in range(len(neighborhoods)):
            nhoodmemsz = neighborhood_members[j].numel()
            nhoodmemsl = slice(
                neighborhood_member_offset, neighborhood_member_offset + nhoodmemsz
            )
            _neighborhood_members[nhoodmemsl] = neighborhood_members[j]
            self.neighborhood_members_slices.append(nhoodmemsl)
            neighborhood_member_offset += nhoodmemsz
            neighborhood_popcounts.append(nhoodmemsz)
        # self.register_buffer("_neighborhood_members", _neighborhood_members)
        # seems that indices want to live on cpu.
        self._neighborhood_members = _neighborhood_members.cpu()
        self.register_buffer("popcounts", torch.tensor(neighborhood_popcounts))

    @classmethod
    def from_channels(cls, channels, n_channels):
        neighborhoods, neighborhood_ids = torch.unique(
            channels, dim=0, return_inverse=True
        )
        return cls(
            n_channels=n_channels,
            neighborhoods=neighborhoods,
            neighborhood_ids=neighborhood_ids,
        )

    def neighborhood_members(self, id):
        return self._neighborhood_members[self.neighborhood_members_slices[id]]

    def subset_neighborhoods(self, channels, min_coverage=1.0, add_to_overlaps=None):
        """Return info on neighborhoods which cover the channel set well enough

        Define coverage for a neighborhood and a channel group as the intersection
        size divided by the neighborhood's size.

        Returns
        -------
        neighborhood_info : dict
            Keys are neighborhood ids, values are tuples
            (neighborhood_channels, neighborhood_member_indices)
        n_spikes : int
            The total number of spikes in the neighborhood.
        """
        inds = self.indicators[channels]
        coverage = inds.sum(0) / self.channel_counts
        (covered_ids,) = torch.nonzero(coverage >= min_coverage, as_tuple=True)
        neighborhood_info = {
            j: (self.neighborhoods[j], self.neighborhood_members(j))
            for j in covered_ids
        }
        n_spikes = self.popcounts[covered_ids].sum()
        if add_to_overlaps is not None:
            add_to_overlaps[covered_ids] += 1
        return neighborhood_info, n_spikes

    def spike_neighborhoods(self, channels, spike_indices, min_coverage=1.0):
        """Like subset_neighborhoods, but for an already chosen collection of spikes

        This is used when subsetting log likelihood calculations.
        In this case, the returned neighborhood_member_indices keys are relative:
        spike_indices[neighborhood_member_indices] are the actual indices.
        """
        spike_ids = self.neighborhood_ids[spike_indices]
        neighborhoods_considered = torch.unique(spike_ids).to(self.indicators.device)
        inds = self.indicators[channels][:, neighborhoods_considered]
        coverage = inds.sum(0) / self.channel_counts[neighborhoods_considered]
        covered_ids = neighborhoods_considered[coverage >= min_coverage].cpu()
        spike_ids = spike_ids.cpu()
        neighborhood_info = {
            j: (self.neighborhoods[j], *torch.nonzero(spike_ids == j, as_tuple=True))
            for j in covered_ids
        }
        n_spikes = self.popcounts[covered_ids].sum()
        return neighborhood_info, n_spikes


# -- helpers


def occupied_chans(spike_data, n_channels, neighborhoods=None):
    if spike_data.neighborhood_ids is None:
        chans = torch.unique(spike_data.channels)
        return chans[chans < n_channels]
    ids = torch.unique(spike_data.neighborhood_ids)
    chans = neighborhoods.neighborhoods[ids]
    chans = torch.unique(chans)
    return chans[chans < n_channels].to(spike_data.channels)


def interp_to_chans(
    spike_data,
    channels,
    prgeom,
    interpolation_method="kriging",
    interpolation_sigma=20.0,
):
    source_pos = prgeom[spike_data.channels]
    target_pos = prgeom[channels]
    shape = len(spike_data), *target_pos.shape
    target_pos = target_pos[None].broadcast_to(shape)
    return interpolation_util.kernel_interpolate(
        spike_data.features,
        source_pos,
        target_pos,
        sigma=interpolation_sigma,
        allow_destroy=False,
        interpolation_method=interpolation_method,
    )


def pad_to_chans(
    spike_data, channels, n_channels, weights=None, target_padded=None, pad_value=0.0
):
    n, r, c = spike_data.features.shape
    c_targ = channels.numel()

    # determine channels to write to
    reindexer = get_channel_reindexer(channels, n_channels)
    target_ixs = reindexer[spike_data.channels]

    # scatter data
    if target_padded is None:
        if pad_value == 0.0:
            target_padded = spike_data.features.new_zeros(n, r, c_targ + 1)
        else:
            target_padded = spike_data.features.new_full((n, r, c_targ + 1), pad_value)
    scatter_ixs = target_ixs.unsqueeze(1).broadcast_to((n, r, c))
    target_padded.scatter_(src=spike_data.features, dim=2, index=scatter_ixs)
    target = target_padded[..., :-1]
    if weights is None:
        return target, None

    # same for weights, if supplied
    assert weights.shape == (n,)
    weights_padded = weights.new_zeros((n, c_targ + 1))
    weights = weights.unsqueeze(1).broadcast_to((n, c))
    weights_padded.scatter_(src=weights, dim=1, index=target_ixs)
    weights = weights_padded[:, :-1]
    return target, weights


def zero_pad_to_chans(
    spike_data, channels, n_channels, weights=None, target_padded=None
):
    return pad_to_chans(spike_data, channels, n_channels, weights=weights)


def get_channel_reindexer(channels, n_channels):
    """
    Arguments
    ---------
    channels : LongTensor
        Shape (n_chans_subset,)
    n_channels : int

    Returns
    -------
    reindexer : LongTensor
        Shape (n_channels + 1,)
        reindexer[i] is the index of i in channels, if present.
        Otherwise, it is n_chans_subset.
        And the last entry is, of course, n_chans_subset.
    """
    reindexer = channels.new_full((n_channels + 1,), channels.numel())
    (rel_ixs,) = torch.nonzero(channels < n_channels, as_tuple=True)
    reindexer[channels[rel_ixs]] = rel_ixs
    return reindexer


def get_shift_info(sorting, motion_est, geom, keep_select, motion_depth_mode):
    """
    shifts = reg_depths - depths
    reg_depths = depths + shifts
    i.e., target_pos = source_pos + shifts
          target_pos - shifts = source_pos
    where by source pos i mean the true moving positions.
    """
    times_s = sorting.times_seconds[keep_select]
    channels = sorting.channels[keep_select]
    if motion_depth_mode == "localization":
        depths = sorting.point_source_localizations[keep_select, 2]
    elif motion_depth_mode == "channel":
        depths = geom[:, 1][channels]
    else:
        assert False

    rdepths = motion_est.correct_s(times_s, depths)
    shifts = rdepths - depths

    return times_s, channels, depths, rdepths, shifts


def get_stable_channels(
    geom,
    depths,
    rdepths,
    times_s,
    channels,
    keep_select,
    channel_index,
    registered_geom,
    motion_est,
    core_radius,
    workers=-1,
    motion_depth_mode="channel",
):
    core_channel_index = waveform_util.make_channel_index(geom, core_radius)

    pitch = drift_util.get_pitch(geom)
    registered_kdtree = drift_util.KDTree(registered_geom)
    match_distance = drift_util.pdist(geom).min() / 2

    n_pitches_shift = drift_util.get_spike_pitch_shifts(
        depths,
        geom=geom,
        motion_est=motion_est,
        times_s=times_s,
        registered_depths_um=rdepths,
        mode="round",
    )

    extract_channels = drift_util.static_channel_neighborhoods(
        geom,
        channels,
        channel_index,
        pitch=pitch,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        workers=workers,
    )
    core_channels = drift_util.static_channel_neighborhoods(
        geom,
        channels,
        core_channel_index,
        pitch=pitch,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        workers=workers,
    )

    return extract_channels, core_channels
