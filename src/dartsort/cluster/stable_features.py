from dataclasses import dataclass, replace
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
        features_on_device: bool = True,
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

        # for to_sorting()
        self.kept_indices = kept_indices
        self.original_sorting = original_sorting

        # pca module, to reconstructing wfs for vis
        self.tpca = tpca

        # neighborhoods module, for querying spikes by channel group
        self.core_neighborhoods = core_neighborhoods

        # channel neighborhoods and features
        # if not self.features_on_device, .spike_data() will .to(self.device)
        if self.features_on_device:
            self.register_buffer("core_channels", core_channels)
            self.register_buffer("extract_channels", extract_channels)
            self.register_buffer("core_features", core_features)
            self.register_buffer("extract_features", extract_features)
        else:
            self.core_channels = core_channels
            self.extract_channels = extract_channels
            self.core_features = core_features
            self.extract_features = extract_features

        # always on device
        self.register_buffer("prgeom", prgeom)

    @property
    def device(self):
        return self.prgeom.device

    def to_sorting(self) -> DARTsortSorting:
        labels = np.full_like(self.original_sorting.labels, -1)
        labels[self.kept_indices] = self.labels.cpu()
        return replace(self.original_sorting, labels=labels)

    @classmethod
    def from_sorting(
        cls,
        sorting,
        motion_est,
        core_radius,
        subsampling_rg=0,
        max_n_spikes=np.inf,
        features_on_device=True,
        interpolation_sigma=20.0,
        interpolation_method="kriging",
        motion_depth_mode="channel",
        features_dataset_name="collisioncleaned_tpca_features",
        show_progress=False,
        store_on_device=True,
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

        # determine core channel neighborhoods
        core_neighborhoods = SpikeNeighborhoods.from_channels(core_channels)

        self = cls(
            kept_indices=kept_indices,
            prgeom=prgeom,
            tpca=tpca,
            extract_channels=extract_channels,
            core_channels=core_channels,
            original_sorting=sorting,
            core_features=core_features,
            extract_features=extract_features,
            core_neighborhoods=core_neighborhoods,
            features_on_device=features_on_device,
        )
        self.to(device)
        return self

    def spike_data(
        self,
        indices: torch.LongTensor,
        neighborhood: str = "extract",
        with_reconstructions: bool = False,
    ) -> "SpikeFeatures":
        if neighborhood == "extract":
            features = self.extract_features[indices]
            channels = self.extract_channels[indices]
        elif neighborhood == "core":
            features = self.core_features[indices]
            channels = self.core_channels[indices]
        else:
            assert False

        if not self.features_on_device:
            features = features.to(self.device)
            channels = channels.to(self.device)

        waveforms = None
        if with_reconstructions:
            raise NotImplementedError

        return SpikeFeatures(
            indices=indices, features=features, channels=channels, waveforms=waveforms
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
    channels: torch.LongTensor
    # n, t, c, only probided when with_reconstructions=True
    waveforms: Optional[torch.Tensor] = None


class SpikeNeighborhoods(torch.nn.Module):
    def __init__(self, neighborhood_ids, neighborhoods, neighborhood_members=None):
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
        self.register_buffer("neighborhood_ids", neighborhood_ids)
        self.register_buffer("neighborhoods", neighborhoods)
        self.n_neighborhoods = len(neighborhoods)

        if neighborhood_members is None:
            # cache lookups
            neighborhood_members = []
            for j in range(len(neighborhoods)):
                (in_nhood,) = torch.nonzero(neighborhood_ids == j, as_tuple=True)
                neighborhood_members.append(in_nhood)
        assert len(neighborhood_members) == self.n_neighborhoods

        # it's a pain to store dicts with register_buffer, so store offsets
        _neighborhood_members = torch.empty(
            sum(v.numel() for v in neighborhood_members.values), dtype=int
        )
        self.neighborhood_member_slices = []
        self.neighborhood_sizes = []
        neighborhood_member_offset = 0
        for j in range(len(neighborhoods)):
            nhoodmemsz = neighborhood_members[j].numel()
            nhoodmemsl = slice(
                neighborhood_member_offset, neighborhood_member_offset + nhoodmemsz
            )
            _neighborhood_members[nhoodmemsl] = neighborhood_members[j]
            self.neighborhood_slices.append(nhoodmemsl)
            neighborhood_member_offset += nhoodmemsz
        self.register_buffer("_neighborhood_members", _neighborhood_members)

    @classmethod
    def from_channels(cls, channels):
        neighborhoods, neighborhood_ids = torch.unique(channels, dim=0)
        return cls(neighborhoods=neighborhoods, neighborhood_ids=neighborhood_ids)

    def neighborhood_members(self, id):
        return self._neighborhood_members[self.neighborhood_members_slices[id]]

    def subset_neighborhood_ids(self, channels, min_coverage=1.0):
        ids = []
        for j, nhood in enumerate(self.neighborhoods):
            coverage = torch.isin(nhood, channels).sum() / self.neighborhood_sizes[j]
            if coverage >= min_coverage:
                ids.append(j)
        return ids


# -- helpers


def get_shift_info(sorting, motion_est, geom, keep_select, motion_depth_mode):
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
