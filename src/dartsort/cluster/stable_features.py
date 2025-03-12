from dataclasses import dataclass
from typing import Optional, Sequence

import h5py
from numba.core.options import Option
import numpy as np
import torch

from dartsort.transform.temporal_pca import TemporalPCAFeaturizer
from dartsort.util import drift_util, interpolation_util, spiketorch, waveform_util
from dartsort.util.data_util import DARTsortSorting, get_tpca

# -- main class


class StableSpikeDataset(torch.nn.Module):
    """
    ## On SpikeNeighborhoods

    Recall, we store spikes on two kinds of channel subsets: the larger
    "extract" and the smaller "core". Extract is used for fitting (M step)
    and core for likelihood computation (E step).

    During fitting, each unit determines some set of channels that it's
    "active" on. At E time, any spike whose core neighborhood is contained
    in those channels gets a likelihood computed (the assumption being that
    it would otherwise have smaller-than-noise likelihood under that unit
    due to determinant).

    It is not cheap to compute likelihoods on spikes from arbitrary channels
    willy nilly. It's better to compute likelihoods for all spikes living
    on the same exatch channel set at once. So, we cache info about what
    neighborhood each spike belongs to, and ways to reverse-lookup spikes
    by neighborhood. We even store the core spike features by neighborhood
    to avoid doing a million index_selects from large arrays.

    ## On splits

    We make use of at least two splits during inference: train and val.
    The train split is the one whose extract features we use during the M
    step, and whose core features we use during the E step. The val split
    is just used for model comparisons during split/merge.

    I say at least two because if the number of spikes is large, we will
    choose to completely ignore a random subset of spikes. The name of that
    split doesn't matter because it's not really used, but let's call it the
    "ignore" split.

    After fitting a model using the train and val sets, we can still
    compute likelihoods for all spikes, even the ignored ones.

    This means that:
     - For "extract" spikes, we only need to load the "train" split and
       make a SpikeNeighborhoods structure for the train split.
     - For "core" spikes, we need to load all of the spikes, and we need
       to cache SpikeNeighborhoods for train, train+val, and all 3.

    ## How are these used from the SpikeMixtureModel?

    In the M step, indices for fitting are chosen from the train split.

    The E step then also only needs to run on the train split most of the
    time. There is a `split='train'` flag for `log_likelihoods()` and for
    `e_step()`. This works by dispatching to the split in question's
    SpikeNeighborhoods structure. The returned sparse array of likelihoods



    """

    def __init__(
        self,
        kept_indices: np.ndarray,
        prgeom: torch.Tensor,
        tpca: TemporalPCAFeaturizer,
        extract_channels: torch.Tensor,
        core_channels: torch.Tensor,
        original_sorting: DARTsortSorting,
        core_features: torch.Tensor,
        train_extract_features: torch.Tensor,
        features_on_device: bool = False,
        interpolation_method: str = "kriging",
        interpolation_sigma: float = 20.0,
        split_names: Optional[Sequence[str]] = None,
        split_mask: Optional[torch.Tensor] = None,
        core_radius: float = 35.0,
        neighborhood_ids=None,
        neighborhood_ix=None,
        device=None,
        _core_feature_splits=("train", "kept"),
    ):
        """Motion-corrected spike data on the registered probe"""
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        self.features_on_device = features_on_device

        # data shapes
        self.rank = tpca.rank
        self.n_channels = prgeom.shape[0] - 1
        self.n_channels_extract = extract_channels.shape[1]
        self.n_channels_core = core_channels.shape[1]
        self.n_spikes = len(original_sorting)
        self.n_spikes_kept = len(kept_indices)
        self.interpolation_method = interpolation_method
        self.interpolation_sigma = interpolation_sigma
        self.core_radius = core_radius

        self.original_sorting = original_sorting

        # spike collection indices
        self.kept_indices = torch.from_numpy(kept_indices)
        # split indices are within kept_indices
        self.has_splits = split_names is not None
        self.split_mask = split_mask
        self.split_indices = dict(full=slice(None), kept=self.kept_indices)
        if self.has_splits:
            assert split_names is not None
            self.split_indices.update(
                (k, (split_mask == j).nonzero().view(-1))
                for j, k in enumerate(split_names)
            )

        # pca module, for reconstructing wfs for vis
        self.tpca = tpca

        # neighborhoods module, for querying spikes by channel group
        self.not_train_indices = torch.tensor([], dtype=int)
        if self.has_splits and "train" in self.split_indices:
            train_ixs = self.split_indices["train"]
            self.n_spikes_train = len(train_ixs)
            self._train_extract_neighborhoods = SpikeNeighborhoods.from_channels(
                extract_channels[train_ixs],
                n_channels=self.n_channels,
                neighborhood_ids=(
                    None if neighborhood_ids is None else neighborhood_ids[train_ixs]
                ),
                neighborhoods=extract_channels[neighborhood_ix],
                deduplicate=True,
                device=device,
                name="extract",
            )
            self._train_extract_channels = extract_channels.cpu()[train_ixs]
            self.not_train_indices = torch.from_numpy(
                np.setdiff1d(np.arange(self.n_spikes), train_ixs)
            )
        core_channel_index = waveform_util.make_channel_index(
            prgeom, core_radius, to_torch=True
        )
        _core_neighborhoods = {
            f"key_{k}": SpikeNeighborhoods.from_channels(
                core_channels[ix],
                n_channels=self.n_channels,
                neighborhood_ids=(
                    None if neighborhood_ids is None else neighborhood_ids[ix]
                ),
                neighborhoods=core_channels[neighborhood_ix],
                features=core_features[ix] if k in _core_feature_splits else None,
                device=device,
                channel_index=core_channel_index,
                name=k,
            )
            for k, ix in self.split_indices.items()
        }
        self._core_neighborhoods = torch.nn.ModuleDict(_core_neighborhoods)
        self.core_channels = core_channels.cpu()

        # channel neighborhoods and features
        # if not self.features_on_device, .spike_data() will .to(self.device)
        times_s = torch.asarray(self.original_sorting.times_seconds[kept_indices])
        self.core_features = core_features
        if self.features_on_device:
            self.register_buffer("_train_extract_features", train_extract_features)
            self.register_buffer("times_seconds", times_s)
        else:
            self._train_extract_features = train_extract_features
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
        motion_est=None,
        core_radius=35.0,
        max_n_spikes=np.inf,
        discard_triaged=False,
        interpolation_sigma=20.0,
        interpolation_method="kriging",
        motion_depth_mode="channel",
        features_dataset_name="collisioncleaned_tpca_features",
        split_names=("train", "val"),
        split_proportions=(0.75, 0.25),
        show_progress=True,
        store_on_device=False,
        workers=-1,
        device=None,
        random_seed=0,
        _core_feature_splits=("train", "kept"),
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # which spikes to keep?
        if discard_triaged:
            keep = sorting.labels >= 0
            keep_select = kept_inds = np.flatnonzero(keep)
        else:
            keep = np.ones(len(sorting), dtype=bool)
            kept_inds = np.arange(len(sorting))
            keep_select = slice(None)
        rg = random_seed
        if kept_inds.size > max_n_spikes:
            rg = np.random.default_rng(rg)
            kept_kept = rg.choice(kept_inds.size, size=int(max_n_spikes), replace=False)
            kept_kept.sort()
            keep[kept_inds] = 0
            keep[kept_inds[kept_kept]] = 1
            keep_select = kept_inds = kept_inds[kept_kept]

        # choose splits
        split_mask = None
        train_mask = None
        if split_names:
            n_splits = len(split_names)
            rg = np.random.default_rng(rg)
            split_mask = torch.full((len(sorting),), -1, dtype=torch.int8)
            split_choices = rg.choice(
                n_splits, p=split_proportions, size=len(kept_inds)
            )
            split_mask[keep_select] = torch.from_numpy(split_choices).to(split_mask)
            if "train" in split_names:
                train_mask = split_mask == split_names.index("train")

        # load information not stored directly on the sorting
        with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
            geom = h5["geom"][:]
            extract_channel_index = h5["channel_index"][:]
            assert np.all(np.diff(extract_channel_index) >= 0)
            if motion_est is None:
                registered_geom = geom
            else:
                registered_geom = drift_util.registered_geometry(
                    geom, motion_est=motion_est
                )
            prgeom = interpolation_util.pad_geom(registered_geom)

            times_s, channels, depths, rdepths, shifts = get_shift_info(
                sorting, motion_est, geom, motion_depth_mode
            )

            # determine channel occupancy of core/extract stable features
            extract_channels, core_channels, *neighb_info = get_stable_channels(
                geom,
                depths,
                rdepths,
                times_s,
                channels,
                extract_channel_index,
                registered_geom,
                motion_est,
                core_radius,
                workers=workers,
                device=device,
            )
            # for all spikes (not just kept), the ID of its shift/chan combo.
            # this determines its channel neighborhood under any registered index.
            # we also get the total counts in each combo, and the indices of
            # the first spikes in each combo, allowing neighbs to be reconstructed
            # as needed
            neighborhood_ids, neighborhood_ix = neighb_info
            extract_channels = torch.from_numpy(extract_channels)
            core_channels = torch.from_numpy(core_channels)
            if store_on_device:
                extract_channels = extract_channels.to(device)
                core_channels = core_channels.to(device)

            # stabilize features with interpolation
            # only load kept extract spikes
            train_extract_features = interpolation_util.interpolate_by_chunk(
                train_mask,
                h5[features_dataset_name],
                geom,
                extract_channel_index,
                sorting.channels[train_mask],
                shifts,
                registered_geom,
                extract_channels[train_mask],
                sigma=interpolation_sigma,
                interpolation_method=interpolation_method,
                device=device,
                store_on_device=store_on_device,
                show_progress=show_progress,
            )
            # always load all core spikes
            core_features = interpolation_util.interpolate_by_chunk(
                np.ones_like(keep),
                h5[features_dataset_name],
                geom,
                extract_channel_index,
                sorting.channels,
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

        self = cls(
            kept_indices=kept_inds,
            prgeom=prgeom,
            tpca=tpca,
            extract_channels=extract_channels,
            core_channels=core_channels,
            neighborhood_ids=neighborhood_ids,
            neighborhood_ix=neighborhood_ix,
            original_sorting=sorting,
            core_features=core_features,
            train_extract_features=train_extract_features,
            features_on_device=store_on_device,
            interpolation_method=interpolation_method,
            interpolation_sigma=interpolation_sigma,
            split_names=split_names,
            split_mask=split_mask,
            core_radius=core_radius,
            device=device,
            _core_feature_splits=_core_feature_splits,
        )
        self.to(device)
        return self

    def spike_data(
        self,
        indices: torch.Tensor,
        split_indices: Optional[torch.Tensor] = None,
        neighborhood: str = "extract",
        with_channels=True,
        with_reconstructions: bool = False,
        with_neighborhood_ids: bool = False,
        split: Optional[str] = "train",
        feature_buffer: Optional[torch.Tensor] = None,
    ) -> "SpikeFeatures":
        withbuf = feature_buffer is not None
        non_blocking = False
        channels = neighborhood_ids = None

        if neighborhood == "extract":
            assert split == "train"
            assert split_indices is not None

            if withbuf:
                features = feature_buffer[: len(indices)]
                non_blocking = features.is_pinned()
                torch.index_select(
                    self._train_extract_features, 0, split_indices, out=features
                )
            else:
                features = self._train_extract_features[split_indices]
            if with_channels:
                channels = self._train_extract_channels[split_indices]
            if with_neighborhood_ids:
                neighborhood_ids = self._train_extract_neighborhoods.neighborhood_ids[
                    split_indices
                ]
        elif neighborhood == "core":
            if withbuf:
                features = feature_buffer[: len(indices)]
                non_blocking = features.is_pinned()
                torch.index_select(self.core_features, 0, indices, out=features)
            else:
                features = self.core_features[indices]
            if with_channels:
                channels = self.core_channels[indices]
            if with_neighborhood_ids:
                _, core_neighborhoods = self.neighborhoods(split="full")
                neighborhood_ids = core_neighborhoods.neighborhood_ids[indices]
        else:
            assert False

        if not self.features_on_device:
            features = features.to(self.device, non_blocking=non_blocking)
            if with_channels:
                channels = channels

        waveforms = None
        if with_reconstructions:
            n = len(features)
            c = channels.shape[1]
            waveforms = features.permute(0, 2, 1).reshape(n * c, -1)
            waveforms = self.tpca._inverse_transform_in_probe(waveforms)
            waveforms = waveforms.reshape(n, c, -1).permute(0, 2, 1)

        return SpikeFeatures(
            indices=indices,
            split_indices=split_indices,
            features=features,
            channels=channels,
            waveforms=waveforms,
            neighborhood_ids=neighborhood_ids,
        )

    def neighborhoods(
        self, neighborhood="core", split="train"
    ) -> tuple[torch.LongTensor, "SpikeNeighborhoods"]:
        split_ixs = self.split_indices[split]

        if neighborhood == "extract":
            assert split == "train"
            return split_ixs, self._train_extract_neighborhoods

        return split_ixs, self._core_neighborhoods[f"key_{split}"]

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
    indices: torch.Tensor
    # n, r, c
    features: torch.Tensor
    # n, c
    channels: Optional[torch.Tensor] = None
    # n, t, c
    waveforms: Optional[torch.Tensor] = None
    # n
    neighborhood_ids: Optional[torch.Tensor] = None
    split_indices: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, ix):
        """Subset the spikes in this collection with [subset]."""
        waveforms = channels = neighborhood_ids = split_indices = None
        if self.channels is not None:
            channels = self.channels[ix]
        if self.waveforms is not None:
            waveforms = self.waveforms[ix]
        if self.neighborhood_ids is not None:
            neighborhood_ids = self.neighborhood_ids[ix]
        if self.split_indices is not None:
            split_indices = self.split_indices[ix]
        return self.__class__(
            indices=self.indices[ix],
            features=self.features[ix],
            channels=channels,
            waveforms=waveforms,
            neighborhood_ids=neighborhood_ids,
            split_indices=split_indices,
        )

    def __repr__(self):
        indstr = f"indices.shape={self.indices.shape},"
        featstr = f"features.shape={self.features.shape},"
        chanstr = wfstr = idstr = ""
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
        features=None,
        neighborhood_members=None,
        store_on_device: bool = False,
        channel_index=None,
        device=None,
        name=None,
    ):
        """SpikeNeighborhoods

        Sparsely keep track of which channels each spike lives on. Used to query
        which core sets are overlapped completely by unit channel neighborhoods.

        Arguments
        ---------
        neighborhood_ids : torch.Tensor
            Size (n_spikes,), the neighborhood id for each spike
        neighborhoods : list[torch.Tensor]
            The channels in each neighborhood
        neighborhood_members : list[torch.Tensor]
            The indices of spikes in each neighborhood
        """
        super().__init__()
        self.n_channels = n_channels
        self.register_buffer("chans_arange", torch.arange(n_channels))
        if store_on_device:
            self.register_buffer("neighborhood_ids", neighborhood_ids)
        else:
            self.neighborhood_ids = neighborhood_ids.cpu()
        self.register_buffer("neighborhoods", neighborhoods)
        # self.neighborhoods = neighborhoods.cpu()
        self.n_neighborhoods = len(neighborhoods)
        if channel_index is not None:
            self.register_buffer("channel_index", channel_index)
        self.name = name

        # store neighborhoods as an indicator matrix
        # also store nonzero-d masks
        indicators = torch.zeros((n_channels, len(neighborhoods)))
        masks = []
        mask_slices = []
        offset = 0
        for j, nhood in enumerate(neighborhoods):
            jvalid = nhood < n_channels
            indicators[nhood[jvalid], j] = 1.0
            (jvalid,) = jvalid.nonzero(as_tuple=True)
            njvalid = jvalid.numel()
            assert njvalid
            masks.append(jvalid)
            mask_slices.append(slice(offset, offset + njvalid))
            offset += njvalid
        self.register_buffer("indicators", indicators)
        self.register_buffer("channel_counts", indicators.sum(0))
        self.register_buffer("_masks", torch.concatenate(masks, dim=0))
        self._mask_slices = mask_slices

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

        if features is not None:
            _features_valid = []
            for j in range(len(neighborhoods)):
                f = features[self.neighborhood_members(j)]
                f = f[..., self.valid_mask(j).to(f.device)]
                if device is not None and device.type == "cuda":
                    f = f.pin_memory()
                _features_valid.append(f)
            self._features_valid = _features_valid

    @classmethod
    def from_channels(
        cls,
        channels,
        n_channels,
        neighborhood_ids=None,
        neighborhoods=None,
        device=None,
        deduplicate=False,
        features=None,
        channel_index=None,
        name=None,
    ):
        if neighborhood_ids is not None:
            assert neighborhoods is not None
            return cls.from_known_ids(
                channels,
                n_channels,
                neighborhood_ids,
                neighborhoods,
                device=device,
                deduplicate=deduplicate,
                features=features,
                channel_index=channel_index,
                name=name,
            )
        if device is not None:
            channels = channels.to(device)
        neighborhoods, neighborhood_ids = torch.unique(
            channels, dim=0, return_inverse=True
        )
        return cls(
            n_channels=n_channels,
            neighborhoods=neighborhoods,
            neighborhood_ids=neighborhood_ids,
            features=features,
            device=channels.device,
            channel_index=channel_index,
            name=name,
        )

    @classmethod
    def from_known_ids(
        cls,
        channels,
        n_channels,
        neighborhood_ids,
        neighborhoods,
        device=None,
        deduplicate=False,
        features=None,
        channel_index=None,
        name=None,
    ):
        neighborhoods = torch.asarray(neighborhoods)
        if device is not None:
            neighborhoods = neighborhoods.to(device)
            neighborhood_ids = neighborhood_ids.to(device)
        if deduplicate:
            neighborhoods, old2new = torch.unique(
                neighborhoods, dim=0, return_inverse=True
            )
            neighborhood_ids = old2new[neighborhood_ids]
            kept_ids, neighborhood_ids = torch.unique(
                neighborhood_ids, return_inverse=True
            )
        return cls(
            n_channels=n_channels,
            neighborhoods=neighborhoods,
            neighborhood_ids=neighborhood_ids,
            features=features,
            device=device,
            channel_index=channel_index,
            name=name,
        )

    def has_feature_cache(self):
        return hasattr(self, "_features_valid")

    def valid_mask(self, id):
        return self._masks[self._mask_slices[id]]

    def neighborhood_channels(self, id):
        nhc = self.neighborhoods[id]
        return nhc[nhc < self.n_channels]

    def missing_channels(self, id):
        return self.chans_arange[self.indicators[:, id] == 0]

    def neighborhood_members(self, id):
        return self._neighborhood_members[self.neighborhood_members_slices[id]]

    def neighborhood_features(
        self, id, batch_start=None, batch_size=None, batch_buffer=None
    ):
        f = self._features_valid[id]
        if batch_start is not None:
            f = f[batch_start : batch_start + batch_size]
        if batch_buffer is not None:
            batch_buffer[: len(f)] = f
            return batch_buffer[: len(f)]
        else:
            return f

    def subset_neighborhoods(
        self, channels, min_coverage=1.0, add_to_overlaps=None, batch_size=None
    ):
        """Return info on neighborhoods which cover the channel set well enough

        Define coverage for a neighborhood and a channel group as the intersection
        size divided by the neighborhood's size.

        Returns
        -------
        neighborhood_info : list of tuples
            Each entry is, in order,
             - neighborhood id
             - neighborhood channels array
             - neighborhood member indices
             - optional batch start
            representing a batch of spikes living on that neighborhood.
        n_spikes : int
            The total number of spikes in the neighborhood.
        """
        inds = self.indicators[channels]
        coverage = inds.sum(0) / self.channel_counts
        (covered_ids,) = torch.nonzero(coverage >= min_coverage, as_tuple=True)
        n_spikes = self.popcounts[covered_ids].sum()

        neighborhood_info = []
        for j in covered_ids:
            jneighb = self.neighborhoods[j]
            jmems = self.neighborhood_members(j)
            if batch_size is None or len(jmems) < batch_size:
                neighborhood_info.append((j, jneighb, jmems, None))
            else:
                for bs in range(0, len(jmems), batch_size):
                    mem_batch = jmems[bs : bs + batch_size]
                    neighborhood_info.append((j, jneighb, mem_batch, bs))

        return covered_ids, neighborhood_info, n_spikes

    def spike_neighborhoods(
        self, channels, neighborhood_ids=None, spike_indices=None, min_coverage=1.0
    ):
        """Like subset_neighborhoods, but for an already chosen collection of spikes

        This is used when subsetting log likelihood calculations.
        In this case, the returned neighborhood_member_indices keys are relative:
        spike_indices[neighborhood_member_indices] are the actual indices.
        """
        if neighborhood_ids is None:
            assert spike_indices is not None
            neighborhood_ids = self.neighborhood_ids[spike_indices]
        assert neighborhood_ids is not None

        covered_ids = torch.unique(neighborhood_ids)
        if min_coverage:
            covered_ids = covered_ids.to(self.indicators.device)
            inds = self.indicators[channels][:, covered_ids]
            coverage = inds.sum(0) / self.channel_counts[covered_ids]
            covered = coverage >= min_coverage
            covered_ids = covered_ids[covered].cpu()
            neighborhood_ids = neighborhood_ids.cpu()

        neighborhood_info = [
            (
                j,
                self.neighborhoods[j],
                *(neighborhood_ids == j).nonzero(as_tuple=True),
                None,
            )
            for j in covered_ids
        ]
        n_spikes = self.popcounts[covered_ids].sum()
        return neighborhood_info, n_spikes


# -- helpers


def occupied_chans(
    spike_data,
    n_channels,
    neighborhood_ids=None,
    neighborhoods=None,
    fuzz=0,
    weights=None,
):
    if spike_data.neighborhood_ids is None:
        chans = torch.unique(spike_data.channels)
        return chans[chans < n_channels]
    assert neighborhoods is not None
    if neighborhood_ids is None:
        neighborhood_ids = spike_data.neighborhood_ids
    ids, inverse = torch.unique(neighborhood_ids, return_inverse=True)
    if weights is None:
        weights = torch.ones(inverse.shape, device=ids.device)
    id_counts = torch.zeros(ids.shape, device=ids.device)
    spiketorch.add_at_(id_counts, inverse, weights.to(ids.device))

    chans0 = neighborhoods.neighborhoods[ids]
    chans, inverse = torch.unique(chans0, return_inverse=True)
    counts = torch.zeros(chans.shape, device=chans.device)
    spiketorch.add_at_(
        counts,
        inverse.view(-1),
        id_counts[:, None].broadcast_to(chans0.shape).reshape(-1).to(counts.device),
    )

    counts = counts[chans < n_channels]
    chans = chans[chans < n_channels]
    for _ in range(fuzz):
        chans = neighborhoods.channel_index[chans]
        chans = torch.unique(chans)
        chans = chans[chans < n_channels]

    return chans, counts


def interp_to_chans(
    spike_data,
    channels,
    prgeom,
    interpolation_method="kriging",
    interpolation_sigma=20.0,
    batch_size=256,
):
    source_pos = prgeom[spike_data.channels]
    target_pos = prgeom[channels]
    shape = len(spike_data), *target_pos.shape
    output_shape = *spike_data.features.shape[:2], channels.numel()
    output = spike_data.features.new_empty(output_shape)
    if not channels.numel():
        return output
    for batch_start in range(0, len(spike_data), batch_size):
        batch_end = min(batch_start + batch_size, len(spike_data))
        sl = slice(batch_start, batch_end)
        shape = batch_end - batch_start, *target_pos.shape
        target_pos_ = target_pos[None].broadcast_to(shape)
        interpolation_util.kernel_interpolate(
            spike_data.features[sl],
            source_pos[sl],
            target_pos_,
            sigma=interpolation_sigma,
            allow_destroy=False,
            interpolation_method=interpolation_method,
            out=output[sl],
        )
    return output


def pad_to_chans(
    spike_data, channels, n_channels, weights=None, target_padded=None, pad_value=0.0
):
    n, r, c = spike_data.features.shape
    c_targ = channels.numel()

    # determine channels to write to
    reindexer = get_channel_reindexer(channels, n_channels)
    target_ixs = reindexer[spike_data.channels].to(spike_data.features.device)

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
    channels : Tensor
        Shape (n_chans_subset,)
    n_channels : int

    Returns
    -------
    reindexer : Tensor
        Shape (n_channels + 1,)
        reindexer[i] is the index of i in channels, if present.
        Otherwise, it is n_chans_subset.
        And the last entry is, of course, n_chans_subset.
    """
    reindexer = channels.new_full((n_channels + 1,), channels.numel())
    (rel_ixs,) = torch.nonzero(channels < n_channels, as_tuple=True)
    reindexer[channels[rel_ixs]] = rel_ixs
    return reindexer


def get_shift_info(sorting, motion_est, geom, motion_depth_mode):
    """
    shifts = reg_depths - depths
    reg_depths = depths + shifts
    i.e., target_pos = source_pos + shifts
          target_pos - shifts = source_pos
    where by source pos i mean the true moving positions.
    """
    times_s = sorting.times_seconds
    channels = sorting.channels
    if motion_depth_mode == "localization":
        depths = sorting.point_source_localizations[:, 2]
    elif motion_depth_mode == "channel":
        depths = geom[:, 1][channels]
    else:
        assert False

    rdepths = depths
    if motion_est is not None:
        rdepths = motion_est.correct_s(times_s, depths)
    shifts = rdepths - depths

    return times_s, channels, depths, rdepths, shifts


def get_stable_channels(
    geom,
    depths,
    rdepths,
    times_s,
    channels,
    channel_index,
    registered_geom,
    motion_est,
    core_radius,
    workers=-1,
    device=None,
):
    core_channel_index = waveform_util.make_channel_index(geom, core_radius)

    pitch = drift_util.get_pitch(geom)
    registered_kdtree = drift_util.KDTree(registered_geom)
    match_distance = drift_util.pdist(geom).min() / 2

    if motion_est is None:
        n_pitches_shift = np.zeros(len(depths), dtype=int)
    else:
        n_pitches_shift = drift_util.get_spike_pitch_shifts(
            depths,
            geom=geom,
            motion_est=motion_est,
            times_s=times_s,
            registered_depths_um=rdepths,
            mode="round",
        )

    # extract the main unique chans computation
    c = torch.asarray(channels, dtype=torch.int32)
    s = torch.asarray(n_pitches_shift, dtype=torch.int32)
    cs = torch.column_stack((c, s))
    if device is not None:
        cs = cs.to(device)
    uniq_channels_and_shifts, *neighb_info = unique_with_index(cs, dim=0)
    neighborhood_ids, neighborhood_counts, neighborhood_ix = neighb_info
    neighborhood_ix = neighborhood_ix.cpu()
    uniq_channels_and_shifts = uniq_channels_and_shifts.numpy(force=True)
    neighborhood_ids_ = neighborhood_ids.numpy(force=True)

    extract_channels = drift_util.static_channel_neighborhoods(
        geom,
        channels,
        channel_index,
        pitch=pitch,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        uniq_channels_and_shifts=uniq_channels_and_shifts,
        uniq_inv=neighborhood_ids_,
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
        uniq_channels_and_shifts=uniq_channels_and_shifts,
        uniq_inv=neighborhood_ids_,
        workers=workers,
    )

    return (
        extract_channels,
        core_channels,
        neighborhood_ids,
        neighborhood_ix,
    )


def unique_with_index(x, dim=0):
    # https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index
