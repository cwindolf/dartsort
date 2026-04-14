from dataclasses import dataclass
from typing import Literal, Self, Sequence

import h5py
import numpy as np
import torch

from ...transform.temporal_pca import TemporalPCAFeaturizer
from ...util import drift_util, interpolation_util, spiketorch
from ...util.data_util import DARTsortSorting, get_tpca
from ...util.internal_config import (
    InterpolationParams,
    RefinementConfig,
    default_extrapolation_params,
    tps_interp_params,
)
from ...util.interpolation_util import SpikeNeighborhoods
from ...util.job_util import get_global_computation_config
from ...util.motion import MotionInfo
from ...util.torch_util import BModule

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
        core_channels: torch.Tensor | None,
        original_sorting: DARTsortSorting,
        core_features: torch.Tensor | None,
        train_extract_features: torch.Tensor,
        features_on_device: bool = False,
        split_names: Sequence[str] | None = None,
        split_mask: torch.Tensor | None = None,
        core_radius: float | Literal["extract"] | None = 35.0,
        interp_params: InterpolationParams = tps_interp_params,
        extrap_params: InterpolationParams = default_extrapolation_params,
        extract_neighborhoods=None,
        extract_neighborhood_ids=None,
        core_neighborhoods=None,
        core_neighborhood_ids=None,
        device=None,
        _core_feature_splits=("train", "kept"),
    ):
        """Motion-corrected spike data on the registered probe"""
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        self.original_sorting = original_sorting

        self.features_on_device = features_on_device

        # data shapes
        self.rank = tpca.rank
        self.n_channels = prgeom.shape[0] - 1
        self.n_channels_extract = extract_channels.shape[1]
        self.core_is_extract = core_radius == "extract"
        if core_radius is not None:
            assert core_channels is not None
            self.n_channels_core = core_channels.shape[1]
        self.n_spikes = len(original_sorting)
        # train is modified below if there is a train split.
        self.n_spikes_train = self.n_spikes_kept = len(kept_indices)
        self.core_radius = core_radius
        # assert extrap_method != "kriging"
        self.interp_params = interp_params.normalize()
        self.extrap_params = extrap_params.normalize()

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
        self.not_train_indices = torch.tensor([], dtype=torch.long)
        if self.has_splits and "train" in self.split_indices:
            train_ixs = self.split_indices["train"]
            self.n_spikes_train = len(train_ixs)  # type: ignore
            train_enids = None
            if extract_neighborhood_ids is not None:
                train_enids = extract_neighborhood_ids[train_ixs]
            self._train_extract_neighborhoods = SpikeNeighborhoods.from_channels(
                extract_channels[train_ixs],
                n_channels=self.n_channels,
                neighborhood_ids=train_enids,
                neighborhoods=extract_neighborhoods,
                deduplicate=not self.core_is_extract,
                device=device,
                name="extract",
            )
            self._train_extract_channels = extract_channels.cpu()[train_ixs]
            assert torch.is_tensor(train_ixs)
            self.not_train_indices = torch.asarray(
                np.setdiff1d(np.arange(self.n_spikes), train_ixs),
                dtype=torch.long,
            )
        else:
            train_ixs = None

        if core_radius is not None:
            _core_neighborhoods = {
                f"key_{k}": SpikeNeighborhoods.from_channels(
                    channels=core_channels[ix],  # type: ignore
                    n_channels=self.n_channels,
                    neighborhood_ids=(
                        None
                        if core_neighborhood_ids is None
                        else core_neighborhood_ids[ix]
                    ),
                    neighborhoods=core_neighborhoods,
                    features=core_features[ix] if k in _core_feature_splits else None,  # type: ignore
                    device=device,
                    name=k,
                )
                for k, ix in self.split_indices.items()
                if (not self.core_is_extract or k != "train")
            }
            if self.core_is_extract and "train" in self.split_indices:
                assert torch.is_tensor(train_ixs)
                assert core_channels is not None
                assert torch.equal(
                    core_channels[train_ixs], extract_channels[train_ixs]
                )
                _core_neighborhoods["train"] = self._train_extract_neighborhoods
            self._core_neighborhoods = torch.nn.ModuleDict(_core_neighborhoods)
            self.core_channels = core_channels.cpu()  # type: ignore

        # channel neighborhoods and features
        # if not self.features_on_device, .spike_data() will .to(self.device)
        times_s = torch.asarray(self.original_sorting.times_seconds)  # type: ignore[reportAttributeAccessIssue]
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
    def device(self) -> torch.device:
        return self.prgeom.device  # type: ignore

    @property
    def dtype(self):
        return self._train_extract_features.dtype

    @classmethod
    def from_config(
        cls,
        sorting: DARTsortSorting,
        refinement_cfg: RefinementConfig,
        motion: MotionInfo,
        computation_cfg=None,
        _core_feature_splits=("train", "kept"),
    ):
        if computation_cfg is None:
            computation_cfg = get_global_computation_config()
        return cls.from_sorting(
            sorting,
            motion=motion,
            core_radius=refinement_cfg.core_radius,
            max_n_spikes=refinement_cfg.sampling_cfg.max_waveforms_fit,
            interp_params=refinement_cfg.interp_params.normalize(),
            split_proportions=(
                1.0 - refinement_cfg.val_proportion,
                refinement_cfg.val_proportion,
            ),
            device=computation_cfg.actual_device(),
            _core_feature_splits=_core_feature_splits,
        )

    @classmethod
    def from_sorting(
        cls,
        sorting: DARTsortSorting,
        motion: MotionInfo,
        core_radius: float | Literal["extract"] | None = 35.0,
        max_n_spikes: float | int = np.inf,
        kept_indices: np.ndarray | None = None,
        discard_triaged: bool = False,
        interp_params: InterpolationParams = tps_interp_params,
        features_dataset_name="collisioncleaned_tpca_features",
        motion_depth_mode: Literal["channel", "localization"] = "channel",
        split_names=("train", "val"),
        split_proportions=(0.75, 0.25),
        show_progress=True,
        store_on_device=False,
        workers=-1,
        device=None,
        feature_rank: int = 8,
        random_seed: int | np.random.Generator = 0,
        _core_feature_splits=("train", "kept"),
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # load information not stored directly on the sorting
        with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
            geom = h5["geom"][:]  # type: ignore
            extract_channel_index: np.ndarray = h5["channel_index"][:]  # type: ignore
            assert np.all(np.diff(extract_channel_index) >= 0)
            prgeom = interpolation_util.pad_geom(motion.rgeom)
            shifts, n_pitches_shift = motion.pitch_shifts(
                sorting=sorting, motion_depth_mode=motion_depth_mode
            )

            # determine channel occupancy of core/extract stable features
            res = drift_util.get_stable_channels(
                motion=motion,
                channels=sorting.channels,
                channel_index=extract_channel_index,
                n_pitches_shift=n_pitches_shift,
                core_radius=core_radius,
                workers=workers,
                device=device,
            )

            extract_channels, extract_neighborhoods, extract_neighborhood_ids = res[:3]
            core_channels, core_neighborhoods, core_neighborhood_ids = res[3:]
            if core_radius is not None:
                assert core_channels is not None
                assert core_neighborhoods is not None
                assert core_neighborhood_ids is not None

            if core_radius == "extract":
                assert core_neighborhoods is not None
                assert core_neighborhood_ids is not None
                assert np.array_equal(extract_neighborhoods, core_neighborhoods)
                assert np.array_equal(extract_neighborhood_ids, core_neighborhood_ids)

            rg = np.random.default_rng(random_seed)

            assert len(split_names) >= 1
            assert "train" in split_names
            if kept_indices is not None:
                pass
            elif discard_triaged:
                assert sorting.labels is not None
                kept_indices = np.flatnonzero(sorting.labels >= 0)
            elif len(sorting) <= max_n_spikes:
                kept_indices = np.arange(len(sorting))
            else:
                assert isinstance(max_n_spikes, int)
                kept_indices = rg.choice(len(sorting), size=max_n_spikes, replace=False)
                kept_indices.sort()

            split_mask = torch.full((len(sorting),), -1, dtype=torch.int8)
            # train set initialized to everything
            assert split_names[0] == "train"
            split_mask[kept_indices] = 0
            # val takes from uncovered
            if "val" in split_names:
                assert "val" == split_names[1]
                assert split_names == ("train", "val")
                n_val = int(np.ceil(split_proportions[1] * len(kept_indices)))
                val_candidates = rg.choice(kept_indices, size=n_val, replace=False)
                val_candidates.sort()
                split_mask[val_candidates] = 1
            else:
                assert split_names == ("train",)
            train_mask = (split_mask == 0).numpy()

            # for all spikes (not just kept), the ID of its shift/chan combo.
            # this determines its channel neighborhood under any registered index.
            # we also get the total counts in each combo, and the indices of
            # the first spikes in each combo, allowing neighbs to be reconstructed
            # as needed
            extract_channels = torch.from_numpy(extract_channels)
            if store_on_device:
                extract_channels = extract_channels.to(device)

            if core_radius is not None:
                core_channels = torch.from_numpy(core_channels)
                if store_on_device:
                    core_channels = core_channels.to(device)

            # stabilize features with interpolation
            # only load kept extract spikes
            train_extract_features = interpolation_util.interpolate_by_chunk(
                train_mask,
                h5[features_dataset_name],
                geom,
                extract_channel_index,
                sorting.channels[train_mask],
                shifts[train_mask],
                motion.rgeom,
                extract_channels[train_mask],
                trim_to_rank=feature_rank,
                params=interp_params,
                device=device,
                store_on_device=store_on_device,
                show_progress=show_progress,
            )
            # always load all core spikes
            core_features = None
            if core_radius is not None:
                core_features = interpolation_util.interpolate_by_chunk(
                    np.ones(len(sorting), dtype=np.bool_),
                    h5[features_dataset_name],
                    geom,
                    extract_channel_index,
                    sorting.channels,
                    shifts,
                    motion.rgeom,
                    core_channels,
                    trim_to_rank=feature_rank,
                    params=interp_params,
                    device=device,
                    store_on_device=store_on_device,
                    show_progress=show_progress,
                )

        # load temporal PCA
        tpca = get_tpca(sorting)

        if core_radius == "extract":
            core_radius = -np.inf
            for j in range(len(extract_channel_index)):
                chans = extract_channel_index[j]
                assert isinstance(geom, np.ndarray)
                chans = chans[chans < len(geom)]
                tmp = np.subtract(geom[chans], geom[j])
                np.square(tmp, out=tmp)
                max_dist = tmp.sum(axis=1).max()
                core_radius = max(core_radius, np.sqrt(max_dist))

        self = cls(
            kept_indices=kept_indices,
            prgeom=prgeom,
            tpca=tpca,  # type: ignore
            extract_channels=extract_channels,
            extract_neighborhoods=extract_neighborhoods,
            extract_neighborhood_ids=extract_neighborhood_ids,
            core_channels=core_channels,  # type: ignore
            core_neighborhoods=core_neighborhoods,
            core_neighborhood_ids=core_neighborhood_ids,
            original_sorting=sorting,
            core_features=core_features,
            train_extract_features=train_extract_features,
            features_on_device=store_on_device,
            split_names=split_names,
            split_mask=split_mask,
            core_radius=core_radius,
            device=device,
            interp_params=interp_params,
            _core_feature_splits=_core_feature_splits,
        )
        self.to(device)
        return self

    def spike_data(
        self,
        indices: torch.Tensor,
        split_indices: torch.Tensor | None = None,
        neighborhood: str = "extract",
        with_channels=True,
        with_reconstructions: bool = False,
        with_neighborhood_ids: bool = False,
        split: str | None = "train",
        feature_buffer: torch.Tensor | None = None,
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
                neighborhood_ids = self._train_extract_neighborhoods.b.neighborhood_ids[
                    split_indices
                ]
        elif neighborhood == "core":
            if withbuf:
                features = feature_buffer[: len(indices)]
                non_blocking = features.is_pinned()
                torch.index_select(self.core_features, 0, indices, out=features)  # type: ignore
            else:
                features = self.core_features[indices]  # type: ignore
            if with_channels:
                channels = self.core_channels[indices]
            if with_neighborhood_ids:
                _, core_neighborhoods = self.neighborhoods(split="full")
                neighborhood_ids = core_neighborhoods.b.neighborhood_ids[indices]
        else:
            assert False

        if not self.features_on_device:
            features = features.to(self.device, non_blocking=non_blocking)
            if with_channels:
                channels = channels

        waveforms = None
        if with_reconstructions:
            n = len(features)
            assert channels is not None
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
    ) -> tuple[torch.Tensor | slice, "SpikeNeighborhoods"]:
        split_ixs = self.split_indices[split]

        if neighborhood == "extract":
            assert split == "train"
            return split_ixs, self._train_extract_neighborhoods

        return split_ixs, self._core_neighborhoods[f"key_{split}"]  # type: ignore

    def interp_to_chans(self, spike_data, channels):
        return interp_to_chans(
            spike_data, channels, self.prgeom, params=self.extrap_params
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
    channels: torch.Tensor | None = None
    # n, t, c
    waveforms: torch.Tensor | None = None
    # n
    neighborhood_ids: torch.Tensor | None = None
    split_indices: torch.Tensor | None = None

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
        inverse.view(-1).to(counts.device),
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
    params,
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
            params=params,
            allow_destroy=False,
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
