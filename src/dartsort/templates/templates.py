from dataclasses import dataclass, replace
import gc
from logging import getLogger
from pathlib import Path
from sys import getrefcount

import numpy as np
import torch
from tqdm.auto import trange

from ..localize.localize_util import localize_waveforms
from ..util import data_util, drift_util, job_util
from ..util.data_util import DARTsortSorting
from ..util.internal_config import default_waveform_cfg
from ..util.spiketorch import fast_nanmedian, nanmean

from .get_templates import get_templates, apply_time_shifts
from .superres_util import superres_sorting
from .template_util import get_realigned_sorting, weighted_average

_motion_error_prefix = (
    "If template_cfg has registered_templates==True or superres_templates=True, then "
)
_aware_error = "motion_est must be passed to TemplateData.from_config()"

logger = getLogger(__name__)


@dataclass
class TemplateData:
    # (n_templates, spike_length_samples, n_registered_channels or n_channels)
    templates: np.ndarray
    # (n_templates,) maps template index to unit index (multiple templates can share a unit index)
    unit_ids: np.ndarray
    # (n_templates,) spike count for each template
    spike_counts: np.ndarray
    # (n_templates, n_registered_channels or n_channels) spike count for each channel
    spike_counts_by_channel: np.ndarray | None = None
    # (n_templates, spike_length_samples, n_registered_channels or n_channels)
    raw_std_dev: np.ndarray | None = None

    registered_geom: np.ndarray | None = None
    trough_offset_samples: int = 42

    # stores (n_templates, *) arrays of template properties
    # possibilities:
    #  - motion_estimate_bin_centers
    #    For motion_estimate binned superres sortings, these are the drift
    #    bin centers for each template.
    properties: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        assert self.trough_offset_samples < self.spike_length_samples

        ntemp = len(self.templates)
        assert self.unit_ids.shape == (ntemp,)
        assert self.spike_counts.shape == (ntemp,)
        if self.spike_counts_by_channel is not None:
            assert self.spike_counts_by_channel.ndim == 2
            assert self.spike_counts_by_channel.shape[0] == ntemp
        if self.raw_std_dev is not None:
            assert self.raw_std_dev.shape == self.templates.shape

        nc = self.templates.shape[2]
        if self.spike_counts_by_channel is not None:
            assert self.spike_counts_by_channel.shape[1] == nc
        if self.registered_geom is not None:
            assert self.registered_geom.ndim == 2
            assert self.registered_geom.shape[0] == nc
        if self.properties:
            for v in self.properties.values():
                assert v.shape[0] == ntemp

    @property
    def spike_length_samples(self):
        return self.templates.shape[1]

    def snrs_by_channel(self):
        amp_vecs = np.nan_to_num(np.ptp(self.templates, axis=1), nan=-np.inf)
        if self.spike_counts_by_channel is not None:
            amp_vecs *= np.sqrt(self.spike_counts_by_channel)
        return amp_vecs

    def main_channels(self):
        return self.snrs_by_channel().argmax(axis=1)

    def template_locations(self, mode="channel", radius=100.0):
        assert mode in ("localization", "channel")

        if mode == "channel":
            assert self.registered_geom is not None
            return self.registered_geom[self.main_channels()]

        assert mode == "localization"
        rdepths = localize_waveforms(
            waveforms=self.templates,
            geom=self.registered_geom,
            main_channels=self.main_channels(),
            radius=radius,
        )
        rdepths = np.c_[rdepths["x"], rdepths["z_abs"]]
        return rdepths

    def registered_depths_um(self, mode="channel", radius=100.0):
        return self.template_locations(mode=mode, radius=radius)[:, 1]

    @classmethod
    def from_npz(cls, npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            data = dict(**data)
            if "spike_length_samples" in data:
                del data["spike_length_samples"]  # todo: remove
            if "parent_sorting_hdf5_path" in data:
                del data["parent_sorting_hdf5_path"]  # todo: remove

            properties = {}
            for k, v in data.items():
                if k.startswith("__prop_"):
                    properties[k.removeprefix("__prop_")] = v
            for k in properties:
                del data[f"__prop_{k}"]
            if not properties:
                properties = None

            return cls(**data, properties=properties)

    def to_npz(self, npz_path):
        to_save = dict(
            templates=self.templates,
            unit_ids=self.unit_ids,
            spike_counts=self.spike_counts,
            trough_offset_samples=self.trough_offset_samples,
        )
        if self.registered_geom is not None:
            to_save["registered_geom"] = self.registered_geom
        if self.spike_counts_by_channel is not None:
            to_save["spike_counts_by_channel"] = self.spike_counts_by_channel
        if self.raw_std_dev is not None:
            to_save["raw_std_dev"] = self.raw_std_dev
        if not npz_path.parent.exists():
            npz_path.parent.mkdir()
        if self.properties is not None:
            for k, p in self.properties.items():
                to_save[f"__prop_{k}"] = p
        np.savez(npz_path, **to_save)  # type: ignore

    def __getitem__(self, subset):
        if not np.array_equal(self.unit_ids, np.arange(len(self.unit_ids))):
            subset_ixs = np.searchsorted(self.unit_ids, subset, side="right") - 1
            matched = self.unit_ids[subset_ixs] == subset
            assert matched.all()
            subset = subset_ixs
        if self.properties is not None:
            properties = {k: p[subset] for k, p in self.properties.items()}
        else:
            properties = None
        return self.__class__(
            templates=self.templates[subset],
            unit_ids=self.unit_ids[subset],
            spike_counts=self.spike_counts[subset],
            spike_counts_by_channel=(
                self.spike_counts_by_channel[subset]
                if self.spike_counts_by_channel is not None
                else None
            ),
            raw_std_dev=(
                self.raw_std_dev[subset] if self.raw_std_dev is not None else None
            ),
            registered_geom=self.registered_geom,
            trough_offset_samples=self.trough_offset_samples,
            properties=properties,
        )

    def coarsen(self):
        """Weighted average all templates that share a unit id."""
        # update templates
        unit_ids_unique, flat_ids = np.unique(self.unit_ids, return_inverse=True)
        templates = weighted_average(flat_ids, self.templates, self.spike_counts)

        # collect spike counts
        spike_counts = np.zeros(len(templates))
        np.add.at(spike_counts, flat_ids, self.spike_counts)

        return replace(
            self,
            templates=templates,
            unit_ids=unit_ids_unique,
            spike_counts=spike_counts,
            properties=None,
        )

    def unit_mask(self, unit_id):
        return np.isin(self.unit_ids, unit_id)

    def unit_templates(self, unit_id):
        return self.templates[self.unit_mask(unit_id)]

    @classmethod
    def from_config(
        cls,
        recording,
        sorting,
        template_cfg,
        waveform_cfg=default_waveform_cfg,
        save_folder: Path | None = None,
        overwrite=False,
        motion_est=None,
        save_npz_name: str | None = "template_data.npz",
        units_per_job=8,
        tsvd=None,
        computation_cfg=None,
    ):
        self, _ = cls.from_config_with_realigned_sorting(
            recording=recording,
            sorting=sorting,
            template_cfg=template_cfg,
            waveform_cfg=waveform_cfg,
            save_folder=save_folder,
            overwrite=overwrite,
            motion_est=motion_est,
            save_npz_name=save_npz_name,
            units_per_job=units_per_job,
            tsvd=tsvd,
            computation_cfg=computation_cfg,
        )

        gc.collect()
        torch.cuda.empty_cache()

        return self

    @classmethod
    def from_config_with_realigned_sorting(
        cls,
        recording,
        sorting,
        template_cfg,
        waveform_cfg=default_waveform_cfg,
        save_folder=None,
        overwrite=False,
        motion_est=None,
        save_npz_name: str | None = "template_data.npz",
        units_per_job=8,
        tsvd=None,
        computation_cfg=None,
    ) -> tuple["TemplateData", DARTsortSorting]:
        self, realigned_sorting = _from_config_with_realigned_sorting(
            cls,
            recording,
            sorting,
            template_cfg,
            waveform_cfg=waveform_cfg,
            save_folder=save_folder,
            overwrite=overwrite,
            motion_est=motion_est,
            save_npz_name=save_npz_name,
            units_per_job=units_per_job,
            tsvd=tsvd,
            computation_cfg=computation_cfg,
        )
        return self, realigned_sorting


def _from_config_with_realigned_sorting(
    cls,
    recording,
    sorting: DARTsortSorting,  # type: ignore
    template_cfg,
    waveform_cfg=default_waveform_cfg,
    save_folder=None,
    overwrite=False,
    motion_est=None,
    save_npz_name: str | None = "template_data.npz",
    units_per_job=8,
    tsvd=None,
    computation_cfg=None,
    show_progress=True,
) -> tuple[TemplateData, DARTsortSorting]:
    if computation_cfg is None:
        computation_cfg = job_util.get_global_computation_config()

    npz_path = None
    if save_folder is not None:
        save_folder = Path(save_folder)
        if not save_folder.exists():
            save_folder.mkdir()
        assert save_npz_name is not None
        npz_path = save_folder / save_npz_name
        if npz_path.exists() and not overwrite:
            return cls.from_npz(npz_path), sorting

    if sorting.extra_features and "time_shifts" in sorting.extra_features:
        logger.info("Sorting had time_shifts, applying before getting templates.")
        new_times_samples = sorting.times_samples + sorting.time_shifts
        ef = {k: v for k, v in sorting.extra_features.items() if k != "time_shifts"}
        sorting: DARTsortSorting = replace(
            sorting, times_samples=new_times_samples, extra_features=ef
        )

    if template_cfg.actual_algorithm() == "by_chunk":
        template_data, realigned_sorting = get_templates_by_chunk(
            sorting=sorting,
            recording=recording,
            tsvd=tsvd,
            motion_est=motion_est,
            waveform_cfg=waveform_cfg,
            computation_cfg=computation_cfg,
            template_cfg=template_cfg,
            show_progress=show_progress,
        )
        return template_data, realigned_sorting

    if sorting is None:
        raise ValueError(
            "TemplateData.from_config needs a sorting when its .npz file "
            "does not exist."
        )

    fs = recording.sampling_frequency
    trough_offset_samples = waveform_cfg.trough_offset_samples(fs)
    spike_length_samples = waveform_cfg.spike_length_samples(fs)
    realign_max_sample_shift = waveform_cfg.ms_to_samples(
        template_cfg.realign_shift_ms, sampling_frequency=fs
    )

    if template_cfg.denoising_method in (None, "none"):
        low_rank_denoising = False
    else:
        assert template_cfg.denoising_method == "exp_weighted"
        low_rank_denoising = True

    motion_aware = motion_est is not None and (
        template_cfg.registered_templates or template_cfg.superres_templates
    )
    localizations_dataset_name = template_cfg.localizations_dataset_name
    has_localizations = hasattr(sorting, localizations_dataset_name)
    if motion_aware and not has_localizations:
        raise ValueError(
            f"{_motion_error_prefix}"
            "sorting must contain localizations in the attribute "
            f"{localizations_dataset_name=}. Using load_simple_features"
            "=True in DARTsortSorting.from_peeling_hdf5() would put them "
            "there."
        )

    # load motion features if necessary
    geom = recording.get_channel_locations()
    motion_kw = {}
    rgeom = geom
    if template_cfg.registered_templates and motion_est is not None:
        rgeom = drift_util.registered_geometry(geom, motion_est=motion_est)
        motion_kw["registered_geom"] = rgeom
        motion_kw["pitch_shifts"] = drift_util.get_spike_pitch_shifts(
            geom=geom, sorting=sorting, motion_est=motion_est
        )

    # handle superresolved templates
    if template_cfg.superres_templates:
        superres_data = superres_sorting(
            sorting,
            geom,
            motion_est=motion_est,
            strategy=template_cfg.superres_strategy,
            superres_bin_size_um=template_cfg.superres_bin_size_um,
            min_spikes_per_bin=template_cfg.superres_bin_min_spikes,
        )
        group_ids = superres_data["group_ids"]
        sorting: DARTsortSorting = superres_data["sorting"]  # type: ignore
        properties = superres_data["properties"]
    else:
        group_ids = None
        properties = {}

    # main!
    results = get_templates(
        recording=recording,
        sorting=sorting,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        spikes_per_unit=template_cfg.spikes_per_unit,
        denoising_rank=template_cfg.denoising_rank,
        recompute_tsvd=template_cfg.recompute_tsvd,
        denoising_fit_radius=template_cfg.denoising_fit_radius,
        denoising_snr_threshold=template_cfg.exp_weight_snr_threshold,
        units_per_job=units_per_job,
        with_raw_std_dev=template_cfg.with_raw_std_dev,
        reducer=nanmean if template_cfg.reduction == "mean" else fast_nanmedian,
        low_rank_denoising=low_rank_denoising,
        denoising_tsvd=tsvd,
        realign_peaks=template_cfg.realign_peaks,
        realign_max_sample_shift=realign_max_sample_shift,
        device=computation_cfg.actual_device(),
        n_jobs=computation_cfg.actual_n_jobs(),
        show_progress=show_progress,
        **motion_kw,
    )
    if template_cfg.superres_templates:
        assert group_ids is not None
        unit_ids = group_ids[results["unit_ids"]]  # type: ignore
    else:
        unit_ids = results["unit_ids"]
    obj = cls(
        results["templates"],
        unit_ids=unit_ids,
        spike_counts=results["spike_counts"],
        spike_counts_by_channel=results["spike_counts_by_channel"],
        raw_std_dev=results["raw_std_devs"],
        registered_geom=rgeom,
        trough_offset_samples=trough_offset_samples,
        properties=properties,
    )
    if save_folder is not None:
        obj.to_npz(npz_path)

    return obj, sorting


def get_chunked_templates(
    recording,
    template_cfg,
    global_sorting=None,
    chunk_sortings=None,
    chunk_time_ranges_s=None,
    global_realign_peaks=True,
    save_folder=None,
    overwrite=False,
    motion_est=None,
    save_npz_name="chunked_template_data.npz",
    localizations_dataset_name="point_source_localizations",
    units_per_job=8,
    tsvd=None,
    random_seed=0,
    computation_cfg=None,
    waveform_cfg=default_waveform_cfg,
):
    """Save the effort of recomputing several TPCAs"""
    rg = np.random.default_rng(random_seed)

    if computation_cfg is None:
        computation_cfg = job_util.get_global_computation_config()

    if global_sorting is None:
        assert chunk_sortings is not None
        global_realign_peaks = False
        global_sorting = data_util.combine_sortings(chunk_sortings, dodge=True)

    done = False
    if save_folder is not None:
        save_folder = Path(save_folder)
        if (save_folder / save_npz_name).exists():
            done = not overwrite

    if not done and template_cfg.realign_peaks and global_realign_peaks:
        # realign globally, before chunking
        global_sorting = get_realigned_sorting(
            recording,
            global_sorting,
            trough_offset_samples=waveform_cfg.trough_offset_samples(
                recording.sampling_frequency
            ),
            spike_length_samples=waveform_cfg.spike_length_samples(
                recording.sampling_frequency
            ),
            spikes_per_unit=template_cfg.spikes_per_unit,
            realign_max_sample_shift=template_cfg.realign_max_sample_shift,
            n_jobs=computation_cfg.actual_n_jobs(),
            random_seed=rg,
        )
        template_cfg = replace(template_cfg, realign_peaks=False)

    # now, break each unit into pieces matching the chunks
    if chunk_sortings is None:
        assert global_sorting is not None
        chunk_time_ranges_s, chunk_sortings = data_util.time_chunk_sortings(
            global_sorting, recording=recording, chunk_time_ranges_s=chunk_time_ranges_s
        )

    # combine into a single sorting, so that we can use the template computer's
    # parallelism in one big loop
    (
        label_to_sorting_index,
        label_to_original_label,
        combined_sorting,
    ) = data_util.combine_sortings(chunk_sortings, dodge=True)

    # compute templates in combined label space
    full_template_data = TemplateData.from_config(
        recording,
        combined_sorting,
        template_cfg,
        save_folder=save_folder,
        overwrite=overwrite,
        motion_est=motion_est,
        save_npz_name=save_npz_name,
        units_per_job=units_per_job,
        computation_cfg=computation_cfg,
        waveform_cfg=waveform_cfg,
        tsvd=tsvd,
    )

    # break it up back into chunks
    chunk_template_data = []
    for i in range(len(chunk_sortings)):
        chunk_mask = np.flatnonzero(
            label_to_sorting_index[full_template_data.unit_ids] == i
        )
        # chunk_unit_ids = np.flatnonzero(label_to_sorting_index == i)
        # chunk_mask = np.flatnonzero(np.isin(full_template_data.unit_ids, chunk_unit_ids))
        orig_unit_ids = label_to_original_label[full_template_data.unit_ids[chunk_mask]]
        chunk_template_data.append(
            replace(
                full_template_data,
                templates=full_template_data.templates[chunk_mask],
                unit_ids=orig_unit_ids,
                spike_counts=full_template_data.spike_counts[chunk_mask],
            )
        )

    return chunk_template_data


def get_templates_by_chunk(
    *,
    sorting: DARTsortSorting,
    recording,
    tsvd,
    motion_est,
    waveform_cfg,
    computation_cfg,
    template_cfg,
    show_progress: bool,
    block_size=384,
    hard_block_size=512,
):
    # TODO: smarter chunk size.
    assert sorting.labels is not None
    unit_ids = np.unique(sorting.labels)
    unit_ids = unit_ids[unit_ids >= 0]
    n_units = unit_ids.shape[0]
    n_blocks = max(1, n_units // block_size)
    n_blocks += n_units / n_blocks > hard_block_size
    assert n_units / n_blocks <= hard_block_size
    units_per_block = int(np.ceil(n_units / n_blocks).astype(int).item())

    labels_tmp = np.full_like(sorting.labels, -1)
    realigned_spike_times = sorting.times_samples.copy()
    template_datas = []
    for block_start in trange(
        0, n_units, units_per_block, desc=f"Template blocks [{n_units}:{n_blocks}]"
    ):
        block_end = min(block_start + units_per_block, n_units)

        ids_in_block = unit_ids[block_start:block_end]
        spikes_in_block = np.flatnonzero(np.isin(sorting.labels, ids_in_block))
        labels_tmp[:] = -1
        labels_tmp[spikes_in_block] = sorting.labels[spikes_in_block]

        block_sorting = replace(sorting, labels=labels_tmp)
        block_realigned_sorting, block_realigned_templates = (
            _get_templates_by_chunk_block(
                sorting=block_sorting,
                recording=recording,
                tsvd=tsvd,
                motion_est=motion_est,
                waveform_cfg=waveform_cfg,
                computation_cfg=computation_cfg,
                template_cfg=template_cfg,
                show_progress=show_progress,
            )
        )
        template_datas.append(block_realigned_templates)
        assert np.array_equal(block_realigned_templates.unit_ids, ids_in_block)
        assert (
            block_realigned_sorting.times_samples.shape == realigned_spike_times.shape
        )
        if block_realigned_sorting.extra_features is not None:
            if "mask_indices" in block_realigned_sorting.extra_features:
                ix = block_realigned_sorting.extra_features["mask_indices"]
                realigned_spike_times[ix] = block_realigned_sorting.times_samples[ix]
        else:
            assert block_realigned_sorting.times_samples.shape == realigned_spike_times.shape

    realigned_sorting = replace(sorting, times_samples=realigned_spike_times)
    template_data = stack_template_datas(template_datas)
    assert np.array_equal(template_data.unit_ids, unit_ids)
    assert template_data.templates.shape[0] == unit_ids.shape[0]
    return template_data, realigned_sorting


def _get_templates_by_chunk_block(
    sorting: DARTsortSorting,
    recording,
    tsvd,
    motion_est,
    waveform_cfg,
    computation_cfg,
    template_cfg,
    show_progress: bool,
):
    from ..peel.running_template import RunningTemplates

    peeler = RunningTemplates.from_config(
        sorting=sorting,
        recording=recording,
        tsvd=tsvd,
        motion_est=motion_est,
        waveform_cfg=waveform_cfg,
        template_cfg=template_cfg,
        computation_cfg=computation_cfg,
        show_progress=show_progress,
    )
    template_data = peeler.compute_template_data(
        show_progress=show_progress, computation_cfg=computation_cfg
    )
    realigned_sorting = apply_time_shifts(
        sorting,
        template_data=template_data,
        trough_offset_samples=peeler.trough_offset_samples,
        spike_length_samples=peeler.spike_length_samples,
        recording_length_samples=recording.get_total_samples(),
    )
    assert getrefcount(peeler) == 2, (
        f"Leaking the template peeler {getrefcount(peeler)=}."
    )
    del peeler
    gc.collect()
    torch.cuda.empty_cache()

    return realigned_sorting, template_data


def stack_template_datas(template_datas):
    if len(template_datas) == 1:
        return template_datas[0]
    assert len(template_datas) > 0

    if template_datas[0].spike_counts_by_channel is not None:
        spike_counts_by_channel = np.concatenate(
            [td.spike_counts_by_channel for td in template_datas]
        )
    else:
        spike_counts_by_channel = None

    if template_datas[0].raw_std_dev is not None:
        raw_std_dev = np.concatenate([td.raw_std_dev for td in template_datas])
    else:
        raw_std_dev = None

    if template_datas[0].properties is not None:
        properties = {
            k: np.concatenate([td.properties[k] for td in template_datas])
            for k in template_datas[0].properties
        }
    else:
        properties = None

    return TemplateData(
        templates=np.concatenate([td.templates for td in template_datas]),
        unit_ids=np.concatenate([td.unit_ids for td in template_datas]),
        spike_counts=np.concatenate([td.spike_counts for td in template_datas]),
        spike_counts_by_channel=spike_counts_by_channel,
        raw_std_dev=raw_std_dev,
        registered_geom=template_datas[0].registered_geom,
        trough_offset_samples=template_datas[0].trough_offset_samples,
        properties=properties,
    )
