from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..localize.localize_util import localize_waveforms
from ..util import data_util, drift_util, job_util
from ..util.internal_config import default_waveform_config

from .get_templates import get_templates
from .superres_util import superres_sorting
from .template_util import get_realigned_sorting, weighted_average

_motion_error_prefix = (
    "If template_config has registered_templates==True "
    "or superres_templates=True, then "
)
_aware_error = "motion_est must be passed to TemplateData.from_config()"


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
    spike_length_samples: int = 121

    # always set if initialized from a sorting which has one
    parent_sorting_hdf5_path: str | None = None

    def main_channels(self):
        amp_vecs = np.nan_to_num(np.ptp(self.templates, axis=1), nan=-np.inf)
        if self.spike_counts_by_channel is not None:
            amp_vecs *= np.sqrt(self.spike_counts_by_channel)
        return amp_vecs.argmax(axis=1)

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
            parent_sorting_hdf5_path = data.pop("parent_sorting_hdf5_path", None)
            if parent_sorting_hdf5_path is not None:
                parent_sorting_hdf5_path = parent_sorting_hdf5_path.item()
            data["parent_sorting_hdf5_path"] = parent_sorting_hdf5_path
            return cls(**data)

    def to_npz(self, npz_path):
        to_save = dict(
            templates=self.templates,
            unit_ids=self.unit_ids,
            spike_counts=self.spike_counts,
        )
        if self.registered_geom is not None:
            to_save["registered_geom"] = self.registered_geom
        if self.spike_counts_by_channel is not None:
            to_save["spike_counts_by_channel"] = self.spike_counts_by_channel
        if self.raw_std_dev is not None:
            to_save["raw_std_dev"] = self.raw_std_dev
        if self.parent_sorting_hdf5_path is not None:
            to_save["parent_sorting_hdf5_path"] = np.array(
                [self.parent_sorting_hdf5_path]
            )
        if not npz_path.parent.exists():
            npz_path.parent.mkdir()
        np.savez(npz_path, **to_save)

    def __getitem__(self, subset):
        if not np.array_equal(self.unit_ids, np.arange(len(self.unit_ids))):
            subset_ixs = np.searchsorted(self.unit_ids, subset, side="right") - 1
            matched = self.unit_ids[subset_ixs] == subset
            assert matched.all()
            subset = subset_ixs
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
            spike_length_samples=self.spike_length_samples,
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
        template_config,
        waveform_config=default_waveform_config,
        save_folder=None,
        overwrite=False,
        motion_est=None,
        save_npz_name="template_data.npz",
        localizations_dataset_name="point_source_localizations",
        units_per_job=8,
        tsvd=None,
        computation_config=None,
    ):
        self, _ = cls.from_config_with_realigned_sorting(
            recording=recording,
            sorting=sorting,
            template_config=template_config,
            waveform_config=waveform_config,
            save_folder=save_folder,
            overwrite=overwrite,
            motion_est=motion_est,
            save_npz_name=save_npz_name,
            localizations_dataset_name=localizations_dataset_name,
            units_per_job=units_per_job,
            tsvd=tsvd,
            computation_config=computation_config,
        )
        return self

    @classmethod
    def from_config_with_realigned_sorting(
        cls,
        recording,
        sorting,
        template_config,
        waveform_config=default_waveform_config,
        save_folder=None,
        overwrite=False,
        motion_est=None,
        save_npz_name="template_data.npz",
        localizations_dataset_name="point_source_localizations",
        units_per_job=8,
        tsvd=None,
        computation_config=None,
    ):
        if computation_config is None:
            computation_config = job_util.get_global_computation_config()

        npz_path = None
        if save_folder is not None:
            save_folder = Path(save_folder)
            if not save_folder.exists():
                save_folder.mkdir()
            npz_path = save_folder / save_npz_name
            if npz_path.exists() and not overwrite:
                return cls.from_npz(npz_path), None

        if sorting is None:
            raise ValueError(
                "TemplateData.from_config needs sorting!=None when its .npz file does not exist."
            )

        parent_sorting_hdf5_path = sorting.parent_h5_path

        fs = recording.sampling_frequency
        trough_offset_samples = waveform_config.trough_offset_samples(fs)
        spike_length_samples = waveform_config.spike_length_samples(fs)
        realign_max_sample_shift = int(template_config.realign_shift_ms * (fs / 1000))

        motion_aware = (
            (template_config.registered_templates or template_config.superres_templates)
            and motion_est is not None
        )
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
        spike_depths_um = None
        if motion_aware and has_localizations:
            # load spike depths
            # TODO: relying on this index feels wrong
            spike_depths_um = sorting.extra_features[localizations_dataset_name][:, 2]
            spike_x_um = sorting.extra_features[localizations_dataset_name][:, 0]

        kwargs = dict(
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            spikes_per_unit=template_config.spikes_per_unit,
            # realign handled in advance below, not needed in kwargs
            # realign_peaks=False,
            realign_max_sample_shift=realign_max_sample_shift,
            denoising_rank=template_config.denoising_rank,
            denoising_fit_radius=template_config.denoising_fit_radius,
            denoising_snr_threshold=template_config.denoising_snr_threshold,
            device=computation_config.actual_device(),
            units_per_job=units_per_job,
        )
        rgeom = geom
        if template_config.registered_templates and motion_est is not None:
            rgeom = drift_util.registered_geometry(
                geom, motion_est=motion_est
            )
            kwargs["registered_geom"] = rgeom
            assert spike_depths_um is not None
            kwargs["pitch_shifts"] = drift_util.get_spike_pitch_shifts(
                spike_depths_um,
                geom,
                times_s=sorting.times_seconds,
                motion_est=motion_est,
            )

        # realign before superres
        if template_config.realign_peaks:
            sorting = get_realigned_sorting(
                recording,
                sorting,
                **kwargs,
                realign_peaks=True,
                low_rank_denoising=False,
                n_jobs=computation_config.actual_n_jobs(),
            )
        kwargs["low_rank_denoising"] = template_config.low_rank_denoising
        kwargs["realign_peaks"] = False
        kwargs["denoising_tsvd"] = tsvd

        # handle superresolved templates
        if template_config.superres_templates:
            unit_ids, sorting = superres_sorting(
                sorting,
                sorting.times_seconds,
                spike_depths_um,
                geom,
                motion_est=motion_est,
                strategy=template_config.superres_strategy,
                superres_bin_size_um=template_config.superres_bin_size_um,
                min_spikes_per_bin=template_config.superres_bin_min_spikes,
                spike_x_um=spike_x_um,
                adaptive_bin_size=template_config.adaptive_bin_size,
            )

        # main!
        results = get_templates(
            recording, sorting, n_jobs=computation_config.actual_n_jobs(), **kwargs
        )

        # handle registered templates
        if template_config.registered_templates and motion_est is not None:
            obj = cls(
                results["templates"],
                unit_ids=results["unit_ids"],
                spike_counts=results["spike_counts"],
                spike_counts_by_channel=results["spike_counts_by_channel"],
                registered_geom=rgeom,
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
                parent_sorting_hdf5_path=parent_sorting_hdf5_path,
            )
        else:
            obj = cls(
                results["templates"],
                unit_ids=results["unit_ids"],
                spike_counts=results["spike_counts"],
                spike_counts_by_channel=results["spike_counts_by_channel"],
                registered_geom=rgeom,
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
                parent_sorting_hdf5_path=parent_sorting_hdf5_path,
            )
        if save_folder is not None:
            obj.to_npz(npz_path)

        return obj, sorting


def get_chunked_templates(
    recording,
    template_config,
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
    computation_config=None,
    waveform_config=default_waveform_config,
):
    """Save the effort of recomputing several TPCAs"""
    rg = np.random.default_rng(random_seed)

    if computation_config is None:
        computation_config = job_util.get_global_computation_config()

    if global_sorting is None:
        assert chunk_sortings is not None
        global_realign_peaks = False
        global_sorting = data_util.combine_sortings(chunk_sortings, dodge=True)

    done = False
    if save_folder is not None:
        save_folder = Path(save_folder)
        if (save_folder / save_npz_name).exists():
            done = not overwrite

    if not done and template_config.realign_peaks and global_realign_peaks:
        # realign globally, before chunking
        global_sorting = get_realigned_sorting(
            recording,
            global_sorting,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            spikes_per_unit=template_config.spikes_per_unit,
            realign_max_sample_shift=template_config.realign_max_sample_shift,
            n_jobs=computation_config.actual_n_jobs(),
            random_seed=rg,
        )
        template_config = replace(template_config, realign_peaks=False)

    # now, break each unit into pieces matching the chunks
    if chunk_sortings is None:
        assert global_sorting is not None
        chunk_time_ranges_s, chunk_sortings = data_util.time_chunk_sortings(
            global_sorting, recording=recording, chunk_time_ranges_s=chunk_time_ranges_s
        )

    # combine into a single sorting, so that we can use the template computer's
    # parallelism in one big loop
    label_to_sorting_index, label_to_original_label, combined_sorting = (
        data_util.combine_sortings(chunk_sortings, dodge=True)
    )

    # compute templates in combined label space
    full_template_data = TemplateData.from_config(
        recording,
        combined_sorting,
        template_config,
        save_folder=save_folder,
        overwrite=overwrite,
        motion_est=motion_est,
        save_npz_name=save_npz_name,
        localizations_dataset_name=localizations_dataset_name,
        units_per_job=units_per_job,
        computation_config=computation_config,
        waveform_config=waveform_config,
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
