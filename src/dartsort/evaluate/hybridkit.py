import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import spikeinterface.core as sc
import spikeinterface.preprocessing as spre
from spikeinterface.core.baserecording import BaseRecording

from ..templates import TemplateData
from ..util.data_util import DARTsortSorting, ensure_path
from ..util.internal_config import raw_template_cfg
from ..util.job_util import ensure_computation_config
from ..util.motion import MotionInfo, resample_motion
from ..util.py_util import databag
from .analysis import DARTsortAnalysis
from .sim_template_tools import TemplateLibrarySimulator, get_template_simulator
from .simlib import InjectSpikesPreprocessor


@databag
class HybridDataset:
    gt_sorting: DARTsortSorting
    gt_templates: TemplateData
    si_templates: sc.Templates
    gt_unit_info: pd.DataFrame
    recording: sc.BaseRecording
    motion: MotionInfo
    metadata: dict[str, Any]

    def gt_analysis(
        self,
        trough_offset: int = 42,
        spike_length: int = 121,
        override_recording: sc.BaseRecording | None = None,
        override_motion: MotionInfo | None = None,
        recompute_templates: bool = False,
        template_cfg=raw_template_cfg,
    ) -> DARTsortAnalysis:
        if override_motion is None:
            if recompute_templates:
                template_data = None
            elif spike_length != self.gt_templates.spike_length_samples:
                i0 = self.gt_templates.trough_offset_samples - trough_offset
                assert i0 >= 0
                template_data = replace(
                    self.gt_templates,
                    templates=self.gt_templates.templates[:, i0 : i0 + spike_length],
                    trough_offset_samples=trough_offset,
                )
            else:
                assert trough_offset == self.gt_templates.spike_length_samples
                template_data = self.gt_templates
            motion = self.motion
        else:
            template_data = None
            motion = override_motion
        rec = self.recording if override_recording is None else override_recording
        return DARTsortAnalysis.from_sorting(
            recording=rec,
            sorting=self.gt_sorting,
            template_data=template_data,
            motion=motion,
            name="GT",
            template_cfg=template_cfg,
        )


def load_hybrid_recording(folder: str | Path) -> HybridDataset | None:
    folder = ensure_path(folder)
    if not folder.exists():
        return None
    recording_dir = folder / "recording"
    templates_npz = folder / "templates.npz"
    sorting_h5 = folder / "dartsort_sorting.h5"
    unit_info_csv = folder / "unit_information.csv"
    meta_json = folder / "metadata.json"
    if not meta_json.exists():
        return None

    recording = sc.read_binary_folder(recording_dir)
    templates = TemplateData.from_npz(templates_npz)
    sorting = DARTsortSorting.from_peeling_hdf5(sorting_h5)
    motion = MotionInfo.try_load(folder)
    assert motion is not None
    unit_info_df = pd.read_csv(unit_info_csv)
    with meta_json.open("r") as jsonf:
        metadata = json.load(jsonf)
    si_templates = sc.Templates.from_zarr(folder / "si_templates.zarr")

    return HybridDataset(
        recording=recording,
        gt_templates=templates,
        si_templates=si_templates,
        gt_sorting=sorting,
        motion=motion,
        gt_unit_info=unit_info_df,
        metadata=metadata,
    )


def make_hybrid_recording(
    *,
    folder: str | Path,
    overwrite: bool = False,
    target_recording: sc.BaseRecording,
    injected_sorting: sc.BaseSorting,
    motion: MotionInfo,
    templates: sc.Templates,
    filter: bool = True,
    trim_t_start_rel: float = 0.0,
    trim_t_len: float | None = None,
    remove_bad_channels: bool = True,
    template_peak_range: tuple[float, float] | None = (25.0, 100.0),
    template_x_align: Literal["center", "amplitude"] = "center",
    reset_times: bool = True,
    target_sampling_frequency: float = 30_000.0,
    amplitude_jitter: float = 0.05,
    amp_jitter_family: Literal["gamma", "uniform", "normal"] = "normal",
    temporal_jitter: int = 8,
    save_chunk_len_s: float = 0.5,
    template_simulator_kwargs: dict | None = None,
    metadata: dict[str, Any] | None = None,
    computation_cfg=None,
    pool_engine="thread",
    extra_unit_information: pd.DataFrame | None = None,
    rg: np.random.Generator | int = 0,
) -> HybridDataset:
    assert templates.is_in_uV
    assert templates.num_units == injected_sorting.get_num_units()
    assert injected_sorting._recording is None
    assert target_recording.get_num_segments() == 1
    assert injected_sorting.get_num_segments() == 1
    assert injected_sorting.get_last_spike_frame() < target_recording.get_num_frames()
    assert abs(templates.sampling_frequency - target_sampling_frequency) < 10

    if folder is not None and not overwrite:
        if (ret := load_hybrid_recording(folder)) is not None:
            return ret
    computation_cfg = ensure_computation_config(computation_cfg)

    if trim_t_len:
        frame_start = int(trim_t_start_rel * target_recording.sampling_frequency)
        frame_end = frame_start + int(trim_t_len * target_recording.sampling_frequency)
        target_recording = target_recording.frame_slice(frame_start, frame_end)
        injected_sorting = injected_sorting.frame_slice(frame_start, frame_end)

    if reset_times:
        # for motion:
        motion_query_times = np.arange(
            target_recording.get_start_time() + save_chunk_len_s / 2,
            target_recording.get_end_time(),
            save_chunk_len_s,
        )

        target_recording.reset_times()
        if abs(target_recording.sampling_frequency - target_sampling_frequency) > 10.0:
            raise ValueError("Sampling...")
        target_recording._sampling_frequency = target_sampling_frequency
        target_recording.reset_times()

        injected_sorting._sampling_frequency = target_sampling_frequency

        motion = resample_motion(
            motion=motion,
            query_time_bin_centers=motion_query_times,
            new_motion_time_bin_centers=np.arange(
                save_chunk_len_s / 2,
                target_recording.get_total_duration(),
                save_chunk_len_s,
            ),
        )

    if template_peak_range is not None:
        templates = rescale_templates(templates, template_peak_range, rg)

    if filter:
        target_recording = spre.highpass_filter(target_recording, dtype="float32")
        if "inter_sample_shift" in target_recording.get_property_keys():
            target_recording = spre.phase_shift(target_recording)
    if remove_bad_channels:
        assert filter
        bcids = spre.detect_bad_channels(target_recording, seed=0)
        target_recording = target_recording.remove_channels(bcids[0])
    target_recording = spre.scale_to_uV(target_recording)  # ty: ignore[invalid-argument-type]

    motion = update_motion_geom(motion, target_recording)

    template_simulator_kwargs = dict(template_simulator_kwargs or {})
    template_simulator_kwargs.setdefault("trough_offset_samples", templates.nbefore)
    template_simulator_kwargs.setdefault("x_align", template_x_align)
    template_simulator_kwargs.setdefault("common_reference", False)
    if templates.probe is not None:
        template_simulator_kwargs.setdefault(
            "source_geom", templates.get_channel_locations()
        )
    template_simulator = get_template_simulator(
        n_units=templates.num_units,
        templates_kind="library",
        template_library=templates.templates_array,
        geom=target_recording.get_channel_locations(),
        sampling_frequency=target_sampling_frequency,
        temporal_jitter=temporal_jitter,
        random_seed=rg,
        **template_simulator_kwargs,
    )
    assert isinstance(template_simulator, TemplateLibrarySimulator)
    hybrid_recording = InjectSpikesPreprocessor(
        recording=target_recording,
        sorting=injected_sorting,
        motion=motion,
        template_simulator=template_simulator,
        amplitude_jitter=amplitude_jitter,
        amp_jitter_family=amp_jitter_family,
        temporal_jitter=temporal_jitter,
        extract_radius=0.0,
        random_seed=rg,
        compute_collision_waveforms=False,
    )

    hybrid_recording.save_simulation(
        folder,
        overwrite=overwrite,
        n_jobs=computation_cfg.actual_n_jobs(),
        featurization_cfg=None,
        computation_cfg=computation_cfg,
        chunk_len_s=save_chunk_len_s,
        save_injected_waveforms=False,
        save_noise_waveforms=False,
        save_collision_waveforms=False,
        save_collisioncleaned_waveforms=False,
        save_collidedness=False,
        extra_unit_information=extra_unit_information,
        n_residual_snips=0,
        pool_engine=pool_engine,
    )
    metadata = dict(metadata or {})
    metadata.update(
        filter=filter,
        trim_t_start_rel=trim_t_start_rel,
        trim_t_len=trim_t_len,
        remove_bad_channels=remove_bad_channels,
        template_peak_range=template_peak_range,
        template_x_align=template_simulator_kwargs["x_align"],
        template_interp_method=template_simulator.interp_method,
        template_grid_avg_offsets=template_simulator.grid_avg_offsets.tolist(),
        reset_times=reset_times,
        target_sampling_frequency=target_sampling_frequency,
        amplitude_jitter=amplitude_jitter,
        amp_jitter_family=amp_jitter_family,
        temporal_jitter=temporal_jitter,
    )
    with (ensure_path(folder) / "metadata.json").open("w") as jsonf:
        json.dump(metadata, jsonf)
    templates.to_zarr(ensure_path(folder) / "si_templates.zarr")

    res = load_hybrid_recording(folder)
    assert res is not None
    return res


def rescale_templates(
    templates: sc.Templates,
    template_peak_range: tuple[float, float] = (25.0, 100.0),
    rg: np.random.Generator | int = 0,
):
    t = deepcopy(templates)
    del templates
    rg = np.random.default_rng(rg)
    for i in range(t.num_units):
        pk = np.abs(t.templates_array[i]).max()
        assert pk > 0
        targ = rg.uniform(*template_peak_range)
        t.templates_array[i] *= targ / pk
    return t


def update_motion_geom(motion: MotionInfo, rec: BaseRecording):
    new_geom = rec.get_channel_locations()
    if np.array_equal(motion.geom, new_geom):
        return motion
    return MotionInfo.from_motion_est(
        geom=new_geom,
        dredge_motion_est=motion.dredge_motion_est,
        si_motion=motion.si_motion,
    )
