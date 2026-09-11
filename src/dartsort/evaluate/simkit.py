import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dredge import motion_util
from scipy.signal import sawtooth
from spikeinterface.core import BaseRecording, read_binary_folder

from ..templates import TemplateData
from ..util.data_util import DARTsortSorting, ensure_path
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.py_util import panic
from .noise_recording_tools import get_background_recording
from .sim_template_tools import TemplateSimulator, get_template_simulator
from .simlib import (
    InjectSpikesPreprocessor,
    default_sim_featurization_cfg,
    default_temporal_kernel_npy,
    simulate_sorting,
    simulate_twostate_switching,
)

logger = get_logger(__name__)


def generate_simulation(
    folder: str | Path | None,
    noise_recording_folder: str | Path | None,
    *,
    n_units: int,
    duration_seconds: float,
    # probe parameters
    geom=None,
    probe_kwargs=None,
    # sorting parameters
    firing_kind="uniform",
    min_fr_hz=1.0,
    max_fr_hz=20.0,
    state_affinity: float = 0.3,
    down_max_fr_hz: float = 5.0,
    up_min_fr_hz: float = 8.0,
    # noise args
    noise_kind="stationary_factorized_rbf",
    noise_spatial_kernel_bandwidth=15.0,
    noise_temporal_kernel: np.ndarray | str | Path = default_temporal_kernel_npy,
    noise_fft_t=121,
    white_noise_scale=1.0,
    # template args
    templates_kind="3exp",
    template_library=None,
    template_simulator_kwargs: dict | None = None,
    template_simulator: TemplateSimulator | None = None,
    # general parameters
    drift_type="triangle",
    drift_speed=0.0,
    drift_period=30.0,
    temporal_jitter=16,
    temporal_jitter_family="uniform",
    amplitude_jitter=0.05,
    amp_jitter_family="uniform",
    common_reference=True,
    sampling_frequency=30000.0,
    refractory_ms=1.0,
    globally_refractory=False,
    extract_radius=150.0,
    recording_dtype="float16",
    features_dtype="float32",
    featurization_cfg=default_sim_featurization_cfg,
    computation_cfg=None,
    save_injected_waveforms=False,
    save_noise_waveforms=False,
    save_collision_waveforms=False,
    save_collisioncleaned_waveforms=True,
    save_collidedness=False,
    n_residual_snips=4096,
    # control
    max_drift_per_chunk=0.5,
    max_chunk_len_s=1.0,
    random_seed=0,
    noise_in_memory=False,
    overwrite=False,
    no_save=False,
    just_noise=False,
):
    computation_cfg = ensure_computation_config(computation_cfg)
    if folder is not None and not (overwrite or just_noise or no_save):
        try:
            return load_simulation(folder)
        except Exception:  # noqa: BLE001, S110
            pass

    if noise_recording_folder is not None:
        noise_recording_folder = ensure_path(noise_recording_folder)
    else:
        assert noise_in_memory
    duration_samples = int(duration_seconds * sampling_frequency)
    with warnings.catch_warnings(record=True) as ws:
        noise_recording = get_background_recording(
            noise_recording_folder,
            duration_samples=duration_samples,
            geom=geom,
            probe_kwargs=probe_kwargs,
            noise_kind=noise_kind,
            noise_spatial_kernel_bandwidth=noise_spatial_kernel_bandwidth,
            noise_temporal_kernel=noise_temporal_kernel,
            random_seed=random_seed,
            dtype=recording_dtype,
            noise_fft_t=noise_fft_t,
            white_noise_scale=white_noise_scale,
            sampling_frequency=sampling_frequency,
            n_jobs=computation_cfg.actual_n_jobs(),
            in_memory=noise_in_memory,
            overwrite=overwrite,
        )
        for w in ws:
            msg = str(w.message)
            if msg.startswith("The extractor is not serializable "):
                continue
            if msg.startswith("auto_cast_uint"):
                continue
            raise ValueError(w)
    assert isinstance(noise_recording, BaseRecording)
    assert noise_recording.dtype == np.dtype(recording_dtype)
    assert noise_recording.sampling_frequency == sampling_frequency
    assert noise_recording.get_num_frames() == duration_samples

    if just_noise:
        return

    if folder is not None:
        folder = ensure_path(folder)
    else:
        assert no_save

    if template_simulator is None:
        template_simulator = get_template_simulator(
            n_units=n_units,
            templates_kind=templates_kind,
            template_library=template_library,
            geom=noise_recording.get_channel_locations(),
            sampling_frequency=sampling_frequency,
            common_reference=common_reference,
            random_seed=random_seed,
            temporal_jitter=temporal_jitter,
            **(template_simulator_kwargs or {}),
        )
    sim_recording = make_simulated_recording(
        noise_recording,
        firing_kind=firing_kind,
        min_fr_hz=min_fr_hz,
        max_fr_hz=max_fr_hz,
        state_affinity=state_affinity,
        down_max_fr_hz=down_max_fr_hz,
        up_min_fr_hz=up_min_fr_hz,
        template_simulator=template_simulator,
        drift_type=drift_type,
        drift_speed=drift_speed,
        drift_period=drift_period,
        amplitude_jitter=amplitude_jitter,
        temporal_jitter_family=temporal_jitter_family,
        temporal_jitter=temporal_jitter,
        random_seed=random_seed,
        refractory_ms=refractory_ms,
        globally_refractory=globally_refractory,
        amp_jitter_family=amp_jitter_family,
        extract_radius=extract_radius,
        features_dtype=features_dtype,
        compute_collision_waveforms=save_collision_waveforms or save_collidedness,
    )
    if no_save:
        return sim_recording, template_simulator

    if drift_speed is None:
        chunk_len_s = max_chunk_len_s
    else:
        drift_per_chunk = drift_speed * max_chunk_len_s
        chunk_len_s = min(
            max_chunk_len_s, max_drift_per_chunk / max(drift_per_chunk, 1e-10)
        )
    assert chunk_len_s

    sim_recording.save_simulation(
        folder,
        overwrite=overwrite,
        n_jobs=computation_cfg.actual_n_jobs(),
        featurization_cfg=featurization_cfg,
        computation_cfg=computation_cfg,
        chunk_len_s=chunk_len_s,
        save_injected_waveforms=save_injected_waveforms,
        save_noise_waveforms=save_noise_waveforms,
        save_collision_waveforms=save_collision_waveforms,
        save_collisioncleaned_waveforms=save_collisioncleaned_waveforms,
        save_collidedness=save_collidedness,
        n_residual_snips=n_residual_snips,
    )
    return load_simulation(folder)


def load_simulation(folder):
    folder = ensure_path(folder, strict=True)
    recording_dir = folder / "recording"
    templates_npz = folder / "templates.npz"
    sorting_h5 = folder / "dartsort_sorting.h5"
    unit_info_csv = folder / "unit_information.csv"

    recording = read_binary_folder(recording_dir)
    templates = TemplateData.from_npz(templates_npz)
    sorting = DARTsortSorting.from_peeling_hdf5(sorting_h5)
    motion = MotionInfo.try_load(folder)
    assert motion is not None
    unit_info_df = pd.read_csv(unit_info_csv)

    return dict(
        recording=recording,
        templates=templates,
        sorting=sorting,
        motion=motion,
        unit_info_df=unit_info_df,
    )


def make_simulated_recording(
    recording: BaseRecording,
    *,
    template_simulator: TemplateSimulator,
    firing_kind: Literal["uniform", "switching_two_state"] = "uniform",
    min_fr_hz: float = 1.0,
    max_fr_hz: float = 20.0,
    state_affinity: float = 0.3,
    down_max_fr_hz: float = 5.0,
    up_min_fr_hz: float = 8.0,
    refractory_ms: float = 1.0,
    globally_refractory: bool = False,
    drift_type: Literal["line", "triangle"] = "triangle",
    drift_speed: float = 0.0,
    drift_period: float = 30.0,
    amplitude_jitter: float = 0.0,
    amp_jitter_family: Literal["gamma", "uniform"] = "uniform",
    temporal_jitter: int = 1,
    temporal_jitter_family: Literal["uniform", "by_unit"] = "uniform",
    extract_radius: float = 100.0,
    random_seed: int = 0,
    features_dtype="float32",
    compute_collision_waveforms: bool = False,
) -> InjectSpikesPreprocessor:
    """Sample a spike train and a drift trajectory, and inject spikes into recording

    This is the sampling half of the simulation: it draws the units' firing rates
    (and their latent states, if any), the spike train, and the motion. The
    injection itself is handled by the InjectSpikesPreprocessor.
    """
    fs = recording.sampling_frequency
    n_samples = recording.get_num_frames()
    n_units = template_simulator.n_units
    extra = {}

    # firing rates, and the latent state which drives them
    rg = np.random.default_rng(random_seed)
    n_bins = int(np.ceil(n_samples / fs))
    if firing_kind == "uniform":
        firing_rates = rg.uniform(min_fr_hz, max_fr_hz, size=n_units)
    elif firing_kind == "switching_two_state":
        states, firing_rates = simulate_twostate_switching(
            rg=rg,
            n_bins=n_bins,
            state_affinity=state_affinity,
            n_units=n_units,
            min_fr=min_fr_hz,
            down_max_fr=down_max_fr_hz,
            up_min_fr=up_min_fr_hz,
            max_fr=max_fr_hz,
        )
        extra["states"] = states
    else:
        raise ValueError(f"Unknown {firing_kind=}")

    sorting = simulate_sorting(
        n_units,
        n_samples,
        firing_rates=firing_rates,
        rg=rg,
        nbefore=template_simulator.trough_offset_samples(),
        spike_length_samples=template_simulator.spike_length_samples(),
        sampling_frequency=fs,
        refractory_samples=int(refractory_ms * (fs / 1000.0)),
        globally_refractory=globally_refractory,
    )

    motion = simulate_motion(
        recording,
        drift_type=drift_type,
        drift_speed=drift_speed,
        drift_period=drift_period,
    )

    return InjectSpikesPreprocessor(
        recording,
        sorting=sorting,
        motion=motion,
        template_simulator=template_simulator,
        amplitude_jitter=amplitude_jitter,
        amp_jitter_family=amp_jitter_family,
        temporal_jitter=temporal_jitter,
        temporal_jitter_family=temporal_jitter_family,
        extract_radius=extract_radius,
        random_seed=random_seed,
        features_dtype=features_dtype,
        compute_collision_waveforms=compute_collision_waveforms,
        extra_data_to_save=extra,
    )


def simulate_drift(
    recording: BaseRecording,
    t_samples,
    *,
    drift_type: Literal["line", "triangle"],
    drift_speed: float,
    drift_period: float,
):
    """Displacement (um) of the whole probe at the times t_samples."""
    if not drift_speed:
        return np.zeros_like(t_samples)

    if drift_type == "line":
        t_center = recording.get_num_frames() / 2
        dt = (t_samples - t_center) / recording.sampling_frequency
        return dt * drift_speed

    if drift_type == "triangle":
        t_seconds = recording.sample_index_to_time(t_samples)
        phase = t_seconds * (2 * np.pi / drift_period)
        wave = sawtooth(phase, width=0.5)
        # -1 to 1 and back to -1, so divide by 4 to have 2*ptp=drift_speed*drift_period.
        return wave * (drift_speed * drift_period / 4.0)

    panic(drift_type)


def simulate_motion(
    recording: BaseRecording,
    *,
    drift_type: Literal["line", "triangle"],
    drift_speed: float,
    drift_period: float,
) -> MotionInfo:
    """Rigid MotionInfo describing the simulated drift, binned at 1s."""
    assert drift_type in ("line", "triangle")
    geom = recording.get_channel_locations()
    if not drift_speed:
        return MotionInfo.from_motion_est(geom=geom)

    segment = recording._recording_segments[0]
    duration_s = np.ceil(segment.get_end_time() - segment.get_start_time())
    t = np.arange(duration_s)
    time_bin_centers = t + 0.5 * np.diff(t).mean()
    tbc_samples = time_bin_centers * recording.sampling_frequency
    displacement = simulate_drift(
        recording,
        tbc_samples,
        drift_type=drift_type,
        drift_speed=drift_speed,
        drift_period=drift_period,
    )
    dredge_me = motion_util.get_motion_estimate(
        displacement=displacement, time_bin_centers_s=time_bin_centers
    )
    return MotionInfo.from_motion_est(geom=geom, dredge_motion_est=dredge_me)
