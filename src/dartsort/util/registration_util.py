from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from dredge.dredge_ap import register as dredge_register
from dredge.motion_util import (
    MotionEstimate,
    NonrigidMotionEstimate,
    RigidMotionEstimate,
    speed_limit_filter,
)
from spikeinterface.core import BaseRecording, Motion

from .data_util import DARTsortSorting
from .internal_config import (
    ComputationConfig,
    MotionEstimationConfig,
    default_motion_estimation_cfg,
)
from .job_util import ensure_computation_config
from .multiprocessing_util import handle_negative_jobs
from .py_util import databag


def dredge_estimate_motion(
    recording: BaseRecording,
    spikes: DARTsortSorting | Path,
    motion_cfg: MotionEstimationConfig = default_motion_estimation_cfg,
    localizations_dataset_name="point_source_localizations",
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    computation_cfg: ComputationConfig | None = None,
) -> MotionEstimate | None:
    if not motion_cfg.do_motion_estimation:
        return None
    computation_cfg = ensure_computation_config(computation_cfg)

    if (
        isinstance(spikes, DARTsortSorting)
        and spikes.get_mask_indices() is None
        and all(
            spikes.has_dataset_on_disk_not_loaded(k)
            for k in (
                localizations_dataset_name,
                amplitudes_dataset_name,
                "times_seconds",
            )
        )
    ):
        assert spikes.parent_h5_path is not None
        spikes = spikes.parent_h5_path

    geom = recording.get_channel_locations()
    spike_batches = MotionSpikeBatches(
        spikes=spikes,
        geom=geom,
        motion_cfg=motion_cfg,
        localizations_dataset_name=localizations_dataset_name,
        amplitudes_dataset_name=amplitudes_dataset_name,
    )
    pad = motion_cfg.probe_boundary_padding_um
    n_jobs = 0
    if computation_cfg.low_memory and len(spike_batches) >= 50:
        _, n_jobs = handle_negative_jobs(
            computation_cfg.actual_n_jobs(small=True, cpu=True)
        )

    # run registration
    with TemporaryDirectory(
        dir=computation_cfg.maybe_tmpdir_parent(), ignore_cleanup_errors=True
    ) as tmpdir:
        dredge_motion_est, _ = dredge_register(
            spike_batches=spike_batches,
            low_memory=computation_cfg.low_memory,
            tmpdir=Path(tmpdir),
            depth_limits_um=(geom[:, 1].min() - pad, geom[:, 1].max() + pad),
            time_limits_s=(recording.get_start_time(), recording.get_end_time()),
            n_jobs=n_jobs,
            rigid=motion_cfg.rigid,
            bin_um=motion_cfg.spatial_bin_length_um,
            bin_s=motion_cfg.temporal_bin_length_s,
            win_step_um=motion_cfg.window_step_um,
            weights_threshold_low=motion_cfg.weight_threshold,
            weights_threshold_high=motion_cfg.weight_threshold,
            win_scale_um=motion_cfg.window_scale_um,
            win_margin_um=motion_cfg.window_margin_um,
            max_disp_um=motion_cfg.max_disp_um,
            max_dt_s=motion_cfg.max_dt_s,
            mincorr=motion_cfg.correlation_threshold,
            gaussian_smoothing_sigma_um=motion_cfg.smoothing_um,
            gaussian_smoothing_sigma_s=motion_cfg.smoothing_s,
            batching_mode=motion_cfg.motion_batching_mode,
            chunk_len_s=motion_cfg.motion_chunk_len_s,
            coarse_bin_s=motion_cfg.motion_coarse_bin_s,
            coarse_max_dt_s=motion_cfg.motion_coarse_max_dt_s,
            fine_max_disp_um=motion_cfg.motion_fine_max_disp_um,
            device=computation_cfg.actual_device(),
        )
    dredge_motion_est = speed_limit_filter(
        dredge_motion_est,
        band_width=motion_cfg.median_neighborhood_bins,
        band_limit=motion_cfg.max_dist_from_median_um,
        speed_limit_um_per_s=motion_cfg.speed_limit_um_per_s,
    )

    return dredge_motion_est


@databag
class MotionSpikeBatches(Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    spikes: DARTsortSorting | Path
    geom: np.ndarray
    motion_cfg: MotionEstimationConfig
    localizations_dataset_name: str
    amplitudes_dataset_name: str
    batch_size: int = 2**20

    def __len__(self) -> int:
        with self._datasets() as (_, _, times, batch_size):
            return len(range(0, times.shape[0], batch_size))

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # ty: ignore
        pad = self.motion_cfg.probe_boundary_padding_um
        xmin = self.geom[:, 0].min() - pad
        xmax = self.geom[:, 0].max() + pad
        zmin = self.geom[:, 1].min() - pad
        zmax = self.geom[:, 1].max() + pad

        with self._datasets() as (locs, amps, times, batch_size):
            n = times.shape[0]
            i0 = i * batch_size
            if i < 0 or i0 >= n:
                raise IndexError(i)
            i1 = min(n, i0 + batch_size)
            batch_locs = locs[i0:i1]
            x = batch_locs[:, 0]
            z = batch_locs[:, 2]
            a = amps[i0:i1]
            valid = np.logical_and(x == x.clip(xmin, xmax), z == z.clip(zmin, zmax))
            if self.motion_cfg.min_amplitude:
                np.logical_and(valid, a >= self.motion_cfg.min_amplitude, out=valid)
            return a[valid], z[valid], times[i0:i1][valid]

    @contextmanager
    def _datasets(self):
        if isinstance(self.spikes, Path):
            with h5py.File(self.spikes, "r", locking=False) as h5:
                locs = h5[self.localizations_dataset_name]
                amps = h5[self.amplitudes_dataset_name]
                times = h5["times_seconds"]
                assert isinstance(locs, h5py.Dataset)
                assert isinstance(amps, h5py.Dataset)
                assert isinstance(times, h5py.Dataset)
                yield locs, amps, times, self.batch_size
        else:
            locs = getattr(self.spikes, self.localizations_dataset_name)
            amps = getattr(self.spikes, self.amplitudes_dataset_name)
            times = self.spikes.times_seconds
            yield locs, amps, times, max(1, times.shape[0])


def dredge_to_si(dredge_motion_est: MotionEstimate) -> Motion:
    disp = np.asarray(dredge_motion_est.displacement)
    t = np.asarray(dredge_motion_est.time_bin_centers_s)
    if disp.ndim == 1:
        return Motion(
            displacement=disp[:, None],
            temporal_bins_s=t,
            spatial_bins_um=np.zeros(1),
        )
    z = np.asarray(dredge_motion_est.spatial_bin_centers_um)
    assert disp.shape == (z.size, t.size)
    return Motion(displacement=disp.T, temporal_bins_s=t, spatial_bins_um=z)


def si_to_dredge(si_motion: Motion) -> MotionEstimate:
    assert si_motion.num_segments == 1
    assert si_motion.direction == "y"
    disp = np.asarray(si_motion.displacement[0])
    t = np.asarray(si_motion.temporal_bins_s[0])
    z = np.asarray(si_motion.spatial_bins_um)
    if z.size == 1:
        return RigidMotionEstimate(disp[:, 0], time_bin_centers_s=t)
    return NonrigidMotionEstimate(
        disp.T, time_bin_centers_s=t, spatial_bin_centers_um=z
    )
