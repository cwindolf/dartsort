from dataclasses import replace
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

import numpy as np
from dredge.motion_util import MotionEstimate
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from spikeinterface.core import BaseRecording, Motion
from torch import Tensor, is_tensor, asarray

if TYPE_CHECKING:
    from .data_util import DARTsortSorting
from .drift_util import get_pitch, registered_geometry
from .internal_config import (
    ComputationConfig,
    MotionEstimationConfig,
    WaveformConfig,
    FitSamplingConfig,
    FeaturizationConfig,
    default_motion_estimation_cfg,
    default_waveform_cfg,
    default_peeling_fit_sampling_cfg,
)
from .job_util import ensure_computation_config
from .py_util import databag, resolve_path
from .registration_util import dredge_estimate_motion, dredge_to_si


def try_load_motion_info(
    output_directory: Path | str | None, filename="motion.pkl"
) -> "MotionInfo | None":
    """Return the saved MotionInfo if there is any, else None."""
    if output_directory is None:
        return None
    return MotionInfo.try_load(output_directory, filename)


def get_motion_info(
    *,
    output_directory: Path | str | None = None,
    filename: str = "motion.pkl",
    recording: BaseRecording,
    sorting: "DARTsortSorting | None",
    detect_new_peaks: bool = False,
    si_motion: Motion | None = None,
    dredge_motion_est: MotionEstimate | None = None,
    motion_cfg: MotionEstimationConfig = default_motion_estimation_cfg,
    waveform_cfg: WaveformConfig | None = None,
    sampling_cfg: FitSamplingConfig | None = None,
    computation_cfg: ComputationConfig | None = None,
    localizations_dataset_name="point_source_localizations",
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    overwrite: bool = False,
    show_progress: bool = True,
) -> "MotionInfo":
    """Get a MotionInfo object by loading from disk, from SI/dredge, or by computing it

    Will save to disk so that future calls can re-load if output_directory is set.
    If no motion has been computed yet, this will call out to DREDge to estimate the
    motion based on the spike locations in the sorting object, using parameters
    from motion_cfg.
    """
    if (motion := try_load_motion_info(output_directory, filename)) is not None:
        return motion

    if detect_new_peaks:
        assert output_directory is not None
        assert sorting is not None
        assert sampling_cfg is not None
        assert waveform_cfg is not None
        motion_sorting = threshold_for_motion(
            output_directory=output_directory,
            recording=recording,
            previous_sorting=sorting,
            motion_cfg=motion_cfg,
            computation_cfg=computation_cfg,
            sampling_cfg=sampling_cfg,
            waveform_cfg=waveform_cfg,
            overwrite=overwrite,
            show_progress=show_progress,
        )
    else:
        motion_sorting = sorting
    assert motion_sorting is not None

    have_si = si_motion is not None
    have_dredge = dredge_motion_est is not None
    assert not (have_si and have_dredge)
    if not (have_si or have_dredge):
        dredge_motion_est = dredge_estimate_motion(
            recording=recording,
            sorting=motion_sorting,
            motion_cfg=motion_cfg,
            device=ensure_computation_config(computation_cfg).actual_device(),
            localizations_dataset_name=localizations_dataset_name,
            amplitudes_dataset_name=amplitudes_dataset_name,
        )
    motion = MotionInfo.from_motion_est(
        geom=recording.get_channel_locations(),
        dredge_motion_est=dredge_motion_est,
        si_motion=si_motion,
    )
    if output_directory is not None:
        motion.save(output_directory, filename, overwrite)

    return motion


@databag
class MotionInfo:
    drifting: bool
    geom: np.ndarray
    rgeom: np.ndarray
    dredge_motion_est: MotionEstimate | None
    si_motion: Motion | None
    geom_kdt: KDTree
    rgeom_kdt: KDTree
    min_dist: float
    pitch: float

    @classmethod
    def from_motion_est(
        cls,
        *,
        geom: np.ndarray | Tensor,
        dredge_motion_est: MotionEstimate | None = None,
        si_motion: Motion | None = None,
        rgeom: np.ndarray | Tensor | None = None,
    ) -> Self:
        """Main constructor for MotionInfo objects

        Precomputes and saves motion-related data structures for use through all
        of dartsort. Notably, the probe pitch, the min inter-channel distance,
        and the "registered geometry". Also, k-d trees which are used everywhere.

        If neither dredge_motion_est nor si_motion is supplied, drifting is set
        to False and there is assumed to be no motion.
        """
        if is_tensor(geom):
            geom = geom.numpy(force=True)
        if is_tensor(rgeom):
            rgeom = rgeom.numpy(force=True)

        have_dredge = dredge_motion_est is not None
        have_si = si_motion is not None
        drifting = have_dredge or have_si

        pitch = get_pitch(geom, allow_horizontal=True)
        if pitch == 0.0:
            assert not drifting
        if geom.shape[0] > 1:
            min_dist = float(np.sqrt(pdist(geom, metric="sqeuclidean").min()))
        else:
            assert not drifting
            min_dist = 0.0

        # get rgeom and its KDTree
        geom_kdt = KDTree(geom)
        if not drifting:
            rgeom = geom
            rgeom_kdt = geom_kdt
        elif rgeom is None:
            if have_dredge:
                assert dredge_motion_est is not None
                d = dredge_motion_est.displacement
            elif have_si:
                assert si_motion is not None
                d = np.asarray(si_motion.displacement)
            else:
                assert False
            rgeom = registered_geometry(
                geom, displacement=d, pitch=pitch, min_distance=min_dist
            )
            rgeom_kdt = KDTree(rgeom)
        else:
            assert rgeom is not None
            rgeom_kdt = KDTree(rgeom)

        return cls(
            drifting=drifting,
            geom=geom,
            rgeom=rgeom,
            dredge_motion_est=dredge_motion_est,
            si_motion=si_motion,
            geom_kdt=geom_kdt,
            min_dist=min_dist,
            rgeom_kdt=rgeom_kdt,
            pitch=pitch,
        )

    @classmethod
    def static(cls, geom: np.ndarray):
        return cls.from_motion_est(geom=geom, dredge_motion_est=None, si_motion=None)

    @property
    def is_nonrigid(self) -> bool:
        if not self.drifting:
            return False
        if self.dredge_motion_est is not None:
            if not hasattr(self.dredge_motion_est, "spatial_bin_centers_um"):
                return False
            elif self.dredge_motion_est.spatial_bin_centers_um is None:
                return False
            else:
                return self.dredge_motion_est.spatial_bin_centers_um.size > 1
        elif self.si_motion is not None:
            return self.si_motion.spatial_bins_um.size > 1
        else:
            assert False

    @property
    def time_bins_s(self) -> np.ndarray:
        if not self.drifting:
            return np.zeros((1,))
        if self.dredge_motion_est is not None:
            b = self.dredge_motion_est.time_bin_centers_s
            if b is None:
                return np.zeros((1,))
            else:
                return b
        elif self.si_motion is not None:
            assert self.si_motion.num_segments == 1
            return self.si_motion.temporal_bins_s[0]
        else:
            assert False

    @property
    def spatial_bins_um(self) -> np.ndarray:
        if not self.is_nonrigid:
            return np.zeros((1,))
        if self.dredge_motion_est is not None:
            assert hasattr(self.dredge_motion_est, "spatial_bin_centers_um")
            assert self.dredge_motion_est.spatial_bin_centers_um is not None
            return self.dredge_motion_est.spatial_bin_centers_um
        elif self.si_motion is not None:
            return self.si_motion.spatial_bins_um
        else:
            assert False

    def disp_at_s(
        self, times_s: np.ndarray, depths_um: np.ndarray, grid: bool = False
    ) -> np.ndarray:
        if self.dredge_motion_est is not None:
            d = self.dredge_motion_est.disp_at_s(
                t_s=times_s, depth_um=depths_um, grid=grid
            )
        elif self.si_motion is not None:
            d = self.si_motion.get_displacement_at_time_and_depth(
                times_s=times_s, locations_um=depths_um, grid=grid
            )
        else:
            assert False
        return d.astype(depths_um.dtype)

    def correct_s(self, times_s: np.ndarray, depths_um: np.ndarray) -> np.ndarray:
        if not self.drifting:
            return np.asarray(depths_um, copy=True)
        z = np.asarray(depths_um)
        return z - self.disp_at_s(times_s, depths_um)

    def uncorrect_s(
        self, times_s: float | np.ndarray, reg_depths_um: np.ndarray
    ) -> np.ndarray:
        """Attempt to invert the motion estimate to un-register reg_depths_um."""
        if self.is_nonrigid:
            assert np.isscalar(times_s)

        if not self.drifting:
            return np.asarray(reg_depths_um, copy=True)

        times_s = np.broadcast_to(times_s, reg_depths_um.shape)

        if not self.is_nonrigid:
            return reg_depths_um + self.disp_at_s(times_s, depths_um=reg_depths_um)

        zc = self.spatial_bins_um
        tt = np.full_like(zc, times_s)
        bin_disp = self.disp_at_s(tt, zc)
        reg_zc = zc - bin_disp
        assert np.all(np.diff(reg_zc) > 0), "Motion invertibility issue."
        reg_depths_um_clipped = reg_depths_um.clip(reg_zc.min(), reg_zc.max())
        disps = np.interp(reg_depths_um_clipped, reg_zc, zc)
        disps = disps.astype(reg_depths_um.dtype)
        return reg_depths_um + disps

    def pitch_shifts(
        self,
        *,
        sorting: "DARTsortSorting | None" = None,
        times_s: np.ndarray | None = None,
        depths_um: np.ndarray | None = None,
        reg_depths_um: np.ndarray | None = None,
        shift_mode: Literal["round", "floor"] = "round",
        motion_depth_mode: Literal["channel", "localization"] = "channel",
        localizations_dset="point_source_localizations",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Figure out coarse pitch shifts based on spike positions

        Determine the number of pitches the probe would need to shift in
        order to coarsely align a waveform to its registered position.
        """
        if depths_um is None:
            assert sorting is not None
            times_s = cast(np.ndarray, getattr(sorting, "times_seconds"))
            if motion_depth_mode == "localization":
                depths_um = cast(np.ndarray, getattr(sorting, localizations_dset))
            elif motion_depth_mode == "channel":
                depths_um = self.geom[sorting.channels, 1]
            else:
                assert False
        assert depths_um is not None
        if not self.drifting:
            out_like = reg_depths_um if reg_depths_um is not None else depths_um
            probe_disp = np.zeros_like(out_like)
            n_pitches_shift = np.zeros(out_like.shape, dtype=np.int32)
            return probe_disp, n_pitches_shift

        if reg_depths_um is None:
            assert times_s is not None
            assert times_s.ndim == 1
            assert times_s.shape == depths_um.shape
            probe_disp = -self.disp_at_s(times_s, depths_um)
        else:
            # rz = z - disp => rz - z = -disp.
            depths_um, reg_depths_um = np.broadcast_arrays(depths_um, reg_depths_um)
            probe_disp = reg_depths_um - depths_um
        assert np.isfinite(probe_disp).all()
        probe_disp = probe_disp.astype(depths_um.dtype)

        if shift_mode == "floor":
            n_pitches_shift = (probe_disp / self.pitch).astype(int)
        elif shift_mode == "round":
            n_pitches_shift = np.round(probe_disp / self.pitch).astype(int)
        else:
            assert False

        return probe_disp, n_pitches_shift

    @classmethod
    def try_load(
        cls, output_directory: Path | str, filename="motion.pkl"
    ) -> Self | None:
        fn = resolve_path(output_directory) / filename
        if not fn.exists():
            return None
        with open(fn, "rb") as jar:
            v = pickle.load(jar)
        return cls.from_motion_est(**v)

    def save(
        self,
        output_directory: Path | str,
        filename="motion.pkl",
        overwrite: bool = False,
    ):
        fn = resolve_path(output_directory) / filename
        if not overwrite and fn.exists():
            return
        v = dict(
            geom=self.geom,
            rgeom=self.rgeom,
            dredge_motion_est=self.dredge_motion_est,
            si_motion=self.si_motion,
        )
        with open(fn, "wb") as jar:
            pickle.dump(v, jar)

    def to_spikeinterface(self) -> Motion | None:
        if self.si_motion is not None:
            return self.si_motion
        elif self.dredge_motion_est is not None:
            return dredge_to_si(self.dredge_motion_est)
        else:
            assert not self.drifting
            return None


def threshold_for_motion(
    *,
    output_directory: Path | str,
    hdf5_filename="motionthreshold.h5",
    model_subdir="motionthreshold_models",
    recording: BaseRecording,
    previous_sorting: "DARTsortSorting | None",
    motion_cfg: MotionEstimationConfig,
    computation_cfg: ComputationConfig | None = None,
    sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    overwrite: bool = False,
    show_progress: bool = False,
):
    """Thresholding detection and localization to get spikes for motion estimation."""
    from ..main import threshold
    from ..transform import WaveformPipeline
    from .data_util import try_get_denoising_pipeline
    from .waveform_util import make_channel_index

    computation_cfg = ensure_computation_config(computation_cfg)

    # load previous denoisers if possible, stack on features
    # else fall back to a more default-like feat cfg
    rad = motion_cfg.localization_radius_um
    sampling_cfg = replace(sampling_cfg, n_residual_snips=0)
    featurization_cfg = FeaturizationConfig(
        save_input_tpca_projs=False,
        save_collidedness=False,
        learn_cleaned_tpca_basis=False,
        extract_radius=rad,
        localization_radius=rad,
        tpca_rank=motion_cfg.tpca_rank,
    )
    if previous_sorting is not None:
        denoising_pipeline, geom, channel_index = try_get_denoising_pipeline(
            previous_sorting
        )
    else:
        denoising_pipeline = None
        geom = asarray(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, featurization_cfg.extract_radius, to_torch=True
        )

    if denoising_pipeline is not None:
        featurization_cfg = replace(
            featurization_cfg, do_nn_denoise=False, do_tpca_denoise=False
        )
    pipeline = WaveformPipeline.from_config(
        featurization_cfg=featurization_cfg,
        waveform_cfg=waveform_cfg,
        geom=geom,
        channel_index=channel_index,
        sampling_frequency=recording.sampling_frequency,
    )
    if denoising_pipeline is not None:
        pipeline = WaveformPipeline(
            transformers=denoising_pipeline.transformers + pipeline.transformers
        )

    return threshold(
        output_dir=output_directory,
        recording=recording,
        waveform_cfg=waveform_cfg,
        thresholding_cfg=motion_cfg.threshold_cfg,
        featurization_cfg=featurization_cfg,
        featurization_pipeline=pipeline,
        sampling_cfg=sampling_cfg,
        extract_channel_index=channel_index,
        chunk_starts_samples=None,
        overwrite=overwrite,
        show_progress=show_progress,
        hdf5_filename=hdf5_filename,
        model_subdir=model_subdir,
        computation_cfg=computation_cfg,
    )
