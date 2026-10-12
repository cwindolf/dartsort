import numpy as np
import probeinterface
import pytest
import spikeinterface.core as sc
from dredge import motion_util

from dartsort.evaluate import hybridkit, sim_template_tools, simkit, simlib
from dartsort.evaluate.noise_recording_tools import get_background_recording
from dartsort.util.internal_config import ComputationConfig
from dartsort.util.motion import MotionInfo

fs = 30_000.0
duration_s = 2.0
n_units = 5
chunk_len_s = 0.5
source_pitch = 5.0
source_margin_um = 30.0

drift_period_s = 4.0
drift_params = {
    "y": dict(drift_speed=10.0, nonrigid=False),
    "n": dict(drift_speed=0.0, nonrigid=False),
    "ynr": dict(drift_speed=10.0, nonrigid=True),
}
max_drift_um = (
    max(d["drift_speed"] for d in drift_params.values()) * drift_period_s / 4.0
)


def probe_from_geom(geom):
    probe = probeinterface.Probe(ndim=2)
    probe.set_contacts(positions=geom)
    probe.set_device_channel_indices(np.arange(len(geom)))
    return probe


def dense_grid_geom(target_geom, pitch=source_pitch, margin=source_margin_um):
    x_center = (target_geom[:, 0].min() + target_geom[:, 0].max()) / 2.0
    x_half = np.ptp(target_geom[:, 0]) / 2.0 + margin
    nx = int(np.ceil(x_half / pitch))
    xs = x_center + pitch * np.arange(-nx, nx + 1)

    z_low = target_geom[:, 1].min() - margin
    z_high = target_geom[:, 1].max() + margin
    nz = int(np.ceil((z_high - z_low) / pitch))
    zs = z_low + pitch * np.arange(nz + 1)

    x_grid, z_grid = np.meshgrid(xs, zs)
    return np.c_[x_grid.ravel(), z_grid.ravel()]


def narrow_grid_geom(
    target_geom, n_columns=3, pitch=source_pitch, margin=source_margin_um
):
    """Like dense_grid_geom, but spanning less x than target_geom does."""
    geom = dense_grid_geom(target_geom, pitch=pitch, margin=margin)
    x_center = (target_geom[:, 0].min() + target_geom[:, 0].max()) / 2.0
    x_half = pitch * (n_columns - 1) / 2.0
    assert 2 * x_half < np.ptp(target_geom[:, 0])
    return geom[np.abs(geom[:, 0] - x_center) <= x_half + 1e-6]


@pytest.fixture(scope="module")
def target_geom():
    """A shortened NP1-like probe."""
    return simlib.generate_geom(
        num_columns=2, num_contact_per_column=12, y_shift_per_column="flat"
    )


@pytest.fixture(scope="module")
def simulator(target_geom):
    return sim_template_tools.get_template_simulator(
        n_units=n_units,
        templates_kind="3exp",
        geom=target_geom,
        sampling_frequency=fs,
        common_reference=False,
        temporal_jitter=1,
        random_seed=0,
        min_rms_distance=1.0,
    )


def point_source_library(simulator, geom):
    """Evaluate simulator's PS3exp templates on the other geom geom"""
    geom3 = np.zeros((len(geom), 3), dtype=simulator.dtype)
    geom3[:, [0, 2]] = geom
    return sim_template_tools.singlechan_to_probe(
        simulator.template_pos,
        simulator.template_alpha,
        simulator.singlechan_templates,
        geom3,
        decay_model=simulator.decay_model,
    )


def rigid_to_nonrigid(motion, n_bins=3):
    assert not motion.is_nonrigid
    me = motion.to_dredge()
    assert me is not None
    depths = motion.geom[:, 1]
    spatial_bin_centers_um = np.linspace(depths.min(), depths.max(), num=n_bins)
    displacement = np.broadcast_to(me.displacement, (n_bins, me.displacement.size))
    nonrigid_me = motion_util.get_motion_estimate(
        displacement=displacement.copy(),
        time_bin_centers_s=me.time_bin_centers_s,
        spatial_bin_centers_um=spatial_bin_centers_um,
    )
    nonrigid = MotionInfo.from_motion_est(
        geom=motion.geom, dredge_motion_est=nonrigid_me
    )
    assert nonrigid.is_nonrigid
    np.testing.assert_array_equal(nonrigid.rgeom, motion.rgeom)
    return nonrigid


def zeros_recording(geom, num_samples):
    recording = get_background_recording(
        None,
        duration_samples=num_samples,
        geom=geom,
        noise_kind="zero",
        sampling_frequency=fs,
        dtype="float32",
    )
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    return recording


@pytest.mark.parametrize("drift", list(drift_params))
def test_hybrid_matches_sim(tmp_path, target_geom, simulator, drift):
    """Sim ~= hybrid with zeros background, dense grid hybrid geom."""
    drift_speed = drift_params[drift]["drift_speed"]
    nonrigid = drift_params[drift]["nonrigid"]
    computation_cfg = ComputationConfig.from_n_jobs(1)

    sim = simkit.generate_simulation(
        tmp_path / "sim",
        tmp_path / "noise",
        n_units=n_units,
        duration_seconds=duration_s,
        geom=target_geom,
        sampling_frequency=fs,
        template_simulator=simulator,
        noise_kind="white",
        white_noise_scale=0.0,
        recording_dtype="float32",
        amplitude_jitter=0.0,
        temporal_jitter=1,
        min_fr_hz=20.0,
        max_fr_hz=25.0,
        drift_type="triangle",
        drift_speed=drift_speed,
        drift_period=drift_period_s,
        max_chunk_len_s=chunk_len_s,
        max_drift_per_chunk=np.inf,
        featurization_cfg=None,
        computation_cfg=computation_cfg,
    )
    assert isinstance(sim, dict)
    gt_sorting = sim["sorting"]
    num_samples = sim["recording"].get_num_frames()

    source_geom = dense_grid_geom(target_geom)
    library = point_source_library(simulator, source_geom)
    main_channels = np.abs(library).max(1).argmax(1)
    si_templates = sc.Templates(
        templates_array=library,
        sampling_frequency=fs,
        nbefore=simulator.trough_offset_samples(),
        is_in_uV=True,
        probe=probe_from_geom(source_geom),
    )

    assert np.array_equal(np.unique(gt_sorting.labels), np.arange(n_units))
    injected_sorting = sc.NumpySorting.from_samples_and_labels(
        [gt_sorting.times_samples], [gt_sorting.labels], sampling_frequency=fs
    )

    motion = sim["motion"]
    if nonrigid:
        motion = rigid_to_nonrigid(motion)

    hybrid = hybridkit.make_hybrid_recording(
        folder=tmp_path / "hybrid",
        target_recording=zeros_recording(target_geom, num_samples),
        injected_sorting=injected_sorting,
        motion=motion,
        templates=si_templates,
        filter=False,
        remove_bad_channels=False,
        template_peak_range=None,
        amplitude_jitter=0.0,
        temporal_jitter=1,
        target_sampling_frequency=fs,
        save_chunk_len_s=chunk_len_s,
        template_simulator_kwargs=dict(
            interp_method="grid_sample",
            extract_radius=10 * np.ptp(source_geom[:, 1]),
            depths=source_geom[main_channels, 1],
        ),
        computation_cfg=computation_cfg,
    )

    np.testing.assert_array_equal(
        hybrid.gt_sorting.times_samples, gt_sorting.times_samples
    )
    np.testing.assert_array_equal(hybrid.gt_sorting.labels, gt_sorting.labels)

    sim_templates = sim["templates"].templates
    scale = np.abs(sim_templates).max()
    assert scale > 1.0
    np.testing.assert_allclose(
        hybrid.gt_templates.templates, sim_templates, atol=0.01 * scale
    )
    np.testing.assert_array_equal(
        hybrid.gt_templates.registered_geom, sim["templates"].registered_geom
    )

    sim_traces = sim["recording"].get_traces()
    hybrid_traces = hybrid.recording.get_traces()
    assert hybrid_traces.shape == sim_traces.shape
    trace_scale = np.abs(sim_traces).max()
    assert trace_scale > 1.0
    rms_error = np.sqrt(np.mean(np.square(hybrid_traces - sim_traces)))
    assert rms_error < 0.005 * np.sqrt(np.mean(np.square(sim_traces)))
    np.testing.assert_allclose(hybrid_traces, sim_traces, atol=0.01 * trace_scale)

    sim_units = sim["unit_info_df"]
    hybrid_units = hybrid.gt_unit_info
    assert np.abs(hybrid_units.x - sim_units.x).max() < source_pitch
    assert np.abs(hybrid_units.z - sim_units.z).max() < source_pitch


@pytest.mark.parametrize("trough_shift", [0, -3, 5])
@pytest.mark.parametrize(
    "interp_method", ["dart", "grid_sample", "grid_sample_average"]
)
def test_misaligned_library(
    tmp_path, target_geom, simulator, trough_shift, interp_method
):
    computation_cfg = ComputationConfig.from_n_jobs(1)
    nbefore = simulator.trough_offset_samples()
    library = np.roll(simulator.templates()[1], trough_shift, axis=1)
    si_templates = sc.Templates(
        templates_array=library,
        sampling_frequency=fs,
        nbefore=nbefore,
        is_in_uV=True,
        probe=probe_from_geom(target_geom),
    )

    spike_length = simulator.spike_length_samples()
    refrac = 4 * spike_length
    num_samples = int(duration_s * fs)
    times = np.arange(spike_length, num_samples - refrac, refrac)
    labels = np.arange(times.size) % n_units
    injected_sorting = sc.NumpySorting.from_samples_and_labels(
        [times], [labels], sampling_frequency=fs
    )

    hybrid = hybridkit.make_hybrid_recording(
        folder=tmp_path / "hybrid",
        target_recording=zeros_recording(target_geom, num_samples),
        injected_sorting=injected_sorting,
        motion=MotionInfo.static(target_geom),
        templates=si_templates,
        filter=False,
        remove_bad_channels=False,
        template_peak_range=None,
        amplitude_jitter=0.0,
        temporal_jitter=1,
        target_sampling_frequency=fs,
        save_chunk_len_s=chunk_len_s,
        template_simulator_kwargs=dict(
            interp_method=interp_method,
            extract_radius=10 * np.ptp(target_geom[:, 1]),
        ),
        computation_cfg=computation_cfg,
    )
    assert hybrid.metadata["template_interp_method"] == interp_method

    # check sim times are shifted by the same shifts
    np.testing.assert_array_equal(hybrid.gt_sorting.times_samples, times + trough_shift)

    # check trace argminima
    traces = hybrid.recording.get_traces()
    half = spike_length // 2
    for time in hybrid.gt_sorting.times_samples:
        snippet = traces[time - half : time + half + 1]
        trough_t, _ = np.unravel_index(snippet.argmin(), snippet.shape)
        assert trough_t == half


def test_narrow_source_x_align(tmp_path, target_geom, simulator):
    computation_cfg = ComputationConfig.from_n_jobs(1)
    source_geom = narrow_grid_geom(target_geom)
    assert np.ptp(source_geom[:, 0]) < np.ptp(target_geom[:, 0])
    library = point_source_library(simulator, source_geom)
    main_channels = np.abs(library).max(1).argmax(1)
    si_templates = sc.Templates(
        templates_array=library,
        sampling_frequency=fs,
        nbefore=simulator.trough_offset_samples(),
        is_in_uV=True,
        probe=probe_from_geom(source_geom),
    )

    spike_length = simulator.spike_length_samples()
    refrac = 4 * spike_length
    num_samples = int(duration_s * fs)
    times = np.arange(spike_length, num_samples - refrac, refrac)
    labels = np.arange(times.size) % n_units
    injected_sorting = sc.NumpySorting.from_samples_and_labels(
        [times], [labels], sampling_frequency=fs
    )
    simulator_kwargs = dict(
        interp_method="grid_sample",
        extract_radius=10 * np.ptp(source_geom[:, 1]),
        depths=source_geom[main_channels, 1],
    )

    hybrid = hybridkit.make_hybrid_recording(
        folder=tmp_path / "hybrid",
        target_recording=zeros_recording(target_geom, num_samples),
        injected_sorting=injected_sorting,
        motion=MotionInfo.static(target_geom),
        templates=si_templates,
        filter=False,
        remove_bad_channels=False,
        template_peak_range=None,
        template_x_align="amplitude",
        amplitude_jitter=0.0,
        temporal_jitter=1,
        target_sampling_frequency=fs,
        save_chunk_len_s=chunk_len_s,
        template_simulator_kwargs=simulator_kwargs,
        computation_cfg=computation_cfg,
    )
    assert hybrid.metadata["template_x_align"] == "amplitude"
    np.testing.assert_array_equal(hybrid.gt_templates.registered_geom, target_geom)

    # units to the left of the source probe went to the target's left column
    x_low, x_high = target_geom[:, 0].min(), target_geom[:, 0].max()
    source_center = (source_geom[:, 0].min() + source_geom[:, 0].max()) / 2.0
    left = simulator.template_pos[:, 0] < source_center
    assert left.any() and not left.all()

    ptps = np.ptp(hybrid.gt_templates.templates, axis=1)
    peak_x = target_geom[ptps.argmax(1), 0]
    np.testing.assert_array_equal(peak_x, np.where(left, x_low, x_high))

    library_ptps = np.ptp(library, axis=1).max(1)
    assert (ptps.max(1) > 0.85 * library_ptps).all()

    # whereas centering would have cost them most of it
    centered = sim_template_tools.TemplateLibrarySimulator.from_template_library(
        source_geom=source_geom,
        target_geom=target_geom,
        n_units=n_units,
        templates=library,
        x_align="center",
        trough_offset_samples=simulator.trough_offset_samples(),
        **simulator_kwargs,  # ty: ignore[invalid-argument-type]
    )
    centered_ptps = np.ptp(centered.templates()[1], axis=1)
    assert (centered_ptps.max(1) < 0.6 * ptps.max(1)).all()


def test_grid_sample_average(target_geom, simulator):
    source_geom = dense_grid_geom(target_geom)
    library = point_source_library(simulator, source_geom)
    main_channels = np.abs(library).max(1).argmax(1)
    offset = source_pitch / 2.0
    shared = dict(
        source_geom=source_geom,
        n_units=n_units,
        templates=library,
        depths=source_geom[main_channels, 1],
        extract_radius=10 * np.ptp(source_geom[:, 1]),
        trough_offset_samples=simulator.trough_offset_samples(),
    )

    avg_temp_sim = sim_template_tools.TemplateLibrarySimulator.from_template_library(
        target_geom=target_geom,
        interp_method="grid_sample_average",
        grid_average_offset_um=offset,
        **shared,  # ty: ignore[invalid-argument-type]
    )
    np.testing.assert_array_equal(avg_temp_sim.grid_avg_offsets, [-offset, offset])

    # shifting the target geom is the same as offsetting the interpolation
    # check that at a couple of drifts
    sample_sims = [
        sim_template_tools.TemplateLibrarySimulator.from_template_library(
            target_geom=target_geom + [0.0, shift],
            interp_method="grid_sample",
            **shared,  # ty: ignore[invalid-argument-type]
        )
        for shift in (-offset, offset)
    ]
    for drift_um in (0.0, max_drift_um, -max_drift_um):
        drift = np.full(n_units, drift_um)
        templates = avg_temp_sim.templates(drift=drift)[1]
        mean_of_shifted = np.mean(
            [sim.templates(drift=drift)[1] for sim in sample_sims], axis=0
        )
        scale = np.abs(mean_of_shifted).max()
        assert scale > 1.0
        np.testing.assert_allclose(templates, mean_of_shifted, atol=1e-5 * scale)

    # check that we are testing something, ie that not doing any of this is different
    plain = sim_template_tools.TemplateLibrarySimulator.from_template_library(
        target_geom=target_geom,
        interp_method="grid_sample",
        **shared,  # ty: ignore[invalid-argument-type]
    )
    np.testing.assert_array_equal(plain.grid_avg_offsets, [0.0])
    avg_temps = avg_temp_sim.templates()[1]
    avg_ptps = np.ptp(avg_temps, axis=1).max(1)
    plain_temps = plain.templates()[1]
    plain_ptps = np.ptp(plain_temps, axis=1).max(1)
    assert not np.allclose(avg_temps, plain_temps, atol=1e-5 * scale)  # ty:ignore[possibly-unresolved-reference, unused-ignore-comment]
    assert (avg_ptps < plain_ptps).all()
    assert (avg_ptps > 0.9 * plain_ptps).all()


def test_drift_is_actually_tested_but_not_too_much():
    assert max_drift_um >= 2 * source_pitch
    assert max_drift_um + 2 * source_pitch <= source_margin_um
