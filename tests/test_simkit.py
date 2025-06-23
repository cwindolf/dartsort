import h5py
import numpy as np
import neuropixel
import pytest

from dartsort.evaluate import simkit, simlib
from dartsort.util.noise_util import StationaryFactorizedNoise
from dartsort.util.registration_util import estimate_motion


f_dt = "float32"
r_dt = "float16"


def test_np1_geom_is_default():
    g0 = neuropixel.dense_layout()
    g0 = np.c_[g0["x"], g0["y"]]
    assert np.array_equal(g0, simlib.generate_geom())


@pytest.mark.parametrize("globally_refractory", [False, True])
@pytest.mark.parametrize("noise_kind", ["zero", "white"])
def test_exact_injections(tmp_path, tmp_path_factory, globally_refractory, noise_kind):
    nc = 4
    nu = nc
    nt = 1
    fs = 10_000.0
    minfr = 20.0
    refractory_ms = 2.0

    simple_template_library = np.zeros((nu, nt, nc), dtype=f_dt)
    ids = np.arange(nu)
    simple_template_library[ids, 0, ids] = ids.astype(f_dt) + 1

    sim = simkit.generate_simulation(
        tmp_path / "sim",
        tmp_path / "noise",
        noise_kind=noise_kind,
        white_noise_scale=1e-8,
        n_units=nu,
        duration_seconds=1,
        sampling_frequency=fs,
        min_fr_hz=minfr,
        amplitude_jitter=0,
        temporal_jitter=1,
        probe_kwargs=dict(
            num_columns=1, num_contact_per_column=nc, y_shift_per_column=None
        ),
        recording_dtype=r_dt,
        features_dtype=f_dt,
        templates_kind="library",
        template_library=simple_template_library,
        globally_refractory=globally_refractory,
        refractory_ms=refractory_ms,
        template_simulator_kwargs=dict(
            trough_offset_samples=0, randomize_position=False
        ),
        include_tpca_feature=False,
        include_residual=False,
        common_reference=False,
    )
    ns = int(fs)

    target = np.zeros((ns, nc), dtype=r_dt)
    st = sim["sorting"]
    target[st.times_samples, st.labels] = st.labels.astype(r_dt) + 1.0
    traces = sim["recording"].get_traces()
    assert np.allclose(target, traces, atol=1e-5)
    ii, jj = np.nonzero(np.abs(traces) > 0.1)
    assert np.array_equal(ii, st.times_samples)
    assert np.array_equal(jj, st.labels)
    assert np.array_equal(st.channels, st.labels)
    if globally_refractory:
        assert np.diff(ii).min() >= ((fs / 1000) * refractory_ms)
    else:
        for j in range(nu):
            inj = np.flatnonzero(jj == j)
            assert np.diff(ii[inj]).min() >= ((fs / 1000) * refractory_ms)
            print(f"{j=} {inj.size=}")
            assert inj.size > (minfr * 0.5)
    assert np.allclose(sim["templates"].templates, simple_template_library, atol=1e-5)
    assert sim["motion_est"] is None
    u, c = np.unique(st.labels, return_counts=True)
    assert np.array_equal(c, sim["unit_info_df"].gt_spike_count.values)
    assert np.allclose(st.ptp_amplitudes, 1.0 + st.labels.astype(f_dt), atol=1e-5)


@pytest.mark.parametrize("globally_refractory", [False, True])
@pytest.mark.parametrize("templates_kind", ["3exp", "library"])
@pytest.mark.parametrize("noise_kind", ["zero", "white", "stationary_factorized_rbf"])
def test_reproducible_and_residual(
    tmp_path, globally_refractory, templates_kind, noise_kind
):
    sims = []
    kw = {}
    if templates_kind == "library":
        rg = np.random.default_rng(0)
        kw["template_library"] = 10 * rg.normal(size=(10, 121, 48))
    for j, n_jobs in enumerate((1, 4)):
        sim = simkit.generate_simulation(
            tmp_path / f"sim{j}",
            tmp_path / f"noise{j}",
            n_units=10,
            probe_kwargs=dict(num_contact_per_column=12),
            noise_kind=noise_kind,
            globally_refractory=globally_refractory,
            n_jobs=n_jobs,
            sampling_frequency=10_000.0,
            duration_seconds=8.1,
        )
        sims.append(sim)
    sim0, sim1 = sims
    st0 = sim0["sorting"]
    st1 = sim1["sorting"]
    assert np.array_equal(
        sim0["recording"].get_traces(), sim1["recording"].get_traces()
    )
    assert np.array_equal(st0.times_samples, st1.times_samples)
    assert np.array_equal(st0.channels, st1.channels)
    assert np.array_equal(st0.labels, st1.labels)
    assert np.array_equal(st0.times_seconds, st1.times_seconds)
    assert np.array_equal(st0.localizations, st1.localizations)
    assert np.array_equal(st0.ptp_amplitudes, st1.ptp_amplitudes)
    assert np.array_equal(sim0["unit_info_df"].values, sim1["unit_info_df"].values)

    tpca_vals = []
    for st in (st0, st1):
        with h5py.File(st.parent_h5_path, "r", locking=False) as h5:
            f = h5["collisioncleaned_tpca_features"][:]
            np.nan_to_num(f, nan=-111111.0, copy=False)
            tpca_vals.append(f)
    assert np.array_equal(*tpca_vals)
    del tpca_vals, f

    residuals = []
    for st in (st0, st1):
        with h5py.File(st.parent_h5_path, "r", locking=False) as h5:
            f = h5["residual"][:]
            np.nan_to_num(f, nan=-111111.0, copy=False)
            residuals.append(f)
    assert np.array_equal(*residuals)
    del residuals

    # check that the residual has the right stats
    if noise_kind == "zero":
        assert (f == 0.0).all()
    else:
        noise = StationaryFactorizedNoise.estimate(f)
        nc = f.shape[2]
        if noise_kind == "white":
            assert np.allclose(noise.spatial_std, 1.0, atol=0.05)
            assert np.allclose(noise.spatial_cov(), np.eye(nc), atol=0.075)
            assert np.allclose(noise.kernel_fft, 1.0, atol=0.05)
        else:
            gs, gv = simlib.rbf_kernel_sqrt(st0.geom)
            gtk = np.load(simlib.default_temporal_kernel_npy)
            gc = gs * gv.T
            gc = gc @ gc.T
            assert np.allclose(noise.spatial_std, gs[::-1], atol=0.05)
            assert np.allclose(noise.spatial_cov(), gc, atol=0.075)


@pytest.mark.parametrize("drift_speed", [0.0, -1.0, 5.0])
def test_motion(tmp_path, drift_speed):
    sim = simkit.generate_simulation(
        tmp_path / f"sim",
        tmp_path / f"noise",
        n_units=256,
        probe_kwargs=dict(num_columns=1, num_contact_per_column=96, y_shift_per_column=None),
        template_simulator_kwargs=dict(pos_margin_um_z=-80),
        noise_kind="zero",
        sampling_frequency=5_000.0,
        duration_seconds=8,
        drift_speed=drift_speed,
        min_fr_hz=25.0,
        max_fr_hz=25.0,
        include_tpca_feature=False,
        include_residual=False,
    )
    if not drift_speed:
        assert sim["motion_est"] is None
        return

    me0 = sim["motion_est"]
    d0 = me0.displacement.ravel()
    assert np.allclose(np.diff(d0), drift_speed)
    me1 = estimate_motion(
        sim["recording"],
        sim["sorting"],
        rigid=True,
        amplitudes_dataset_name="ptp_amplitudes",
        localizations_dataset_name="localizations",
    )
    d1 = me1.displacement.ravel()
    assert np.array_equal(me0.time_bin_centers_s, me1.time_bin_centers_s)
    assert np.isclose(np.mean(np.square(np.diff(d0)[1:-1] - np.diff(d1)[1:-1])), 0.0, atol=0.1)
