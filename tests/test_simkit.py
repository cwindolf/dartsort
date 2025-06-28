import h5py
import numpy as np
import pytest

from dartsort.evaluate import simkit, simlib
from dartsort.util.noise_util import StationaryFactorizedNoise
from dartsort.util.registration_util import estimate_motion


f_dt = "float32"
r_dt = "float16"


def test_np1_geom_is_default():
    assert np.array_equal(np1_dense_layout, simlib.generate_geom())


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
@pytest.mark.parametrize("templates_kind", ["3exp", "library", "librarygrid"])
@pytest.mark.parametrize("noise_kind", ["zero", "white", "stationary_factorized_rbf"])
def test_reproducible_and_residual(
    tmp_path, globally_refractory, templates_kind, noise_kind
):
    sims = []

    kw = {}
    if templates_kind.startswith("library"):
        rg = np.random.default_rng(0)
        kw["template_library"] = 10 * rg.normal(size=(10, 121, 48))
    if templates_kind == "librarygrid":
        kw["template_simulator_kwargs"] = dict(
            interp_method="griddata", interp_kernel_name="linear"
        )

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
            templates_kind=templates_kind,
            **kw,
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
        probe_kwargs=dict(
            num_columns=1, num_contact_per_column=96, y_shift_per_column=None
        ),
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
    assert np.isclose(
        np.mean(np.square(np.diff(d0)[1:-1] - np.diff(d1)[1:-1])), 0.0, atol=0.1
    )


# NP1 dense layout as generated by ibl-neuropixel
np1str = "43,20,11,20,59,40,27,40,43,60,11,60,59,80,27,80,43,100,11,100,59,120,27,120,43,140,11,140,59,160,27,160,43,180,11,180,59,200,27,200,43,220,11,220,59,240,27,240,43,260,11,260,59,280,27,280,43,300,11,300,59,320,27,320,43,340,11,340,59,360,27,360,43,380,11,380,59,400,27,400,43,420,11,420,59,440,27,440,43,460,11,460,59,480,27,480,43,500,11,500,59,520,27,520,43,540,11,540,59,560,27,560,43,580,11,580,59,600,27,600,43,620,11,620,59,640,27,640,43,660,11,660,59,680,27,680,43,700,11,700,59,720,27,720,43,740,11,740,59,760,27,760,43,780,11,780,59,800,27,800,43,820,11,820,59,840,27,840,43,860,11,860,59,880,27,880,43,900,11,900,59,920,27,920,43,940,11,940,59,960,27,960,43,980,11,980,59,1000,27,1000,43,1020,11,1020,59,1040,27,1040,43,1060,11,1060,59,1080,27,1080,43,1100,11,1100,59,1120,27,1120,43,1140,11,1140,59,1160,27,1160,43,1180,11,1180,59,1200,27,1200,43,1220,11,1220,59,1240,27,1240,43,1260,11,1260,59,1280,27,1280,43,1300,11,1300,59,1320,27,1320,43,1340,11,1340,59,1360,27,1360,43,1380,11,1380,59,1400,27,1400,43,1420,11,1420,59,1440,27,1440,43,1460,11,1460,59,1480,27,1480,43,1500,11,1500,59,1520,27,1520,43,1540,11,1540,59,1560,27,1560,43,1580,11,1580,59,1600,27,1600,43,1620,11,1620,59,1640,27,1640,43,1660,11,1660,59,1680,27,1680,43,1700,11,1700,59,1720,27,1720,43,1740,11,1740,59,1760,27,1760,43,1780,11,1780,59,1800,27,1800,43,1820,11,1820,59,1840,27,1840,43,1860,11,1860,59,1880,27,1880,43,1900,11,1900,59,1920,27,1920,43,1940,11,1940,59,1960,27,1960,43,1980,11,1980,59,2000,27,2000,43,2020,11,2020,59,2040,27,2040,43,2060,11,2060,59,2080,27,2080,43,2100,11,2100,59,2120,27,2120,43,2140,11,2140,59,2160,27,2160,43,2180,11,2180,59,2200,27,2200,43,2220,11,2220,59,2240,27,2240,43,2260,11,2260,59,2280,27,2280,43,2300,11,2300,59,2320,27,2320,43,2340,11,2340,59,2360,27,2360,43,2380,11,2380,59,2400,27,2400,43,2420,11,2420,59,2440,27,2440,43,2460,11,2460,59,2480,27,2480,43,2500,11,2500,59,2520,27,2520,43,2540,11,2540,59,2560,27,2560,43,2580,11,2580,59,2600,27,2600,43,2620,11,2620,59,2640,27,2640,43,2660,11,2660,59,2680,27,2680,43,2700,11,2700,59,2720,27,2720,43,2740,11,2740,59,2760,27,2760,43,2780,11,2780,59,2800,27,2800,43,2820,11,2820,59,2840,27,2840,43,2860,11,2860,59,2880,27,2880,43,2900,11,2900,59,2920,27,2920,43,2940,11,2940,59,2960,27,2960,43,2980,11,2980,59,3000,27,3000,43,3020,11,3020,59,3040,27,3040,43,3060,11,3060,59,3080,27,3080,43,3100,11,3100,59,3120,27,3120,43,3140,11,3140,59,3160,27,3160,43,3180,11,3180,59,3200,27,3200,43,3220,11,3220,59,3240,27,3240,43,3260,11,3260,59,3280,27,3280,43,3300,11,3300,59,3320,27,3320,43,3340,11,3340,59,3360,27,3360,43,3380,11,3380,59,3400,27,3400,43,3420,11,3420,59,3440,27,3440,43,3460,11,3460,59,3480,27,3480,43,3500,11,3500,59,3520,27,3520,43,3540,11,3540,59,3560,27,3560,43,3580,11,3580,59,3600,27,3600,43,3620,11,3620,59,3640,27,3640,43,3660,11,3660,59,3680,27,3680,43,3700,11,3700,59,3720,27,3720,43,3740,11,3740,59,3760,27,3760,43,3780,11,3780,59,3800,27,3800,43,3820,11,3820,59,3840,27,3840"
np1_dense_layout = np.array([float(z) for z in np1str.split(",")])
np1_dense_layout = np1_dense_layout.reshape(384, 2)
