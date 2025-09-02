import pytest
from dartsort.evaluate import simkit, config_grid


common_params = dict(
    probe_kwargs=dict(
        num_columns=2, num_contact_per_column=12, y_shift_per_column=None
    ),
    temporal_jitter=4,
    noise_kind="white",
)
drift_params = {"y": dict(drift_speed=1.0), "n": dict(drift_speed=0.0)}


@pytest.fixture(scope="session")
def mini_simulations(tmp_path_factory):
    sim_settings = config_grid(
        common_params=common_params,
        config_cls=None,
        drift=drift_params,
        sz={
            "mini": dict(
                duration_seconds=3.0, n_units=20, min_fr_hz=30.0, max_fr_hz=41.0
            )
        },
    )
    assert len(sim_settings) == 2

    sims = {}
    for sim_name, kw in sim_settings.items():
        p = tmp_path_factory.mktemp(f"simdata_{sim_name}")
        sims[sim_name] = simkit.generate_simulation(p / "sim", p / "noise", **kw)

    return sims


@pytest.fixture(scope="session")
def simulations(tmp_path_factory, mini_simulations):
    sim_settings = config_grid(
        common_params=common_params,
        config_cls=None,
        drift=drift_params,
        sz={
            "reg": dict(
                duration_seconds=9.9, n_units=40, min_fr_hz=20.0, max_fr_hz=31.0
            )
        },
    )
    assert len(sim_settings) == 2

    sims = {**mini_simulations}
    for sim_name, kw in sim_settings.items():
        p = tmp_path_factory.mktemp(f"simdata_{sim_name}")
        sims[sim_name] = simkit.generate_simulation(p / "sim", p / "noise", **kw)

    return sims
