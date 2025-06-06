from dataclasses import replace

import pytest

from dartsort.util import noise_util, data_util
from dartsort.eval import simkit


@pytest.fixture(scope="session")
def sim_recordings(tmp_path_factory):
    geom = simkit.generate_geom()

    # a static example
    tmp_path_static = tmp_path_factory.mktemp("sim_recording_static")
    rec_sim = simkit.SimulatedRecording(
        duration_samples=10 * 30_000,
        template_simulator=simkit.PointSource3ExpSimulator(
            geom, n_units=40, temporal_jitter=4
        ),
        noise=noise_util.WhiteNoise(len(geom)),
        min_fr_hz=20.0,
        max_fr_hz=31.0,
    )
    h5_path = tmp_path_static / "sim.h5"
    rec_static = rec_sim.simulate(h5_path)
    info_static = dict(
        rec=rec_static,
        motion_est=rec_sim.motion_estimate(),
        template_data=replace(
            rec_sim.template_data(), parent_sorting_hdf5_path=h5_path
        ),
        sorting=data_util.DARTsortSorting.from_peeling_hdf5(h5_path),
    )

    # a drifting example
    tmp_path_drifting = tmp_path_factory.mktemp("sim_recording_drifting")
    rec_sim = simkit.SimulatedRecording(
        duration_samples=10 * 30_000,
        template_simulator=simkit.PointSource3ExpSimulator(
            geom, n_units=40, temporal_jitter=4
        ),
        noise=noise_util.WhiteNoise(len(geom)),
        min_fr_hz=20.0,
        max_fr_hz=31.0,
        drift_speed=1.0,
    )
    h5_path = tmp_path_drifting / "sim.h5"
    rec_drifting = rec_sim.simulate(h5_path)
    info_drifting = dict(
        rec=rec_drifting,
        motion_est=rec_sim.motion_estimate(),
        template_data=replace(
            rec_sim.template_data(), parent_sorting_hdf5_path=h5_path
        ),
        sorting=data_util.DARTsortSorting.from_peeling_hdf5(h5_path),
    )

    recs = dict(static=info_static, drifting=info_drifting)

    return recs
