from dataclasses import replace
from itertools import product

import pytest

from dartsort.util import noise_util, data_util
from dartsort.eval import simkit



@pytest.fixture(scope="session")
def sim_recordings(tmp_path_factory):
    geom = simkit.generate_geom()

    recs = {}

    for drifting, mini in product((True, False), (True, False)):
        kind = "drifting" if drifting else "static"
        suffix = "_mini" if mini else ""
        t_s = 4 if mini else 9.9
        n_units = 20 if mini else 40
        fr_boost = 10 if mini else 0

        # a static example
        tmp_path = tmp_path_factory.mktemp(f"sim_recording_{kind}{suffix}")
        rec_sim = simkit.SimulatedRecording(
            duration_samples=t_s * 30_000,
            template_simulator=simkit.PointSource3ExpSimulator(
                geom, n_units=n_units, temporal_jitter=4
            ),
            noise=noise_util.WhiteNoise(len(geom)),
            min_fr_hz=20.0 + fr_boost,
            max_fr_hz=31.0 + fr_boost,
            drift_speed=1.0 * drifting,
        )
        h5_path = tmp_path / "sim.h5"
        # add tpca features to the mini ones only

        with pytest.warns(Warning) as warninfo:
            rec = rec_sim.simulate(h5_path, with_tpca_features=mini)
        # recording is too short to extract all residual snips so there's a warning
        warnings = {(w.category, w.message.args[0][:10]) for w in warninfo}
        expected = {(UserWarning, "Can't extr")}
        assert warnings == expected

        info = dict(
            rec=rec,
            motion_est=rec_sim.motion_estimate(),
            template_data=replace(
                rec_sim.template_data(), parent_sorting_hdf5_path=h5_path
            ),
            sorting=data_util.DARTsortSorting.from_peeling_hdf5(h5_path),
        )
        recs[f"{kind}{suffix}"] = info

    return recs
