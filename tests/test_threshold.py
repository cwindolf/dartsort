import tempfile
from pathlib import Path

import dartsort
from dartsort.util import simkit, noise_util


def test_sim():
    geom = simkit.generate_geom()
    rec_sim = simkit.SimulatedRecording(
        duration_samples=3 * 30_000,
        n_units=40,
        template_simulator=simkit.PointSource3ExpSimulator(geom),
        noise=noise_util.WhiteNoise(len(geom)),
        min_fr_hz=20.0,
        max_fr_hz=31.0,
        temporal_jitter=4,
    )

    with tempfile.TemporaryDirectory() as tempdir:
        test_h5 = Path(tempdir) / "sim.h5"
        rec = rec_sim.simulate(test_h5)
        st = dartsort.threshold(
            recording=rec, output_dir=dartsort.resolve_path(tempdir)
        )
        st0 = dartsort.DARTsortSorting.from_peeling_hdf5(test_h5)
        assert abs(len(st) - len(st0)) / len(st0) < 0.2
