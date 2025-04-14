import dartsort
from dartsort.util import simkit
import tempfile


def test_sim():
    geom = simkit.generate_geom()
    rec_sim = simkit.StaticSimulatedRecording(
        duration_samples=3 * 30_000,
        n_units=40,
        template_simulator=simkit.PointSource3ExpSimulator(geom),
        noise=simkit.WhiteNoise(len(geom)),
        min_fr_hz=20.0,
        max_fr_hz=31.0,
        temporal_jitter=4,
    )
    rec = rec_sim.simulate()

    with tempfile.TemporaryDirectory() as tempdir:
        st = dartsort.threshold(rec, output_dir=dartsort.resolve_path(tempdir))
