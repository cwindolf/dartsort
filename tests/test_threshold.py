import tempfile

import dartsort


def test_sim(sim_recordings):
    rec = sim_recordings["static"]["rec"]
    gt_st = sim_recordings["static"]["sorting"]

    with tempfile.TemporaryDirectory() as tempdir:
        st = dartsort.threshold(
            recording=rec, output_dir=dartsort.resolve_path(tempdir)
        )
        assert abs(len(st) - len(gt_st)) / len(gt_st) < 0.2
