import tempfile

import dartsort


def test_sim(tmp_path, sim_recordings):
    rec = sim_recordings["static"]["rec"]
    gt_st = sim_recordings["static"]["sorting"]

    st = dartsort.threshold(
        recording=rec, output_dir=tmp_path
    )
    assert abs(len(st) - len(gt_st)) / len(gt_st) < 0.2
