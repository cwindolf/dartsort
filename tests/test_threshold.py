import dartsort


def test_sim(tmp_path, simulations):
    rec = simulations["driftn_szmini"]["recording"]
    gt_st = simulations["driftn_szmini"]["sorting"]

    st = dartsort.threshold(recording=rec, output_dir=tmp_path)
    assert abs(len(st) - len(gt_st)) / len(gt_st) < 0.2
