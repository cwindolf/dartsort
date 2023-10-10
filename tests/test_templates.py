import numpy as np
import spikeinterface.core as sc
from dartsort import config
from dartsort.templates import get_templates, template_util, templates
from dartsort.util.data_util import DARTsortSorting
from dredge.motion_util import get_motion_estimate


def test_static_templates():
    rec = np.zeros((11, 10))
    # geom = np.c_[np.zeros(10), np.arange(10).astype(float)]
    rec[0, 1] = 1
    rec[3, 5] = 2
    rec = sc.NumpyRecording(rec, 1)
    # rec.set_dummy_probe_from_locations(geom)

    sorting = DARTsortSorting(
        times_samples=[0, 2], labels=[0, 1], channels=[1, 5], sampling_frequency=1
    )

    res = get_templates.get_templates(
        rec,
        sorting,
        trough_offset_samples=0,
        spike_length_samples=2,
        realign_peaks=False,
        low_rank_denoising=False,
    )
    temps = res["raw_templates"]
    assert temps.shape == (2, 2, 10)

    assert temps[0, 0, 1] == 1
    temps[0, 0, 1] -= 1
    assert temps[1, 1, 5] == 2
    temps[1, 1, 5] -= 2
    assert np.all(temps == 0)


def test_drifting_templates():
    geom = np.c_[np.zeros(7), np.arange(7).astype(float)]
    rec = np.zeros((11, 7))
    rec[0, 1] = 1
    rec[2, 2] = 1
    rec[6, 5] = 2
    rec[8, 6] = 2
    rec = sc.NumpyRecording(rec, 1)
    rec.set_dummy_probe_from_locations(geom)

    me = get_motion_estimate(
        0.5 * np.arange(11), time_bin_centers_s=np.arange(11).astype(float)
    )

    sorting = DARTsortSorting(
        times_samples=[0, 2, 6, 8],
        labels=[0, 0, 1, 1],
        channels=[0, 0, 0, 0],
        sampling_frequency=1,
    )
    t_s = [0, 2, 6, 8]

    res = template_util.get_registered_templates(
        rec,
        sorting,
        spike_times_s=t_s,
        spike_depths_um=[0, 0, 0, 0],
        geom=geom,
        motion_est=me,
        registered_template_depths_um=[0, 0],
        trough_offset_samples=0,
        spike_length_samples=2,
        realign_peaks=False,
        low_rank_denoising=False,
        show_progress=False,
    )
    reg_temps = res["templates"]
    registered_geom = res["registered_geom"]

    temps0 = template_util.templates_at_time(
        0,
        reg_temps,
        geom,
        registered_template_depths_um=[0, 0],
        registered_geom=registered_geom,
        motion_est=me,
    )
    assert temps0.shape == (2, 2, 7)
    assert temps0[0, 0, 1] == 1
    assert temps0[1, 0, 2] == 2

    temps6 = template_util.templates_at_time(
        6,
        reg_temps,
        geom,
        registered_template_depths_um=[0, 0],
        registered_geom=registered_geom,
        motion_est=me,
    )
    assert temps6.shape == (2, 2, 7)
    assert temps6[0, 0, 4] == 1
    assert temps6[1, 0, 5] == 2

    temps8 = template_util.templates_at_time(
        8,
        reg_temps,
        geom,
        registered_template_depths_um=[0, 0],
        registered_geom=registered_geom,
        motion_est=me,
    )
    assert temps8.shape == (2, 2, 7)
    assert temps8[0, 0, 5] == 1
    assert temps8[1, 0, 6] == 2


def test_main_object():
    probe = np.c_[np.zeros(7), np.arange(7).astype(float)]
    rec = np.zeros((11, 7))
    rec[0, 1] = 1
    rec[2, 2] = 1
    rec[6, 5] = 2
    rec[8, 6] = 2
    rec = sc.NumpyRecording(rec, 1)
    rec.set_dummy_probe_from_locations(probe)

    me = get_motion_estimate(
        0.5 * np.arange(11), time_bin_centers_s=np.arange(11).astype(float)
    )
    sorting = DARTsortSorting(
        times_samples=[0, 2, 6, 8],
        labels=[0, 0, 1, 1],
        channels=[0, 0, 0, 0],
        sampling_frequency=1,
        extra_features=dict(point_source_localizations=np.zeros((4, 4)), times_seconds=[0, 2, 6, 8]),
    )
    tdata = templates.TemplateData.from_config(
        rec,
        sorting,
        config.TemplateConfig(trough_offset_samples=0, spike_length_samples=2, realign_peaks=False),
        motion_est=me,
    )


if __name__ == "__main__":
    test_static_templates()
    test_drifting_templates()
    test_main_object()
