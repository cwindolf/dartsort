import numpy as np
from dartsort.util import drift_util, waveform_util


def test_shifted_waveforms():
    geom = np.c_[[0, 0, 0], [1, 2, 3]]
    displacement = np.array([0, 1, 2, 3, 4])

    reg_geom = drift_util.registered_geometry(
        geom,
        upward_drift=displacement.max(),
        downward_drift=-displacement.min(),
    )
    assert np.array_equal(np.unique(reg_geom[:, 0]), [0])
    assert np.array_equal(np.unique(reg_geom[:, 1]), [1, 2, 3, 4, 5, 6, 7])

    # test shifted channel neighborhoods
    shifted_chans = drift_util.shifted_channel_neighborhood(
        0,
        np.arange(7),
        geom,
        registered_geom=reg_geom,
    )
    assert np.array_equal(shifted_chans, [0, 1, 2, 3, 3, 3, 3])
    shifted_chans = drift_util.shifted_channel_neighborhood(
        -1,
        np.arange(7),
        geom,
        registered_geom=reg_geom,
    )
    assert np.array_equal(shifted_chans, [1, 2, 3, 3, 3, 3, 3])
    shifted_chans = drift_util.shifted_channel_neighborhood(
        1,
        np.arange(7),
        geom,
        registered_geom=reg_geom,
    )
    assert np.array_equal(shifted_chans, [3, 0, 1, 2, 3, 3, 3])

    # ntc
    waveforms = np.zeros((5, 1, 3))
    waveforms[0, :, 0] = 1
    waveforms[1, :, 1] = 1
    waveforms[2, :, 2] = 1
    waveforms[3, :, 2] = 1
    waveforms[4, :, 0] = 1
    n_pitches_shift = np.array([3, 2, 1, 1, 3])
    shifted_waveforms = drift_util.get_waveforms_on_shifted_channel_subset(
        waveforms,
        main_channels=np.zeros(5, dtype=int),
        channel_index=waveform_util.full_channel_index(3),
        target_channels=np.array([3]),
        n_pitches_shift=n_pitches_shift,
        geom=geom,
        registered_geom=reg_geom,
        fill_value=np.nan,
    )
    assert np.all(shifted_waveforms == 1)


if __name__ == "__main__":
    test_shifted_waveforms()
