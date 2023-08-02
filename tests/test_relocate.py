"""
This tests relocate.py and the get_waveforms_on... functions
in drift_util.py
"""
import numpy as np
from dartsort.cluster import relocate
from dartsort.util import drift_util, waveform_util
from neuropixel import dense_layout

h = dense_layout()
geom = np.c_[h["x"], h["y"]]
geom = geom[np.lexsort(geom.T)]


def test_relocate_fixed_chans():
    rg = np.random.default_rng(0)
    # need enough random spikes to trigger corner cases
    # nspikes = 1003
    nspikes = 3

    # random starting positions
    x = rg.normal(size=nspikes)
    y = np.square(rg.normal(size=nspikes))
    z = rg.uniform(geom[:, 1].min(), geom[:, 1].max(), size=nspikes)
    alpha = 10 + np.square(rg.normal(size=nspikes))

    # ending positions and target channels
    z_dest = np.full_like(z, geom[:, 1].mean())
    dest_chans = np.arange(len(geom) // 2 - 10, len(geom) // 2 + 10)

    # amplitudes on the full probe
    ptps = relocate.point_source_amplitude_vectors(
        x, y, z, alpha, geom, channels=None
    )
    assert np.array_equal(
        ptps[:, dest_chans],
        drift_util.get_waveforms_on_static_channels(
            ptps,
            geom,
            target_channels=dest_chans,
        ),
    )

    # targets on target chans
    targ_ptps = relocate.point_source_amplitude_vectors(
        x, y, z_dest, alpha, geom, channels=dest_chans
    )
    # one sample wfs with perfect point source amplitudes
    wfs = ptps[:, None, :]
    targ_wfs = targ_ptps[:, None, :]

    shifted = relocate.relocated_waveforms_on_static_channels(
        wfs,
        np.zeros(nspikes, dtype=int),
        waveform_util.full_channel_index(len(geom)),
        dest_chans,
        np.c_[x, y, z, alpha],
        z_dest,
        geom,
        fill_value=np.nan,
    )

    assert shifted.shape == targ_wfs.shape
    pitch = drift_util.get_pitch(geom)
    displacement = z_dest - z
    pitch_shifts = (displacement / pitch).astype(int)
    had_nan = np.isnan(shifted).any(axis=(1, 2))
    # 46 = ceil((384/2 - 10)/4), where 4 = pitch here
    assert np.array_equal(had_nan, np.abs(pitch_shifts) >= 46)
    assert np.isclose(
        np.nan_to_num(shifted), np.isfinite(shifted) * np.nan_to_num(targ_wfs)
    ).all()


def test_relocate_varying_chans():
    rg = np.random.default_rng(0)
    # need enough random spikes to trigger corner cases
    nspikes = 1003

    # random starting positions
    x = rg.normal(size=nspikes)
    y = np.square(rg.normal(size=nspikes))
    z = rg.uniform(geom[:, 1].min(), geom[:, 1].max(), size=nspikes)
    alpha = 10 + np.square(rg.normal(size=nspikes))
    main_channels = np.array(
        [
            np.argmin(np.square(geom - [xx, zz]).sum(axis=1))
            for xx, zz in zip(x, z)
        ]
    )
    channel_index = waveform_util.make_channel_index(geom, radius=100)

    # ending positions and target channels
    z_dest = np.full_like(z, geom[:, 1].mean())
    dest_chans = np.arange(len(geom) // 2 - 10, len(geom) // 2 + 10)

    # amplitudes on the full probe
    ptps = relocate.point_source_amplitude_vectors(
        x, y, z, alpha, geom, channels=None
    )
    ptps = np.pad(ptps, [(0, 0), (0, 1)], constant_values=np.nan)
    ptps = np.array(
        [ptps[i][channel_index[mc]] for i, mc in enumerate(main_channels)]
    )

    # targets on target chans
    targ_ptps = relocate.point_source_amplitude_vectors(
        x, y, z_dest, alpha, geom, channels=dest_chans
    )
    # one sample wfs with perfect point source amplitudes
    wfs = ptps[:, None, :]
    targ_wfs = targ_ptps[:, None, :]

    shifted = relocate.relocated_waveforms_on_static_channels(
        wfs,
        main_channels,
        channel_index,
        dest_chans,
        np.c_[x, y, z, alpha],
        z_dest,
        geom,
        fill_value=np.nan,
    )

    assert np.isclose(
        np.nan_to_num(shifted), np.isfinite(shifted) * np.nan_to_num(targ_wfs)
    ).all()

    # using registered geometry
    pitch = drift_util.get_pitch(geom)
    reg_geom = drift_util.registered_geometry(
        geom, upward_drift=10 * pitch, downward_drift=10 * pitch
    )
    assert np.array_equal(reg_geom[40:-40], geom)
    shifted1 = relocate.relocated_waveforms_on_static_channels(
        wfs,
        main_channels,
        channel_index,
        40 + dest_chans,
        np.c_[x, y, z, alpha],
        z_dest,
        geom,
        registered_geom=reg_geom,
        fill_value=np.nan,
    )
    assert np.array_equal(np.isnan(shifted), np.isnan(shifted1))
    assert np.array_equal(np.nan_to_num(shifted), np.nan_to_num(shifted1))
    assert np.isclose(
        np.nan_to_num(shifted), np.isfinite(shifted) * np.nan_to_num(targ_wfs)
    ).all()


if __name__ == "__main__":
    test_relocate_fixed_chans()
    test_relocate_varying_chans()
