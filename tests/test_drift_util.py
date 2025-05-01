import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from dartsort.util import drift_util, waveform_util, simkit
import dredge.motion_util as mu


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

    # fixed check
    waveforms = np.arange(15).reshape(5, 3)[:, None, :].astype(np.float32)
    w = drift_util.get_waveforms_on_static_channels(waveforms, geom=geom)
    assert np.array_equal(w, waveforms)
    w = drift_util.get_waveforms_on_static_channels(
        waveforms, geom=geom, target_channels=[1]
    )
    assert np.array_equal(w, waveforms[:, :, [1]])
    w = drift_util.get_waveforms_on_static_channels(
        waveforms, geom=geom, target_channels=np.arange(2)
    )
    assert np.array_equal(w, waveforms[:, :, np.arange(2)])

    # ntc
    waveforms = np.zeros((5, 1, 3))
    waveforms[0, :, 0] = 1
    waveforms[1, :, 1] = 1
    waveforms[2, :, 2] = 1
    waveforms[3, :, 2] = 1
    waveforms[4, :, 0] = 1
    n_pitches_shift = np.array([3, 2, 1, 1, 3])
    shifted_waveforms = drift_util.get_waveforms_on_static_channels(
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
    shifted_waveforms = drift_util.get_waveforms_on_static_channels(
        waveforms,
        target_channels=np.array([3]),
        n_pitches_shift=n_pitches_shift,
        geom=geom,
        registered_geom=reg_geom,
        fill_value=np.nan,
    )
    assert np.all(shifted_waveforms == 1)
    shifted_avg = drift_util.registered_average(
        waveforms,
        n_pitches_shift=n_pitches_shift,
        geom=geom,
        registered_geom=reg_geom,
        pad_value=0.0,
    )
    assert np.array_equal(shifted_avg, [[0.0, 0, 0, 1, 0, 0, 0]])


@pytest.fixture
def example_geoms():
    geom = simkit.generate_geom(num_contact_per_column=48)
    geom0 = geom

    # make some holes
    rg = np.random.default_rng(0)
    choices = rg.choice(len(geom), replace=False, size=(2 * len(geom)) // 3)
    geom1 = geom[np.sort(choices)]
    geom2 = geom[choices]  # unsorted version

    # single column geom
    geom3 = np.c_[np.zeros(10), np.arange(10)]

    # single column with holes
    geom4 = geom2[[0, 1, 3, 4, 5, 7, 8, 9]]

    # oddball column with one friend, and it's not even sorted
    geom5 = np.concatenate([geom4, [[1, 5.2]]], axis=0)

    return [geom0, geom1, geom2, geom3, geom4, geom5]


@pytest.mark.parametrize("geom_ix", range(6))
@pytest.mark.parametrize("drift_speed", [0, 10, 100, -10, -132])
def test_registered_geometry(example_geoms, geom_ix, drift_speed):
    geom = example_geoms[geom_ix]

    # create fake motion estimate
    T_seconds = 10
    time_bin_centers = np.arange(T_seconds) + 0.5
    drift = drift_speed * (time_bin_centers - T_seconds / 2)
    motion_est = mu.get_motion_estimate(drift, time_bin_centers_s=time_bin_centers)

    # this is the old impl of registered_geometry
    pitch = drift_util.get_pitch(geom)
    downward_drift = max(0, motion_est.displacement.max())
    upward_drift = max(0, -motion_est.displacement.min())
    assert upward_drift >= 0
    assert downward_drift >= 0
    pitches_pad_up = int(np.ceil(upward_drift / pitch))
    pitches_pad_down = int(np.ceil(downward_drift / pitch))
    min_distance = pdist(geom, metric="sqeuclidean").min() / 2
    unique_shifted_positions = list(geom)
    for shift in range(-pitches_pad_down, pitches_pad_up + 1):
        shifted_geom = geom + [0, pitch * shift]
        for site in shifted_geom:
            if np.square(unique_shifted_positions - site).sum(dim=1).min() > min_distance:
                unique_shifted_positions.append(site)
    unique_shifted_positions = np.array(unique_shifted_positions)
    registered_geom0 = unique_shifted_positions[np.lexsort(unique_shifted_positions.T)]
    assert len(np.unique(registered_geom0, axis=0)) == len(registered_geom0)

    registered_geom1 = drift_util.registered_geometry(geom, motion_est=motion_est)
    assert len(np.unique(registered_geom1, axis=0)) == len(registered_geom1)
    assert np.array_equal(registered_geom0, registered_geom1)


def _check_chans_neighbs_nids_consistent(chans, neighbs, nids):
    assert nids.max() == len(neighbs) - 1
    nids_unique, nids_first_ix = np.unique(nids, return_index=True)
    assert np.array_equal(nids_unique, np.arange(len(neighbs)))
    for j in range(len(neighbs)):
        # first = np.flatnonzero(nids == j)[0]
        first = nids_first_ix[j]
        assert nids[first] == j
        assert np.array_equal(chans[first], neighbs[j])


@pytest.mark.parametrize("geom_ix", range(6))
@pytest.mark.parametrize("drift_speed", [0, 10, 100, -10, -132])
@pytest.mark.parametrize("radius", [0, 10, 35])
def test_stable_channels(example_geoms, geom_ix, drift_speed, radius):
    geom = example_geoms[geom_ix]
    pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=np.nan)
    nc = len(geom)
    T_seconds = 100
    n_spikes = 2048
    rg = np.random.default_rng(0)
    ci = waveform_util.make_channel_index(geom, radius)
    max_distance = pdist(geom).min() / 2

    # create fake motion estimate
    time_bin_centers = np.arange(T_seconds) + 0.5
    drift = drift_speed * (time_bin_centers - T_seconds / 2)
    motion_est = mu.get_motion_estimate(drift, time_bin_centers_s=time_bin_centers)

    # pick random main channels and times
    drifted_chans = np.arange(n_spikes) % nc
    times = rg.uniform(0, T_seconds, size=n_spikes)
    drifted_pos = geom[drifted_chans]
    drifted_depths = drifted_pos[:, 1]
    shifts = motion_est.disp_at_s(times, drifted_depths)
    reg_depths = motion_est.correct_s(times, drifted_depths)
    assert np.isclose(reg_depths, drifted_depths - shifts).all()
    reg_pos = np.c_[drifted_pos[:, 0], reg_depths]

    # registered geometry -- kdt since we'll do lots of nn queries
    rgeom = drift_util.registered_geometry(geom, motion_est)
    kdt = KDTree(rgeom)

    # original (drifted) and registered channel neighborhood positions
    drifted_neighbs = ci[drifted_chans]
    assert drifted_neighbs.min() == 0
    orig_chan_pos = pgeom[drifted_neighbs]
    assert orig_chan_pos.shape == (n_spikes, ci.shape[1], 2)
    reg_chan_pos = orig_chan_pos.copy()
    reg_chan_pos[:, :, 1] -= shifts[:, None]

    # get pitch shifts and check that they agree with the drifted channel position
    n_pitches_shift = drift_util.get_spike_pitch_shifts(
        drifted_depths, geom, reg_depths
    )
    pitch = drift_util.get_pitch(geom)
    pitch_shifts = n_pitches_shift * pitch
    pitch_reg_depths = drifted_depths + pitch_shifts
    assert ((np.abs(reg_depths - pitch_reg_depths) // pitch) == 0).all()
    pitch_reg_pos = np.c_[drifted_pos[:, 0], pitch_reg_depths]
    d_pitch, i_pitch = kdt.query(
        pitch_reg_pos, distance_upper_bound=max_distance, workers=-1
    )
    assert (i_pitch < kdt.n).all()
    i_pitch: npt.NDArray[np.intp] = i_pitch

    # pitch-registered channel positions
    pitch_reg_chan_pos = orig_chan_pos.copy()
    pitch_reg_chan_pos[:, :, 1] += pitch_shifts[:, None]

    # -- different ways of getting the registered neighborhoods
    # 1 compute target channels directly
    ii, jj = np.nonzero(drifted_neighbs < len(geom))
    _, chans_1_ = kdt.query(
        pitch_reg_chan_pos[ii, jj], distance_upper_bound=max_distance, workers=-1
    )
    assert (chans_1_ < kdt.n).all()
    chans_1 = np.full_like(drifted_neighbs, kdt.n)
    chans_1[ii, jj] = chans_1_
    assert (chans_1 == i_pitch[:, None]).any(1).all()

    # 2 use static_channel_neighborhoods fn without precomputed uniq
    chans_2, neighbs_2, nids_2 = drift_util.static_channel_neighborhoods(
        geom,
        drifted_chans,
        ci,
        registered_geom=rgeom,
        target_kdtree=kdt,
        n_pitches_shift=n_pitches_shift,
        workers=-1,
    )
    assert chans_2.shape == chans_1.shape
    assert np.array_equal(chans_1, chans_2)
    _check_chans_neighbs_nids_consistent(chans_2, neighbs_2, nids_2)

    # 3 with precomputed uniq
    cs = np.c_[drifted_chans, n_pitches_shift]
    cs_uniq, cs_ix, cs_inv = np.unique(
        cs, axis=0, return_inverse=True, return_index=True
    )
    assert cs_inv.shape == (n_spikes,)
    chans_3, neighbs_3, nids_3 = drift_util.static_channel_neighborhoods(
        geom,
        drifted_chans,
        ci,
        registered_geom=rgeom,
        target_kdtree=kdt,
        n_pitches_shift=n_pitches_shift,
        uniq_channels_and_shifts=cs_uniq,
        uniq_inv=cs_inv,
        workers=-1,
    )
    assert np.array_equal(chans_1, chans_3)
    assert np.array_equal(neighbs_2, neighbs_3)
    assert np.array_equal(nids_2, nids_3)
    _check_chans_neighbs_nids_consistent(chans_3, neighbs_3, nids_3)

    # -- different ways of getting the neighborhood ids
    # 1 from precomputed uniq ixs
    neighborhoods_1 = chans_1[cs_ix]
    neighborhood_ids_1 = cs_inv
    static_chans_1 = neighborhoods_1[neighborhood_ids_1]
    assert np.array_equal(static_chans_1, chans_1)
    _check_chans_neighbs_nids_consistent(static_chans_1, neighborhoods_1, neighborhood_ids_1)

    # 2 from chans uniq
    neighborhoods_2, neighborhood_ids_2 = np.unique(
        chans_1, axis=0, return_inverse=True
    )
    static_chans_2 = neighborhoods_2[neighborhood_ids_2]
    # test that (1) is a version of (2) with redundancy
    assert neighborhoods_1.shape[0] >= neighborhoods_2.shape[0]
    for id1, neighb1 in enumerate(neighborhoods_1):
        assert (neighborhoods_2 == neighb1).all(1).sum() == 1
    assert np.array_equal(static_chans_2, chans_1)
    _check_chans_neighbs_nids_consistent(static_chans_2, neighborhoods_2, neighborhood_ids_2)

    # 3 deduplicated precomputed uniq
    neighborhoods_3, n1_inv = np.unique(neighborhoods_1, axis=0, return_inverse=True)
    neighborhood_ids_3 = n1_inv[neighborhood_ids_1]
    static_chans_3 = neighborhoods_3[neighborhood_ids_3]
    assert np.array_equal(neighborhoods_2, neighborhoods_3)
    assert np.array_equal(neighborhood_ids_2, neighborhood_ids_3)
    assert np.array_equal(static_chans_3, chans_1)

    # 4 from get_stable_channels
    static_chans_4, neighborhoods_4, neighborhood_ids_4, *_ = (
        drift_util.get_stable_channels(geom, drifted_chans, ci, rgeom, n_pitches_shift)
    )
    assert _ == [None, None, None]
    assert np.array_equal(neighborhoods_2, neighborhoods_4)
    assert np.array_equal(static_chans_1, static_chans_4)
    assert np.array_equal(neighborhood_ids_2, neighborhood_ids_4)

    # // check that these equal neighbs_{2,3}, nids_{2,3} from above
    assert np.array_equal(neighbs_2, neighborhoods_2)
    assert np.array_equal(neighbs_3, neighborhoods_2)
    assert np.array_equal(nids_2, neighborhood_ids_2)
    assert np.array_equal(nids_3, neighborhood_ids_2)


if __name__ == "__main__":
    test_shifted_waveforms()
