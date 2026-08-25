"""
Test conversions among DARTsortSorting, NumpySorting, and HDF5 formats.
"""

import tempfile
from pathlib import Path
from typing import cast

import dredge.motion_util as mu
import h5py
import numpy as np
import pytest
from spikeinterface import NumpyRecording

from dartsort import MotionInfo
from dartsort.util import data_util, waveform_util
from dartsort.util.data_util import DARTsortSorting, check_recording

times_samples = np.arange(0, 1000, 10)


def test_to_numpy_sorting():
    """Test conversion to SI NumpySorting"""
    rg = np.random.default_rng(0)
    channels = rg.integers(0, 384, size=(100,))
    labels = rg.integers(0, 10, size=(100,))
    dsorting = DARTsortSorting(
        times_samples=times_samples, channels=channels, labels=labels
    )
    npsorting = dsorting.to_numpy_sorting()

    si_spiketrain = cast(np.recarray, npsorting.to_spike_vector())
    si_times = si_spiketrain["sample_index"]
    si_labels = si_spiketrain["unit_index"]
    assert np.array_equal(si_times, times_samples)
    assert np.array_equal(si_labels, labels)
    assert np.array_equal(npsorting.get_unit_ids(), np.arange(0, 10))


def test_from_peeling():
    """Test initialization from HDF5 stored peeling"""
    rg = np.random.default_rng(0)
    channels = rg.integers(0, 384, size=(100,))
    labels = rg.integers(0, 10, size=(100,))
    with tempfile.TemporaryDirectory() as tempdir:
        peeling_h5 = Path(tempdir) / "test.h5"
        with h5py.File(peeling_h5, "w") as h:
            h.create_dataset("sampling_frequency", data=1)
            h.create_dataset("times_samples", data=times_samples)
            h.create_dataset("channels", data=channels)
            h.create_dataset("labels", data=labels)

        dsorting = DARTsortSorting.from_peeling_hdf5(peeling_h5)
        assert np.array_equal(dsorting.times_samples, times_samples)
        assert np.array_equal(dsorting.channels, channels)
        assert dsorting.labels is not None
        assert np.array_equal(dsorting.labels, labels)


def test_check_recording():
    """Test spike rate and data range sanity checks performed by this method."""
    rg = np.random.default_rng(0)
    x = rg.normal(size=(5 * 30000, 384)).astype(np.float32) * 1e4
    rec = NumpyRecording(x, sampling_frequency=30000)
    rec.set_dummy_probe_from_locations(np.c_[np.zeros(384), 100 * np.arange(384)])

    with pytest.warns(Warning) as warninfo:
        check_recording(rec, copy_flag=False)
    warnings = {(w.category, w.message.args[0][:11]) for w in warninfo}  # type: ignore
    expected = {
        (RuntimeWarning, "Detected 53"),
        (RuntimeWarning, "Recording s"),
        (RuntimeWarning, "Recording v"),
        (RuntimeWarning, "Your (prepr"),
    }

    assert warnings == expected


def _amp_vec_sorting(tempdir, geom, channel_index, times_seconds, channels, amp_vecs):
    """Write a peeling-style h5 holding amplitude vectors and load it."""
    peeling_h5 = Path(tempdir) / "ampvecs.h5"
    with h5py.File(peeling_h5, "w") as h:
        h.create_dataset("sampling_frequency", data=1000.0)
        h.create_dataset("times_samples", data=(1000 * times_seconds).astype(np.int64))
        h.create_dataset("times_seconds", data=times_seconds)
        h.create_dataset("channels", data=channels)
        h.create_dataset("geom", data=geom)
        h.create_dataset("channel_index", data=channel_index)
        h.create_dataset("amplitude_vectors", data=amp_vecs)
    return DARTsortSorting.from_peeling_hdf5(peeling_h5)


@pytest.mark.parametrize("drift_speed", [0.0, 1.0])
def test_interpolate_main_channel_amplitudes(drift_speed):
    rg = np.random.default_rng(0)
    geom = np.c_[np.tile([0.0, 20.0], 12), np.repeat(20.0 * np.arange(12), 2)]
    nc = len(geom)
    pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=np.nan)
    ci = waveform_util.make_channel_index(geom, 45.0)
    T_seconds = 100.0
    n_spikes = 256

    if drift_speed:
        time_bin_centers = np.arange(T_seconds) + 0.5
        motion = MotionInfo.from_motion_est(
            geom=geom,
            dredge_motion_est=mu.get_motion_estimate(
                drift_speed * (time_bin_centers - T_seconds / 2),
                time_bin_centers_s=time_bin_centers,
            ),
        )
    else:
        motion = MotionInfo.from_motion_est(geom=geom)

    times_seconds = np.sort(rg.uniform(0, T_seconds, size=n_spikes))
    channels = rg.integers(0, nc, size=n_spikes)
    neighb_pos = pgeom[ci[channels]].astype(np.float32)

    coefs = rg.normal(size=2).astype(np.float32)
    offset = np.float32(10.0)
    amp_vecs = offset + neighb_pos @ coefs

    with tempfile.TemporaryDirectory() as tempdir:
        sorting = _amp_vec_sorting(
            tempdir, geom, ci, times_seconds, channels, amp_vecs
        )
        data_util.interpolate_main_channel_amplitudes(
            sorting, motion, show_progress=False
        )
        with h5py.File(sorting.parent_h5_path, "r") as h5:
            amps = cast(h5py.Dataset, h5["motion_corrected_amplitudes"])[:]

    assert amps.shape == (n_spikes,)
    assert np.isfinite(amps).all()

    shifts, n_pitches_shift = motion.pitch_shifts(sorting=sorting)
    target_pos = geom[channels].copy()
    target_pos[:, 1] += n_pitches_shift * motion.pitch - shifts
    assert np.isclose(amps, offset + target_pos @ coefs, rtol=1e-4).all()

    if not drift_speed:
        assert (n_pitches_shift == 0).all()
        main_ix = np.array([np.flatnonzero(ci[c] == c).item() for c in channels])
        assert np.isclose(
            amps, amp_vecs[np.arange(n_spikes), main_ix], rtol=1e-4
        ).all()
