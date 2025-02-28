"""
Test conversions among DARTsortSorting, NumpySorting, and HDF5 formats.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from dartsort.util.data_util import DARTsortSorting, check_recording
from spikeinterface import NumpyRecording

times_samples = np.arange(0, 1000, 10)


def test_to_numpy_sorting():
    """Test conversion to SI NumpySorting"""
    rg = np.random.default_rng(0)
    channels = rg.integers(0, 384, size=(100,))
    labels = rg.integers(0, 10, size=(100,))
    dsorting = DARTsortSorting(times_samples, channels, labels=labels)
    npsorting = dsorting.to_numpy_sorting()

    si_spiketrain = npsorting.to_spike_vector()
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
        assert np.array_equal(dsorting.labels, labels)


def test_check_recording():
    """Test spike rate and data range sanity checks performed by this method."""
    rg = np.random.default_rng(0)
    x = rg.normal(size=(5 * 30000, 384)).astype(np.float32) * 1e4
    rec = NumpyRecording(x, sampling_frequency=30000)
    rec.set_dummy_probe_from_locations(np.c_[np.zeros(384), 100 * np.arange(384)])

    with pytest.warns(Warning) as warninfo:
        check_recording(rec)
    warnings = {(w.category, w.message.args[0][:11]) for w in warninfo}
    expected = {
        (RuntimeWarning, "Detected 76"),
        (RuntimeWarning, "Recording v"),
    }

    assert warnings == expected
