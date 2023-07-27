"""
Test conversions among DARTsortSorting, NumpySorting, and HDF5 formats.
"""
import tempfile
from pathlib import Path

import h5py
import numpy as np
from dartsort.util.data_util import DARTsortSorting, check_recording
from spikeinterface import NumpyRecording
import pytest

times = np.arange(0, 1000, 10)
rng = np.random.default_rng(0)
channels = rng.integers(0, 384, size=(100,))
labels = rng.integers(0, 10, size=(100,))


def test_to_numpy_sorting():
    """
    Test conversion to SI NumpySorting
    """
    dsorting = DARTsortSorting(times, channels, labels=labels)
    npsorting = dsorting.to_numpy_sorting()

    si_times, si_labels = npsorting.get_all_spike_trains()[0]
    assert np.array_equal(si_times, times)
    assert np.array_equal(si_labels, labels)
    assert np.array_equal(npsorting.get_unit_ids(), np.arange(0, 10))


def test_from_peeling():
    """
    Test initialization from HDF5 stored peeling
    """
    with tempfile.TemporaryDirectory() as tempdir:
        peeling_h5 = Path(tempdir) / "test.h5"
        with h5py.File(peeling_h5, "w") as h:
            h.create_dataset("sampling_frequency", data=1)
            h.create_dataset("times", data=times)
            h.create_dataset("channels", data=channels)
            h.create_dataset("labels", data=labels)

        dsorting = DARTsortSorting.from_peeling_hdf5(peeling_h5)
        assert np.array_equal(dsorting.times, times)
        assert np.array_equal(dsorting.channels, channels)
        assert np.array_equal(dsorting.labels, labels)

def test_check_recording():
    """
    Test spike rate and data range sanity 
    checks performed by this method.
    """

    x = rng.normal(size=(5*30000, 384)).astype(np.float32) * 1e4
    rec = NumpyRecording(x, sampling_frequency=30000)

    with pytest.warns(Warning) as warninfo:
        check_recording(rec)
    warnings = {(w.category, w.message.args[0][:12]) for w in warninfo}
    expected = {
                    (RuntimeWarning, "Data range e"),
                    (RuntimeWarning, "Average spik")
                }
    
    assert warnings == expected


