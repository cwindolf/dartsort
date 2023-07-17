from dartsort.util.data_util import DARTsortSorting
import spikeinterface.core as sc
import numpy as np
import h5py
import tempfile
from pathlib import Path

"""
Test conversions among DARTsortSorting, NumpySorting, and HDF5 formats.
"""

times = np.arange(0,1000,10)
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
    assert np.array_equal(npsorting.get_unit_ids(), np.arange(0,10))

def test_from_peeling():
    """
    Test initialization from HDF5 stored peeling
    """
    with tempfile.TemporaryDirectory() as tempdir:
        peeling_h5 = Path(tempdir) / "test.h5"
        with h5py.File(peeling_h5, "w") as h:
            h.create_dataset("times", data=times)
            h.create_dataset("channels", data=channels)
            h.create_dataset("labels", data=labels)

        dsorting = DARTsortSorting.from_peeling_hdf5(peeling_h5)
        assert np.array_equal(dsorting.times, times)
        assert np.array_equal(dsorting.channels, channels)
        assert np.array_equal(dsorting.labels, labels)

