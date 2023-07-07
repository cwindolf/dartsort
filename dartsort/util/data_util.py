from collections import namedtuple

SpikeDataset = namedtuple("SpikeDataset", ["name", "shape_per_spike", "dtype"])


class SpikeTrain:
    pass


class H5FeatureExtractor:
    def __init__(
        self, hdf5_filename, require_datasets=["times", "amplitude_vectors"]
    ):
        pass
