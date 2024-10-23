import importlib.metadata

from .config import *
from .localize.localize_util import (localize_amplitude_vectors, localize_hdf5,
                                     localize_waveforms)
from .main import (ObjectiveUpdateTemplateMatchingPeeler, SubtractionPeeler,
                   check_recording, cluster, dartsort, match, run_peeler,
                   split_merge, subtract, estimate_motion)
from .peel.grab import GrabAndFeaturize
from .templates import TemplateData
from .transform import WaveformPipeline
from .util.analysis import DARTsortAnalysis
from .util.data_util import DARTsortSorting
from .util.drift_util import registered_geometry
from .util.waveform_util import make_channel_index
from .cluster import merge, postprocess, density
from . import util
from .util import noise_util, data_util, waveform_util, spiketorch
from .util.noise_util import EmbeddedNoise
from .cluster.initial import initial_clustering
from .cluster.stable_features import StableSpikeDataset
from .cluster.gaussian_mixture import SpikeMixtureModel, GaussianUnit

__version__ = importlib.metadata.version("dartsort")
