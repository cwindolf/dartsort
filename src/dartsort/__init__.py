import importlib.metadata

from . import detect, util, cluster
from .cluster import cluster_util, density, kmeans, merge, postprocess, ppcalib
from .cluster.gaussian_mixture import GaussianUnit, SpikeMixtureModel
from .cluster.initial import initial_clustering
from .cluster.stable_features import (
    SpikeFeatures,
    SpikeNeighborhoods,
    StableSpikeDataset,
)
from .cluster.refine import refine_clustering
from .config import *
from .localize.localize_util import (
    localize_amplitude_vectors,
    localize_hdf5,
    localize_waveforms,
)
from .main import (
    ObjectiveUpdateTemplateMatchingPeeler,
    SubtractionPeeler,
    check_recording,
    dartsort,
    estimate_motion,
    match,
    run_peeler,
    subtract,
    run_dev_tasks,
)
from .peel.grab import GrabAndFeaturize
from .templates import TemplateData
from .transform import WaveformPipeline
from .util import data_util, noise_util, spiketorch, waveform_util
from .util.analysis import DARTsortAnalysis
from .util.comparison import DARTsortGroundTruthComparison
from .util.data_util import DARTsortSorting
from .util.drift_util import registered_geometry
from .util.noise_util import EmbeddedNoise
from .util.waveform_util import make_channel_index

__version__ = importlib.metadata.version("dartsort")
