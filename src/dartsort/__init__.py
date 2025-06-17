import importlib.metadata

from .util import logging_util
from .eval import hybrid_util

from . import detect, util, cluster
from .cluster import cluster_util, density, kmeans, merge, postprocess, ppcalib
from .cluster.gaussian_mixture import GaussianUnit, SpikeMixtureModel
from .cluster import get_clusterer, get_clustering_features, clustering_strategies, refinement_strategies
from .cluster.stable_features import (
    SpikeFeatures,
    SpikeNeighborhoods,
    StableSpikeDataset,
)
from .util.internal_config import *
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
    grab,
    threshold,
    logger,
)
from .util.internal_config import *
from .peel.grab import GrabAndFeaturize
from .templates import TemplateData, realign_sorting, realign_templates
from .transform import WaveformPipeline
from .util import data_util, noise_util, spiketorch, waveform_util
from .eval.analysis import DARTsortAnalysis
from .eval.comparison import DARTsortGroundTruthComparison
from .util.data_util import DARTsortSorting, get_featurization_pipeline
from .util.drift_util import registered_geometry
from .util.noise_util import EmbeddedNoise
from .util.waveform_util import make_channel_index
from .util.py_util import resolve_path

__version__ = importlib.metadata.version("dartsort")
