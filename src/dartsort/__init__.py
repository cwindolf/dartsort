import importlib.metadata

from . import util
from .util import logging_util
from .util.internal_config import *
from .util import data_util, noise_util, spiketorch, waveform_util
from .util.data_util import DARTsortSorting, get_featurization_pipeline
from .util.drift_util import registered_geometry
from .util.noise_util import EmbeddedNoise
from .util.waveform_util import make_channel_index
from .util.py_util import resolve_path

from . import detect

from .localize.localize_util import (
    localize_amplitude_vectors,
    localize_hdf5,
    localize_waveforms,
)


from .peel.grab import GrabAndFeaturize

from .templates import TemplateData, realign_sorting, realign_templates

from .transform import WaveformPipeline

from .cluster import cluster_util, density, kmeans, merge, postprocess, ppcalib
from .cluster import (
    get_clusterer,
    get_clustering_features,
    clustering_strategies,
    refinement_strategies,
)

from .evaluate import hybrid_util, config_grid, simkit, simlib
from .evaluate.analysis import DARTsortAnalysis
from .evaluate.comparison import DARTsortGroundTruthComparison, DARTsortGTVersus

from .config import *

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
    cluster,
    universal_match,
)

__version__ = importlib.metadata.version("dartsort")
