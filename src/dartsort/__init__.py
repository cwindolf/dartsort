import importlib.metadata

from . import detect, util
from .clustering import (
    cluster_util,
    clustering_strategies,
    density,
    get_clusterer,
    get_clustering_features,
    kmeans,
    merge,
    ppcalib,
    refinement_strategies,
)
from .config import *
from .evaluate import config_grid, hybrid_util, simkit, simlib
from .evaluate.analysis import DARTsortAnalysis
from .evaluate.comparison import DARTsortGroundTruthComparison, DARTsortGTVersus
from .localize.localize_util import (
    localize_amplitude_vectors,
    localize_hdf5,
    localize_waveforms,
)
from .main import (
    ObjectiveUpdateTemplateMatchingPeeler,
    SubtractionPeeler,
    check_recording,
    cluster,
    dartsort,
    estimate_motion,
    grab,
    logger,
    match,
    run_peeler,
    subtract,
    threshold,
    universal_match,
)
from .peel.grab import GrabAndFeaturize
from .templates import (
    TemplateData,
    estimate_template_library,
    postprocess_util,
    realign,
)
from .transform import WaveformPipeline
from .util import data_util, logging_util, noise_util, spiketorch, waveform_util
from .util.data_util import (
    DARTsortSorting,
    get_featurization_pipeline,
    load_h5,
    get_tpca,
    load_stored_tsvd,
    check_recording,
)
from .util.drift_util import registered_geometry
from .util.internal_config import *
from .util.logging_util import (
    DARTSORTDEBUG,
    DARTSORTVERBOSE,
    DARTsortLogger,
    get_logger,
)
from .util.noise_util import EmbeddedNoise
from .util.py_util import resolve_path
from .util.registration_util import try_load_motion_est
from .util.waveform_util import make_channel_index

__version__ = importlib.metadata.version("dartsort")
