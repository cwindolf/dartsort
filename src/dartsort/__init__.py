import importlib.metadata

from . import detect, util
from .clustering import (
    SimpleMatrixFeatures,
    StableWaveformFeatures,
    cluster_util,
    clustering_strategies,
    density,
    get_clusterer,
    kmeans,
    merge,
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
    DARTsortResult,
    check_recording,
    cluster,
    dartsort,
    grab,
    logger,
    match,
    run_peeler,
    subtract,
    threshold,
)
from .peel.grab import GrabAndFeaturize
from .peel.matching import MatchingTemplates
from .templates import (
    TemplateData,
    estimate_template_library,
    postprocess_util,
    realign,
)
from .transform import WaveformPipeline
from .util import (
    data_util,
    logging_util,
    noise_util,
    spikeio,
    spiketorch,
    waveform_util,
)
from .util.data_util import (
    DARTsortSorting,
    check_recording,
    get_featurization_pipeline,
    get_tpca,
    load,
    load_stored_tsvd,
)
from .util.drift_util import registered_geometry
from .util.internal_config import *
from .util.interpolation_util import (
    FromFullProbeInterpolator,
    StableFeaturesInterpolator,
    ToFullProbeInterpolator,
    pad_geom,
)
from .util.job_util import ensure_computation_config, set_global_computation_config
from .util.logging_util import (
    DARTSORTDEBUG,
    DARTSORTVERBOSE,
    DARTsortLogger,
    get_logger,
    set_log_level,
)
from .util.motion import MotionInfo, get_motion_info, try_load_motion_info
from .util.noise_util import EmbeddedNoise
from .util.preprocess_util import preprocess
from .util.py_util import databag, ensure_path
from .util.waveform_util import full_channel_index, make_channel_index

__version__ = importlib.metadata.version("dartsort")
