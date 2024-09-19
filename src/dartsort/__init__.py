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
from .util.waveform_util import make_channel_index
from .cluster import merge, postprocess

__version__ = importlib.metadata.version("dartsort")
