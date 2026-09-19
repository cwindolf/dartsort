"""Module where "peeler" classes live, which can run spike detection, extraction, and featurization pipelines."""

from .grab import GrabAndFeaturize
from .matching import ObjectiveUpdateTemplateMatchingPeeler
from .matching_util import *
from .reduction_template import ReductionTemplateData, TemplateReduction
from .shaver import Shaver
from .subtract import SubtractionPeeler
from .threshold import Threshold
