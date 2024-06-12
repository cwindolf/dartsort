from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .. import config
from ..cluster import merge
from ..templates import templates
from ..templates.get_templates import fit_tsvd
from ..util import data_util, spike_features, analysis
from . import analysis_plots, scatterplots, layout, unit

basic_template_config = config.TemplateConfig(
    realign_peaks=False, superres_templates=False
)


# -- over_time_summary stuff


class GMMPlot(layout.BasePlot):
    width = 1
    height = 1
    kind = "gmm"

    def draw(self, panel, sorting_analysis, unit_id):
        raise NotImplementedError

class TrainEmbedsPlot(GMMPlot):
    kind = "embeds"
    width = 3
    height = 1.5
    pass

class FullEmbedsPlot(GMMPlot):
    kind = "embeds"
    width = 3
    height = 1.5
    pass

class WaveformsPlot(GMMPlot):
    width = 5
    height = 5
    pass