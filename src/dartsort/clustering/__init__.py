from .clustering_features import SimpleMatrixFeatures, StableWaveformFeatures
from .clustering import get_clusterer, clustering_strategies, refinement_strategies, TMMRefinement
from .agglomerate import deduplicate_spikes
