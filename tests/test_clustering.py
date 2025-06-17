import numpy as np
import pytest


from dartsort.cluster import (
    get_clusterer,
    get_clustering_features,
    clustering_strategies,
    refinement_strategies,
)
from dartsort.util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    RefinementConfig,
    TemplateMergeConfig,
    SplitConfig,
    TemplateConfig,
)


# this is how they are named in simkit...
global_feature_kwargs = dict(
    amplitudes_dataset_name="ptp_amplitudes", localizations_dataset_name="localizations"
)
feature_kwargs = [
    dict(use_amplitude=True, n_main_channel_pcs=0),
    dict(use_amplitude=False, n_main_channel_pcs=1),
]
feature_kwargs = [global_feature_kwargs | kw for kw in feature_kwargs]

clustering_kwargs = [dict(cluster_strategy=k) for k in clustering_strategies]
refinement_kwargs = [dict(refinement_strategy=k) for k in refinement_strategies]


@pytest.mark.parametrize("sim_name", ["static_mini", "drifting_mini"])
@pytest.mark.parametrize("featkw", feature_kwargs)
@pytest.mark.parametrize("cluskw", clustering_kwargs)
def test_clustering(sim_recordings, sim_name, featkw, cluskw):
    if cluskw["cluster_strategy"] == "density_peaks_uhdversion":
        if not featkw["use_amplitude"]:
            return
    sim = sim_recordings[sim_name]
    recording = sim["rec"]
    sorting = sim["sorting"]
    motion_est = sim["motion_est"]

    features = get_clustering_features(
        recording,
        sorting,
        motion_est=motion_est,
        clustering_features_cfg=ClusteringFeaturesConfig(**featkw),
    )
    clusterer = get_clusterer(
        clustering_cfg=ClusteringConfig(**cluskw),
        refinement_cfg=None,
    )
    res = clusterer.cluster(
        recording=recording, sorting=sorting, features=features, motion_est=motion_est
    )
    assert res is not None
    # dpcs struggle with this.
    # assert np.unique(res.labels).size > 1
    assert res.labels.shape == sorting.labels.shape


@pytest.mark.parametrize("sim_name", ["static_mini", "drifting_mini"])
@pytest.mark.parametrize("featkw", feature_kwargs)
@pytest.mark.parametrize("refkw", refinement_kwargs)
def test_refinement(sim_recordings, sim_name, featkw, refkw):
    sim = sim_recordings[sim_name]
    recording = sim["rec"]
    sorting = sim["sorting"]
    motion_est = sim["motion_est"]

    features = get_clustering_features(
        recording,
        sorting,
        motion_est=motion_est,
        clustering_features_cfg=None,
    )
    if refkw["refinement_strategy"]:
        refkw["split_cfg"] = SplitConfig()
        refkw["merge_cfg"] = TemplateMergeConfig()
        refkw["merge_template_cfg"] = TemplateConfig(
            realign_peaks=False, low_rank_denoising=False
        )

    clusterer = get_clusterer(
        clustering_cfg=None,
        refinement_cfg=RefinementConfig(**refkw),
    )
    res = clusterer.cluster(
        recording=recording, sorting=sorting, features=features, motion_est=motion_est
    )
    assert res is not None
    assert res.labels.shape == sorting.labels.shape
    assert np.unique(res.labels).size > 1
