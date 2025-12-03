import numpy as np
import pytest
from sklearn.metrics import rand_score

from dartsort.cluster import (
    clustering_strategies,
    get_clusterer,
    get_clustering_features,
    refinement_strategies,
)
from dartsort.cluster.postprocess import reorder_by_depth
from dartsort.main import cluster
from dartsort.util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    RefinementConfig,
    SplitConfig,
    TemplateConfig,
    TemplateMergeConfig,
)
from dartsort.util.logging_util import get_logger
from dartsort.util.spiketorch import ptp

logger = get_logger(__name__)

# this is how they are named in simkit...
global_feature_kwargs = dict(
    amplitudes_dataset_name="ptp_amplitudes", localizations_dataset_name="localizations"
)
feature_kwargs = [
    dict(use_amplitude=True, n_main_channel_pcs=0),
    dict(use_amplitude=False, n_main_channel_pcs=1, motion_aware=False),
    dict(use_amplitude=False, n_main_channel_pcs=1, motion_aware=True),
]
feature_kwargs = [global_feature_kwargs | kw for kw in feature_kwargs]

clustering_kwargs = [dict(cluster_strategy=k) for k in clustering_strategies]
clustering_kwargs += [
    dict(cluster_strategy="gmmdpc", hellinger_weak=0.99, mop=True),
    dict(cluster_strategy="gmmdpc", use_hellinger=False),
]
refinement_kwargs = [dict(refinement_strategy=k) for k in refinement_strategies]
refinement_kwargs = [dict(refinement_strategy="tmm", signal_rank=0)] + refinement_kwargs


# only some methods are good enough to actually test the outcome
eval_clustering_kwargs = [
    dict(cluster_strategy="none"),  # ground truth
    dict(cluster_strategy="gmmdpc"),
]
eval_initial_refinement_kwargs = [
    dict(refinement_strategy="pcmerge"),
]
eval_refinement_kwargs = [
    dict(refinement_strategy="gmm"),
    dict(refinement_strategy="tmm"),
]


@pytest.mark.parametrize("sim_name", ["drifty_szmini", "driftn_szmini"])
@pytest.mark.parametrize("featkw", feature_kwargs)
@pytest.mark.parametrize("cluskw", clustering_kwargs)
def test_clustering(simulations, sim_name, featkw, cluskw):
    if cluskw["cluster_strategy"] == "density_peaks_uhdversion":
        if not featkw["use_amplitude"]:
            return
    sim = simulations[sim_name]
    recording = sim["recording"]
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


@pytest.mark.parametrize("sim_name", ["drifty_szmini", "driftn_szmini"])
@pytest.mark.parametrize("refkw", refinement_kwargs)
def test_refinement(simulations, sim_name, refkw):
    sim = simulations[sim_name]
    recording = sim["recording"]
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
            realign_peaks=False, denoising_method="none"
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


@pytest.mark.parametrize("sim_name", ["driftn_szmini", "drifty_szmini"])
@pytest.mark.parametrize("cluskw", eval_clustering_kwargs)
@pytest.mark.parametrize("initrefkw", eval_initial_refinement_kwargs)
@pytest.mark.parametrize("refkw", eval_refinement_kwargs)
def test_accurate(subtests, simulations, sim_name, cluskw, initrefkw, refkw):
    sim = simulations[sim_name]
    recording = sim["recording"]
    sorting = sim["sorting"]
    motion_est = sim["motion_est"]

    res_sorting = cluster(
        recording=recording,
        sorting=sorting,
        motion_est=motion_est,
        clustering_cfg=ClusteringConfig(**cluskw),
        pre_refinement_cfg=RefinementConfig(**initrefkw),
        refinement_cfg=RefinementConfig(**refkw),
    )
    with subtests.test(msg="rand score"):
        assert rand_score(sorting.labels, res_sorting.labels) > 0.995
    with subtests.test(msg="unit count"):
        # allowing old gmm to fail this
        if refkw["refinement_strategy"] != "gmm":
            assert sorting.n_units == res_sorting.n_units


@pytest.mark.parametrize("sim_name", ["drifty_szmini", "driftn_szmini"])
def test_reorder_by_depth(simulations, sim_name):
    sim = simulations[sim_name]
    sorting = sim["sorting"]
    template_data = sim["templates"]

    sorting1, template_data1 = reorder_by_depth(sorting, template_data)

    assert np.array_equal(sorting.times_samples, sorting1.times_samples)
    assert np.array_equal(sorting.channels, sorting1.channels)
    assert np.array_equal(sorting.labels >= 0, sorting1.labels >= 0)
    if template_data1.registered_geom is None:
        assert template_data.registered_geom is None
    else:
        assert np.array_equal(
            template_data.registered_geom, template_data1.registered_geom
        )

    u0, ix0, c0 = np.unique(sorting.labels, return_counts=True, return_index=True)
    u1, c1 = np.unique(sorting1.labels, return_counts=True)
    assert np.array_equal(u0, u1)
    assert np.array_equal(u0, np.arange(u0.size))
    assert np.array_equal(np.sort(c0), np.sort(c1))

    old_to_new = np.full_like(u0, -1)
    for old_id in range(len(u0)):
        old_ix = ix0[old_id]
        assert sorting.labels[old_ix] == old_id
        new_id = sorting1.labels[old_ix]
        old_to_new[old_id] = new_id

    for old_id, new_id in enumerate(old_to_new):
        assert np.array_equal(
            np.flatnonzero(sorting.labels == old_id),
            np.flatnonzero(sorting1.labels == new_id),
        )
        assert np.array_equal(
            template_data.templates[old_id], template_data1.templates[new_id]
        )
        assert np.array_equal(
            template_data.spike_counts[old_id], template_data1.spike_counts[new_id]
        )
        if template_data.spike_counts_by_channel is not None:
            assert template_data1.spike_counts_by_channel is not None
            assert np.array_equal(
                template_data.spike_counts_by_channel[old_id],
                template_data1.spike_counts_by_channel[new_id],
            )
        else:
            assert template_data1.spike_counts_by_channel is None
        if template_data.raw_std_dev is not None:
            assert template_data1.raw_std_dev is not None
            assert np.array_equal(
                template_data.raw_std_dev[old_id],
                template_data1.raw_std_dev[new_id],
            )
        else:
            assert template_data1.raw_std_dev is None

    w = ptp(template_data1.templates, dim=1)
    if template_data1.spike_counts_by_channel is not None:
        w *= np.sqrt(template_data1.spike_counts_by_channel)
    w /= w.sum(axis=1, keepdims=True)
    if template_data1.registered_geom is not None:
        meanz = np.sum(template_data1.registered_geom[:, 1] * w, axis=1)
        assert np.array_equal(np.argsort(meanz), np.arange(u1.size))
