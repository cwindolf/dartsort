import numpy as np
import pytest
import torch
from sklearn.decomposition import TruncatedSVD

from dartsort.clustering import SimpleMatrixFeatures, clustering_features
from dartsort.util.internal_config import ClusteringFeaturesConfig
from dartsort.util.job_util import ensure_computation_config

# this is how they are named in simkit...
global_feature_kwargs = dict(
    amplitudes_dataset_name="ptp_amplitudes", localizations_dataset_name="localizations"
)


@pytest.mark.parametrize("sim_name", ["driftn_szmini", "drifty_szmini"])
@pytest.mark.parametrize("motion_aware", [False, True])
def test_multi_channel_pcs(simulations, sim_name, motion_aware):
    sim = simulations[sim_name]
    sorting = sim["sorting"]
    motion = sim["motion"]
    n_pcs = 3

    cfg = ClusteringFeaturesConfig(
        **global_feature_kwargs,  # ty: ignore[invalid-argument-type]
        use_x=False,
        use_z=False,
        use_amplitude=False,
        use_signed_amplitude=False,
        n_main_channel_pcs=0,
        n_multi_channel_pcs=n_pcs,
        motion_aware=motion_aware,
        raise_for_numerics=True,
    )
    features = SimpleMatrixFeatures.from_config(
        sorting=sorting,
        motion=motion,
        clustering_features_cfg=cfg,
        computation_cfg=None,
    )
    assert features.features.shape == (len(sorting), n_pcs)
    assert np.isfinite(features.features).all()

    feats, neighbs, nids, n_target_channels = (
        clustering_features._multi_channel_features(
            sorting=sorting,
            clustering_features_cfg=cfg,
            motion=motion,
            computation_cfg=ensure_computation_config(None),
        )
    )
    pcs = clustering_features._multi_channel_embedding(
        feats=feats,
        neighborhoods=neighbs,
        neighborhood_ids=nids,
        n_target_channels=n_target_channels,
        n_pcs=n_pcs,
    )
    observed = torch.asarray(neighbs) < n_target_channels
    full = observed.all(dim=1)
    # check missing tested
    assert not full.all()
    spike_is_full = full[torch.asarray(nids)]
    assert spike_is_full.any()
    assert torch.isnan(feats[spike_is_full]).sum() == 0
    assert torch.isnan(feats[spike_is_full.logical_not()]).sum() > 0

    # basis is fit only to fully obs, isometry check against tsvd
    x = feats[spike_is_full].reshape(int(spike_is_full.sum()), -1).numpy()
    skpcs = TruncatedSVD(n_components=n_pcs, algorithm="arpack").fit_transform(x)
    assert np.allclose(
        np.linalg.norm(skpcs, axis=1),
        np.linalg.norm(pcs[spike_is_full].numpy(), axis=1),
        rtol=1e-4,
    )

    # check Ilin and Raiko eq 8
    rank = feats.shape[1]
    basis = np.linalg.lstsq(x, pcs[spike_is_full].numpy(), rcond=None)[0]
    basis = basis.reshape(rank, -1, n_pcs)
    (partial,) = spike_is_full.logical_not().nonzero(as_tuple=True)
    for j in partial[:: max(1, len(partial) // 10)].tolist():
        obs = observed[nids[j]].numpy()
        w = basis[:, obs].reshape(-1, n_pcs)
        y = feats[j].numpy()[:, obs].ravel()
        expected = np.linalg.inv(w.T @ w) @ w.T @ y
        assert np.allclose(expected, pcs[j].numpy(), rtol=1e-3, atol=1e-4)
