from tempfile import TemporaryDirectory

import numpy as np
import pytest

from dartsort import (
    ClusteringFeaturesConfig,
    FeaturizationConfig,
    MatchingConfig,
    RefinementConfig,
    WhiteningConfig,
    match,
)


@pytest.mark.parametrize("drift", ["n", "y"])
def test_mixture_classifier(tmp_path, simulations, drift):
    # basis non explosion test, running gt template matching on mini sim
    sim = simulations[f"drift{drift}_szmini"]
    templates = sim["templates"]
    rec = sim["recording"]
    motion = sim["motion"]

    # this config will do as little as possible while still running the code
    with TemporaryDirectory(dir=tmp_path, ignore_cleanup_errors=True) as tdir:
        st = match(
            output_dir=tdir,
            recording=rec,
            motion=motion,
            template_data=templates,
            matching_cfg=MatchingConfig(
                whitening=WhiteningConfig(),
            ),
            featurization_cfg=FeaturizationConfig(
                save_input_tpca_projs=False,
                compute_input_tpca_projs_regardless=True,
                do_localization=False,
                use_gmm_classifier=True,
                pre_gmm_refinement_cfgs=[],
                gmm_clustering_features_cfg=ClusteringFeaturesConfig(),
                gmm_refinement_cfg=RefinementConfig(mixture_steps=()),
            ),
        )
        assert len(st) > 0
        assert hasattr(st, "labels")
        assert st.labels is not None
        assert hasattr(st, "template_inds")
        assert hasattr(st, "gmm_candidates")
        assert hasattr(st, "gmm_log_liks")
        assert hasattr(st, "gmm_responsibilities")
        assert np.isin(np.unique(st.labels), np.unique(st.template_inds)).all()
        assert (st.gmm_candidates >= 0).sum() > len(st) / 10
        np.testing.assert_array_equal(
            st.gmm_candidates >= 0, np.isfinite(st.gmm_log_liks[:, :-1])
        )
        assert not np.isposinf(st.gmm_log_liks).any()
        assert not np.isnan(st.gmm_log_liks).any()
        assert not np.isnan(st.gmm_responsibilities).any()
        assert np.all(
            np.logical_or(
                st.gmm_candidates[:, 0] < 0, st.gmm_candidates[:, 0] == st.labels
            )
        )
        # shouldn't be too diff from matching result
        assert (st.gmm_candidates[:, 0] == st.template_inds).mean() > 0.95
