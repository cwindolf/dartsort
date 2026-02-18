import dataclasses
import pytest
import subprocess
import warnings

import dartsort


@pytest.mark.parametrize("do_motion_estimation", [True])
@pytest.mark.parametrize("sim_size", ["mini"])
def test_fakedata_nonn(tmp_path, sim_size, simulations, do_motion_estimation):
    sim_name = "drifty" if do_motion_estimation else "driftn"
    sim_name = f"{sim_name}_sz{sim_size}"
    sim_recording = simulations[sim_name]["recording"]

    cfg = dartsort.DARTsortInternalConfig(
        initial_detection_cfg=dartsort.SubtractionConfig(
            subtraction_denoising_cfg=dartsort.FeaturizationConfig(
                denoise_only=True, do_nn_denoise=False
            ),
            residnorm_decrease_threshold=16.0,
        ),
        featurization_cfg=dartsort.FeaturizationConfig(n_residual_snips=512),
        motion_estimation_cfg=dartsort.MotionEstimationConfig(
            do_motion_estimation=do_motion_estimation, rigid=True
        ),
        work_in_tmpdir=True,
        save_intermediate_features=True,
        save_intermediate_labels=True,
        matching_iterations=0,
    )
    res = dartsort.dartsort(sim_recording, output_dir=tmp_path, cfg=cfg)
    assert res["sorting"].parent_h5_path.exists()
    assert (tmp_path / "dartsort_sorting.npz").exists()
    assert (tmp_path / "subtraction.h5").exists()
    assert not (tmp_path / "matching1.h5").exists()

    # test the fast-forward thing
    cfg1 = dataclasses.replace(cfg, matching_iterations=1)
    res = dartsort.dartsort(sim_recording, output_dir=tmp_path, cfg=cfg1)
    assert res["sorting"].parent_h5_path.exists()
    assert (tmp_path / "dartsort_sorting.npz").exists()
    assert (tmp_path / "subtraction.h5").exists()
    assert (tmp_path / "matching1.h5").exists()

    # test the fast-forward thing again
    res = dartsort.dartsort(sim_recording, output_dir=tmp_path, cfg=cfg1)
    assert res["sorting"].parent_h5_path.exists()
    assert (tmp_path / "dartsort_sorting.npz").exists()
    assert (tmp_path / "subtraction.h5").exists()
    assert (tmp_path / "matching1.h5").exists()


usual_sdcfg = dartsort.FeaturizationConfig(denoise_only=True)
decollider_sdcfg = dartsort.FeaturizationConfig(
    denoise_only=True,
    do_nn_denoise=True,
    nn_denoiser_class_name="Decollider",
    # not good parameters -- don't want to explode CI
    nn_denoiser_train_epochs=25,
    nn_denoiser_epoch_size=256,
    nn_denoiser_pretrained_path=None,
    nn_denoiser_extra_kwargs=dict(hidden_dims=[512] * 2, batch_size=32),
)


@pytest.mark.parametrize("sdcfg", [usual_sdcfg, decollider_sdcfg])
@pytest.mark.parametrize("sim_size", ["mini"])
def test_fakedata(tmp_path, sim_size, simulations, sdcfg):
    sim_recording = simulations[f"driftn_sz{sim_size}"]["recording"]

    cfg = dartsort.DARTsortInternalConfig(
        initial_detection_cfg=dartsort.SubtractionConfig(
            subtraction_denoising_cfg=sdcfg,
            first_denoiser_thinning=0.0,
        ),
        # test pc based clust
        clustering_features_cfg=dartsort.ClusteringFeaturesConfig(
            use_amplitude=False, n_main_channel_pcs=1
        ),
        refinement_cfg=dartsort.RefinementConfig(),
        featurization_cfg=dartsort.FeaturizationConfig(
            n_residual_snips=512, nn_localization=False
        ),
        motion_estimation_cfg=dartsort.MotionEstimationConfig(
            do_motion_estimation=False
        ),
        # test the dev tasks pipeline
        save_intermediate_labels=True,
        save_intermediate_features=False,
    )
    res = dartsort.dartsort(sim_recording, output_dir=tmp_path, cfg=cfg)
    assert res["sorting"].parent_h5_path.exists()
    assert (tmp_path / "dartsort_sorting.npz").exists()
    assert not (tmp_path / "subtraction.h5").exists()
    assert (tmp_path / "matching1.h5").exists()


def test_cli_help():
    # at least make sure the cli can do -h
    res = subprocess.run(["dartsort", "-h"], capture_output=True)
    assert not res.returncode, res.stderr.decode()


@pytest.mark.parametrize(
    "type", ["subtract", "threshold", "match", "drifty_match"]
)
def test_initial_detection_swap(tmp_path, simulations, type):
    sim = simulations["driftn_szmini"]
    sim["templates"].to_npz(tmp_path / "temps.npz")

    cfg_add = {}
    if type == "drifty_match":
        type = "match"
        cfg_add["matching_template_type"] = "drifty"
        cfg_add["matching_up_method"] = "keys4"
        cfg_add["matching_template_min_amplitude"] = 1.0

    cfg = dartsort.DeveloperConfig(
        dredge_only=True,
        detection_type=type.removesuffix("_cumulant"),
        precomputed_templates_npz=str(tmp_path / "temps.npz"),
        save_intermediates=True,
        **cfg_add,
    )
    with warnings.catch_warnings() as ws:
        warnings.filterwarnings(
            "ignore", message="Can't extract this many non-overlapping snips."
        )
        res = dartsort.dartsort(sim["recording"], output_dir=tmp_path, cfg=cfg)
    assert res["sorting"].parent_h5_path.exists()
    if type.startswith("subtract"):
        h5_name = "subtraction"
    elif type == "threshold":
        h5_name = "threshold"
    elif type == "universal":
        h5_name = "universal"
    elif type == "match":
        h5_name = "matching0"
    else:
        assert False
    assert (tmp_path / f"{h5_name}.h5").exists()
    assert not (tmp_path / "matching1.h5").exists()

    if type == "match":
        count_dif_tol = 0.035
    elif type == "universal":
        count_dif_tol = 0.01
    elif type == "subtract":
        count_dif_tol = 0.15
    elif type == "threshold":
        count_dif_tol = 0.35
    else:
        assert False

    c0 = len(sim["sorting"])
    c1 = len(res["sorting"])
    assert abs(c0 - c1) < count_dif_tol * c0
