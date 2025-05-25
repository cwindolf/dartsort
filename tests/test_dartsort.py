import dataclasses
import pytest
import subprocess

import dartsort


@pytest.mark.parametrize("do_motion_estimation", [False, True])
def test_fakedata_nonn(tmp_path, sim_recordings, do_motion_estimation):
    sim_recording = (
        sim_recordings["drifting"] if do_motion_estimation else sim_recordings["static"]
    )
    sim_recording = sim_recording["rec"]

    cfg = dartsort.DARTsortInternalConfig(
        subtraction_config=dartsort.SubtractionConfig(
            subtraction_denoising_config=dartsort.FeaturizationConfig(
                denoise_only=True, do_nn_denoise=False
            )
        ),
        initial_refinement_config=dartsort.RefinementConfig(
            min_count=10, n_total_iters=1, one_split_only=True
        ),
        refinement_config=dartsort.RefinementConfig(min_count=10, n_total_iters=1),
        featurization_config=dartsort.FeaturizationConfig(n_residual_snips=512),
        motion_estimation_config=dartsort.MotionEstimationConfig(
            do_motion_estimation=do_motion_estimation, rigid=True
        ),
        work_in_tmpdir=True,
        save_intermediate_features=True,
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
def test_fakedata(tmp_path, sim_recordings, sdcfg):
    sim_recording = sim_recordings["static"]["rec"]

    cfg = dartsort.DARTsortInternalConfig(
        subtraction_config=dartsort.SubtractionConfig(
            subtraction_denoising_config=sdcfg,
            first_denoiser_thinning=0.0,
        ),
        # test pc based clust
        clustering_config=dartsort.ClusteringConfig(
            use_amplitude=False, n_main_channel_pcs=1
        ),
        refinement_config=dartsort.RefinementConfig(
            min_count=10, channels_strategy="count", n_total_iters=1
        ),
        featurization_config=dartsort.FeaturizationConfig(
            n_residual_snips=512, nn_localization=False
        ),
        motion_estimation_config=dartsort.MotionEstimationConfig(
            do_motion_estimation=False
        ),
        matching_config=dartsort.MatchingConfig(threshold="fp_control"),
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
    assert not res.returncode


if __name__ == "__main__":
    test_fakedata(sim_recording())
