from pathlib import Path
import pytest
import tempfile
import subprocess


import dartsort
from dartsort.util import simkit


@pytest.fixture
def sim_recording():
    geom = simkit.generate_geom()
    rec_sim = simkit.StaticSimulatedRecording(
        duration_samples=10 * 30_000,
        n_units=40,
        template_simulator=simkit.PointSource3ExpSimulator(geom),
        noise=simkit.WhiteNoise(len(geom)),
        min_fr_hz=20.0,
        max_fr_hz=31.0,
        temporal_jitter=4,
    )
    rec = rec_sim.simulate()
    return rec


@pytest.mark.parametrize("do_motion_estimation", [False, True])
def test_fakedata_nonn(sim_recording, do_motion_estimation):
    with tempfile.TemporaryDirectory() as tempdir:
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
        )
        res = dartsort.dartsort(sim_recording, output_dir=tempdir, cfg=cfg)
        assert res["sorting"].parent_h5_path.exists()
        assert (Path(tempdir) / "dartsort_sorting.npz").exists()
        assert (Path(tempdir) / "subtraction.h5").exists()
        assert (Path(tempdir) / "matching1.h5").exists()


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
def test_fakedata(sim_recording, sdcfg):
    with tempfile.TemporaryDirectory() as tempdir:
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
        res = dartsort.dartsort(sim_recording, output_dir=tempdir, cfg=cfg)
        assert res["sorting"].parent_h5_path.exists()
        assert (Path(tempdir) / "dartsort_sorting.npz").exists()
        assert not (Path(tempdir) / "subtraction.h5").exists()
        assert (Path(tempdir) / "matching1.h5").exists()


def test_cli_help():
    # at least make sure the cli can do -h
    res = subprocess.run(["dartsort", "-h"])
    assert not res.returncode


if __name__ == "__main__":
    test_fakedata(sim_recording())
