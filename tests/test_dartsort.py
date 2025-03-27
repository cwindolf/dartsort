import numpy as np
import tempfile
import subprocess


import dartsort
from dartsort.util import simkit


def test_fakedata():
    geom = simkit.generate_geom()
    rec_sim = simkit.StaticSimulatedRecording(
        template_simulator=simkit.PointSource3ExpSimulator(geom),
        noise=simkit.WhiteNoise(len(geom)),
        firing_rates=np.arange(20.0, 31.0),
        jitter=4,
    )
    rec, gt_sorting = rec_sim.simulate(t_samples=10 * 30_000)

    with tempfile.TemporaryDirectory() as tempdir:
        cfg = dartsort.DARTsortInternalConfig(
            subtraction_config=dartsort.SubtractionConfig(
                subtraction_denoising_config=dartsort.FeaturizationConfig(
                    denoise_only=True, do_nn_denoise=False
                )
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
        )
        res = dartsort.dartsort(rec, output_dir=tempdir, cfg=cfg)

    for do_motion_estimation in (False, True):
        with tempfile.TemporaryDirectory() as tempdir:
            cfg = dartsort.DARTsortInternalConfig(
                subtraction_config=dartsort.SubtractionConfig(
                    subtraction_denoising_config=dartsort.FeaturizationConfig(
                        denoise_only=True, do_nn_denoise=False
                    )
                ),
                refinement_config=dartsort.RefinementConfig(
                    min_count=10, n_total_iters=1
                ),
                featurization_config=dartsort.FeaturizationConfig(n_residual_snips=512),
                motion_estimation_config=dartsort.MotionEstimationConfig(
                    do_motion_estimation=do_motion_estimation, rigid=True
                ),
            )
            res = dartsort.dartsort(rec, output_dir=tempdir, cfg=cfg)


def test_cli_help():
    # at least make sure the cli can do -h
    res = subprocess.run(["dartsort", "-h"])
    assert not res.returncode


if __name__ == "__main__":
    test_fakedata()
