"""
This is very work in progress!
I'm not sure that things will work this way at all in the future!
Not sure we'll keep using click -- I want to auto generate documentation
from the config objects?
"""

import importlib.util
from pathlib import Path

import click
import spikeinterface.full as si

from .main import dartsort, default_dartsort_config
from .vis.vismain import visualize_all_sorting_steps

# -- entry points


@click.command()
@click.argument("si_rec_path")
@click.argument("output_directory")
@click.option("--config_path", type=str, default=None)
@click.option("--take_subtraction_from", type=str, default=None)
@click.option("--n_jobs_gpu", default=None, type=int)
@click.option("--n_jobs_cpu", default=None, type=int)
@click.option("--overwrite", default=False, flag_value=True, is_flag=True)
@click.option("--no_show_progress", default=False, flag_value=True, is_flag=True)
@click.option("--device", type=str, default=None)
@click.option("--rec_to_memory", default=False, flag_value=True, is_flag=True)
def dartsort_si_config_py(
    si_rec_path,
    output_directory,
    config_path=None,
    take_subtraction_from=None,
    n_jobs_gpu=None,
    n_jobs_cpu=None,
    overwrite=False,
    no_show_progress=False,
    device=None,
    rec_to_memory=False,
):
    run_from_si_rec_path_and_config_py(
        si_rec_path,
        output_directory,
        config_path=config_path,
        take_subtraction_from=take_subtraction_from,
        n_jobs_gpu=n_jobs_gpu,
        n_jobs_cpu=n_jobs_cpu,
        overwrite=overwrite,
        show_progress=not no_show_progress,
        device=device,
        rec_to_memory=rec_to_memory,
    )


@click.command()
@click.argument("si_rec_path")
@click.argument("dartsort_dir")
@click.argument("visualizations_dir")
@click.option("--channel_show_radius_um", default=50.0)
@click.option("--pca_radius_um", default=75.0)
@click.option("--no_superres_templates", default=False, flag_value=True, is_flag=True)
@click.option("--n_jobs_gpu", default=0)
@click.option("--n_jobs_cpu", default=0)
@click.option("--overwrite", default=False, flag_value=True, is_flag=True)
@click.option("--no_scatterplots", default=False, flag_value=True, is_flag=True)
@click.option("--no_summaries", default=False, flag_value=True, is_flag=True)
@click.option("--no_animations", default=False, flag_value=True, is_flag=True)
@click.option("--rec_to_memory", default=False, flag_value=True, is_flag=True)
def dartvis_si_all(
    si_rec_path,
    dartsort_dir,
    visualizations_dir,
    channel_show_radius_um=50.0,
    pca_radius_um=75.0,
    no_superres_templates=False,
    n_jobs_gpu=0,
    n_jobs_cpu=0,
    overwrite=False,
    no_scatterplots=False,
    no_summaries=False,
    no_animations=False,
    rec_to_memory=False,
):
    recording = si.load_extractor(si_rec_path)
    if rec_to_memory:
        recording = recording.save_to_memory(n_jobs=n_jobs_cpu)
    visualize_all_sorting_steps(
        recording,
        dartsort_dir,
        visualizations_dir,
        superres_templates=not no_superres_templates,
        channel_show_radius_um=channel_show_radius_um,
        pca_radius_um=pca_radius_um,
        make_scatterplots=not no_scatterplots,
        make_unit_summaries=not no_summaries,
        make_animations=not no_animations,
        n_jobs=n_jobs_gpu,
        n_jobs_templates=n_jobs_cpu,
        overwrite=overwrite,
    )


# -- scripting utils


def run_from_si_rec_path_and_config_py(
    si_rec_path,
    output_directory,
    config_path=None,
    take_subtraction_from=None,
    n_jobs_gpu=None,
    n_jobs_cpu=None,
    overwrite=False,
    show_progress=True,
    device=None,
    rec_to_memory=False,
):
    # stub for eventual function that reads a config file
    # I'm not sure this will be the way we actually do configuration
    # maybe we'll end up deserializing DARTsortConfigs from a non-python
    # config language
    if config_path is None:
        cfg = default_dartsort_config
    else:
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.cfg

    recording = si.load_extractor(si_rec_path)
    print(f"{recording=}")

    if rec_to_memory:
        recording = recording.save_to_memory()

    if take_subtraction_from is not None:
        symlink_subtraction_and_motion(
            take_subtraction_from,
            output_directory,
        )

    return dartsort(
        recording,
        output_directory,
        cfg=cfg,
        motion_est=None,
        n_jobs_gpu=n_jobs_gpu,
        n_jobs_cpu=n_jobs_cpu,
        overwrite=overwrite,
        show_progress=show_progress,
        device=device,
    )


def symlink_subtraction_and_motion(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    sub_h5 = input_dir / "subtraction.h5"
    if not sub_h5.exists():
        print(f"Can't symlink {sub_h5}")
        return

    targ_sub_h5 = output_dir / "subtraction.h5"
    if not targ_sub_h5.exists():
        targ_sub_h5.symlink_to(sub_h5)

    sub_models = input_dir / "subtraction_models"
    targ_sub_models = output_dir / "subtraction_models"
    if not targ_sub_models.exists():
        targ_sub_models.symlink_to(sub_models, target_is_directory=True)

    motion_est_pkl = input_dir / "motion_est.pkl"
    if motion_est_pkl.exists():
        targ_me_pkl = output_dir / "motion_est.pkl"
        if not targ_me_pkl.exists():
            targ_me_pkl.symlink_to(motion_est_pkl)
