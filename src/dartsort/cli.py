"""
This is very work in progress!
I'm not sure that things will work this way at all in the future!
Not sure we'll keep using click -- I want to auto generate documentation
from the config objects?
"""

import importlib.util

import click
import spikeinterface.full as si

from .main import dartsort, default_dartsort_config
from .vis.vismain import visualize_all_sorting_steps


def run_from_binary_folder_and_config_py(
    binary_folder,
    output_directory,
    config_path=None,
    n_jobs=0,
    n_jobs_cluster=0,
    overwrite=False,
    show_progress=True,
    device=None,
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

    recording = si.read_binary_folder(binary_folder)

    return dartsort(
        recording,
        output_directory,
        cfg=cfg,
        motion_est=None,
        n_jobs=n_jobs,
        n_jobs_cluster=n_jobs_cluster,
        overwrite=overwrite,
        show_progress=show_progress,
        device=device,
    )


@click.command()
@click.argument("binary_folder")
@click.argument("output_directory")
@click.option("--config_path", type=str, default=None)
@click.option("--n_jobs", default=0)
@click.option("--n_jobs_cluster", default=0)
@click.option("--overwrite", default=False, flag_value=True, is_flag=True)
@click.option("--no_show_progress", default=False, flag_value=True, is_flag=True)
@click.option("--device", type=str, default=None)
def dartsort_binary_folder_config_py(
    binary_folder,
    output_directory,
    config_path=None,
    n_jobs=0,
    n_jobs_cluster=0,
    overwrite=False,
    no_show_progress=False,
    device=None,
):
    run_from_binary_folder_and_config_py(
        binary_folder,
        output_directory,
        config_path=config_path,
        n_jobs=n_jobs,
        n_jobs_cluster=n_jobs_cluster,
        overwrite=overwrite,
        show_progress=not no_show_progress,
        device=device,
    )


@click.command()
@click.argument("dartsort_dir")
@click.argument("visualizations_dir")
@click.option("--n_jobs", default=0)
@click.option("--overwrite", default=False, flag_value=True, is_flag=True)
@click.option("--no_scatterplots", default=False, flag_value=True, is_flag=True)
@click.option("--no_summaries", default=False, flag_value=True, is_flag=True)
def dartvis_binary_folder_all(
    binary_folder,
    dartsort_dir,
    visualizations_dir,
    n_jobs=0,
    overwrite=False,
    no_scatterplots=False,
    no_summaries=False,
):
    recording = si.read_binary_folder(binary_folder)
    visualize_all_sorting_steps(
        recording,
        dartsort_dir,
        visualizations_dir,
        make_scatterplots=not no_scatterplots,
        make_unit_summaries=not no_summaries,
        n_jobs=n_jobs,
        overwrite=overwrite,
    )
