import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ..evaluate.analysis import (DARTsortAnalysis, basic_template_cfg,
                                        no_realign_template_cfg)
from ..util.data_util import DARTsortSorting
from ..evaluate.hybrid_util import load_dartsort_step_sortings
from . import over_time, scatterplots, unit
from .sorting import make_sorting_summary

try:
    from dredge import motion_util

    have_dredge = True
except ImportError:
    have_dredge = False


def visualize_sorting(
    recording,
    sorting,
    output_directory,
    sorting_path=None,
    motion_est=None,
    make_scatterplots=True,
    make_sorting_summaries=True,
    make_unit_summaries=True,
    make_animations=False,
    superres_templates=False,
    sorting_analysis=None,
    amplitudes_dataset_name='denoised_ptp_amplitudes',
    channel_show_radius_um=50.0,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    chunk_length_s=300.0,
    frame_interval=1500,
    dpi=200,
    layout_max_height=4,
    layout_figsize=(11, 8.5),
    overwrite=False,
):
    output_directory.mkdir(exist_ok=True, parents=True)

    if sorting is None and sorting_path is not None:
        if sorting_path.name.endswith(".h5"):
            sorting = DARTsortSorting.from_peeling_hdf5(sorting_path)
        elif sorting_path.name.endswith(".npz"):
            sorting = DARTsortSorting.load(sorting_path)
        else:
            assert False

    if make_scatterplots:
        scatter_unreg = output_directory / "scatter_unreg.png"
        if overwrite or not scatter_unreg.exists():
            fig = plt.figure(figsize=layout_figsize)
            fig, axes, scatters = scatterplots.scatter_spike_features(
                sorting=sorting, figure=fig, amplitude_color_cutoff=amplitude_color_cutoff, amplitudes_dataset_name=amplitudes_dataset_name,
            )
            if have_dredge and motion_est is not None:
                motion_util.plot_me_traces(motion_est, axes[2], color="r", lw=1)
            fig.savefig(scatter_unreg, dpi=dpi)
            plt.close(fig)

        scatter_reg = output_directory / "scatter_reg.png"
        if motion_est is not None and (overwrite or not scatter_reg.exists()):
            fig = plt.figure(figsize=layout_figsize)
            fig, axes, scatters = scatterplots.scatter_spike_features(
                sorting=sorting,
                motion_est=motion_est,
                registered=True,
                figure=fig,
                amplitude_color_cutoff=amplitude_color_cutoff, amplitudes_dataset_name=amplitudes_dataset_name,
            )
            fig.savefig(scatter_reg, dpi=dpi)
            plt.close(fig)

    # figure out if we need a sorting analysis object
    need_analysis = False
    is_labeled = sorting.n_units > 1
    if make_sorting_summaries and is_labeled:
        sorting_summary_png = output_directory / "sorting_summary.png"
        need_analysis = need_analysis or not sorting_summary_png.exists()
    summaries_done = False
    if make_unit_summaries and is_labeled:
        unit_summary_dir = output_directory / "single_unit_summaries"
        summaries_done = not overwrite and unit.all_summaries_done(
            sorting.unit_ids, unit_summary_dir
        )
        need_analysis = need_analysis or not summaries_done
    if make_animations and is_labeled:
        animation_png = output_directory / "animation.mp4"
        need_analysis = need_analysis or (make_animations and not animation_png.exists())
    if need_analysis and sorting_analysis is None:
        template_cfg = no_realign_template_cfg if superres_templates else basic_template_cfg
        sorting_analysis = DARTsortAnalysis.from_sorting(
            recording=recording,
            sorting=sorting,
            motion_est=motion_est,
            name=output_directory.stem,
            template_cfg=template_cfg,
            allow_template_reload="match" in output_directory.stem,
        )

    if make_sorting_summaries and is_labeled:
        if overwrite or not sorting_summary_png.exists():
            fig = make_sorting_summary(
                sorting_analysis,
                max_height=layout_max_height,
                figsize=layout_figsize,
                figure=None,
            )
            fig.savefig(sorting_summary_png, dpi=dpi)

    if make_animations and is_labeled:
        if overwrite or not animation_png.exists():
            over_time.sorting_scatter_animation(
                sorting_analysis,
                animation_png,
                chunk_length_samples=chunk_length_s * recording.sampling_frequency,
                interval=frame_interval,
            )

    if make_unit_summaries and is_labeled:
        if not summaries_done:
            unit.make_all_summaries(
                sorting_analysis,
                unit_summary_dir,
                channel_show_radius_um=channel_show_radius_um,
                amplitude_color_cutoff=amplitude_color_cutoff,
                amplitudes_dataset_name=amplitudes_dataset_name,
                pca_radius_um=pca_radius_um,
                max_height=layout_max_height,
                figsize=layout_figsize,
                dpi=dpi,
                show_progress=True,
                overwrite=overwrite,
            )


def visualize_all_sorting_steps(
    recording,
    dartsort_dir,
    visualizations_dir,
    make_scatterplots=True,
    make_sorting_summaries=True,
    make_unit_summaries=True,
    make_animations=False,
    step_dir_name_format="step{step:02d}_{step_name}",
    motion_est_pkl="motion_est.pkl",
    initial_sortings=("subtraction.h5", "initial_clustering.npz"),
    step_refinements=("split{step}.npz", "merge{step}.npz"),
    match_step_sorting="matching{step}.h5",
    superres_templates=False,
    channel_show_radius_um=50.0,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    layout_max_height=4,
    layout_figsize=(11, 8.5),
    dpi=200,
    overwrite=False,
):
    dartsort_dir = Path(dartsort_dir)
    visualizations_dir = Path(visualizations_dir)

    motion_est_pkl = dartsort_dir / motion_est_pkl
    if motion_est_pkl.exists():
        with open(motion_est_pkl, "rb") as jar:
            motion_est = pickle.load(jar)

    step_sortings = load_dartsort_step_sortings(
        dartsort_dir,
        load_simple_features=True,
        load_feature_names=('times_seconds', 'point_source_localizations', 'denoised_ptp_amplitudes'),
    )

    with tqdm(step_sortings, desc="Sorting steps", mininterval=0) as prog:
        for j, (step_name, step_sorting) in enumerate(prog):
            prog.write(f"Vis step  {j}: {step_name}.\n{step_sorting}")
            step_dir_name = step_dir_name_format.format(step=j, step_name=step_name)
            visualize_sorting(
                recording=recording,
                sorting=step_sorting,
                output_directory=visualizations_dir / step_dir_name,
                motion_est=motion_est,
                make_scatterplots=make_scatterplots,
                make_sorting_summaries=make_sorting_summaries,
                make_unit_summaries=make_unit_summaries,
                make_animations=make_animations,
                superres_templates=superres_templates,
                channel_show_radius_um=channel_show_radius_um,
                amplitude_color_cutoff=amplitude_color_cutoff,
                pca_radius_um=pca_radius_um,
                dpi=dpi,
                layout_max_height=layout_max_height,
                layout_figsize=layout_figsize,
                overwrite=overwrite,
            )
