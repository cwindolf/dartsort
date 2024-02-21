import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ..util.analysis import DARTsortAnalysis
from ..util.data_util import DARTsortSorting
from . import scatterplots, unit

try:
    from dredge import motion_util

    have_dredge = True
except ImportError:
    have_dredge = False


def visualize_sorting(
    recording,
    sorting,
    output_directory,
    motion_est=None,
    make_scatterplots=True,
    make_unit_summaries=True,
    gt_sorting=None,
    dpi=200,
    n_jobs=0,
    n_jobs_templates=0,
    overwrite=False,
):
    output_directory.mkdir(exist_ok=True, parents=True)

    if make_scatterplots:
        scatter_unreg = output_directory / "scatter_unreg.png"
        if overwrite or not scatter_unreg.exists():
            fig, axes, scatters = scatterplots.scatter_spike_features(sorting=sorting)
            if have_dredge and motion_est is not None:
                motion_util.plot_me_traces(motion_est, axes[2], color="r", lw=1)
            fig.savefig(scatter_unreg, dpi=dpi)
            plt.close(fig)

        scatter_reg = output_directory / "scatter_reg.png"
        if motion_est is not None and (overwrite or not scatter_reg.exists()):
            fig, axes, scatters = scatterplots.scatter_spike_features(
                sorting=sorting, motion_est=motion_est, registered=True
            )
            fig.savefig(scatter_reg, dpi=dpi)
            plt.close(fig)

    if make_unit_summaries and sorting.n_units > 1:
        unit_summary_dir = output_directory / "single_unit_summaries"
        sorting_analysis = DARTsortAnalysis.from_sorting(
            recording=recording,
            sorting=sorting,
            motion_est=motion_est,
            n_jobs_templates=n_jobs_templates,
        )
        unit.make_all_summaries(
            sorting_analysis,
            unit_summary_dir,
            channel_show_radius_um=50.0,
            amplitude_color_cutoff=15.0,
            dpi=dpi,
            n_jobs=n_jobs,
            show_progress=True,
            overwrite=overwrite,
        )


def visualize_all_sorting_steps(
    recording,
    dartsort_dir,
    visualizations_dir,
    make_scatterplots=True,
    make_unit_summaries=True,
    gt_sorting=None,
    step_dir_name_format="step{step:02d}_{step_name}",
    motion_est_pkl="motion_est.pkl",
    initial_sortings=("subtraction.h5", "initial_clustering.npz"),
    step_refinements=("split{step}.npz", "merge{step}.npz"),
    match_step_sorting="matching{step}.h5",
    dpi=200,
    n_jobs=0,
    n_jobs_templates=0,
    overwrite=False,
):
    dartsort_dir = Path(dartsort_dir)
    visualizations_dir = Path(visualizations_dir)

    step_paths = list(initial_sortings)
    n_match_steps = sum(
        1 for _ in dartsort_dir.glob(match_step_sorting.format(step="*"))
    )
    match_step_sortings = step_refinements + (match_step_sorting,)
    for step in range(n_match_steps):
        step_paths.extend(s.format(step=step) for s in match_step_sortings)

    motion_est_pkl = dartsort_dir / motion_est_pkl
    if motion_est_pkl.exists():
        with open(motion_est_pkl, "rb") as jar:
            motion_est = pickle.load(jar)

    for j, path in enumerate(tqdm(step_paths, desc="Sorting steps")):
        sorting_path = dartsort_dir / path
        step_name = sorting_path.name
        if sorting_path.name.endswith(".h5"):
            sorting = DARTsortSorting.from_peeling_hdf5(sorting_path)
            step_name = step_name.removesuffix(".h5")
        elif sorting_path.name.endswith(".npz"):
            sorting = DARTsortSorting.load(sorting_path)
            step_name = step_name.removesuffix(".npz")
        else:
            assert False

        step_dir_name = step_dir_name_format.format(step=j, step_name=step_name)
        visualize_sorting(
            recording=recording,
            sorting=sorting,
            output_directory=visualizations_dir / step_dir_name,
            motion_est=motion_est,
            make_scatterplots=make_scatterplots,
            make_unit_summaries=make_unit_summaries,
            gt_sorting=gt_sorting,
            dpi=dpi,
            n_jobs=n_jobs,
            n_jobs_templates=n_jobs_templates,
            overwrite=overwrite,
        )
