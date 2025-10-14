from logging import getLogger
import pickle
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from tqdm.auto import tqdm


from ..evaluate.analysis import DARTsortAnalysis
from ..evaluate.comparison import DARTsortGroundTruthComparison, DARTsortGTVersus
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    raw_template_cfg,
    unshifted_template_cfg,
    ComputationConfig,
)
from ..util.job_util import get_global_computation_config
from ..evaluate.hybrid_util import load_dartsort_step_sortings
from . import over_time, scatterplots, unit, gt, unit_comparison, versus
from .sorting import make_sorting_summary

try:
    from dredge import motion_util

    have_dredge = True
except ImportError:
    have_dredge = False
    motion_util = None


logger = getLogger(__name__)


def visualize_sorting(
    recording,
    sorting,
    output_directory,
    sorting_name=None,
    sorting_path=None,
    motion_est=None,
    gt_analysis=None,
    other_analyses=None,
    gt_comparison_with_distances=True,
    make_scatterplots=True,
    make_sorting_summaries=True,
    make_unit_summaries=True,
    make_animations=False,
    make_gt_overviews=True,
    make_unit_comparisons=True,
    make_versus=True,
    sorting_analysis=None,
    template_cfg=unshifted_template_cfg,
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    channel_show_radius_um=50.0,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    chunk_length_s=300.0,
    frame_interval=1500,
    exhaustive_gt=True,
    dpi=200,
    overwrite=False,
    computation_cfg=None,
    errors_to_warnings=True,
):
    output_directory.mkdir(exist_ok=True, parents=True)
    if computation_cfg is None:
        computation_cfg = get_global_computation_config()

    if sorting is None and sorting_path is not None:
        if sorting_path.name.endswith(".h5"):
            sorting = DARTsortSorting.from_peeling_hdf5(sorting_path)
        elif sorting_path.name.endswith(".npz"):
            sorting = DARTsortSorting.load(sorting_path)
        else:
            assert False

    try:
        if make_scatterplots:
            sorting_scatterplots(
                output_directory,
                sorting,
                motion_est=motion_est,
                amplitude_color_cutoff=amplitude_color_cutoff,
                amplitudes_dataset_name=amplitudes_dataset_name,
                dpi=dpi,
                overwrite=overwrite,
            )
    except Exception as e:
        if errors_to_warnings:
            warnings.warn(str(e))
        else:
            raise
        plt.close("all")

    # figure out if we need a sorting analysis object and hide some
    # logic for figuring out which steps need running
    sorting_analysis, gt_comparison, gt_vs, *paths_or_nones = _plan_vis(
        output_directory,
        recording,
        sorting,
        sorting_name=sorting_name,
        motion_est=motion_est,
        make_sorting_summaries=make_sorting_summaries,
        make_unit_summaries=make_unit_summaries,
        make_animations=make_animations,
        make_gt_overviews=make_gt_overviews,
        make_unit_comparisons=make_unit_comparisons,
        make_versus=make_versus,
        sorting_analysis=sorting_analysis,
        gt_analysis=gt_analysis,
        other_analyses=other_analyses,
        overwrite=overwrite,
        template_cfg=template_cfg,
        computation_cfg=computation_cfg,
        exhaustive_gt=exhaustive_gt,
        gt_comparison_with_distances=gt_comparison_with_distances,
    )
    sum_png, unit_sum_dir, anim_png, comp_png, unit_comp_dir, vs_png = paths_or_nones

    try:
        if sum_png is not None:
            if overwrite or not sum_png.exists():
                fig = make_sorting_summary(
                    sorting_analysis,
                    figure=None,
                )
                fig.savefig(sum_png, dpi=dpi)
    except Exception as e:
        if errors_to_warnings:
            warnings.warn(str(e))
        else:
            raise

    if anim_png is not None:
        if overwrite or not anim_png.exists():
            over_time.sorting_scatter_animation(
                sorting_analysis,
                anim_png,
                chunk_length_samples=chunk_length_s * recording.sampling_frequency,
                interval=frame_interval,
            )

    if comp_png is not None and gt_analysis is not None:
        if overwrite or not comp_png.exists():
            assert gt_comparison is not None
            plots = (
                gt.full_gt_overview_plots
                if gt_comparison_with_distances
                else gt.default_gt_overview_plots
            )
            fig = gt.make_gt_overview_summary(gt_comparison, plots=plots)
            fig.savefig(comp_png, dpi=dpi)

    if vs_png is not None and gt_vs is not None:
        fig = versus.make_versus_summary(gt_vs)
        fig.savefig(vs_png, dpi=dpi)

    if unit_sum_dir is not None:
        unit.make_all_summaries(
            sorting_analysis,
            unit_sum_dir,
            channel_show_radius_um=channel_show_radius_um,
            amplitude_color_cutoff=amplitude_color_cutoff,
            amplitudes_dataset_name=amplitudes_dataset_name,
            pca_radius_um=pca_radius_um,
            dpi=dpi,
            show_progress=True,
            overwrite=overwrite,
            n_jobs=computation_cfg.n_jobs_cpu,
        )

    if unit_comp_dir is not None and gt_analysis is not None:
        assert gt_comparison is not None
        unit_comparison.make_all_unit_comparisons(
            gt_comparison,
            unit_comp_dir,
            channel_show_radius_um=channel_show_radius_um,
            amplitude_color_cutoff=amplitude_color_cutoff,
            amplitudes_dataset_name=amplitudes_dataset_name,
            pca_radius_um=pca_radius_um,
            dpi=dpi,
            show_progress=True,
            overwrite=overwrite,
            n_jobs=computation_cfg.n_jobs_cpu,
        )


def visualize_all_sorting_steps(
    recording,
    dartsort_dir,
    visualizations_dir,
    gt_analysis=None,
    other_analyses=None,
    make_scatterplots=True,
    make_sorting_summaries=True,
    make_unit_summaries=True,
    make_animations=False,
    make_gt_overviews=True,
    make_unit_comparisons=True,
    make_versus=True,
    step_sortings=None,
    template_cfg=unshifted_template_cfg,
    gt_comparison_with_distances=True,
    step_dir_name_format="step{step:02d}_{step_name}",
    step_name_formatter=None,
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    motion_est=None,
    motion_est_pkl="motion_est.pkl",
    channel_show_radius_um=50.0,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    exhaustive_gt=True,
    start_from_matching=False,
    stop_after=None,
    dpi=200,
    overwrite=False,
    load_step_sortings_kw=None,
    reverse=False,
    computation_cfg=None,
):
    dartsort_dir = Path(dartsort_dir)
    visualizations_dir = Path(visualizations_dir)

    if motion_est is None:
        motion_est_pkl = dartsort_dir / motion_est_pkl
        if motion_est_pkl.exists():
            with open(motion_est_pkl, "rb") as jar:
                motion_est = pickle.load(jar)

    fnames = ["times_seconds"]
    if make_scatterplots or make_sorting_summaries:
        fnames += ["point_source_localizations", amplitudes_dataset_name]
    if step_sortings is None:
        step_sortings = load_dartsort_step_sortings(
            dartsort_dir,
            load_simple_features=True,
            load_feature_names=fnames,
            name_formatter=step_name_formatter,
            **(load_step_sortings_kw or {}),
        )
    assert step_sortings is not None

    steps = enumerate(step_sortings)
    if reverse:
        logger.info("Reversing the steps...")
        steps = list(steps)
        nsteps = len(steps)
        steps = reversed(steps)
    else:
        nsteps = None

    count = 0
    with tqdm(steps, desc="Sorting steps", mininterval=0, total=nsteps) as prog:
        for j, (step_name, step_sorting) in prog:
            if step_name is None:
                continue
            if start_from_matching:
                h5p = step_sorting.parent_h5_path
                if h5p is None:
                    continue
                if not Path(h5p).stem.startswith("match"):
                    continue
            assert all(hasattr(step_sorting, fn) for fn in fnames)
            prog.write(f"Vis step  {j}: {step_name}.\n{step_sorting}")
            step_dir_name = step_dir_name_format.format(step=j, step_name=step_name)
            visualize_sorting(
                recording=recording,
                sorting=step_sorting,
                sorting_name=step_name,
                output_directory=visualizations_dir / step_dir_name,
                motion_est=motion_est,
                make_scatterplots=make_scatterplots,
                make_sorting_summaries=make_sorting_summaries,
                make_unit_summaries=make_unit_summaries,
                make_animations=make_animations,
                make_gt_overviews=make_gt_overviews,
                make_unit_comparisons=make_unit_comparisons,
                make_versus=make_versus,
                gt_analysis=gt_analysis,
                other_analyses=other_analyses,
                exhaustive_gt=exhaustive_gt,
                gt_comparison_with_distances=gt_comparison_with_distances,
                channel_show_radius_um=channel_show_radius_um,
                amplitude_color_cutoff=amplitude_color_cutoff,
                pca_radius_um=pca_radius_um,
                template_cfg=template_cfg,
                dpi=dpi,
                overwrite=overwrite,
                computation_cfg=computation_cfg,
            )

            count += 1
            if stop_after is not None and count >= stop_after:
                break


# -- helpers


def sorting_scatterplots(
    output_directory,
    sorting,
    motion_est=None,
    amplitude_color_cutoff=15.0,
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    dpi=200,
    overwrite=False,
):
    scatter_unreg = output_directory / "scatter_unreg.png"
    if overwrite or not scatter_unreg.exists():
        fig, axes, scatters = scatterplots.scatter_spike_features(
            sorting=sorting,
            amplitude_color_cutoff=amplitude_color_cutoff,
            amplitudes_dataset_name=amplitudes_dataset_name,
        )
        if have_dredge and motion_est is not None:
            assert motion_util is not None
            motion_util.plot_me_traces(motion_est, axes[2], color="r", lw=1)
        fig.savefig(scatter_unreg, dpi=dpi)
        plt.close(fig)

    scatter_reg = output_directory / "scatter_reg.png"
    if motion_est is not None and (overwrite or not scatter_reg.exists()):
        fig, axes, scatters = scatterplots.scatter_spike_features(
            sorting=sorting,
            motion_est=motion_est,
            registered=True,
            amplitude_color_cutoff=amplitude_color_cutoff,
            amplitudes_dataset_name=amplitudes_dataset_name,
        )
        fig.savefig(scatter_reg, dpi=dpi)
        plt.close(fig)


def _plan_vis(
    output_directory,
    recording,
    sorting,
    sorting_name=None,
    motion_est=None,
    make_sorting_summaries=False,
    make_unit_summaries=False,
    make_animations=False,
    make_gt_overviews=False,
    make_unit_comparisons=False,
    make_versus=False,
    sorting_analysis=None,
    other_analyses=None,
    gt_analysis=None,
    template_cfg=unshifted_template_cfg,
    exhaustive_gt=True,
    gt_comparison_with_distances=True,
    overwrite=False,
    computation_cfg=None,
):
    if computation_cfg is None:
        computation_cfg = get_global_computation_config()

    # goal of this fn is to figure out if we need to instantiate these
    # SortingAnalysis and GTComparison objects, or if we can skip everything
    need_analysis = False
    need_comparison = False
    need_vs = False

    # can't compare or analyze units if there aren't any
    is_labeled = sorting.n_units > 1

    if make_sorting_summaries and is_labeled:
        sorting_summary_png = output_directory / "sorting_summary.png"
        need_summary = overwrite or not sorting_summary_png.exists()
        need_analysis = need_analysis or need_summary
        if not need_summary:
            sorting_summary_png = None
    else:
        sorting_summary_png = None

    if make_unit_summaries and is_labeled:
        unit_summary_dir = output_directory / "single_unit_summaries"
        if overwrite:
            need_summaries = True
        else:
            need_summaries = not unit.all_summaries_done(
                sorting.unit_ids, unit_summary_dir
            )
        need_analysis = need_analysis or need_summaries
        if not need_summaries:
            unit_summary_dir = None
    else:
        unit_summary_dir = None

    if make_animations and is_labeled:
        animation_png = output_directory / "animation.mp4"
        need_anim = overwrite or not animation_png.exists()
        need_analysis = need_analysis or need_anim
        if not need_anim:
            animation_png = None
    else:
        animation_png = None

    can_gt = gt_analysis is not None and is_labeled
    if can_gt and make_gt_overviews:
        comparison_png = output_directory / "gt_comparison.png"
        need_comp = overwrite or not comparison_png.exists()
        need_analysis = need_analysis or need_comp
        need_comparison = need_comparison or need_comp
        if not need_comp:
            comparison_png = None
    else:
        comparison_png = None

    if can_gt and make_unit_comparisons:
        unit_comparison_dir = output_directory / "gt_unit_comparisons"
        if overwrite:
            need_ucomps = True
        else:
            # TODO: unit_comparison.all_summaries_done
            need_ucomps = not unit.all_summaries_done(
                gt_analysis.sorting.unit_ids,
                unit_comparison_dir,
                sorting_analysis=gt_analysis,
                namebyamp=True,
            )
        need_analysis = need_analysis or need_ucomps
        need_comparison = need_comparison or need_ucomps
        if not need_ucomps:
            unit_comparison_dir = None
    else:
        unit_comparison_dir = None

    if can_gt and other_analyses is not None and make_versus:
        gtn = gt_analysis.name
        on = "_vs_".join(oa.name for oa in other_analyses)
        vs_png = output_directory / f"{gtn}_study_{on}.png"
        need_vs = overwrite or not vs_png.exists()
        need_analysis = need_analysis or need_vs
        need_comparison = need_comparison or need_vs
        if not need_vs:
            vs_png = None
    else:
        vs_png = None

    if need_analysis and sorting_analysis is None:
        if sorting_name is None:
            sorting_name = output_directory.stem
        sorting_analysis = DARTsortAnalysis.from_sorting(
            recording=recording,
            sorting=sorting,
            motion_est=motion_est,
            name=sorting_name,
            template_cfg=template_cfg,
            allow_template_reload="match" in output_directory.stem,
            computation_cfg=computation_cfg,
            compute_distances=True,
        )

    if need_comparison:
        assert sorting_analysis is not None
        assert gt_analysis is not None
        gt_comparison = DARTsortGroundTruthComparison(
            gt_analysis=gt_analysis,
            tested_analysis=sorting_analysis,
            exhaustive_gt=exhaustive_gt,
            compute_distances=gt_comparison_with_distances,
        )
    else:
        gt_comparison = None

    if need_vs:
        assert sorting_analysis is not None
        assert gt_analysis is not None
        assert gt_comparison is not None
        assert other_analyses is not None

        comparison_kw = dict(exhaustive_gt=exhaustive_gt, compute_distances=gt_comparison_with_distances)
        cmps = [gt_comparison] + ([None] * len(other_analyses))
        gt_vs = DARTsortGTVersus(
            gt_analysis,
            sorting_analysis,
            *other_analyses,
            comparison_kw=comparison_kw,
            comparisons=cmps,
        )
    else:
        gt_vs = None

    return (
        sorting_analysis,
        gt_comparison,
        gt_vs,
        sorting_summary_png,
        unit_summary_dir,
        animation_png,
        comparison_png,
        unit_comparison_dir,
        vs_png,
    )
