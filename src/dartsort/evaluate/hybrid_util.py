import dataclasses
from logging import getLogger
from pathlib import Path
import time
from typing import Generator, Any
import warnings

import numpy as np
from probeinterface import Probe
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from spikeinterface.generation.drift_tools import (
    InjectDriftingTemplatesRecording,
    DriftingTemplates,
    move_dense_templates,
)
import torch
from tqdm.auto import tqdm

from ..templates import TemplateData
from ..util.data_util import DARTsortSorting
from ..util.internal_config import unshifted_raw_template_cfg, ComputationConfig
from ..config import DeveloperConfig
from . import simkit, comparison, analysis


logger = getLogger(__name__)


def get_drifty_hybrid_recording(
    recording,
    templates,
    motion_estimate,
    sorting=None,
    displacement_sampling_frequency=5.0,
    seed=0,
    firing_rates=None,
    peak_channels=None,
    amplitude_scale_std=0.1,
    amplitude_factor=None,
):
    """
    :param: recording
    :param: templates object
    :param: motion estimate object
    :param: firing_rates
    :param: peak_channels
    :param: amplitude_scale_std -- std of gamma distributed amplitude variation if
        amplitude_factor is None
    :param: amplitude_factor array of length n_spikes with amplitude factors
    """
    num_units = templates.num_units
    rg = np.random.default_rng(seed=seed)

    if peak_channels is None:
        (central_disp_index,) = np.flatnonzero(np.all(templates.displacements == 0, 1))
        central_templates = templates.templates_array_moved[central_disp_index]
        peak_channels = np.ptp(central_templates, 1).argmax(1)

    if sorting is None:
        sorting = simkit.simulate_sorting(
            num_units,
            recording.get_num_samples(),
            firing_rates=firing_rates,
            rg=rg,
            spike_length_samples=templates.num_samples,
            sampling_frequency=recording.sampling_frequency,
        )
    n_spikes = sorting.count_total_num_spikes()

    # Default amplitude scalings for spikes drawn from gamma
    if amplitude_factor is None:
        if amplitude_scale_std:
            shape = 1.0 / (amplitude_scale_std**1.5)
            amplitude_factor = rg.gamma(shape, scale=1.0 / (shape - 1), size=n_spikes)
        else:
            amplitude_factor = np.ones(n_spikes)

    depths = recording.get_probe().contact_positions[:, 1][peak_channels]
    t_start = recording.sample_index_to_time(0)
    t_end = recording.sample_index_to_time(recording.get_num_samples() - 1)
    motion_times_s = np.arange(
        t_start, t_end, step=1.0 / displacement_sampling_frequency
    )

    disp_y = motion_estimate.disp_at_s(motion_times_s, depths, grid=True)

    disp = np.zeros((motion_times_s.shape[0], 2, num_units))
    disp[:, 1, :] = disp_y.T
    # this tricks SI into using one displacement per unit
    displacement_unit_factor = np.eye(num_units)

    if not sorting.check_serializability(type="json"):
        warnings.warn(
            "Your sorting is not serializable, which could lead to problems later."
        )

    rec = InjectDriftingTemplatesRecording(
        sorting=sorting,
        drifting_templates=templates,
        parent_recording=recording,
        displacement_vectors=[disp],
        displacement_sampling_frequency=displacement_sampling_frequency,
        displacement_unit_factor=displacement_unit_factor,
        amplitude_factor=amplitude_factor,
    )
    rec.annotate(peak_channel=peak_channels.tolist())
    return rec


def precompute_displaced_registered_templates(
    template_data: TemplateData,
    geometry: np.array,
    displacements: np.array,
    sampling_frequency: float = 30000,
    template_subset=slice(None),
) -> DriftingTemplates:
    """Use spikeinterface tools to turn templates on registered geom into
    precomputed drifting templates on the regular geom.
    """
    source_probe = Probe(ndim=template_data.registered_geom.ndim)
    source_probe.set_contacts(positions=template_data.registered_geom)
    target_probe = Probe(ndim=geometry.ndim)
    target_probe.set_contacts(positions=geometry)

    shifted_templates = move_dense_templates(
        templates_array=template_data.templates[template_subset],
        displacements=displacements,
        source_probe=source_probe,
        dest_probe=target_probe,
    )

    ret = DriftingTemplates.from_precomputed_templates(
        templates_array_moved=shifted_templates,
        displacements=displacements,
        sampling_frequency=sampling_frequency,
        nbefore=template_data.trough_offset_samples,
        probe=target_probe,
    )
    return ret


def closest_clustering(
    gt_st, peel_st, geom=None, match_dt_ms=0.1, match_radius_um=0.0, p=2.0
):
    frames_per_ms = gt_st.sampling_frequency / 1000
    delta_frames = match_dt_ms * frames_per_ms
    rescale = [delta_frames]
    gt_pos = gt_st.times_samples[:, None]
    peel_pos = peel_st.times_samples[:, None]
    if match_radius_um:
        rescale = rescale + (geom.shape[1] * [match_radius_um])
        gt_pos = np.c_[gt_pos, geom[gt_st.channels]]
        peel_pos = np.c_[peel_pos, geom[peel_st.channels]]
    else:
        gt_pos = gt_pos.astype(float)
        peel_pos = peel_pos.astype(float)
    gt_pos /= rescale
    peel_pos /= rescale
    peelix2gtix = greedy_match(gt_pos, peel_pos, dx=1.0 / frames_per_ms)
    labels = peelix2gtix.copy()
    labels[labels >= 0] = gt_st.labels[labels[labels >= 0]]

    extra_features = peel_st.extra_features or {}
    extra_features = extra_features.copy()
    extra_features["match_ix"] = torch.from_numpy(peelix2gtix)
    return dataclasses.replace(peel_st, labels=labels, extra_features=extra_features)


def greedy_match(
    gt_coords,
    test_coords,
    max_val=1.0,
    dx=1.0 / 30,
    workers=-1,
    p=2.0,
    show_progress=True,
):
    """Greedily match spikes in a test sorting to those in a GT sorting

    Iteratively expands a spatiotemporal neighborhood around each test spike
    until a GT spike lands in the neighborhood; then arbitrarily pick from
    the GT spikes in the neighborhood as a match.

    Each spike in the GT and test spike trains will be matched to only one
    partner in the other sorting.

    Returns gt unit labels for each tested spike and an array describing
    whether each gt spike was matched.

    TODO: Be a little smarter. If there are ties, pick "best friend" unit.
    Could be done with two iterations.

    TODO: Return the gt->test assignments, which are easily computable here,
    if needed.
    """
    assignments = np.full(len(test_coords), -1)
    gt_unmatched = np.ones(len(gt_coords), dtype=bool)

    thresholds = np.arange(0.0 + 1e-5, max_val + dx + 2e-5, dx)
    if show_progress:
        thresholds = tqdm(thresholds, desc="Greedy match")

    for j, thresh in enumerate(thresholds):
        test_unmatched = np.flatnonzero(assignments < 0)
        if not test_unmatched.size:
            break
        test_kdtree = KDTree(test_coords[test_unmatched])
        gt_ix = np.flatnonzero(gt_unmatched)
        d, i = test_kdtree.query(
            gt_coords[gt_ix],
            k=1,
            distance_upper_bound=min(thresh, max_val),
            workers=workers,
            p=p,
        )
        # handle multiple gt spikes getting matched to the same peel ix
        thresh_matched = i < test_kdtree.n
        _, ii = np.unique(i, return_index=True)
        i = i[ii]
        thresh_matched = thresh_matched[ii]

        gt_ix = gt_ix[ii]
        gt_ix = gt_ix[thresh_matched]
        i = i[thresh_matched]
        assignments[test_unmatched[i]] = gt_ix
        gt_unmatched[gt_ix] = False

        if not gt_unmatched.any():
            break
        if thresh > max_val:
            break

    return assignments, gt_unmatched


def greedy_match_counts(
    gt_sorting,
    tested_sorting,
    radius_um=35.0,
    radius_frames=12,
    show_progress=True,
):
    """A greedy confusion matrix computed using greedy_match()."""
    from scipy.optimize import linear_sum_assignment

    gt_t = gt_sorting.times_samples / radius_frames
    tested_t = tested_sorting.times_samples / radius_frames

    geom = getattr(gt_sorting, 'geom', getattr(tested_sorting, 'geom', None))
    assert geom is not None
    gt_x = geom[gt_sorting.channels] / radius_um
    tested_x = geom[tested_sorting.channels] / radius_um

    step = min(1.0 / radius_frames, pdist(geom).min() / radius_um) / 2

    test2gt_spike, gt_unmatched = greedy_match(np.c_[gt_t, gt_x], np.c_[tested_t, tested_x], dx=step)
    counts = np.zeros(
        (gt_sorting.unit_ids.max() + 1, tested_sorting.unit_ids.max() + 1),
        dtype=np.int32,
    )
    test_matched_spike = np.flatnonzero(np.logical_and(test2gt_spike >= 0, tested_sorting.labels >= 0))

    matched_gt_labels = gt_sorting.labels[test2gt_spike[test_matched_spike]]
    matched_test_labels = tested_sorting.labels[test_matched_spike]

    np.add.at(counts, (matched_gt_labels, matched_test_labels), 1)

    row_ind, col_ind = linear_sum_assignment(counts, maximize=True)

    return dict(
        counts=counts,
        test2gt_spike=test2gt_spike,
        gt_unmatched=gt_unmatched,
        test_matched_spike=test_matched_spike,
        matched_gt_labels=matched_gt_labels,
        matched_test_labels=matched_test_labels,
        counts_ord=counts[row_ind][:, col_ind],
        gt_unit_order=row_ind,
        test_unit_order=col_ind,
    )


def sorting_from_times_labels(
    times_samples,
    labels,
    recording=None,
    motion_est=None,
    sampling_frequency=None,
    determine_channels=True,
    template_cfg=unshifted_raw_template_cfg,
    n_jobs=0,
    spikes_per_unit=50,
):
    channels = np.zeros_like(labels)
    if sampling_frequency is None:
        if recording is not None:
            sampling_frequency = recording.sampling_frequency

    extra_features = {}
    if recording is not None:
        extra_features['times_seconds'] = recording.sample_index_to_time(times_samples)

    sorting = DARTsortSorting(
        times_samples=times_samples,
        channels=channels,
        labels=labels,
        sampling_frequency=sampling_frequency,
        extra_features=extra_features,
    )

    if not determine_channels:
        return sorting

    assert recording is not None

    _, labels_flat = np.unique(labels, return_inverse=True)
    sorting = dataclasses.replace(sorting, labels=labels_flat)
    template_cfg = dataclasses.replace(template_cfg, spikes_per_unit=spikes_per_unit)
    comp_cfg = ComputationConfig(n_jobs_cpu=n_jobs, n_jobs_gpu=n_jobs)
    td = TemplateData.from_config(
        recording, sorting, template_cfg, computation_cfg=comp_cfg
    )

    channels = np.nan_to_num(np.ptp(td.coarsen().templates, 1)).argmax(1)[labels_flat]
    if motion_est is not None:
        from scipy.spatial import KDTree

        rgeom = td.registered_geom
        guess_pos = rgeom[channels]
        times_seconds = recording.sample_index_to_time(times_samples)
        # anti-correct these already stable positions so that they start movin
        guess_pos[:, 1] += motion_est.disp_at_s(times_seconds, depth_um=guess_pos[:, 1])

        gkdt = KDTree(recording.get_channel_locations())
        # closest original channels to shifted positions
        # these positions can drift off the probe if the main channel does! so in that case
        # we can't really upper bound the distance query. i guess it would be, like, the
        # largest distance that a unit would ever extend, or something, but let's not worry.
        d, channels = gkdt.query(guess_pos, workers=n_jobs)

    sorting = dataclasses.replace(sorting, channels=channels)

    return sorting, td


def sorting_from_spikeinterface(
    sorting,
    recording=None,
    determine_channels=True,
    template_cfg=unshifted_raw_template_cfg,
    n_jobs=0,
):
    sv = sorting.to_spike_vector()
    return sorting_from_times_labels(
        sv["sample_index"],
        sv["unit_index"],
        sampling_frequency=sorting.sampling_frequency,
        recording=recording,
        determine_channels=determine_channels,
        template_cfg=template_cfg,
        n_jobs=n_jobs,
    )


def _same(x):
    return x


def load_dartsort_step_sortings(
    sorting_dir,
    load_simple_features=False,
    load_feature_names=("times_seconds",),
    detection_h5_names=(
        "subtraction.h5",
        "threshold.h5",
        "universal.h5",
        "matching0.h5",
    ),
    detection_h5_path: Path | str | None = None,
    step_format="refined{step}",
    recluster_format="recluster{step}",
    mtime_gap_minutes=0,
    name_formatter=None,
) -> Generator[tuple[str, DARTsortSorting], None, None]:
    """Returns list of step names and sortings, ordered.

    The mtime thing is trying to prevent reading hdf5 files which are in active
    use, although its not a guarantee... h5 locking... need to figure it out.
    """
    mtime_dt = mtime_gap_minutes * 60 if mtime_gap_minutes else 0
    if detection_h5_path is None:
        for dh5n in detection_h5_names:
            detection_h5_path = sorting_dir / dh5n
            if not detection_h5_path.exists():
                continue
            if mtime_dt:
                age = time.time() - detection_h5_path.stat().st_mtime
                if age < mtime_dt:
                    continue
            h5s = [detection_h5_path]
            break
        else:
            h5s = []
    else:
        h5s = [detection_h5_path]

    for j in range(1, 100):
        mh5 = sorting_dir / f"matching{j}.h5"
        if not mh5.exists():
            break
        if mtime_dt:
            if time.time() - mh5.stat().st_mtime < mtime_dt:
                break
        h5s.append(mh5)

    # let's check that there is at least something to do...
    labels_npys = sorting_dir.glob("*_labels.npy")
    relevant_files = [sorting_dir / "dartsort_sorting.npz", *labels_npys, *h5s[1:]]
    if not any(f.exists() for f in relevant_files):
        h5s = []

    if name_formatter is None:
        name_formatter = _same

    for step, h5 in enumerate(h5s):
        if not h5.exists():
            continue
        st0 = DARTsortSorting.from_peeling_hdf5(
            h5,
            load_simple_features=load_simple_features,
            load_feature_names=load_feature_names,
        )

        # initial clust or later step res?
        if h5 == h5s[0]:
            npy = sorting_dir / "initial_labels.npy"
            if npy.exists():
                yield (
                    name_formatter("initial"),
                    dataclasses.replace(st0, labels=np.load(npy)),
                )
            else:
                warnings.warn(f"Initial {npy} does not exist.")
                yield None, None

            other_initial_npys = sorted(sorting_dir.glob("initial_*_labels.npy"))
            for npy in other_initial_npys:
                npy = sorting_dir / npy
                stem = npy.stem.removesuffix("_labels")
                yield (
                    name_formatter(stem),
                    dataclasses.replace(st0, labels=np.load(npy)),
                )
        else:
            yield (name_formatter(h5.stem), st0)

        # reclustering, if applicable
        reclustr = recluster_format.format(step=step)
        recluster_npy = sorting_dir / f"{reclustr}_labels.npy"
        if recluster_npy.exists():
            yield (
                name_formatter(reclustr),
                dataclasses.replace(st0, labels=np.load(recluster_npy)),
            )

        # refinement steps
        stepstr = step_format.format(step=step)
        for npy in sorted(sorting_dir.glob(f"{stepstr}refstep*.npy")):
            yield (
                name_formatter(npy.stem.removesuffix("_labels")),
                dataclasses.replace(st0, labels=np.load(npy)),
            )

        # refinement final
        npy = sorting_dir / f"{stepstr}_labels.npy"
        if npy.exists():
            yield (
                name_formatter(stepstr),
                dataclasses.replace(st0, labels=np.load(npy)),
            )


def load_dartsort_step_unit_info_dataframes(
    sorting_dir,
    gt_analysis,
    recording,
    sorting_name=None,
    detection_h5_names=(
        "subtraction.h5",
        "threshold.h5",
        "universal.h5",
        "matching0.h5",
    ),
    detection_h5_path: Path | str | None = None,
    step_format="refined{step}",
):
    step_sortings = load_dartsort_step_sortings(
        sorting_dir,
        detection_h5_names=detection_h5_names,
        detection_h5_path=detection_h5_path,
        step_format=step_format,
    )
    for step_ix, (step_name, step_sorting) in enumerate(step_sortings):
        name = f"{sorting_name}: {step_name}" if sorting_name else step_name
        step_analysis = analysis.DARTsortAnalysis(step_sorting, recording, name=name)
        step_comparison = comparison.DARTsortGroundTruthComparison(
            gt_analysis, step_analysis
        )
        df = step_comparison.unit_info_dataframe()
        df["stepix"] = step_ix
        df["stepname"] = step_name
        yield df


def config_grid(
    common_params: dict | None = None,
    name_prefix="",
    config_cls: Any = DeveloperConfig,
    **grid_params,
):
    """Configuration grid search helper

    Generate a set product of aptly named configuration dictionaries or objects.

    Example usage:

    ```python
    config_grid(
        {'save_intermediate_labels': True, 'n_jobs_cpu': 2},
        badness={".1":dict(denoiser_badness_factor=0.1),".25":dict(denoiser_badness_factor=0.25)},
        crit={"hell":dict(merge_criterion="heldout_loglik"),"helb":dict(merge_criterion="heldout_elbo")},
    )
    => nice nested grid with names
    ```

    Arguments
    ---------
    common_params: dict
        Arguments shared across all settings.
    name_prefix: str
        Will appear, with _, at the start of all keys in the returned dict.
    config_cls: class or None
    **grid_params
        Each key-value pair here should be as follows. The key is like the
        "tag" for this setting. The value is a dict whose keys are setting
        nicknames and values are themselves dicts containing the actual
        configurations. See the example above.

    Returns
    -------
    grid: dict[str, config_cls]
        My length is product(v for k, v in grid_params.items()).
    """
    if common_params is None:
        common_params = {}

    # base case (won't actually end up in the final grid)
    grid = [(name_prefix, common_params)]

    for param_nickname, param_settings in grid_params.items():
        new_grid = []

        for setting_nickname, kw in param_settings.items():
            # induction
            for cur_name, cur_params in grid:
                prefix = f"{cur_name}_" if cur_name else ""
                new_name = f"{prefix}{param_nickname}{setting_nickname}"
                # new overrides old if overlapping
                new_grid.append((new_name, cur_params | kw))

        grid = new_grid

    # convert to dict of configs
    if config_cls is not None:
        grid = {name: config_cls(**kw) for name, kw in grid}
    else:
        grid = dict(grid)

    return grid
