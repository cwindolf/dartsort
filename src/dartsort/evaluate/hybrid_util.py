import dataclasses
from pathlib import Path
from typing import Generator, Any
import warnings

import numpy as np
from probeinterface import Probe
from scipy.spatial import KDTree
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


def greedy_match(gt_coords, test_coords, max_val=1.0, dx=1.0 / 30, workers=-1, p=2.0):
    assignments = np.full(len(test_coords), -1)
    gt_unmatched = np.ones(len(gt_coords), dtype=bool)

    for j, thresh in enumerate(
        tqdm(np.arange(0.0, max_val + dx + 2e-5, dx), desc="match")
    ):
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

    return assignments


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
    sorting = DARTsortSorting(
        times_samples=times_samples,
        channels=channels,
        labels=labels,
        sampling_frequency=sampling_frequency,
    )

    if not determine_channels:
        return sorting

    assert recording is not None

    _, labels_flat = np.unique(labels, return_inverse=True)
    sorting = DARTsortSorting(
        times_samples=times_samples,
        channels=channels,
        labels=labels_flat,
        sampling_frequency=sorting.sampling_frequency,
    )
    template_cfg = dataclasses.replace(template_cfg, spikes_per_unit=spikes_per_unit)
    comp_cfg = ComputationConfig(n_jobs_cpu=n_jobs, n_jobs_gpu=n_jobs)
    td = TemplateData.from_config(
        recording,
        sorting,
        template_cfg,
        computation_cfg=comp_cfg,
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

    sorting = DARTsortSorting(
        times_samples=times_samples,
        channels=channels,
        labels=labels_flat,
        sampling_frequency=sorting.sampling_frequency,
    )
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


def load_dartsort_step_sortings(
    sorting_dir,
    load_simple_features=False,
    load_feature_names=("times_seconds",),
    detection_h5_names=("subtraction.h5", "threshold.h5", "universal.h5", "matching0.h5"),
    detection_h5_path: Path | str | None = None,
    step_format="refined{step}",
    recluster_format="recluster{step}",
) -> Generator[tuple[str, DARTsortSorting], None, None]:
    """Returns list of step names and sortings, ordered."""
    if detection_h5_path is None:
        for dh5n in detection_h5_names:
            detection_h5_path = sorting_dir / dh5n
            if detection_h5_path.exists():
                h5s = [detection_h5_path]
                break
        else:
            h5s = []
    else:
        h5s = [detection_h5_path]

    for j in range(1, 100):
        if (sorting_dir / f"matching{j}.h5").exists():
            h5s.append(sorting_dir / f"matching{j}.h5")
        else:
            break

    # let's check that there is at least something to do...
    labels_npys = sorting_dir.glob("*_labels.npy")
    relevant_files = [sorting_dir / "dartsort_sorting.npz", *labels_npys, *h5s[1:]]
    if not any(f.exists() for f in relevant_files):
        h5s = []

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
                yield ("initial", dataclasses.replace(st0, labels=np.load(npy)))
            else:
                warnings.warn(f"Initial {npy} does not exist.")
                yield None, None
        else:
            yield (h5.stem, st0)

        # reclustering, if applicable
        reclustr = recluster_format.format(step=step)
        recluster_npy = sorting_dir / f"{reclustr}_labels.npy"
        if recluster_npy.exists():
            yield (reclustr, dataclasses.replace(st0, labels=np.load(recluster_npy)))

        # refinement steps
        stepstr = step_format.format(step=step)
        for npy in sorted(sorting_dir.glob(f"{stepstr}refstep*.npy")):
            yield (
                npy.stem.removesuffix("_labels"),
                dataclasses.replace(st0, labels=np.load(npy)),
            )

        # refinement final
        npy = sorting_dir / f"{stepstr}_labels.npy"
        if npy.exists():
            yield (stepstr, dataclasses.replace(st0, labels=np.load(npy)))


def load_dartsort_step_unit_info_dataframes(
    sorting_dir,
    gt_analysis,
    recording,
    sorting_name=None,
    detection_h5_names=("subtraction.h5", "threshold.h5", "universal.h5", "matching0.h5"),
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
    common_params: dict | None=None,
    name_prefix="",
    config_cls: Any=DeveloperConfig,
    **grid_params,
):
    """Configuration grid search helper

    Generate a set product of aptly named configuration dictionaries or objects.

    Example usage:

    ```python
    dartsort_config_grid(
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
