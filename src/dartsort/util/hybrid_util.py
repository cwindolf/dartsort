import dataclasses
import numpy as np
from tqdm.auto import tqdm
import warnings
from spikeinterface.extractors import NumpySorting
from spikeinterface.generation.drift_tools import (
    InjectDriftingTemplatesRecording,
    DriftingTemplates,
    move_dense_templates,
)
from probeinterface import Probe
from scipy.spatial import KDTree

from ..templates import TemplateData
from .data_util import DARTsortSorting
from ..config import unshifted_raw_template_config, ComputationConfig
from . import simkit


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
    labels = greedy_match(gt_pos, peel_pos, dx=1.0 / frames_per_ms)
    labels[labels >= 0] = gt_st.labels[labels[labels >= 0]]

    return dataclasses.replace(peel_st, labels=labels)


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
    template_config=unshifted_raw_template_config,
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
    template_config = dataclasses.replace(
        template_config, spikes_per_unit=spikes_per_unit
    )
    comp_cfg = ComputationConfig(n_jobs_cpu=n_jobs, n_jobs_gpu=n_jobs)
    td = TemplateData.from_config(
        recording,
        sorting,
        template_config,
        with_locs=False,
        computation_config=comp_cfg,
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
    template_config=unshifted_raw_template_config,
    n_jobs=0,
):
    sv = sorting.to_spike_vector()
    return sorting_from_times_labels(
        sv["sample_index"],
        sv["unit_index"],
        sampling_frequency=sorting.sampling_frequency,
        recording=recording,
        determine_channels=determine_channels,
        template_config=template_config,
        n_jobs=n_jobs,
    )
