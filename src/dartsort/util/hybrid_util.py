import dataclasses
import numpy as np
from tqdm.auto import tqdm
import warnings
from spikeinterface.extractors import NumpySorting
from spikeinterface.generation.drift_tools import InjectDriftingTemplatesRecording, DriftingTemplates, move_dense_templates
from probeinterface import Probe
from scipy.spatial import KDTree
from scipy.sparse import csgraph, coo_array

from ..templates import TemplateData
from .data_util import DARTsortSorting
from ..config import unshifted_raw_template_config, ComputationConfig


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
    amplitude_factor=None
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
        (central_disp_index,) = np.flatnonzero(
            np.all(templates.displacements == 0, 1)
        )
        central_templates = templates.templates_array_moved[central_disp_index]
        peak_channels = np.ptp(central_templates, 1).argmax(1)

    if sorting is None:
        sorting = get_sorting(num_units, recording, firing_rates=firing_rates, rg=rg, spike_length_samples=templates.num_samples)
    n_spikes = sorting.count_total_num_spikes()

    # Default amplitude scalings for spikes drawn from gamma
    if amplitude_factor is None:
        if amplitude_scale_std:
            shape = 1. / (amplitude_scale_std ** 1.5)
            amplitude_factor = rg.gamma(shape, scale=1./(shape-1), size=n_spikes)
        else:
            amplitude_factor = np.ones(n_spikes)

    depths = recording.get_probe().contact_positions[:, 1][peak_channels]
    t_start = recording.sample_index_to_time(0)
    t_end = recording.sample_index_to_time(recording.get_num_samples() - 1)
    motion_times_s = np.arange(t_start, t_end, step=1.0 / displacement_sampling_frequency)

    disp_y = motion_estimate.disp_at_s(motion_times_s, depths, grid=True)

    disp = np.zeros((motion_times_s.shape[0], 2, num_units))
    disp[:, 1, :] = disp_y.T
    # this tricks SI into using one displacement per unit
    displacement_unit_factor = np.eye(num_units)

    if not sorting.check_serializability(type='json'):
        warnings.warn("Your sorting is not serializable, which could lead to problems later.")

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


def get_sorting(num_units, recording, firing_rates=None, rg=0, nbefore=42, spike_length_samples=128):
    rg = np.random.default_rng(rg)

    # Default firing rates drawn uniformly from 1-10Hz
    if firing_rates is not None:
        assert firing_rates.shape[0] == num_units, "Number of firing rates must match number of units in templates."
    else:
        firing_rates = rg.uniform(1.0, 10.0, num_units)

    spike_trains = [
        refractory_poisson_spike_train(
            firing_rates[i], 
            recording.get_num_samples(), 
            trough_offset_samples=nbefore,
            spike_length_samples=spike_length_samples,
            seed=rg,
        ) for i in range(num_units)
    ]

    spike_labels = np.repeat(np.arange(num_units) + 1, np.array([spike_trains[i].shape[0] for i in range(num_units)]))

    sorting = NumpySorting.from_times_labels(
        [np.concatenate(spike_trains)], 
        [spike_labels], 
        sampling_frequency=recording.sampling_frequency,
    )

    return sorting


def simulate_spike_trains(
    n_units,
    duration_samples,
    spike_rates_range_hz=(1.0, 10.0),
    refractory_samples=40,
    trough_offset_samples=42,
    spike_length_samples=121,
    sampling_frequency=30000.0,
    rg=0,
):
    rg = np.random.default_rng(rg)

    labels = []
    times_samples = []
    for u in range(n_units):
        rate_hz = rg.uniform(*spike_rates_range_hz)
        st = refractory_poisson_spike_train(
            rate_hz,
            duration_samples,
            rg=rg,
            refractory_samples=refractory_samples,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            sampling_frequency=sampling_frequency,
        )
        labels.append(np.broadcast_to([u], st.size))
        times_samples.append(st)

    times_samples = np.concatenate(times_samples)
    order = np.argsort(times_samples)
    times_samples = times_samples[order]
    labels = np.concatenate(labels)[order]

    return times_samples, labels


def refractory_poisson_spike_train(
    rate_hz,
    duration_samples,
    seed=0,
    refractory_samples=40,
    trough_offset_samples=42,
    spike_length_samples=121,
    sampling_frequency=30000.0,
    overestimation=1.5,
):
    """Sample a refractory Poisson spike train

    Arguments
    ---------
    rate : float
        Spikes / second, well, except it'll be slower due to refractoriness.
    duration : float
    """
    rg = np.random.default_rng(seed)

    seconds_per_sample = 1.0 / sampling_frequency
    refractory_s = refractory_samples * seconds_per_sample
    duration_s = duration_samples * seconds_per_sample

    # overestimate the number of spikes needed
    mean_interval_s = 1.0 / rate_hz
    estimated_spike_count = int((duration_s / mean_interval_s) * overestimation)

    # generate interspike intervals
    intervals = rg.exponential(
        scale=mean_interval_s, size=estimated_spike_count
    )
    intervals += refractory_s
    intervals_samples = np.floor(intervals * sampling_frequency).astype(int)

    # determine spike times and restrict to ones which we can actually
    # add into / read from a recording with this duration and trough offset
    spike_samples = np.cumsum(intervals_samples)
    max_spike_time = duration_samples - (
        spike_length_samples - trough_offset_samples
    )
    # check that we overestimated enough
    assert spike_samples.max() > max_spike_time
    valid = spike_samples == spike_samples.clip(
        trough_offset_samples, max_spike_time
    )
    spike_samples = spike_samples[valid]
    assert spike_samples.size

    return spike_samples


def piecewise_refractory_poisson_spike_train(rates, bins, binsize_samples, **kwargs):
    """
    Returns a spike train with variable firing rate using refractory_poisson_spike_train().

    :param rates: list of firing rates in Hz
    :param bins: bin starting samples (same shape as rates)
    :param binsize_samples: number of samples per bin
    :param **kwargs: kwargs to feed to refractory_poisson_spike_train()
    """
    st = []
    for rate, bin in zip(rates, bins):
        if rate < 0.1:
            continue
        binst = refractory_poisson_spike_train(rate, binsize_samples, **kwargs)
        st.append(bin + binst)
    st = np.concatenate(st)
    return st


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


def closest_clustering(gt_st, peel_st, geom=None, match_dt_ms=0.1, match_radius_um=0.0, p=2.0):
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


def greedy_match(gt_coords, test_coords, max_val=1.0, dx=1./30, workers=-1, p=2.0):
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
        times_samples=times_samples, channels=channels, labels=labels, sampling_frequency=sampling_frequency
    )

    if not determine_channels:
        return sorting

    assert recording is not None

    _, labels_flat = np.unique(labels, return_inverse=True)
    sorting = DARTsortSorting(
        times_samples=times_samples, channels=channels, labels=labels_flat, sampling_frequency=sorting.sampling_frequency
    )
    template_config = dataclasses.replace(template_config, spikes_per_unit=spikes_per_unit)
    comp_cfg = ComputationConfig(n_jobs_cpu=n_jobs, n_jobs_gpu=n_jobs)
    td = TemplateData.from_config(recording, sorting, template_config, with_locs=False, computation_config=comp_cfg)

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
        times_samples=times_samples, channels=channels, labels=labels_flat, sampling_frequency=sorting.sampling_frequency
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
        sv['sample_index'], sv['unit_index'], sampling_frequency=sorting.sampling_frequency, recording=recording, determine_channels=determine_channels, template_config=template_config, n_jobs=n_jobs
    )
