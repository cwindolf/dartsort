import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.extractors import NumpySorting
from spikeinterface.generation.drift_tools import InjectDriftingTemplatesRecording
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor, BasePreprocessorSegment)

from ..templates import TemplateData
from .analysis import DARTsortAnalysis
from .data_util import DARTsortSorting

def get_drifty_hybrid_recording(
                recording,
                 templates,
                 motion_estimate,
                 seed=None,
                 firing_rates=None,
                 peak_channels=None,
                 amplitude_scale_std=0.1,
):
    """

    :param: recording
    :param: templates object
    :param: motion estimate object
    :param: firing_rates
    :param: peak_channels
    :param: amplitude_factor
    """

    num_units = templates.num_units
    
    if seed is None:
        _rg = np.random.default_rng()
        seed = _rg.integers(0)
    rg = np.random.default_rng(seed=seed)

    if peak_channels is None:
        raise(NotImplementedError, "Autodetection of peak channels not implemented yet.")
        
    # Default firing rates drawn uniformly from 1-10Hz
    if firing_rates:
        assert firing_rates.shape[0] == num_units, "Number of firing rates must match number of units in templates."
    else:
        firing_rates = rg.uniform(1.0, 10.0, num_units)

    spike_trains = [refractory_poisson_spike_train(
                        firing_rates[i], 
                        recording.get_num_samples(), 
                        spike_length_samples=templates.num_samples,
                        seed=seed
                    ) for i in range(num_units)
                ]

    spike_labels = np.repeat(np.arange(num_units) + 1, np.array([spike_trains[i].shape[0] for i in range(num_units)]))

    sorting = NumpySorting.from_times_labels(
        [np.concatenate(spike_trains)], 
        [spike_labels], 
        sampling_frequency=recording.get_sampling_frequency()
    )

    
    # Default amplitude scalings for spikes drawn from gamma
    shape = 1. / (amplitude_scale_std ** 1.5)
    amplitude_factor = rg.gamma(shape, scale=1./(shape-1), size=sorting.to_spike_vector().shape)
        
    depths = recording.get_probe().contact_positions[:, 1][peak_channels]
    motion_times_s = np.arange(int(recording.get_duration())+1)

    disp_y = motion_estimate.disp_at_s(motion_times_s, depths, grid=True)

    disp = np.zeros((motion_times_s.shape[0], 2, num_units))
    disp[:, 1, :] = disp_y.swapaxes(0, 1)


    return InjectDriftingTemplatesRecording(
        sorting=sorting,
        drifting_templates=templates,
        parent_recording=recording,
        displacement_vectors=[disp],
        displacement_sampling_frequency=1.0,
        displacement_unit_factor=np.eye(num_units),
        amplitude_factor=amplitude_factor
    ), sorting


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
