import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor, BasePreprocessorSegment)


class HybridRecording(BasePreprocessor):
    name = "hybrid_recording"
    installed = True

    def __init__(
        self,
        recording: BaseRecording,
        templates: np.ndarray,
        times_samples=None,
        labels=None,
        template_indices=None,
        trough_offset_samples=42,
        spike_train_kwargs=None,
        random_seed=0,
        dtype="float32",
    ):
        """A basic hybrid recording factory"""
        assert templates.ndim == 3
        assert templates.shape[2] == recording.get_num_channels()
        assert 0 <= trough_offset_samples < templates.shape[1]
        assert recording.get_num_segments() == 1

        if times_samples is None:
            assert labels is None and template_indices is None
            if spike_train_kwargs is None:
                spike_train_kwargs = {}
            spike_train_kwargs["trough_offset_samples"] = trough_offset_samples
            spike_train_kwargs["spike_length_samples"] = templates.shape[1]
            spike_train_kwargs["rg"] = random_seed
            times_samples, labels = simulate_spike_trains(
                n_units=templates.shape[0],
                duration_samples=recording.get_num_samples(),
                **spike_train_kwargs,
            )
        else:
            assert labels is not None
            times_samples = np.asarray(times_samples)
            labels = np.asarray(labels)

        assert times_samples.ndim == 1
        assert np.all(np.diff(times_samples) >= 0)
        if template_indices is None:
            template_indices = labels
        else:
            assert template_indices.max() < templates.shape[0]
        assert times_samples.shape == labels.shape == template_indices.shape

        self.times_samples = times_samples
        self.labels = labels
        self.template_indices = template_indices
        self.templates = templates

        dtype_ = dtype
        if dtype_ is None:
            dtype_ = recording.dtype
        assert dtype_ == templates.dtype
        BasePreprocessor.__init__(self, recording, dtype=dtype_)
        for parent_segment in recording._recording_segments:
            rec_segment = HybridRecordingSegment(
                parent_segment,
                times_samples,
                template_indices,
                templates,
                trough_offset_samples=trough_offset_samples,
                dtype=dtype_,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            times_samples=times_samples,
            labels=labels,
            template_indices=template_indices,
            templates=templates,
            trough_offset_samples=trough_offset_samples,
            spike_train_kwargs=spike_train_kwargs,
            random_seed=random_seed,
            dtype=dtype_,
        )


class HybridRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment: BaseRecordingSegment,
        times_samples,
        template_indices,
        templates,
        trough_offset_samples=0,
        dtype="float32",
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.times_samples = times_samples
        self.template_indices = template_indices
        self.templates = templates
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = templates.shape[1]
        self.post_trough_samples = (
            self.spike_length_samples - self.trough_offset_samples
        )
        self.margin_left = self.spike_length_samples
        self.margin_right = self.spike_length_samples
        self.dtype = dtype

        # this is a helper for indexing operations below
        self.time_domain_offset = (
            np.arange(self.spike_length_samples)
            - self.trough_offset_samples
            + self.margin_left
        )

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        parent_traces = self.parent_recording_segment.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=slice(None),
        )
        # we want to copy since we will modify and can't have a memmap
        # and, we will use np.pad to add margin since we don't care
        # about the edges
        traces_pad = np.pad(
            parent_traces.astype(self.dtype, copy=False),
            [(self.margin_left, self.margin_right), (0, 0)],
        )

        # get spike times_samples/template_indices in this part, offset by start frame
        ix_low = np.searchsorted(
            self.times_samples, start_frame - self.post_trough_samples, side="left"
        )
        ix_high = np.searchsorted(
            self.times_samples, end_frame + self.trough_offset_samples, side="right"
        )
        times_samples = self.times_samples[ix_low:ix_high] - start_frame
        template_indices = self.template_indices[ix_low:ix_high]

        # just add with a for loop
        for t, c in zip(times_samples, template_indices):
            traces_pad[t + self.time_domain_offset] += self.templates[c]

        traces = traces_pad[self.margin_left : traces_pad.shape[0] - self.margin_right]
        return traces


hybrid_recording = define_function_from_class(
    source_class=HybridRecording, name="hybrid_recording"
)


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
    rg=0,
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
    rg = np.random.default_rng(rg)

    seconds_per_sample = 1.0 / sampling_frequency
    refractory_s = refractory_samples * seconds_per_sample
    duration_s = duration_samples * seconds_per_sample

    # overestimate the number of spikes needed
    mean_interval_s = 1.0 / rate_hz
    estimated_spike_count = int((duration_s / mean_interval_s) * overestimation)

    # generate interspike intervals
    intervals = rg.exponential(scale=mean_interval_s, size=estimated_spike_count)
    intervals += refractory_s
    intervals_samples = np.floor(intervals * sampling_frequency).astype(int)

    # determine spike times and restrict to ones which we can actually
    # add into / read from a recording with this duration and trough offset
    spike_samples = np.cumsum(intervals_samples)
    max_spike_time = duration_samples - (spike_length_samples - trough_offset_samples)
    # check that we overestimated enough
    assert spike_samples.max() > max_spike_time
    valid = spike_samples == spike_samples.clip(trough_offset_samples, max_spike_time)
    spike_samples = spike_samples[valid]
    assert spike_samples.size

    return spike_samples
