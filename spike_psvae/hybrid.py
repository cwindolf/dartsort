import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)
from spikeinterface.core.core_tools import define_function_from_class


class HybridRecording(BasePreprocessor):
    name = "hybrid_recording"
    installed = True

    def __init__(
        self,
        recording: BaseRecording,
        times,
        labels,
        templates: np.ndarray,
        trough_offset_samples=42,
        dtype="float32",
    ):
        """A basic hybrid recording factory"""
        times = np.asarray(times)
        labels = np.asarray(labels)
        assert times.ndim == 1
        assert times.shape == labels.shape
        assert labels.max() < templates.shape[0]
        assert templates.ndim == 3
        assert templates.shape[2] == recording.get_num_channels()
        assert 0 <= trough_offset_samples < templates.shape[1]
        assert recording.get_num_segments() == 1
        assert np.all(np.diff(times) >= 0)

        dtype_ = dtype
        if dtype_ is None:
            dtype_ = recording.dtype
        BasePreprocessor.__init__(self, recording, dtype=dtype_)
        for parent_segment in recording._recording_segments:
            rec_segment = HybridRecordingSegment(
                parent_segment,
                times,
                labels,
                templates,
                trough_offset_samples=trough_offset_samples,
                dtype=dtype_,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            times=times,
            labels=labels,
            templates=templates,
            trough_offset_samples=trough_offset_samples,
            dtype=dtype_,
        )


class HybridRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment: BaseRecordingSegment,
        times,
        labels,
        templates,
        trough_offset_samples=0,
        dtype="float32",
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.times = times
        self.labels = labels
        self.templates = templates
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = templates.shape[1]
        self.post_trough_samples = self.spike_length_samples - self.trough_offset_samples
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

        # get spike times/labels in this part, offset by start frame
        ix_low = np.searchsorted(
            self.times, start_frame - self.post_trough_samples, side="left"
        )
        ix_high = np.searchsorted(
            self.times, end_frame + self.trough_offset_samples, side="right"
        )
        times = self.times[ix_low:ix_high] - start_frame
        labels = self.labels[ix_low:ix_high]

        # just add with a for loop
        for t, c in zip(times, labels):
            traces_pad[t + self.time_domain_offset] += self.templates[c]

        traces = traces_pad[self.margin_left : traces_pad.shape[0] - self.margin_right]
        return traces


# function for API
hybrid_recording = define_function_from_class(
    source_class=HybridRecording, name="hybrid_recording"
)
