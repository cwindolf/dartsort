from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)


class PeeledResidualRecording(BasePreprocessor):

    def __init__(self, peeler):
        # this does not handle segments at all!
        super().__init__(peeler.recording, dtype=peeler.dtype)
        assert peeler.recording.get_num_segments() == 1
        self.add_recording_segment(PeeledResidualSegment(peeler))
        self._kwargs = dict(peeler=peeler)


class PeeledResidualSegment(BasePreprocessorSegment):
    def __init__(self, peeler):
        self.peeler = peeler

    def get_traces(self, start_frame, end_frame, channel_indices):
        # right now the peeler dives into segment 0 in process_chunk
        # i guess that's something to think about...
        stuff = self.peeler.process_chunk(
            chunk_start_samples=start_frame,
            chunk_end_samples=end_frame,
            return_residual=True,
            skip_features=True,
        )
        return stuff["residual"].numpy(force=True)[:, channel_indices]
