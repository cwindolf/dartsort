from .base import BasePeeler


class GrabAndLocalize(BasePeeler):
    def __init__(
        self,
        recording,
        channel_index,
        waveform_pipeline,
        spike_times,
        spike_channels=None,
        spike_labels=None,
        templates=None,
    ):
        super().__init__(recording, channel_index, waveform_pipeline)
