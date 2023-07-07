from .base import BasePeeler


class Subtraction(BasePeeler):
    def __init__(
        self,
        recording,
        channel_index,
        subtracted_waveform_pipeline,
        cleaned_waveform_pipeline,
        chunk_length_samples=30_000,
        n_seconds_fit=40,
        fit_subsampling_random_state=0,
    ):
        super().___init__(
            recording,
            channel_index,
            cleaned_waveform_pipeline,
            chunk_length_samples=chunk_length_samples,
            n_seconds_fit=n_seconds_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )
        self.add_module(
            "subtracted_waveform_pipeline", subtracted_waveform_pipeline
        )

    def peel_chunk(self, traces):
        pass
