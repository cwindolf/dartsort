from pathlib import Path
import tempfile

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment, read_binary_folder
from spikeinterface.core.recording_tools import get_chunk_with_margin
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)

from ..util.noise_util import StationaryFactorizedNoise
from .simlib import default_temporal_kernel_npy, rbf_kernel_sqrt, generate_geom


def get_background_recording(
    noise_recording_folder,
    duration_samples,
    probe_kwargs=None,
    noise_kind="stationary_factorized_rbf",
    noise_spatial_kernel_bandwidth=15.0,
    noise_temporal_kernel: np.ndarray | str | Path = default_temporal_kernel_npy,
    noise_fft_t=121,
    random_seed=0,
    dtype="float16",
    sampling_frequency=30_000.0,
    white_noise_scale=1.0,
    n_jobs=1,
    overwrite=False,
):
    if (noise_recording_folder / "binary.json").exists():
        return read_binary_folder(noise_recording_folder)

    geom = generate_geom(**probe_kwargs or {})
    n_channels = len(geom)
    scale = 0.0 if noise_kind == "zero" else white_noise_scale
    recording = WhiteNoiseRecording(
        duration_samples,
        n_channels,
        sampling_frequency,
        scale=scale,
        random_seed=random_seed,
        dtype=dtype,
    )
    recording.set_dummy_probe_from_locations(geom)

    if noise_kind == "zero":
        return recording

    if noise_kind == "white":
        return recording.save_to_folder(noise_recording_folder, n_jobs=1)

    assert noise_kind == "stationary_factorized_rbf"
    if not isinstance(noise_temporal_kernel, np.ndarray):
        noise_temporal_kernel = np.load(noise_temporal_kernel)
    assert isinstance(noise_temporal_kernel, np.ndarray)
    assert noise_temporal_kernel.ndim == 1
    block_size = len(noise_temporal_kernel) * 2 - 1
    spatial_std, spatial_vt = rbf_kernel_sqrt(
        geom, bandwidth=noise_spatial_kernel_bandwidth
    )
    noise_temporal_kernel = noise_temporal_kernel.astype(spatial_vt.dtype) + 0j
    noise = StationaryFactorizedNoise(
        spatial_std=spatial_std,
        vt_spatial=spatial_vt,
        kernel_fft=noise_temporal_kernel,
        block_size=block_size,
        t=noise_fft_t,
    )

    # white noise must be cached before convolving
    with tempfile.TemporaryDirectory() as tdir:
        recording = recording.save_to_folder(Path(tdir) / "noiserecording", n_jobs=1)
        recording = UnwhitenPreprocessor(noise, recording)
        recording = recording.save_to_folder(
            noise_recording_folder, n_jobs=n_jobs, pool_engine="thread", overwrite=overwrite
        )

    return recording


class WhiteNoiseRecording(BaseRecording):
    def __init__(
        self,
        duration_samples,
        n_channels,
        sampling_frequency,
        scale=1.0,
        n_segments=1,
        random_seed=0,
        dtype="float32",
        channel_ids=None,
    ):
        if channel_ids is None:
            channel_ids = np.arange(n_channels)
        else:
            channel_ids = np.asarray(channel_ids)
            assert channel_ids.shape == (n_channels,)
        super().__init__(sampling_frequency, channel_ids, dtype)
        self._serializability["json"] = False
        self._serializability["pickle"] = False
        segment = WhiteNoiseRecordingSegment(
            duration_samples, n_channels, sampling_frequency, dtype, random_seed, scale
        )
        self.add_recording_segment(segment)


class WhiteNoiseRecordingSegment(BaseRecordingSegment):
    def __init__(
        self, duration_samples, n_channels, sampling_frequency, dtype, rg, scale
    ):
        super().__init__(sampling_frequency=sampling_frequency)
        self.duration_samples = duration_samples
        self.chans_arange = np.arange(n_channels)
        self.sampling_frequency = sampling_frequency
        self.dtype = dtype
        self.rg = np.random.default_rng(rg)
        self.scale = scale

    def get_num_samples(self):
        return self.duration_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        start_frame = start_frame or 0
        end_frame = self.get_num_samples() if end_frame is None else end_frame
        channel_indices = slice(None) if channel_indices is None else channel_indices
        n_frames = end_frame - start_frame
        n_channels = self.chans_arange[channel_indices].size
        if not self.scale:
            traces = np.zeros((n_frames, n_channels), dtype=self.dtype)
        else:
            traces = self.rg.normal(scale=self.scale, size=(n_frames, n_channels))
        return traces.astype(self.dtype, copy=False)


class UnwhitenPreprocessor(BasePreprocessor):
    def __init__(self, noise, recording):
        super().__init__(recording)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(
                UnwhitenPreprocessorSegment(parent_segment, noise)
            )


class UnwhitenPreprocessorSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, noise):
        super().__init__(parent_recording_segment)
        self.noise = noise

    def get_traces(self, start_frame, end_frame, channel_indices):
        start_frame = start_frame or 0
        end_frame = self.get_num_samples() if end_frame is None else end_frame
        traces, lm, rm = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            margin=self.noise.margin,
            add_reflect_padding=True,
            channel_indices=None,
        )
        assert lm == rm == self.noise.margin
        dtype = traces.dtype
        traces = self.noise.unwhiten(traces).numpy(force=True)
        traces = traces.astype(dtype, copy=False)
        assert traces.shape[0] == end_frame - start_frame
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces
