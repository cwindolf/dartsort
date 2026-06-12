"""Grab and featurize events at known times."""

from typing import Mapping

import numpy as np
import torch
from spikeinterface import BaseRecording

from ..transform import WaveformPipeline
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    FeaturizationConfig,
    FitSamplingConfig,
    WaveformConfig,
    default_waveform_cfg,
)
from ..util.spiketorch import _nonzero_static, grab_spikes
from ..util.waveform_util import make_channel_index
from .peel_base import (
    BasePeeler,
    PeelingBatchResult,
    SpikeDataset,
    peeling_empty_result,
)


class GrabAndFeaturize(BasePeeler):
    """Extract and featurize raw waveform snippets at known times."""

    peel_kind = "Grab and featurize"

    def __init__(
        self,
        recording,
        channel_index,
        featurization_pipeline,
        *,
        times_samples,
        fixed_properties: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        chunk_length_samples=30_000,
        fit_sampling_cfg: FitSamplingConfig = FitSamplingConfig(n_residual_snips=0),
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        batch_size: int = 2048,
        dtype=torch.float,
    ):
        fixed_properties = fixed_properties or {}
        fixed_property_keys = tuple(fixed_properties.keys())
        trough_offset_samples = waveform_cfg.trough_offset_samples(
            recording.sampling_frequency
        )
        spike_length_samples = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        self.batch_size = batch_size
        assert not fit_sampling_cfg.n_residual_snips
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=max(
                trough_offset_samples, spike_length_samples - trough_offset_samples
            ),
            waveform_cfg=waveform_cfg,
            fit_sampling_cfg=fit_sampling_cfg,
            fixed_property_keys=fixed_property_keys,
            dtype=dtype,
        )
        self.register_buffer("times_samples", torch.asarray(times_samples))
        for k, v in fixed_properties.items():
            self.register_buffer(k, torch.asarray(v))
        assert self.times_samples.ndim == 1
        assert self.times_samples.shape == self.channels.shape

    def out_datasets(self):
        datasets = super().out_datasets()
        datasets.append(
            SpikeDataset(name="indices", shape_per_spike=(), dtype=np.int64)
        )
        return datasets

    def process_chunk(
        self,
        chunk_start_samples: int,
        *,
        chunk_end_samples: int | None = None,
        return_residual: bool = False,
        skip_features: bool = False,
        n_resid_snips: int | None = None,
        to_cpu: bool = True,
    ) -> "PeelingBatchResult":
        """Override process_chunk to skip empties."""
        if chunk_end_samples is None:
            chunk_end_samples = min(
                self.recording.get_num_samples(),
                chunk_start_samples + self.chunk_length_samples,
            )
        t_clip = self.b.times_samples.clip(chunk_start_samples, chunk_end_samples - 1)
        in_chunk = self.b.times_samples == t_clip
        n_in_chunk = int(in_chunk.sum().item())
        if not n_in_chunk:
            return peeling_empty_result

        in_chunk = _nonzero_static(in_chunk, size=n_in_chunk)
        assert in_chunk.shape == (n_in_chunk, 1)
        in_chunk = in_chunk[:, 0]
        return_waveforms = not skip_features and bool(self.featurization_pipeline)

        chunk, chunk_end_samples_, left_margin, right_margin = self.get_chunk(
            chunk_start_samples, chunk_end_samples
        )
        assert chunk_end_samples == chunk_end_samples_

        batch_results = []
        for i0 in range(0, n_in_chunk, self.batch_size):
            i1 = min(n_in_chunk, i0 + self.batch_size)
            batch_in_chunk = in_chunk[i0:i1]
            batch_peel_result = self.peel_chunk(
                traces=chunk,
                chunk_start_samples=chunk_start_samples,
                left_margin=left_margin,
                right_margin=right_margin,
                return_residual=return_residual,
                in_chunk=batch_in_chunk,
            )
            assert batch_peel_result["n_spikes"] == i1 - i0
            batch_res = self.featurize_chunk_result(
                peel_result=batch_peel_result,
                to_cpu=to_cpu,
                return_waveforms=return_waveforms,
                chunk_start_samples=chunk_start_samples,
                chunk_end_samples=chunk_end_samples,
                device=chunk.device,
                n_resid_snips=n_resid_snips,
            )
            batch_results.append(batch_res)
        res = _cat_results(batch_results)
        return res

    @classmethod
    def from_config(
        cls,
        recording: BaseRecording,
        *,
        sorting: DARTsortSorting,
        fixed_property_keys: list[str] | None = None,
        chunk_length_samples: int = 30_000,
        waveform_cfg: WaveformConfig,
        featurization_cfg: FeaturizationConfig,
        sampling_cfg: FitSamplingConfig,
    ):
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, featurization_cfg.extract_radius, to_torch=True
        )
        featurization_pipeline = WaveformPipeline.from_config(
            geom=geom,
            channel_index=channel_index,
            featurization_cfg=featurization_cfg,
            waveform_cfg=waveform_cfg,
            sampling_frequency=recording.sampling_frequency,
        )
        fixed_property_keys = fixed_property_keys or []
        fixed_property_keys = list(fixed_property_keys)
        if "channels" not in fixed_property_keys:
            fixed_property_keys.append("channels")
        fixed_properties = {k: getattr(sorting, k) for k in fixed_property_keys}

        return cls(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            times_samples=sorting.times_samples,
            waveform_cfg=waveform_cfg,
            chunk_length_samples=chunk_length_samples,
            fixed_properties=fixed_properties,
            dtype=torch.float,
            fit_sampling_cfg=sampling_cfg,
        )

    def peel_chunk(
        self,
        traces: torch.Tensor,
        *,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
        in_chunk: torch.Tensor | None = None,
    ) -> PeelingBatchResult:
        assert not return_residual

        if in_chunk is None:
            max_t = chunk_start_samples + self.chunk_length_samples - 1
            in_chunk = self.b.times_samples == self.b.times_samples.clip(
                chunk_start_samples, max_t
            )
            (in_chunk,) = in_chunk.nonzero(as_tuple=True)

        if not in_chunk.numel():
            return peeling_empty_result

        chunk_left = chunk_start_samples - left_margin
        times_rel = self.b.times_samples[in_chunk] - chunk_left
        channels = self.b.channels[in_chunk]

        assert times_rel.ndim == 1
        assert channels.ndim == 1

        n_spikes = in_chunk.numel()
        res = PeelingBatchResult(
            n_spikes=n_spikes,
            indices=in_chunk,
            times_samples=self.b.times_samples[in_chunk],
            channels=channels,
        )
        if return_waveforms:
            res["collisioncleaned_waveforms"] = grab_spikes(
                traces,
                times_rel,
                channels,
                self.channel_index,
                trough_offset=self.trough_offset_samples,
                spike_length_samples=self.spike_length_samples,
                already_padded=False,
                pad_value=torch.nan,
            )
        dev = times_rel.device
        for k in self.fixed_property_keys:
            res[k] = getattr(self.b, k)[in_chunk].to(device=dev)
        return res


def _cat_results(results):
    # not ideal, should work on typing these things better, but no time now.
    assert len(results) > 0
    if len(results) == 1:
        return results[0]
    rdict: dict[str, int | float | list[torch.Tensor] | list[np.ndarray]] = {
        "n_spikes": 0
    }
    for res in results:
        for k, v in res.items():
            if k == "n_spikes":
                rdict[k] += v
            elif k == "chunk_center_s":
                rdict[k] = v
            elif isinstance(v, (int, float)):
                if k not in rdict:
                    rdict[k] = v
                else:
                    assert rdict[k] == v
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                if k not in rdict:
                    rdict[k] = []
                rdict[k].append(v)  # type: ignore
            else:
                assert False
    res = {}
    for k, v in rdict.items():
        if k in ("n_spikes", "chunk_center_s"):
            res[k] = v
        elif isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                res[k] = torch.concatenate(v)  # type: ignore
            elif isinstance(v[0], np.ndarray):
                res[k] = np.concatenate(v)
            else:
                assert False
        else:
            res[k] = v
    return res
