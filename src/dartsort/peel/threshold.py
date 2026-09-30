"""Simple thresholding spike detection."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording

from ..detect import detect_and_deduplicate
from ..transform import BaseWaveformModule, WaveformPipeline
from ..util.data_util import SpikeDataset
from ..util.internal_config import (
    FeaturizationConfig,
    FitSamplingConfig,
    ThresholdingConfig,
    WaveformConfig,
    default_peeling_fit_sampling_cfg,
    default_thresholding_cfg,
    default_waveform_cfg,
)
from ..util.py_util import panic
from ..util.waveform_util import make_channel_index
from .peel_base import BasePeeler, PeelingBatchResult, peeling_empty_result
from .peel_lib import extract_detections, threshold_chunk
from .subtract_util import (
    LinearMatchedFilterProposer,
    proposal_filters_from_pipeline,
    with_proposal_filters,
)


class Threshold(BasePeeler):
    def __init__(
        self,
        recording: BaseRecording,
        channel_index: np.ndarray | torch.Tensor,
        featurization_pipeline: WaveformPipeline | None = None,
        p: ThresholdingConfig = default_thresholding_cfg,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        peak_channel_index=None,
        dedup_channel_index=None,
        fit_sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
        spatial_jitter_channel_index=None,
        save_collidedness=False,
        thinning: float | None = None,
        dtype=torch.float,
    ):
        fixed_prop_keys = ("channels", "times_seconds")
        if save_collidedness:
            fixed_prop_keys = fixed_prop_keys + ("collidedness",)
        spike_length_samples = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=p.chunk_length_samples,
            waveform_cfg=waveform_cfg,
            chunk_margin_samples=self.next_margin(spike_length_samples, factor=5),
            fit_sampling_cfg=fit_sampling_cfg,
            dtype=dtype,
            fixed_property_keys=fixed_prop_keys,
        )
        self.p = p
        self.peel_kind = f"Threshold {p.voltage_threshold}"
        self.save_collidedness = save_collidedness
        self.dedup_batch_size = self.nearest_batch_length()

        geom = recording.get_channel_locations()
        if peak_channel_index is None and p.relative_peak_radius_um:
            peak_channel_index = make_channel_index(
                geom, p.relative_peak_radius_um, to_torch=True
            )
        if dedup_channel_index is None and p.spatial_dedup_radius_um:
            dedup_channel_index = make_channel_index(
                geom, p.spatial_dedup_radius_um, to_torch=True
            )
        if spatial_jitter_channel_index is None and p.spatial_jitter_radius:
            spatial_jitter_channel_index = make_channel_index(
                geom, p.spatial_jitter_radius, to_torch=True
            )
        self.register_buffer_or_none(
            "spatial_jitter_channel_index", spatial_jitter_channel_index
        )
        self.register_buffer_or_none("dedup_channel_index", dedup_channel_index)
        self.register_buffer_or_none("peak_channel_index", peak_channel_index)

        # parameters for random perturbation of detections
        if thinning is None:
            self.thinning = p.thinning
        else:
            self.thinning = thinning
        self.is_random = (
            thinning or p.time_jitter or spatial_jitter_channel_index is not None
        )

        self.register_buffer_or_none("proposal_filters", None)
        self.proposal_filter_trough_offset: int | None = None
        self._in_fit_pass = False

    @classmethod
    def from_config(
        cls,
        *,
        recording: BaseRecording,
        waveform_cfg: WaveformConfig,
        thresholding_cfg: ThresholdingConfig,
        featurization_cfg: FeaturizationConfig,
        sampling_cfg: FitSamplingConfig,
        extract_channel_index: torch.Tensor | None = None,
        featurization_pipeline: WaveformPipeline | None = None,
    ):
        geom = torch.tensor(recording.get_channel_locations())
        if extract_channel_index is None:
            channel_index = make_channel_index(
                geom, featurization_cfg.extract_radius, to_torch=True
            )
        else:
            channel_index = extract_channel_index

        featurization_cfg = with_proposal_filters(
            featurization_cfg,
            thresholding_cfg.detection_proposal,
            thresholding_cfg.proposal_filters,
        )
        if featurization_pipeline is None:
            featurization_pipeline = WaveformPipeline.from_config(
                geom=geom,
                channel_index=channel_index,
                featurization_cfg=featurization_cfg,
                waveform_cfg=waveform_cfg,
                sampling_frequency=recording.sampling_frequency,
            )

        save_collidedness = (
            featurization_cfg.save_collidedness and not featurization_cfg.skip
        )

        return cls(
            recording,
            channel_index,
            featurization_pipeline=featurization_pipeline,
            p=thresholding_cfg,
            waveform_cfg=waveform_cfg,
            fit_sampling_cfg=sampling_cfg,
            save_collidedness=save_collidedness,
        )

    def out_datasets(self):
        datasets = super().out_datasets()
        if self.is_random:
            datasets.append(
                SpikeDataset(
                    name="orig_times_samples", shape_per_spike=(), dtype=np.int64
                )
            )
            datasets.append(
                SpikeDataset(name="orig_channels", shape_per_spike=(), dtype=np.int64)
            )
        if self.save_collidedness:
            datasets.append(SpikeDataset("collidedness", (), "float32"))
        datasets.append(
            SpikeDataset(name="voltages", shape_per_spike=(), dtype="float32")
        )
        return datasets

    def peeling_needs_fit(self):
        if self._in_fit_pass or self.p.detection_proposal == "voltage":
            return False
        return self.b.proposal_filters is None

    def fit_models(
        self, save_folder, tmp_dir=None, overwrite=False, computation_cfg=None
    ):
        self._in_fit_pass = True
        try:
            super().fit_models(
                save_folder,
                tmp_dir=tmp_dir,
                overwrite=overwrite,
                computation_cfg=computation_cfg,
            )
        finally:
            self._in_fit_pass = False
        self._install_proposal_filters()

    def load_models(self, save_folder: str | Path):
        super().load_models(save_folder)
        if not any(
            cast(BaseWaveformModule, p).needs_fit()
            for p in self.featurization_pipelines.values()
        ):
            self._install_proposal_filters()

    def _install_proposal_filters(self):
        if self.p.detection_proposal == "voltage":
            return
        assert self.featurization_pipeline is not None
        filters, trough_offset = proposal_filters_from_pipeline(
            self.featurization_pipeline,
            self.p.detection_proposal,
            self.p.proposal_filters,
        )
        filters = filters.to(self.b.channel_index.device)
        if self.b.proposal_filters is None:
            self.del_none_buffer("proposal_filters")
            self.register_buffer("proposal_filters", filters)
        else:
            self.proposal_filters = filters
        self.proposal_filter_trough_offset = trough_offset

    def _build_proposer(self) -> LinearMatchedFilterProposer:
        if self.b.proposal_filters is None:
            raise ValueError(f"{self.p.detection_proposal=}, not fit...")
        assert self.proposal_filter_trough_offset is not None
        if self.p.detection_proposal == "tpca":
            reduction = "sum"
        elif self.p.detection_proposal == "vq":
            reduction = "max"
        else:
            panic(self.p.detection_proposal)
        return LinearMatchedFilterProposer(
            filters=self.b.proposal_filters,
            filter_trough_offset=self.proposal_filter_trough_offset,
            threshold=self.p.proposal_threshold**2,
            reduction=reduction,
            relative_peak_radius=self.p.relative_peak_radius_samples,
            detect_dedup_radius=self.p.temporal_dedup_radius_samples,
            peak_channel_index=self.b.peak_channel_index,
            remove_exact_duplicates=self.p.remove_exact_duplicates,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            field_patch_channel_index=self.b.channel_index,
        )

    def peel_chunk(
        self,
        traces,
        *,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ) -> PeelingBatchResult:
        perturb = not self._in_fit_pass
        extract_kw = dict(
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            thinning=self.thinning if perturb else 0.0,
            time_jitter=self.p.time_jitter if perturb else 0,
            spatial_jitter_channel_index=(
                self.spatial_jitter_channel_index if perturb else None
            ),
            rg=self.rg if self.is_random else None,
            max_spikes_per_chunk=None,
            return_waveforms=return_waveforms,
            quiet=False,
        )
        proposal = "voltage" if self._in_fit_pass else self.p.detection_proposal
        if proposal == "voltage":
            threshold_res = threshold_chunk(
                traces,
                self.b.channel_index,
                voltage_threshold=self.p.voltage_threshold,
                peak_sign=self.p.peak_sign,
                dedup_channel_index=self.b.dedup_channel_index,
                peak_channel_index=self.b.peak_channel_index,
                temporal_dedup_radius_samples=self.p.temporal_dedup_radius_samples,
                remove_exact_duplicates=self.p.remove_exact_duplicates,
                trough_priority=self.p.trough_priority,
                **extract_kw,
            )
        elif proposal == "tpca" or proposal == "vq":
            proposer = self._build_proposer()
            field = proposer.full_proposal_score_field(
                F.pad(traces, (0, 1), value=torch.nan)
            )
            times_rel, channels, field_values = detect_and_deduplicate(
                field[:, :-1],
                threshold=proposer.threshold,
                peak_sign="pos",
                relative_peak_radius=self.p.relative_peak_radius_samples,
                peak_channel_index=self.b.peak_channel_index,
                dedup_temporal_radius=self.p.temporal_dedup_radius_samples,
                dedup_neighborhoods=self.b.dedup_channel_index,
                remove_exact_duplicates=self.p.remove_exact_duplicates,
                return_energies=True,
            )
            threshold_res = extract_detections(
                traces,
                times_rel,
                channels,
                field_values,
                channel_index=self.b.channel_index,
                **extract_kw,
            )
        else:
            panic(proposal)
        if not threshold_res["n_spikes"]:
            return peeling_empty_result

        # get absolute times
        times_rel = threshold_res["times_rel"]
        assert times_rel is not None
        times_samples = times_rel + chunk_start_samples

        peel_result = PeelingBatchResult(
            n_spikes=threshold_res["n_spikes"],
            times_samples=times_samples,
            channels=threshold_res["channels"],
            voltages=threshold_res["voltages"],
        )
        if return_waveforms:
            assert "waveforms" in threshold_res
            peel_result["collisioncleaned_waveforms"] = threshold_res["waveforms"]
        if return_waveforms and self.save_collidedness:
            peel_result["collidedness"] = threshold_res["waveforms"].new_full(
                (int(threshold_res["n_spikes"]),), torch.nan
            )
        if self.is_random:
            orig_times_rel = threshold_res["orig_times_rel"]
            assert orig_times_rel is not None
            peel_result["orig_times_samples"] = orig_times_rel + chunk_start_samples
            peel_result["orig_channels"] = threshold_res["orig_channels"]
        if return_residual:
            # note, this is same as the input.
            peel_result["residual"] = traces[
                left_margin : traces.shape[0] - right_margin
            ]
        return peel_result
