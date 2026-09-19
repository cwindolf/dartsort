from typing import TYPE_CHECKING, Protocol, Self

import torch
import torch.nn.functional as F
from torch import Tensor

from ..detect import detect_and_deduplicate
from ..detect.detect import (
    deduplicate_globally,
    is_extreme_transpose_no_pad,
    peak_sign_to_pos,
    update_peak_map,
)
from ..util.internal_config import PeakSign, SubtractionConfig
from ..util.py_util import databag, panic
from ..util.spiketorch import grab_spikes, subtract_spikes_
from ..util.torch_util import BModule
from ..util.waveform_util import get_relative_subset
from .peel_lib import check_residual_decrease

if TYPE_CHECKING:
    from ..transform.pipeline import WaveformPipeline


# -- messenger classes


@databag(frozen=True)
class Dilation:
    """Proposers need different lengths of context to update after residual changes"""

    time: int

    def __or__(self, other: Self) -> Self:
        return self.__class__(time=max(self.time, other.time))


@databag
class PeakProposals:
    times_samples: Tensor
    channels: Tensor
    voltages: Tensor

    def __len__(self) -> int:
        return self.times_samples.numel()

    def __getitem__(self, ix: Tensor) -> Self:
        return self.__class__(
            times_samples=self.times_samples[ix],
            channels=self.channels[ix],
            voltages=self.voltages[ix],
        )


@databag
class ExclusionFootprint:
    time_ix: Tensor
    chan_ix: Tensor

    def __getitem__(self, ix: Tensor) -> Self:
        return self.__class__(time_ix=self.time_ix[ix], chan_ix=self.chan_ix[ix])


@databag
class AcceptedPeaks:
    times_samples: Tensor
    channels: Tensor
    waveforms: Tensor
    features: dict[str, Tensor]

    def __len__(self) -> int:
        return self.times_samples.numel()


@databag
class ChunkSubtractionResult:
    n_spikes: int
    times_samples: Tensor
    channels: Tensor
    collisioncleaned_waveforms: Tensor
    denoised_waveforms: Tensor | None
    residual: Tensor
    features: dict[str, Tensor]


# -- peak proposers


class PeakProposer(Protocol):
    """Detects and deduplicates the peaks to try subtracting this iteration."""

    @property
    def dilation(self) -> Dilation: ...

    def setup(self, residual: Tensor) -> None: ...

    def update(self, residual: Tensor, times_samples: Tensor) -> None: ...

    def cleanup(self) -> None: ...

    def propose_peaks(
        self, residual: Tensor, detection_mask: Tensor | None
    ) -> PeakProposals: ...


@databag
class GlobalPeakProposer:
    """Global deduplication strategy"""

    threshold: float
    peak_sign: PeakSign
    relative_peak_radius: int
    detect_dedup_radius: int
    peak_channel_index: Tensor | None
    trough_priority: float | None
    remove_exact_duplicates: bool
    trough_offset_samples: int
    spike_length_samples: int
    audit: bool = False
    peak_map: Tensor | None = None

    @property
    def dilation(self) -> Dilation:
        return Dilation(time=self.relative_peak_radius)

    def setup(self, residual: Tensor) -> None:
        self.peak_map = self._full_peak_map(residual)

    def cleanup(self) -> None:
        self.peak_map = None

    def update(self, residual: Tensor, times_samples: Tensor) -> None:
        assert self.peak_map is not None
        if not times_samples.numel():
            return

        # after subtraction, residual is "dirty" and the peak map is invalidated
        # invalidation region is a spike length plus one "dilation". but an extra
        # dilation of context is needed to fill in the first one!
        dt = self.dilation.time
        width = self.spike_length_samples + 4 * dt
        lead = self.trough_offset_samples + 2 * dt
        starts = (times_samples - lead).clamp_(0, residual.shape[0] - width)

        update_peak_map(
            self.peak_map,
            residual,
            starts,
            width,
            dt,
            peak_sign=self.peak_sign,
            relative_peak_radius=self.relative_peak_radius,
            peak_channel_index=self.peak_channel_index,
        )

        if self.audit:
            reference = self._full_peak_map(residual)
            n_bad = int((reference != self.peak_map).sum())
            if n_bad:
                panic(f"patched peak map differs from full recompute at {n_bad} sites")

    def propose_peaks(
        self, residual: Tensor, detection_mask: Tensor | None
    ) -> PeakProposals:
        assert self.peak_map is not None
        times_samples, channels = deduplicate_globally(
            residual,
            peak_sign_to_pos(residual, self.peak_sign),
            self.peak_map,
            self.threshold,
            peak_sign=self.peak_sign,
            dedup_temporal_radius=self.detect_dedup_radius,
            trough_priority=self.trough_priority,
            remove_exact_duplicates=self.remove_exact_duplicates,
            detection_mask=None if detection_mask is None else detection_mask[:, :-1],
        )
        return PeakProposals(
            times_samples=times_samples,
            channels=channels,
            voltages=residual[times_samples, channels],
        )

    def _full_peak_map(self, residual: Tensor) -> Tensor:
        return is_extreme_transpose_no_pad(
            peak_sign_to_pos(residual, self.peak_sign),
            dt=self.relative_peak_radius,
            neighbors=self.peak_channel_index,
        )


@databag
class LocalPeakProposer:
    """Local deduplication strategy"""

    threshold: float
    peak_sign: PeakSign
    relative_peak_radius: int
    detect_dedup_radius: int
    peak_channel_index: Tensor | None
    sub_dedup_channel_index: Tensor
    trough_priority: float | None
    remove_exact_duplicates: bool

    @property
    def dilation(self) -> Dilation:
        return Dilation(time=self.relative_peak_radius)

    def setup(self, residual: Tensor) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def update(self, residual: Tensor, times_samples: Tensor) -> None:
        """Not patched yet: local detection batches over time inside
        detect_and_deduplicate, so its peak map isn't separable the way the
        global one is."""

    def propose_peaks(
        self, residual: Tensor, detection_mask: Tensor | None
    ) -> PeakProposals:
        times_samples, channels = detect_and_deduplicate(
            residual[:, :-1],
            self.threshold,
            peak_channel_index=self.peak_channel_index,
            dedup_neighborhoods=self.sub_dedup_channel_index,
            peak_sign=self.peak_sign,
            relative_peak_radius=self.relative_peak_radius,
            dedup_temporal_radius=self.detect_dedup_radius,
            remove_exact_duplicates=self.remove_exact_duplicates,
            detection_mask=None if detection_mask is None else detection_mask[:, :-1],
            trough_priority=self.trough_priority,
        )
        return PeakProposals(
            times_samples=times_samples,
            channels=channels,
            voltages=residual[times_samples, channels],
        )


def make_peak_proposer(
    p: SubtractionConfig,
    spike_length_samples: int,
    trough_offset_samples: int,
    peak_channel_index: Tensor | None,
    sub_dedup_channel_index: Tensor,
    audit: bool = False,
) -> PeakProposer:
    if p.subtract_global_dedup:
        return GlobalPeakProposer(
            threshold=p.detection_threshold,
            peak_sign=p.peak_sign,
            relative_peak_radius=p.relative_peak_radius_samples,
            detect_dedup_radius=spike_length_samples,
            peak_channel_index=peak_channel_index,
            trough_priority=p.trough_priority,
            remove_exact_duplicates=p.remove_exact_duplicates,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            audit=audit,
        )
    return LocalPeakProposer(
        threshold=p.detection_threshold,
        peak_sign=p.peak_sign,
        relative_peak_radius=p.relative_peak_radius_samples,
        detect_dedup_radius=spike_length_samples,
        peak_channel_index=peak_channel_index,
        sub_dedup_channel_index=sub_dedup_channel_index,
        trough_priority=p.trough_priority,
        remove_exact_duplicates=p.remove_exact_duplicates,
    )


# -- the subtracter


class ChunkSubtracter(BModule):
    def __init__(
        self,
        *,
        channel_index: Tensor,
        denoising_pipeline: "WaveformPipeline",
        proposer: PeakProposer,
        trough_offset_samples: int,
        spike_length_samples: int,
        residnorm_decrease_threshold: float,
        extract_index: Tensor | None = None,
        extract_mask: Tensor | None = None,
        dedup_channel_index: Tensor | None = None,
        subtract_rel_inds: Tensor | None = None,
        local_whiteners: Tensor | None = None,
        whitening_kernel: Tensor | None = None,
        peak_sign: PeakSign = "both",
        exclusion_time_radius: int = 7,
        pos_exclusion_time_radius: int | None = None,
        realign_to_denoiser: bool = False,
        denoiser_realignment_shift: int = 5,
        max_iter: int = 100,
        save_iteration: bool = False,
        save_residnorm_decrease: bool = False,
        compute_collidedness: bool = False,
    ):
        super().__init__()

        self.denoising_pipeline = denoising_pipeline
        self.proposer = proposer
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.post_trough_samples = spike_length_samples - trough_offset_samples
        self.residnorm_decrease_threshold = residnorm_decrease_threshold
        self.peak_sign = peak_sign
        self.exclusion_time_radius = exclusion_time_radius
        self.pos_exclusion_time_radius = pos_exclusion_time_radius
        self.realign_to_denoiser = realign_to_denoiser
        self.denoiser_realignment_shift = denoiser_realignment_shift
        self.max_iter = max_iter
        self.save_iteration = save_iteration
        self.save_residnorm_decrease = save_residnorm_decrease
        self.compute_collidedness = compute_collidedness

        self.register_buffer("channel_index", channel_index)
        self.register_buffer(
            "extract_index", channel_index if extract_index is None else extract_index
        )
        self.register_buffer_or_none("extract_mask", extract_mask)
        self.register_buffer_or_none("dedup_channel_index", dedup_channel_index)
        self.register_buffer_or_none("subtract_rel_inds", subtract_rel_inds)
        self.register_buffer_or_none("local_whiteners", local_whiteners)
        self.register_buffer_or_none("whitening_kernel", whitening_kernel)
        assert extract_index is None or extract_mask is not None

        self.residual: Tensor | None = None
        self.detection_mask: Tensor | None = None
        self._exclusion_time_ix: Tensor | None = None
        self._pos_exclusion_time_ix: Tensor | None = None
        self._mask_touched = False

    # -- run before and after each chunk

    def setup(self, traces: Tensor) -> None:
        self.residual = F.pad(traces, (0, 1), value=torch.nan)
        self.detection_mask = torch.ones_like(self.residual, dtype=torch.bool)
        self._mask_touched = False

        device = traces.device
        self._exclusion_time_ix = torch.arange(
            -self.exclusion_time_radius, self.exclusion_time_radius + 1, device=device
        )
        if self.pos_exclusion_time_radius:
            self._pos_exclusion_time_ix = torch.arange(
                -self.pos_exclusion_time_radius,
                self.pos_exclusion_time_radius + 1,
                device=device,
            )
        else:
            self._pos_exclusion_time_ix = None

        self.proposer.setup(self.residual)

    def cleanup(self) -> None:
        self.residual = None
        self.detection_mask = None
        self._exclusion_time_ix = None
        self._pos_exclusion_time_ix = None
        self.proposer.cleanup()

    # -- main

    def subtract_chunk(
        self,
        traces: Tensor,
        left_margin: int = 0,
        right_margin: int = 0,
        return_denoised_waveforms: bool = False,
    ) -> ChunkSubtractionResult:
        assert 0 <= left_margin < traces.shape[0]
        assert 0 <= right_margin < traces.shape[0]
        assert traces.shape[1] == self.b.channel_index.shape[0]

        self.setup(traces)
        try:
            spikes = self._peel()
            return self._stack_sort_and_get_waveforms(
                spikes, left_margin, right_margin, return_denoised_waveforms
            )
        finally:
            self.cleanup()

    def _peel(self) -> list[AcceptedPeaks]:
        iter_peaks = []
        for it in range(self.max_iter):
            proposals = self._propose_peaks()
            if not len(proposals):
                break

            # all proposals get excluded next iter, not just valid/accepted
            exclusion = self._exclusion_footprint(proposals)

            valid_ix = self._check_valid_times(proposals)
            if not valid_ix.numel():
                break
            valid_peaks = proposals[valid_ix]

            waveforms, features, accept_mask = self._denoise_and_check_score(
                valid_peaks
            )
            self._exclude(exclusion, valid_peaks, valid_ix)

            (accept_ix,) = accept_mask.nonzero(as_tuple=True)
            if not accept_ix.numel():
                continue
            peaks = AcceptedPeaks(
                times_samples=valid_peaks.times_samples[accept_ix],
                channels=valid_peaks.channels[accept_ix],
                waveforms=waveforms[accept_ix],
                features={k: v[accept_ix] for k, v in features.items()},
            )
            if self.save_iteration:
                peaks.features["iteration"] = torch.full_like(peaks.times_samples, it)

            self._subtract(peaks)
            assert self.residual is not None
            self.proposer.update(self.residual, peaks.times_samples)
            iter_peaks.append(peaks)

        return iter_peaks

    # -- loop steps

    def _propose_peaks(self) -> PeakProposals:
        assert self.residual is not None
        mask = self.detection_mask if self._mask_touched else None
        return self.proposer.propose_peaks(self.residual, mask)

    def _check_valid_times(self, peaks: PeakProposals) -> Tensor:
        assert self.residual is not None
        max_trough_time = self.residual.shape[0] - self.post_trough_samples
        times = peaks.times_samples
        keep = times == times.clamp(self.trough_offset_samples, max_trough_time)
        (keep,) = keep.nonzero(as_tuple=True)
        return keep

    def _exclusion_footprint(self, peaks: PeakProposals) -> ExclusionFootprint:
        assert self.residual is not None
        assert self._exclusion_time_ix is not None
        time_ix = peaks.times_samples.unsqueeze(1) + self._exclusion_time_ix
        time_ix = time_ix.clamp_(0, self.residual.shape[0] - 1)
        if self.b.dedup_channel_index is not None:
            chan_ix = self.b.dedup_channel_index[peaks.channels]
        else:
            chan_ix = peaks.channels.unsqueeze(1)
        return ExclusionFootprint(time_ix=time_ix, chan_ix=chan_ix)

    def _denoise_and_check_score(
        self, peaks: PeakProposals
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        assert self.residual is not None
        waveforms = grab_spikes(
            self.residual,
            peaks.times_samples,
            peaks.channels,
            self.b.channel_index,
            trough_offset=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            buffer=0,
            already_padded=True,
        )

        if self.residnorm_decrease_threshold:
            original_waveforms = waveforms.nan_to_num()
        else:
            original_waveforms = None

        waveforms, features = self.denoising_pipeline(
            waveforms, channels=peaks.channels
        )

        accept, new_feats = check_residual_decrease(
            original_waveforms,
            waveforms,
            threshold=self.residnorm_decrease_threshold,
            save_residnorm_decrease=self.save_residnorm_decrease,
            local_whiteners=self.b.local_whiteners,
            whitening_kernel=self.b.whitening_kernel,
            channels=peaks.channels,
        )
        features.update(new_feats)

        if self.realign_to_denoiser:
            assert self.b.subtract_rel_inds is not None
            features["time_shifts"], realign_mask = denoiser_time_shifts(
                waveforms=waveforms,
                channels=peaks.channels,
                voltages=peaks.voltages,
                subtract_rel_inds=self.b.subtract_rel_inds,
                trough_offset_samples=self.trough_offset_samples,
                spike_length_samples=self.spike_length_samples,
                peak_sign=self.peak_sign,
                denoiser_realignment_shift=self.denoiser_realignment_shift,
                detection_mask=self.detection_mask,
                times_samples=peaks.times_samples,
            )
            if realign_mask is not None:
                accept.logical_and_(realign_mask)

        return waveforms, features, accept

    def _exclude(
        self, excl: ExclusionFootprint, peaks: PeakProposals, peak_ix: Tensor
    ) -> None:
        """Ignore neighborhoods around peaks in the future by updating self.detection_mask

        Takes extra care to exclude positive peaks appearing near stronger troughs.
        """
        assert self.detection_mask is not None
        assert self.residual is not None
        self.detection_mask[excl.time_ix[:, :, None], excl.chan_ix[:, None, :]] = 0
        self._mask_touched = True

        if not self.pos_exclusion_time_radius:
            return
        assert self._pos_exclusion_time_ix is not None

        (neg,) = (peaks.voltages < 0).nonzero(as_tuple=True)
        time_ix = peaks.times_samples[neg].unsqueeze(1)
        time_ix = (time_ix + self._pos_exclusion_time_ix).clamp_(
            0, self.residual.shape[0] - 1
        )
        chan_ix = excl.chan_ix[peak_ix[neg]]

        pos_mask = torch.ones_like(self.detection_mask)
        pos_mask[time_ix[:, :, None], chan_ix[:, None, :]] = 0
        pos_mask.logical_or_(self.residual < 0)
        self.detection_mask.logical_and_(pos_mask)

    def _subtract(self, peaks: AcceptedPeaks) -> None:
        assert self.residual is not None
        self.residual = subtract_spikes_(
            self.residual,
            peaks.times_samples,
            peaks.channels,
            self.b.channel_index,
            peaks.waveforms,
            trough_offset=self.trough_offset_samples,
            buffer=0,
            already_padded=True,
            in_place=True,
        )

    def _stack_sort_and_get_waveforms(
        self,
        iter_peaks: list[AcceptedPeaks],
        left_margin: int,
        right_margin: int,
        return_denoised_waveforms: bool,
    ) -> ChunkSubtractionResult:
        assert self.residual is not None
        n_samples = self.residual.shape[0]
        trimmed_residual = self.residual[left_margin : n_samples - right_margin, :-1]

        if not iter_peaks:
            return empty_chunk_subtraction_result(
                self.spike_length_samples,
                self.b.extract_index,
                trimmed_residual,
                return_denoised_waveforms,
            )

        spike_times = torch.concatenate([a.times_samples for a in iter_peaks])
        spike_channels = torch.concatenate([a.channels for a in iter_peaks])
        subtracted_waveforms = torch.concatenate([a.waveforms for a in iter_peaks])
        spike_features = {
            k: torch.concatenate([a.features[k] for a in iter_peaks])
            for k in iter_peaks[0].features
        }

        # discard spikes in the margins and sort times_samples for caller
        max_valid_t = n_samples - right_margin - 1
        keep = spike_times == spike_times.clamp(left_margin, max_valid_t)
        (keep,) = keep.nonzero(as_tuple=True)
        if not keep.numel():
            return empty_chunk_subtraction_result(
                self.spike_length_samples,
                self.b.extract_index,
                trimmed_residual,
                return_denoised_waveforms,
            )

        keep = keep[torch.argsort(spike_times[keep])]
        subtracted_waveforms = subtracted_waveforms[keep]
        spike_times = spike_times[keep]
        spike_channels = spike_channels[keep]
        spike_features = {k: v[keep] for k, v in spike_features.items()}

        collisioncleaned_waveforms, denoised_waveforms, collidedness = (
            extract_output_waveforms(
                self.residual,
                spike_times,
                spike_channels,
                subtracted_waveforms,
                channel_index=self.b.channel_index,
                extract_index=self.b.extract_index,
                extract_mask=self.b.extract_mask,
                denoising_pipeline=self.denoising_pipeline,
                trough_offset_samples=self.trough_offset_samples,
                spike_length_samples=self.spike_length_samples,
                return_denoised_waveforms=return_denoised_waveforms,
                compute_collidedness=self.compute_collidedness,
            )
        )
        if collidedness is not None:
            spike_features["collidedness"] = collidedness

        spike_times -= left_margin
        if "time_shifts" in spike_features:
            spike_times += spike_features["time_shifts"]

        return ChunkSubtractionResult(
            n_spikes=spike_times.numel(),
            times_samples=spike_times,
            channels=spike_channels,
            collisioncleaned_waveforms=collisioncleaned_waveforms,
            denoised_waveforms=denoised_waveforms,
            residual=trimmed_residual.cpu(),
            features=spike_features,
        )


# -- helpers


def denoiser_time_shifts(
    waveforms: Tensor,
    channels: Tensor,
    voltages: Tensor,
    subtract_rel_inds: Tensor,
    trough_offset_samples: int,
    spike_length_samples: int,
    peak_sign: "PeakSign",
    denoiser_realignment_shift: int,
    detection_mask: Tensor | None = None,
    times_samples: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    # extract main channel traces
    main_channel_rel_inds = subtract_rel_inds[channels]
    denoised_main_channel_traces = waveforms.take_along_dim(
        dim=2, indices=main_channel_rel_inds[:, None, None]
    )
    nwf = len(waveforms)
    assert denoised_main_channel_traces.shape == (nwf, spike_length_samples, 1)
    denoised_main_channel_traces = denoised_main_channel_traces[:, :, 0]

    # extract window around trough
    start = trough_offset_samples - denoiser_realignment_shift
    end = trough_offset_samples + denoiser_realignment_shift + 1
    snips = denoised_main_channel_traces[:, start:end]
    assert snips.shape == (nwf, 2 * denoiser_realignment_shift + 1)

    # handle the sign of the events so that we can align to maxima
    if peak_sign == "both":
        snips.mul_(torch.sign(voltages)[:, None])
    elif peak_sign == "neg":
        snips.neg_()
    else:
        assert peak_sign == "pos"

    # find shifts just by argmax
    peaks = snips.argmax(dim=1)
    dt = peaks.sub_(denoiser_realignment_shift)

    # did we land in the drink?
    if detection_mask is not None:
        assert times_samples is not None
        mask = detection_mask[times_samples + dt, channels]
    else:
        mask = None

    return dt, mask


def extract_output_waveforms(
    residual: Tensor,
    times_samples: Tensor,
    channels: Tensor,
    subtracted_waveforms: Tensor,
    channel_index: Tensor,
    extract_index: Tensor,
    extract_mask: Tensor | None,
    denoising_pipeline: "WaveformPipeline",
    trough_offset_samples: int,
    spike_length_samples: int,
    return_denoised_waveforms: bool = False,
    compute_collidedness: bool = False,
    batch_size: int = 1024,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Deal with logic of going to extract index with/without denoising etc."""
    n = times_samples.numel()
    # intermediate index, before going to extract
    grab_index = channel_index if return_denoised_waveforms else extract_index
    shape = (n, spike_length_samples, extract_index.shape[1])
    collisioncleaned_waveforms = residual.new_empty(shape)
    denoised_waveforms = (
        residual.new_empty(shape) if return_denoised_waveforms else None
    )
    collidedness = residual.new_empty((n,)) if compute_collidedness else None

    def to_extract(waveforms, chans):
        if extract_mask is None:
            return waveforms
        return get_relative_subset(waveforms, chans, extract_mask)

    for i0 in range(0, n, batch_size):
        bs = slice(i0, min(n, i0 + batch_size))
        chans = channels[bs]
        waveforms = grab_spikes(
            residual,
            times_samples[bs],
            chans,
            grab_index,
            trough_offset=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            buffer=0,
            already_padded=True,
        )

        # collidedness is always extract neighborhood rms, costs an extra extract
        if collidedness is not None:
            resid = (
                to_extract(waveforms, chans) if return_denoised_waveforms else waveforms
            )
            collidedness[bs] = resid.square().nanmean(dim=(1, 2)).sqrt_()

        if return_denoised_waveforms:
            waveforms += subtracted_waveforms[bs]
            # copy out before denoising, which may overwrite its input
            collisioncleaned_waveforms[bs] = to_extract(waveforms, chans)
            denoised, _ = denoising_pipeline(waveforms, channels=chans)
            assert denoised_waveforms is not None
            denoised_waveforms[bs] = to_extract(denoised, chans)
        else:
            waveforms += to_extract(subtracted_waveforms[bs], chans)
            collisioncleaned_waveforms[bs] = waveforms

    return collisioncleaned_waveforms, denoised_waveforms, collidedness


def empty_chunk_subtraction_result(
    spike_length_samples, channel_index, residual, return_denoised
):
    empty_waveforms = torch.empty(
        (0, spike_length_samples, channel_index.shape[1]),
        dtype=residual.dtype,
    )
    empty_times_or_chans = torch.empty((0,), dtype=torch.long)
    return ChunkSubtractionResult(
        n_spikes=0,
        times_samples=empty_times_or_chans,
        channels=empty_times_or_chans,
        collisioncleaned_waveforms=empty_waveforms,
        denoised_waveforms=empty_waveforms if return_denoised else None,
        residual=residual,
        features={},
    )
