from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import correlate
from spikeinterface.core import BaseRecording
from torch import Tensor

from ...peel.matching import (
    ChunkTemplateData,
    MatchingPeaks,
    MatchingTemplates,
    ObjectiveUpdateTemplateMatchingPeeler,
)
from ...peel.matching_util.matching_base import subtract_precomputed_pconv
from ...templates.templates import TemplateData
from ..data_util import DARTsortSorting
from ..internal_config import ComputationConfig, MatchingConfig
from ..job_util import ensure_computation_config
from ..py_util import databag
from ..waveform_util import upsample_multichan


def reference_shared_temporal_convolution(
    temporal: np.ndarray, temporal_up: np.ndarray
):
    """Compute the same convolution of temporal basis two ways."""
    rank, up, t = temporal_up.shape
    assert temporal.shape == (rank, t)
    conv_len = 2 * t - 1

    tconv0 = np.zeros((rank, rank, up, conv_len), dtype=np.float32)
    tconv00 = np.zeros((rank, rank, up, conv_len), dtype=np.float32)
    for p in range(rank):
        for q in range(rank):
            for u in range(up):
                tconv0[p, q, u] = correlate(
                    temporal[p], temporal_up[q, u], mode="full", method="direct"
                )[::-1]
                tconv00[p, q, u] = correlate(
                    temporal[p][::-1],
                    temporal_up[q, u][::-1],
                    mode="full",
                    method="direct",
                )
    return tconv0, tconv00


def reference_pairwise_convolution(
    templates: np.ndarray,
    templates_up: np.ndarray,
    accum_dtype: np.typing.DTypeLike = np.float32,
):
    K, up, t, nc = templates_up.shape
    assert templates.shape == (K, t, nc)
    conv_len = 2 * t - 1
    pconv = np.zeros((K, K, up, conv_len), dtype=np.float32)
    tmp = np.zeros(conv_len, dtype=accum_dtype)
    for i in range(K):
        for j in range(K):
            for u in range(up):
                tmp[:] = 0.0
                for c in range(nc):
                    tmp += correlate(
                        templates[i, :, c].astype(accum_dtype)[::-1],
                        templates_up[j, u, :, c].astype(accum_dtype)[::-1],
                        mode="full",
                        method="direct",
                    )
                pconv[i, j, u] = tmp.astype(np.float32)
    return pconv


def yield_step_results(
    matcher: ObjectiveUpdateTemplateMatchingPeeler,
    chunk,
    t_s: float,
    max_iter: int = 5,
    obj_mode=False,
):
    device = matcher.b.channel_index.device
    chunk = torch.asarray(chunk, device=device)
    assert matcher.matching_templates is not None
    chunk_data = matcher.matching_templates.data_at_time(
        t_s,
        scaling=matcher.is_scaling,
        inv_lambda=matcher.inv_lambda,
        scale_min=matcher.amp_scale_min,
        scale_max=matcher.amp_scale_max,
    )

    cur_residual = chunk.clone()
    for _ in range(max_iter):
        pre_conv = chunk_data.convolve(cur_residual.T, padding=matcher.obj_pad_len)
        if obj_mode:
            pre_conv = chunk_data.obj_from_conv(
                conv=pre_conv,
                out=torch.zeros_like(pre_conv),
                scalings_out=torch.zeros_like(pre_conv) if matcher.is_scaling else None,
            )

        chk = matcher.match_chunk(
            cur_residual,
            chunk_data,
            return_conv=True,
            return_residual=True,
            max_iter=1,
        )
        cur_residual = chk["residual"].clone()

        pre_conv = pre_conv.numpy(force=True)
        resid = chk["residual"].numpy(force=True)
        if obj_mode:
            conv = chunk_data.obj_from_conv(
                conv=chk["conv"],
                out=torch.zeros_like(chk["conv"]),
                scalings_out=torch.zeros_like(chk["conv"])
                if matcher.is_scaling
                else None,
            ).numpy(force=True)
        else:
            conv = chk["conv"].numpy(force=True)
        if not chk["n_spikes"]:
            break
        times_samples = chk["times_samples"].numpy(force=True)
        labels = chk["labels"].numpy(force=True)
        channels = chk["channels"].numpy(force=True)

        yield resid, pre_conv, conv, times_samples, labels, channels


def visualize_step_results(
    matcher: ObjectiveUpdateTemplateMatchingPeeler,
    chunk: np.ndarray,
    t_s: float,
    chunk_start_samples: int = 0,
    max_iter: int = 5,
    cmap="berlin",
    figsize=(10, 10),
    s=10,
    vis_start=None,
    vis_end=None,
    obj_mode=False,
    chunk_vis_style: Literal["im", "trace"] = "im",
    gt_sorting: DARTsortSorting | None = None,
):
    import matplotlib.pyplot as plt

    from ...vis import glasbey1024

    if vis_start is None:
        vis_start = 0
    if vis_end is None:
        vis_end = chunk.shape[0]
    vis_len = vis_end - vis_start
    chunk_sl = slice(vis_start, vis_end)

    if gt_sorting is not None:
        gt_t = gt_sorting.times_samples - chunk_start_samples
        gtvalid = np.flatnonzero(gt_t == gt_t.clip(vis_start, vis_end - 1))
        gt_t = gt_t[gtvalid] - vis_start
        gt_chan = gt_sorting.channels[gtvalid]
        gt_l = cast(np.ndarray, gt_sorting.labels)[gtvalid]
        gt_c = glasbey1024[gt_l]
    else:
        gt_t = gt_chan = gt_c = None

    obj_sl = slice(
        max(vis_start + matcher.obj_pad_len, matcher.obj_pad_len),
        min(vis_end, chunk.shape[0]),
    )
    obj_domain = matcher.trough_offset_samples + np.arange(obj_sl.stop - obj_sl.start)

    t_full = np.zeros(chunk.size, dtype=np.int64)
    c_full = np.zeros(chunk.size, dtype=np.int64)
    l_full = np.zeros(chunk.size, dtype=np.int64)
    n = 0

    it = 0
    for resid, pre_conv, conv, times_samples, labels, channels in yield_step_results(
        matcher=matcher,
        chunk=chunk,
        t_s=t_s,
        max_iter=max_iter,
        obj_mode=obj_mode,
    ):
        it += 1

        v = np.flatnonzero(times_samples == times_samples.clip(vis_start, vis_end - 1))
        times_samples = times_samples[v] - vis_start
        labels = labels[v]
        channels = channels[v]

        nnew = times_samples.shape[0]
        if not nnew:
            continue
        t_full[n : n + nnew] = times_samples
        c_full[n : n + nnew] = channels
        l_full[n : n + nnew] = labels

        panel = plt.figure(figsize=figsize, layout="constrained")
        axes = panel.subplots(nrows=6, sharex=True)

        for x, ax, name in zip(
            (chunk, resid, chunk - resid), axes, ("chunk", "resid", "signal")
        ):
            if chunk_vis_style == "im":
                ax.imshow(
                    x[chunk_sl].T,
                    vmin=-5,
                    vmax=5,
                    aspect="auto",
                    cmap=cmap,
                    origin="lower",
                    interpolation="none",
                )
            elif chunk_vis_style == "trace":
                for j, trace in enumerate(x[chunk_sl].T):
                    ax.plot(j, color="k", zorder=1)
                    ax.plot(trace + j, color="k", zorder=2)
            ax.scatter(
                times_samples, channels, c=glasbey1024[labels], s=s, ec="w", lw=1
            )
            ax.set_ylabel(name)

        if gt_t is not None:
            axes[-3].scatter(gt_t, gt_chan, c=gt_c, s=4 * s, lw=0, marker="o")

        axes[-3].scatter(
            t_full[:n], c_full[:n], c=glasbey1024[l_full[:n]], s=s, lw=1, ec="k"
        )
        axes[-3].set_ylabel("channel scatter")
        axes[-3].scatter(
            times_samples, channels, c=glasbey1024[labels], s=s, ec="w", lw=1
        )
        n += nnew

        for j, c in enumerate(pre_conv):
            axes[-2].plot(obj_domain, c[obj_sl], color=glasbey1024[j], lw=0.5)
        axes[-2].set_ylabel("pre-step " + ("obj" if obj_mode else "conv"))
        for t, l in zip(times_samples, labels):
            axes[-2].axvline(t, color=glasbey1024[l], lw=1, ls=":")
        for j, c in enumerate(conv):
            axes[-1].plot(obj_domain, c[obj_sl], color=glasbey1024[j], lw=0.5)
        axes[-1].set_ylabel("post-step " + ("obj" if obj_mode else "conv"))
        for ax in axes[-2:]:
            ax.grid()
        if obj_mode:
            for ax in axes[-2:]:
                ax.set_ylim([-100, pre_conv[:, obj_sl].max() * 1.05])

        panel.suptitle(f"iteration {it}", fontsize=12)

        plt.show()
        plt.close(panel)


# -- reference implementation for upsampled matching


class DebugMatchingTemplates(MatchingTemplates):
    """The goal with this one is to be obviously correct."""

    template_type = "debug"

    def __init__(self, templates_up: Tensor, refrac_radius: int):
        super().__init__()
        self.register_buffer("templates_up", templates_up)
        pconv = reference_pairwise_convolution(
            templates=templates_up[:, 0].numpy(force=True),
            templates_up=templates_up.numpy(force=True),
        )
        pconv = torch.asarray(pconv, device=templates_up.device)
        self.register_buffer("pconv", pconv)
        refrac_ix = torch.arange(-refrac_radius, refrac_radius + 1, device=pconv.device)
        conv_lags = torch.arange(
            -templates_up.shape[2] + 1, templates_up.shape[2], device=pconv.device
        )
        self.register_buffer("refrac_ix", refrac_ix)
        self.register_buffer("conv_lags", conv_lags)
        main_channels = templates_up[:, 0].square().sum(dim=1).argmax(dim=1)
        self.register_buffer("main_channels", main_channels)

    @classmethod
    def _from_config(
        cls,
        save_folder: Path,
        recording: BaseRecording,
        template_data: TemplateData,
        matching_cfg: MatchingConfig,
        computation_cfg: ComputationConfig | None = None,
        motion_est=None,
        overwrite: bool = False,
        dtype=torch.float,
    ) -> Self:
        assert motion_est is None
        computation_cfg = ensure_computation_config(computation_cfg)
        device = computation_cfg.actual_device()
        templates_up = upsample_multichan(
            template_data.templates, matching_cfg.template_temporal_upsampling_factor
        )
        assert templates_up.shape[0] == template_data.templates.shape[0]
        assert templates_up.shape[1] == matching_cfg.template_temporal_upsampling_factor
        assert templates_up.shape[3] == recording.get_num_channels()
        assert template_data.templates.shape[2] == recording.get_num_channels()
        templates_up = torch.asarray(templates_up, device=device, dtype=dtype)
        return cls(
            templates_up=templates_up,
            refrac_radius=matching_cfg.refractory_radius_frames,
        )

    def data_at_time(
        self,
        t_s: float,
        scaling: bool,
        inv_lambda: float,
        scale_min: float,
        scale_max: float,
    ) -> ChunkTemplateData:
        return DebugChunkTemplateData(
            spike_length_samples=self.b.templates_up.shape[2],
            unit_ids=torch.arange(
                self.b.templates_up.shape[0], device=self.b.pconv.device
            ),
            main_channels=self.b.main_channels,
            templates_up=self.b.templates_up,
            obj_normsq=self.b.templates_up[:, 0].square().sum(dim=(1, 2)),
            normsq_up=self.b.templates_up.square().sum(dim=(2, 3)),
            obj_n_templates=self.b.templates_up.shape[0],
            pconv=self.b.pconv,
            refrac_ix=self.b.refrac_ix,
            conv_lags=self.b.conv_lags,
            inv_lambda=torch.asarray(inv_lambda, device=self.b.pconv.device),
            scale_min=torch.asarray(scale_min, device=self.b.pconv.device),
            scale_max=torch.asarray(scale_max, device=self.b.pconv.device),
            scaling=scaling,
            needs_fine_pass=self.b.templates_up.shape[1] > 1 or scale_max > 3.0,
            upsampling=self.b.templates_up.shape[1] > 1,
            up_factor=self.b.templates_up.shape[1],
        )


@databag
class DebugChunkTemplateData(ChunkTemplateData):
    spike_length_samples: int
    unit_ids: Tensor
    main_channels: Tensor
    obj_normsq: Tensor
    normsq_up: Tensor
    obj_n_templates: int
    templates_up: Tensor
    pconv: Tensor
    conv_lags: Tensor
    refrac_ix: Tensor
    inv_lambda: Tensor
    scale_min: Tensor
    scale_max: Tensor
    scaling: bool
    upsampling: bool
    up_factor: int
    needs_fine_pass: bool
    coarse_objective: bool = False

    def convolve(self, traces: Tensor, padding: int = 0, out: Tensor | None = None):
        assert traces.shape[0] == self.templates_up.shape[3]
        T = traces.shape[1]
        t = self.templates_up.shape[2]

        out_len = T - (t - 1) + 2 * padding
        if out is None:
            out = traces.new_zeros((self.templates_up.shape[0], out_len))
        else:
            assert out.shape == (self.templates_up.shape[0], out_len)

        traces = traces.double()
        for i, tplt in enumerate(self.templates_up[:, 0]):
            res_i = F.conv1d(traces[None], tplt.T[None].double(), padding=padding)
            assert res_i.shape == (1, 1, out_len)
            out[i] = res_i[0, 0].to(dtype=out.dtype)

        return out

    def subtract(self, traces: Tensor, peaks: "MatchingPeaks", sign: int = -1):
        if not peaks.n_spikes:
            return

        assert peaks.template_inds is not None
        assert peaks.up_inds is not None
        wfs = self.templates_up[peaks.template_inds, peaks.up_inds]
        if peaks.scalings is not None:
            wfs = wfs * peaks.scalings[:, None, None]

        time_ix = torch.arange(wfs.shape[1], device=wfs.device)
        times = peaks.times
        assert times is not None
        for t, wf in zip(times, wfs):
            if sign == -1:
                traces[t + time_ix, :-1] -= wf
            elif sign == 1:
                traces[t + time_ix, :-1] += wf
            else:
                assert False

    def subtract_conv(
        self, conv: Tensor, peaks: "MatchingPeaks", padding=0, batch_size=256, sign=-1
    ):
        subtract_precomputed_pconv(
            conv=conv,
            pconv=self.pconv,
            peaks=peaks,
            padding=padding,
            conv_lags=self.conv_lags,
            sign=sign,
            batch_size=batch_size,
        )

    def get_clean_waveforms(
        self,
        peaks: MatchingPeaks,
        channels: Tensor,
        channel_index: Tensor,
        add_into: Tensor | None = None,
    ):
        if not peaks.n_spikes:
            return add_into
        assert peaks.template_inds is not None
        assert peaks.up_inds is not None
        wfs = self.templates_up[peaks.template_inds, peaks.up_inds]
        wfs = F.pad(wfs, (0, 1))
        chan_ix = channel_index[channels][:, None, :]
        wfs = wfs.take_along_dim(dim=2, indices=chan_ix)
        if peaks.scalings is not None:
            wfs = wfs * peaks.scalings[:, None, None]
        if add_into is None:
            return wfs
        else:
            return add_into.add_(wfs)

    def _enforce_refractory(
        self, mask: Tensor, peaks: MatchingPeaks, offset: int = 0, value=-torch.inf
    ):
        if not peaks.n_spikes:
            return
        assert peaks.times is not None
        assert peaks.obj_template_inds is not None
        time_ix = peaks.times[:, None] + (self.refrac_ix[None, :] + offset)
        row_ix = peaks.obj_template_inds[:, None]
        mask[row_ix, time_ix] = value

    def fine_match(
        self, *, peaks: MatchingPeaks, residual: Tensor, conv: Tensor, padding: int = 0
    ) -> MatchingPeaks:
        nt = self.templates_up.shape[2]
        if not peaks.n_spikes:
            return peaks

        assert peaks.times is not None
        times = peaks.times.clone()
        template_inds = peaks.template_inds
        assert times is not None
        assert template_inds is not None
        if self.scaling:
            scalings = conv.new_ones(times.shape)
        else:
            scalings = None
        up_inds = torch.zeros_like(template_inds)
        scores = conv.new_zeros(times.shape)

        for n, (t, l) in enumerate(zip(times, template_inds)):
            bank = self.templates_up[l]
            resid_chunk = residual[t : t + nt + 1]
            T = resid_chunk.shape[0]
            snips = [resid_chunk[t0 : t0 + nt] for t0 in range(T - nt + 1)]
            snips = torch.stack(snips, dim=0)
            assert snips.shape[1] == nt
            dots = torch.einsum("utc,stc->us", bank, snips)
            if self.scaling:
                b = dots + self.inv_lambda
                a = self.normsq_up[l, :, None] + self.inv_lambda
                sc = (b / a).clamp(min=self.scale_min, max=self.scale_max)
                objs = -a * sc * sc + 2.0 * b * sc - self.inv_lambda
            else:
                sc = None
                objs = 2.0 * dots - self.normsq_up[l, :, None]
            objs[0].add_(1e-5)
            best_val, best_flat = objs.view(-1).max(dim=0)
            best_u = best_flat // dots.shape[1]
            best_s = best_flat % dots.shape[1]
            assert objs[best_u, best_s] == best_val

            if self.scaling:
                assert scalings is not None
                assert sc is not None
                scalings[n] = sc[best_u, best_s]
            up_inds[n] = best_u
            times[n] += best_s
            scores[n] = best_val

        up_half = self.up_factor // 2
        time_shifts = (up_inds > up_half).long().neg_()

        return MatchingPeaks(
            times=times,
            obj_template_inds=peaks.obj_template_inds,
            template_inds=template_inds,
            scalings=scalings,
            up_inds=up_inds,
            scores=scores,
            time_shifts=time_shifts,
        )

    def reconstruct_up_templates(self):
        return self.templates_up
