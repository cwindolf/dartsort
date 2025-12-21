from pathlib import Path
from typing import Self

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
from ...templates.template_util import get_main_channels_and_alignments
from ...templates.templates import TemplateData
from ..internal_config import ComputationConfig, MatchingConfig
from ..job_util import ensure_computation_config
from ..py_util import databag
from ..waveform_util import upsample_multichan
from ...peel.matching_util.matchlib import subtract_precomputed_pconv


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


def reference_pairwise_convolution(templates: np.ndarray, templates_up: np.ndarray):
    K, up, t, nc = templates_up.shape
    assert templates.shape == (K, t, nc)
    conv_len = 2 * t - 1
    pconv = np.zeros((K, K, up, conv_len), dtype=np.float32)
    tmp = np.zeros(conv_len, dtype=np.float64)
    for i in range(K):
        for j in range(K):
            for u in range(up):
                tmp[:] = 0.0
                for c in range(nc):
                    tmp += correlate(
                        templates[i, :, c].astype(np.float64)[::-1],
                        templates_up[j, u, :, c].astype(np.float64)[::-1],
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
            pre_conv = chunk_data.obj_from_conv(pre_conv)

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
            conv = chunk_data.obj_from_conv(chk["conv"]).numpy(force=True)
        else:
            conv = chk["conv"].numpy(force=True)
        times_samples = chk["times_samples"].numpy(force=True)
        labels = chk["labels"].numpy(force=True)
        channels = chk["channels"].numpy(force=True)
        if not times_samples.size:
            break

        yield resid, pre_conv, conv, times_samples, labels, channels


def visualize_step_results(
    matcher: ObjectiveUpdateTemplateMatchingPeeler,
    chunk,
    t_s: float,
    max_iter: int = 5,
    cmap="berlin",
    figsize=(10, 10),
    s=10,
    vis_start=None,
    vis_end=None,
    obj_mode=False,
):
    import matplotlib.pyplot as plt

    from ...vis import glasbey1024

    if vis_start is None:
        vis_start = 0
    if vis_end is None:
        vis_end = chunk.shape[0]
    vis_len = vis_end - vis_start
    chunk_sl = slice(vis_start, vis_end)

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

        for x, ax in zip((chunk, resid, chunk - resid), axes):
            ax.imshow(
                x[chunk_sl].T,
                vmin=-5,
                vmax=5,
                aspect="auto",
                cmap=cmap,
                origin="lower",
                interpolation="none",
            )
            ax.scatter(
                times_samples, channels, c=glasbey1024[labels], s=s, ec="w", lw=1
            )

        axes[-3].scatter(
            t_full[:n], c_full[:n], c=glasbey1024[l_full[:n]], s=s, lw=1, ec="k"
        )
        axes[-3].scatter(
            times_samples, channels, c=glasbey1024[labels], s=s, ec="w", lw=1
        )
        n += nnew

        for j, c in enumerate(pre_conv):
            axes[-2].plot(obj_domain, c[obj_sl], color=glasbey1024[j], lw=0.5)
        for t, l in zip(times_samples, labels):
            axes[-2].axvline(t, color=glasbey1024[l], lw=1, ls=":")
        for j, c in enumerate(conv):
            axes[-1].plot(obj_domain, c[obj_sl], color=glasbey1024[j], lw=0.5)
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

    def __init__(
        self, templates_up: Tensor, trough_shifts_up: Tensor, refrac_radius: int
    ):
        super().__init__()
        self.register_buffer("templates_up", templates_up)
        pconv = reference_pairwise_convolution(
            templates=templates_up[:, 0].numpy(force=True),
            templates_up=templates_up.numpy(force=True),
        )
        pconv = torch.asarray(pconv, device=templates_up.device)
        self.register_buffer("pconv", pconv)
        self.register_buffer("trough_shifts_up", trough_shifts_up)
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
        _, _, trough_shifts_up = get_main_channels_and_alignments(
            templates=templates_up.reshape(-1, *templates_up.shape[-2:])
        )
        trough_shifts_up = trough_shifts_up - template_data.trough_offset_samples
        trough_shifts_up = trough_shifts_up.reshape(*templates_up.shape[:2])
        templates_up = torch.asarray(templates_up, device=device, dtype=dtype)
        trough_shifts_up = torch.asarray(
            trough_shifts_up, device=device, dtype=torch.long
        )
        return cls(
            templates_up=templates_up,
            trough_shifts_up=trough_shifts_up,
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
            trough_shifts_up=self.b.trough_shifts_up,
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
    trough_shifts_up: Tensor
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
        wfs = self.templates_up[peaks.template_indices, peaks.upsampling_indices]
        wfs = wfs * peaks.scalings[:, None, None]
        time_ix = torch.arange(wfs.shape[1], device=wfs.device)
        for t, wf in zip(peaks.times, wfs):
            if sign == -1:
                traces[t + time_ix, :-1] -= wf
            elif sign == 1:
                traces[t + time_ix, :-1] += wf
            else:
                assert False

    def subtract_conv(
        self, conv: Tensor, peaks: "MatchingPeaks", padding=0, batch_size=256, sign=-1
    ):
        assert conv.shape[0] == self.pconv.shape[0]
        assert sign in (-1, 1)
        padded_lags = padding + self.conv_lags
        subtract_precomputed_pconv(
            conv=conv,
            pconv=self.pconv,
            template_indices=peaks.template_indices,
            upsampling_indices=peaks.upsampling_indices,
            scalings=peaks.scalings,
            times=peaks.times,
            padded_conv_lags=padded_lags,
            neg=sign == -1,
            batch_size=batch_size,
        )

    def get_clean_waveforms(
        self,
        peaks: MatchingPeaks,
        channels: Tensor,
        channel_index: Tensor,
        add_into: Tensor | None = None,
    ):
        wfs = self.templates_up[peaks.template_indices, peaks.upsampling_indices]
        wfs = F.pad(wfs, (0, 1))
        chan_ix = channel_index[channels][:, None, :]
        wfs = wfs.take_along_dim(dim=2, indices=chan_ix)
        wfs = wfs * peaks.scalings[:, None, None]
        if add_into is None:
            return wfs
        else:
            return add_into.add_(wfs)

    def _enforce_refractory(self, mask, peaks, offset=0, value=-torch.inf):
        time_ix = peaks.times[:, None] + (self.refrac_ix[None, :] + offset)
        row_ix = peaks.objective_template_indices[:, None]
        mask[row_ix, time_ix] = value

    def fine_match(
        self, *, peaks: MatchingPeaks, residual: Tensor, conv: Tensor, padding: int = 0
    ) -> MatchingPeaks:
        nt = self.templates_up.shape[2]
        for n, (t, l) in enumerate(zip(peaks.times, peaks.template_indices)):
            bank = self.templates_up[l]
            resid_chunk = residual[t : t + nt + 1]
            snips = torch.stack(
                [
                    resid_chunk[t0 : t0 + nt]
                    for t0 in range(resid_chunk.shape[0] - nt + 1)
                ],
                dim=0,
            )
            assert snips.shape[1] == nt
            dots = torch.einsum("utc,stc->us", bank, snips)
            if self.scaling:
                b = dots + self.inv_lambda
                a = self.normsq_up[l, None] + self.inv_lambda
                sc = (b / a).clamp(min=self.scale_min, max=self.scale_max)
                objs = -a * sc * sc + 2.0 * b * sc - self.inv_lambda
            else:
                sc = None
                objs = 2.0 * dots - self.normsq_up[l, None]
            best_val, best_flat = objs.view(-1).max(dim=0)
            best_u = best_flat // dots.shape[1]
            best_s = best_flat % dots.shape[1]
            assert objs[best_u, best_s] == best_val

            if self.scaling:
                assert sc is not None
                peaks.scalings[n] = sc[best_u, best_s]
            peaks.upsampling_indices[n] = best_u
            peaks.times[n] += best_s

        return peaks

    def trough_shifts(self, peaks: "MatchingPeaks") -> Tensor:
        return self.trough_shifts_up[peaks.template_indices, peaks.upsampling_indices]

    def reconstruct_up_templates(self):
        return self.templates_up
