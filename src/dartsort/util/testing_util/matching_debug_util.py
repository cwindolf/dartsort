import numpy as np
import torch
from ...peel.matching import ObjectiveUpdateTemplateMatchingPeeler


def yield_step_results(
    matcher: ObjectiveUpdateTemplateMatchingPeeler, chunk, t_s: float, max_iter: int = 5, obj_mode=False,
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
        matcher=matcher, chunk=chunk, t_s=t_s, max_iter=max_iter, obj_mode=obj_mode,
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
                x[chunk_sl].T, vmin=-5, vmax=5, aspect="auto", cmap=cmap, origin="lower", interpolation="none"
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
