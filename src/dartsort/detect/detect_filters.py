import torch
import torch.nn.functional as F


def vexity(snips):
    snips = torch.asarray(snips)
    assert snips.ndim == 2
    radius = snips.shape[1] // 2
    assert snips.shape[1] == 2 * radius + 1

    # get the autocovariance of the temporal derivative, with sign change
    # across the center. it's positive for convex (or convave) functions.
    ddt = snips.diff(dim=1)
    ddt[:, radius:].neg_()
    del snips
    vex = (ddt[:, :-1] * ddt[:, 1:]).sum(dim=1)

    return vex


def convexity_filter(traces, times, channels, threshold=None, radius=3):
    if threshold is None or not radius:
        return slice(None)

    # throw away spikes whose vexedness can't be checked
    keep = times.clip(radius, traces.shape[0] - radius - 1) == times
    (keep,) = keep.nonzero(as_tuple=True)

    # get temporal snips along main channels
    tix = times[keep, None] + torch.arange(-radius, radius + 1, device=times.device)
    snips = traces[tix[:, :], channels[keep, None]]
    assert snips.shape == (len(keep), 2 * radius + 1)
    vex = vexity(snips)
    assert vex.shape == keep.shape

    keep = keep[vex >= threshold]
    return keep
