import torch


def convexity_filter(traces, times, channels, threshold=None, radius=3):
    if threshold is None or not radius:
        return slice(None)

    # get temporal snips along main channels
    tix = times[:, None] + torch.arange(-radius, radius + 1)
    snips = traces[tix[:, :], channels[:, None]]
    assert snips.shape == (len(times), 2 * radius + 1)

    # get the autocovariance of the temporal derivative, with sign change
    # across the center. it's positive for convex (or convave) functions.
    ddt = snips.diff(dim=1)
    ddt[:, radius:].neg_()
    del snips
    vex = (ddt[:, :-1] * ddt[:, 1:]).sum(dim=1)
    assert vex.shape == times.shape

    (keep,) = (vex >= threshold).nonzero(as_tuple=True)
    return keep
