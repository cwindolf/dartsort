"""Point source relocation of spikes

Can we remove the effect of location "analytically"?

TODO: Not sure how much of this code assumes NP2 geom specific stuff.
"""
import torch
from .localization import get_local_geom


def point_source_ptp(local_geom, x, y, z, alpha):
    # figure out geometry
    local_geom = torch.as_tensor(local_geom)
    B, C, _ = local_geom.shape
    xz = torch.stack([torch.as_tensor(x), torch.as_tensor(z)], axis=-1)
    geom_rel = local_geom.view(B, C, 2) - xz.view(-1, 1, 2)
    dists = torch.sqrt(
        torch.sum(
            torch.as_tensor(y * y).view(-1, 1, 1)
            + torch.square(geom_rel),
            dim=2,
        )
    )
    ptp = torch.squeeze(torch.as_tensor(alpha).view(-1, 1) / dists)
    return ptp


def stereotypical_ptp(local_geom, y=15.0, alpha=150.0):
    assert local_geom.shape[1] % 2 == 0
    xspacing = torch.abs(local_geom[0, 1, 0] - local_geom[0, 0, 0])
    x = xspacing / 2
    r = point_source_ptp(local_geom, x, y, 0, alpha)
    return r


def relocate_simple(wf, geom, maxchan, x, y, z_rel, alpha):
    """r is the result of stereotypical_ptp"""
    B, T, C = wf.shape
    assert C % 2 == 0
    geom = geom.copy()
    local_geom = torch.stack(
        [torch.as_tensor(get_local_geom(geom, mc, 10)) for mc in maxchan],
        axis=0,
    )
    r = stereotypical_ptp(local_geom)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)
    wf_ = wf * (r.view(B, C) / q).unsqueeze(1)
    return wf_, r, q
