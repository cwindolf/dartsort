"""Point source relocation of spikes

Can we remove the effect of location "analytically"?

TODO: Not sure how much of this code assumes NP2 geom specific stuff.
"""
import torch
from .waveform_utils import get_local_geom


def point_source_ptp(local_geom, x, y, z, alpha):
    # figure out geometry
    local_geom = torch.as_tensor(local_geom)
    B, C, _ = local_geom.shape
    xz = torch.stack(
        [torch.as_tensor(x), torch.broadcast_to(torch.as_tensor(z), x.shape)],
        axis=-1,
    )
    geom_rel = local_geom - xz.view(-1, 1, 2)
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
    x = local_geom[:, :, 0].mean(1)
    r = point_source_ptp(local_geom, x, y, 0, alpha)
    return r


def relocate_simple(wf, geom, maxchan, x, y, z_rel, alpha, channel_radius=10):
    """r is the result of stereotypical_ptp"""
    B, T, C = wf.shape
    geom = geom.copy()
    ptp = wf.ptp(axis=1)
    local_geom = torch.stack(
        [
            torch.as_tensor(
                get_local_geom(geom, maxchan[n], channel_radius, ptp[n])
            )
            for n in range(B)
        ],
        axis=0,
    )
    # ptp of "standard location" spike
    r = stereotypical_ptp(local_geom)
    # ptp predicted from this localization (x,y,z,alpha)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)
    wf_ = torch.as_tensor(wf) * (r.view(B, C) / q).unsqueeze(1)
    return wf_, r, q
