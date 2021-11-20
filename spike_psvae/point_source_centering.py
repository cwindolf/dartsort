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
            torch.as_tensor(y * y).view(-1, 1, 1) + torch.square(geom_rel),
            dim=2,
        )
    )
    ptp = torch.squeeze(torch.as_tensor(alpha).view(-1, 1) / dists)
    return ptp


def stereotypical_ptp(local_geom, x=None, y=15.0, z=None, alpha=150.0):
    assert local_geom.shape[1] % 2 == 0
    # Center of xz... could spell trouble for spikes on the edge of the probe,
    # where this is not the maxchan's Z coord.
    if x is None:
        x = local_geom[:, :, 0].mean(axis=1)
    if z is None:
        z = local_geom[:, :, 1].mean(axis=1)
    r = point_source_ptp(local_geom, x, y, z, alpha)
    return r


def relocate_simple(
    waveforms,
    geom,
    maxchan,
    x,
    y,
    z_rel,
    alpha,
    channel_radius=10,
    geomkind="updown",
    relocate_dims="xyza",
):
    """'Relocate' waveforms according to the point source model
    
    This computes the PTP predicted for a waveform from the point source model,
    divides the waveform by that PTP, and multiplies it by the predicted PTP at a
    "standard" location.
    """
    B, T, C = waveforms.shape
    geom = geom.copy()
    ptp = waveforms.ptp(axis=1)
    local_geom = torch.stack(
        [
            torch.as_tensor(
                get_local_geom(
                    geom, maxchan[n], channel_radius, ptp[n], geomkind=geomkind
                )
            )
            for n in range(B)
        ],
        axis=0,
    )
    
    # ptp of "standard location" spike
    stereo_kwargs = {}
    if len(relocate_dims) < 4:
        assert len(relocate_dims) > 0
        if "x" not in relocate_dims:
            stereo_kwargs["x"] = x
        if "y" not in relocate_dims:
            stereo_kwargs["y"] = y
        if "z" not in relocate_dims:
            stereo_kwargs["z"] = z_rel
        if "a" not in relocate_dims:
            stereo_kwargs["alpha"] = alpha
    r = stereotypical_ptp(local_geom, **stereo_kwargs)
    
    # ptp predicted from this localization (x,y,z,alpha)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)
    
    # relocate and return
    waveforms_relocated = torch.as_tensor(waveforms) * (r.view(B, C) / q).unsqueeze(1)
    return waveforms_relocated, r, q
