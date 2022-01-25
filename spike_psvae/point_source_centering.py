"""Point source relocation of spikes

Can we remove the effect of location "analytically"?

IDK why I did this torch instead of np but easy to change. uhh

TODO: Not sure how much of this code assumes NP2 geom specific stuff.
"""
# from scipy import ndimage
import torch
import numpy as np

from .waveform_utils import get_local_geom
from .torch_utils import translate
from .localization import localize_ptp


def point_source_ptp(local_geom, x, y, z, alpha):
    # figure out geometry
    local_geom = torch.as_tensor(np.array(local_geom))
    B, C, _ = local_geom.shape
    xz = torch.stack(
        [torch.as_tensor(x), torch.broadcast_to(torch.as_tensor(z), x.shape)],
        axis=-1,
    )
    geom_rel = local_geom - xz.view(-1, 1, 2)
    dists = torch.sqrt(
        torch.as_tensor(y * y).view(-1, 1)
        + torch.square(geom_rel).sum(dim=2),
    )
    ptp = torch.squeeze(torch.as_tensor(alpha).view(-1, 1) / dists)
    return ptp


def stereotypical_ptp(local_geom, x=None, y=15.0, z=0.0, alpha=150.0):
    assert local_geom.shape[1] % 2 == 0
    # Center of xz... could spell trouble for spikes on the edge of the probe,
    # where this is not the maxchan's Z coord.
    if x is None:
        x = local_geom[:, :, 0].mean(axis=1)
    r = point_source_ptp(local_geom, x, y, z, alpha)
    return r


def ptp_fit(
    waveforms,
    geom,
    maxchans,
    x,
    y,
    z_rel,
    alpha,
    channel_radius=10,
    firstchans=None,
    geomkind="updown",
):
    B, T, C = waveforms.shape
    geom = geom.copy()
    ptp = waveforms.ptp(axis=1)
    local_geom = torch.stack(
        [
            torch.as_tensor(
                get_local_geom(
                    geom,
                    maxchans[n],
                    channel_radius,
                    ptp[n],
                    firstchan=firstchans[n]
                    if firstchans is not None
                    else None,  # noqa
                    geomkind=geomkind,
                )
            )
            for n in range(B)
        ],
        axis=0,
    )

    # ptp predicted from this localization (x,y,z,alpha)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)

    return ptp, q.numpy()


def shift(
    waveform,
    maxchan,
    geom,
    dx=0,
    dz=0,
    y1=None,
    alpha1=None,
    channel_radius=10,
    firstchan=None,
    geomkind="updown",
):
    ptp = waveform.ptp(0)
    x0, y0, z_rel0, z_abs0, alpha0 = localize_ptp(
        ptp, maxchan, geom, firstchan=firstchan, geomkind=geomkind
    )
    local_geom = get_local_geom(
        geom,
        maxchan,
        channel_radius,
        ptp,
        firstchan=firstchan,
        geomkind=geomkind,
    )
    x1 = x0 + dx
    z1 = z_rel0 + dz
    if y1 is None:
        y1 = y0
    if alpha1 is None:
        alpha1 = alpha0
    ptp0 = point_source_ptp([local_geom], x0, y0, z_rel0, alpha0)
    qtq = point_source_ptp([local_geom], x1, y1, z1, alpha1)
    shifted = waveform * (qtq.numpy() / ptp0.numpy())[None, :]
    return shifted, qtq


def relocate_simple(
    waveforms,
    geom,
    maxchans,
    x,
    y,
    z_rel,
    alpha,
    channel_radius=10,
    firstchans=None,
    geomkind="updown",
    relocate_dims="xyza",
    interp_xz=False,
):
    """Shift waveforms according to the point source model

    This computes the PTP predicted for a waveform from the point
    source model, divides the waveform by that PTP, and multiplies
    it by the predicted PTP at a "standard" location.

    Optionally (if `interp_xz`), any relocation done on x/z dims will
    be performed by shifting the image rather than multiplying by
    PTPs.

    Arguments
    ---------
    waveforms : array-like (batches, time, local channels)
    geom : array-like (global channels, 2)
        XZ coords
    maxchans : integer array-like (batches,)
        Max channel in global channel space, not local.
    x, y, z_rel, alpha : array-likes, all (batches,)
        Localizations for this batch of spikes
    channel_radius : int
    geomkind : str, "updown" or "standard"
    relocate_dims : str
        Should be a string containing some or all of the characters "xyza"
        These are the dimensions that will be relocated. For example,
        if you set it to "yza", then the localization for `x` will be used
        when constructing the target/stereotpyical PTP, so that nothing
        will happen to your `waveforms` along the x dimension
    interp_xz : bool
        Rather than shifting X/Z by means of PTPs, use image interpolation
        instead. Might preserve more info. Y/alpha cannot be handled this
        way.

    Returns
    -------
    waveforms_relocated, r, q : array-likes
        waveforms_relocated is your relocated waveforms, `r` is the "target"
        PTP (what the point source model predicts at standard locations for
        the `relocate_dims`), `q` is the ptp that the point source model
        predicts from the localizations you supplied here.
    """
    B, T, C = waveforms.shape
    geom = geom.copy()
    ptp = waveforms.ptp(axis=1)
    local_geom = torch.stack(
        [
            torch.as_tensor(
                get_local_geom(
                    geom,
                    maxchans[n],
                    channel_radius,
                    ptp[n],
                    firstchan=firstchans[n]
                    if firstchans is not None
                    else None,  # noqa
                    geomkind=geomkind,
                )
            )
            for n in range(B)
        ],
        axis=0,
    )

    # who are we moving? set the standard locs accordingly
    # if we are interpolating for x/z shift, then we don't want
    # to touch those with the PTP rescaling. so, put them into
    # these kwargs to set them as origins for the target PTP.
    stereo_kwargs = {}
    if len(relocate_dims) < 4:
        assert len(relocate_dims) > 0
        if interp_xz or "x" not in relocate_dims:
            stereo_kwargs["x"] = x
        if "y" not in relocate_dims:
            stereo_kwargs["y"] = y
        if interp_xz or "z" not in relocate_dims:
            stereo_kwargs["z"] = z_rel
        if "a" not in relocate_dims:
            stereo_kwargs["alpha"] = alpha
    r = stereotypical_ptp(local_geom, **stereo_kwargs)

    # ptp predicted from this localization (x,y,z,alpha)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)

    # relocate by PTP rescaling
    waveforms_relocated = torch.as_tensor(waveforms) * (
        r.view(B, C) / q
    ).unsqueeze(1)

    # deal with interp x/z
    if interp_xz:
        # get the shifts in pixel units
        dx = dz = 0
        if "x" in relocate_dims:
            cx = local_geom[:, :, 0].mean(axis=1)
            hx = local_geom[0, 1, 0] - local_geom[0, 0, 0]
            dx = (cx - x) / hx
        if "z" in relocate_dims:
            hz = local_geom[0, 2, 1] - local_geom[0, 0, 1]
            dz = -z_rel / hz
        shifts = torch.stack(
            (
                torch.as_tensor(dx).broadcast_to(B),
                torch.as_tensor(dz).broadcast_to(B),
            ),
            dim=1,
        )

        # apply shifts
        waveforms_relocated = translate(
            waveforms_relocated.reshape(B, T, C // 2, 2),
            shifts,
        ).reshape(B, T, C)

    return waveforms_relocated, r, q
