"""Point source relocation of spikes

Can we remove the effect of location "analytically"?

IDK why I did this torch instead of np but easy to change. uhh

TODO: Not sure how much of this code assumes NP2 geom specific stuff.
"""
import torch
import numpy as np

# from .waveform_utils import get_local_geom
from .localization import localize_ptp


def point_source_ptp(local_geom, x, y, z, alpha):
    local_geom = torch.as_tensor(local_geom)
    dists = torch.sqrt(
        torch.as_tensor(y * y).view(-1, 1)
        + torch.square(local_geom[:, :, 0] - torch.as_tensor(x).view(-1, 1))
        + torch.square(local_geom[:, :, 1] - torch.as_tensor(z).view(-1, 1))
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
    firstchans,
    maxchans,
    x,
    y,
    z_rel,
    alpha,
):
    B, T, C = waveforms.shape
    geom = geom.copy()
    ptp = waveforms.ptp(axis=1)
    local_geom = torch.stack(
        [
            torch.as_tensor(
                get_local_geom(geom, firstchans[n], maxchans[n], C)
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
    firstchan,
    maxchan,
    geom,
    dx=0,
    dz=0,
    y1=None,
    alpha1=None,
    loc0=None,
):
    ptp = waveform.ptp(0)
    n_channels = ptp.shape[0]
    if loc0 is None:
        x0, y0, z_rel0, z_abs0, alpha0 = localize_ptp(
            ptp, firstchan, maxchan, geom
        )
    else:
        x0, y0, z_rel0, z_abs0, alpha0 = loc0
    local_geom = get_local_geom(geom, firstchan, maxchan, n_channels)
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
    firstchans,
    maxchans,
    x,
    y,
    z_abs,
    alpha,
    relocate_dims="xyza",
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
    ix = firstchans[:, None] + np.arange(C)[None, :]
    local_geom = torch.as_tensor(geom[ix])
    z_mc = geom[maxchans, 1]
    local_geom[:, :, 1] -= z_mc[:, None]
    z_rel = z_abs - z_mc

    # who are we moving? set the standard locs accordingly
    # if we are interpolating for x/z shift, then we don't want
    # to touch those with the PTP rescaling. so, put them into
    # these kwargs to set them as origins for the target PTP.
    stereo_kwargs = {}
    if len(relocate_dims) < 4:
        assert len(relocate_dims) > 0
        if "x" not in relocate_dims:
            stereo_kwargs["x"] = x
        if "y" not in relocate_dims:
            stereo_kwargs["y"] = y
        if "z" not in relocate_dims:
            stereo_kwargs["z"] = z_abs
        if "a" not in relocate_dims:
            stereo_kwargs["alpha"] = alpha
    r = stereotypical_ptp(local_geom, **stereo_kwargs)

    # ptp predicted from this localization (x,y,z,alpha)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)

    # relocate by PTP rescaling
    waveforms_relocated = torch.as_tensor(waveforms) * (
        r.view(B, C) / q
    ).unsqueeze(1)

    return waveforms_relocated, r, q


def relocating_ptps(
    geom,
    firstchans,
    maxchans,
    x,
    y,
    z_abs,
    alpha,
    nchans,
    relocate_dims="xyza",
):
    geom = geom.copy()
    ix = firstchans[:, None] + np.arange(nchans)[None, :]
    local_geom = torch.as_tensor(geom[ix])
    z_mc = geom[maxchans, 1]
    local_geom[:, :, 1] -= z_mc[:, None]
    z_rel = z_abs - z_mc

    # who are we moving? set the standard locs accordingly
    # if we are interpolating for x/z shift, then we don't want
    # to touch those with the PTP rescaling. so, put them into
    # these kwargs to set them as origins for the target PTP.
    stereo_kwargs = {}
    if len(relocate_dims) < 4:
        assert len(relocate_dims) > 0
        if "x" not in relocate_dims:
            stereo_kwargs["x"] = x
        if "y" not in relocate_dims:
            stereo_kwargs["y"] = y
        if "z" not in relocate_dims:
            stereo_kwargs["z"] = z_abs
        if "a" not in relocate_dims:
            stereo_kwargs["alpha"] = alpha
    r = stereotypical_ptp(local_geom, **stereo_kwargs)

    # ptp predicted from this localization (x,y,z,alpha)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)

    return r, q


def relocating_ptps_index(
    geom,
    channel_index,
    maxchans,
    x,
    y,
    z_abs,
    alpha,
    relocate_dims="xyza",
):
    geom = geom.copy()
    ix = channel_index[maxchans]
    local_geom = torch.as_tensor(
        np.pad(geom, [(0, 1), (0, 0)], constant_values=np.nan)[ix]
    )
    z_mc = geom[maxchans, 1]
    local_geom[:, :, 1] -= z_mc[:, None]
    z_rel = z_abs - z_mc

    # who are we moving? set the standard locs accordingly
    # if we are interpolating for x/z shift, then we don't want
    # to touch those with the PTP rescaling. so, put them into
    # these kwargs to set them as origins for the target PTP.
    stereo_kwargs = {}
    if len(relocate_dims) < 4:
        assert len(relocate_dims) > 0
        if "x" not in relocate_dims:
            stereo_kwargs["x"] = x
        if "y" not in relocate_dims:
            stereo_kwargs["y"] = y
        if "z" not in relocate_dims:
            stereo_kwargs["z"] = z_abs
        if "a" not in relocate_dims:
            stereo_kwargs["alpha"] = alpha
    r = stereotypical_ptp(local_geom, **stereo_kwargs)

    # ptp predicted from this localization (x,y,z,alpha)
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)

    return r, q


def relocate_index(
    waveforms,
    geom,
    channel_index,
    maxchans,
    x,
    y,
    z_abs,
    alpha,
    relocate_dims="xyza",
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
    r, q = relocating_ptps_index(
        geom,
        channel_index,
        maxchans,
        x,
        y,
        z_abs,
        alpha,
        relocate_dims=relocate_dims,
    )

    # relocate by PTP rescaling
    waveforms_relocated = torch.as_tensor(waveforms) * (
        r.view(B, C) / q
    ).unsqueeze(1)

    return waveforms_relocated, r, q
