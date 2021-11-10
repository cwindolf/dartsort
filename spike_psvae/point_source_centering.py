"""Point source relocation of spikes

Can we remove the effect of location "analytically"?

TODO: Not sure how much of this code assumes NP2 geom specific stuff.
"""
import torch


def point_source_ptp(local_geom, x, y, z, alpha):
    B, C, _ = local_geom.shape
    shanks = local_geom.shape[1] // 2
    zspacing = torch.abs(local_geom[0, 2, 1] - local_geom[0, 0, 1])
    # print("x", torch.as_tensor(x).shape)
    # print("z", torch.as_tensor(z).shape)
    xz = torch.stack([x, z + shanks * zspacing / 2], axis=-1)
    # print("xz", xz, xz.shape)
    # print("local_geom", local_geom)
    geom_rel = local_geom.view(B, C, 2) - xz.view(-1, 1, 2)
    # print("geom_rel", geom_rel.shape)
    # print("y", torch.as_tensor(y).shape)
    dists = torch.sqrt(
        torch.sum(
            torch.as_tensor(y * y).view(-1, 1, 1)
            + torch.square(geom_rel),
            dim=2,
        )
    )
    # print("dists", dists.shape)
    # print("alpha", torch.as_tensor(alpha).view(-1, 1).shape)
    ptp = torch.squeeze(torch.as_tensor(alpha).view(-1, 1) / dists)
    # print("ptp", ptp.shape)
    return ptp


def stereotypical_ptp(local_geom, y=15.0, alpha=150.0):
    assert local_geom.shape[1] % 2 == 0
    xspacing = torch.abs(local_geom[0, 1, 0] - local_geom[0, 0, 0])
    x = xspacing / 2
    # print("xspacing", xspacing)
    z = 0
    r = point_source_ptp(local_geom, x, y, z, alpha)
    return r


def relocate_simple(wf, geom, maxchan, x, y, z_rel, alpha):
    """r is the result of stereotypical_ptp"""
    B, T, C = wf.shape
    assert C % 2 == 0
    geom = geom.copy()
    local_geom = torch.stack(
        [geom[mc - 10 : mc + 10] for mc in maxchan],
        axis=0,
    )
    local_geom -= torch.min(local_geom, dim=1, keepdim=True).values
    # print("local_geom", local_geom.shape)
    # p = wf.max(dim=1) - wf.min(dim=1)
    # print("STEREO")
    r = stereotypical_ptp(local_geom)
    # print("QQQ")
    q = point_source_ptp(local_geom, x, y, z_rel, alpha)
    # print(wf.shape, r.shape, q.shape, (r.view(B, C) / q).shape)
    wf_ = wf * (r.view(B, C) / q).unsqueeze(1)
    return wf_, r, q
