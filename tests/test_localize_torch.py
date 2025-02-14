import numpy as np
import torch
from dartsort.localize import localize_torch
from dartsort.util.waveform_util import make_channel_index
from test_util import dense_layout


def test_localize_torch():
    # this test checks to see if localization can recover its own
    # prediction in neuropixels geometry, checking that model
    # is implemented OK and that code works

    n_spikes = 101
    rg = np.random.default_rng(0)

    # make geometry and channel sparsity
    h = dense_layout()
    g = np.c_[h["x"], h["y"]]
    padded_g = np.pad(g, [(0, 1), (0, 0)])
    ci = make_channel_index(g, 75)
    assert ci.shape[0] == g.shape[0]
    ncloc = ci.shape[1]
    in_probe_ci = ci < g.shape[0]

    # fake spike positions and main channels
    chans = rg.integers(0, len(g), size=n_spikes)
    chan_zs = g[chans, 1]
    xs = rg.normal(loc=g[:, 0].mean(), scale=10, size=n_spikes)
    ys = rg.uniform(1e-3, 100, size=n_spikes)
    z_rels = rg.normal(scale=10, size=n_spikes)
    z_abss = chan_zs + z_rels
    alphas = rg.uniform(5.0, 15, size=n_spikes)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    z_rels = torch.tensor(z_rels)
    z_abss = torch.tensor(z_abss)
    alphas = torch.tensor(alphas)
    chans = torch.tensor(chans)
    chan_zs = torch.tensor(chan_zs)
    padded_g = torch.tensor(padded_g)
    in_probe_ci = torch.tensor(in_probe_ci).to(torch.double)

    # local geometries
    local_geoms = padded_g[ci[chans]]
    assert local_geoms.shape == (n_spikes, ncloc, 2)

    # model predicted amplitude distributions
    pred_ampvecs0 = np.array(
        [
            localize_torch.point_source_amplitude_at(x, y, za, a, lg).numpy()
            for x, y, za, a, lg in zip(xs, ys, z_abss, alphas, local_geoms)
        ]
    )
    local_geoms[:, :, 1] -= torch.tensor(g[chans, 1][:, None])
    pred_ampvecs = np.array(
        [
            localize_torch.point_source_amplitude_at(x, y, z, a, lg).numpy()
            for x, y, z, a, lg in zip(xs, ys, z_rels, alphas, local_geoms)
        ]
    )
    assert np.isclose(pred_ampvecs0, pred_ampvecs).all()

    # test find_alpha
    in_probe_mask = in_probe_ci[chans]
    pred_ampvecs = torch.tensor(pred_ampvecs, dtype=torch.double)
    alpha1 = localize_torch.vmap_point_source_find_alpha(
        pred_ampvecs,
        in_probe_mask,
        xs,
        ys,
        z_rels,
        local_geoms,
    )
    assert np.isclose(alpha1, alphas).all()

    # recover positions
    res = localize_torch.localize_amplitude_vectors(pred_ampvecs, g, chans, ci)
    # it's not that good I guess :|
    tol = 3
    assert (res["x"] - xs).abs().max() < tol
    assert (res["z_rel"] - z_rels).abs().max() < tol
    assert (res["z_abs"] - z_abss).abs().max() < tol
    assert (res["alpha"] - alphas).abs().max() < tol
    assert (res["y"] - ys).abs().max() < tol

    rescom = localize_torch.localize_amplitude_vectors(
        pred_ampvecs, g, chans, ci, model="com"
    )
    # well, com is way worse :)
    tol = 90
    assert (rescom["x"] - xs).abs().max() < tol
    assert (rescom["z_rel"] - z_rels).abs().max() < tol
    assert (rescom["z_abs"] - z_abss).abs().max() < tol
