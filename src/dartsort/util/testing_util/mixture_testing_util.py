"""Utilities for making toy data and fitting PPCA or mixtures thereof to it"""

from typing import Literal

import numpy as np
import torch


def simulate_moppca(
    N=2**14,
    rank=4,
    nc=8,
    M=3,
    n_missing=2,
    K=5,
    t_mu: Literal["zero", "random"] = "random",
    t_cov: Literal["eye", "random"] = "eye",
    # zero, hot, random,
    t_w: Literal["zero", "hot", "random"] = "zero",
    t_missing: Literal[None, "random", "by_cluster"] = None,
    init_label_corruption: float = 0.0,
    snr: float = 10.0,
    rg=0,
):
    import dartsort

    rg = np.random.default_rng(0)
    D = rank * nc

    if t_mu == "zero":
        mu = np.zeros((K, rank, nc))
    elif t_mu == "random":
        mu = snr * rg.normal(size=(K, rank, nc))
    else:
        assert False

    if t_cov == "eye":
        cov = np.eye(D)
        noise = dartsort.EmbeddedNoise(
            rank, nc, cov_kind="scalar", global_std=torch.tensor(1.0, dtype=torch.float)
        )
    elif t_cov == "random":
        _c = rg.normal(size=(D, 10 * D))
        cov = _c @ _c.T / (10 * D)
        full_cov = torch.asarray(cov, dtype=torch.float).view(rank, nc, rank, nc)
        noise = dartsort.EmbeddedNoise(rank, nc, cov_kind="full", full_cov=full_cov)
    else:
        assert False

    if t_w == "hot":
        W = np.zeros((K, rank, nc, M))
        for j in range(K):
            for m in range(M):
                W[j, m, j, m] = 10 - m
    elif t_w == "random":
        W = rg.normal(size=(K, rank, nc, M))
    elif t_w == "zero":
        W = np.zeros((K, rank, nc, M))
    else:
        assert False

    labels = rg.integers(K, size=N)
    u = rg.normal(size=(N, M))
    eps = rg.multivariate_normal(mean=np.zeros(D), cov=cov, size=N)

    labels = torch.asarray(labels)
    cov = noise.marginal_covariance()
    print(f"{cov.logdet()=}")
    u = torch.asarray(u, dtype=torch.float)
    eps = torch.asarray(eps, dtype=torch.float)
    mu = torch.asarray(mu, dtype=torch.float)
    W = torch.asarray(W, dtype=torch.float)

    y = mu[labels] + torch.einsum("nrcm,nm->nrc", W[labels], u) + eps.view(N, rank, nc)

    if not t_missing:
        x = y
        channels = torch.arange(nc).unsqueeze(0).broadcast_to(N, nc)
    elif t_missing == "random":
        possible_neighbs = np.zeros(nc - n_missing, dtype=int)[None]
        npair = (nc * (nc - 1)) // 2
        possible_neighbs = np.broadcast_to(possible_neighbs, (npair, nc - n_missing))
        possible_neighbs = np.ascontiguousarray(possible_neighbs)
        mask = np.ones(nc, dtype=bool)
        ct = 0
        for i in range(nc):
            mask[i] = False
            for j in range(i + 1, nc):
                mask[j] = False
                possible_neighbs[ct] = np.flatnonzero(mask)
                mask[j] = True
                ct += 1
            mask[i] = True
        assert ct == npair
        channels = possible_neighbs[rg.integers(npair, size=N)]
    elif t_missing == "by_cluster":
        clus_neighbs = [
            rg.choice(nc, size=nc - n_missing, replace=False) for _ in range(K)
        ]
        clus_neighbs = np.array(clus_neighbs)
        channels = clus_neighbs[labels]

    if t_missing:
        channels = torch.asarray(channels)
        x = torch.take_along_dim(y, channels.unsqueeze(1), dim=2)

    channels = torch.asarray(channels)
    neighbs = dartsort.SpikeNeighborhoods.from_channels(channels, nc)

    init_labels = labels.clone()
    if init_label_corruption:
        to_corrupt = rg.binomial(1, init_label_corruption, size=N)
        init_labels[to_corrupt] = rg.integers(K, size=to_corrupt.sum())
    print(f"{init_labels=}")

    init_sorting = dartsort.DARTsortSorting(
        times_samples=torch.arange(N),
        channels=torch.zeros(N, dtype=int),
        labels=init_labels,
        sampling_frequency=100.0,
        extra_features=dict(times_seconds=torch.arange(N) / 100.0),
    )

    _tpca = dartsort.transform.TemporalPCAFeaturizer(
        channel_index=torch.zeros(nc, nc - n_missing, dtype=int),
        rank=rank,
    )
    data = dartsort.StableSpikeDataset(
        original_sorting=init_sorting,
        kept_indices=torch.arange(N),
        prgeom=torch.arange(nc + 1, dtype=torch.float).unsqueeze(1),
        tpca=_tpca,
        extract_channels=channels,
        core_channels=channels,
        core_features=x,
        extract_features=x,
        core_neighborhoods=neighbs,
        extract_neighborhoods=neighbs,
    )
    return dict(
        data=data,
        init_sorting=init_sorting,
        neighborhoods=neighbs,
        channels=channels,
        noise=noise,
        M=M,
        mu=mu,
        W=W,
        labels=labels,
        cov=cov,
    )


def fit_moppcas(
    data,
    noise,
    M=3,
    cov_kind="zero",
    inner_em_iter=100,
    em_converged_atol=0.05,
    with_noise_unit=True,
    return_before_fit=False,
):
    import dartsort

    N = data.n_spikes

    mm = dartsort.SpikeMixtureModel(
        data,
        noise,
        n_spikes_fit=N,
        cov_kind=cov_kind,
        ppca_rank=M,
        prior_pseudocount=0.0,
        ppca_inner_em_iter=inner_em_iter,
        ppca_atol=em_converged_atol,
        em_converged_churn=1e-6,
        em_converged_atol=1e-6,
        em_converged_prop=1e-6,
        n_threads=1,
        with_noise_unit=with_noise_unit,
    )
    torch.manual_seed(0)
    if not return_before_fit:
        mm.em()

    return mm


def fit_ppca(
    data,
    noise,
    neighborhoods,
    M=1,
    n_iter=50,
    show_progress=True,
    normalize=True,
    em_converged_atol=1e-6,
):
    from dartsort.cluster import ppcalib

    nc = neighborhoods.n_channels
    print(f"{nc=}")
    res = ppcalib.ppca_em(
        data.spike_data(indices=slice(None), with_neighborhood_ids=True),
        noise,
        neighborhoods,
        active_channels=torch.arange(nc),
        M=M,
        n_iter=n_iter,
        show_progress=show_progress,
        normalize=normalize,
        em_converged_atol=em_converged_atol,
    )
    return res


def compare_subspaces(
    mu,
    W,
    umu=None,
    uW=None,
    gmm=None,
    k=None,
    make_vis=True,
    figsize=(4, 3),
    title=None,
):
    if k is not None:
        mu = mu[k]
        W = W[k]
    else:
        if mu.ndim == 3:
            assert mu.shape[0] == 1
            mu = mu[0]
            W = W[0]
    if torch.is_tensor(mu):
        mu = mu.numpy()
        W = W.numpy()

    if umu is None:
        unit = gmm[k]
        uW = getattr(unit, "W", np.zeros_like(W))
        umu = getattr(unit, "mean", np.zeros_like(mu))
    if torch.is_tensor(uW):
        uW = uW.numpy()
    if torch.is_tensor(umu):
        umu = umu.numpy()

    assert umu.shape == mu.shape, f"{umu.shape=} {mu.shape=}"
    assert uW.shape == W.shape, f"{uW.shape=} {W.shape=}"

    rank, nc, M = W.shape
    WTW = W.reshape(rank * nc, M)
    WTW = WTW @ WTW.T
    uWTW = uW.reshape(rank * nc, M)
    uWTW = uWTW @ uWTW.T

    muerr = mu - umu
    werr = WTW - uWTW
    if not make_vis:
        return muerr, werr, None

    import matplotlib.pyplot as plt
    import dartsort.vis as dartvis

    panel = plt.figure(figsize=figsize, layout="constrained")
    top, bot = panel.subfigures(nrows=2)
    ax_mu, ax_muerr, ax_werr = top.subplots(ncols=3)
    ax_w, ax_uw, ax_dw = bot.subplots(ncols=3, sharex=True, sharey=True)

    ax_mu.plot(mu.ravel(), color="k", label="gt")
    color = dartvis.glasbey1024[k] if k is not None else "b"
    ax_mu.plot(umu.ravel(), color=color, label="est")
    ax_mu.legend(title="mean")
    ax_muerr.hist(muerr.ravel(), bins=32, log=True)
    ax_muerr.set_xlabel("mean errors")
    ax_werr.hist(werr.ravel(), bins=32, log=True)
    ax_werr.set_xlabel("subspace errors")

    vm = max(np.abs(WTW).max(), 0.9 * np.abs(uWTW).max())
    kw = dict(vmin=-vm, vmax=vm, cmap=plt.cm.seismic, interpolation="none")
    im = ax_w.imshow(WTW, **kw)
    ax_w.set_title("gt subspace")
    ax_uw.imshow(uWTW, **kw)
    ax_uw.set_title("est subspace")
    ax_dw.imshow(WTW - uWTW, **kw)
    ax_dw.set_title("diff")
    plt.colorbar(im, ax=ax_dw, shrink=0.75, aspect=10)
    if title:
        panel.suptitle(title)

    return muerr, werr, panel


def visually_compare_means(gmm, mu, figsize=(3, 2)):
    K = len(mu)
    import matplotlib.pyplot as plt
    from dartsort.vis import glasbey1024

    fig, axes = plt.subplots(figsize=figsize)
    for k in range(K):
        axes.plot(mu[k].view(-1), color=glasbey1024[k])
        axes.plot(gmm[k].mean.view(-1), color=glasbey1024[k], ls=":")
    return fig, axes


def test_ppca(
    N=2**14,
    rank=4,
    nc=8,
    M=3,
    t_mu: Literal["zero", "random"] = "random",
    t_cov: Literal["eye", "random"] = "eye",
    # zero, hot, random,
    t_w: Literal["zero", "hot", "random"] = "zero",
    t_missing: Literal[None, "random"] = None,
    n_missing=2,
    em_iter=100,
    em_converged_atol=1e-6,
    make_vis=True,
    show_vis=False,
    figsize=(4, 3),
    normalize=True,
    rg=0,
):
    rg = np.random.default_rng(rg)
    sim_res = simulate_moppca(
        N=N,
        rank=rank,
        nc=nc,
        M=M,
        n_missing=n_missing,
        K=1,
        t_mu=t_mu,
        t_cov=t_cov,
        t_w=t_w,
        t_missing=t_missing,
        rg=rg,
    )
    torch.manual_seed(
        np.frombuffer(np.random.default_rng(0).bytes(8), dtype=np.int64).item()
    )
    ppca_res = fit_ppca(
        sim_res["data"],
        sim_res["noise"],
        sim_res["neighborhoods"],
        M=M,
        n_iter=em_iter,
        show_progress=make_vis,
        normalize=normalize,
        em_converged_atol=em_converged_atol,
    )

    muerr, werr, panel = compare_subspaces(
        mu=sim_res["mu"],
        W=sim_res["W"],
        umu=ppca_res["mu"],
        uW=ppca_res["W"],
        make_vis=make_vis,
        figsize=figsize,
        title=f"{t_mu=} {t_cov=} {t_w=} {t_missing=}",
    )
    if make_vis and show_vis:
        import matplotlib.pyplot as plt

        # panel.show()
        plt.show()
        plt.close(panel)
    res = dict(sim_res=sim_res, ppca_res=ppca_res, muerr=muerr, Werr=werr, panel=panel)
    return res


def test_moppcas(
    N=2**14,
    rank=4,
    nc=8,
    M=3,
    n_missing=2,
    K=5,
    t_mu: Literal["zero", "random"] = "random",
    t_cov: Literal["eye", "random"] = "eye",
    # zero, hot, random,
    t_w: Literal["zero", "hot", "random"] = "zero",
    t_missing: Literal[None, "random", "by_cluster"] = None,
    init_label_corruption: float = 0.0,
    inner_em_iter=100,
    em_converged_atol=0.05,
    with_noise_unit=True,
    make_vis=True,
    figsize=(4, 3),
    return_before_fit=False,
    snr=10.0,
    rg=0,
):
    rg = np.random.default_rng(rg)
    sim_res = simulate_moppca(
        N=N,
        rank=rank,
        nc=nc,
        M=M,
        n_missing=n_missing,
        K=K,
        t_mu=t_mu,
        t_cov=t_cov,
        t_w=t_w,
        t_missing=t_missing,
        init_label_corruption=init_label_corruption,
        snr=snr,
        rg=rg,
    )
    mm = fit_moppcas(
        sim_res["data"],
        sim_res["noise"],
        M=M,
        cov_kind="zero" if t_w == "zero" else "ppca",
        inner_em_iter=inner_em_iter,
        em_converged_atol=em_converged_atol,
        with_noise_unit=with_noise_unit,
        return_before_fit=return_before_fit,
    )
    if return_before_fit:
        return dict(sim_res=sim_res, gmm=mm)

    acc = (mm.labels == sim_res["init_sorting"].labels).sum() / N
    print(f"accuracy: {acc}")

    muerrs = []
    Werrs = []
    for k in range(K):
        muerr, werr, panel = compare_subspaces(
            sim_res["mu"],
            sim_res["W"],
            gmm=mm,
            k=k,
            make_vis=make_vis,
            figsize=figsize,
            title=f"{t_mu=} {t_cov=} {t_w=} {t_missing=} | {k=}",
        )
        muerrs.append(muerr)
        Werrs.append(werr)
        if make_vis:
            import matplotlib.pyplot as plt

            panel.show()
            plt.close(panel)
    muerrs = np.stack(muerrs, axis=0)
    Werrs = np.stack(Werrs, axis=0)
    results = dict(sim_res=sim_res, gmm=mm, acc=acc, muerrs=muerrs, Werrs=Werrs)
    return results