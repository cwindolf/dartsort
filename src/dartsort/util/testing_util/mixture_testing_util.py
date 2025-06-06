"""Utilities for making toy data and fitting PPCA or mixtures thereof to it"""

from typing import Literal

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score


def simulate_moppca(
    Nper=2**12,
    rank=4,
    nc=8,
    M=3,
    n_missing=2,
    K=5,
    t_mu: Literal["zero", "random"] = "random",
    t_cov: Literal["eye", "random"] = "eye",
    # zero, hot, random,
    t_w: Literal["zero", "hot", "random"] = "zero",
    t_missing: Literal[None, "random", "skewed", "by_cluster"] = None,
    init_label_corruption: float = 0.0,
    snr: float = 10.0,
    rg=0,
):
    import dartsort

    N = Nper * K

    rg = np.random.default_rng(rg)
    D = rank * nc

    clus_neighbs = None
    clus_mask = None
    if t_missing == "by_cluster":
        # in this case, missing chans per clus are never observed,
        # so we should treat them as 0 in W and mu to avoid issues with metrics
        clus_neighbs = [
            rg.choice(nc, size=nc - n_missing, replace=False) for _ in range(K)
        ]
        clus_neighbs = np.array(clus_neighbs)
        clus_neighbs = np.sort(clus_neighbs, axis=1)
        clus_mask = np.zeros((K, nc))
        for j, n in enumerate(clus_neighbs):
            clus_mask[j, n] = 1.0

    if t_mu == "zero":
        mu = np.zeros((K, rank, nc))
    elif t_mu == "random":
        mu = snr * rg.normal(size=(K, rank, nc))
    else:
        assert False
    if clus_mask is not None:
        mu = mu * clus_mask[:, None]

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
    if clus_mask is not None:
        W = W * clus_mask[:, None, :, None]

    labels = rg.integers(K, size=N)
    u = rg.normal(size=(N, M))
    eps = rg.multivariate_normal(mean=np.zeros(D), cov=cov, size=N)

    labels = torch.asarray(labels)
    cov = noise.marginal_covariance()
    u = torch.asarray(u, dtype=torch.float)
    eps = torch.asarray(eps, dtype=torch.float)
    mu = torch.asarray(mu, dtype=torch.float)
    W = torch.asarray(W, dtype=torch.float)

    y = mu[labels] + torch.einsum("nrcm,nm->nrc", W[labels], u) + eps.view(N, rank, nc)

    x = None
    if not t_missing:
        x = y
        channels = torch.arange(nc).unsqueeze(0).broadcast_to(N, nc)
    elif t_missing == "random":
        possible_neighbs = np.zeros(nc - n_missing, dtype=np.int64)[None]
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
    elif t_missing == "skewed":
        nc_miss = min(2, nc - 2)
        assert nc_miss > 0
        n0 = np.arange(nc - nc_miss)
        n1 = nc_miss + n0
        neighbs = np.stack((n0, n1), axis=0)
        ns_missing = int(N / 3)
        choices = np.concatenate(
            (
                np.zeros(ns_missing, dtype=np.int64),
                np.ones(N - ns_missing, dtype=np.int64),
            )
        )
        rg.shuffle(choices)
        channels = neighbs[choices]
    elif t_missing == "by_cluster":
        assert clus_neighbs is not None
        channels = clus_neighbs[labels]
    channels = torch.asarray(channels, dtype=torch.long)

    if t_missing:
        x = torch.take_along_dim(y, channels.unsqueeze(1), dim=2)
    assert x is not None

    neighbs = dartsort.SpikeNeighborhoods.from_channels(channels, nc)

    init_labels = labels.clone()
    if init_label_corruption:
        to_corrupt = rg.binomial(1, init_label_corruption, size=N).astype(bool)
        init_labels[to_corrupt] = torch.from_numpy(
            rg.integers(K, size=to_corrupt.sum())
        )

    init_sorting = dartsort.DARTsortSorting(
        times_samples=torch.arange(N).to(torch.long),
        channels=torch.zeros(N, dtype=torch.long),
        labels=init_labels,
        sampling_frequency=100.0,
        extra_features=dict(times_seconds=torch.arange(N) / 100.0),
    )

    _tpca = dartsort.transform.TemporalPCAFeaturizer(
        channel_index=torch.zeros(nc, nc - n_missing, dtype=torch.long),
        rank=rank,
    )
    splits = rg.binomial(1, p=0.3, size=N)
    data = dartsort.StableSpikeDataset(
        original_sorting=init_sorting,
        kept_indices=np.arange(N),
        prgeom=torch.arange(nc + 1, dtype=torch.float).unsqueeze(1),
        tpca=_tpca,
        extract_channels=channels,
        core_channels=channels,
        core_features=x,
        train_extract_features=x[splits == 0],
        split_names=["train", "val"],
        split_mask=torch.from_numpy(splits),
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
        K=K,
        cov=cov,
        x=x,
    )


def fit_moppcas(
    data,
    noise,
    M=3,
    cov_kind="zero",
    inner_em_iter=100,
    n_em_iters=50,
    em_converged_atol=0.05,
    with_noise_unit=True,
    return_before_fit=False,
    channels_strategy="count",
    inference_algorithm="em",
    n_refinement_iters=0,
    gmm_kw={},
):
    import dartsort

    N = data.n_spikes

    mm = dartsort.SpikeMixtureModel(
        data,
        noise,
        n_spikes_fit=N,
        cov_kind=cov_kind,
        ppca_rank=M,
        ppca_inner_em_iter=inner_em_iter,
        ppca_atol=em_converged_atol,
        em_converged_churn=1e-6,
        em_converged_atol=1e-6,
        em_converged_prop=1e-6,
        n_threads=1,
        n_em_iters=n_em_iters,
        with_noise_unit=with_noise_unit,
        channels_strategy=channels_strategy,
        **gmm_kw,
    )
    torch.manual_seed(0)
    elbos = []

    if return_before_fit:
        return mm, dict(elbos=elbos)

    if inference_algorithm == "em":
        mm.log_liks = mm.em()
    elif inference_algorithm in ("tem", "tvi"):
        res = mm.tvi()
        mm.log_liks = res["log_liks"]
        elbos.append(np.array([r["obs_elbo"] for r in res["records"]]))
    elif inference_algorithm == "tsgd":
        res = mm.tvi(algorithm="adam")
        mm.log_liks = res["log_liks"]
    else:
        assert False

    for j in range(n_refinement_iters):
        mm.split()

        if inference_algorithm == "em":
            mm.log_liks = mm.em()
        elif inference_algorithmin("tem", "tvi"):
            res = mm.tvi()
            mm.log_liks = res["log_liks"]
            elbos.append(np.array([r["obs_elbo"] for r in res["records"]]))
        elif inference_algorithm == "tsgd":
            res = mm.tvi(algorithm="adam")
            mm.log_liks = res["log_liks"]
        else:
            assert False

        mm.merge(mm.log_liks)

        if inference_algorithm == "em":
            mm.log_liks = mm.em()
        elif inference_algorithmin("tem", "tvi"):
            res = mm.tvi()
            mm.log_liks = res["log_liks"]
            elbos.append(np.array([r["obs_elbo"] for r in res["records"]]))
        elif inference_algorithm == "tsgd":
            res = mm.tvi(algorithm="adam")
            mm.log_liks = res["log_liks"]
        else:
            assert False

    return mm, dict(elbos=elbos)


def fit_ppca(
    data,
    noise,
    neighborhoods,
    M=1,
    n_iter=50,
    show_progress=True,
    normalize=True,
    em_converged_atol=1e-6,
    cache_local_direct=False,
    W_initialization="svd",
    prior_pseudocount=0,
    laplace_ard=False,
):
    from dartsort.cluster import ppcalib

    nc = neighborhoods.n_channels
    res = ppcalib.ppca_em(
        data.spike_data(
            indices=slice(None), split_indices=slice(None), with_neighborhood_ids=True
        ),
        noise,
        neighborhoods,
        active_channels=torch.arange(nc),
        M=M,
        n_iter=n_iter,
        show_progress=show_progress,
        normalize=normalize,
        em_converged_atol=em_converged_atol,
        cache_local_direct=cache_local_direct,
        W_initialization=W_initialization,
        prior_pseudocount=prior_pseudocount,
        laplace_ard=laplace_ard,
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
        unit = {}
        if k in gmm:
            unit = gmm[k]
        uW = getattr(unit, "W", np.zeros_like(W))
        umu = getattr(unit, "mean", np.zeros_like(mu))
    M = 0
    werr = None
    if torch.is_tensor(uW):
        uW = uW.numpy()
        assert uW.shape == W.shape, f"{uW.shape=} {W.shape=}"
        rank, nc, M = W.shape
        WTW = W.reshape(rank * nc, M)
        WTW = WTW @ WTW.T
        uWTW = uW.reshape(rank * nc, M)
        uWTW = uWTW @ uWTW.T
        werr = WTW - uWTW
    if torch.is_tensor(umu):
        umu = umu.numpy()
        assert umu.shape == mu.shape, f"{umu.shape=} {mu.shape=}"

    muerr = mu - umu
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

    if M:
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
    Nper=2**12,
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
    em_converged_atol=1e-3,
    make_vis=False,
    show_vis=False,
    figsize=(4, 3),
    normalize=True,
    cache_local=False,
    W_initialization="svd",
    prior_pseudocount=0,
    laplace_ard=False,
    sim_res=None,
    rg=0,
):
    rg = np.random.default_rng(rg)
    if sim_res is None:
        sim_res = simulate_moppca(
            Nper=Nper,
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
        M=M * (t_w != "zero"),
        n_iter=em_iter,
        show_progress=make_vis,
        normalize=normalize,
        em_converged_atol=em_converged_atol,
        cache_local_direct=cache_local,
        W_initialization=W_initialization,
        prior_pseudocount=prior_pseudocount,
        laplace_ard=laplace_ard,
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
    Nper=2**12,
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
    em_converged_atol=1e-3,
    with_noise_unit=True,
    make_vis=False,
    figsize=(4, 3),
    return_before_fit=False,
    channels_strategy="count",
    snr=10.0,
    rg=0,
    inference_algorithm="em",
    n_refinement_iters=0,
    n_em_iters=50,
    do_comparison=True,
    sim_res=None,
    zero_radius=None,
    gmm_kw={},
):
    rg = np.random.default_rng(rg)
    if sim_res is None:
        sim_res = simulate_moppca(
            Nper=Nper,
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
    sim_res["noise"].zero_radius = zero_radius
    mm, fit_info = fit_moppcas(
        sim_res["data"],
        sim_res["noise"],
        M=M,
        n_em_iters=n_em_iters,
        cov_kind="zero" if t_w == "zero" else "ppca",
        inner_em_iter=inner_em_iter,
        em_converged_atol=em_converged_atol,
        with_noise_unit=with_noise_unit,
        return_before_fit=return_before_fit,
        channels_strategy=channels_strategy,
        inference_algorithm=inference_algorithm,
        n_refinement_iters=n_refinement_iters,
        gmm_kw=gmm_kw,
    )
    if return_before_fit:
        return dict(sim_res=sim_res, gmm=mm, fit_info=fit_info)

    N = K * Nper
    assert mm.labels.shape == sim_res["labels"].shape == (N,)
    acc = (mm.labels == sim_res["labels"]).sum() / N
    print(f"accuracy: {acc}")
    ari = adjusted_rand_score(sim_res["labels"], mm.labels)
    print(f"ari: {ari}")
    ids, means, covs, logdets = mm.stack_units()

    muerrs = []
    Werrs = []
    if do_comparison:
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
    results = dict(
        sim_res=sim_res,
        gmm=mm,
        acc=acc,
        muerrs=muerrs,
        Werrs=Werrs,
        mm_means=means,
        mm_W=covs,
        mm_logdets=logdets,
        init_label_corruption=init_label_corruption,
        M=M,
        ari=ari,
        fit_info=fit_info,
    )
    return results
