"""Utilities for making toy data and fitting PPCA or mixtures thereof to it"""

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score


def simulate_moppca(
    Nper=2**12,
    rank=4,
    nc=8,
    M=2,
    n_missing=2,
    K=5,
    t_mu: Literal["zero", "smooth", "random"] = "smooth",
    t_cov: Literal["eye", "random", "eyesmall"] = "eye",
    # zero, hot, random,
    t_w: Literal["zero", "hot", "smooth", "random"] = "zero",
    t_missing: Literal[
        None, "random", "random_no_extrap", "skewed", "by_cluster"
    ] = None,
    init_label_corruption: float = 0.0,
    snr: float = 10.0,
    rg: int | np.random.Generator = 0,
    device=None,
):
    from dartsort.transform import TemporalPCAFeaturizer
    from dartsort.util.data_util import DARTsortSorting
    from dartsort.util.noise_util import EmbeddedNoise
    from dartsort.clustering.gmm.stable_features import (
        StableSpikeDataset,
        SpikeNeighborhoods,
    )

    N = Nper * K

    rg = np.random.default_rng(rg)
    D = rank * nc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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
    if t_missing in (None, "none"):
        n_missing = 0

    if t_mu == "zero":
        mu = np.zeros((K, rank, nc))
    elif t_mu == "random":
        mu = snr * rg.normal(size=(K, rank, nc))
    elif t_mu == "smooth":
        phase = rg.uniform(0.0, 2 * np.pi, size=(K, rank, 1))
        freq = rg.uniform(0.2, 1.0, size=(K, rank, 1))
        amp = rg.uniform(1.0, 2.0, size=(K, rank, 1))
        domain = np.linspace(0, 2 * np.pi, endpoint=False, num=nc)
        mu = snr * amp * np.sin(phase + freq * domain)
    else:
        assert False
    if clus_mask is not None:
        mu = mu * clus_mask[:, None]

    if t_cov == "eye":
        cov = np.eye(D)
        noise = EmbeddedNoise(
            rank,
            nc,
            cov_kind="scalar",
            global_std=torch.tensor(1.0, dtype=torch.float, device=device),
        )
    elif t_cov == "eyesmall":
        cov = 1e-4 * np.eye(D)
        noise = EmbeddedNoise(
            rank,
            nc,
            cov_kind="scalar",
            global_std=torch.tensor(0.01, dtype=torch.float, device=device),
        )
    elif t_cov == "random":
        _c = rg.normal(size=(D, 10 * D))
        cov = _c @ _c.T / (10 * D)
        full_cov = torch.asarray(cov, dtype=torch.float, device=device).view(
            rank, nc, rank, nc
        )
        noise = EmbeddedNoise(rank, nc, cov_kind="full", full_cov=full_cov)
    else:
        assert False

    if t_w == "hot":
        W = np.zeros((K, rank, nc, M))
        for j in range(K):
            for m in range(M):
                W[j, m, j, m] = 10 - m
    elif t_w == "random":
        W = rg.normal(size=(K, rank, nc, M))
    elif t_w == "smooth":
        phase = rg.uniform(0.0, 2 * np.pi, size=(K, rank, 1, M))
        freq = rg.uniform(0.2, 1.0, size=(K, rank, 1, M))
        amp = rg.uniform(1.0, 2.0, size=(K, rank, 1, M))
        domain = np.linspace(0, 2 * np.pi, endpoint=False, num=nc)
        W = amp * np.sin(phase + freq * domain[:, None])
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
    elif t_missing.startswith("random"):
        assert n_missing == 2
        no_extrap = int(t_missing == "random_no_extrap")
        possible_neighbs = np.zeros(nc - n_missing, dtype=np.int64)[None]
        npair = ((nc - 2 * no_extrap) * (nc - 1 - 2 * no_extrap)) // 2
        possible_neighbs = np.broadcast_to(possible_neighbs, (npair, nc - n_missing))
        possible_neighbs = np.ascontiguousarray(possible_neighbs)
        mask = np.ones(nc, dtype=bool)
        ct = 0
        for i in range(no_extrap, nc - no_extrap):
            mask[i] = False
            for j in range(i + 1, nc - no_extrap):
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
    else:
        assert False
    channels = torch.asarray(channels, dtype=torch.long)

    if t_missing:
        x = torch.take_along_dim(y, channels.unsqueeze(1), dim=2)
    assert x is not None

    neighbs = SpikeNeighborhoods.from_channels(channels, nc, device=device)

    init_labels = labels.clone()
    if init_label_corruption:
        to_corrupt = rg.binomial(1, init_label_corruption, size=N).astype(bool)
        init_labels[to_corrupt] = torch.from_numpy(
            rg.integers(K, size=to_corrupt.sum())
        )

    init_sorting = DARTsortSorting(
        times_samples=np.arange(N),
        channels=np.zeros(N, dtype=np.int64),
        labels=init_labels.numpy(force=True),
        sampling_frequency=100.0,
        ephemeral_features=dict(times_seconds=np.arange(N) / 100.0),
    )

    _tpca = TemporalPCAFeaturizer(
        channel_index=torch.zeros(nc, nc - n_missing, dtype=torch.long),
        rank=rank,
    )
    _tpca = _tpca.to(device)
    splits = rg.binomial(1, p=0.3, size=N)

    prgeom = 15.0 * torch.arange(nc, dtype=torch.float)
    prgeom = torch.stack((torch.zeros(nc), prgeom), dim=1)
    prgeom = F.pad(prgeom, (0, 0, 0, 1), value=torch.nan)

    data = StableSpikeDataset(
        original_sorting=init_sorting,
        kept_indices=np.arange(N),
        prgeom=prgeom,
        tpca=_tpca,
        extract_channels=channels,
        core_channels=channels,
        core_features=x,
        train_extract_features=x[splits == 0],
        split_names=["train", "val"],
        split_mask=torch.from_numpy(splits),
        device=device,
    )

    data = data.to(device)
    x = x.to(device)
    noise = noise.to(device)
    neighbs = neighbs.to(device)
    noise_log_priors = noise.detection_prior_log_prob(mu)
    noise_log_priors = noise_log_priors[labels]

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
        channel_observed_by_unit=clus_mask,
        noise_log_priors=noise_log_priors,
        n_channels=nc,
    )



def compare_subspaces(mu, W, umu=None, uW=None, gmm=None, k=None):
    if k is not None:
        mu = mu[k]
        W = W[k]
    else:
        if mu.ndim == 3:
            assert mu.shape[0] == 1
            mu = mu[0]
            W = W[0]
    if torch.is_tensor(mu):
        mu = mu.numpy(force=True)
        W = W.numpy(force=True)

    if umu is None:
        unit = {}
        if k in gmm:  # type: ignore
            unit = gmm[k]  # type: ignore
        uW = getattr(unit, "W", np.zeros_like(W))
        umu = getattr(unit, "mean", np.zeros_like(mu))
    M = 0
    werr = None
    if torch.is_tensor(uW):
        uW = uW.numpy(force=True)
        assert uW.shape == W.shape, f"{uW.shape=} {W.shape=}"
        rank, nc, M = W.shape
        WTW = W.reshape(rank * nc, M)
        WTW = WTW @ WTW.T
        uWTW = uW.reshape(rank * nc, M)
        uWTW = uWTW @ uWTW.T
        werr = WTW - uWTW
    if torch.is_tensor(umu):
        umu = umu.numpy(force=True)
        assert umu.shape == mu.shape, f"{umu.shape=} {mu.shape=}"

    muerr = mu - umu
    return muerr, werr

