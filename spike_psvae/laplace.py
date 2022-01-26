import numpy as np
import torch
from torch.autograd.functional import hessian


def normal_lpdf(x, mu=0, sigma=1):
    return -0.5 * torch.square((x - mu) / sigma).sum()


def point_source_post_lpdf(x, y, z, alpha, ptp, lgeom, ptp_sigma=0.1):
    pred_ptp = alpha / torch.sqrt(
        torch.square(x - lgeom[:, 0])
        + torch.square(z - lgeom[:, 1])
        + torch.square(y)
    )
    return normal_lpdf(ptp - pred_ptp, sigma=ptp_sigma)


def point_source_post_lpdf_hessian(
    x0, y0, z0, alpha0, ptp, lgeom, ptp_sigma=0.1
):
    z0 = torch.tensor(z0)
    alpha0 = torch.tensor(alpha0)
    ptp = torch.tensor(ptp)
    lgeom = torch.tensor(lgeom)

    def closure(xy):
        return point_source_post_lpdf(
            xy[0], xy[1], z0, alpha0, ptp, lgeom, ptp_sigma=ptp_sigma
        )

    H = hessian(
        closure,
        (
            torch.tensor([x0, y0])
        ),
    )

    return H.numpy()


def laplace_approx_samples(
    x0,
    y0,
    z0,
    alpha0,
    ptp,
    lgeom,
    ptp_sigma=0.1,
    n_samples=4000,
    seed=0,
):
    """Sample x,y from our truncated Laplace approx"""
    H = point_source_post_lpdf_hessian(
        x0, y0, z0, alpha0, ptp, lgeom, ptp_sigma=ptp_sigma
    )

    rg = np.random.default_rng(seed)
    x_samples = []
    y_samples = []
    while sum(len(s) for s in x_samples) < n_samples:
        new_samples = rg.multivariate_normal([x0, y0], -H, size=1000)
        new_samples = new_samples[new_samples[:, 1] > 0]
        x_samples.append(new_samples[:, 0])
        y_samples.append(new_samples[:, 1])

    return np.hstack(x_samples)[:n_samples], np.hstack(y_samples)[:n_samples]
