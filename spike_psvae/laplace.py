import numpy as np
import scipy.linalg as la
from scipy.stats import norm
import torch
from torch.autograd.functional import hessian


def tensor_wrap(func):
    def wrapper(*args, **kwargs):
        if all(not isinstance(a, torch.Tensor) for a in args):
            return func(
                *map(lambda x: torch.tensor(x, requires_grad=False), args),
                **kwargs
            ).numpy()
        else:
            return func(*args, **kwargs)

    return wrapper


def normal_lpdf(x, mu=0, sigma=1):
    return -0.5 * torch.square((x - mu) / sigma).sum()


@tensor_wrap
def point_source_post_lpdf(x, y, z, alpha, ptp, lgeom, ptp_sigma=0.1):
    # print(x, y, z, alpha, ptp, lgeom, ptp_sigma)
    pred_ptp = alpha / torch.sqrt(
        torch.square(x - lgeom[:, 0])
        + torch.square(z - lgeom[:, 1])
        + torch.square(y)
    )
    return normal_lpdf(ptp - pred_ptp, sigma=ptp_sigma)


def point_source_post_lpdf_hessian_xy(
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

    H = hessian(closure, torch.tensor([x0, y0]))

    return H.numpy()


def point_source_post_lpdf_hessian_xy_polar(
    r0, theta0, z0, alpha0, ptp, lgeom, ptp_sigma=0.1
):
    z0 = torch.tensor(z0)
    alpha0 = torch.tensor(alpha0)
    ptp = torch.tensor(ptp)
    lgeom = torch.tensor(lgeom)

    def closure(rtheta):
        r, theta = rtheta
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return point_source_post_lpdf(
            x, y, z0, alpha0, ptp, lgeom, ptp_sigma=ptp_sigma
        )

    H = hessian(closure, torch.tensor([r0, theta0]))

    return H.numpy()


def point_source_post_lpdf_hessian_xyza(
    x0, y0, z0, alpha0, ptp, lgeom, ptp_sigma=0.1
):
    ptp = torch.tensor(ptp)
    lgeom = torch.tensor(lgeom)

    def closure(xyza):
        return point_source_post_lpdf(
            xyza[0], xyza[1], xyza[2], xyza[3], ptp, lgeom, ptp_sigma=ptp_sigma
        )

    H = hessian(closure, torch.tensor([x0, y0, z0, alpha0]))

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
    H = point_source_post_lpdf_hessian_xyza(
        x0, y0, z0, alpha0, ptp, lgeom, ptp_sigma=ptp_sigma
    )

    rg = np.random.default_rng(seed)
    x_samples = []
    y_samples = []
    while sum(len(s) for s in x_samples) < n_samples:
        new_samples = rg.multivariate_normal(
            [x0, y0, z0, alpha0], -la.inv(H), size=1000
        )
        new_samples = new_samples[new_samples[:, 1] > 0]
        x_samples.append(new_samples[:, 0])
        y_samples.append(new_samples[:, 1])

    return np.hstack(x_samples)[:n_samples], np.hstack(y_samples)[:n_samples]


def laplace_approx_samples_polar(
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
    r0 = np.sqrt(x0 ** 2 + y0 ** 2)
    theta0 = np.arctan2(y0, x0)
    print(x0, y0, r0, theta0)
    H = point_source_post_lpdf_hessian_xy_polar(
        r0, theta0, z0, alpha0, ptp, lgeom, ptp_sigma=ptp_sigma
    )

    rg = np.random.default_rng(seed)
    x_samples = []
    y_samples = []
    while sum(len(s) for s in x_samples) < n_samples:
        new_samples = rg.multivariate_normal(
            [r0, theta0], -la.inv(H), size=1000
        )
        x = new_samples[:, 0] * np.cos(new_samples[:, 1])
        y = new_samples[:, 0] * np.sin(new_samples[:, 1])
        x_samples.append(x[y > 0])
        y_samples.append(y[y > 0])

    return np.hstack(x_samples)[:n_samples], np.hstack(y_samples)[:n_samples]


def laplace_correct(x0, y0, z0, alpha0, ptp, lgeom, ptp_sigma=0.1):
    H = point_source_post_lpdf_hessian_xyza(
        x0, y0, z0, alpha0, ptp, lgeom, ptp_sigma=ptp_sigma
    )
    V = -la.inv(H)
    x_std = np.sqrt(V[0, 0])
    y_std = np.sqrt(V[1, 1])
    rho = V[0, 1] / (x_std * y_std)
    phi_alpha = norm.pdf(-y0 / y_std)
    Phi_alpha = norm.cdf(-y0 / y_std)

    ey = y0 + y_std * phi_alpha / (1 - Phi_alpha)
    eu = rho * phi_alpha / (1 - Phi_alpha)
    ex = x0 + x_std * eu
    return ex, ey
