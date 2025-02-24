import numpy as np
import torch
from linear_operator import operators
from scipy.stats import multivariate_normal

from dartsort.util import more_operators, spiketorch


def _test_lowranksolve(cov_kind="eye"):
    N = 2**5
    D = 8
    mean = np.zeros(D)

    # test case: normal log likelihoods
    # using scipy for reference point
    rg = np.random.default_rng(0)

    if cov_kind == "eye":
        cov0 = np.eye(D)
    elif cov_kind == "random":
        cov0 = rg.normal(size=(D, 2 * D))
        cov0 = cov0 @ cov0.T
        cov0 += 0.01 * np.eye(D)
    else:
        assert False

    # make a low rank + identity covariance
    v = rg.normal(size=(D, 2))
    cov = cov0 + v @ v.T

    # draws...
    y = rg.multivariate_normal(mean=mean, cov=cov, size=N)

    # log liks
    rv = multivariate_normal(mean=mean, cov=cov)
    scipy_lls = rv.logpdf(y)

    # convert to torch for below
    mean = torch.asarray(mean, dtype=torch.float)
    v = torch.asarray(v, dtype=torch.float)
    cov = torch.asarray(cov, dtype=torch.float)
    y = torch.asarray(y, dtype=torch.float)

    # some different identities
    if cov_kind == "eye":
        dense_eye = operators.DenseLinearOperator(torch.eye(D))
        diag_eye = operators.DiagLinearOperator(torch.ones(D))
        cov0s = [dense_eye, diag_eye]
    elif cov_kind == "random":
        cov0 = operators.DenseLinearOperator(torch.tensor(cov0, dtype=torch.float))
        chol_cov0 = operators.CholLinearOperator(cov0.cholesky())
        cov0s = [cov0, chol_cov0]
    else:
        assert False

    # the low rank part...
    root = operators.LowRankRootLinearOperator(v)

    for cov0 in cov0s:
        # guests of honor
        my_root_cov = more_operators.LowRankRootSumLinearOperator(cov0, root)

        # test inv_quad_logdet
        my_lls = spiketorch.ll_via_inv_quad(my_root_cov, y)
        assert np.isclose(scipy_lls, my_lls).all()
        assert np.isclose(my_root_cov.logdet(), np.linalg.slogdet(cov).logabsdet)

        # test solve
        solve_gt = torch.linalg.solve(cov, y.T).T
        my_solve = my_root_cov.solve(y.T).T
        assert np.isclose(solve_gt, my_solve, atol=1e-5).all()


def test_lowranksolve_eye():
    _test_lowranksolve()


def test_lowranksolve_random():
    _test_lowranksolve("random")


if __name__ == "__main__":
    test_lowranksolve_random()
