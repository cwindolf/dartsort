import numpy as np
import torch
from linear_operator import operators
from scipy.stats import multivariate_normal

from dartsort.util import more_operators, spiketorch


def test_lowranksolve():
    N = 2**5
    D = 8
    mean = np.zeros(D)

    # test case: normal log likelihoods
    # using scipy for reference point
    rg = np.random.default_rng(0)

    # make a low rank + identity covariance
    v = rg.normal(size=(D, 2))
    cov = np.eye(D) + v @ v.T

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
    dense_eye = operators.DenseLinearOperator(torch.eye(D))
    diag_eye = operators.DiagLinearOperator(torch.ones(D))

    # the low rank part...
    root = operators.LowRankRootLinearOperator(v)

    # guests of honor
    my_root_cov_dense = more_operators.LowRankRootSumLinearOperator(dense_eye, root)
    my_root_cov_diag = more_operators.LowRankRootSumLinearOperator(diag_eye, root)

    # test inv_quad_logdet
    dense_root_lls = spiketorch.ll_via_inv_quad(my_root_cov_dense, y)
    diag_root_lls = spiketorch.ll_via_inv_quad(my_root_cov_diag, y)
    assert np.isclose(scipy_lls, dense_root_lls).all()
    assert np.isclose(scipy_lls, diag_root_lls).all()

    # test solve
    solve_gt = torch.linalg.solve(cov, y.T).T
    dense_solve = my_root_cov_dense.solve(y.T).T
    diag_solve = my_root_cov_diag.solve(y.T).T
    assert np.isclose(solve_gt, dense_solve, atol=1e-5).all()
    assert np.isclose(solve_gt, diag_solve, atol=1e-5).all()
