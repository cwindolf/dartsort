from itertools import product

import numpy as np
import pytest

from dartsort.util.testing_util import mixture_testing_util

mu_atol = 0.01
wtw_rtol = 0.01

t_mu_test = ("zero", "random")
t_cov_test = ("eye", "random")
t_w_test = ("zero", "hot", "random")
t_missing_test = (None, "random")


@pytest.fixture(scope="module")
def ppca_simulations():
    simulations = {}
    for t_mu, t_cov, t_w, t_missing in product(
        t_mu_test, t_cov_test, t_w_test, t_missing_test
    ):
        simulations[(t_mu, t_cov, t_w, t_missing)] = (
            mixture_testing_util.simulate_moppca(
                t_mu=t_mu, t_cov=t_cov, t_w=t_w, t_missing=t_missing, K=1
            )
        )
    return simulations


@pytest.mark.parametrize("w_init", ["random", "svd"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("t_mu", t_mu_test)
@pytest.mark.parametrize("t_cov", t_cov_test)
@pytest.mark.parametrize("t_w", t_w_test)
@pytest.mark.parametrize("t_missing", t_missing_test)
@pytest.mark.parametrize("pcount_ard", [(0, False), (5, False), (5, True)])
def test_ppca(
    ppca_simulations, w_init, normalize, t_mu, t_cov, t_w, t_missing, pcount_ard
):
    prior_pseudocount, laplace_ard = pcount_ard
    print(f"{t_mu=} {t_cov=} {t_w=} {t_missing=}")
    res = mixture_testing_util.test_ppca(
        t_mu=t_mu,
        t_cov=t_cov,
        t_w=t_w,
        t_missing=t_missing,
        em_converged_atol=min(mu_atol, wtw_rtol) / 4,
        em_iter=512,
        figsize=(3, 2.5),
        make_vis=False,
        show_vis=False,
        normalize=normalize,
        W_initialization=w_init,
        prior_pseudocount=prior_pseudocount,
        laplace_ard=laplace_ard,
        sim_res=ppca_simulations[t_mu, t_cov, t_w, t_missing],
    )

    mumse = np.square(res["muerr"]).mean()
    assert mumse < mu_atol
    if "Werr" in res and t_w != "zero":
        W0 = res["sim_res"]["W"]
        W = res["ppca_res"]["W"]
        rank, nc, M = W.shape
        W = W.reshape(-1, M)
        WTW = W @ W.T
        mss = np.square(WTW.numpy(force=True)).mean()
        mse = np.square(res["Werr"]).mean()
        w_rel_err = mse / mss
        assert w_rel_err < wtw_rtol
