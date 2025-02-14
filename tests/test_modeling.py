import numpy as np
import torch

from dartsort.util.testing_util import mixture_testing_util

mu_atol = 0.05
wtw_rtol = 0.01

t_missing_test = (None, "random")
t_mu_test = ("zero", "random")
t_cov_test = ("eye", "random")
t_w_test = ("zero", "hot", "random")
t_channels_strategy_test = ("count", "count_core")


def _test_ppca(w_init="random", normalize=False):
    for t_channels_strategy in t_channels_strategy_test:
        for t_missing in t_missing_test:
            for t_mu in t_mu_test:
                for t_cov in t_cov_test:
                    for t_w in t_w_test:
                        print(f"{t_mu=} {t_cov=} {t_w=} {t_missing=}")
                        res = mixture_testing_util.test_ppca(
                            t_mu=t_mu,
                            t_cov=t_cov,
                            t_w=t_w,
                            t_missing=t_missing,
                            em_converged_atol=1e-3,
                            em_iter=1000,
                            figsize=(3, 2.5),
                            make_vis=False,
                            show_vis=False,
                            normalize=normalize,
                            W_initialization=w_init,
                            cache_local=t_channels_strategy.endswith("core"),
                        )

                        mumse = np.square(res["muerr"]).mean()
                        mugood = mumse < mu_atol
                        assert mugood
                        Wgood = True
                        wmse = 0
                        if "Werr" in res and t_w != "zero":
                            W0 = res["sim_res"]["W"]
                            W = res["ppca_res"]["W"]
                            rank, nc, M = W.shape
                            W = W.reshape(-1, M)
                            WTW = W @ W.T
                            mss = np.square(WTW.numpy(force=True)).mean()
                            mse = np.square(res["Werr"]).mean()
                            Wgood = mse / mss < wtw_rtol
                        assert Wgood

                        # print(f"{mumse=} {wmse=}")
                        # if not (mugood and Wgood):
                        #     print(f"{mugood=} {Wgood=}")
                        #     plt.show()
                        #     plt.close(res['panel'])
                        #     assert False
                        # plt.close(res['panel'])


def test_ppca():
    _test_ppca("random", False)
    _test_ppca("svd", False)
    _test_ppca("random", True)
    _test_ppca("svd", True)


def test_mixture():
    for t_mu in ("random",):
        for t_cov in t_cov_test:
            for t_w in t_w_test:
                # for t_w in ("zero", "random"):
                for t_missing in t_missing_test:
                    for t_channels_strategy in t_channels_strategy_test:
                        print(f"{t_mu=} {t_cov=} {t_w=} {t_missing=}")
                        kw = dict(
                            t_mu=t_mu,
                            t_cov=t_cov,
                            t_w=t_w,
                            t_missing=t_missing,
                            em_converged_atol=1e-3,
                            inner_em_iter=100,
                            figsize=(3, 2.5),
                            make_vis=False,
                            with_noise_unit=False,
                            channels_strategy=t_channels_strategy,
                            snr=10.0,
                        )
                        res = mixture_testing_util.test_moppcas(
                            **kw, return_before_fit=False
                        )

                        sf = res["sim_res"]["data"]
                        train = sf.split_indices["train"]
                        corechans1 = sf.core_channels[train]

                        tec = sf._train_extract_channels
                        tecnc = torch.column_stack(
                            (tec, torch.full(tec.shape[:1], sf.n_channels))
                        )
                        assert (
                            (corechans1[:, :, None] == tecnc[:, None, :]).any(2).all()
                        )
                        _, coretrainneighb = sf.neighborhoods()
                        corechans2 = coretrainneighb.neighborhoods[
                            coretrainneighb.neighborhood_ids
                        ]
                        assert torch.equal(corechans1, corechans2)

                        mugood = np.square(res["muerrs"]).mean() < mu_atol
                        assert mugood
                        agood = res["acc"] == 1.0
                        assert agood

                        Wgood = True
                        if "W" in res:
                            W = res["W"]
                            k, rank, nc, M = W.shape
                            mss = np.square(WTW).mean()
                            mse = np.square(res["Werrs"]).mean()
                            print(f"{mse=} {mss=}")
                            Wgood = mse / mss < wtw_rtol
                        assert Wgood
                        # if not (mugood and Wgood and agood):
                        #     print(f"rerun. {mugood=} {Wgood=} {agood=}")
                        #     mixture_testing_util.test_moppcas(**kw)

                        #     assert False


if __name__ == "__main__":
    test_ppca()
    test_mixture()
