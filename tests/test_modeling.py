import numpy as np
import torch

from dartsort.util.testing_util import mixture_testing_util

mu_atol = 0.05
wtw_rtol = 0.05

t_missing_test = (None, "random")
t_mu_test = ("zero", "random")
t_cov_test = ("eye", "random")
t_w_test = ("zero", "hot", "random")
t_channels_strategy_test = ("count", "count_core")


def test_ppca():
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
                            em_converged_atol=1e-1,
                            em_iter=100,
                            figsize=(3, 2.5),
                            make_vis=False,
                            show_vis=False,
                            normalize=False,
                            cache_local=t_channels_strategy.endswith("core"),
                        )

                        mumse = np.square(res["muerr"]).mean()
                        mugood = mumse < mu_atol
                        assert mugood
                        Wgood = True
                        wmse = 0
                        if "W" in res:
                            W = res["W"]
                            W = rank, nc, M = W.shape
                            WTW = W.reshape(rank * nc, M)
                            wmse = np.square(res["Werr"]).mean()
                            Wgood = wmse / np.square(WTW).mean() < wtw_rtol
                        assert Wgood

                        # print(f"{mumse=} {wmse=}")
                        # if not (mugood and Wgood):
                        #     print(f"{mugood=} {Wgood=}")
                        #     plt.show()
                        #     plt.close(res['panel'])
                        #     assert False
                        # plt.close(res['panel'])


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
                        assert torch.vmap(torch.isin)(
                            corechans1, sf._train_extract_channels
                        ).all()
                        _, coretrainneighb = sf.neighborhoods()
                        corechans2 = coretrainneighb.neighborhoods[
                            coretrainneighb.neighborhood_ids
                        ]
                        assert torch.equal(corechans1, corechans2)

                        mugood = np.square(res["muerrs"]).mean() < mu_atol
                        assert mugood
                        agood = res["acc"] >= 1.0
                        assert agood

                        Wgood = True
                        if "W" in res:
                            W = res["W"]
                            k, rank, nc, M = W.shape
                            mss = 0.0
                            for ww in W:
                                WTW = W.reshape(rank * nc, M)
                                mss = max(mss, np.square(WTW).mean())
                            Wgood = np.square(res["Werrs"]).mean() / mss < wtw_rtol
                        assert Wgood
                        # if not (mugood and Wgood and agood):
                        #     print(f"rerun. {mugood=} {Wgood=} {agood=}")
                        #     mixture_testing_util.test_moppcas(**kw)

                        #     assert False
