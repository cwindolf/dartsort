from itertools import product
import numpy as np
import pytest
import torch

from dartsort.cluster import truncated_mixture
from dartsort.util.sparse_util import integers_without_inner_replacement
from dartsort.util.testing_util import mixture_testing_util

mu_atol = 0.05
wtw_rtol = 0.4
elbo_atol = 5e-3


test_t_mu = ("random",)
test_t_cov = ("eye", "random")
test_t_w = ("zero", "random")
test_t_missing = (None, "random", "by_cluster")

# test_t_mu = ("random",)
# test_t_cov = ("eye",)
# test_t_w = ("zero",)
# test_t_w = ("random",)
# test_t_missing = ("random",)
# test_t_missing = ("by_cluster",)


@pytest.fixture(scope="module")
def moppca_simulations():
    simulations = {}
    for t_mu, t_cov, t_w, t_missing in product(
        test_t_mu, test_t_cov, test_t_w, test_t_missing
    ):
        simulations[(t_mu, t_cov, t_w, t_missing)] = (
            mixture_testing_util.simulate_moppca(
                t_mu=t_mu, t_cov=t_cov, t_w=t_w, t_missing=t_missing
            )
        )
    return simulations


@pytest.mark.parametrize("inference_algorithm", ["em", "tvi", "tvi_nlp"])
@pytest.mark.parametrize("n_refinement_iters", [0])
@pytest.mark.parametrize("t_mu", test_t_mu)
@pytest.mark.parametrize("t_cov_zrad", [("eye", None), ("eye", 2.0), ("random", None)])
@pytest.mark.parametrize("t_w", test_t_w)
@pytest.mark.parametrize("t_missing", test_t_missing)
@pytest.mark.parametrize(
    "pcount_ard_psm",
    [(0, False, False), (0, False, True), (5, False, True), (5, True, False)],
)
@pytest.mark.parametrize("dist_and_search_type", ["kl", "cos"])
def test_mixture(
    moppca_simulations,
    inference_algorithm,
    n_refinement_iters,
    t_mu,
    t_cov_zrad,
    t_w,
    t_missing,
    pcount_ard_psm,
    dist_and_search_type,
):
    t_cov, zrad = t_cov_zrad
    prior_pseudocount, laplace_ard, prior_scales_mean = pcount_ard_psm

    if dist_and_search_type == "kl":
        dist_search_kw = dict(distance_metric="kl", search_type="topk")
    elif dist_and_search_type == "cos":
        dist_search_kw = dict(
            distance_metric="cosine",
            search_type="random",
            merge_distance_threshold=0.5,
            distance_normalization_kind="none",
        )
    else:
        assert False

    use_nlp = inference_algorithm.endswith("_nlp")
    inference_algorithm = inference_algorithm.removesuffix("_nlp")
    is_truncated = inference_algorithm == "tvi"

    kw = dict(
        t_mu=t_mu,
        t_cov=t_cov,
        t_w=t_w,
        t_missing=t_missing,
        inference_algorithm=inference_algorithm,
        n_refinement_iters=n_refinement_iters,
        gmm_kw=dict(
            laplace_ard=laplace_ard,
            prior_pseudocount=prior_pseudocount,
            prior_scales_mean=prior_scales_mean,
            tvi_n_random_search_iter=0,
            **dist_search_kw,
        ),
        sim_res=moppca_simulations[(t_mu, t_cov, t_w, t_missing)],
        zero_radius=zrad,
        use_nlp=use_nlp,
    )
    # res_no_fit = mixture_testing_util.test_moppcas(**kw, return_before_fit=True)
    res = mixture_testing_util.test_moppcas(**kw, return_before_fit=False)

    # -- test that channel neighborhoods are handled correctly in stable feats
    data = res["gmm"].data
    train_ixs, train_extract_neighbs = data.neighborhoods(
        neighborhood="extract", split="train"
    )
    train_ixs_, train_core_neighbs = data.neighborhoods(
        neighborhood="core", split="train"
    )
    full_ixs, full_core_neighbs = data.neighborhoods(neighborhood="core", split="full")
    # internal consistency
    assert torch.equal(train_ixs, train_ixs_)
    assert torch.equal(
        train_extract_neighbs.neighborhoods, train_core_neighbs.neighborhoods
    )
    assert torch.equal(
        train_extract_neighbs.neighborhood_ids, train_core_neighbs.neighborhood_ids
    )
    assert full_ixs == slice(None)
    assert torch.equal(
        full_core_neighbs.neighborhoods, train_core_neighbs.neighborhoods
    )
    assert torch.equal(
        full_core_neighbs.neighborhood_ids[train_ixs],
        train_core_neighbs.neighborhood_ids,
    )
    cchans_ = train_core_neighbs.neighborhoods[full_core_neighbs.neighborhood_ids]
    assert torch.equal(data.core_channels, cchans_.cpu())
    # external
    assert torch.equal(data.core_channels, res["sim_res"]["channels"])

    if is_truncated:
        # test that channel neighborhoods are handled correctly in TMM
        proc = res["gmm"].tmm.processor
        neighbs = train_extract_neighbs
        nhoods = neighbs.neighborhoods
        assert nhoods.dtype == torch.long
        assert torch.equal(proc.obs_ix, nhoods)
        assert proc.n_neighborhoods == len(nhoods)
        nc = data.n_channels
        neighb_nc = nhoods.shape[1]
        rank = data.rank
        for j in range(proc.n_neighborhoods):
            vmask = neighbs.valid_mask(j).numpy(force=True)
            imask = np.setdiff1d(np.arange(neighb_nc), vmask)
            assert (nhoods[j][vmask] < nc).all()
            assert (nhoods[j][imask] == nc).all()
            obs_row = proc.obs_ix[j].numpy(force=True)
            miss_row = proc.miss_ix[j].numpy(force=True)
            (miss_nc,) = miss_row.shape
            miss_vmask = miss_row < nc
            miss_imask = np.logical_not(miss_vmask)
            assert np.intersect1d(miss_row[miss_vmask], obs_row[vmask]).size == 0

            for joobuf in (proc.Coo_inv, proc.Coo_invsqrt):
                assert (joobuf[..., imask, :] == 0).all()
                assert (joobuf[..., :, imask] == 0).all()

            if not miss_nc:
                continue
            for jjmbuf in (proc.Cooinv_Com, proc.Cmo_Cooinv_x):
                assert jjmbuf.shape[-1] == miss_nc * rank
                jjmbuf = jjmbuf.view(-1, rank, miss_nc)
                if t_cov == "eye":
                    assert (jjmbuf[..., miss_vmask] == 0).all()
                else:
                    assert (jjmbuf[..., miss_vmask] != 0).any((1, 2)).all()
                assert (jjmbuf[..., miss_imask] == 0).all()
                assert (jjmbuf[..., miss_imask] == 0).all()

        # test that lookup tables are handled correctly in TMM
        train_labels = torch.asarray(res["sim_res"]["init_sorting"].labels)[train_ixs]
        dense_init = integers_without_inner_replacement(
            np.random.default_rng(0),
            high=res["sim_res"]["K"],
            size=(*train_labels.shape, res["gmm"].tmm.n_candidates),
        )
        assert np.array_equal(np.unique(dense_init), np.arange(res["sim_res"]["K"]))
        dense_init[:, 0] = train_labels.numpy(force=True)
        for initializer in (train_labels, dense_init):
            initializer = torch.asarray(initializer)
            print(f"{res['gmm'].data.device=}")
            tmm = truncated_mixture.SpikeTruncatedMixtureModel(
                data=res["gmm"].data,
                noise=res["gmm"].noise,
                M=res["gmm"].ppca_rank * (res["gmm"].cov_kind == "ppca"),
                alpha0=res["gmm"].prior_pseudocount,
                laplace_ard=res["gmm"].laplace_ard,
                prior_scales_mean=res["gmm"].prior_scales_mean,
                neighborhood_adjacency_overlap=0.5,
                search_neighborhood_steps=1,
                n_random_search_iter=0,
            )
            tmm.set_sizes(res["sim_res"]["K"])

            div = None
            if dist_and_search_type == "kl":
                # artificial kl for testing
                div = torch.arange(res["sim_res"]["K"])
                div = div[:, None] - div[None, :]
                div = div.abs_().to(res["gmm"].data.device)

            tmm.set_parameters(
                labels=initializer,
                means=res["sim_res"]["mu"],
                bases=torch.asarray(res["sim_res"]["W"]).permute(0, 3, 1, 2),
                log_proportions=-torch.log(
                    torch.ones(res["sim_res"]["K"]) * res["sim_res"]["K"]
                ),
                noise_log_prop=torch.tensor(-50.0),
                divergences=div,
            )

            assert tmm.candidates._initialized
            assert torch.equal(
                initializer.view(len(train_labels), -1)[:, 0],
                torch.asarray(train_labels),
            )
            assert torch.equal(
                tmm.candidates.candidates[:, 0], torch.asarray(train_labels)
            )

            if dist_and_search_type == "kl":
                assert torch.equal(
                    tmm.candidates.candidates[
                        :, 1 : tmm.candidates.n_candidates
                    ].unique(),
                    torch.arange(res["sim_res"]["K"]),
                )
            assert (
                tmm.candidates.candidates[:, : tmm.candidates.n_candidates]
                .sort(dim=1)
                .values.diff(dim=1)
                > 0
            ).all()

            search_neighbors = tmm.candidates.search_sets(tmm.divergences, constrain_searches=False)
            candidates, unit_neighborhood_counts = tmm.candidates.propose_candidates(
                tmm.divergences
            )
            assert unit_neighborhood_counts.shape == (
                res["sim_res"]["K"],
                len(neighbs.neighborhoods),
            )
            assert torch.equal(candidates[:, 0], torch.asarray(train_labels))
            if dist_and_search_type == "kl":
                assert torch.equal(
                    candidates[:, 1 : tmm.candidates.n_candidates].unique(),
                    torch.arange(res["sim_res"]["K"]),
                )
            assert (
                candidates[:, : tmm.candidates.n_candidates]
                .sort(dim=1)
                .values.diff(dim=1)
                > 0
            ).all()
            assert (
                candidates.untyped_storage()
                == tmm.candidates.candidates.untyped_storage()
            )

            tmm.processor.update(
                tmm.log_proportions,
                tmm.means,
                tmm.noise_log_prop,
                tmm.bases,
                unit_neighborhood_counts=unit_neighborhood_counts,
            )

            counts2 = np.zeros_like(unit_neighborhood_counts)
            neighbs_bc = neighbs.neighborhood_ids[:, None].broadcast_to(
                candidates.shape
            )

            np.add.at(
                counts2, (candidates[candidates >= 0], neighbs_bc[candidates >= 0]), 1
            )

            lut_units = tmm.processor.lut_units.numpy(force=True)
            lut_neighbs = tmm.processor.lut_neighbs.numpy(force=True)
            uuu, nnn = unit_neighborhood_counts.nonzero()
            assert np.array_equal(lut_units, uuu)
            assert np.array_equal(lut_neighbs, nnn)
            masks = {
                uu: np.flatnonzero((candidates == uu).any(dim=1))
                for uu in range(res["sim_res"]["K"])
            }
            for uu, nn in zip(lut_units, lut_neighbs):
                assert 0 <= uu < res["sim_res"]["K"]
                assert 0 <= nn < len(neighbs.neighborhoods)
                inu = masks[int(uu.item())]
                assert (neighbs.neighborhood_ids[inu] == nn).any()

            # check parameters
            if t_cov == "eye":
                pnames = {"Cmo_Cooinv_x"}
                check = list(tmm.processor.state_dict())
                for pname in check:
                    if "om" not in pname and "mo" not in pname:
                        continue
                    if pname.startswith("_"):
                        continue
                    pnames.add(pname)
                for pname in pnames:
                    assert (getattr(tmm.processor, pname) == 0).all()

            # run the tmm and check that it doesn't do something terrible
            tmm_res = tmm.step(hard_label=True)
            u, c = tmm_res["labels"].unique(return_counts=True)
            assert torch.equal(u.cpu(), torch.arange(res["sim_res"]["K"]))
            assert ((c / c.sum()) >= 0.5 / res["sim_res"]["K"]).all()
            assert (tmm.log_proportions.exp() >= 0.5 / res["sim_res"]["K"]).all()

            tmm_elbos = []
            for j in range(10):
                rec = tmm.step()
                tmm_elbos.append(rec["obs_elbo"])
            tmm_res = tmm.step(hard_label=True)
            tmm_elbos.append(tmm_res["obs_elbo"])
            u, c = tmm_res["labels"].unique(return_counts=True)
            assert torch.equal(u.cpu(), torch.arange(res["sim_res"]["K"]))
            assert ((c / c.sum()) >= 0.5 / res["sim_res"]["K"]).all()
            assert (tmm.log_proportions.exp() >= 0.5 / res["sim_res"]["K"]).all()
            assert np.diff(tmm_elbos).min() >= -elbo_atol

            channels, counts = tmm.channel_occupancy(tmm_res["labels"], min_count=1)
            assert len(channels) == len(counts) == res["sim_res"]["K"]
            for j in range(len(channels)):
                assert np.array_equal(counts[j].nonzero(), (channels[j],))
                if t_missing == "by_unit":
                    assert torch.equal(torch.unique(counts[j]), torch.tensor([0, c[j]]))

        # test elbo decreasing
        assert len(res["fit_info"]["elbos"])
        for elbo in res["fit_info"]["elbos"]:
            assert np.diff(elbo).min() >= -elbo_atol

    sf = res["sim_res"]["data"]
    train = sf.split_indices["train"]
    corechans1 = sf.core_channels[train]

    tec = sf._train_extract_channels
    tecnc = torch.column_stack((tec, torch.full(tec.shape[:1], sf.n_channels)))
    assert (corechans1[:, :, None] == tecnc[:, None, :]).any(2).all()
    _, coretrainneighb = sf.neighborhoods()
    corechans2 = coretrainneighb.neighborhoods[coretrainneighb.neighborhood_ids]
    assert torch.equal(corechans1, corechans2.cpu())

    assert res["sim_res"]["mu"].shape == res["mm_means"].shape

    mu_err = np.square(res["muerrs"]).mean()
    print(f"{mu_err=} {res['ari']=}")
    assert mu_err < mu_atol
    assert res["ari"] == 1.0

    if t_w != "zero":
        W0 = res["sim_res"]["W"]
        W = res["W"]
        assert W0 is not None
        assert W is not None

        k, rank, nc, M = W.shape

        W = W.reshape(k, rank * nc, M)
        W0 = W0.reshape(k, rank * nc, M)

        WTW = np.einsum("nij,nkj->nik", W, W)
        WTW0 = np.einsum("nij,nkj->nik", W0, W0)
        Werr = np.abs(WTW - WTW0)

        W_rel_err_0 = np.square(Werr).mean() / np.square(WTW0).mean()
        W_rel_err_1 = np.square(Werr).mean() / np.square(WTW).mean()
        assert W_rel_err_0 < wtw_rtol
        assert W_rel_err_1 < wtw_rtol

        norm0 = np.linalg.norm(W, axis=(1, 2))
        norm1 = np.linalg.norm(W, axis=(1, 2))
        assert (np.abs(norm1 - norm0) / norm0).max() < wtw_rtol
