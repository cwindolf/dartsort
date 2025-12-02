from itertools import product
import numpy as np
import pytest
import torch
import torch.nn.functional as F


from dartsort.cluster.gmm import truncated_mixture, mixture
from dartsort.util.internal_config import RefinementConfig
from dartsort.util.logging_util import get_logger
from dartsort.util.sparse_util import integers_without_inner_replacement
from dartsort.util.testing_util import mixture_testing_util


logger = get_logger(__name__)

mu_atol = 0.05
wtw_rtol = 0.4
elbo_atol = 5e-3

TEST_RANK = 4
TMM_ELBO_ATOL = 1e-3


test_t_mu = ("smooth",)
test_t_cov = ("eye", "random")
test_t_w = ("zero", "random")
test_t_missing = (None, "random", "by_cluster")
test_corruption = (0.0, 0.2)


@pytest.fixture(scope="module")
def moppca_simulations():
    simulations = {}
    for t_mu, t_cov, t_w, t_missing, corrupt_p in product(
        test_t_mu, test_t_cov, test_t_w, test_t_missing, test_corruption
    ):
        simulations[(t_mu, t_cov, t_w, t_missing, corrupt_p)] = (
            mixture_testing_util.simulate_moppca(
                t_mu=t_mu,
                t_cov=t_cov,
                t_w=t_w,
                t_missing=t_missing,
                init_label_corruption=corrupt_p,
                rank=TEST_RANK,
            )
        )
    return simulations


@pytest.mark.parametrize("t_mu", ("smooth",))
@pytest.mark.parametrize("t_cov_zrad", [("eye", None), ("eye", 2.0), ("random", None)])
@pytest.mark.parametrize("t_w", test_t_w)
@pytest.mark.parametrize("t_missing", test_t_missing)
@pytest.mark.parametrize("corruption", test_corruption)
def test_truncated_mixture(
    moppca_simulations, t_mu, t_cov_zrad, t_w, t_missing, corruption
):
    mixture.pnoid = True

    t_cov, zrad = t_cov_zrad
    sim = moppca_simulations[(t_mu, t_cov, t_w, t_missing, corruption)]
    true_labels = sim["labels"]
    init_sorting = sim["init_sorting"]
    stable_data = sim["data"]
    noise = sim["noise"]
    K = sim["K"]
    M = sim["M"]
    cmask = sim["channel_observed_by_unit"]
    nc = sim["n_channels"]
    e_true = F.one_hot(true_labels, K)

    mu = sim["mu"].view(K, -1)
    mu_norm = torch.linalg.norm(mu, dim=1)
    gt_cosines = mu @ mu.T
    gt_cosines /= mu_norm[:, None] * mu_norm
    gt_cosines.fill_diagonal_(-torch.inf)

    rg = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ref_cfg = RefinementConfig(
        refinement_strategy="tmm",
        signal_rank=M * (t_w != "zero"),
        n_candidates=K,
    )

    # copy-pasting from tmm_demix here
    neighb_cov, erp, train_data, val_data, full_data, noise = (
        mixture.get_truncated_datasets(
            init_sorting,
            motion_est=None,
            refinement_cfg=ref_cfg,
            device=device,
            rg=rg,
            noise=noise,
            stable_data=stable_data,
        )
    )
    tmm = mixture.TruncatedMixtureModel.from_config(
        noise=noise,
        erp=erp,
        neighb_cov=neighb_cov,
        train_data=train_data,
        refinement_cfg=ref_cfg,
        seed=rg,
    )
    D = tmm.unit_distance_matrix()
    lut = train_data.bootstrap_candidates(D)
    tmm.update_lut(lut)
    eval_scores = None

    # loop: em, then split+em, then merge+em, then false-merge+split+em, false-split+merge+em.
    for it in range(5):
        if it == 3:
            # introduce a false merge, and hope to split it below
            # merge the highest-cosine pair
            ii, jj = (gt_cosines == gt_cosines.amax()).nonzero(as_tuple=True)
            ii = ii[0].item()
            jj = jj[0].item()
            ii, jj = min(ii, jj), max(ii, jj)
            assert ii != jj
            print(f"False merge: {ii} and {jj}, with cos {gt_cosines[ii, jj]}.")
            logger.info(f"False merge: {ii} and {jj}, with cos {gt_cosines[ii, jj]}.")

            # update unit ii by just averaging
            tmm.b.log_proportions[ii] = torch.logaddexp(
                tmm.b.log_proportions[ii], tmm.b.log_proportions[jj]
            )
            tmm.b.means[ii] = 0.5 * (tmm.b.means[ii] + tmm.b.means[jj])
            if tmm.signal_rank:
                tmm.b.bases[ii] = 0.5 * (tmm.b.bases[ii] + tmm.b.bases[jj])
            tmm.b.log_proportions[jj] = -torch.inf
            assert torch.isclose(
                tmm.b.log_proportions.logsumexp(dim=0), torch.zeros(()), atol=1e-6
            )

            # create a label remapping which deletes jj and shuffles everyone larger back by one
            mapping = torch.arange(K)
            mapping[jj] = ii
            mapping = mixture.UnitRemapping(mapping=mapping)
            flat_map = tmm.cleanup(mapping)

            # re-bootstrap everything
            distances = tmm.unit_distance_matrix()
            lut = train_data.remap(distances=distances, remapping=flat_map)
            tmm.update_lut(lut)
            em_res = tmm.em(train_data)

        if it == 4:
            # similarly, do two false splits here
            # pick 2 random units and just accept the kmeans proposal
            to_split = rg.choice(K, size=2, replace=False)
            n_pieces = [3, 3]
            scores = tmm.soft_assign(full_data, needs_bootstrap=True)
            tmm_labels = torch.asarray(mixture.labels_from_scores(scores))
            for k in range(K):
                logger.info(
                    f"{k=} {tmm_labels[true_labels==k].unique(return_counts=True)=}"
                )
            assert eval_scores is not None
            print(f"False split: {to_split} into {n_pieces} parts.")
            logger.info(f"False split: {to_split} into {n_pieces} parts.")
            split_results = []
            for unit_id, kmeansk in zip(to_split, n_pieces):
                split_data = train_data.dense_slice_by_unit(unit_id)
                assert split_data is not None
                kmeans_responsibliities = mixture.try_kmeans(
                    split_data,
                    k=kmeansk,
                    erp=tmm.erp,
                    gen=tmm.rg,
                    feature_rank=tmm.noise.rank,
                    min_count=tmm.min_count,
                )
                assert kmeans_responsibliities is not None
                split_model, _ = (
                    mixture.TruncatedMixtureModel.initialize_from_dense_data_with_fixed_responsibilities(
                        data=split_data,
                        responsibilities=kmeans_responsibliities,
                        signal_rank=tmm.signal_rank,
                        erp=tmm.erp,
                        min_count=tmm.min_count,
                        min_channel_count=tmm.min_channel_count,
                        noise=tmm.noise,
                        max_group_size=tmm.max_group_size,
                        max_distance=tmm.max_distance,
                        neighb_cov=tmm.neighb_cov,
                        min_iter=tmm.criterion_em_iters,
                        max_iter=tmm.em_iters,
                        elbo_atol=tmm.elbo_atol,
                        prior_pseudocount=tmm.prior_pseudocount,
                        cl_alpha=tmm.cl_alpha,
                        total_log_proportion=tmm.b.log_proportions[unit_id].item(),
                    )
                )
                assert split_model.n_units > 1
                assert split_model.n_units == kmeans_responsibliities.shape[1]

                split_result = mixture.SuccessfulUnitSplitResult(
                    unit_id=unit_id,
                    n_split=split_model.n_units,
                    train_indices=split_data.indices,
                    train_assignments=kmeans_responsibliities.argmax(dim=1),
                    means=split_model.b.means,
                    sub_proportions=kmeans_responsibliities.mean(0),
                    bases=split_model.b.bases if tmm.signal_rank else None,
                )
                split_results.append(split_result)

        if it:
            # get scores that split needs
            if val_data is not None:
                eval_scores = tmm.soft_assign(val_data, needs_bootstrap=True)
                true_eval_labels = true_labels[stable_data.split_indices["val"]]
            else:
                eval_scores = tmm.soft_assign(
                    train_data, needs_bootstrap=False, max_iter=1
                )
                true_eval_labels = true_labels[stable_data.split_indices["train"]]
            eval_labels = torch.asarray(mixture.labels_from_scores(eval_scores))
            logger.info(f"At {it=}, label breakdown before split or merge is:")
            for k in range(K):
                logger.info(
                    f"{k=} {eval_labels[true_eval_labels==k].unique(return_counts=True)=}"
                )
        else:
            eval_scores = None

        if it in (1, 3):
            print("Split.")
            logger.info("Split.")
            assert eval_scores is not None
            split_res = tmm.split(train_data, val_data, scores=eval_scores)
            logger.info(f"Split created {split_res.n_new_units} new units.")
            if it == 1:
                assert split_res.n_new_units == 0

        if it in (2, 4):
            print("Merge.")
            logger.info("Merge.")
            assert eval_scores is not None
            merge_map = tmm.merge(train_data, val_data, scores=eval_scores)
            logger.info(f"{gt_cosines=}.")
            logger.info(f"{true_labels.unique(return_counts=True)=}.")
            logger.info(
                f"Merge {merge_map.mapping.shape[0]} -> {merge_map.nuniq()} units."
            )
            assert merge_map.nuniq() == K

        # the false merges and splits can lead to label permutations. let's quickly fix those here.
        if it in (3, 4):
            em_res = tmm.em(train_data)
            assert tmm.n_units == K
            assert tmm.unit_ids.shape[0] == K
            assert np.diff(em_res.elbos).min(initial=0.0) >= -TMM_ELBO_ATOL

            scores = tmm.soft_assign(full_data, needs_bootstrap=True)
            tmm_labels = scores.candidates[:, 0]
            e_tmm = F.one_hot(tmm_labels, K)
            agreement = (e_true[:, :, None] == e_tmm[:, None, :]).float().mean(0)
            max_agreements, matches = agreement.max(dim=1)
            assert torch.allclose(max_agreements, torch.ones_like(max_agreements))
            print(f"Correcting permutation {matches} (agreements: {max_agreements}).")
            logger.info(
                f"Correcting permutation {matches} (agreements: {max_agreements})."
            )

            perm = mixture.UnitRemapping(mapping=torch.argsort(matches))
            flat_map = tmm.cleanup(perm)

            # re-bootstrap everything
            distances = tmm.unit_distance_matrix()
            lut = train_data.remap(distances=distances, remapping=flat_map)
            tmm.update_lut(lut)

        em_res = tmm.em(train_data)

        # check count
        assert tmm.n_units == K
        assert tmm.unit_ids.shape[0] == K

        # check elbos
        assert np.diff(em_res.elbos).min(initial=0.0) >= -TMM_ELBO_ATOL

        # check labels
        scores = tmm.soft_assign(full_data, needs_bootstrap=True)
        tmm_labels = scores.candidates[:, 0]
        tmm_labels_noise = torch.asarray(mixture.labels_from_scores(scores))
        for lpred in (tmm_labels, tmm_labels_noise):
            acc = (true_labels == lpred).double().mean().item()
            assert acc >= 0.995
            assert lpred.max() + 1 == K

        # check parameters
        # if there is missingness by cluster, lets ignore the always-unobserved stuff.
        _, c = true_labels.unique(return_counts=True)
        standard_error = 1.0 / c.sqrt()
        # uhm. what would the bonferroni be?
        z = 10 * (1 + 5 * (t_w != "zero") + 5 * (corruption != 0.0))
        diff = tmm.b.means.view(K, -1) - mu.view(K, -1)
        if cmask is not None:
            cmask = torch.asarray(cmask).to(diff)
            diff.view(K, -1, nc).mul_(cmask[:, None])
        assert torch.all(diff.abs().amax(dim=1) <= z * standard_error)
        if t_w != "zero":
            w0 = sim["W"].permute(0, 3, 1, 2)
            w_ = tmm.b.bases
            assert w0.shape[:2] == (K, M)
            assert w_.shape[:2] == (K, M)
            w0 = w0.view(K, M, -1)
            w_ = w_.view(K, M, -1)
            assert w0.shape == w_.shape
            wtw0 = w0.mT.bmm(w0)
            wtw_ = w_.mT.bmm(w_)
            diff = wtw0 - wtw_
            if cmask is not None:
                diff = diff.view(K, TEST_RANK, nc, TEST_RANK, nc)
                wcmask = cmask[:, None, :, None, None] * cmask[:, None, None, None, :]
                diff.mul_(wcmask)
            assert torch.all(
                diff.abs().view(K, -1).amax(dim=1) <= 5 * z * standard_error
            )


@pytest.mark.parametrize("inference_algorithm", ["em", "tvi"])  # , "tvi_nlp"])
@pytest.mark.parametrize("n_refinement_iters", [0])
@pytest.mark.parametrize("t_mu", test_t_mu)
@pytest.mark.parametrize("t_cov_zrad", [("eye", None), ("eye", 2.0), ("random", None)])
@pytest.mark.parametrize("t_w", test_t_w)
@pytest.mark.parametrize("t_missing", test_t_missing)
@pytest.mark.parametrize(
    "pcount_ard_psm",
    # [(0, False, False), (0, False, True), (5, False, True), (5, True, False)],
    [(0, False, False), (5, False, True)],
)
@pytest.mark.parametrize("dist_and_search_type", ["kl", "cos"])
def test_original_mixture(
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
            search_type="topk",
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

            if dist_and_search_type == "kl" and t_missing in (None, "random"):
                assert torch.equal(
                    tmm.candidates.candidates[
                        :, 1 : tmm.candidates.n_candidates
                    ].unique(),
                    torch.arange(res["sim_res"]["K"]),
                )
            scand = (
                tmm.candidates.candidates[:, : tmm.candidates.n_candidates]
                .sort(dim=1)
                .values
            )
            assert torch.logical_or(scand.diff(dim=1) > 0, scand[:, 1:] < 0).all()

            search_neighbors = tmm.candidates.search_sets(
                tmm.divergences, constrain_searches=False
            )
            candidates, unit_neighborhood_counts = tmm.candidates.propose_candidates(
                tmm.divergences
            )
            assert unit_neighborhood_counts.shape == (
                res["sim_res"]["K"],
                len(neighbs.neighborhoods),
            )
            assert torch.equal(candidates[:, 0], torch.asarray(train_labels))
            if dist_and_search_type == "kl" and t_missing in (None, "random"):
                assert torch.equal(
                    candidates[:, 1 : tmm.candidates.n_candidates].unique(),
                    torch.arange(res["sim_res"]["K"]),
                )
            scand = candidates[:, : tmm.candidates.n_candidates].sort(dim=1).values
            assert torch.logical_or(scand.diff(dim=1) > 0, scand[:, 1:] < 0).all()
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
