from itertools import product
import numpy as np
import pytest
import torch
import torch.nn.functional as F


from dartsort.cluster.gmm import (
    truncated_mixture,
    mixture,
    stable_features,
    gaussian_mixture,
)
from dartsort.util.internal_config import RefinementConfig
from dartsort.util.logging_util import get_logger
from dartsort.util.sparse_util import integers_without_inner_replacement
from dartsort.util.spiketorch import spawn_torch_rg
from dartsort.util.testing_util import mixture_testing_util


logger = get_logger(__name__)

mu_atol = 0.1
wtw_rtol = 1.0
elbo_atol = 5e-3

TEST_RANK = 4
TMM_ELBO_ATOL = 1e-3


test_t_mu = ("smooth",)
test_t_cov = ("eye", "random")
test_t_w = ("zero", "random", "smooth")
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
            sorting=init_sorting,
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
            assert eval_scores is not None
            print(f"False split: {to_split} into {n_pieces} parts.")
            logger.info(f"False split: {to_split} into {n_pieces} parts.")
            split_results = []
            for unit_id, kmeansk in zip(to_split, n_pieces):
                split_data = train_data.dense_slice_by_unit(unit_id, gen=tmm.rg)
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
                split_model, _, _, any_discarded, _, _ = (
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
                assert not any_discarded
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
                eval_scores = tmm.soft_assign(
                    data=val_data, full_proposal_view=True, needs_bootstrap=False
                )
            else:
                eval_scores = tmm.soft_assign(
                    data=train_data,
                    full_proposal_view=False,
                    needs_bootstrap=False,
                    max_iter=1,
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

            scores = tmm.soft_assign(
                data=full_data, full_proposal_view=True, needs_bootstrap=False
            )
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
        scores = tmm.soft_assign(
            data=full_data, full_proposal_view=True, needs_bootstrap=False
        )
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
        # with missing channels, non-smooth stuff is hard.
        # by hard, I mean that a low atol is required.
        z = 10 * (
            1
            + 5 * (t_w != "zero")
            + 5 * (corruption != 0.0)
            + 5 * (t_w not in ("zero", "smooth"))
        )
        diff = tmm.b.means.view(K, -1).cpu() - mu.view(K, -1)
        if cmask is not None:
            cmask = torch.asarray(cmask).to(diff)
            diff.view(K, -1, nc).mul_(cmask[:, None])
        assert torch.all(diff.abs().amax(dim=1) <= z * standard_error)

        if t_w == "zero":
            return

        # hard to estimate arbitrary w with missing channels!
        zw = z * (5 + 10 * (t_w != "smooth") + 5 * (corruption != 0.0))
        w0 = sim["W"].permute(0, 3, 1, 2)
        w_ = tmm.b.bases
        assert w0.shape[:2] == (K, M)
        assert w_.shape[:2] == (K, M)
        w0 = w0.view(K, M, -1)
        w_ = w_.view(K, M, -1)
        assert w0.shape == w_.shape
        wtw0 = w0.mT.bmm(w0)
        wtw_ = w_.mT.bmm(w_)
        diff = wtw0 - wtw_.cpu()
        if cmask is not None:
            diff = diff.view(K, TEST_RANK, nc, TEST_RANK, nc)
            wcmask = cmask[:, None, :, None, None] * cmask[:, None, None, None, :]
            diff.mul_(wcmask)
        assert torch.all(diff.abs().view(K, -1).amax(dim=1) <= zw * standard_error)


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
        sim_res=moppca_simulations[(t_mu, t_cov, t_w, t_missing, 0.0)],
        zero_radius=zrad,
        use_nlp=use_nlp,
    )
    # res_no_fit = mixture_testing_util.test_moppcas(**kw, return_before_fit=True)
    res = mixture_testing_util.test_moppcas(**kw, return_before_fit=False)  # type: ignore
    K = res["sim_res"]["K"]  # type: ignore
    assert isinstance(K, int)

    # -- test that channel neighborhoods are handled correctly in stable feats
    data = res["gmm"].data  # type: ignore
    assert isinstance(data, stable_features.StableSpikeDataset)
    train_ixs, train_extract_neighbs = data.neighborhoods(
        neighborhood="extract", split="train"
    )
    train_ixs_, train_core_neighbs = data.neighborhoods(
        neighborhood="core", split="train"
    )
    full_ixs, full_core_neighbs = data.neighborhoods(neighborhood="core", split="full")
    # internal consistency
    if torch.is_tensor(train_ixs):
        assert torch.is_tensor(train_ixs_)
        assert torch.equal(train_ixs, train_ixs_)
    else:
        assert train_ixs == train_ixs_
    assert torch.equal(
        train_extract_neighbs.b.neighborhoods, train_core_neighbs.b.neighborhoods
    )
    assert torch.equal(
        train_extract_neighbs.b.neighborhood_ids, train_core_neighbs.b.neighborhood_ids
    )
    assert full_ixs == slice(None)
    assert torch.equal(
        full_core_neighbs.b.neighborhoods, train_core_neighbs.b.neighborhoods
    )
    assert torch.equal(
        full_core_neighbs.b.neighborhood_ids[train_ixs],
        train_core_neighbs.b.neighborhood_ids,
    )
    cchans_ = train_core_neighbs.b.neighborhoods[full_core_neighbs.b.neighborhood_ids]
    assert torch.equal(data.core_channels, cchans_.cpu())
    # external
    assert torch.equal(data.core_channels, res["sim_res"]["channels"])  # type: ignore

    if is_truncated:
        # test that channel neighborhoods are handled correctly in TMM
        proc = res["gmm"].tmm.processor  # type: ignore
        assert isinstance(proc, truncated_mixture.TruncatedExpectationProcessor)
        neighbs = train_extract_neighbs
        nhoods = neighbs.b.neighborhoods
        assert nhoods.dtype == torch.long
        assert torch.equal(proc.obs_ix, nhoods)  # type: ignore
        assert proc.n_neighborhoods == len(nhoods)
        nc = data.n_channels
        neighb_nc = nhoods.shape[1]
        rank = data.rank
        for j in range(proc.n_neighborhoods):  # type: ignore
            vmask = neighbs.valid_mask(j).numpy(force=True)
            imask = np.setdiff1d(np.arange(neighb_nc), vmask)
            assert (nhoods[j][vmask] < nc).all()
            assert (nhoods[j][imask] == nc).all()
            obs_row = proc.obs_ix[j].numpy(force=True)  # type: ignore
            miss_row = proc.miss_ix[j].numpy(force=True)  # type: ignore
            (miss_nc,) = miss_row.shape
            miss_vmask = miss_row < nc
            miss_imask = np.logical_not(miss_vmask)
            assert np.intersect1d(miss_row[miss_vmask], obs_row[vmask]).size == 0

            for joobuf in (proc.Coo_inv, proc.Coo_invsqrt):
                assert (joobuf[..., imask, :] == 0).all()  # type: ignore
                assert (joobuf[..., :, imask] == 0).all()  # type: ignore

            if not miss_nc:
                continue
            for jjmbuf in (proc.Cooinv_Com, proc.Cmo_Cooinv_x):
                assert jjmbuf.shape[-1] == miss_nc * rank  # type: ignore
                jjmbuf = jjmbuf.view(-1, rank, miss_nc)  # type: ignore
                if t_cov == "eye":
                    assert (jjmbuf[..., miss_vmask] == 0).all()  # type: ignore
                else:
                    assert (jjmbuf[..., miss_vmask] != 0).any((1, 2)).all()  # type: ignore
                assert (jjmbuf[..., miss_imask] == 0).all()
                assert (jjmbuf[..., miss_imask] == 0).all()

        # test that lookup tables are handled correctly in TMM
        train_labels = torch.asarray(res["sim_res"]["init_sorting"].labels)[train_ixs]  # type: ignore
        dense_init = integers_without_inner_replacement(
            np.random.default_rng(0),
            high=K,  # type: ignore
            size=(*train_labels.shape, res["gmm"].tmm.n_candidates),  # type: ignore
        )
        assert np.array_equal(np.unique(dense_init), np.arange(K))
        dense_init[:, 0] = train_labels.numpy(force=True)
        for initializer in (train_labels, dense_init):
            initializer = torch.asarray(initializer)
            gmm = res["gmm"]  # type: ignore
            assert isinstance(gmm, gaussian_mixture.SpikeMixtureModel)
            tmm = truncated_mixture.SpikeTruncatedMixtureModel(
                data=gmm.data,  # type: ignore
                noise=gmm.noise,
                M=gmm.ppca_rank * (gmm.cov_kind == "ppca"),
                alpha0=gmm.prior_pseudocount,
                laplace_ard=gmm.laplace_ard,
                prior_scales_mean=gmm.prior_scales_mean,
            )
            tmm.set_sizes(K)

            div = None
            if dist_and_search_type == "kl":
                # artificial kl for testing
                div = torch.arange(K)
                div = div[:, None] - div[None, :]
                div = div.abs_().to(res["gmm"].data.device)  # type: ignore

            tmm.set_parameters(
                labels=initializer,
                means=res["sim_res"]["mu"],  # type: ignore
                bases=torch.asarray(res["sim_res"]["W"]).permute(0, 3, 1, 2),  # type: ignore
                log_proportions=-torch.log(torch.ones(K) * K),
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
                    torch.arange(K),
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
                K,
                len(neighbs.b.neighborhoods),
            )
            assert torch.equal(candidates[:, 0], torch.asarray(train_labels))
            if dist_and_search_type == "kl" and t_missing in (None, "random"):
                assert torch.equal(
                    candidates[:, 1 : tmm.candidates.n_candidates].unique(),
                    torch.arange(K),
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
            neighbs_bc = neighbs.b.neighborhood_ids[:, None].broadcast_to(
                candidates.shape
            )

            np.add.at(
                counts2,
                (candidates[candidates >= 0].cpu(), neighbs_bc[candidates >= 0].cpu()),
                1,
            )

            lut_units = tmm.processor.lut_units.numpy(force=True)
            lut_neighbs = tmm.processor.lut_neighbs.numpy(force=True)
            uuu, nnn = unit_neighborhood_counts.nonzero()
            assert np.array_equal(lut_units, uuu)
            assert np.array_equal(lut_neighbs, nnn)
            masks = {
                uu: np.flatnonzero((candidates == uu).any(dim=1)) for uu in range(K)
            }
            for uu, nn in zip(lut_units, lut_neighbs):
                assert 0 <= uu < K
                assert 0 <= nn < len(neighbs.b.neighborhoods)
                inu = masks[int(uu.item())]
                assert (neighbs.b.neighborhood_ids[inu] == nn).any()

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
            ll = torch.asarray(tmm_res["labels"])
            u, c = ll.unique(return_counts=True)
            assert torch.equal(u.cpu(), torch.arange(K))
            assert ((c / c.sum()) >= 0.5 / K).all()
            assert (tmm.log_proportions.exp() >= 0.5 / K).all()

            tmm_elbos = []
            for j in range(10):
                rec = tmm.step()
                tmm_elbos.append(rec["obs_elbo"])
            tmm_res = tmm.step(hard_label=True)
            tmm_elbos.append(tmm_res["obs_elbo"])
            u, c = tmm_res["labels"].unique(return_counts=True)  # type: ignore
            assert torch.equal(u.cpu(), torch.arange(K))
            assert ((c / c.sum()) >= 0.5 / K).all()
            assert (tmm.log_proportions.exp() >= 0.5 / K).all()
            assert np.diff(tmm_elbos).min() >= -elbo_atol

            channels, counts = tmm.channel_occupancy(tmm_res["labels"], min_count=1)
            assert len(channels) == len(counts) == K
            for j in range(len(channels)):
                assert np.array_equal(counts[j].nonzero(), (channels[j],))
                if t_missing == "by_unit":
                    assert torch.equal(torch.unique(counts[j]), torch.tensor([0, c[j]]))

        # test elbo decreasing
        assert len(res["fit_info"]["elbos"])  # type: ignore
        for elbo in res["fit_info"]["elbos"]:  # type: ignore
            assert np.diff(elbo).min() >= -elbo_atol

    sf = res["sim_res"]["data"]  # type: ignore
    assert isinstance(sf, stable_features.StableSpikeDataset)
    train = sf.split_indices["train"]
    corechans1 = sf.core_channels[train]

    tec = sf._train_extract_channels
    tecnc = torch.column_stack((tec, torch.full(tec.shape[:1], sf.n_channels)))
    assert (corechans1[:, :, None] == tecnc[:, None, :]).any(2).all()
    _, coretrainneighb = sf.neighborhoods()
    corechans2 = coretrainneighb.b.neighborhoods[coretrainneighb.b.neighborhood_ids]
    assert torch.equal(corechans1, corechans2.cpu())

    assert res["sim_res"]["mu"].shape == res["mm_means"].shape  # type: ignore

    mu_err = np.square(res["muerrs"]).mean()  # type: ignore
    print(f"{mu_err=} {res['ari']=}")
    assert mu_err < mu_atol
    assert res["ari"] == 1.0

    if t_w != "zero":
        W0 = res["sim_res"]["W"]  # type: ignore
        W = res["W"]
        assert W0 is not None
        assert W is not None

        k, rank, nc, M = W.shape  # type: ignore

        W = W.reshape(k, rank * nc, M)  # type: ignore
        W0 = W0.reshape(k, rank * nc, M)  # type: ignore

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


@pytest.mark.parametrize("link", ("single", "complete"))
@pytest.mark.parametrize("max_distance", (0.0, 0.1, 1.0, 2.0))
@pytest.mark.parametrize("max_group_size", (1, 2, 5))
@pytest.mark.parametrize("dist_kind", ("flipeye", "randn", "zero", "real"))
@pytest.mark.parametrize("K", (1, 2, 5))
def test_tree_groups(K, dist_kind, max_group_size, max_distance, link):
    logger.debug(f"{K=} {dist_kind=} {max_group_size=} {max_distance=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rg = spawn_torch_rg(0, device=device)

    if dist_kind == "flipeye":
        dist = 1.0 - torch.eye(K, device=device)
    elif dist_kind == "randn":
        dist = torch.randn(K, 2 * K, device=device, generator=rg)
        dist = dist @ dist.T
        dist.fill_diagonal_(0.0)
    elif dist_kind == "zero":
        dist = torch.zeros((K, K), device=device)
    elif dist_kind == "real":
        dist = _real_dist().to(device)
        if K > 1:
            return
        K = dist.shape[0]
    else:
        assert False

    groups = mixture.tree_groups(
        dist, max_group_size=max_group_size, max_distance=max_distance, link=link
    )

    # check distances meet linkage criterion
    for g in groups:
        dg = dist[g][:, g]
        if link == "complete":
            assert dg.max() <= max_distance
        elif link == "single":
            if g.numel() > 1:
                dg.fill_diagonal_(max_distance + 1)
                assert dg.amin(dim=0).amax() <= max_distance
                assert dg.amin(dim=1).amax() <= max_distance

    # check coverage and no duplicates
    all_ids = torch.concatenate(groups)
    all_ids = all_ids.sort().values
    assert torch.equal(all_ids, torch.arange(K, device=all_ids.device))

    # check group sizes
    assert all(g.numel() <= max_group_size for g in groups)

    # check that there is SOME grouping happening, if possible...
    if K > 1 and max_group_size > 1:
        min_dist = dist[*torch.triu_indices(*dist.shape, offset=1)].amin()
        if min_dist <= max_distance:
            assert max(g.numel() for g in groups) > 1


# fmt: off
def _real_dist():
    x = r"[[0.0,0.5407323241233826,0.5185264348983765,1.186301589012146,1.3640413284301758,1.3625680208206177,1.3643501996994019,1.3633673191070557,0.8287628293037415,0.824342668056488,0.8216472268104553,0.8140830993652344,1.0429106950759888,1.0468953847885132,1.1978167295455933,1.1857260465621948,1.1794891357421875,1.4037282466888428,1.4039306640625,1.4040076732635498,1.4022308588027954,1.402640461921692,1.4025861024856567,1.3439334630966187,1.3997373580932617,1.4013144969940186,1.3984349966049194,1.4007179737091064,1.4141825437545776,1.414223313331604,1.4142059087753296,1.414214015007019,1.4142206907272339,1.4142134189605713,1.4142040014266968,1.4142308235168457,1.4142131805419922,1.4093965291976929,1.4107441902160645,1.4066569805145264,1.4058283567428589,1.4076207876205444,0.9987942576408386,0.9939128160476685,0.9140720963478088,0.9473844766616821,1.241729736328125,0.8480154275894165,0.8451475501060486,0.8421363234519958],[0.5407323241233826,0.0,0.046521443873643875,1.1360185146331787,1.3917930126190186,1.3912107944488525,1.3915326595306396,1.391201376914978,0.9598490595817566,0.9567139148712158,0.9534347057342529,0.9523178339004517,1.0352966785430908,1.0402659177780151,1.2147512435913086,1.204429268836975,1.1984297037124634,1.4037678241729736,1.4040100574493408,1.4041575193405151,1.399926781654358,1.4003174304962158,1.4003257751464844,1.3291643857955933,1.39676833152771,1.3984706401824951,1.3951679468154907,1.3982278108596802,1.4141989946365356,1.4142374992370605,1.414206624031067,1.4142082929611206,1.4142181873321533,1.414209246635437,1.4141873121261597,1.4142203330993652,1.414206862449646,1.412293553352356,1.4128614664077759,1.4109838008880615,1.4106923341751099,1.4114936590194702,1.163755178451538,1.1595370769500732,1.0669684410095215,1.0896323919296265,1.3093029260635376,1.046993613243103,1.0439462661743164,1.037471055984497],[0.5185264348983765,0.046522725373506546,0.0,1.1344568729400635,1.3926935195922852,1.3920644521713257,1.392437219619751,1.3920586109161377,0.9536961913108826,0.9510942697525024,0.9481849074363708,0.9475924372673035,1.0281325578689575,1.0329310894012451,1.2101887464523315,1.2001020908355713,1.194420337677002,1.4042407274246216,1.4044550657272339,1.4045145511627197,1.4002755880355835,1.4005950689315796,1.4007456302642822,1.3261522054672241,1.3971577882766724,1.398834466934204,1.3957728147506714,1.3988677263259888,1.4141855239868164,1.4142180681228638,1.4142258167266846,1.414228081703186,1.4142390489578247,1.4142041206359863,1.414198875427246,1.4142308235168457,1.4142178297042847,1.4123810529708862,1.4129202365875244,1.411201000213623,1.4109336137771606,1.4116653203964233,1.148486852645874,1.1442291736602783,1.0615730285644531,1.0842386484146118,1.3072181940078735,1.0258184671401978,1.0230072736740112,1.0169250965118408],[1.186301589012146,1.1360185146331787,1.1344568729400635,0.0,1.3313323259353638,1.3272100687026978,1.3321171998977661,1.331070065498352,1.1864666938781738,1.1869897842407227,1.1898293495178223,1.1941304206848145,0.9027742147445679,0.9012443423271179,1.079931378364563,1.0750609636306763,1.0692583322525024,1.2364922761917114,1.2398320436477661,1.2442444562911987,1.1112834215164185,1.1174569129943848,1.1090844869613647,0.9534761309623718,1.0703915357589722,1.0814731121063232,1.0439637899398804,1.0600522756576538,1.393007516860962,1.3944687843322754,1.4034785032272339,1.4037513732910156,1.4033596515655518,1.3815075159072876,1.3777793645858765,1.378212332725525,1.3774548768997192,1.414199709892273,1.4142436981201172,1.414198875427246,1.4142733812332153,1.414271593093872,1.3862009048461914,1.3860161304473877,1.3329766988754272,1.3278254270553589,1.4084168672561646,1.3715683221817017,1.3726929426193237,1.3719068765640259],[1.3640413284301758,1.3917930126190186,1.3926935195922852,1.3313323259353638,0.0,0.06415465474128723,0.05878468230366707,0.05504058301448822,1.1836305856704712,1.1862154006958008,1.1864193677902222,1.1904215812683105,1.073354721069336,1.0755620002746582,0.8937291502952576,0.8818015456199646,0.8655376434326172,1.1531206369400024,1.1581475734710693,1.1711009740829468,1.351852536201477,1.3483648300170898,1.3462573289871216,1.3194934129714966,1.3707804679870605,1.372383713722229,1.371126413345337,1.3697997331619263,1.4118937253952026,1.411843180656433,1.3990100622177124,1.399315357208252,1.3984839916229248,1.406036615371704,1.3858814239501953,1.3863776922225952,1.3856232166290283,1.414176106452942,1.414219617843628,1.4141795635223389,1.4142537117004395,1.4142482280731201,1.3955936431884766,1.3952176570892334,1.3127707242965698,1.2947298288345337,1.4120709896087646,1.4081158638000488,1.4084333181381226,1.4086304903030396],[1.3625680208206177,1.3912107944488525,1.3920644521713257,1.3272100687026978,0.06415373086929321,0.0,0.11984097212553024,0.09551235288381577,1.1794575452804565,1.1824986934661865,1.1827203035354614,1.1867693662643433,1.0696868896484375,1.0708831548690796,0.8876376152038574,0.8754018545150757,0.8596193194389343,1.1448698043823242,1.1496360301971436,1.1621811389923096,1.3479608297348022,1.3445078134536743,1.342617154121399,1.3145288228988647,1.36776864528656,1.3692163228988647,1.367824673652649,1.3668726682662964,1.4117770195007324,1.4117283821105957,1.3981043100357056,1.3984222412109375,1.3975462913513184,1.405574083328247,1.3843846321105957,1.3849799633026123,1.3839973211288452,1.414203405380249,1.4142465591430664,1.4141916036605835,1.4142647981643677,1.4142591953277588,1.3951287269592285,1.3947162628173828,1.3103312253952026,1.292406439781189,1.4120467901229858,1.4079307317733765,1.4083608388900757,1.408546805381775],[1.3643501996994019,1.3915326595306396,1.392437219619751,1.3321171998977661,0.05878366902470589,0.11984147131443024,0.0,0.052266623824834824,1.183501124382019,1.185678482055664,1.1860487461090088,1.1901798248291016,1.071324110031128,1.0745149850845337,0.8951224684715271,0.883659303188324,0.8672259449958801,1.152244210243225,1.1572016477584839,1.1699401140213013,1.3516128063201904,1.348051905632019,1.3459982872009277,1.3193765878677368,1.3702024221420288,1.3718430995941162,1.3708515167236328,1.3691353797912598,1.411743402481079,1.4116789102554321,1.3991761207580566,1.3995137214660645,1.3987362384796143,1.4056580066680908,1.3854423761367798,1.3859317302703857,1.3852248191833496,1.4141993522644043,1.4142423868179321,1.414190411567688,1.4142653942108154,1.4142587184906006,1.3956105709075928,1.3952765464782715,1.3130066394805908,1.2951520681381226,1.4120562076568604,1.4081429243087769,1.4083454608917236,1.4085944890975952],[1.3633673191070557,1.391201376914978,1.3920586109161377,1.331070065498352,0.055041663348674774,0.09551297873258591,0.05226776376366615,0.0,1.1756203174591064,1.1784735918045044,1.1794860363006592,1.1842703819274902,1.0587470531463623,1.0614433288574219,0.8763548135757446,0.8653841018676758,0.8497677445411682,1.1387338638305664,1.1430596113204956,1.154812216758728,1.3476450443267822,1.3439944982528687,1.3424619436264038,1.3140506744384766,1.3676199913024902,1.369049310684204,1.3684959411621094,1.3666889667510986,1.4117711782455444,1.4116641283035278,1.398193359375,1.398526668548584,1.397765040397644,1.4053654670715332,1.3833976984024048,1.3840956687927246,1.3830339908599854,1.4142091274261475,1.4142522811889648,1.414201259613037,1.4142764806747437,1.4142684936523438,1.3952758312225342,1.3949264287948608,1.3114162683486938,1.2936811447143555,1.4121376276016235,1.408002257347107,1.4082915782928467,1.4085261821746826],[0.8287628889083862,0.9598490595817566,0.9536961317062378,1.1864666938781738,1.1836305856704712,1.1794575452804565,1.183501124382019,1.1756203174591064,0.0,0.04194674268364906,0.08040718734264374,0.1230604350566864,0.7057951092720032,0.7118921279907227,0.7561678290367126,0.7343478798866272,0.7237499356269836,1.3851895332336426,1.3851441144943237,1.3849964141845703,1.3987408876419067,1.3981401920318604,1.3989627361297607,1.2964444160461426,1.3965860605239868,1.3996585607528687,1.3963526487350464,1.3985795974731445,1.4142156839370728,1.414222002029419,1.41422438621521,1.4142186641693115,1.4142252206802368,1.414228081703186,1.414229154586792,1.4142341613769531,1.4142265319824219,1.4126272201538086,1.4131226539611816,1.411638855934143,1.411497712135315,1.412092924118042,1.1116700172424316,1.1094125509262085,0.6997151374816895,0.7561507225036621,1.3238643407821655,1.2119475603103638,1.2120152711868286,1.2112935781478882],[0.824342668056488,0.9567139744758606,0.9510942697525024,1.1869899034500122,1.1862154006958008,1.1824986934661865,1.185678482055664,1.1784735918045044,0.04194674268364906,0.0,0.05110651254653931,0.09209766238927841,0.7213616967201233,0.7282538414001465,0.7732533812522888,0.750747561454773,0.7390496134757996,1.3845778703689575,1.3846147060394287,1.3846341371536255,1.3981518745422363,1.397667407989502,1.398292064666748,1.2999285459518433,1.3960431814193726,1.3990556001663208,1.3956129550933838,1.3978172540664673,1.4142098426818848,1.4142193794250488,1.4142158031463623,1.4142100811004639,1.4142160415649414,1.4142080545425415,1.4142107963562012,1.4142138957977295,1.4142087697982788,1.4124956130981445,1.413045883178711,1.4113843441009521,1.4112145900726318,1.4118950366973877,1.1153415441513062,1.113171935081482,0.6932845115661621,0.7513318061828613,1.323025107383728,1.2136726379394531,1.2135670185089111,1.2126675844192505],[0.8216472864151001,0.9534347057342529,0.9481849074363708,1.1898293495178223,1.1864193677902222,1.1827203035354614,1.1860487461090088,1.1794860363006592,0.08040793240070343,0.05110534653067589,0.0,0.051495734602212906,0.745547890663147,0.7528330087661743,0.7986311316490173,0.7750856280326843,0.7620636820793152,1.3834377527236938,1.3836028575897217,1.383910059928894,1.3978137969970703,1.3973649740219116,1.3978244066238403,1.3045300245285034,1.395808219909668,1.3987377882003784,1.3951562643051147,1.3973854780197144,1.4142158031463623,1.4142282009124756,1.414212703704834,1.4142062664031982,1.4142125844955444,1.4142168760299683,1.4142206907272339,1.4142249822616577,1.4142175912857056,1.4122878313064575,1.4128950834274292,1.411116123199463,1.4108995199203491,1.4116753339767456,1.1235110759735107,1.1211882829666138,0.6926835179328918,0.75203537940979,1.3232375383377075,1.2165400981903076,1.2160533666610718,1.2152236700057983],[0.8140830993652344,0.9523178339004517,0.9475924372673035,1.1941304206848145,1.1904215812683105,1.1867693662643433,1.1901798248291016,1.1842703819274902,0.1230599507689476,0.09209766238927841,0.051495734602212906,0.0,0.7737143635749817,0.7809257507324219,0.8262442946434021,0.8018477559089661,0.7878842949867249,1.3831952810287476,1.3834377527236938,1.3839428424835205,1.3976221084594727,1.3972382545471191,1.3974846601486206,1.3101521730422974,1.395742416381836,1.3985843658447266,1.3948928117752075,1.3971000909805298,1.4142234325408936,1.4142394065856934,1.4142475128173828,1.4142404794692993,1.4142448902130127,1.4142390489578247,1.4142546653747559,1.414260745048523,1.4142507314682007,1.4121919870376587,1.4128191471099854,1.4108957052230835,1.4106709957122803,1.4114627838134766,1.1328518390655518,1.1302742958068848,0.6926482319831848,0.753722071647644,1.3228780031204224,1.2222894430160522,1.2214281558990479,1.220473289489746],[1.0429106950759888,1.0352966785430908,1.0281325578689575,0.9027742743492126,1.073354721069336,1.0696868896484375,1.071324110031128,1.0587470531463623,0.7057951092720032,0.7213616371154785,0.745547890663147,0.7737143635749817,0.0,0.03403989225625992,0.45695534348487854,0.4592358469963074,0.4640863537788391,1.3272722959518433,1.3274284601211548,1.3279937505722046,1.3482190370559692,1.3479207754135132,1.3491270542144775,1.1359522342681885,1.3404624462127686,1.3460429906845093,1.3382078409194946,1.3436851501464844,1.4127163887023926,1.4125946760177612,1.412007212638855,1.4120612144470215,1.412028193473816,1.4112123250961304,1.408710241317749,1.4090211391448975,1.4085057973861694,1.4141931533813477,1.4142428636550903,1.4141875505447388,1.4142943620681763,1.4142848253250122,1.3064652681350708,1.3060437440872192,1.1193324327468872,1.125149130821228,1.390816569328308,1.3181918859481812,1.3198996782302856,1.319712519645691],[1.0468953847885132,1.0402659177780151,1.0329310894012451,0.9012444019317627,1.0755620002746582,1.0708831548690796,1.0745149850845337,1.0614433288574219,0.7118921279907227,0.7282538414001465,0.7528330087661743,0.7809257507324219,0.03403814136981964,0.0,0.4535122215747833,0.4561457633972168,0.4621555507183075,1.3289105892181396,1.329099178314209,1.3296656608581543,1.349281668663025,1.349034309387207,1.350212574005127,1.1347085237503052,1.3419550657272339,1.347609281539917,1.3395277261734009,1.3451780080795288,1.4128142595291138,1.412691354751587,1.4121272563934326,1.4121862649917603,1.4121588468551636,1.4113852977752686,1.4090585708618164,1.4093643426895142,1.4088565111160278,1.4141851663589478,1.4142340421676636,1.4141812324523926,1.41428804397583,1.4142773151397705,1.3073805570602417,1.306911587715149,1.1235928535461426,1.129044771194458,1.3911932706832886,1.3194199800491333,1.3213294744491577,1.3211005926132202],[1.1978167295455933,1.2147513628005981,1.2101887464523315,1.079931378364563,0.8937292098999023,0.8876376152038574,0.8951225280761719,0.8763548135757446,0.7561678290367126,0.773253321647644,0.7986311912536621,0.8262443542480469,0.45695534348487854,0.4535123407840729,0.0,0.042578574270009995,0.07830104231834412,1.284427285194397,1.2851693630218506,1.2874823808670044,1.3474727869033813,1.346154808998108,1.3465428352355957,1.1920379400253296,1.3483902215957642,1.3536559343338013,1.3469454050064087,1.3501653671264648,1.4126975536346436,1.4125800132751465,1.4105443954467773,1.4106072187423706,1.4105441570281982,1.4106407165527344,1.406410574913025,1.406933069229126,1.4060940742492676,1.414161205291748,1.41421639919281,1.414169430732727,1.4142770767211914,1.4142684936523438,1.3305599689483643,1.3298423290252686,1.1568028926849365,1.1554837226867676,1.3984161615371704,1.363466501235962,1.3663452863693237,1.3659191131591797],[1.1857260465621948,1.204429268836975,1.2001020908355713,1.0750609636306763,0.8818015456199646,0.8754018545150757,0.883659303188324,0.8653841018676758,0.7343478202819824,0.750747561454773,0.7750856280326843,0.8018476963043213,0.4592359662055969,0.45614564418792725,0.042578574270009995,0.0,0.040064360946416855,1.2822092771530151,1.2831976413726807,1.286001205444336,1.3456722497940063,1.34438157081604,1.3445310592651367,1.1923720836639404,1.3464744091033936,1.3518050909042358,1.344753623008728,1.3481265306472778,1.412606120109558,1.412493109703064,1.41060209274292,1.4106643199920654,1.4106007814407349,1.410577416419983,1.40651535987854,1.4070324897766113,1.4062039852142334,1.41422438621521,1.4142787456512451,1.4142237901687622,1.4143345355987549,1.4143248796463013,1.3281060457229614,1.3272136449813843,1.1404496431350708,1.1403639316558838,1.3976309299468994,1.3616666793823242,1.3643587827682495,1.3638967275619507],[1.1794891357421875,1.1984297037124634,1.194420337677002,1.0692583322525024,0.8655375838279724,0.8596193194389343,0.8672259449958801,0.8497677445411682,0.7237499356269836,0.7390496134757996,0.7620636820793152,0.7878842949867249,0.4640863537788391,0.4621555507183075,0.07830028235912323,0.04006585106253624,0.0,1.2824651002883911,1.2838127613067627,1.287376880645752,1.348167896270752,1.3471293449401855,1.3465994596481323,1.197522759437561,1.349145770072937,1.3546606302261353,1.3466496467590332,1.350317120552063,1.4128941297531128,1.4127819538116455,1.4109437465667725,1.4109935760498047,1.4109405279159546,1.411077857017517,1.4074006080627441,1.4078476428985596,1.4071288108825684,1.4142292737960815,1.4142833948135376,1.4142078161239624,1.4143226146697998,1.4143099784851074,1.3281614780426025,1.3272117376327515,1.1306774616241455,1.1310656070709229,1.3973710536956787,1.3616949319839478,1.3641892671585083,1.3637065887451172],[1.4037282466888428,1.4037678241729736,1.4042407274246216,1.2364922761917114,1.1531206369400024,1.1448698043823242,1.152244210243225,1.1387338638305664,1.3851895332336426,1.3845778703689575,1.3834377527236938,1.3831952810287476,1.3272722959518433,1.3289105892181396,1.284427285194397,1.2822092771530151,1.2824651002883911,0.0,0.037815798074007034,0.09832534193992615,0.8250777721405029,0.8282062411308289,0.8287257552146912,0.8737210631370544,0.9566425681114197,0.9544012546539307,0.9529202580451965,0.950960636138916,1.3080387115478516,1.3146822452545166,1.2060143947601318,1.2096601724624634,1.198384404182434,1.2137876749038696,1.0138146877288818,1.0121607780456543,1.017462134361267,1.4141978025436401,1.41423761844635,1.4141979217529297,1.4142528772354126,1.4142462015151978,1.4131613969802856,1.4131001234054565,1.4050227403640747,1.3989810943603516,1.4142037630081177,1.4135104417800903,1.413666844367981,1.4136269092559814],[1.4039306640625,1.4040100574493408,1.4044550657272339,1.2398320436477661,1.1581475734710693,1.149635910987854,1.1572016477584839,1.1430596113204956,1.3851441144943237,1.3846147060394287,1.3836028575897217,1.3834377527236938,1.3274284601211548,1.329099178314209,1.2851693630218506,1.2831976413726807,1.2838127613067627,0.037817373871803284,0.0,0.06420388072729111,0.8289210796356201,0.8309625387191772,0.8335135579109192,0.8789229989051819,0.9571669101715088,0.9543624520301819,0.9554064869880676,0.9526339173316956,1.3067599534988403,1.3128842115402222,1.2035713195800781,1.2070759534835815,1.1964038610458374,1.214331030845642,1.0173652172088623,1.0166065692901611,1.0202257633209229,1.414199948310852,1.4142390489578247,1.4141995906829834,1.4142532348632812,1.414247751235962,1.4130823612213135,1.4130194187164307,1.4049901962280273,1.3992197513580322,1.4141978025436401,1.4134901762008667,1.4136289358139038,1.4135981798171997],[1.4040076732635498,1.4041575193405151,1.4045145511627197,1.2442444562911987,1.1711009740829468,1.1621811389923096,1.1699401140213013,1.154812216758728,1.3849964141845703,1.3846341371536255,1.383910059928894,1.3839428424835205,1.3279937505722046,1.3296656608581543,1.2874823808670044,1.286001205444336,1.287376880645752,0.09832534193992615,0.06420480459928513,0.0,0.8322257399559021,0.8323236107826233,0.8387771844863892,0.8848509788513184,0.9537080526351929,0.949770987033844,0.9556465744972229,0.9512656331062317,1.3068537712097168,1.312089443206787,1.202683687210083,1.2059766054153442,1.1965224742889404,1.2164095640182495,1.0278615951538086,1.0284993648529053,1.029554843902588,1.4142146110534668,1.4142508506774902,1.4142100811004639,1.4142627716064453,1.4142560958862305,1.4128645658493042,1.4128026962280273,1.405007243156433,1.3998262882232666,1.4141596555709839,1.4133481979370117,1.413507103919983,1.4134618043899536],[1.4022308588027954,1.399926781654358,1.4002755880355835,1.1112834215164185,1.351852536201477,1.3479608297348022,1.3516128063201904,1.3476450443267822,1.3987408876419067,1.3981518745422363,1.3978137969970703,1.3976221084594727,1.3482190370559692,1.349281668663025,1.3474727869033813,1.3456722497940063,1.348167896270752,0.8250777721405029,0.8289210200309753,0.8322257399559021,0.0,0.049066800624132156,0.043630871921777725,0.34644147753715515,0.31365278363227844,0.32984834909439087,0.2691563665866852,0.2883336842060089,1.1937015056610107,1.209401249885559,1.285323977470398,1.2867815494537354,1.2815877199172974,1.1087331771850586,1.0748653411865234,1.0727282762527466,1.0764330625534058,1.4141782522201538,1.4142112731933594,1.4141919612884521,1.4142526388168335,1.414243221282959,1.4134931564331055,1.413513422012329,1.409726858139038,1.407339096069336,1.414226770401001,1.4133058786392212,1.4134317636489868,1.4133952856063843],[1.402640461921692,1.4003174304962158,1.4005950689315796,1.1174569129943848,1.3483648300170898,1.3445078134536743,1.348051905632019,1.3439944982528687,1.3981401920318604,1.397667407989502,1.3973649740219116,1.3972382545471191,1.3479207754135132,1.349034309387207,1.346154808998108,1.34438157081604,1.3471293449401855,0.8282062411308289,0.8309624195098877,0.8323236107826233,0.04906558617949486,0.0,0.08086400479078293,0.3584505021572113,0.29706287384033203,0.311229407787323,0.2653207778930664,0.2790229320526123,1.1928119659423828,1.2075053453445435,1.2854770421981812,1.2867125272750854,1.2819838523864746,1.111002802848816,1.0818414688110352,1.0806940793991089,1.0825939178466797,1.4142367839813232,1.4142680168151855,1.4142248630523682,1.4142881631851196,1.4142735004425049,1.4133009910583496,1.4133137464523315,1.4095109701156616,1.4071582555770874,1.4141746759414673,1.41313898563385,1.413257122039795,1.4132425785064697],[1.4025861024856567,1.4003257751464844,1.4007456302642822,1.1090844869613647,1.3462573289871216,1.342617154121399,1.3459982872009277,1.3424619436264038,1.3989627361297607,1.398292064666748,1.3978244066238403,1.3974846601486206,1.3491270542144775,1.350212574005127,1.3465428352355957,1.3445310592651367,1.3465994596481323,0.8287257552146912,0.8335135579109192,0.8387771844863892,0.043630871921777725,0.08086400479078293,0.0,0.3436315357685089,0.33096662163734436,0.3488720953464508,0.27700376510620117,0.30043286085128784,1.1960949897766113,1.2129546403884888,1.287224292755127,1.28865385055542,1.2831710577011108,1.109397053718567,1.0727430582046509,1.0701038837432861,1.0746926069259644,1.4141740798950195,1.4142073392868042,1.4141758680343628,1.414235234260559,1.4142241477966309,1.4136261940002441,1.413635492324829,1.4097979068756104,1.4071272611618042,1.4142296314239502,1.4135046005249023,1.413614273071289,1.413586139678955],[1.3439334630966187,1.3291643857955933,1.3261522054672241,0.9534761309623718,1.3194934129714966,1.3145288228988647,1.3193765878677368,1.3140506744384766,1.2964445352554321,1.2999285459518433,1.3045300245285034,1.3101521730422974,1.1359522342681885,1.1347085237503052,1.1920379400253296,1.1923720836639404,1.197522759437561,0.8737210631370544,0.8789229989051819,0.8848509192466736,0.3464416563510895,0.358450323343277,0.343631386756897,0.0,0.4787335693836212,0.4965193569660187,0.43634042143821716,0.46213558316230774,1.1972250938415527,1.2145577669143677,1.287787675857544,1.2890692949295044,1.2834889888763428,1.1176189184188843,1.0858187675476074,1.0824817419052124,1.0881491899490356,1.4141567945480347,1.4141992330551147,1.414167046546936,1.4142488241195679,1.41423499584198,1.405349850654602,1.4055625200271606,1.3855433464050293,1.3821901082992554,1.4141777753829956,1.4028475284576416,1.404286503791809,1.4042465686798096],[1.3997373580932617,1.39676833152771,1.3971577882766724,1.0703915357589722,1.3707804679870605,1.3677685260772705,1.3702024221420288,1.3676199913024902,1.3965860605239868,1.3960431814193726,1.395808219909668,1.395742416381836,1.3404624462127686,1.3419550657272339,1.3483902215957642,1.3464744091033936,1.349145770072937,0.9566425681114197,0.9571669101715088,0.9537080526351929,0.31365278363227844,0.2970626652240753,0.33096662163734436,0.47873368859291077,0.0,0.038048360496759415,0.11248764395713806,0.06871853768825531,1.2356520891189575,1.2467472553253174,1.333065390586853,1.3340845108032227,1.3315125703811646,1.1833231449127197,1.179828405380249,1.1794886589050293,1.179028034210205,1.4142249822616577,1.4142398834228516,1.4142298698425293,1.414254903793335,1.4142504930496216,1.4133228063583374,1.4133421182632446,1.409535527229309,1.4073829650878906,1.4142025709152222,1.412929892539978,1.4130536317825317,1.4130374193191528],[1.4013144969940186,1.3984706401824951,1.398834466934204,1.0814731121063232,1.372383713722229,1.3692163228988647,1.3718430995941162,1.369049310684204,1.3996585607528687,1.3990556001663208,1.3987377882003784,1.3985843658447266,1.3460429906845093,1.347609281539917,1.3536559343338013,1.3518050909042358,1.3546606302261353,0.9544012546539307,0.9543624520301819,0.9497710466384888,0.32984834909439087,0.311229407787323,0.3488720953464508,0.4965193569660187,0.038048360496759415,0.0,0.14046978950500488,0.09527179598808289,1.2405445575714111,1.2509956359863281,1.3362338542938232,1.3372302055358887,1.3347485065460205,1.1907254457473755,1.1881141662597656,1.1879782676696777,1.187191367149353,1.4142130613327026,1.4142258167266846,1.4142138957977295,1.4142335653305054,1.4142324924468994,1.4134899377822876,1.4135191440582275,1.4102997779846191,1.4083670377731323,1.4141967296600342,1.4131171703338623,1.4132360219955444,1.4131903648376465],[1.3984349966049194,1.3951679468154907,1.3957728147506714,1.0439637899398804,1.371126413345337,1.367824673652649,1.3708515167236328,1.3684959411621094,1.3963526487350464,1.3956129550933838,1.3951561450958252,1.3948928117752075,1.3382078409194946,1.3395277261734009,1.3469454050064087,1.344753623008728,1.3466496467590332,0.9529202580451965,0.9554064869880676,0.9556465744972229,0.2691563665866852,0.2653209865093231,0.27700376510620117,0.4363405704498291,0.11248817294836044,0.14046978950500488,0.0,0.05617038533091545,1.2366398572921753,1.2505769729614258,1.3335533142089844,1.3347004652023315,1.3314796686172485,1.1795543432235718,1.1713653802871704,1.1699156761169434,1.1714316606521606,1.4142053127288818,1.414223313331604,1.4142080545425415,1.414233922958374,1.414231777191162,1.4134676456451416,1.4134901762008667,1.4094996452331543,1.4070082902908325,1.4142098426818848,1.4130476713180542,1.413124442100525,1.4131675958633423],[1.4007179737091064,1.3982278108596802,1.3988677263259888,1.0600522756576538,1.3697997331619263,1.3668725490570068,1.3691353797912598,1.3666889667510986,1.3985795974731445,1.3978172540664673,1.3973854780197144,1.3971000909805298,1.3436851501464844,1.3451780080795288,1.3501653671264648,1.3481265306472778,1.350317120552063,0.950960636138916,0.9526339173316956,0.9512656331062317,0.2883336842060089,0.2790229320526123,0.30043286085128784,0.4621354341506958,0.06871767342090607,0.09527179598808289,0.05617144703865051,0.0,1.2385375499725342,1.2512167692184448,1.335656762123108,1.33692467212677,1.333837628364563,1.1847435235977173,1.1777819395065308,1.1767624616622925,1.1776394844055176,1.4141916036605835,1.4142074584960938,1.414183497428894,1.4142050743103027,1.4142050743103027,1.4135278463363647,1.413536548614502,1.4098206758499146,1.407547950744629,1.4141767024993896,1.4133384227752686,1.413395643234253,1.4133821725845337],[1.4141825437545776,1.4141989946365356,1.4141855239868164,1.393007516860962,1.4118937253952026,1.4117770195007324,1.411743402481079,1.4117711782455444,1.4142156839370728,1.4142098426818848,1.4142158031463623,1.4142234325408936,1.4127163887023926,1.4128142595291138,1.4126975536346436,1.412606120109558,1.4128941297531128,1.3080387115478516,1.3067599534988403,1.3068537712097168,1.1937015056610107,1.1928119659423828,1.1960949897766113,1.1972250938415527,1.2356520891189575,1.2405445575714111,1.2366398572921753,1.2385375499725342,0.0,0.07274653017520905,1.0885875225067139,1.08530855178833,1.0848699808120728,0.6504170298576355,0.9574797749519348,0.9603684544563293,0.9492464065551758,1.414168119430542,1.4142265319824219,1.4141806364059448,1.414313554763794,1.4142887592315674,1.4142005443572998,1.4142283201217651,1.414153814315796,1.4142060279846191,1.4142966270446777,1.4141881465911865,1.4141839742660522,1.414189338684082],[1.414223313331604,1.4142374992370605,1.4142180681228638,1.3944687843322754,1.411843180656433,1.4117283821105957,1.4116789102554321,1.4116641283035278,1.414222002029419,1.4142193794250488,1.4142282009124756,1.4142394065856934,1.4125946760177612,1.412691354751587,1.4125800132751465,1.412493109703064,1.4127819538116455,1.3146822452545166,1.3128843307495117,1.312089443206787,1.209401249885559,1.2075053453445435,1.2129546403884888,1.2145577669143677,1.2467472553253174,1.2509956359863281,1.2505769729614258,1.2512167692184448,0.07274653017520905,0.0,1.0896556377410889,1.0867252349853516,1.0874966382980347,0.649046003818512,0.961571216583252,0.9655743837356567,0.9530541896820068,1.414206624031067,1.4142627716064453,1.4141970872879028,1.4143295288085938,1.4143016338348389,1.4142402410507202,1.4142658710479736,1.4141942262649536,1.4142451286315918,1.4143102169036865,1.4142279624938965,1.4142245054244995,1.4142292737960815],[1.4142059087753296,1.414206624031067,1.4142258167266846,1.4034785032272339,1.3990100622177124,1.3981043100357056,1.3991761207580566,1.398193359375,1.41422438621521,1.4142158031463623,1.414212703704834,1.4142475128173828,1.412007212638855,1.4121272563934326,1.4105443954467773,1.41060209274292,1.4109437465667725,1.2060143947601318,1.2035713195800781,1.202683687210083,1.285323977470398,1.2854770421981812,1.287224292755127,1.287787675857544,1.333065390586853,1.3362339735031128,1.3335533142089844,1.335656762123108,1.0885875225067139,1.0896556377410889,0.0,0.03437097743153572,0.05443952605128288,0.8988308906555176,0.658526599407196,0.6643484234809875,0.6490858793258667,1.414174199104309,1.414255976676941,1.414146900177002,1.4143539667129517,1.4143191576004028,1.4142311811447144,1.414241075515747,1.4141641855239868,1.4141595363616943,1.4143065214157104,1.4141626358032227,1.4142600297927856,1.4142177104949951],[1.414214015007019,1.4142082929611206,1.414228081703186,1.4037513732910156,1.399315357208252,1.3984222412109375,1.3995137214660645,1.398526668548584,1.4142186641693115,1.4142100811004639,1.4142062664031982,1.4142404794692993,1.4120612144470215,1.4121862649917603,1.4106072187423706,1.4106643199920654,1.4109935760498047,1.2096601724624634,1.2070759534835815,1.2059766054153442,1.2867815494537354,1.2867125272750854,1.28865385055542,1.2890692949295044,1.3340845108032227,1.3372302055358887,1.3347004652023315,1.33692467212677,1.08530855178833,1.0867252349853516,0.03437097743153572,0.0,0.05256682634353638,0.8970663547515869,0.6647807955741882,0.6708881258964539,0.6544787883758545,1.4142045974731445,1.4142866134643555,1.4141687154769897,1.4143766164779663,1.4143414497375488,1.4142367839813232,1.414247989654541,1.4141539335250854,1.4141532182693481,1.414319634437561,1.414193034172058,1.4142929315567017,1.4142495393753052],[1.4142206907272339,1.4142181873321533,1.4142390489578247,1.4033596515655518,1.3984839916229248,1.3975462913513184,1.3987362384796143,1.397765040397644,1.4142252206802368,1.4142160415649414,1.4142125844955444,1.4142448902130127,1.412028193473816,1.4121588468551636,1.4105441570281982,1.4106007814407349,1.4109405279159546,1.198384404182434,1.1964038610458374,1.1965224742889404,1.2815877199172974,1.2819838523864746,1.2831710577011108,1.2834889888763428,1.3315125703811646,1.3347485065460205,1.3314796686172485,1.333837628364563,1.0848699808120728,1.0874966382980347,0.05443952605128288,0.05256682634353638,0.0,0.894950270652771,0.6518253684043884,0.6568794846534729,0.6420836448669434,1.414201021194458,1.4142831563949585,1.41416335105896,1.4143717288970947,1.4143356084823608,1.4142557382583618,1.414263129234314,1.4141854047775269,1.4141714572906494,1.414329171180725,1.4141937494277954,1.4142937660217285,1.4142515659332275],[1.4142134189605713,1.414209246635437,1.4142041206359863,1.3815075159072876,1.406036615371704,1.405574083328247,1.4056580066680908,1.4053654670715332,1.414228081703186,1.4142080545425415,1.4142168760299683,1.4142390489578247,1.4112123250961304,1.4113852977752686,1.4106407165527344,1.410577416419983,1.411077857017517,1.2137876749038696,1.214331030845642,1.2164095640182495,1.1087331771850586,1.111002802848816,1.109397053718567,1.1176189184188843,1.1833231449127197,1.1907254457473755,1.1795543432235718,1.1847435235977173,0.6504170298576355,0.649046003818512,0.8988309502601624,0.8970663547515869,0.894950270652771,0.0,0.6513919830322266,0.6588942408561707,0.6406587958335876,1.4142025709152222,1.4142495393753052,1.4141782522201538,1.4143497943878174,1.4143123626708984,1.4141899347305298,1.4142333269119263,1.414159893989563,1.4141900539398193,1.4143314361572266,1.4141947031021118,1.4142632484436035,1.4142323732376099],[1.4142040014266968,1.4141873121261597,1.414198875427246,1.3777793645858765,1.3858814239501953,1.3843846321105957,1.3854423761367798,1.3833976984024048,1.414229154586792,1.4142107963562012,1.4142206907272339,1.4142546653747559,1.408710241317749,1.4090585708618164,1.406410574913025,1.40651535987854,1.4074006080627441,1.0138146877288818,1.0173652172088623,1.0278615951538086,1.0748653411865234,1.0818414688110352,1.0727430582046509,1.0858186483383179,1.179828405380249,1.1881141662597656,1.1713653802871704,1.1777819395065308,0.9574797749519348,0.961571216583252,0.658526599407196,0.6647809147834778,0.651825487613678,0.6513921022415161,0.0,0.04162866249680519,0.041437793523073196,1.4141960144042969,1.4142539501190186,1.4141680002212524,1.4143450260162354,1.4143110513687134,1.4142086505889893,1.4142206907272339,1.414170265197754,1.4141640663146973,1.4143279790878296,1.4141672849655151,1.414290428161621,1.4142156839370728],[1.4142308235168457,1.4142203330993652,1.4142308235168457,1.378212332725525,1.3863776922225952,1.3849799633026123,1.3859317302703857,1.3840956687927246,1.4142341613769531,1.4142138957977295,1.4142249822616577,1.414260745048523,1.4090211391448975,1.4093643426895142,1.406933069229126,1.4070324897766113,1.4078476428985596,1.0121607780456543,1.0166065692901611,1.0284993648529053,1.0727282762527466,1.0806940793991089,1.0701038837432861,1.0824817419052124,1.1794886589050293,1.1879782676696777,1.1699156761169434,1.1767624616622925,0.9603684544563293,0.9655743837356567,0.6643484830856323,0.6708881258964539,0.6568794846534729,0.6588941812515259,0.04162866249680519,0.0,0.0741681382060051,1.4142019748687744,1.4142624139785767,1.4141616821289062,1.4143391847610474,1.4143043756484985,1.414228081703186,1.4142414331436157,1.414175271987915,1.4141645431518555,1.4143376350402832,1.4141991138458252,1.4143216609954834,1.4142483472824097],[1.4142131805419922,1.414206862449646,1.4142178297042847,1.3774548768997192,1.3856232166290283,1.3839973211288452,1.3852248191833496,1.3830339908599854,1.4142265319824219,1.4142087697982788,1.4142175912857056,1.4142507314682007,1.4085057973861694,1.4088565111160278,1.4060940742492676,1.4062039852142334,1.4071288108825684,1.017462134361267,1.0202257633209229,1.029554843902588,1.0764329433441162,1.0825939178466797,1.0746926069259644,1.0881491899490356,1.179028034210205,1.187191367149353,1.1714316606521606,1.1776394844055176,0.9492464065551758,0.9530541896820068,0.6490859389305115,0.6544787883758545,0.6420836448669434,0.6406587958335876,0.0414392314851284,0.0741681382060051,0.0,1.4141974449157715,1.4142543077468872,1.4141640663146973,1.4143415689468384,1.4143072366714478,1.4141931533813477,1.4142078161239624,1.414154052734375,1.4141489267349243,1.414317011833191,1.4141819477081299,1.4143050909042358,1.414229393005371],[1.4093965291976929,1.412293553352356,1.4123810529708862,1.414199709892273,1.414176106452942,1.414203405380249,1.4141993522644043,1.4142091274261475,1.4126272201538086,1.4124956130981445,1.4122878313064575,1.4121919870376587,1.4141931533813477,1.4141851663589478,1.414161205291748,1.41422438621521,1.4142292737960815,1.4141978025436401,1.414199948310852,1.4142146110534668,1.4141782522201538,1.4142367839813232,1.4141740798950195,1.4141567945480347,1.4142249822616577,1.4142130613327026,1.4142053127288818,1.4141916036605835,1.414168119430542,1.414206624031067,1.414174199104309,1.4142045974731445,1.414201021194458,1.4142025709152222,1.4141960144042969,1.4142019748687744,1.4141974449157715,0.0,0.028897423297166824,0.5092541575431824,0.4910562038421631,0.5397685766220093,1.3898245096206665,1.3897345066070557,1.4058120250701904,1.3560558557510376,1.2751208543777466,1.3795751333236694,1.3794139623641968,1.3796879053115845],[1.4107441902160645,1.4128614664077759,1.4129202365875244,1.4142436981201172,1.414219617843628,1.4142465591430664,1.4142423868179321,1.4142522811889648,1.4131226539611816,1.413045883178711,1.4128950834274292,1.4128191471099854,1.4142428636550903,1.4142340421676636,1.41421639919281,1.4142787456512451,1.4142833948135376,1.41423761844635,1.4142390489578247,1.4142508506774902,1.4142112731933594,1.4142680168151855,1.4142073392868042,1.4141992330551147,1.4142398834228516,1.4142258167266846,1.414223313331604,1.4142074584960938,1.4142265319824219,1.4142627716064453,1.414255976676941,1.4142866134643555,1.4142831563949585,1.4142495393753052,1.4142539501190186,1.4142624139785767,1.4142543077468872,0.028899485245347023,0.0,0.5066592693328857,0.4897572994232178,0.5351381897926331,1.3939290046691895,1.39389967918396,1.4083905220031738,1.3558052778244019,1.2846847772598267,1.383604884147644,1.3834816217422485,1.3839327096939087],[1.4066569805145264,1.4109838008880615,1.411201000213623,1.414198875427246,1.4141795635223389,1.4141916036605835,1.414190411567688,1.414201259613037,1.411638855934143,1.4113843441009521,1.411116123199463,1.4108957052230835,1.4141875505447388,1.4141812324523926,1.414169430732727,1.4142237901687622,1.4142078161239624,1.4141979217529297,1.4141995906829834,1.4142100811004639,1.4141919612884521,1.4142248630523682,1.4141758680343628,1.414167046546936,1.4142298698425293,1.4142138957977295,1.4142080545425415,1.414183497428894,1.4141806364059448,1.4141970872879028,1.414146900177002,1.4141687154769897,1.41416335105896,1.4141782522201538,1.4141680002212524,1.4141616821289062,1.4141640663146973,0.5092540383338928,0.5066592693328857,0.0,0.0508187972009182,0.06882860511541367,1.3664010763168335,1.365946888923645,1.3989360332489014,1.296971082687378,1.1587233543395996,1.355413794517517,1.35573148727417,1.3552626371383667],[1.4058283567428589,1.4106923341751099,1.4109336137771606,1.4142733812332153,1.4142537117004395,1.4142647981643677,1.4142653942108154,1.4142764806747437,1.411497712135315,1.4112145900726318,1.4108995199203491,1.4106709957122803,1.4142943620681763,1.41428804397583,1.4142770767211914,1.4143345355987549,1.4143226146697998,1.4142528772354126,1.4142532348632812,1.4142627716064453,1.4142526388168335,1.4142881631851196,1.414235234260559,1.4142488241195679,1.414254903793335,1.4142335653305054,1.414233922958374,1.4142050743103027,1.414313554763794,1.4143295288085938,1.4143539667129517,1.4143766164779663,1.4143717288970947,1.4143497943878174,1.4143450260162354,1.4143391847610474,1.4143415689468384,0.4910562038421631,0.4897574186325073,0.05081762373447418,0.0,0.11427189409732819,1.365462064743042,1.36494779586792,1.3976993560791016,1.3051003217697144,1.1613454818725586,1.35446298122406,1.3546898365020752,1.3540829420089722],[1.4076207876205444,1.4114936590194702,1.4116653203964233,1.414271593093872,1.4142482280731201,1.4142591953277588,1.4142587184906006,1.4142684936523438,1.412092924118042,1.4118950366973877,1.4116753339767456,1.4114627838134766,1.4142848253250122,1.4142773151397705,1.4142684936523438,1.4143248796463013,1.4143099784851074,1.4142462015151978,1.414247751235962,1.4142560958862305,1.414243221282959,1.4142735004425049,1.4142241477966309,1.41423499584198,1.4142504930496216,1.4142324924468994,1.414231777191162,1.4142050743103027,1.4142887592315674,1.4143016338348389,1.4143191576004028,1.4143414497375488,1.4143356084823608,1.4143123626708984,1.4143110513687134,1.4143043756484985,1.4143072366714478,0.5397686958312988,0.5351381897926331,0.06882860511541367,0.11427189409732819,0.0,1.3688502311706543,1.368403434753418,1.4014087915420532,1.2885112762451172,1.161368489265442,1.3559726476669312,1.3562877178192139,1.3560367822647095],[0.9987942576408386,1.163755178451538,1.148486852645874,1.3862009048461914,1.3955936431884766,1.3951287269592285,1.3956105709075928,1.3952758312225342,1.1116700172424316,1.1153415441513062,1.1235110759735107,1.1328518390655518,1.3064652681350708,1.3073805570602417,1.3305599689483643,1.3281060457229614,1.3281614780426025,1.4131613969802856,1.4130823612213135,1.4128645658493042,1.4134931564331055,1.4133009910583496,1.4136261940002441,1.405349850654602,1.4133228063583374,1.4134899377822876,1.4134676456451416,1.4135278463363647,1.4142005443572998,1.4142402410507202,1.4142311811447144,1.4142367839813232,1.4142557382583618,1.4141899347305298,1.4142086505889893,1.414228081703186,1.4141931533813477,1.3898245096206665,1.3939290046691895,1.3664010763168335,1.365462064743042,1.3688502311706543,0.0,0.04611092060804367,0.6923372745513916,0.6952300071716309,0.9568553566932678,0.780988335609436,0.7900177836418152,0.7839375734329224],[0.9939128160476685,1.1595369577407837,1.1442291736602783,1.3860161304473877,1.3952176570892334,1.3947162628173828,1.3952765464782715,1.3949264287948608,1.1094125509262085,1.113171935081482,1.1211882829666138,1.1302742958068848,1.3060437440872192,1.306911587715149,1.3298423290252686,1.3272136449813843,1.3272117376327515,1.4131001234054565,1.4130194187164307,1.4128026962280273,1.413513422012329,1.4133137464523315,1.413635492324829,1.4055625200271606,1.4133421182632446,1.4135191440582275,1.4134901762008667,1.413536548614502,1.4142283201217651,1.4142658710479736,1.414241075515747,1.414247989654541,1.414263129234314,1.4142333269119263,1.4142206907272339,1.4142414331436157,1.4142078161239624,1.3897345066070557,1.39389967918396,1.365946888923645,1.36494779586792,1.368403434753418,0.04611092060804367,0.0,0.6889588832855225,0.6934245228767395,0.9547014236450195,0.779371976852417,0.787280797958374,0.7813718318939209],[0.9140720367431641,1.0669684410095215,1.0615730285644531,1.3329766988754272,1.3127707242965698,1.3103312253952026,1.3130066394805908,1.3114162683486938,0.6997150182723999,0.6932845711708069,0.6926836371421814,0.6926482319831848,1.1193324327468872,1.1235928535461426,1.1568028926849365,1.1404495239257812,1.1306774616241455,1.4050227403640747,1.4049901962280273,1.405007243156433,1.409726858139038,1.4095109701156616,1.4097979068756104,1.3855433464050293,1.409535527229309,1.4102997779846191,1.4094996452331543,1.4098206758499146,1.414153814315796,1.4141942262649536,1.4141641855239868,1.4141539335250854,1.4141854047775269,1.414159893989563,1.414170265197754,1.414175271987915,1.414154052734375,1.4058120250701904,1.4083905220031738,1.3989360332489014,1.3976993560791016,1.4014087915420532,0.6923373341560364,0.6889587640762329,0.0,0.31846532225608826,1.1615924835205078,1.0805491209030151,1.0834871530532837,1.07864511013031],[0.9473844766616821,1.0896323919296265,1.0842386484146118,1.3278254270553589,1.2947298288345337,1.292406439781189,1.2951520681381226,1.2936811447143555,0.7561506032943726,0.7513318657875061,0.7520353198051453,0.753722071647644,1.125149130821228,1.1290446519851685,1.1554837226867676,1.1403639316558838,1.1310654878616333,1.3989810943603516,1.3992197513580322,1.3998262882232666,1.407339096069336,1.4071582555770874,1.4071272611618042,1.3821901082992554,1.4073829650878906,1.4083670377731323,1.4070082902908325,1.407547950744629,1.4142060279846191,1.4142451286315918,1.4141595363616943,1.4141532182693481,1.4141714572906494,1.4141900539398193,1.4141640663146973,1.4141645431518555,1.4141489267349243,1.3560558557510376,1.3558052778244019,1.296971082687378,1.3051003217697144,1.2885112762451172,0.6952300071716309,0.6934245228767395,0.31846532225608826,0.0,1.098583698272705,1.0710586309432983,1.0750017166137695,1.0708459615707397],[1.241729736328125,1.3093029260635376,1.3072181940078735,1.4084168672561646,1.4120709896087646,1.4120467901229858,1.4120562076568604,1.4121376276016235,1.3238643407821655,1.323025107383728,1.3232375383377075,1.3228778839111328,1.390816569328308,1.3911932706832886,1.3984161615371704,1.3976309299468994,1.3973710536956787,1.4142037630081177,1.4141978025436401,1.4141596555709839,1.414226770401001,1.4141746759414673,1.4142296314239502,1.4141777753829956,1.4142025709152222,1.4141967296600342,1.4142098426818848,1.4141767024993896,1.4142966270446777,1.4143102169036865,1.4143065214157104,1.414319634437561,1.414329171180725,1.4143314361572266,1.4143279790878296,1.4143376350402832,1.414317011833191,1.2751208543777466,1.2846847772598267,1.1587233543395996,1.1613454818725586,1.161368489265442,0.9568553566932678,0.9547014236450195,1.1615924835205078,1.098583698272705,0.0,0.939510703086853,0.9399570822715759,0.9275019764900208],[0.8480154275894165,1.046993613243103,1.0258184671401978,1.3715683221817017,1.4081158638000488,1.4079307317733765,1.4081429243087769,1.408002257347107,1.2119475603103638,1.2136726379394531,1.2165400981903076,1.2222894430160522,1.3181918859481812,1.3194199800491333,1.363466501235962,1.3616666793823242,1.3616949319839478,1.4135104417800903,1.4134901762008667,1.4133481979370117,1.4133058786392212,1.41313898563385,1.4135046005249023,1.4028475284576416,1.412929892539978,1.4131171703338623,1.4130476713180542,1.4133384227752686,1.4141881465911865,1.4142279624938965,1.4141626358032227,1.414193034172058,1.4141937494277954,1.4141947031021118,1.4141672849655151,1.4141991138458252,1.4141819477081299,1.3795751333236694,1.383604884147644,1.355413794517517,1.35446298122406,1.3559726476669312,0.780988335609436,0.7793720364570618,1.0805491209030151,1.0710586309432983,0.939510703086853,0.0,0.034887343645095825,0.04481692984700203],[0.8451475501060486,1.0439462661743164,1.0230072736740112,1.3726929426193237,1.4084333181381226,1.4083608388900757,1.4083454608917236,1.4082915782928467,1.2120152711868286,1.2135670185089111,1.2160533666610718,1.2214280366897583,1.3198996782302856,1.3213294744491577,1.3663452863693237,1.3643587827682495,1.3641892671585083,1.413666844367981,1.4136289358139038,1.413507103919983,1.4134317636489868,1.413257122039795,1.413614273071289,1.404286503791809,1.4130536317825317,1.4132360219955444,1.413124442100525,1.413395643234253,1.4141839742660522,1.4142245054244995,1.4142600297927856,1.4142929315567017,1.4142937660217285,1.4142632484436035,1.414290428161621,1.4143216609954834,1.4143050909042358,1.3794139623641968,1.3834816217422485,1.35573148727417,1.3546898365020752,1.3562877178192139,0.7900177836418152,0.787280797958374,1.0834871530532837,1.07500159740448,0.9399570822715759,0.03488563746213913,0.0,0.04063309729099274],[0.8421363234519958,1.037471055984497,1.0169252157211304,1.3719068765640259,1.4086304903030396,1.408546805381775,1.4085944890975952,1.4085261821746826,1.2112935781478882,1.2126675844192505,1.2152236700057983,1.220473289489746,1.319712519645691,1.3211005926132202,1.3659191131591797,1.3638967275619507,1.3637065887451172,1.4136269092559814,1.4135981798171997,1.4134618043899536,1.4133952856063843,1.4132425785064697,1.413586139678955,1.4042465686798096,1.4130374193191528,1.4131903648376465,1.4131675958633423,1.4133821725845337,1.414189338684082,1.4142292737960815,1.4142177104949951,1.4142495393753052,1.4142515659332275,1.4142323732376099,1.4142156839370728,1.4142483472824097,1.414229393005371,1.3796879053115845,1.3839327096939087,1.3552627563476562,1.3540829420089722,1.3560367822647095,0.7839375734329224,0.7813717722892761,1.07864511013031,1.0708459615707397,0.9275019764900208,0.04481692984700203,0.04063309729099274,0.0]]"  # type: ignore
    x = eval(x)
    return torch.tensor(x, dtype=torch.float)
# fmt: on
