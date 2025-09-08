import numpy as np
import pytest
import torch

import dartsort
from dartsort.cluster.refine_util import initialize_gmm


TEST_N_EM_ITERS = 10


@pytest.mark.parametrize("final_split", ["train", "kept"])
# sorry for the cryptography
# negative means run a prepare_step loop
# positive means run TVI step by step and check after each
# falsy means run the tvi() function. 0 means with TEST_N_EM_ITERS, False means 1 iter.
@pytest.mark.parametrize("stepwise", [-1, 1, False, 0, TEST_N_EM_ITERS, -3])
@pytest.mark.parametrize("tmm_initialization", ["nearest", "topk"])
@pytest.mark.parametrize("initial_e_step", [False, True])
@pytest.mark.parametrize("val_proportion", [0.0, 0.2])
@pytest.mark.parametrize("distance_metric", ["kl", "cosine"])
@pytest.mark.parametrize("drift", ["y", "n"])
def test_topk_candidates(
    mini_simulations,
    drift,
    distance_metric,
    val_proportion,
    initial_e_step,
    tmm_initialization,
    stepwise,
    final_split,
):
    # test the topk constrained candidates strategy
    # test step by step and after running a large batch of steps
    sim = mini_simulations[f"drift{drift}_szmini"]

    refinement_cfg = dartsort.RefinementConfig(
        signal_rank=0,
        search_type="topk",
        distance_metric=distance_metric,
        em_converged_atol=0.0,
        em_converged_churn=0.0,
        em_converged_prop=0.0,
        val_proportion=val_proportion,
        n_em_iters=TEST_N_EM_ITERS if stepwise is False else 1,
    )
    gmm = initialize_gmm(sim["sorting"], refinement_cfg, sim["motion_est"])

    # initialization of parameters and likelihoods
    gmm.m_step()
    lls = gmm.log_likelihoods()
    gmm.update_proportions(lls)
    if initial_e_step:
        _, _, lls, _ = gmm.e_step()
    else:
        lls = gmm.log_likelihoods(split="kept")

    # initialize tmm
    gmm.tvi(lls=lls, initialize_tmm_only=True, initialization=tmm_initialization)
    check_tmm_invariants(
        gmm,
        including_labels=initial_e_step,
        including_top=tmm_initialization != "nearest",
        reinit_neighborhoods=True,
        including_search_explore=False,
    )

    if stepwise < 0:
        for j in range(-stepwise):
            cnew, cnu = gmm.tmm.prepare_step()
            assert not cnu
            check_tmm_invariants(gmm, including_labels=initial_e_step, d=True)
        return

    for j in range(stepwise):
        res = gmm.tmm.step(hard_label=True)
        (kept,) = (res["labels"] >= 0).nonzero(as_tuple=True)
        assert torch.equal(res["labels"][kept], gmm.tmm.candidates.candidates[kept, 0])
        check_tmm_invariants(
            gmm, including_labels=False, including_adjacency=False, d=True
        )

    if not stepwise:
        gmm.tvi(lls=lls, final_split=final_split)
        check_tmm_invariants(
            gmm,
            d=True,
            including_adjacency=False,
            including_labels=final_split == "train",
        )

    cnew, cnu = gmm.tmm.prepare_step()
    assert not cnu
    # last check for adjacency
    check_tmm_invariants(
        gmm, including_labels=not stepwise and final_split == "train", d=True
    )


def check_tmm_invariants(
    gmm,
    including_labels=True,
    including_top=True,
    including_adjacency=True,
    including_search_explore=True,
    reinit_neighborhoods=False,
    d=False,
):
    no_search_explore = not including_search_explore
    no_top = not including_top

    p = print if d else lambda x: None
    p("/ " * 40)

    tmm = gmm.tmm
    train_ixs = gmm.data.split_indices["train"]
    train_neighbs = gmm.data._train_extract_neighborhoods
    candidates = gmm.tmm.candidates.candidates
    train_labels = gmm.labels[train_ixs]
    n_candidates = tmm.n_candidates
    assert tmm.candidates.n_candidates == n_candidates
    neighb_ids = tmm.candidates.neighborhood_ids
    assert torch.equal(neighb_ids, train_neighbs.neighborhood_ids)
    n_units = len(tmm.means)

    # test invariants:
    #  a. same tvi() and candidates[:,0] labels after every iteration, modulo noise assigns
    #  b. candidates 1:n_candidates are adjacent
    #  c. search candidates adjacency
    #  d. explore candidates adjacency
    #  e. empirical adjacency of all {1:top, search, explore} spikes'
    #     neighborhoods to the neighborhoods of candidates[:,0] spikes
    #     for all units
    #  f. equality of neighborhoods of candidates[:,0] spikes with the LUT table
    #  g. equality of neighborhoods of candidates[:,0] spikes with the unit_neighb_adj thing
    #  h. uniqueness of candidates except -1s
    #  i. top candidate is >=0
    # above, adjacent means that the neighborhoods overlap either fully or by at least
    # some fraction where appropriate. it is determined by the adjacency() method of
    # spikeneighborhoods objects as computed below.
    # impl checklist:
    #  a [x] b [x] c [x] d [x] e [x] f [x] g [x] h [x] i [x]

    # a. rest are below.
    if including_labels:
        is_noise = train_labels == -1
        assert torch.logical_or(train_labels == candidates[:, 0], is_noise).all()
        del is_noise

    # -- -- -- precompute/grab for below asserts  -- -- --
    #  - neighborhood adjacencies
    #  - unit-neighborhood empirical adjacencies
    #  - LUT unit-neighborhood adjacencies
    #  - un_adj unit-neighborhood adjacencies
    #  - unique group candidate unit ID sets by unit for groups:
    #    - 1:n_candidates
    #    - search
    #    - explore
    #  - unique group candidate neighborhood ID sets for same.

    # store channels by neighb id
    neighb_chans = []
    for nid in range(train_neighbs.n_neighborhoods):
        neighb_chans.append(train_neighbs.neighborhood_channels(nid))

    # neighb-neighb adjacency 2 ways: partial and full overlap
    train_neighborhoods = gmm.data._train_extract_neighborhoods
    full_adjacency = train_neighborhoods.adjacency(overlap=1.0)
    some_adjacency = train_neighborhoods.adjacency(
        overlap=tmm.candidates.neighborhood_adjacency_overlap
    )
    assert np.array_equal(
        tmm.candidates.neighb_adjacency, some_adjacency.numpy(force=True)
    )
    assert tmm.candidates._initialized  # don't run this before initializing candidates
    assert (some_adjacency >= full_adjacency).all()

    # this is a unit-neighb adjacency, see below, getting now to read a runtime shape
    if reinit_neighborhoods:
        un_adj = tmm.candidates.reinit_neighborhoods(candidates[:, 0], neighb_ids)
        tmm.candidates.ensure_adjacent(
            candidates[:, : tmm.n_candidates], neighb_ids, un_adj
        )
    else:
        assert tmm.candidates.un_adj is not None
        un_adj = tmm.candidates.un_adj
    adj_uu, adj_nn, unit_neighb_adj, un_adj_lut = un_adj

    # unit-(neighb,unit) adjacency:
    # - target search sets
    unit_search_neighbors = tmm.candidates.search_sets(tmm.divergences, un_adj=un_adj)

    # best, top, search, explore sets
    n_search = unit_search_neighbors.shape[1]
    assert 0 < n_search <= tmm.n_search
    best = candidates[:, 0]
    assert best.ndim == 1
    assert len(best) == len(train_ixs)
    assert (best >= 0).all()
    top = candidates[:, 1 : tmm.n_candidates]
    assert 0 < top.shape[1] == tmm.n_candidates - 1
    search = candidates[
        :, tmm.n_candidates : tmm.n_candidates + n_search * tmm.n_candidates
    ]
    assert search.shape[1] == n_search * tmm.n_candidates
    explore = candidates[
        :,
        (tmm.n_candidates + n_search * tmm.n_candidates) : (
            tmm.n_candidates + n_search * tmm.n_candidates + tmm.n_explore
        ),
    ]
    assert explore.shape[1] == tmm.n_explore

    # unit-neighb adjacency several ways:
    #  - tmm unit_neighborhood_counts (used by neighb_lut())
    #    this is the empirical count of top units in each neighborhood
    #  - tmm un_adj (used by search_sets(); computed by reinit_neighborhoods())
    #    this can be a superset of above if: search_neighborhood_steps > 0 (not default)
    #    or if there are neighborhoods which are subsets of unions of other neighborhoods
    #    so the condition to check is: either this equals the above, or it is a superset
    #    and there is that subset relation
    #    (this is ^^^ right above, before best,top, search, explore)
    #  - set of channels occupied by neighb ids in each units best sets
    #  - empirical sets of neighborhood IDs in each top unit
    #  - empirical sets of neighborhood IDs in gmm label unit
    #  -                   """                      search """
    #  -                   """                      explore """
    unc = torch.as_tensor(tmm.candidates.unit_neighborhood_counts, dtype=torch.float)
    assert unc is not None  # reinit_neighborhoods() has been called
    assert unc.shape == (n_units + 1, train_neighbs.n_neighborhoods)
    assert (unc > 0).any()
    best_neighb_ids = []
    top_neighb_ids = []
    search_neighb_ids = []
    explore_neighb_ids = []
    for j in range(n_units):
        nids_j = neighb_ids[candidates[:, 0] == j].unique()
        best_neighb_ids.append(nids_j)
        if including_labels:
            assert torch.equal(neighb_ids[train_labels == j].unique(), nids_j)

        top_neighb_ids.append(neighb_ids[(top == j).any(1)].unique())
        search_neighb_ids.append(neighb_ids[(search == j).any(1)].unique())
        explore_neighb_ids.append(neighb_ids[(explore == j).any(1)].unique())

    # simple unit-unit adjacency
    # this is: they have at least one fully-overlapping neighborhood in common
    uu_adj_full = (unc @ full_adjacency @ unc.T) > 0
    # this is: they have at least one partially-overlapping neighborhood in common
    uu_adj_some = (unc @ some_adjacency @ unc.T) > 0
    assert (uu_adj_some >= uu_adj_full).all()
    # empirical unit co-occurrences
    uu_top = []
    uu_search = []
    uu_explore = []
    for j in range(n_units):
        (in_j,) = (candidates[:, 0] == j).nonzero(as_tuple=True)

        topj = top[in_j].unique()
        topj = topj[topj >= 0]
        uu_top.append(topj)

        searchj = search[in_j].unique()
        searchj = searchj[searchj >= 0]
        uu_search.append(searchj)

        explorej = explore[in_j].unique()
        explorej = explorej[explorej >= 0]
        uu_explore.append(explorej)

    # empirical unit-neighb co-occurences
    # also, channel sets for best+top
    top_channel_sets = []
    best_channel_sets = []
    rest_neighb_ids = []
    rest_channel_sets = []
    nu_search = []
    for j in range(n_units):
        nu_searchj = []
        top_chansj = []
        rest_chansj = []
        rest_nidsj = []
        for k in range(candidates.shape[1]):
            (in_jk,) = (candidates[:, k] == j).nonzero(as_tuple=True)
            if not in_jk.numel():
                continue

            nids_jk = neighb_ids[in_jk]
            nids_jk_unique = nids_jk.unique()
            (chans_jk,) = (
                train_neighborhoods.indicators[:, nids_jk_unique]
                .sum(1)
                .nonzero(as_tuple=True)
            )
            if k == 0:
                best_channel_sets.append(chans_jk)

            if k >= tmm.n_candidates:
                rest_nidsj.append(nids_jk_unique)
                rest_chansj.append(chans_jk)
                continue

            # top only below
            top_chansj.append(chans_jk)
            sub_search_range = slice(k * n_search, (k + 1) * n_search)
            usearchjk = search[in_jk, sub_search_range]
            nu_searchjk = [
                torch.stack(
                    (nids_jk[usearchjk_row >= 0], usearchjk_row[usearchjk_row >= 0]),
                    dim=1,
                ).unique(dim=0)
                for usearchjk_row in usearchjk.T
            ]
            nu_searchj.extend(nu_searchjk)
        nu_searchj = torch.concatenate(nu_searchj, dim=0).unique(dim=0)
        nu_search.append(nu_searchj)
        top_channel_sets.append(torch.concatenate(top_chansj).unique())

        if len(rest_chansj):
            rest_channel_sets.append(torch.concatenate(rest_chansj).unique())
            rest_neighb_ids.append(torch.concatenate(rest_nidsj).unique())
        else:
            rest_channel_sets.append(torch.tensor([], dtype=torch.long))
            rest_neighb_ids.append(torch.tensor([], dtype=torch.long))

    # -- -- -- -- -- -- check invariants -- -- -- -- -- --
    # f.i unc is best+top
    # immediately after reinit_neighborhoods, it is only best. top gets
    # added in after candidates are proposed.
    for j in range(n_units):
        (uncj,) = unc[j].nonzero(as_tuple=True)
        if reinit_neighborhoods:
            assert torch.equal(uncj, best_neighb_ids[j])
        else:
            emp_nj = torch.concatenate([best_neighb_ids[j], top_neighb_ids[j]]).unique()
            # subset, because uncj includes search+explore
            assert torch.isin(emp_nj, uncj).all()
            emp_nj_full = torch.concatenate([emp_nj, rest_neighb_ids[j]]).unique()
            assert torch.equal(emp_nj_full, uncj)

    # f.ii chan sets match
    # same note as f.i
    for j in range(n_units):
        inds_j = train_neighborhoods.indicators[:, *unc[j].nonzero(as_tuple=True)]
        (chans_j,) = inds_j.sum(dim=1).nonzero(as_tuple=True)
        if reinit_neighborhoods:
            assert torch.equal(best_channel_sets[j], chans_j)
        else:
            emp_chans = torch.concatenate(
                [best_channel_sets[j], top_channel_sets[j]]
            ).unique()
            assert torch.isin(emp_chans, chans_j).all()
            emp_chans_full = torch.concatenate(
                [emp_chans, rest_channel_sets[j]]
            ).unique()
            assert torch.equal(emp_chans_full, chans_j)

    # g.
    if including_adjacency and including_search_explore:
        for i in range(len(adj_uu)):
            # from adj lut
            uu = adj_uu[i]
            nn = adj_nn[i]
            assert un_adj_lut[uu, nn] == i
            target_uunn = unit_search_neighbors[i]
            target_uunn = target_uunn[target_uunn >= 0]

            # from nu_search (empirical)
            nsearch_uu, usearch_uu = nu_search[uu].T
            (nn_ix,) = (nn == nsearch_uu).nonzero(as_tuple=True)
            assert len(nn_ix) >= 0
            found = usearch_uu[nn_ix]

            # assert empirical is subset of target
            # why subset? if searches were present in top, they are -1d out
            if len(nn_ix) > 0:
                assert torch.isin(found, target_uunn).all()

    # b.
    if including_top:
        for j in range(n_units):
            assert uu_adj_some[j][uu_top[j]].all()
            assert uu_adj_full[j][uu_top[j]].all()

    # c.
    if including_search_explore:
        for j in range(n_units):
            assert uu_adj_some[j][uu_search[j]].all()
            assert uu_adj_full[j][uu_search[j]].all()

    # d.
    if including_search_explore:
        for j in range(n_units):
            assert uu_adj_some[j][uu_explore[j]].all()
            assert uu_adj_full[j][uu_explore[j]].all()

    # e preface
    # check neighborhood subsets
    for j in range(train_neighborhoods.n_neighborhoods):
        for jj in np.flatnonzero(tmm.candidates.neighborhood_subset[j]):
            assert (
                train_neighborhoods.indicators[jj] <= train_neighborhoods.indicators[j]
            ).all()

    # e.i,
    # this would not be true, but that's why ensure_adjacent() is called in
    # propose_candidates().
    # ensure_adjacent() enforces that if unit U is a top unit for spike j, then
    # neighb_ids[j] is in the "best" neighb ids set for unit U. OR, it is a subset
    # of a neighborhood ID which is in the best neighb ids set.
    if including_adjacency and including_top:
        for j in range(n_units):
            bnids_j = np.atleast_1d(best_neighb_ids[j])
            best_or_subset_j = tmm.candidates.neighborhood_subset[bnids_j].sum(0)
            best_or_subset_j = torch.as_tensor(np.flatnonzero(best_or_subset_j))
            assert torch.isin(top_neighb_ids[j], best_or_subset_j).all()
            for nid_j in top_neighb_ids[j]:
                assert torch.isin(neighb_chans[nid_j], best_channel_sets[j]).all()

    # e.ii,
    if including_adjacency and including_search_explore:
        for j in range(n_units):
            for nid_j in search_neighb_ids[j]:
                assert torch.isin(neighb_chans[nid_j], best_channel_sets[j]).all()

    # e.iii,
    if including_adjacency and including_search_explore:
        for j in range(n_units):
            for nid_j in explore_neighb_ids[j]:
                assert torch.isin(neighb_chans[nid_j], best_channel_sets[j]).all()

    # h.
    c_sorted = candidates[:, :n_candidates].sort(dim=1).values
    c_diff = c_sorted.diff(dim=1)
    c_valid = torch.logical_or(c_sorted[:, 1:] == -1, c_diff > 0)
    assert c_valid.all()
    del c_sorted, c_diff, c_valid

    # i.
    assert torch.all(candidates[:, 0] >= 0)
