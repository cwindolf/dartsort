import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from .pyks_ccg import ccg_metrics, ccg
from . import waveform_utils, cluster_viz_index


def ccg_outliers(
    spike_train,
    outlier_scores,
    unit,
    contam_ratio_threshold=0.2,
    contam_alpha=0.05,
    nbins=50,
    tbin=30,
    verbose=False,
):
    in_unit = np.flatnonzero(spike_train[:, 1] == unit)
    if not in_unit.size:
        return None, [], [], [], []

    unit_times = spike_train[in_unit, 0]
    unit_scores = outlier_scores[in_unit]

    # scores in increasing order
    scores = np.unique(unit_scores)
    contam_ratios = np.full_like(scores, -1)
    p_values = np.full_like(scores, -1)

    for j, score in enumerate(scores[::-1]):
        inliers = unit_scores >= score
        times = unit_times[inliers]
        contam_ratio, p_value = ccg_metrics(times, times, nbins, tbin)
        contam_ratios[-1 - j] = contam_ratio
        p_values[-1 - j] = p_value
        # print(f"{score=}, {contam_ratio=}, {p_value=}", end="... ")

        p_good = p_value < contam_alpha
        contam_ok = contam_ratio < contam_ratio_threshold

        if p_good and not contam_ok:
            break

    if verbose:
        print(
            f"kept {100*inliers.mean():0.1f}% of the {in_unit.size} spikes in unit {unit}"
        )
        print(f"{score=}, {unit_scores.max()=}")

    return score, in_unit[inliers], in_unit[~inliers], contam_ratios, p_values


def outlier_viz(
    score,
    inliers,
    outliers,
    contam_ratios,
    p_values,
    spike_train,
    outlier_scores,
    unit,
    cleaned_waveforms=None,
    cleaned_tpca_projs=None,
    cleaned_tpca=None,
    maxchans=None,
    residual_path=None,
    geom=None,
    channel_index=None,
    ms_frames=30,
    metric_name="outlier scores",
):
    in_unit = spike_train[:, 1] == unit

    fig, ((aa, ab, ae), (ac, ad, af), (ax, ay, az)) = plt.subplots(
        nrows=3, ncols=3, figsize=(14, 14)
    )
    aa.hist(outlier_scores[in_unit], bins=64)
    aa.axvline(score, color="k")

    ab.scatter(
        np.unique(outlier_scores[in_unit])[contam_ratios > -1],
        contam_ratios[contam_ratios > -1],
        s=1,
    )
    ab.set_xlabel(metric_name)
    ab.set_ylabel("contam ratio")
    ab.axhline(0.2, color="k")

    ae.scatter(
        np.unique(outlier_scores[in_unit])[contam_ratios > -1],
        p_values[contam_ratios > -1],
        s=1,
    )
    ae.set_xlabel(metric_name)
    ae.set_ylabel("p val")
    ae.axhline(0.05, color="k")

    times_all = spike_train[in_unit, 0]
    isis = np.diff(times_all)
    isis_bin = isis / (ms_frames / 2)
    ac.hist(isis_bin, bins=np.arange(100) / 2)
    ac.set_xlabel("all isi (ms)")

    times = spike_train[inliers, 0]
    isis = np.diff(times)
    isis_bin = isis / (ms_frames / 2)
    ad.hist(isis_bin, bins=np.arange(100) / 2)
    ad.set_xlabel("inlier isi (ms)")

    nbins = 50
    ccg_all = ccg(times_all, times_all, nbins, ms_frames)
    ccg_in = ccg(times, times, nbins, ms_frames)
    ccg_all[nbins] = ccg_in[nbins] = 0

    ax.bar(np.arange(nbins * 2 + 1) - nbins, ccg_all)
    ax.set_xlabel("isi (ms)")
    ax.set_ylabel("all auto-ccg count")
    ay.bar(np.arange(nbins * 2 + 1) - nbins, ccg_in)
    ay.set_xlabel("isi (ms)")
    ay.set_ylabel("inlier auto-ccg count")

    fig.tight_layout()
    fig.suptitle(f"{in_unit.sum()} spikes, {len(inliers)} inliers")

    rg = np.random.default_rng(0)
    if cleaned_waveforms is not None or (
        cleaned_tpca_projs is not None and cleaned_tpca is not None
    ):
        nchans_plot = 4
        plotci_subset = waveform_utils.channel_index_subset(
            geom,
            channel_index,
            n_channels=nchans_plot,
        )
        plotci = np.full((len(geom), nchans_plot), len(geom))
        for c in range(len(geom)):
            plotci[c, : plotci_subset[c].sum()] = channel_index[
                c, plotci_subset[c]
            ]

        if outliers.size:
            outlier_choices = rg.choice(
                outliers, size=min(250, outliers.size), replace=False
            )
            outlier_choices.sort()
            if cleaned_waveforms is not None:
                cwfs = cleaned_waveforms[outlier_choices]
            else:
                cwfs = cleaned_tpca.inverse_transform(
                    cleaned_tpca_projs[outlier_choices],
                    maxchans[outlier_choices],
                    channel_index,
                )
            outlier_wfs = waveform_utils.get_channel_subset(
                cwfs,
                maxchans[outlier_choices],
                plotci_subset,
            )

            cluster_viz_index.pgeom(
                outlier_wfs,
                maxchans[outlier_choices],
                plotci,
                geom,
                color="r",
                # label="outliers",
                ax=af,
                lw=1,
                alpha=0.25,
            )

        if inliers.size:
            inlier_choices = rg.choice(
                inliers, size=min(250, inliers.size), replace=False
            )
            inlier_choices.sort()
            if cleaned_waveforms is not None:
                cwfs = cleaned_waveforms[inlier_choices]
            else:
                cwfs = cleaned_tpca.inverse_transform(
                    cleaned_tpca_projs[inlier_choices],
                    maxchans[inlier_choices],
                    channel_index,
                )
            inlier_wfs = waveform_utils.get_channel_subset(
                cwfs,
                maxchans[inlier_choices],
                plotci_subset,
            )
            cluster_viz_index.pgeom(
                inlier_wfs,
                maxchans[inlier_choices],
                plotci,
                geom,
                color="b",
                # label="inliers",
                ax=az,
                lw=1,
                alpha=0.25,
            )
        af.axis("off")
        af.set_title("cleaned wfs, outliers")
        az.axis("off")
        az.set_title("cleaned wfs, inliers")

    return fig


def runs_to_ranges(x, one_more=False):
    ranges = []
    cur = x[0]
    b = x[0]
    for a, b in zip(x, x[1:]):
        assert b > a
        if b - a == 1:
            continue
        else:
            ranges.append(range(cur, a + 1 + one_more))
            cur = b
    ranges.append(range(cur, b + 1 + one_more))
    return ranges


def enforce_refractory_in_run(
    run_times,
    run_scores,
    min_dt_frames,
):
    # try to keep as many as we can
    n_spikes = len(run_times)

    for n in range(n_spikes - 1, -1, -1):
        best_comb = None
        best_score = np.inf
        for inds in map(np.array, combinations(range(n_spikes), n)):
            if (n == 1) or (np.diff(run_times[inds]) >= min_dt_frames).all():
                score = run_scores[inds].sum()
                if score < best_score:
                    best_score = score
                    best_comb = inds
        if best_comb is not None:
            return np.setdiff1d(np.arange(n_spikes), best_comb)


def enforce_refractory_by_score(
    in_unit,
    times,
    scores,
    min_dt_frames=10,
):
    if not in_unit.size:
        return in_unit

    times_u = times[in_unit]
    scores_u = scores[in_unit]

    # i in runs if t[i+1] too close to t[i]
    viol = np.flatnonzero(np.diff(times_u) < min_dt_frames)
    if not viol.size:
        return np.arange(len(in_unit))
    runs = runs_to_ranges(viol, one_more=True)

    delete_ix = []
    for run in runs:
        run_arr = np.array(list(run))
        run_times = times_u[run]
        run_scores = scores_u[run]
        violators = enforce_refractory_in_run(
            run_times, run_scores, min_dt_frames
        )
        delete_ix += list(run_arr[violators])

    keeps = np.setdiff1d(np.arange(len(times_u)), delete_ix)

    return keeps


def outlier_viz_mini(
    scores,
    inliers,
    threshold,
    spike_train,
    unit,
    maxchans=None,
    cleaned_waveforms=None,
    cleaned_tpca_projs=None,
    cleaned_tpca=None,
    geom=None,
    channel_index=None,
    keep_lt=True,
    ms_frames=30,
):
    in_unit = np.flatnonzero(spike_train[:, 1] == unit)
    # if keep_lt:
    #     inliers = scores[in_unit] <= threshold
    # else:
    #     inliers = scores[in_unit] >= threshold
    outliers = ~inliers

    fig, axes = plt.subplot_mosaic(
        "aa\n" "bc\n" "de\n",
        figsize=(5, 8),
    )

    n, bins, _ = axes["a"].hist(
        scores[in_unit], bins=64, label="All outlier scores"
    )
    axes["a"].hist(
        scores[in_unit][inliers], histtype="step", bins=bins, label="Inliers"
    )
    axes["a"].axvline(threshold, color="k", label="Cutoff")
    axes["a"].legend(title=f"kept {100*inliers.mean():.1f}% of {inliers.size}")
    axes["a"].semilogy()

    times_all = spike_train[in_unit, 0]
    isis = np.diff(times_all)
    isis_bin = isis / ms_frames
    axes["b"].hist(isis_bin, bins=np.arange(50))
    axes["b"].set_xlabel("isi (ms)")

    times = times_all[inliers]
    isis = np.diff(times)
    isis_bin = isis / ms_frames
    axes["b"].hist(isis_bin, histtype="step", bins=np.arange(50))
    axes["b"].set_ylabel("isi frequency")

    nbins = 50
    ccg_all = ccg(times_all, times_all, nbins, ms_frames)
    ccg_in = ccg(times, times, nbins, ms_frames)
    ccg_all[nbins] = ccg_in[nbins] = 0

    axes["c"].bar(
        np.arange(nbins * 2 + 1) - nbins, ccg_all, width=1, ec="none"
    )
    axes["c"].set_xlabel("isi (ms)")
    axes["c"].set_ylabel("autocorrelogram count")
    axes["c"].bar(np.arange(nbins * 2 + 1) - nbins, ccg_in, width=1, ec="none")
    # axes["c"].set_xlabel("isi (ms)")
    # axes["c"].set_ylabel("inlier auto-ccg count")

    rg = np.random.default_rng(0)
    if cleaned_waveforms is not None or (
        cleaned_tpca_projs is not None and cleaned_tpca is not None
    ):
        nchans_plot = 4
        plotci_subset = waveform_utils.channel_index_subset(
            geom,
            channel_index,
            n_channels=nchans_plot,
        )
        plotci = np.full((len(geom), nchans_plot), len(geom))
        for c in range(len(geom)):
            plotci[c, : plotci_subset[c].sum()] = channel_index[
                c, plotci_subset[c]
            ]

        if outliers.any():
            outlier_choices = rg.choice(
                np.flatnonzero(outliers),
                size=min(250, outliers.sum()),
                replace=False,
            )
            outlier_choices.sort()
            if cleaned_waveforms is not None:
                cwfs = cleaned_waveforms[in_unit[outlier_choices]]
            else:
                cwfs = cleaned_tpca.inverse_transform(
                    cleaned_tpca_projs[in_unit[outlier_choices]],
                    maxchans[in_unit[outlier_choices]],
                    channel_index,
                )
            outlier_wfs = waveform_utils.get_channel_subset(
                cwfs,
                maxchans[in_unit[outlier_choices]],
                plotci_subset,
            )

            cluster_viz_index.pgeom(
                outlier_wfs,
                maxchans[in_unit[outlier_choices]],
                plotci,
                geom,
                color="b",
                # label="outliers",
                ax=axes["d"],
                lw=1,
                alpha=0.25,
            )

        if inliers.any():
            inlier_choices = rg.choice(
                np.flatnonzero(inliers),
                size=min(250, inliers.sum()),
                replace=False,
            )
            inlier_choices.sort()
            if cleaned_waveforms is not None:
                cwfs = cleaned_waveforms[in_unit[inlier_choices]]
            else:
                cwfs = cleaned_tpca.inverse_transform(
                    cleaned_tpca_projs[in_unit[inlier_choices]],
                    maxchans[in_unit[inlier_choices]],
                    channel_index,
                )
            inlier_wfs = waveform_utils.get_channel_subset(
                cwfs,
                maxchans[in_unit[inlier_choices]],
                plotci_subset,
            )
            cluster_viz_index.pgeom(
                inlier_wfs,
                maxchans[in_unit[inlier_choices]],
                plotci,
                geom,
                color="orange",
                # label="inliers",
                ax=axes["e"],
                lw=1,
                alpha=0.25,
            )
        axes["d"].axis("off")
        axes["d"].set_title("cleaned wfs, outliers")
        axes["e"].axis("off")
        axes["e"].set_title("cleaned wfs, inliers")

    return fig, axes
