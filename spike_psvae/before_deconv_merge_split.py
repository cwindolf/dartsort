import numpy as np
import h5py


try:
    from isosplit import isosplit
except ImportError:
    pass
from functools import wraps
from hdbscan import HDBSCAN
from hdbscan.robust_single_linkage_ import RobustSingleLinkage
from inspect import getfullargspec
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import OPTICS, MeanShift
from tqdm.auto import tqdm

from .multiprocessing_utils import get_pool
from .waveform_utils import get_channel_subset
from .isocut5 import isocut5, isosplit1d
from .chunk_features import TPCA
from .snr_templates import get_raw_template_single
from .deconv_resid_merge import calc_resid_matrix
from . import relocation


# -- split step


# starts with helper functions
# this is a parallelism helper: this function will be run
# before the split steps below, and stores helpful stuff
# for them to use.

# the main function is `split_clusters` below


class H5Extractor:
    def __init__(
        self,
        h5_path,
        raw_data_bin,
        log_c=5,
        feature_scales=(1, 1, 50),
        waveforms_kind="cleaned",
        loc_feature=None,
    ):
        self.raw_data_bin = raw_data_bin

        # get the clustering features in memory
        h5 = h5py.File(h5_path)
        if loc_feature is None:
            str_loc = "localizations"
        else:
            str_loc = "localizations{}".format(loc_feature)
        self.x = x = h5[str_loc][:, 0]
        self.z = z = h5["z_reg"][:]
        self.maxptp = maxptp = h5["maxptps"][:]
        self.spike_times, self.max_channels = h5["spike_index"][:].T
        self.channel_index = h5["channel_index"][:]
        self.n_channels = self.channel_index.shape[0]
        self.features = np.c_[x, z, np.log(log_c + maxptp)]
        self.features *= feature_scales
        self.geom = h5["geom"][:]

        # things we will need if doing relocated clustering
        self.y, self.z_abs, self.alpha = h5[str_loc][:, 1:4].T
        self.log_c = log_c
        self.feature_scales = feature_scales

        # refrence to waveform tpca embeddings in h5 file
        # we could load them in memory here if that seems helpful
        self.tpca_projs = h5[f"{waveforms_kind}_tpca_projs"]

        # load sklearn PCA object from the h5 so that split steps can
        # reconstruct waveforms from the tpca projections
        tpca_feat = TPCA(
            self.tpca_projs.shape[1],
            self.channel_index,
            waveforms_kind,
        )
        tpca_feat.from_h5(h5)
        self.tpca = tpca_feat
        self.T = tpca_feat.T


def herding_split(
    in_unit,
    min_size_split=25,
    n_pca_features=2,
    clusterer="hdbscan",
    hdbscan_kwargs=dict(min_cluster_size=25, min_samples=25),
    dpgmm_kwargs=dict(
        n_components=5,
        weight_concentration_prior_type="dirichlet_distribution",
        max_iter=20000,
    ),
    meanshift_kwargs=dict(
        bandwidth=5,
        max_iter=2000,
    ),
    chans_method="distance",
    chans_amplitude_n_channels=5,
    chans_distance_radius=75,
    relocated=False,
    extractor=None,
    use_features=True,
    return_features=False,
):
    n_spikes = in_unit.size

    # bail early if too small
    if n_spikes < min_size_split:
        return False, None, None

    # we'll store new labels here, including -1 for triage
    new_labels = np.zeros(in_unit.shape, dtype=int)

    # pick the subset of channels to use
    template = get_raw_template_single(
        extractor.spike_times[in_unit],
        extractor.raw_data_bin,
        extractor.n_channels,
    )
    if chans_method == "amplitude":
        which_chans = (-template.ptp(0)).argsort()[:chans_amplitude_n_channels]
    elif chans_method == "distance":
        which_chans = np.flatnonzero(
            cdist(
                extractor.geom[template.ptp(0).argmax()][None], extractor.geom
            )
            < chans_distance_radius
        )
    else:
        raise ValueError("chans_method not in ('amplitude', 'distance')")

    # get pca projections on channel subset
    if relocated:
        unit_waveform_features, relocated_maxptps = get_relocated_wfs_on_channel_subset(
            in_unit,
            extractor.tpca_projs,
            extractor.max_channels,
            extractor.channel_index,
            which_chans,
            extractor.tpca,
            extractor.x,
            extractor.y,
            extractor.z_abs,
            extractor.z,
            extractor.alpha,
            extractor.geom,
            T=extractor.T,
        )
    else:
        unit_waveform_features = get_pca_projs_on_channel_subset(
            in_unit,
            extractor.tpca_projs,
            extractor.max_channels,
            extractor.channel_index,
            which_chans,
        )

    # some spikes may not exist on all these channels.
    # this should be exceedingly rare but everything that can
    # happen will. for now, let's triage them away
    too_far = np.isnan(unit_waveform_features).any(axis=(1, 2))
    new_labels[too_far] = -1
    kept = np.flatnonzero(new_labels >= 0)

    # bail if too small
    if kept.size < min_size_split and not return_features:
        return False, None, None

    # fit a pca projection to what we got
    pca_projs = PCA(n_pca_features, whiten=True).fit_transform(
        unit_waveform_features[kept].reshape(kept.size, -1)
    )
    del unit_waveform_features

    # create features for hdbscan, scaling pca projs to match
    # the current feature set
    if use_features:
        unit_features = extractor.features[in_unit[kept]]
        if relocated:
            unit_features[:, 2] = extractor.feature_scales[2] * np.log(extractor.log_c + relocated_maxptps[kept])
        pca_projs *= unit_features.std(axis=0).mean()
        unit_features = np.c_[unit_features, pca_projs]
    else:
        unit_features = pca_projs

    # if user requested the features we can get here while too small
    if kept.size < min_size_split:
        return False, None, None, unit_features

    # run hdbscan
    if clusterer == "hdbscan":
        clust = HDBSCAN(**hdbscan_kwargs)
        clust.fit(unit_features)
        new_labels[kept] = clust.labels_
    elif clusterer == "robustsinglelinkage":
        clust = RobustSingleLinkage()
        clust.fit(unit_features)
        new_labels[kept] = clust.labels_
    elif clusterer == "optics":
        clust = OPTICS(min_samples=10)
        clust.fit(unit_features)
        new_labels[kept] = clust.labels_
    elif clusterer == "meanshift":
        clust = MeanShift(**meanshift_kwargs)
        clust.fit(unit_features)
        new_labels[kept] = clust.predict(unit_features)
    elif clusterer == "isosplit":
        new_labels[kept] = isosplit(unit_features.T)
    elif clusterer == "dpgmm":
        clust = BayesianGaussianMixture(**dpgmm_kwargs)
        clust.fit(unit_features)
        labs = clust.predict(unit_features)
        u, c = np.unique(
            np.concatenate([np.arange(5), labs]), return_counts=True
        )
        c -= 1
        k = (c >= 25).sum()
        if k <= 1:
            return False, None, None
        clust2 = GaussianMixture(
            n_components=k,
            weights_init=clust.weights_[c >= 25]
            / clust.weights_[c >= 25].sum(),
            means_init=clust.means_[c >= 25],
            precisions_init=clust.precisions_[c >= 25],
        )
        clust2.fit(unit_features)
        new_labels[kept] = clust2.predict(unit_features)
    elif clusterer == "isosplit1d_gmm":
        # -- Yizi's method
        # guess the number of clusters
        k = max(
            np.unique(labels).size
            for labels, _ in map(isosplit1d, map(np.unique, unit_features.T))
        )
        if k <= 1:
            if unit_features:
                return False, None, None, unit_features
            return False, None, None
        # fit a GMM
        clust = GaussianMixture(n_components=k)
        clust.fit(unit_features)
        new_labels[kept] = clust.predict(unit_features)
    else:
        raise ValueError(f"{clusterer=} not understood.")

    is_split = np.setdiff1d(np.unique(new_labels), [-1]).size > 1
    if return_features:
        return is_split, new_labels, in_unit, unit_features
    return is_split, new_labels, in_unit


def maxchan_lda_split(
    in_unit,
    min_size_split=25,
    n_channels=5,
    threshold_diptest=1.0,
    hdbscan_kwargs=dict(min_cluster_size=25, min_samples=25),
    chans_method="distance",
    chans_distance_radius=75,
    extractor=None,
    relocated=False,
):
    n_spikes = in_unit.size

    # bail early if too small
    if n_spikes < min_size_split:
        return False, None, None

    # this step relies on there being multiple max channels
    # so let's bail if this is not the case
    unit_max_chans = extractor.max_channels[in_unit]
    if np.unique(unit_max_chans).size <= 1:
        return False, None, None

    # we'll store new labels here, including -1 for triage
    new_labels = np.zeros(in_unit.shape, dtype=int)

    # pick the subset of channels to use
    template = get_raw_template_single(
        extractor.spike_times[in_unit],
        extractor.raw_data_bin,
        extractor.n_channels,
    )
    if chans_method == "amplitude":
        which_chans = (-template.ptp(0)).argsort()[:n_channels]
    elif chans_method == "distance":
        which_chans = np.flatnonzero(
            cdist(
                extractor.geom[template.ptp(0).argmax()][None], extractor.geom
            )[0]
            < chans_distance_radius
        )
    else:
        raise ValueError("chans_method not in ('amplitude', 'distance')")

    # get pca projections on channel subset
    if relocated:
        unit_features, relocated_maxptps = get_relocated_wfs_on_channel_subset(
            in_unit,
            extractor.tpca_projs,
            extractor.max_channels,
            extractor.channel_index,
            which_chans,
            extractor.tpca,
            extractor.x,
            extractor.y,
            extractor.z_abs,
            extractor.z,
            extractor.alpha,
            extractor.geom,
            T=extractor.T,
        )
    else:
        unit_features = get_pca_projs_on_channel_subset(
            in_unit,
            extractor.tpca_projs,
            extractor.max_channels,
            extractor.channel_index,
            which_chans,
        )

    # some spikes may not exist on all these channels.
    # this should be exceedingly rare but everything that can
    # happen will. for now, let's triage them away
    too_far = np.isnan(unit_features).any(axis=(1, 2))
    new_labels[too_far] = -1
    kept = np.flatnonzero(new_labels >= 0)

    # bail if too small or too few mcs
    if kept.size < min_size_split:
        return False, None, None
    n_max_chans = np.unique(unit_max_chans[kept]).size
    if n_max_chans <= 1:
        return False, None, None

    # fit the lda model
    n_lda_components = min(n_max_chans - 1, 2)
    lda_projs = LDA(n_components=n_lda_components).fit_transform(
        unit_features[kept].reshape(kept.size, -1), unit_max_chans[kept]
    )
    del unit_features

    # perform the split with isocut if not enough dims for hdbscan
    if n_lda_components == 1:
        dipscore, cutpoint = isocut5(lda_projs.squeeze())
        is_split = dipscore > threshold_diptest
        if is_split:
            new_labels[kept] = lda_projs.squeeze() > cutpoint
    elif n_lda_components > 1:
        clust = HDBSCAN(**hdbscan_kwargs)
        clust.fit(lda_projs)
        new_labels[kept] = clust.labels_
        is_split = np.setdiff1d(np.unique(new_labels), [-1]).size > 1
    else:
        assert False

    return is_split, new_labels, in_unit


def ks_bimodal_pursuit_split(
    in_unit,
    unit_rank=3,
    top_pc_init=True,
    aucsplit=0.85,
    min_size_split=50,
    max_split_corr=0.9,
    min_amp_sim=0.2,
    min_split_prop=0.05,
    load_batch_size=512,
    extractor=None,
):
    """Adapted from PyKS"""
    tpca = extractor.tpca.tpca
    n_spikes = in_unit.size

    # bail early if too small
    if n_spikes < min_size_split:
        return False, None, None

    # load pca embeddings on the max channel
    unit_max_chans = extractor.max_channels[in_unit]
    ix0, unit_rel_max_chans = np.nonzero(
        unit_max_chans[:, None] == extractor.channel_index[unit_max_chans]
    )
    assert np.array_equal(ix0, np.arange(unit_max_chans.shape[0]))
    unit_features = np.empty(
        (in_unit.size, extractor.tpca_projs.shape[1]),
        dtype=extractor.tpca_projs.dtype,
    )
    for bs in range(0, in_unit.size, load_batch_size):
        be = min(in_unit.size, bs + load_batch_size)
        unit_features[bs:be] = extractor.tpca_projs[in_unit[bs:be]][
            np.arange(be - bs), :, unit_rel_max_chans[bs:be]
        ]

    if unit_rank < unit_features.shape[1]:
        unit_features = PCA(unit_rank).fit_transform(unit_features)

    if top_pc_init:
        # input should be centered so no problem with centered pca?
        w = PCA(1).fit(unit_features).components_.squeeze()
    else:
        # initialize with the mean of NOT drift-corrected trace
        w = unit_features.mean(axis=0)
        w /= np.linalg.norm(w)

    # initial projections of waveform PCs onto 1D vector
    x = unit_features @ w
    x_mean = x.mean()
    # initialize estimates of variance for the first
    # and second gaussian in the mixture of 1D gaussians
    s1 = x[x > x_mean].var()
    s2 = x[x < x_mean].var()
    # initialize the means as well
    mu1 = x[x > x_mean].mean()
    mu2 = x[x < x_mean].mean()
    # and the probability that a spike is assigned to the first Gaussian
    p = (x > x_mean).mean()

    # initialize matrix of log probabilities that each spike is assigned to the first
    # or second cluster
    logp = np.zeros((x.shape[0], 2), order="F")
    # do 50 pursuit iteration
    logP = np.zeros(50)  # used to monitor the cost function

    # TODO: move_to_config - maybe...
    for k in range(50):
        if min(s1, s2) < 1e-6:
            break

        # for each spike, estimate its probability to come from either Gaussian cluster
        logp[:, 0] = np.log(s1) / 2 - ((x - mu1) ** 2) / (2 * s1) + np.log(p)
        logp[:, 1] = (
            np.log(s2) / 2 - ((x - mu2) ** 2) / (2 * s2) + np.log(1 - p)
        )

        lMax = logp.max(axis=1)
        # subtract the max for floating point accuracy
        logp = logp - lMax[:, np.newaxis]
        rs = np.exp(logp)

        # get the normalizer and add back the max
        pval = np.log(np.sum(rs, axis=1)) + lMax
        # this is the cost function: we can monitor its increase
        logP[k] = pval.mean()
        # normalize so that probabilities sum to 1
        rs /= np.sum(rs, axis=1)[:, np.newaxis]
        if rs.sum(0).min() < 1e-6:
            break

        # mean probability to be assigned to Gaussian 1
        p = rs[:, 0].mean()
        # new estimate of mean of cluster 1 (weighted by "responsibilities")
        mu1 = np.dot(rs[:, 0], x) / np.sum(rs[:, 0])
        # new estimate of mean of cluster 2 (weighted by "responsibilities")
        mu2 = np.dot(rs[:, 1], x) / np.sum(rs[:, 1])

        # new estimates of variances
        s1 = np.dot(rs[:, 0], (x - mu1) ** 2) / np.sum(rs[:, 0])
        s2 = np.dot(rs[:, 1], (x - mu2) ** 2) / np.sum(rs[:, 1])

        if min(s1, s2) < 1e-6:
            break

        if (k >= 10) and (k % 2 == 0):
            # starting at iteration 10, we start re-estimating the pursuit direction
            # that is, given the Gaussian cluster assignments, and the mean and variances,
            # we re-estimate w
            # these equations follow from the model
            StS = (
                np.matmul(
                    unit_features.T,
                    unit_features
                    * (rs[:, 0] / s1 + rs[:, 1] / s2)[:, np.newaxis],
                )
                / unit_features.shape[0]
            )
            StMu = (
                np.dot(
                    unit_features.T, rs[:, 0] * mu1 / s1 + rs[:, 1] * mu2 / s2
                )
                / unit_features.shape[0]
            )

            # this is the new estimate of the best pursuit direction
            w = np.linalg.solve(StS.T, StMu)
            w /= np.linalg.norm(w)
            x = unit_features @ w

    # these spikes are assigned to cluster 1
    rs = np.exp(logp)
    ilow = rs[:, 0] > rs[:, 1]
    # the smallest cluster has this proportion of all spikes
    nremove = min(ilow.mean(), (~ilow).mean())
    if nremove < min_split_prop:
        return False, None, None

    # the mean probability of spikes assigned to cluster 1/2
    plow = rs[ilow, 0].mean()
    phigh = rs[~ilow, 1].mean()

    # now decide if the split would result in waveforms that are too similar
    # the reconstructed mean waveforms for putative cluster 1
    # c1 = cp.matmul(wPCA, cp.reshape((mean(clp0[ilow, :], 0), 3, -1), order='F'))
    c1 = tpca.inverse_transform(unit_features[ilow].mean())
    c2 = tpca.inverse_transform(unit_features[~ilow].mean())
    # correlation of mean waveforms
    cc = np.corrcoef(c1.ravel(), c2.ravel())[0, 1]
    n1 = np.linalg.norm(c1)  # the amplitude estimate 1
    n2 = np.linalg.norm(c2)  # the amplitude estimate 2

    r0 = 2 * abs((n1 - n2) / (n1 + n2))

    # if the templates are correlated, and their amplitudes are similar, stop the split!!!
    if (cc > max_split_corr) and (r0 < min_amp_sim):
        return False, None, None

    # finaly criteria to continue with the split: if the split piece is more than 5% of all
    # spikes, if the split piece is more than 300 spikes, and if the confidences for
    # assigning spikes to # both clusters exceeds a preset criterion ccsplit
    if (
        (nremove > min_split_prop)
        and (min(plow, phigh) > aucsplit)
        # and (min(cp.sum(ilow), cp.sum(~ilow)) > 300)
    ):
        new_labels = np.zeros(in_unit.size, dtype=int)
        new_labels[~ilow] = 1
        return True, new_labels, in_unit

    return False, None, None


# main function


def split_clusters(
    labels,
    raw_data_bin,
    h5_path,
    n_workers=1,
    feature_scales=(1, 1, 50),
    log_c=5,
    waveforms_kind="cleaned",
    split_steps=(maxchan_lda_split, herding_split, ks_bimodal_pursuit_split),
    recursive_steps=(False, True, True),
    split_step_kwargs=None,
    relocated=False,
    loc_feature=None,
):
    contig = labels.max() + 1 == np.unique(labels[labels >= 0]).size
    if not contig:
        raise ValueError("Please pass contiguous labels to the split step.")

    with h5py.File(h5_path, "r") as h5:
        if labels.shape[0] != h5["spike_index"].shape[0]:
            raise ValueError(
                "labels shape does not match h5. "
                "Maybe you have passed triaged labels?"
            )
        assert f"{waveforms_kind}_tpca_projs" in h5
        assert np.all(np.diff(h5["channel_index"][:], axis=1) >= 0)

    # result goes here
    new_labels = labels.copy()
    del labels

    # process kwargs
    if split_step_kwargs is None:
        split_step_kwargs = [{}] * len(split_steps)
    split_step_kwargs = [kw or {} for kw in split_step_kwargs]

    # set up multiprocessing.
    # Mock has better error messages, will be used with n_workers in (0, 1)
    Executor, context = get_pool(n_workers)
    with Executor(
        max_workers=n_workers,
        mp_context=context,
        initializer=split_worker_init,
        initargs=(
            h5_path,
            log_c,
            feature_scales,
            waveforms_kind,
            raw_data_bin,
            relocated,
            loc_feature,
        ),
    ) as pool:
        # we will do each split step one after the other, each
        # starting with the labels set output by the previous step
        for split_step, recursive, extra_kwargs in zip(
            split_steps, recursive_steps, split_step_kwargs
        ):
            split_step_wrapped = split_fn_wrapper(split_step, extra_kwargs)
            cur_labels_set = np.setdiff1d(new_labels, [-1])
            cur_max_label = cur_labels_set.max()
            nlabels_cur = cur_max_label + 1

            jobs = [
                pool.submit(
                    split_step_wrapped, np.flatnonzero(new_labels == i)
                )
                for i in cur_labels_set
            ]
            iterator = tqdm(
                jobs,
                desc=f"Split step: {split_step.__name__}",
                total=len(cur_labels_set),
                smoothing=0,
            )
            for future in iterator:
                # would be better to do this like "imap style" but not
                # sure how to do that... anyway its ok.
                is_split, unit_new_labels, in_unit = future.result()

                if not is_split:
                    continue

                # -1 will become -1, 0 will keep its current label
                # 1 and on will start at next_label
                unit_new_labels[unit_new_labels > 0] += cur_max_label
                new_labels[in_unit[unit_new_labels < 0]] = unit_new_labels[
                    unit_new_labels < 0
                ]
                new_labels[in_unit[unit_new_labels > 0]] = unit_new_labels[
                    unit_new_labels > 0
                ]
                cur_max_label = new_labels[in_unit].max()

                if recursive:
                    new_jobs = np.setdiff1d(new_labels[in_unit], [-1])
                    jobs.extend(
                        pool.submit(
                            split_step_wrapped, np.flatnonzero(new_labels == i)
                        )
                        for i in new_jobs
                    )
                    iterator.total += len(new_jobs)

            print(f"{new_labels.max() + 1 - nlabels_cur} new units.")

    return new_labels


# -- merge step


def lda_diptest_merge(
    in_unit_a,  # <= np.flatnonzero(labels == unit_a)
    in_unit_b,
    template_a,
    template_b,
    shift,
    max_channels,
    tpca_projs,
    tpca,
    channel_index,
    tpca_rank=5,
    n_channels=10,
    min_spikes=10,
    max_spikes=250,
    threshold_diptest=0.5,
    relocated=False,
    x=None,
    y=None,
    z_abs=None,
    z_reg=None,
    alpha=None,
    geom=None,
    seed=0,
):
    T = template_a.shape[0]
    assert template_b.shape == template_a.shape

    # randomly subset waveforms to balance the problem
    rg = np.random.default_rng(seed)
    max_spikes = min(max_spikes, in_unit_a.size, in_unit_b.size)
    in_unit_a = rg.choice(in_unit_a, max_spikes, replace=False)
    in_unit_a.sort()
    in_unit_b = rg.choice(in_unit_b, max_spikes, replace=False)
    in_unit_b.sort()

    # load cleaned wf tpca projections for both
    # units on the channels subset
    which_chans = np.argsort(-(template_a.ptp(0) + template_b.ptp(0)))[
        :n_channels
    ]
    if relocated:
        feats_a, relocated_maxptps = get_relocated_wfs_on_channel_subset(
            in_unit_a,
            tpca_projs,
            max_channels,
            channel_index,
            which_chans,
            tpca,
            x,
            y,
            z_abs,
            z_reg,
            alpha,
            geom,
            T=T,
        )
    else:
        feats_a = get_pca_projs_on_channel_subset(
            in_unit_a,
            tpca_projs,
            max_channels,
            channel_index,
            which_chans,
        )
    too_far_a = np.isnan(feats_a).any(axis=(1, 2))
    feats_a = feats_a[~too_far_a]
    if relocated:
        feats_b, relocated_maxptps = get_relocated_wfs_on_channel_subset(
            in_unit_b,
            tpca_projs,
            max_channels,
            channel_index,
            which_chans,
            tpca,
            x,
            y,
            z_abs,
            z_reg,
            alpha,
            geom,
            T=T,
        )
    else:
        feats_b = get_pca_projs_on_channel_subset(
            in_unit_b,
            tpca_projs,
            max_channels,
            channel_index,
            which_chans,
        )
    too_far_b = np.isnan(feats_b).any(axis=(1, 2))
    feats_b = feats_b[~too_far_b]
    del in_unit_a, in_unit_b

    if min(feats_a.shape[0], feats_b.shape[0]) < min_spikes:
        return False

    # invert the tpca and align the times according to shift
    # shift is trough[b] - trough[a] here
    if relocated:
        wfs_a = feats_a
        wfs_b = feats_b
    else:
        wfs_a = invert_tpca(feats_a, tpca.tpca)
        wfs_b = invert_tpca(feats_b, tpca.tpca)
    del feats_a, feats_b

    if shift > 0:
        wfs_a = wfs_a[:, :-shift, :]
        wfs_b = wfs_b[:, shift:, :]
    elif shift < 0:
        wfs_a = wfs_a[:, -shift:, :]
        wfs_b = wfs_b[:, :shift, :]

    # apply a shared pca to both units (???)
    wfs = np.concatenate((wfs_a, wfs_b))
    Ntot, T, C = wfs.shape
    wfs = wfs.transpose(0, 2, 1).reshape(Ntot * C, T)
    shared_tpca = PCA(tpca_rank)
    wfs = shared_tpca.inverse_transform(shared_tpca.fit_transform(wfs))
    wfs = wfs.reshape(Ntot, C * T)

    # LDA project and dip test
    labels = np.ones(Ntot, dtype=int)
    labels[: wfs_a.shape[0]] = 0
    lda_projs = LDA(n_components=1).fit_transform(wfs, labels)
    dipscore, cutpoint = isocut5(lda_projs.squeeze())

    # import matplotlib.pyplot as plt
    # fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 5))
    # aa.hist(lda_projs, bins=64)
    # aa.set_title(dipscore)
    # ta = wfs_a.mean(0)
    # tb = wfs_b.mean(0)
    # ab.plot(ta[:, ta.ptp(0).argmax()])
    # ab.plot(tb[:, tb.ptp(0).argmax()])
    # plt.show()
    # plt.close(fig)

    return dipscore < threshold_diptest


def get_proposed_pairs_for_unit(
    unit,
    other_units,
    templates_dict,
    max_resid_dist=5,
    deconv_threshold_mul=0.9,
    n_jobs=-1,
    lambd=0.001,
    allowed_scale=0.1,
):
    other_templates = np.stack(
        [templates_dict[j] for j in other_units], axis=0
    )
    deconv_threshold = deconv_threshold_mul * min(
        np.square(templates_dict[unit]).sum(),
        np.square(other_templates).sum(axis=(1, 2)).min(),
    )

    # shifts[i, j] is like trough[j] - trough[i]
    resids, shifts = calc_resid_matrix(
        templates_dict[unit][None],
        np.array([unit]),
        other_templates,
        np.array(other_units),
        thresh=deconv_threshold,
        n_jobs=n_jobs,
        vis_ptp_thresh=1,
        auto=True,
        pbar=False,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )
    resids = resids.squeeze()
    shifts = shifts.squeeze().astype(int)

    # get pairs with resid max norm < threshold
    ix = np.flatnonzero(resids < max_resid_dist)
    if not ix.size:
        return (), ()
    proposals = np.array([other_units[j] for j in ix])
    prop_resids = resids[ix]

    # sort them in order of increasing distance
    # so that the greedy merge is really greedy
    sort = np.argsort(prop_resids)
    proposals = proposals[sort]
    shifts = shifts[ix[sort]]

    return proposals, shifts


def merge_clusters(
    h5_path,
    raw_data_bin,
    labels,
    templates,
    proposal_max_resid_dist=15,
    threshold_diptest=0.5,
    waveforms_kind="cleaned",
    recursive=True,
    relocated=False,
    trough_offset=42,
    n_jobs=-1,
):

    orig_labels, orig_counts = np.unique(
        labels[labels >= 0], return_counts=True
    )
    assert labels.max() + 1 == orig_labels.size == templates.shape[0]
    n_channels = templates.shape[2]

    # result will go here
    new_labels = labels.copy()
    del labels

    # turn templates into a dict for this step
    snrs = templates.ptp(1).max(1) * np.sqrt(orig_counts)
    templates_dict = {i: templates[i] for i in orig_labels}
    spike_length_samples = templates.shape[1]
    del templates

    # load some stuff from the h5
    h5 = h5py.File(h5_path, "r")
    geom = h5["geom"][:]
    spike_times, max_channels = h5["spike_index"][:].T
    # TODO: these right now are just use for computing templates
    #       the shifts when merging already merged units are not
    #       really handled correctly, but it's hard to do it right
    aligned_times = spike_times.copy()
    channel_index = h5["channel_index"][:]
    tpca_projs = h5[f"{waveforms_kind}_tpca_projs"]
    tpca_feat = TPCA(tpca_projs.shape[1], channel_index, waveforms_kind)
    tpca_feat.from_h5(h5)

    x = y = z_abs = alpha = z_reg = None
    if relocated:
        x, y, z_abs, alpha = h5["localizations"][:, :4].T
        z_reg = h5["z_reg"][:]
    # loop by order of snr
    # high snr will be the last element here
    labels_to_process = list(np.argsort(snrs))
    t = tqdm(desc="Merge", total=len(labels_to_process))
    while len(labels_to_process) > 1:
        # pop removes from the end of list
        label = labels_to_process.pop()
        in_label = np.flatnonzero(new_labels == label)

        # get canditate matches for this unit with the rest of
        # the unprocessed units
        proposals, shifts = get_proposed_pairs_for_unit(
            label,
            labels_to_process,
            templates_dict,
            max_resid_dist=proposal_max_resid_dist,
            n_jobs=n_jobs,
        )
        # print(label, proposals.size)

        # loop through candidates until we find a merge
        # if we find a merge, update the state and add the merged
        # unit back into the labels to process if recursive
        # if not recursive, the merged unit will no longer be
        # considered in future merges
        for candidate, shift in zip(proposals, shifts):
            in_candidate = np.flatnonzero(new_labels == candidate)

            is_merge = lda_diptest_merge(
                in_label,
                in_candidate,
                templates_dict[label],
                templates_dict[candidate],
                shift,
                max_channels,
                tpca_projs,
                tpca_feat,
                channel_index,
                threshold_diptest=threshold_diptest,
                relocated=relocated,
                x=x,
                y=y,
                z_abs=z_abs,
                z_reg=z_reg,
                alpha=alpha,
                geom=geom,
            )

            if is_merge:
                # update the state
                # print("merge", label, candidate)

                # update spike times, using the alignment of the
                # higher snr template. since we are iterating in
                # (roughly) order of decreasing snr, that means
                # we can align to unit `label`
                # shift is like trough[candidate] - trough[label]
                # if >0, trough of label is behind trough of candidate
                # adding will decrease the trough of candidate to match
                aligned_times[in_candidate] += shift

                # by convention, we'll keep the lower label
                new_labels[in_candidate] = label

                # remove candidate from the labels to process
                # we know this item is in there, so this will not ValueError
                labels_to_process.remove(candidate)

                # update with a new template
                del templates_dict[candidate]
                templates_dict[label] = get_raw_template_single(
                    aligned_times[new_labels == label],
                    raw_data_bin,
                    n_channels,
                    trough_offset=trough_offset,
                    spike_length_samples=spike_length_samples,
                )

                if recursive:
                    # we add this to the end of the list,
                    # so it's up next for processing.
                    labels_to_process.append(label)
                    t.total += 1
                    t.refresh()

                break

        t.update()

    h5.close()

    return aligned_times, new_labels


# -- helpers


def get_relocated_wfs_on_channel_subset(
    which,
    tpca_projs,
    max_channels,
    channel_index,
    which_chans,
    tpca,
    x,
    y,
    z_abs,
    z_reg,
    alpha,
    geom,
    T=121,
    load_batch_size=512,
):
    # load pca projected spikes for this unit
    these_wfs = np.empty(
        (which.size, T, tpca_projs.shape[2]), dtype=tpca_projs.dtype
    )
    for bs in range(0, which.size, load_batch_size):
        be = bs + load_batch_size
        these_wfs[bs:be] = tpca.inverse_transform(
            tpca_projs[which[bs:be]],
            max_channels[which[bs:be]],
            channel_index,
        )

    relocated_wfs = relocation.get_relocated_waveforms_on_channel_subset(
        max_channels[which],
        these_wfs,
        np.c_[x[which], y[which], z_abs[which], alpha[which]],
        z_reg[which],
        channel_index,
        geom,
        which_chans,
        fill_value=np.nan,
    )

    relocated_maxptps = np.nanmax(relocated_wfs.ptp(1), axis=1)

    return relocated_wfs, relocated_maxptps


def get_pca_projs_on_channel_subset(
    which,
    tpca_projs,
    max_channels,
    channel_index,
    which_chans,
    load_batch_size=512,
):
    # load pca projected spikes for this unit
    these_tpca_projs = np.empty(
        (which.size, *tpca_projs.shape[1:]), dtype=tpca_projs.dtype
    )
    for bs in range(0, which.size, load_batch_size):
        be = bs + load_batch_size
        these_tpca_projs[bs:be] = tpca_projs[which[bs:be]]

    # which channels do we load, as a function of max channel
    channel_index_mask = np.isin(channel_index, which_chans)

    # gather pca projs on those channels
    these_tpca_projs = get_channel_subset(
        these_tpca_projs,
        max_channels[which],
        channel_index_mask,
    )

    return these_tpca_projs


def invert_tpca(projs, tpca):
    N, R, C = projs.shape
    projs = projs.transpose(0, 2, 1)
    wfs = tpca.inverse_transform(projs.reshape(-1, R))
    return wfs.reshape(N, C, -1).transpose(0, 2, 1)


# -- parallelism helpers
# users don't need to worry about these -- just write your split functions as above.


def split_worker_init(
    h5_path,
    log_c,
    feature_scales,
    waveforms_kind,
    raw_data_bin,
    relocated,
    loc_feature,
):
    """
    Loads hdf5 datasets on each worker process, rather than
    loading them once in every split.
    """
    # we will assign lots of properties here, and
    # the split functions below will load them up
    p = split_worker_init
    p.extractor = H5Extractor(
        h5_path,
        raw_data_bin,
        log_c=log_c,
        feature_scales=feature_scales,
        waveforms_kind=waveforms_kind,
        loc_feature=loc_feature,
    )
    p.relocated = relocated


def split_fn_wrapper(split_fn, extra_kwargs):
    p = split_worker_init
    argspec = getfullargspec(split_fn)

    @wraps(split_fn)
    def split_fn_wrapped(*args, **kwargs):
        kwargs.update(extra_kwargs)
        kwargs["extractor"] = p.extractor
        if "relocated" in argspec.args:
            kwargs["relocated"] = p.relocated
        return split_fn(*args, **kwargs)

    return split_fn_wrapped
