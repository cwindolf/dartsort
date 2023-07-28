import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from tqdm.auto import trange

from .point_source_centering import relocate_simple, relocating_ptps
from .waveform_utils import relativize_waveforms


class LinearRelocAE(BaseEstimator, TransformerMixin):
    r"""Linear "relocated" autoencoder

    Fits a linear autoencoder to relocated waveforms, minimizing a loss in
    the original (un-relocated) space. Any dimension (x,y,z,a) can be
    "standardized" with the `relocate_dims` flag, and this can also handle
    subsetting channels if you want to featurize a subset of channels
    with `n_channels`.

    Let W_n be the nth TxC waveform of N waveforms, q_n its PTP vector as
    predicted by the point source model, and let r_n be its "standard" PTP.

    Then this minimizes the following cost over B (KxTxC matrix, the basis)
    and F (NxK, the features) using an alternating least squares approach:

    \sum_{n=1}^N \sum_{t=1}^T \sum_{c=1}^C
        ( W_{ntc} - (q_{nc} / r_{nc}) \sum_{k=1}^K B_{ktc} F_{nk} )^2

    Well, almost. It also allows for a mean M (TxC), so it really optimizes

    \sum_{n=1}^N \sum_{t=1}^T \sum_{c=1}^C
        (W_{ntc} - [(q_{nc} / r_{nc}) (M_{tc} + \sum_{k=1}^K B_{ktc} F_{nk}])^2

    Arguments
    ---------
    n_components : int
        How many features to learn?
    geom : total_n_channels x 2
        The probe geometry array
    n_channels : int
        If your input waveforms have more channels than this, the
        subset of `n_channels` around the max channel will be
        extracted before featurization.
    relocate_dims : string containing characters from "xyza"
        Which localization parameters are being standardized?
    fit_n_waveforms : int
        How many waveforms to extract from the input when fitting
        with `fit`. These will be chosen randomly (according to
        `random_seed`). If there are fewer waveforms in the data
        passed to `fit`, then all of them will be used.
    B_updates : int
        How many times to run the alternating minimization
    random_seed : int
        For reproducibility
    """

    def __init__(
        self,
        n_components,
        geom,
        n_channels=18,
        relocate_dims="xyza",
        fit_n_waveforms=100_000,
        B_updates=2,
        random_seed=None,
    ):
        self.n_components = n_components
        self.n_channels = n_channels
        self.geom = geom
        self.relocate_dims = relocate_dims
        self.fit_n_waveforms = fit_n_waveforms
        self.B_updates = B_updates
        self.rg = np.random.default_rng(random_seed)
        self.random_seed = random_seed

    def fit(self, waveforms, x, y, z, alpha, firstchans, maxchans):
        """
        waveforms : N x T x C
        x, y, z, alpha : N float
            Note z here is in absolute coordinates, not relative to max channel
        """
        N, T, C = waveforms.shape
        assert (
            (N,)
            == x.shape
            == y.shape
            == z.shape
            == alpha.shape
            == firstchans.shape
        )  # noqa

        # -- which random subset of waveforms will we fit to?
        choice = slice(None)  # all by default
        if N > self.fit_n_waveforms:
            choice = self.rg.choice(
                N, replace=False, size=self.fit_n_waveforms
            )
            choice.sort()
            N = self.fit_n_waveforms

        # trim to our subset
        waveforms = waveforms[choice]
        x = x[choice]
        y = y[choice]
        z = z[choice]
        alpha = alpha[choice]
        firstchans = firstchans[choice]
        maxchans = maxchans[choice]

        # trim to n_channels if necessary
        if C > self.n_channels:
            waveforms, firstchans, *_ = relativize_waveforms(
                waveforms,
                firstchans,
                None,
                self.geom,
                feat_chans=self.n_channels,
                maxchans_orig=maxchans,
            )
            C = self.n_channels

        # -- relocated waveforms and the transformations to get them
        print("hi")
        relocated_waveforms, r, q = relocate_simple(
            waveforms,
            self.geom,
            firstchans,
            maxchans,
            x,
            y,
            z,
            alpha,
            relocate_dims=self.relocate_dims,
        )
        # those are torch but we want numpy
        relocated_waveforms = relocated_waveforms.cpu().numpy()
        r = r.cpu().numpy()
        q = q.cpu().numpy()

        # Nx1xC transformation to invert the relocation
        destandardization = (q / r)[:, None, :]

        # -- initialize B with PCA in relocated space
        reloc_pca = PCA(self.n_components)
        reloc_pca.fit(relocated_waveforms.reshape(N, T * C))
        B = reloc_pca.components_.reshape(self.n_components, T, C)

        # rank 0 model
        relocated_mean = reloc_pca.mean_.reshape(T, C)
        unrelocated_means = relocated_mean[None, :, :] * destandardization
        decentered_waveforms = waveforms - unrelocated_means

        # re-compute the loadings to minimize loss in original space
        features = np.zeros((N, self.n_components))
        err = 0.0
        errors = np.zeros(N)
        for n in range(N):
            A = (
                (destandardization[n, None, :, :] * B)
                .reshape(self.n_components, T * C)
                .T
            )
            b = decentered_waveforms[n].reshape(T * C)
            feat, resid, *_ = np.linalg.lstsq(A, b, rcond=None)
            features[n] = feat
            err += resid
            errors[n] = resid / (T * C)
        err = err / (N * T * C)

        for _ in trange(self.B_updates, desc="B updates"):
            # update B
            # flat view
            B = B.reshape(self.n_components, T * C)
            W = decentered_waveforms.reshape(N, T * C)
            for c in range(C):
                A = destandardization[:, 0, c, None] * features
                for t in range(T):
                    i = t * C + c
                    res, *_ = np.linalg.lstsq(A, W[:, i], rcond=None)
                    B[:, i] = res
            B = B.reshape(self.n_components, T, C)

            # re-update features
            features = np.zeros((N, self.n_components))
            err = 0.0
            for n in range(N):
                A = (
                    (destandardization[n, None, :, :] * B)
                    .reshape(self.n_components, T * C)
                    .T
                )
                b = decentered_waveforms[n].reshape(T * C)
                feat, resid, *_ = np.linalg.lstsq(A, b, rcond=None)
                features[n] = feat
                err += resid
                errors[n] = resid / (T * C)
            err = err / (N * T * C)

        # store and return
        self.B_ = B
        self.mean_ = relocated_mean
        self.train_mse_ = err
        return self

    def transform(
        self,
        waveforms,
        x,
        y,
        z,
        alpha,
        firstchans,
        maxchans,
        return_error=False,
    ):
        N, T, C_ = waveforms.shape
        features = np.zeros((N, self.n_components))
        errors = np.zeros(N)

        # do this in batches in case a huge input has been passed
        for bs in range(0, N, self.fit_n_waveforms):
            be = min(N, bs + self.fit_n_waveforms)
            batch_wfs = waveforms[bs:be]
            batch_fcs = firstchans[bs:be]
            batch_mcs = maxchans[bs:be]

            # trim to n_channels if necessary
            C = C_
            if C_ > self.n_channels:
                batch_wfs, batch_fcs, batch_mcs, _ = relativize_waveforms(
                    batch_wfs,
                    batch_fcs,
                    None,
                    self.geom,
                    feat_chans=self.n_channels,
                    maxchans_orig=batch_mcs,
                )
                C = self.n_channels

            # relocate
            relocated_waveforms, r, q = relocate_simple(
                batch_wfs,
                self.geom,
                batch_fcs,
                batch_mcs,
                x[bs:be],
                y[bs:be],
                z[bs:be],
                alpha[bs:be],
                relocate_dims=self.relocate_dims,
            )
            relocated_waveforms = relocated_waveforms.cpu().numpy()
            r = np.atleast_2d(r.cpu().numpy())
            q = np.atleast_2d(q.cpu().numpy())

            # Nx1xC transformation to invert the relocation
            destandardization = (q / r)[:, None, :]

            # rank 0 model
            unrelocated_means = self.mean_[None, :, :] * destandardization
            decentered_waveforms = batch_wfs - unrelocated_means

            # solve least squares problems to determine the features
            for n in range(be - bs):
                A = (
                    (destandardization[n, None, :, :] * self.B_)
                    .reshape(self.n_components, T * C)
                    .T
                )
                b = decentered_waveforms[n].reshape(T * C)
                feat, resid, *_ = np.linalg.lstsq(A, b, rcond=None)
                features[bs + n] = feat
                errors[bs + n] = resid / (T * C)

        if return_error:
            return features, errors

        return features

    def inverse_transform(
        self,
        features,
        x,
        y,
        z,
        alpha,
        firstchans,
        maxchans,
    ):
        # inverse relocation
        r, q = relocating_ptps(
            self.geom,
            firstchans,
            maxchans,
            x,
            y,
            z,
            alpha,
            self.n_channels,
            relocate_dims=self.relocate_dims,
        )
        # those are torch but we want numpy
        r = np.atleast_2d(r.cpu().numpy())
        q = np.atleast_2d(q.cpu().numpy())
        destandardization = (q / r)[:, None, :]

        return destandardization * (
            self.mean_[None, :, :]
            + np.tensordot(features, self.B_, axes=1)
        )
