import time
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from tqdm.auto import trange

from .point_source_centering import relocate_index, relocating_ptps_index
from .waveform_utils import channel_index_subset


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
        channel_index,
        decorrelated=False,
        n_channels=20,
        relocate_dims="xyza",
        fit_n_waveforms=100_000,
        B_updates=5,
        random_seed=0,
    ):
        self.n_components = n_components
        n_channels = min(n_channels, channel_index.shape[1])
        self.n_channels = n_channels
        self.geom = geom
        self.channel_index = channel_index
        C = channel_index.shape[1]
        if C > self.n_channels:
            channel_subset = channel_index_subset(
                geom, channel_index, n_channels=n_channels
            )
            sub_channel_index = []
            for mask in channel_subset:
                s = np.flatnonzero(mask)
                s = list(s) + [C] * (self.n_channels - len(s))
                sub_channel_index.append(s)
            self.sub_channel_index = np.array(sub_channel_index)
        else:
            self.sub_channel_index = np.broadcast_to(
                np.arange(C)[None, :],
                channel_index.shape,
            )
        self.relocate_dims = relocate_dims
        self.fit_n_waveforms = fit_n_waveforms
        self.B_updates = B_updates
        self.rg = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.decorrelated = decorrelated

    def fit(self, waveforms, x, y, z, alpha, maxchans):
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
            == maxchans.shape
        )

        # -- which random subset of waveforms will we fit to?
        choice = slice(None)
        if N > self.fit_n_waveforms:
            choice = self.rg.choice(
                N, replace=False, size=self.fit_n_waveforms
            )
            choice.sort()
            N = self.fit_n_waveforms

        # trim to our subset
        wfs = []
        for i in range(0, len(choice), 1000):
            wfs.append(waveforms[choice[i:i + 1000]])
        waveforms = np.concatenate(wfs, axis=0)
        x = x[choice]
        y = y[choice]
        z = z[choice]
        alpha = alpha[choice]
        maxchans = maxchans[choice]

        L = None
        if self.decorrelated:
            L = np.c_[x, np.log(y), z, np.log(alpha)]

        # trim to n_channels if necessary
        if C > self.n_channels:
            waveforms = np.pad(
                waveforms, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
            )
            waveforms = waveforms[
                np.arange(N)[:, None, None],
                np.arange(T)[None, :, None],
                self.sub_channel_index[maxchans][:, None, :],
            ]

        # initial mask: what channels are active / in probe
        S = (~np.isnan(waveforms)).astype(waveforms.dtype)

        # -- relocated waveforms and the transformations to get them
        with timer("Relocating"):
            relocated_waveforms, r, q = relocate_index(
                waveforms,
                self.geom,
                self.sub_channel_index,
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
            destandardization[np.isnan(destandardization)] = 0
            S = np.broadcast_to(destandardization, (relocated_waveforms.shape))

        # -- initialize B with PCA in relocated space
        with timer("PCA initialization"):
            reloc_pca = PCA(self.n_components)
            pca_train_data = KNNImputer(copy=False).fit_transform(relocated_waveforms.reshape(N, T * self.n_channels))
            reloc_pca.fit(pca_train_data)
            B = reloc_pca.components_
            B = B.T.reshape(T, self.n_channels, self.n_components)

        # rank 0 model
        relocated_mean = reloc_pca.mean_.reshape(T, self.n_channels)
        # unrelocated_means = relocated_mean[None, :, :] * destandardization
        # decentered_waveforms = waveforms - unrelocated_means
        centered_relocated = relocated_waveforms - relocated_mean[None]

        for _ in trange(self.B_updates):
            F = update_F(S, centered_relocated, B, L=L)
            B = update_B(S, centered_relocated, F)

        # store and return
        self.B_ = B
        self.mean_ = relocated_mean
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
        Fs = []
        for bs in range(0, N, 1000):
            be = min(N, bs + 1000)
            batch_wfs = waveforms[bs:be]
            batch_mcs = maxchans[bs:be]

            # trim to n_channels if necessary
            C = C_
            if C_ > self.n_channels:
                batch_wfs = np.pad(
                    batch_wfs, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
                )
                batch_wfs = batch_wfs[
                    np.arange(N)[:, None, None],
                    np.arange(T)[None, :, None],
                    channel_subset[batch_mcs][:, None, :],
                ]

            # relocate
            relocated_waveforms, r, q = relocate_index(
                batch_wfs,
                self.geom,
                self.sub_channel_index,
                batch_mcs,
                x[bs:be],
                y[bs:be],
                z[bs:be],
                alpha[bs:be],
                relocate_dims=self.relocate_dims,
            )
            relocated_waveforms = relocated_waveforms.cpu().numpy()
            r = r.cpu().numpy()
            q = q.cpu().numpy()

            L = None
            if self.decorrelated:
                L = np.c_[
                    x[bs:be],
                    np.log(y[bs:be]),
                    z[bs:be],
                    np.log(alpha[bs:be]),
                ]

            # Nx1xC transformation to invert the relocation
            destandardization = (q / r)[:, None, :]
            destandardization[np.isnan(destandardization)] = 0
            S = np.broadcast_to(destandardization, (relocated_waveforms.shape))

            centered_relocated = relocated_waveforms - self.mean_[None]

            F = update_F(S, centered_relocated, self.B_, L=L)
            Fs.append(F)

        # KxN -> NxK
        features = np.concatenate(Fs, axis=1).T
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
        r, q = relocating_ptps_index(
            self.geom,
            self.sub_channel_index,
            maxchans,
            x,
            y,
            z,
            alpha,
            self.n_channels,
            relocate_dims=self.relocate_dims,
        )
        # those are torch but we want numpy
        r = r.cpu().numpy()
        q = q.cpu().numpy()
        destandardization = (q / r)[:, None, :]

        return destandardization * (
            self.mean_[None, :, :] + np.tensordot(features, self.B_, axes=1)
        )


def update_F(S, W_prime, B, L=None):
    T, C, K = B.shape
    N, T_, C_ = S.shape
    N_, T__, C__ = W_prime.shape
    assert C == C_ == C__
    assert T == T_ == T__
    if L is not None:
        assert L.shape[0] == N
        
    # construct problem
    y = (S * W_prime).T.ravel()
    dvS = sparse.dia_matrix((S.ravel(), 0), shape=(N * T * C, N * T * C))
    X = dvS @ sparse.kron(sparse.eye(N), B.reshape(T * C, K))

    XTX = X.T @ X
    nyTX = -y.T @ X

    if L is not None:
        A = sparse.kron(L.T, sparse.eye(K))
        zeros = sparse.dok_matrix((4 * K, 4 * K))
        coefts = sparse.bmat(
            [
                [XTX, A.T],
                [A, zeros],
            ],
            format="csc",
        )

        targ = sparse.bmat(
            [
                [nyTX[:, None]],
                [sparse.dok_matrix((4 * K, 1))],
            ],
            format="csc",
        ).toarray()[:, 0]
    else:
        coefts = XTX.tocsc()
        targ = nyTX

    res_lsmr = sparse.linalg.lsmr(
        coefts,
        targ,
        atol=1e-6 / (N * T * C),
        btol=1e-6 / (N * T * C),
    )

    # check good convergence
    if res_lsmr[1] not in (0, 1, 4):
        print("Convergence value in F update was", res_lsmr[1])
    F = res_lsmr[0][: K * N].reshape(K, N)

    return F


def update_B(S, W_prime, F):
    K, N = F.shape
    N_, T, C = S.shape
    N__, T_, C_ = W_prime.shape
    assert N == N_ == N__
    assert T == T_ and C == C_

    y = (S * W_prime).T.ravel()
    dvS = sparse.dia_matrix((S.ravel(), 0), shape=(N * T * C, N * T * C))
    X = dvS @ sparse.kron(F.T, sparse.eye(T * C))
    XTX = X.T @ X
    nyTX = -y.T @ X

    res_lsmr = sparse.linalg.lsmr(
        XTX,
        nyTX,
        atol=1e-6 / (N * T * C),
        btol=1e-6 / (N * T * C),
    )

    # check good convergence
    if res_lsmr[1] not in (0, 1, 4):
        print("Convergence value in B update was", res_lsmr[1])
    B = res_lsmr[0].reshape(T, C, K)

    return B

class timer:
    def __init__(self, name="timer"):
        self.name = name
 
    def __enter__(self):
        self.start = time.time()
        return self
 
    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")
