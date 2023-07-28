import time
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tqdm.auto import trange
import cvxpy as cp
import warnings

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
        n_channels=None,
        radius=100,
        relocate_dims="xyza",
        fit_n_waveforms=100_000,
        initialization="pca",
        B_updates=5,
        random_seed=0,
        unrelocated_loss=False,
    ):
        assert len(geom) == len(channel_index)
        self.n_components = n_components
        self.geom = geom
        self.channel_index = channel_index
        self.initialization = initialization
        self.unrelocated_loss = unrelocated_loss
        C = channel_index.shape[1]
        channel_subset = channel_index_subset(
            geom,
            channel_index,
            n_channels=n_channels,
            radius=radius,
        )
        if n_channels is not None:
            assert n_channels == channel_subset.sum(axis=1).max()
        n_channels = channel_subset.sum(axis=1).max()

        rel_sub_channel_index = []
        for mask in channel_subset:
            s = np.flatnonzero(mask)
            s = list(s) + [C] * (n_channels - len(s))
            rel_sub_channel_index.append(s)
        self.rel_sub_channel_index = np.array(rel_sub_channel_index)
        self.sub_channel_index = np.pad(
            channel_index, [(0, 0), (0, 1)], constant_values=384
        )[np.arange(len(geom))[:, None], self.rel_sub_channel_index]
        self.n_channels = self.sub_channel_index.shape[1]

        self.relocate_dims = relocate_dims
        self.fit_n_waveforms = fit_n_waveforms
        self.B_updates = B_updates
        self.rg = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.decorrelated = decorrelated

    def fit(self, waveforms, x, y, z, alpha, maxchans, which=None):
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
        choice = None

        # accommodate triaging by boolean mask or index array
        if which is not None:
            if which.dtype == bool:
                choice = np.flatnonzero(choice)
            else:
                choice = which
            N = len(choice)

        # random choice of training waveforms
        if N > self.fit_n_waveforms:
            choice = choice if choice is not None else np.arange(N)
            choice = self.rg.choice(
                choice,
                replace=False,
                size=self.fit_n_waveforms,
            )
            choice.sort()
            N = self.fit_n_waveforms

        # trim to our subset
        if choice is not None:
            wfs = np.zeros((N, T, C), waveforms.dtype)
            for i in trange(
                0, len(choice), 1000, desc="Grabbing training subset..."
            ):
                wfs[i : i + 1000] = waveforms[choice[i : i + 1000]]
            waveforms = wfs
        else:
            choice = slice(None)

        x = x[choice]
        y = y[choice]
        z = z[choice]
        alpha = alpha[choice]
        maxchans = maxchans[choice]

        L = None
        if self.decorrelated:
            L = np.c_[x, np.log(y), z  - self.geom[maxchans, 1], np.log(alpha)]

        # trim to n_channels if necessary
        if C > self.n_channels:
            waveforms = np.pad(
                waveforms, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
            )
            waveforms = waveforms[
                np.arange(N)[:, None, None],
                np.arange(T)[None, :, None],
                self.rel_sub_channel_index[maxchans][:, None, :],
            ]

        # initial mask: what channels are active / in probe
        outside_probe = np.isnan(waveforms)
        print("Fraction outside probe:", outside_probe.mean())
        S = (~outside_probe).astype(waveforms.dtype)

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

            # for debug / ptp vis
            self._train_standard_ptp = r
            self._train_predicted_ptp = q
            self._train_original_ptp = waveforms.ptp(1)
            self._train_relocated_ptp = relocated_waveforms.ptp(1)

            # Nx1xC transformation to invert the relocation
            destandardization = (q / r)[:, None, :]
            destandardization[np.isnan(destandardization)] = 0
            if self.unrelocated_loss:
                destandardization[destandardization > 0] = 1
            S = np.broadcast_to(destandardization, (relocated_waveforms.shape))

        # -- initialize B with PCA in relocated space

        if self.initialization == "pca":
            with timer("PCA initialization"):
                reloc_pca = PCA(self.n_components)
                pca_train_data = SimpleImputer(copy=False).fit_transform(
                    relocated_waveforms.reshape(N, T * self.n_channels)
                )
                reloc_pca.fit(pca_train_data)
                B = reloc_pca.components_
                B = B.T.reshape(T, self.n_channels, self.n_components)
                relocated_mean = reloc_pca.mean_.reshape(T, self.n_channels)
        elif self.initialization == "random":
            relocated_mean = np.nanmean(relocated_waveforms, axis=0)
            B = self.rg.normal(size=(T, self.n_channels, self.n_components))
            B /= np.linalg.norm(B, axis=(1, 2), keepdims=True)

        centered_relocated = relocated_waveforms - relocated_mean[None]
        centered_relocated[outside_probe] = 0

        for _ in trange(self.B_updates, desc="Coordinate descent"):
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
        maxchans,
        batch_size=50000,
    ):
        N, T, C = waveforms.shape

        # do this in batches in case a memmap has been passed
        Fs = []
        for bs in trange(0, N, batch_size, desc="Transform"):
            be = min(N, bs + batch_size)
            batch_wfs = waveforms[bs:be]
            batch_mcs = maxchans[bs:be]

            # trim to n_channels if necessary
            if C > self.n_channels:
                batch_wfs = np.pad(
                    batch_wfs, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
                )
                batch_wfs = batch_wfs[
                    np.arange(be - bs)[:, None, None],
                    np.arange(T)[None, :, None],
                    self.rel_sub_channel_index[batch_mcs][:, None, :],
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
                    z[bs:be] - self.geom[maxchans[bs:be], 1],
                    np.log(alpha[bs:be]),
                ]

            # Nx1xC transformation to invert the relocation
            destandardization = (q / r)[:, None, :]
            destandardization[np.isnan(destandardization)] = 0
            if self.unrelocated_loss:
                destandardization[destandardization > 0] = 1
            S = np.broadcast_to(destandardization, (relocated_waveforms.shape))

            centered_relocated = relocated_waveforms - self.mean_[None]
            centered_relocated[np.isnan(centered_relocated)] = 0

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
    print("NTCK", N, T, C, K)
    assert C == C_ == C__
    assert T == T_ == T__
    if L is not None:
        assert L.shape[0] == N
        
    # S = S.reshape(N, T * C).T
    # W = W_prime.reshape(N, T * C).T
    
        
    # F = cp.Variable((K, N))
    # cost = cp.sum_squares(cp.multiply(S, (W - B.reshape(T * C, K) @ F)))
    # constraints = []
    # if L is not None:
    #     print("constrained")
    #     constraints = [N * K * 4 * F @ L == 0]
    # prob = cp.Problem(cp.Minimize(cost), constraints)
    # prob.solve(verbose=True)
    # return F.value

    with timer("construct"):
        # construct problem
        y = (S * W_prime).ravel(order="C")
        dvS = sparse.dia_matrix(
            (S.ravel(order="C"), 0), shape=(N * T * C, N * T * C)
        )
        X = dvS @ sparse.kron(sparse.eye(N), B.reshape(T * C, K))

        XTX = X.T @ X
        nyTX = -y.T @ X

        if L is not None:
            A = sparse.kron(L.T, sparse.eye(K))
#             zeros = sparse.dok_matrix((4 * K, 4 * K))
#             coefts = sparse.bmat(
#                 [
#                     [XTX, A.T],
#                     [A, zeros],
#                 ],
#                 format="csc",
#             )

#             targ = sparse.bmat(
#                 [
#                     [nyTX[:, None]],
#                     [sparse.dok_matrix((4 * K, 1))],
#                 ],
#                 format="csc",
#             ).toarray()[:, 0]
            print(A.toarray().shape)
            u, s, vh = np.linalg.svd(A.toarray())
            print("usv", u.shape, s.shape, vh.shape)
            print(XTX.shape, A.shape, nyTX.shape)
            span = vh[:8]
            ker = vh[8:]
            print("span, ker", span.shape, ker.shape)
            Ps = span.T @ np.linalg.pinv(span @ span.T) @ span
            Pk = ker.T @ np.linalg.pinv(ker @ ker.T) @ ker
            ALinv = Pk @ np.linalg.inv( (XTX @ Pk + Ps) )
            F = ALinv @ nyTX
            return F.reshape(K, N)
        else:
            coefts = XTX
            targ = nyTX

        # preconditioning really helps a lot! both with cond issues and speed
        # simple diagonal preconditioner, see
        # https://stanford.edu/group/SOL/software/lsmr/
#         D = sparse.linalg.norm(coefts, axis=1)
#         D[D == 0] = 1
#         D = sparse.dia_matrix((1 / D, 0), shape=coefts.shape)

#     with timer("F lsmr"):
#         # res_lsmr = sparse.linalg.lsmr(
#         #     coefts @ D,
#         #     targ,
#         #     atol=1e-6 / (N * K + 4 * K * (L is not None)),
#         #     btol=1e-6 / (N * K + 4 * K * (L is not None)),
#         #     maxiter=100 * (N * K + 4 * K * (L is not None)),
#         # )
#         res_lsmr = sparse.linalg.gcrotmk(
#             coefts @ D,
#             targ,
#             tol=1e-6 / (N * K + 4 * K * (L is not None)),
#             # btol=1e-6 / (N * K + 4 * K * (L is not None)),
#             maxiter=100 * (N * K + 4 * K * (L is not None)),
#         )
#         # res = sparse.linalg.minres(
#         #     coefts @ D,
#         #     targ,
#         #     tol=1e-6 / (N * K + 4 * K * (L is not None)),
#         #     # btol=1e-6 / (N * K + 4 * K * (L is not None)),
#         #     maxiter=100 * (N * K + 4 * K * (L is not None)),
#         # )
#         # res_lsmr = (res[0], 0)

#     # check good convergence
#     if res_lsmr[1] not in (0, 1, 4):
#         warnings.warn(f"Convergence value in F update was {res_lsmr[1]}")
#     F = D @ res_lsmr[0]
#     F = F[: K * N].reshape(N, K).T

    #     res_cg = sparse.linalg.cg(
    #         coefts,
    #         targ,
    #         tol=1e-6 / (N * T * C),
    #         M=D,
    #     )

    #     # check good convergence
    #     if res_cg[1] != 0:
    #         warnings.warn(f"Convergence value in F update was {res_cg[1]}")
    #     F = res_cg[0]
    #     F = F[: K * N].reshape(K, N)

    return F


def update_B(S, W_prime, F):
    K, N = F.shape
    N_, T, C = S.shape
    N__, T_, C_ = W_prime.shape
    assert N == N_ == N__
    assert T == T_ and C == C_

    y = (S * W_prime).ravel(order="C")
    dvS = sparse.dia_matrix(
        (S.ravel(order="C"), 0), shape=(N * T * C, N * T * C)
    )
    X = dvS @ sparse.kron(F.T, sparse.eye(T * C))
    XTX = X.T @ X
    nyTX = -y.T @ X

    D = sparse.linalg.norm(XTX, axis=1)
    D[D == 0] = 1
    D = sparse.dia_matrix((1 / D, 0), shape=XTX.shape)

    with timer("B lsmr"):
        res_lsmr = sparse.linalg.lsmr(
            XTX @ D,
            nyTX,
            atol=1e-6 / (K * T * C),
            btol=1e-6 / (K * T * C),
        )

    # check good convergence
    if res_lsmr[1] not in (0, 1, 4):
        warnings.warn(f"Convergence value in B update was {res_lsmr[1]}")
    B = (D @ res_lsmr[0]).reshape(K, T * C).T
    B /= np.linalg.norm(B, axis=0, keepdims=True)
    B = B.reshape(T, C, K)
    # B = res_lsmr[0].reshape(T, C, K)

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
