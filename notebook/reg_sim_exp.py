# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:mysi]
#     language: python
#     name: conda-env-mysi-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt
import spikeinterface.full as si
from spikeinterface.sortingcomponents import motion_estimation, peak_detection, peak_localization

# %%
import neuropixel

# %%
from pathlib import Path

# %%
name = "dd1"

# %%
figdir = Path("/moto/stats/users/ciw2107/mysi_reg_test")
figdir.mkdir(exist_ok=True)

# %%
job_kw = dict(chunk_size=30_000, n_jobs=10)

# %%
rec = si.read_binary(
    "/moto/stats/users/ciw2107/icassp2023/drift-dataset1/destriped_p1_g0_t0.imec0.ap.bin",
    # "/local/p1_g0_t0.imec0.ap.bin",
    sampling_frequency=30000,
    num_chan=384,
    dtype=np.int16,
)

h = neuropixel.dense_layout(version=2)
geom = np.c_[h['x'], h['y']]
rec.set_dummy_probe_from_locations(geom, shape_params=dict(radius=10))

# %%
# !rsync -avP /moto/stats/users/ciw2107/dataset1/p1_g0_t0.imec0.ap.* /local/

# %%
# !ls /local

# %%
# rec = si.read_binary(
#     # "/moto/stats/users/ciw2107/icassp2023/drift-dataset1/destriped_p1_g0_t0.imec0.ap.bin",
#     "/local/p1_g0_t0.imec0.ap.bin",
#     sampling_frequency=30000,
#     num_chan=384,
#     dtype=np.int16,
# )
rec = si.read_spikeglx("/local/")
# h = neuropixel.dense_layout(version=2)
# geom = np.c_[h['x'], h['y']]
# rec.set_dummy_probe_from_locations(geom, shape_params=dict(radius=10))
# rec = si.
rec

# %%
rec = si.bandpass_filter(rec, dtype=np.float32)
rec = si.phase_shift(rec)
rec = si.common_reference(rec)
rec = si.zscore(rec)

# %%
t = rec.get_traces(start_frame=410000, end_frame=420000)
t.min(), t.max(), t.std(axis=0).mean()

# %%
overwrite = False

peaks_npy = figdir / f"{name}_peaks.npy"
if overwrite or not peaks_npy.exists():
    peaks = peak_detection.detect_peaks(
        rec,
        method="locally_exclusive",
        local_radius_um=100,
        peak_sign="both",
        detect_threshold=5,
        **job_kw,
    )
    np.save(peaks_npy, peaks)
peaks = np.load(peaks_npy)

peak_locations_npy = figdir / f"{name}_peak_locations.npy"
if overwrite or not peak_locations_npy.exists():
    peak_locations = peak_localization.localize_peaks(
        rec,
        peaks,
        ms_before=0.3,
        ms_after=0.6,
        method='monopolar_triangulation',
        **{'local_radius_um': 100., 'max_distance_um': 1000., 'optimizer': 'minimize_with_log_penality'},
        **job_kw,
    )
    np.save(peak_locations_npy, peak_locations)
peak_locations = np.load(peak_locations_npy)

# %%
bin_um = 5.0
bin_s = 2.0
margin_um = -1300

# %%
spatial_bin_edges = motion_estimation.get_spatial_bin_edges(rec, "y", margin_um, bin_um)
# non_rigid_windows, non_rigid_window_centers = get_windows(True, bin_um, contact_pos, spatial_bin_edges,
#                                                               margin_um, win_step_um, win_sigma_um, win_shape)
motion_histogram, temporal_hist_bin_edges, spatial_hist_bin_edges = \
            motion_estimation.make_2d_motion_histogram(rec, peaks,
                                     peak_locations,
                                     direction="y",
                                     bin_duration_s=bin_s,
                                     spatial_bin_edges=spatial_bin_edges,
                                     weight_with_amplitude=True,
                                    )
plt.imshow(motion_histogram.T, aspect="auto", vmax=15)

# %%
subsample_p = 0.02 + 0.98 * (1 - peaks["sample_ind"].astype(float) / peaks["sample_ind"].max())
subsample_p = 0.02 + 0.98 * (0.5 + np.sin(np.pi + np.pi * peaks["sample_ind"].astype(float) / peaks["sample_ind"].max()) / 2)
rg = np.random.default_rng(0)
subsamp = rg.binomial(1, subsample_p)
subsamp = np.flatnonzero(subsamp)
peaks_sub = peaks[subsamp]
peak_locations_sub = peak_locations[subsamp]

# %%
spatial_bin_edges = motion_estimation.get_spatial_bin_edges(rec, "y", margin_um, bin_um)
# non_rigid_windows, non_rigid_window_centers = get_windows(True, bin_um, contact_pos, spatial_bin_edges,
#                                                               margin_um, win_step_um, win_sigma_um, win_shape)
motion_histogram, temporal_hist_bin_edges, spatial_hist_bin_edges = \
            motion_estimation.make_2d_motion_histogram(rec, peaks_sub,
                                     peak_locations_sub,
                                     direction="y",
                                     bin_duration_s=bin_s,
                                     spatial_bin_edges=spatial_bin_edges,
                                     weight_with_amplitude=True,
                                    )
plt.imshow(motion_histogram.T, aspect="auto", vmax=15)

# %%
motion_ks, temporal_bins_ks, non_rigid_window_centers_ks, extra_check_ks = motion_estimation.estimate_motion(
    rec, peaks_sub, peak_locations_sub,
    direction='y', bin_duration_s=bin_s, bin_um=bin_um, margin_um=margin_um,
    rigid=True,
    post_clean=False, speed_threshold=30, sigma_smooth_s=None,
    method='iterative_template',
    output_extra_check=True,
    progress_bar=True,
    upsample_to_histogram_bin=False,
    verbose=False,
)

# %%
motion_dec, temporal_bins_dec, non_rigid_window_centers_dec, extra_check_dec = motion_estimation.estimate_motion(
    rec, peaks_sub, peak_locations_sub,
    direction='y', bin_duration_s=bin_s, bin_um=bin_um, margin_um=margin_um,
    rigid=True,
    post_clean=False, 
    method='decentralized',
    convergence_method="lsqr_robust",
    lsqr_robust_n_iter=1,
    output_extra_check=True,
    progress_bar=True,
    upsample_to_histogram_bin=False,
    time_horizon_s=None,
    verbose=False,
)

# %%
extent = [*extra_check_dec['temporal_hist_bin_edges'][[0, -1]], *extra_check_dec['spatial_hist_bin_edges'][[-1, 0]]]
plt.figure(figsize=(10,10))
plt.imshow(extra_check_dec['motion_histogram'].T, aspect="auto", vmax=15, extent=extent)
plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], -100 + motion_dec + non_rigid_window_centers_dec, color="w", lw=1, label="ours");
plt.plot(extra_check_ks['temporal_hist_bin_edges'][:-1], 100 + motion_ks + non_rigid_window_centers_ks, color="y", lw=1, label="ks");
plt.legend(fancybox=False, framealpha=1);

# %%
D_unnorm, S_unnorm = motion_estimation.compute_pairwise_displacement(motion_histogram, bin_um,
                                                  window=np.ones(motion_histogram.shape[1], dtype=motion_histogram.dtype),
                                                  method="conv",
                                                  weight_scale="linear",
                                                  error_sigma=0,
                                                  conv_engine="torch",
                                                  torch_device=None,
                                                  batch_size=64,
                                                  max_displacement_um=100.0,
                                                  normalized_xcorr=False,
                                                  centered_xcorr=False,
                                                  corr_threshold=0.0,
                                                  time_horizon_s=None,
                                                  bin_duration_s=bin_s,
                                                  progress_bar=False)
D_norm, S_norm = motion_estimation.compute_pairwise_displacement(motion_histogram, bin_um,
                                                  window=np.ones(motion_histogram.shape[1], dtype=motion_histogram.dtype),
                                                  method="conv",
                                                  weight_scale="linear",
                                                  error_sigma=0,
                                                  conv_engine="torch",
                                                  torch_device=None,
                                                  batch_size=64,
                                                  max_displacement_um=100.0,
                                                  normalized_xcorr=True,
                                                  centered_xcorr=False,
                                                  corr_threshold=0.0,
                                                  time_horizon_s=None,
                                                  bin_duration_s=bin_s,
                                                  progress_bar=False)
D_corr, S_corr = motion_estimation.compute_pairwise_displacement(motion_histogram, bin_um,
                                                  window=np.ones(motion_histogram.shape[1], dtype=motion_histogram.dtype),
                                                  method="conv",
                                                  weight_scale="linear",
                                                  error_sigma=0,
                                                  conv_engine="torch",
                                                  torch_device=None,
                                                  batch_size=64,
                                                  max_displacement_um=100.0,
                                                  normalized_xcorr=True,
                                                  centered_xcorr=True,
                                                  corr_threshold=0.0,
                                                  time_horizon_s=None,
                                                  bin_duration_s=bin_s,
                                                  progress_bar=False)
T = D_unnorm.shape[0]

# %%
dss = {
    "unnorm": (D_unnorm, S_unnorm),
    "norm": (D_norm, S_norm),
    "corr": (D_corr, S_corr),
}

# %%
1

# %%
from sklearn.gaussian_process.kernels import Matern


# %%
def newt_solve(D, S, Sigma0inv, normalize=None):
    """D is TxT displacement, S is TxT subsampling or soft weights matrix"""
    
    if normalize == "sym":
        uu = 1/np.sqrt((S + S.T).sum(1))
        S = np.einsum("i,j,ij->ij",uu,uu,S)
    
    # forget the factor of 2, we'll put it in later
    # HS = (S + S.T) - np.diag((S + S.T).sum(1))
    HS = S.copy()
    HS += S.T
    np.fill_diagonal(HS, np.diagonal(HS) - S.sum(1) - S.sum(0))
    # grad_at_0 = (S * D - S.T * D.T).sum(1)
    SD = S * D
    grad_at_0 = SD.sum(1) - SD.sum(0)
    # Next line would be (Sigma0inv ./ 2 .- HS) \ grad in matlab
    p = la.solve(Sigma0inv / 2 - HS, grad_at_0)
    return p, HS


# %%
Sigma0inv_bm = np.eye(T) - np.diag(0.5 * np.ones(T - 1), k=1) - np.diag(0.5 * np.ones(T - 1), k=-1) 

ker = Matern(length_scale=10, nu=0.5)
Sigma0_mat05 = ker(np.arange(T)[:, None])
Sigma0inv_mat05 = la.inv(Sigma0_mat05)

Sigma0invs = {"bm": Sigma0inv_bm, "heavy_mat": 0.5 * T * Sigma0inv_mat05}

# %%
for dstype, (D, S) in dss.items():
    for prior, Sigma0inv in Sigma0invs.items():
        for normtype in (None, "sym"):
            pp, HS = newt_solve(D, S, Sigma0inv, normalize=normtype)
            postcov = la.inv(Sigma0inv - 2 * HS)
            postvar = np.diagonal(postcov)
            
            extent = [*extra_check_dec['temporal_hist_bin_edges'][[0, -1]], *extra_check_dec['spatial_hist_bin_edges'][[-1, 0]]]
            plt.figure(figsize=(10,10))
            plt.imshow(extra_check_dec['motion_histogram'].T, aspect="auto", vmax=15, extent=extent)
            plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], -100 + motion_dec + non_rigid_window_centers_dec, color="w", lw=1, label="ours");
            plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], 0 + pp[:,None] + non_rigid_window_centers_dec, color="b", lw=1, label="test");
            plt.fill_between(
                extra_check_dec['temporal_hist_bin_edges'][:-1],
                (0 + pp[:,None] + non_rigid_window_centers_dec - postvar[:, None] * (10 / postvar.mean())).squeeze(),
                (0 + pp[:,None] + non_rigid_window_centers_dec + postvar[:, None] * (10 / postvar.mean())).squeeze(),
                color="w",
                alpha=0.5,
            )
            plt.plot(extra_check_ks['temporal_hist_bin_edges'][:-1], 100 + motion_ks + non_rigid_window_centers_ks, color="y", lw=1, label="ks");
            plt.title(f"{dstype=} {prior=} {normtype=}")
            plt.legend(fancybox=False, framealpha=1);
            plt.show()

# %%
std * std'

# %%
plt.imshow(D_corr.std(axis=1)[:, None] * D_corr.std(axis=1)[None])

# %%
plt.plot(D_corr.std(axis=1))

# %%

# %%

# %%
# centered brownian motion cov
Sigma_bm = np.minimum(np.arange(T)[:, None], np.arange(T)[None, :])
centering = np.full((T, T), -1/T)
np.fill_diagonal(centering, (T-1)/T)
Sigma0_bm = centering @ Sigma_bm @ centering
Sigma0inv_bm = np.eye(T) - np.diag(0.5 * np.ones(T - 1), k=1) - np.diag(0.5 * np.ones(T - 1), k=-1) 

# Matern cov
ker = Matern(length_scale=10, nu=0.5)
Sigma0_mat05 = ker(np.arange(T)[:, None])
Sigma0inv_mat05 = la.inv(Sigma0_mat05)

ker = Matern(length_scale=10, nu=1.5)
Sigma0_mat15 = ker(np.arange(T)[:, None])
Sigma0inv_mat15 = la.inv(Sigma0_mat15)

# weird idea
Scov = S_unnorm + S_unnorm.T
Svar = np.diagonal(Scov)
Sstd = np.sqrt(Svar)
Ostd = Sstd[:, None] * Sstd[None, :]
Scorr = Scov / Ostd
Sigma0_S_unnorm = np.square(np.maximum(Sstd[:, None], Sstd[None, :])) * Scorr - Scov
np.fill_diagonal(Sigma0_S_unnorm, 1)
Sigma0inv_S_unnorm = la.inv(Sigma0_S_unnorm)

# weird idea
# Sigma0_S_corr = S_corr + S_corr.T
# Sigma0inv_S_corr = la.inv(Sigma0_S_corr)

# other needed matrices
_1 = np.ones(T)
eye = sp.eye(T)

# %%
Scov = (S_unnorm + S_unnorm.T) / 2
Svar = np.diagonal(Scov)
Sstd = np.sqrt(Svar)
Scorr = Scov / (Sstd[:, None] * Sstd[None, :])
S_unnorm_max = np.square(np.maximum(Sstd[:, None], Sstd[None, :])) * Scorr

# %%
plt.imshow(la.inv(Scov)); plt.colorbar()

# %%
qqq = D_corr.std(axis=1)[:, None] * D_corr.std(axis=1)[None]
K = -qqq + np.diag(qqq.sum(1))

# %%
# Sigma0inv = (3 * T) * Sigma0inv_mat05
# Sigma0inv = (0.5 * T) * Sigma0inv_mat05
# Sigma0inv = (0.5 * T) * Sigma0inv_mat15
# Sigma0inv = (100 * T) * Sigma0inv_mat05
Sigma0inv = Sigma0inv_bm
# Sigma0inv = D_corr.std(axis=1)[:, None] * D_corr.std(axis=1)[None]
# Sigma0inv = 0.5 * T * Sigma0inv_S_unnorm
# Sigma0inv = Sigma0inv_S_unnorm
# Sigma0inv = 20 * (Sigma0_S_unnorm + 0.025 * T * Sigma0inv_mat05)
# pp, HS = newt_solve(D_unnorm, S_unnorm, Sigma0inv_bm)
pp, HS = newt_solve(D_corr, K, Sigma0inv)

# pp, HS = newt_solve(D_unnorm, -la.inv(Scov), Sigma0inv)
# pp, HS = newt_solve(D_corr, S_corr, Sigma0inv)
# pp, HS = newt_solve(D_corr, S_corr, (0.5 * T) * Sigma0inv_bm)

# %%
postcov = la.inv(Sigma0inv - 2 * HS)
postvar = np.diagonal(postcov)

# %%
plt.plot(postvar);

# %%
extent = [*extra_check_dec['temporal_hist_bin_edges'][[0, -1]], *extra_check_dec['spatial_hist_bin_edges'][[-1, 0]]]
plt.figure(figsize=(10,10))
plt.imshow(extra_check_dec['motion_histogram'].T, aspect="auto", vmax=15, extent=extent)
plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], -100 + motion_dec + non_rigid_window_centers_dec, color="w", lw=1, label="ours");
plt.plot(extra_check_dec['temporal_hist_bin_edges'][:-1], 0 + pp[:,None] + non_rigid_window_centers_dec, color="b", lw=1, label="test");
plt.fill_between(
    extra_check_dec['temporal_hist_bin_edges'][:-1],
    (0 + pp[:,None] + non_rigid_window_centers_dec - postvar[:, None] * (10 / postvar.mean())).squeeze(),
    (0 + pp[:,None] + non_rigid_window_centers_dec + postvar[:, None] * (10 / postvar.mean())).squeeze(),
    color="w",
    alpha=0.5,
)
plt.plot(extra_check_ks['temporal_hist_bin_edges'][:-1], 100 + motion_ks + non_rigid_window_centers_ks, color="y", lw=1, label="ks");
plt.legend(fancybox=False, framealpha=1);

# %%

# %%

# %%
D = D[:T // 2, :T // 2]
S = S[:T // 2, :T // 2]
T = T // 2

# %%
# centered brownian motion cov
Sigma_bm = np.minimum(np.arange(T)[:, None], np.arange(T)[None, :])
centering = np.full((T, T), -1/T)
np.fill_diagonal(centering, (T-1)/T)
Sigma0 = centering @ Sigma_bm @ centering

# other needed matrices
_1 = np.ones(T)
eye = sp.eye(T)

# %%
plt.imshow(la.pinv(Sigma0))

# %%

# %%
e, V = la.eigh(Sigma0)
# this matrix was degenerate
e = e[1:]
V = V[:, 1:]
# Sigma0 ~= v @ diag(s) @ v.T
np.isclose(Sigma0, (V @ np.diag(e) @ V.T)).all()

# %%
np.isclose(la.pinv(Sigma0), (V @ np.diag(1/e) @ V.T)).all()


# %%
def spdia(vals):
    return sp.dia_matrix((vals[None], [0]), shape=(vals.size, vals.size)).tocsr()


# %%
D = spdia(S.ravel())
Dinv = spdia(1 / S.ravel())
U = sp.kron(V, eye, "csr")
E = spdia(e)
C = sp.kron(E, eye, "csr")
Cinv = sp.kron(spdia(1/e), eye)

# %%
# U.T @ Dinv @ U


# %%
woodbury_inner = Cinv + U.T @ Dinv @ U

# %%
woodbury_inner.shape

# %%
woodbury_inner.nnz / np.prod(woodbury_inner.shape)

# %%
woodbury_inner.nnz / T

# %%
woodbury_inner.nnz / (T**2)

# %%
woodbury_inner.nnz / (T**3)

# %%
1/T  # (aka T^3/T^4)

# %%
woodbury_inv = sp.linalg.inv(woodbury_inner)

# %%
