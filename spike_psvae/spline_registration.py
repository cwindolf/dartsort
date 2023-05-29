import os
from pathlib import Path
import numpy as np
import h5py
import scipy.io
import time
import torch
import shutil
import pickle

from sklearn.decomposition import PCA
from spike_psvae import filter_standardize
from spike_psvae.cluster_uhd import run_full_clustering
from spike_psvae.drifty_deconv_uhd import full_deconv_with_update, get_registered_pos
from spike_psvae import subtract, ibme
from spike_psvae.ibme import register_nonrigid
from spike_psvae.ibme_corr import calc_corr_decent
from spike_psvae.ibme import fast_raster
from spike_psvae.ibme_corr import psolvecorr
from spike_psvae.waveform_utils import get_pitch
from spike_psvae.post_processing_uhd import full_post_processing
import spike_psvae.newton_motion_est as newt
import spike_psvae.motion_utils as mu
from spike_psvae import cluster_viz, cluster_utils

from matplotlib import gridspec
from celluloid import Camera
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorcet as ccet

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from scipy.io import loadmat
from scipy.signal import lombscargle


def run_spline_reg(pt, all_output_dir, fs=30000, savefigs=True, 
        threshold_ptp_rigid_reg = 20,
        threshold_ptp_spline = 20,
        spline_degree=5,
        min_cluster_size=100,
        cluster_selection_epsilon=1,
        std_bound=2,
        min_fr=10,
        batch_size_spline=100_000,
    ):
    print(pt)
    pt_outdir = all_output_dir / pt

    # check if done
    assert (pt_outdir / "rez2.mat").exists()

    # load spikes from kilosort
    with h5py.File(pt_outdir / "rez2.mat", "r") as h5:
        st0 = h5["rez"]["st0"][:]
    # times in samples
    t_samples = st0[0]
    # depths
    z = st0[1]
    # KS amplitudes
    a = st0[2]
    # times in seconds
    t_s = t_samples / fs

    h = loadmat(pt_outdir / "ppx_chanMap.mat")
    geom = np.c_[h['xcoords'], h['ycoords']]

    # load our rigid motion estimate
    with open(pt_outdir / "dredgeapksspikes_me.pkl", "rb") as jar:
        dredge_ap_rigid_me = pickle.load(jar)["me"]
    # our registered z positions
    z_reg = dredge_ap_rigid_me.correct_s(t_s, z)

    if savefigs:

        fname_before_fig = Path(pt_outdir) / "dredge_output.png"

        fig, (aa, ab) = plt.subplots(ncols=2, figsize=(10, 5))
        raster, dbe, tbe = mu.fast_raster(a, z, t_s)
        mu.show_raster(raster, dbe, tbe, aa, aspect="auto", vmax=15, cmap=plt.cm.binary)
        mu.plot_me_traces(dredge_ap_rigid_me, aa)
        mu.show_registered_raster(dredge_ap_rigid_me, a, z, t_s, ab,  aspect="auto", vmax=15, cmap=plt.cm.binary)
        aa.set_title("unreg raster with motion trace")
        ab.set_title("dredge ap rigid registered")
        plt.savefig(fname_before_fig)
        plt.close(fig)

    
    max_channels = np.abs(z_reg[:, None] - geom[:, 1][None]).argmin(1)
    
    idx = np.flatnonzero(a>threshold_ptp_rigid_reg)

    clusterer, cluster_centers, spike_index_cluster, x_cluster, z_cluster, maxptps_cluster, original_spike_ids = cluster_utils.cluster_spikes(
            8*np.ones(len(idx)), 
            z_reg[idx],
            a[idx], 
            np.c_[t_s*30000, max_channels][idx], triage_quantile=100, 
            do_copy_spikes=False, split_big=False, do_remove_dups=False, 
            min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)

    if savefigs:
        
        fname_cluster_fig = Path(pt_outdir) / "clustering_output.png"
        plt.figure(figsize=(20, 10))
        for k in np.unique(clusterer.labels_):
            idx_unit = idx[clusterer.labels_==k]
            # plt.scatter(t_s[idx], z_reg[idx], s=1, color = color_array)
            plt.scatter(t_s[idx_unit], z_reg[idx_unit], c = ccolors[k%31], s = 1, alpha = 0.05)
        plt.ylim((0, geom.max()))
        plt.title("clustered spikes")
        plt.savefig(fname_cluster_fig)
        plt.close()


    idx_model_fit = idx[clusterer.labels_>-1][z_reg[idx[clusterer.labels_>-1]]<geom.max()]

    z_centered = z_reg[idx_model_fit].copy()

    for k in range(clusterer.labels_.max()+1):
        idx_k = np.flatnonzero(clusterer.labels_[clusterer.labels_>-1][z_reg[idx[clusterer.labels_>-1]]<geom.max()] == k)
        z_centered[idx_k] -= np.median(z_centered[idx_k]) #.mean()
    z_centered_std = z_centered.std()

    values = z_centered[np.abs(z_centered)<z_centered_std*std_bound]
    # idx_times = np.flatnonzero(clusterer.labels_>-1)[z_reg[idx[clusterer.labels_>-1]]<geom.max()]
    times = t_s[idx_model_fit][np.abs(z_centered)<z_centered_std*std_bound]

    # to implement to make sure FR > ~10spikes per period
    # valid_times, fr = np.unique(times//1, return_counts=True)
    # valid_times = valid_times[fr > min_fr]


    periods = np.linspace(0.01, 1, 1000) #Periods to calculate signal strength
    angular_frequencies = 2 * np.pi * 1.0 / periods

    periodogram = lombscargle(times, values, angular_frequencies)
    normalized_periodogram = np.sqrt(4 * (periodogram / values.shape[0]))

    if savefigs:
        fname_spectro_fig = Path(pt_outdir) / "spectrogram.png"
        plt.figure(figsize=(14,4))
        plt.plot(periods, periodogram, linewidth=1.5)
        plt.xlabel('Periods', size=14)
        plt.ylabel('Spectrum', size=14)
        plt.savefig(fname_spectro_fig)
        plt.close()

    transformer = SplineTransformer(
                degree=spline_degree,
                n_knots=int(times.max()*3/periods[periodogram.argmax()]), #can find this automatically - 2 per period (of ~1.25sec)
            )
    model = make_pipeline(transformer, Ridge(alpha=1e-3))
    model.fit(times.reshape(-1, 1), values.reshape(-1, 1))


    if savefigs:
        fname_spectro_fig = Path(pt_outdir) / "spline_fit.png"
        plt.figure(figsize=(20, 5))
        plt.scatter(times, values, s=1, c='blue')
        plt.plot(times, model.predict(times.reshape(-1, 1))[:, 0], c='red')
        plt.savefig(fname_spectro_fig)
        plt.close()


    inferred_values = np.zeros(len(t_s))
    n_batches = len(t_s)//batch_size_spline+1
    for k in tqdm(range(n_batches)):
        inferred_values[k*batch_size_spline:(k+1)*batch_size_spline] = model.predict(t_s[k*batch_size_spline:(k+1)*batch_size_spline].reshape(-1, 1))[:, 0]

    if savefigs:

        ptp_col = np.log(a+1)
        ptp_col[ptp_col>3.5]=3.5
        ptp_col -= ptp_col.min()
        ptp_col /= ptp_col.max()
        color_array_all = vir(ptp_col)

        fname_final_fig = Path(pt_outdir) / "final_reg.png"
        fig, (aa, ab) = plt.subplots(ncols=2, figsize=(20, 5))
        aa.scatter(t_s, z_reg, s=0.5, color = color_array_all, alpha = ptp_col/ptp_col.max())
        aa.set_title("Dredge registered raster")
        ab.scatter(t_s, z_reg - inferred_values, s=0.5, color = color_array_all, alpha = ptp_col/ptp_col.max())
        ab.set_title("Splines registered raster")
        plt.savefig(fname_before_fig)
        plt.close(fig)

