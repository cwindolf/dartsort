from pathlib import Path
import numpy as np
import h5py
from tqdm.auto import tqdm, trange
import scipy.io
import hdbscan

from spike_psvae import cluster_viz, cluster_utils
from spike_psvae.ibme import register_nonrigid, fast_raster
from spike_psvae.ibme_corr import calc_corr_decent, psolvecorr

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline



# Input
fname_sub_h5 = "subtraction.h5"
sampling_rate=30000
rec_len_sec = 300 # Duration of the recording in seconds
output_dir = "spline_drift_results"
Path(output_dir).mkdir(exist_ok=True)
savefigs=True
if savefigs:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    vir = cm.get_cmap('viridis')

#parameters of output displacement
disp_resolution=30000

# Parameters - I recommend keeping them to default
threshold_ptp_rigid_reg = 8
threshold_ptp_spline = 15
std_bound=5
spline_degree=5
batch_size_spline=20

sub_h5 = Path(fname_sub_h5)

with h5py.File(sub_h5, "r+") as h5:
    geom = h5["geom"][:]
    localization_results = np.array(h5["localizations"][:]) #f.get('localizations').value
    maxptps = np.array(h5["maxptps"][:])
    spike_index = np.array(h5["spike_index"][:])
    z_reg = np.array(h5["z_reg"][:])

if savefigs:
    idx = maxptps>threshold_ptp_rigid_reg
    ptp_col = np.log(maxptps[idx]+1)
    ptp_col[ptp_col>3.5]=3.5
    ptp_col -= ptp_col.min()
    ptp_col /= ptp_col.max()
    color_array = vir(ptp_col)
    plt.figure(figsize=(20, 10))
    plt.scatter(spike_index[idx, 0], z_reg[idx], s=1, color = color_array)
    plt.savefig(Path(output_dir) / "initial_detection_localization.png")
    plt.close()

# Rigid reg
idx = np.flatnonzero(maxptps>threshold_ptp_rigid_reg)

raster, dd, tt = fast_raster(maxptps[idx], z_reg[idx], spike_index[idx, 0]/sampling_rate)
D, C = calc_corr_decent(raster, disp = 100)
displacement_rigid = psolvecorr(D, C)

if savefigs:
    ptp_col = np.log(maxptps[idx]+1)
    ptp_col[ptp_col>3.5]=3.5
    ptp_col -= ptp_col.min()
    ptp_col /= ptp_col.max()
    color_array = vir(ptp_col)
    plt.figure(figsize=(20, 10))
    plt.scatter(spike_index[idx, 0], z_reg[idx], s=1, color = color_array)
    plt.plot(displacement_rigid+geom.max()/2, c='red')
    plt.savefig(Path(output_dir) / "initial_detection_localization.png")
    plt.close()

idx = np.flatnonzero(maxptps>threshold_ptp_spline)

clusterer, cluster_centers, spike_index_cluster, x, z, maxptps_cluster, original_spike_ids = cluster_utils.cluster_spikes(
    localization_results[idx, 0], 
    z_reg[idx]-displacement_rigid[spike_index[idx, 0]//sampling_rate],
    maxptps[idx], spike_index[idx], triage_quantile=100, do_copy_spikes=False, split_big=False, do_remove_dups=False)

if savefigs:
    cluster_viz.array_scatter(clusterer.labels_, 
                                  geom, x, z, 
                                  maxptps_cluster)
    plt.savefig(Path(output_dir) / "clustering_high_ptp_units.png")
    plt.close()

z_centered = z[clusterer.labels_>-1].copy()
for k in range(clusterer.labels_.max()+1):
    idx_k = np.flatnonzero(clusterer.labels_[clusterer.labels_>-1] == k)
    z_centered[idx_k] -= z_centered[idx_k].mean()
z_centered_std = z_centered.std()

values = z_centered[np.abs(z_centered)<z_centered_std*std_bound]
idx_times = np.flatnonzero(clusterer.labels_>-1)
times = spike_index_cluster[idx_times, 0][np.abs(z_centered)<z_centered_std*std_bound]/sampling_rate

transformer = SplineTransformer(
            degree=spline_degree,
            n_knots=int(rec_len_sec*2.5), #can find this automatically - 2 per period (of ~1.25sec)
        )
model = make_pipeline(transformer, Ridge(alpha=1e-3))
model.fit(times.reshape(-1, 1), values.reshape(-1, 1))

if savefigs:
    plt.figure(figsize=(20, 5))
    plt.scatter(times, values, s=1, c='blue')
    plt.plot(times, model.predict(times.reshape(-1, 1))[:, 0], c='red')
    plt.savefig(Path(output_dir) / "spline_fit.png")
    plt.close()

print("Inference")
spline_displacement = np.zeros(rec_len_sec*sampling_rate)
n_batches = rec_len_sec//batch_size_spline
for batch_id in tqdm(range(n_batches)):
    idx_batch = np.arange(batch_id*batch_size_spline*sampling_rate, (batch_id+1)*batch_size_spline*sampling_rate, )
    spline_displacement[idx_batch] = model.predict(idx_batch.reshape(-1, 1)/sampling_rate)[:, 0]

if savefigs:
    idx = np.flatnonzero(maxptps>threshold_ptp_rigid_reg)
    ptp_col = np.log(maxptps[idx]+1)
    ptp_col[ptp_col>3.5]=3.5
    ptp_col -= ptp_col.min()
    ptp_col /= ptp_col.max()
    color_array = vir(ptp_col)
    plt.figure(figsize=(20, 10))
    plt.scatter(spike_index[idx, 0], z_reg[idx]-displacement_rigid[spike_index[idx, 0]//30000] - spline_displacement[spike_index[idx, 0].astype('int')], s=1, color = color_array)
    plt.savefig(Path(output_dir) / "final_registered_raster.png")
    plt.close()

np.save("low_freq_disp.npy", displacement_rigid)
np.save("high_freq_correction.npy", spline_displacement)
