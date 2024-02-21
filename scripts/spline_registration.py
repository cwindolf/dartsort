from pathlib import Path
import numpy as np
import h5py
from tqdm.auto import tqdm, trange
import scipy.io
import hdbscan
import os 

from dredge import dredge_ap, motion_util as mu


import spikeinterface.core as sc
import spikeinterface.full as si

from spike_psvae import cluster_viz, cluster_utils
from spike_psvae.ibme import register_nonrigid, fast_raster
from spike_psvae.ibme_corr import calc_corr_decent, psolvecorr

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

from dartsort.config import TemplateConfig, MatchingConfig, ClusteringConfig, SubtractionConfig, FeaturizationConfig
from dartsort import cluster

# Input
data_dir = Path("UHD_DATA/ZYE_0021___2021-05-01___1")
fname_sub_h5 = data_dir / "subtraction.h5"
raw_data_name = data_dir / "standardized.bin"
dtype_preprocessed = "float32"
sampling_rate = 30000
n_channels = 384

rec_len_sec = int(os.path.getsize(raw_data_name)/4/n_channels/sampling_rate)

output_dir = data_dir / "spline_drift_results"
os.makedirs(Path(output_dir), exist_ok=True)
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

clustering_config_uhd = ClusteringConfig(
    cluster_strategy="density_peaks",
    sigma_regional=25,
    chunk_size_s=300,
    cluster_selection_epsilon=1,
    min_cluster_size = 25,
    min_samples = 25,
    recursive=False,
    remove_big_units=True,
    zstd_big_units=50.0,
)

with h5py.File(sub_h5, "r+") as h5:
    localization_results = np.array(h5["point_source_localizations"][:])
    maxptps = np.array(h5["denoised_ptp_amplitudes"][:])
    times_samples = np.array(h5["times_samples"][:])
    times_seconds = np.array(h5["times_seconds"][:])
    geom = np.array(h5["geom"][:])
    channels = np.array(h5["channels"][:])


recording = sc.read_binary(
    raw_data_name,
    sampling_rate,
    dtype_preprocessed,
    num_channels=n_channels,
    is_filtered=True,
)

recording.set_dummy_probe_from_locations(
    geom, shape_params=dict(radius=10)
)

z = localization_results[:, 2]
wh = (z > geom[:,1].min() - 100) & (z < geom[:,1].max() + 100)
a = maxptps[wh]
z = z[wh]
t = times_seconds[wh]
    
me, extra = dredge_ap.register(a, z, t, max_disp_um=100, win_scale_um=300, win_step_um=200, rigid=False, mincorr=0.6)

z_reg = me.correct_s(times_seconds, localization_results[:, 2])
displacement_rigid = me.displacement

# Rigid reg
idx = np.flatnonzero(maxptps>threshold_ptp_rigid_reg)

if savefigs:
    ptp_col = np.log(maxptps[idx]+1)
    ptp_col[ptp_col>3.5]=3.5
    ptp_col -= ptp_col.min()
    ptp_col /= ptp_col.max()
    color_array = vir(ptp_col)
    plt.figure(figsize=(20, 10))
    plt.scatter(times_seconds[idx], z_reg[idx], s=1, color = color_array)
    plt.plot(displacement_rigid+geom.max()/2, c='red')
    plt.savefig(Path(output_dir) / "initial_detection_localization.png")
    plt.close()

idx = np.flatnonzero(maxptps>threshold_ptp_spline)
 
sorting = cluster(
    sub_h5,
    recording,
    clustering_config=clustering_config_uhd, 
    motion_est=me)

if savefigs:
    cluster_viz.array_scatter(sorting.labels, 
                                  geom, localization_results[:, 0], z_reg, 
                                  maxptps, zlim=(-50, 332))
    plt.savefig(Path(output_dir) / "clustering_high_ptp_units.png")
    plt.close()

z_centered = localization_results[sorting.labels>-1, 2].copy()
for k in range(sorting.labels.max()+1):
    idx_k = np.flatnonzero(sorting.labels[sorting.labels>-1] == k)
    z_centered[idx_k] -= z_centered[idx_k].mean()
z_centered_std = z_centered.std()

values = z_centered[np.abs(z_centered)<z_centered_std*std_bound]
idx_times = np.flatnonzero(sorting.labels>-1)
times = times_seconds[idx_times][np.abs(z_centered)<z_centered_std*std_bound]

transformer = SplineTransformer(
            degree=spline_degree,
            n_knots=int(rec_len_sec*2.5), #can find this automatically - 2 per period (of ~1.25sec)
        )

model = make_pipeline(transformer, Ridge(alpha=1e-3))
model.fit(times.reshape(-1, 1), values.reshape(-1, 1))

print("Inference")
spline_displacement = np.zeros(rec_len_sec*sampling_rate)
n_batches = rec_len_sec//batch_size_spline
for batch_id in tqdm(range(n_batches)):
    idx_batch = np.arange(batch_id*batch_size_spline*sampling_rate, (batch_id+1)*batch_size_spline*sampling_rate, )
    spline_displacement[idx_batch] = model.predict(idx_batch.reshape(-1, 1)/sampling_rate)[:, 0]

z_reg_spline = z_reg - spline_displacement[times_samples.astype('int')]
if savefigs:
    idx = np.flatnonzero(maxptps>threshold_ptp_rigid_reg)
    ptp_col = np.log(maxptps[idx]+1)
    ptp_col[ptp_col>3.5]=3.5
    ptp_col -= ptp_col.min()
    ptp_col /= ptp_col.max()
    color_array = vir(ptp_col)
    plt.figure(figsize=(20, 10))
    plt.scatter(times_seconds[idx], z_reg_spline[idx], s=1, color = color_array)
    plt.savefig(Path(output_dir) / "final_registered_raster.png")
    plt.close()


np.save("high_freq_correction.npy", spline_displacement)
np.save("spline_registered_z.npy", z_reg_spline)

