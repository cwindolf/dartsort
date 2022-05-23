import argparse
import numpy as np
from pathlib import Path
import h5py
import torch
from spike_psvae import (
    denoise,
    ibme,
    residual,
    deconvolve,
    merge_split_cleaned,
    relocalize_after_deconv,
    after_deconv_merge_split,
)

# import matplotlib.pyplot as plt
# from tqdm.auto import tqdm


ap = argparse.ArgumentParser()

ap.add_argument("spike_index_npy")
ap.add_argument("labels_npy")
ap.add_argument("standardized_path")
ap.add_argument("subtraction_dir")
ap.add_argument("output_directory")

args = ap.parse_args()

spike_index = np.load(args.spike_index_npy)
labels = np.load(args.labels_npy)
assert len(spike_index) == len(labels)

standardized_path = Path(args.standardized_path)

sub_dir = Path(args.subtraction_dir)
h5_subtract = next(sub_dir.glob("sub*.h5"))
residual_path = next(sub_dir.glob("res*.bin"))

base_outdir = Path(args.output_directory)
first_outdir = base_outdir / "first_deconv_results"
second_outdir = base_outdir / "second_deconv_results"
for d in (base_outdir, first_outdir, second_outdir):
    d.mkdir(exist_ok=True)

with h5py.File(h5_subtract) as h5:
    # fs = h5["fs"][()]
    fs = 30_000
    geom_array = h5["geom"][:]
geom_path = base_outdir / "geom.npy"
np.save(geom_path, geom_array)

# Run deconvolution
# get_templates reads so that templates have trough at 42
templates_raw = merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(labels).shape[0] - 1,
    spike_index[labels >= 0],
    labels[labels >= 0],
    max_spikes=250,
    n_times=121,
)

# Align templates/spike index to 42!!
# Needed for later? - deconvolve.read_waveforms reads from -60 not -42
# SHOULD WE CHANGE THIS?
for i in range(templates_raw.shape[0]):
    mc = templates_raw[i].ptp(0).argmax(0)
    if templates_raw[i, :, mc].argmin() != 42:
        spike_index[labels == i, 0] += templates_raw[i, :, mc].argmin() - 42

template_spike_train = np.c_[
    spike_index[labels >= 0][:, 0], labels[labels >= 0]
]

result_file_names = deconvolve.deconvolution(
    spike_index[labels >= 0],
    labels[labels >= 0],
    first_outdir,
    standardized_path,
    residual_path,
    template_spike_train,
    geom_path,
    multi_processing=False,
    cleaned_temps=True,
    n_processors=6,
    threshold=40,
)
print(result_file_names)

# Compute residual
residual_path = residual.run_residual(
    result_file_names[0],
    result_file_names[1],
    first_outdir,
    standardized_path,
    geom_path,
    multi_processing=True,
    n_processors=6,
)


"""
CODE TO CHECK RESIDUALS LOOK GOOD

start = 0
viz_len = 1000
n_chans = 384
img = np.fromfile(standardized_path,
                  dtype=np.float32,
                  count=n_chans*viz_len,
                  offset=4*start*n_chans).reshape((viz_len,n_chans))
residual_img = np.fromfile(residual_path,
                  dtype=np.float32,
                  count=n_chans*viz_len,
                           offset=4*start*n_chans).reshape((viz_len,n_chans))

vmin = min(img.min(), residual_img.min())
vmax = max(img.max(), residual_img.max())

fig, axs = plt.subplots(1,2, sharey=True, figsize=(14,6))
axs[0].imshow(img.T, aspect='auto', vmin=vmin, vmax=vmax)
axs[1].imshow(residual_img.T, aspect='auto', vmin=vmin, vmax=vmax)
plt.show()

"""

# Extract subtracted, collision-subtracted, denoised waveforms

# load denoiser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoiser = denoise.SingleChanDenoiser()
denoiser.load()
denoiser.to(device)

deconv_spike_train_up = np.load(result_file_names[1])
deconv_templates_up = np.load(result_file_names[0])

n_spikes = deconv_spike_train_up.shape[0]
print(f"number of deconv spikes: {n_spikes}")
print(f"deconv templates shape: {deconv_templates_up.shape}")

# 42/60 issue :
# deconvolve.read_waveforms used in this function reads at t-60:t+60
# and pass wfs through denoising pipeline

# Save all wfs in first_outdir
n_chans_to_extract = 40

(
    fname_spike_index,
    fname_spike_labels,
    fname_subtracted,
    cleaned_wfs_h5,
    denoised_wfs_h5,
) = relocalize_after_deconv.extract_deconv_wfs(
    h5_subtract,
    residual_path,
    geom_array,
    deconv_spike_train_up,
    deconv_templates_up,
    first_outdir,
    denoiser,
    device,
    n_chans_to_extract=n_chans_to_extract,
)

# Relocalize Waveforms

deconv_spike_index = np.load(fname_spike_index)
# assert deconv_spike_index.shape[0] == n_spikes
print(f"number of deconv spikes: {deconv_spike_index.shape[0]}")

relocalize_after_deconv.relocalize_extracted_wfs(
    denoised_wfs_h5,
    deconv_spike_train_up,
    deconv_spike_index,
    geom_array,
    first_outdir,
)

localization_results_path = first_outdir / "localization_results.npy"
maxptpss = np.load(localization_results_path)[:, 4]
z_absss = np.load(localization_results_path)[:, 1]
times = deconv_spike_train_up[:, 0].copy() / fs


# # Check localization results output
# raster, dd, tt = ibme.fast_raster(maxptpss, z_absss, times)
# plt.figure(figsize=(16,12))
# plt.imshow(raster, aspect='auto')


# Register
z_reg, dispmap = ibme.register_nonrigid(
    maxptpss,
    z_absss,
    times,
    robust_sigma=1,
    rigid_disp=200,
    disp=100,
    denoise_sigma=0.1,
    destripe=False,
    n_windows=10,
    widthmul=0.5,
)
z_reg -= (z_reg - z_absss).mean()
dispmap -= dispmap.mean()
np.save(first_outdir / "z_reg.npy", z_reg)
np.save(first_outdir / "ptps.npy", maxptpss)

# # Check registration output
# registered_raster, dd, tt = ibme.fast_raster(maxptpss, z_reg, times)
# plt.figure(figsize=(16,12))
# plt.imshow(registered_raster, aspect='auto')

# After Deconv Split Merge

deconv_spike_index = np.load(first_outdir / "spike_index.npy")
z_abs = np.load(first_outdir / "localization_results.npy")[:, 1]
firstchans = np.load(first_outdir / "localization_results.npy")[:, 5]
maxptps = np.load(first_outdir / "localization_results.npy")[:, 4]
spike_train_deconv = np.load(first_outdir / "spike_train.npy")
xs = np.load(first_outdir / "localization_results.npy")[:, 0]
z_reg = np.load(first_outdir / "z_reg.npy")

templates_after_deconv = merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0] - 1,
    deconv_spike_index,
    spike_train_deconv[:, 1],
    max_spikes=250,
    n_times=121,
)


for i in range(templates_after_deconv.shape[0]):
    mc = templates_after_deconv[i].ptp(0).argmax(0)
    if templates_after_deconv[i, :, mc].argmin() != 42:
        spike_train_deconv[spike_train_deconv[:, 1] == i, 0] += (
            templates_after_deconv[i, :, mc].argmin() - 42
        )

templates_after_deconv = merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0] - 1,
    deconv_spike_index,
    spike_train_deconv[:, 1],
    max_spikes=250,
    n_times=121,
)


split_labels = after_deconv_merge_split.split(
    spike_train_deconv[:, 1],
    templates_after_deconv,
    maxptps,
    firstchans,
    denoised_wfs_h5,
)

templates_geq_4 = merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0] - 1,
    deconv_spike_index[maxptps > 4],
    spike_train_deconv[maxptps > 4, 1],
    max_spikes=250,
    n_times=121,
)

# take ptp > 4 before next step of deconv

merged_labels = after_deconv_merge_split.merge(
    spike_train_deconv[maxptps > 4, 1],
    templates_geq_4,
    cleaned_wfs_h5,
    xs[maxptps > 4],
    z_reg[maxptps > 4],
    maxptps[maxptps > 4],
)


# Additional Deconv

which = np.flatnonzero(np.logical_and(maxptps > 4, split_labels >= 0))
spt_deconv_after_merge = spike_train_deconv[which]
spt_deconv_after_merge[:, 1] = merged_labels[split_labels >= 0]

spike_index_DAM = np.zeros(spt_deconv_after_merge.shape)
spike_index_DAM[:, 0] = spt_deconv_after_merge[:, 0].copy()
for i in range(templates_geq_4.shape[0]):
    spike_index_DAM[spt_deconv_after_merge[:, 1] == i, 1] = (
        templates_geq_4[i].ptp(0).argmax()
    )

trough_offset = 42
max_time = 5 * 60 * 30000
which = (spt_deconv_after_merge[:, 0] > trough_offset) & (
    spt_deconv_after_merge[:, 0] < max_time - (121 - trough_offset)
)

result_file_names = deconvolve.deconvolution(
    spike_index_DAM[which],
    spt_deconv_after_merge[which, 1],
    second_outdir,
    standardized_path,
    residual_path,
    spt_deconv_after_merge[which],
    geom_path,
    multi_processing=False,
    cleaned_temps=True,
    n_processors=6,
    threshold=40,
)


# Following steps are optional (compute residuals and relocalize )

residual_path = residual.run_residual(
    result_file_names[0],
    result_file_names[1],
    second_outdir,
    standardized_path,
    geom_path,
    multi_processing=True,
    n_processors=6,
)


deconv_spike_train_up = np.load(result_file_names[1])
deconv_templates_up = np.load(result_file_names[0])

n_spikes = deconv_spike_train_up.shape[0]
print(f"number of deconv spikes: {n_spikes}")
print(f"deconv templates shape: {deconv_templates_up.shape}")

# 42/60 issue : deconvolve.read_waveforms used in this function reads at t-60:t+60
# and pass wfs through denoising pipeline

# Save all wfs in first_outdir
n_chans_to_extract = 40

(
    fname_spike_index,
    fname_spike_labels,
    fname_subtracted,
    cleaned_wfs_h5,
    denoised_wfs_h5,
) = relocalize_after_deconv.extract_deconv_wfs(
    h5_subtract,
    residual_path,
    geom_array,
    deconv_spike_train_up,
    deconv_templates_up,
    second_outdir,
    denoiser,
    device,
    n_chans_to_extract=n_chans_to_extract,
)


# Relocalize Waveforms

deconv_spike_index = np.load(fname_spike_index)
# assert deconv_spike_index.shape[0] == n_spikes
print(f"number of deconv spikes: {deconv_spike_index.shape[0]}")

relocalize_after_deconv.relocalize_extracted_wfs(
    denoised_wfs_h5,
    deconv_spike_train_up,
    deconv_spike_index,
    geom_array,
    second_outdir,
)

localization_results_path = second_outdir / "localization_results.npy"
maxptpss = np.load(localization_results_path)[:, 4]
z_absss = np.load(localization_results_path)[:, 1]
times = deconv_spike_train_up[:, 0].copy() / fs


# Register

z_reg, dispmap = ibme.register_nonrigid(
    maxptpss,
    z_absss,
    times,
    robust_sigma=1,
    rigid_disp=200,
    disp=100,
    denoise_sigma=0.1,
    destripe=False,
    n_windows=10,
    widthmul=0.5,
)
z_reg -= (z_reg - z_absss).mean()
dispmap -= dispmap.mean()
np.save(second_outdir / "z_reg.npy", z_reg)
np.save(second_outdir / "ptps.npy", maxptpss)

# # Check registration output
# registered_raster, dd, tt = ibme.fast_raster(maxptpss, z_reg, times)
# plt.figure(figsize=(16,12))
# plt.imshow(registered_raster, aspect='auto')

# After Deconv Split Merge

deconv_spike_index = np.load(second_outdir / "spike_index.npy")
z_abs = np.load(second_outdir / "localization_results.npy")[:, 1]
firstchans = np.load(second_outdir / "localization_results.npy")[:, 5]
maxptps = np.load(second_outdir / "localization_results.npy")[:, 4]
spike_train_deconv = np.load(second_outdir / "spike_train.npy")
xs = np.load(second_outdir / "localization_results.npy")[:, 0]
z_reg = np.load(second_outdir / "z_reg.npy")

templates_after_deconv = merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0] - 1,
    deconv_spike_index,
    spike_train_deconv[:, 1],
    max_spikes=250,
    n_times=121,
)
