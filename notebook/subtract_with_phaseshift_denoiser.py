# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from spike_psvae import subtract
import spikeinterface.full as sf
import spikeinterface.preprocessing as si
import spikeinterface.extractors as se
from pathlib import Path

# %%
cbin_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_69c9a415-f7fa-4208-887b-1417c1479b48_probe_probe00_pID_1a276285-8b0e-4cc9-9f0a-a3a002978724'

# %%
rec_cbin = sf.read_cbin_ibl(Path(cbin_dir))
destriped_cbin = cbin_dir + '/destriped__spikeglx_ephysData_g0_t0.imec0.ap.stream.cbin'

rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

# %%
import cProfile

# %%
import torch
torch.cuda.is_available()

# %% jupyter={"outputs_hidden": true}
cProfile.run("sub_h5 = subtract.subtraction(rec,out_folder=cbin_dir,thresholds=[12, 10, 8, 6, 5],n_sec_pca=40,save_subtracted_tpca_projs=False,save_cleaned_tpca_projs=False,save_denoised_tpca_projs=False,save_denoised_waveforms=True,do_phaseshift = True,n_jobs=1,loc_workers=4,overwrite=True,device = \"cuda\",save_cleaned_pca_projs_on_n_channels=None,loc_feature= (\"ptp\", \"peak\"),out_filename=\"test_n_14_parallelized_subtraction_again.h5\", enforce_decrease_kind=\"none\")", "restats")

# %%
sub_h5 = subtract.subtraction(
                        rec,
                        out_folder=cbin_dir,
                        # thresholds=[12, 10],
                        thresholds=[12, 10, 8, 6, 5],
                        n_sec_pca=40,
                        # save_cleaned_pca_projs_on_n_channels=None,
                        save_subtracted_tpca_projs=False,
                        save_cleaned_tpca_projs=False,
                        save_denoised_tpca_projs=False,
                        save_denoised_waveforms=True,
                        do_phaseshift = True,
                        n_jobs=1,
                        loc_workers=4,
                        overwrite=False,
                        device = "cuda",
                        # n_sec_chunk=args.batchlen,
                        save_cleaned_pca_projs_on_n_channels=None,
                        loc_feature= ("ptp", "peak"),
                        out_filename="test_n_14_parallelized_subtraction_again.h5", 
                        enforce_decrease_kind="none"
                    )

# %% jupyter={"outputs_hidden": true}
import numpy as np
N = 100
row_idx = np.repeat(np.arange(N)[None,:], 7,  axis=1)
np.reshape(row_idx, -1)

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
h5_dir = cbin_dir + '/test_n_14_parallelized_subtraction.h5'
with h5py.File(h5_dir) as h5:
    x_ps = h5['localizations'][:, 0]
    z_ps = h5['localizations'][:, 2]
    ptps_ps = h5['maxptps'][:]
    
h5_dir = cbin_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    x_sup = h5['localizations'][:, 0]
    z_sup = h5['localizations'][:, 2]
    ptps_sup = h5['maxptps'][:]
    
    
h5_dir = cbin_dir + '/no_decreade_enforce_subtraction.h5'
with h5py.File(h5_dir) as h5:
    x = h5['localizations'][:, 0]
    z = h5['localizations'][:, 2]
    ptps = h5['maxptps'][:]
    
    
fig, axs = plt.subplots(1,6, figsize = [12, 20])
maxptps = ptps_ps
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[0].scatter(x_ps, z_ps, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[0].set_xlim([-50,90])
axs[0].set_ylim([0,3900])
axs[0].set_title('phase-shift')

maxptps = ptps_sup
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[1].scatter(x_sup, z_sup, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[1].set_xlim([-50,90])
axs[1].set_ylim([0,3900])
axs[1].set_title('enforce decrease')

maxptps = ptps
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[2].scatter(x, z, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[2].set_xlim([-50,90])
axs[2].set_ylim([0,3900])
axs[2].set_title('no decrease')


h5_dir = cbin_dir + '/test_n_14_parallelized_subtraction.h5'
with h5py.File(h5_dir) as h5:
    x_ps = h5['localizationspeak'][:, 0]
    z_ps = h5['localizationspeak'][:, 2]
    ptps_ps = h5['maxptps'][:]
    
h5_dir = cbin_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    x_sup = h5['localizationspeak'][:, 0]
    z_sup = h5['localizationspeak'][:, 2]
    ptps_sup = h5['maxptps'][:]
    
    
h5_dir = cbin_dir + '/no_decreade_enforce_subtraction.h5'
with h5py.File(h5_dir) as h5:
    x = h5['localizationspeak'][:, 0]
    z = h5['localizationspeak'][:, 2]
    ptps = h5['maxptps'][:]
    
    
# fig, axs = plt.subplots(1,3, figsize = [6, 20])
maxptps = ptps_ps
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[3].scatter(x_ps, z_ps, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[3].set_xlim([-50,90])
axs[3].set_ylim([0,3900])
axs[3].set_title('phase-shift peak')

maxptps = ptps_sup
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[4].scatter(x_sup, z_sup, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[4].set_xlim([-50,90])
axs[4].set_ylim([0,3900])
axs[4].set_title('enforce decrease peak')

maxptps = ptps
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[5].scatter(x, z, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[5].set_xlim([-50,90])
axs[5].set_ylim([0,3900])
axs[5].set_title('no decrease peak')

# plt.savefig(cbin_dir + '/localization_compare.png', dpi = 300)

# %%
import h5py
h5_dir = cbin_dir + '/phase-shift_subtraction.h5'
with h5py.File(h5_dir) as h5:
    x_ps = h5['localizationspeak'][:, 0]
    z_ps = h5['localizationspeak'][:, 2]
    ptps_ps = h5['maxptps'][:]
    
h5_dir = cbin_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    x_sup = h5['localizationspeak'][:, 0]
    z_sup = h5['localizationspeak'][:, 2]
    ptps_sup = h5['maxptps'][:]
    
    
h5_dir = cbin_dir + '/no_decreade_enforce_subtraction.h5'
with h5py.File(h5_dir) as h5:
    x = h5['localizationspeak'][:, 0]
    z = h5['localizationspeak'][:, 2]
    ptps = h5['maxptps'][:]
    
    
fig, axs = plt.subplots(1,3, figsize = [6, 20])
maxptps = ptps_ps
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[0].scatter(x_ps, z_ps, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[0].set_xlim([-50,90])
axs[0].set_ylim([0,3900])
axs[0].set_title('phase-shift')

maxptps = ptps_sup
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[1].scatter(x_sup, z_sup, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[1].set_xlim([-50,90])
axs[1].set_ylim([0,3900])
axs[1].set_title('enforce decrease')

maxptps = ptps
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
axs[2].scatter(x, z, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
axs[2].set_xlim([-50,90])
axs[2].set_ylim([0,3900])
axs[2].set_title('no decrease')

plt.savefig(cbin_dir + '/localization_compare_peak.png', dpi = 300)

# %%
h5_dir = cbin_dir + '/phase-shift_subtraction.h5'
with h5py.File(h5_dir) as h5:
    waveforms = h5['denoised_waveforms'][:]
h5 = h5py.File(h5_dir)
print(h5.keys())
h5.close()

# %%
plt.imshow(waveforms[0].T, aspect = 'auto')

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
plt.figure(figsize = [2, 20])
maxptps = ptps
cmps = np.clip(maxptps, 8, 14)
nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
plt.scatter(x, z, c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'])
plt.xlim([-50,80])




# %%
