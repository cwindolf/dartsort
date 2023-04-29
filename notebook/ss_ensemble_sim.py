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
import warnings; warnings.simplefilter("ignore", category=DeprecationWarning)

# %%
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample
import colorcet as cc
import spikeinterface.full as si
import h5py
from tqdm.auto import tqdm, trange
from scipy import linalg as la
import pickle
import shutil
from spike_psvae.hybrid_analysis import Sorting

# %%
from spike_psvae import (
    subtract,
    cluster_utils,
    cluster_viz_index,
    ibme,
    ibme_corr,
    newms,
    waveform_utils,
    chunk_features,
    drifty_deconv,
    deconvolve,
    spike_train_utils,
    snr_templates,
    extract_deconv,
    localize_index,
    outliers,
    before_deconv_merge_split,
)

# %%
# %matplotlib inline
plt.rc("figure", dpi=200)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# %%
import torch

# %%
torch.cuda.is_available()

# %% [markdown]
# ## Paths / config

# %%
#parameters
bath_path = '/moto/stats/users/ch3676/'
recording_name='recording_5_static/'
sorting_name = 'sorting_static_5min/'
data_path = bath_path + recording_name
sort_path = Path(bath_path + sorting_name)
# !rsync -aP {data_path} /local/bincache/recording_5_static
rec_gt = si.load_extractor(str(data_path))
sort_gt = si.load_extractor(sort_path)
data_name = 'traces_cached_seg0.raw'
raw_data_bin = data_path + data_name
# !rsync -aP {raw_data_bin} /local/bincache/
raw_data_bin = Path(raw_data_bin)
data_path = Path(data_path)
raw_data_bin = Path("/local/bincache/") / raw_data_bin.name
data_path = Path("/local/bincache/") / data_path.name

output_folder = Path("/local/bincache/") / "simulated_results/outputs_ensemble_static_5min"
template_locations = np.load(sort_path / 'template_locations.npy')

raw_data_bin = Path(raw_data_bin)
output_folder = Path(output_folder)

geom = rec_gt.get_probe().contact_positions
print(rec_gt, sort_gt)
pitch = waveform_utils.get_pitch(geom)

dsroot = Path(data_path)
dsout = output_folder
dsout.mkdir(exist_ok=True, parents=True)

(dsout / "probe_and_raw_data").mkdir(exist_ok=True)
(dsout / "scatterplots").mkdir(exist_ok=True)
(dsout / "drift").mkdir(exist_ok=True)

# %%
t_start = 0
t_end = None

args = {}
args["trough_offset"] = trough_offset = 42
args["spike_length_samples"] = spike_length_samples = 121
args["deconv_thresh"] = deconv_thresh = 200 #200 # change this parameter!!
args["merge_thresh_early"] = merge_thresh_early = 3.0 # 2.0
args["merge_thresh_end"] = merge_thresh_end = 3.0
args["tpca_weighted"] = tpca_weighted = False
args["do_reg"] = do_reg = False
args["do_reloc"] = do_reloc = False 
args["subtraction_thresholds"] = subtraction_thresholds = [12, 10, 8, 6, 5, 4]
args["do_neural_net_denoise"] = do_neural_net_denoise = True
args["refractory_period_frames"] = refractory_period_frames = 10
args["fs"] = fs = 32000

with open(dsout / 'sort_params.txt', 'w') as f:
    for key in args.keys():
        f.write('\n' + key + ': ' + str(args[key]))

# %%
subtract.make_channel_index(geom, 100)[0]

# %%
loc_channel_index = subtract.make_contiguous_channel_index(geom.shape[0], n_neighbors=20)

# %%
#visualize raw data
zzz = np.memmap(raw_data_bin, dtype=np.float32).reshape(-1, 384)
plt.imshow(zzz[:1000].T, aspect="auto");
plt.show()

# %% [markdown]
# # Detection / featurization

# %%
from spike_psvae import waveform_utils

sub_dir = output_folder / "sub"
sub_dir.mkdir(exist_ok=True)

# %%
import torch

# %%
sub_h5 = subtract.subtraction_binary(
    raw_data_bin,
    # dsout / "sippx" / "traces_cached_seg0.raw",
    sub_dir,
    geom=geom, 
    # save_waveforms=True,
    save_residual=False,
    # n_sec_pca=80,
    peak_sign="both",
    enforce_decrease_kind="radial",
    neighborhood_kind="circle",
    # sampling_rate=30000, #fs
    do_nn_denoise=do_neural_net_denoise,
    thresholds=subtraction_thresholds,
    n_jobs=8,
    overwrite=True,
    localize_radius=100,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=True,
    save_denoised_tpca_projs=False,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    sampling_rate=fs,
)

# %%
overwrite = True

with h5py.File(next((dsout / "sub").glob("sub*h5")), "r+" if overwrite else "r") as h5:
# with h5py.File("/media/peter/2TB/sherry/IND106_shank1_Mar26/sub/subtraction.h5", "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] # - h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    # dispmap -= dispmap.mean()
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if overwrite and "z_reg" in h5:
            del h5["z_reg"]
            del h5["p"]
        if do_reg:
            # z_reg, dispmap = ibme.register_nonrigid(
            z_reg, p = ibme.register_rigid(
                maxptps,
                z_abs - z_abs.min(),
                (samples - samples.min()) / fs,
                # robust_sigma=1,
                # corr_threshold=0.3,
                adaptive_mincorr_percentile=5,
                disp=300,
                denoise_sigma=0.1,
                # max_dt=100,
                prior_lambda=1,
                batch_size=64,
            )
            z_reg -= (z_reg - z_abs).mean()
        else:
            z_reg = z_abs
            p = np.zeros(np.ceil(t.max()).astype(int))
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)
    t = spike_index[:, 0] / fs

# %%
fig, ax = plt.subplots(figsize=(8, 6))
for zz in np.unique(geom[:, 1]):
    ax.axhline(zz, lw=1, color="k", alpha=0.2)
ax.scatter(t, geom[maxchans, 1], c=np.clip(maxptp, 0, 15), s=50, alpha=1, linewidths=0, marker=".")
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ax,
)
tt = np.arange(0, 100 * (t.max() // 100) + 1, 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("detection channel depth (um)")
plt.title(f"time vs. maxchan.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatterplots" / "initial_detection_t_v_channel.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(t, z, c=np.clip(maxptp, 0, 15), s=5, alpha=0.5, linewidths=0, marker=".")
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ax,
)
ax.plot(geom.max() / 2 + p, color="k", lw=2, label="drift est.")
for zz in np.unique(geom[:, 1]):
    ax.axhline(zz, lw=1, color="k", alpha=0.2)
ax.legend()
tt = np.arange(0, 100 * (t.max() // 100) + 1, 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"time vs. y.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatterplots" / "initial_detection_t_v_y.png")
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t, z_reg, c=np.clip(maxptp, 0, 15), s=5, alpha=0.5, marker=".", linewidths=0)
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=plt.gca(),
)
tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"time vs. registered y.  {len(maxptp)} spikes.")
fig.savefig(dsout / "scatterplots" / "initial_detection_t_v_regy.png")
plt.close(fig)

fig, (aa, ab) = plt.subplots(ncols=2, figsize=(6, 6), gridspec_kw=dict(wspace=0.05))
ordd = np.argsort(maxptp)
aa.scatter(x[ordd], z_abs[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
ab.scatter(np.log(5+maxptp[ordd]), z_abs[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ab,
)
aa.set_xlabel("x (um)")
ab.set_xlabel("log(5 + amplitude)")
ab.set_yticks([])
aa.set_ylabel("depth (um)")
fig.suptitle(f"x vs. y.  {len(maxptp)} spikes.", y=0.92)
fig.savefig(dsout / "scatterplots" / "initial_detection_x_v_y.png")
plt.close(fig)

fig, (aa, ab) = plt.subplots(ncols=2, figsize=(6, 6), gridspec_kw=dict(wspace=0.05))
ordd = np.argsort(maxptp)
aa.scatter(x[ordd], z_reg[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
ab.scatter(np.log(5+maxptp[ordd]), z_reg[ordd], c=np.clip(maxptp[ordd], 0, 15), s=1, alpha=0.25, linewidths=0)
plt.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 15), plt.cm.viridis),
    label="denoised peak-to-peak amp.",
    shrink=0.5,
    pad=0.025,
    ax=ab,
)
aa.set_xlabel("x (um)")
ab.set_xlabel("log(5 + amplitude)")
ab.set_yticks([])
aa.set_ylabel("depth (um)")
fig.suptitle(f"x vs. registered y.  {len(maxptp)} spikes.", y=0.92)
fig.savefig(dsout / "scatterplots" / "initial_detection_x_v_regy.png")
plt.close(fig)

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))

counts, edges, _ = aa.hist(p, bins=128, color="k")
aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")

ab.plot(t_start + np.arange(len(p)), p, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(p)), p, c="k", s=1, zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
fig.suptitle(f"start time {t_start}", y=0.95, fontsize=10)

fig.savefig(dsout / "drift" / f"initial_detection_disp_tstart{t_start}.png", dpi=300)
# np.savetxt(dsout / "drift" / f"initial_detection_disp_tstart{t_start}.csv", p, delimiter=",") #temporary commented out
plt.close(fig) 

fig, (aa, ab) = plt.subplots(ncols=2, figsize=(8, 3))
counts, edges, _ = aa.hist(p, bins=128, color="k")
with h5py.File(sub_h5, "r") as h5:
    p = h5["p"][:]
p_mode = edges[counts.argmax():counts.argmax()+2].mean()
lo = p_mode - pitch / 2
hi = p_mode + pitch / 2
p_good = (lo < p) & (hi > p)

# the command-out code is to run subsampling of 5 mins across the entire session
# p_good = np.zeros(len(p), dtype=bool)
# random_times = np.random.default_rng(0).choice(len(p), size=300, replace=False)
# random_times.sort()
# p_good[random_times] = 1

aa.set_xlabel("est. displacement")
aa.set_ylabel("frequency")
aa.axvline(edges[counts.argmax():counts.argmax()+2].mean(), color=plt.cm.Greens(0.4), label="mode est.", lw=2, ls="--")
aa.legend()
ab.plot(t_start + np.arange(len(p)), p, c="gray", lw=1)
ab.scatter(t_start + np.arange(len(p))[~p_good], p[~p_good], c="k", s=1, label=f"disp too far {(~p_good).sum()}s", zorder=12)
ab.scatter(t_start + np.arange(len(p))[p_good], p[p_good], color=plt.cm.Greens(0.4), s=1, label=f"disp within pitch/2 of mode {(p_good).sum()}s", zorder=12)
ab.set_ylabel("est. displacement")
ab.set_xlabel("time (s)")
ab.legend()

fig.savefig(dsout / "drift" / f"initial_detection_stable_bins.png", dpi=300)


# %% [markdown]
# # Initial clustering

# %%
sub_h5 = next((dsout / "sub").glob("sub*h5"))
(dsout / "clust").mkdir(exist_ok=True)

with h5py.File(sub_h5) as h5:
    spike_index = h5["spike_index"][:]

# %%
st = spike_index.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
print(f"{good_times.sum()=}")
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split_ensemble(
    st,
    geom.shape[0],
    raw_data_bin,
    sub_h5,
    geom,
    dsout / "clust",
    n_workers=20,
    merge_resid_threshold=2.0,
    relocated=do_reloc,
    trough_offset=trough_offset,
    ensemble_percent=.8,
    num_ensemble=3,
    spike_length_samples=spike_length_samples,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
        split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=15,
                    # min_samples=5,
                    cluster_selection_epsilon=20.0,
                ),
            ),
        ),
    )
)

# st = spike_index.copy()
# # st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
# good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
# print(f"{good_times.sum()=}")
# st[~good_times, 1] = -1
# spike_train, templates, order = newms.new_merge_split(
#     st,
#     geom.shape[0],
#     raw_data_bin,
#     sub_h5,
#     geom,
#     dsout / "clust",
#     n_workers=20,
#     merge_resid_threshold=2.0,
#     relocated=do_reloc,
#     trough_offset=trough_offset,
#     # ensemble_percent=.8,
#     # num_ensemble=3,
#     spike_length_samples=spike_length_samples,
#     split_kwargs=dict(
#         split_steps=(
#             before_deconv_merge_split.herding_split,
#         ),
#         recursive_steps=(True,),
#         split_step_kwargs=(
#             dict(
#                 hdbscan_kwargs=dict(
#                     min_cluster_size=15,
#                     # min_samples=5,
#                     cluster_selection_epsilon=20.0,
#                 ),
#             ),
#         ),
#     )
# )

# %%
# for k in ("split", "merge"):
#     visst = np.load(dsout / "clust" / f"{k}_st.npy")
#     vissort = Sorting(
#         dsroot / "traces_cached_seg0.raw",
#         geom,
#         visst[:, 0],
#         spike_train_utils.make_labels_contiguous(visst[:, 1]),
#         name="StableClust" + k.capitalize(),
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
#     od = dsout / "summaries"
#     vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
    
# #     the command-out code is to output cleaned waveforms. Turned off because to reduce time
#     # with h5py.File(sub_h5) as h5:
#     #     vissort.make_unit_summaries(
#     #         out_folder=od / f"{vissort.name_lo}_cleaned",
#     #         stored_maxchans=h5["spike_index"][:, 1],
#     #         stored_order=order,
#     #         stored_channel_index=h5["channel_index"][:],
#     #         stored_tpca_projs=h5["cleaned_tpca_projs"],
#     #         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#     #         show_scatter=False,
#     #         relocated=False,
#     #         n_jobs=1,
#     #     )

# %%
# fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
# cluster_viz_index.array_scatter(
#     spike_train[:, 1],
#     geom,
#     x[order],
#     z_reg[order],
#     maxptp[order],
#     axes=axes,
#     zlim=(geom.min() - 25, geom.max() + 25),
#     c=5,
# )
# cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
# cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
# cbar.solids.set(alpha=1)
# cax.set_yticks([0, 1], labels=[3, 15])
# axes[0].set_ylabel("depth (um)")
# axes[1].set_yticks([])
# axes[1].set_xlabel("log peak-to-peak amplitude")
# axes[2].set_yticks([])
# nunits = np.setdiff1d(np.unique(spike_train[:, 1]), [-1]).size
# axes[1].set_title(f"Spatial view of clustered and triaged spikes. {nunits} units.")
# fig.savefig(dsout / "scatterplots" / "initclust_scatter.png")
# plt.close(fig)

# fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
# cluster_viz_index.array_scatter(
#     spike_train[:, 1],
#     geom,
#     x[order],
#     z_reg[order],
#     maxptp[order],
#     axes=axes,
#     zlim=(500, 1000),
#     c=5,
# )
# cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
# cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
# cbar.solids.set(alpha=1)
# cax.set_yticks([0, 1], labels=[3, 15])
# axes[0].set_ylabel("depth (um)")
# axes[1].set_yticks([])
# axes[1].set_xlabel("log peak-to-peak amplitude")
# axes[2].set_yticks([])
# axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
# fig.savefig(dsout / "scatterplots" / "initclust_scatter_detail.png")

# kept = spike_train[:, 1] >= 0
# triaged = spike_train[:, 1] < 0

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(t[order][triaged & ~good_times], z_reg[order][triaged & ~good_times], color="k", s=5, alpha=0.5, marker=".", linewidths=0, label="outside stable")
# plt.scatter(t[order][triaged & good_times], z_reg[order][triaged & good_times], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# plt.legend(markerscale=2.5)

# tt = np.arange(0, 100 * (t.max() // 100) , 100)
# plt.xticks(tt, t_start + tt)
# plt.xlabel("time (s)")
# plt.ylabel("depth (um)")
# plt.title(f"time vs. registered y.  {kept.sum()} sorted spikes.")
# fig.savefig(dsout / "scatterplots" / "initclust_scatter_sorted_t_v_regy.png", dpi=300)

# kept = spike_train[:, 1] >= 0
# triaged = spike_train[:, 1] < 0

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(t[order][triaged & ~good_times], z[order][triaged & ~good_times], color="k", s=5, alpha=0.5, marker=".", linewidths=0, label="outside stable")
# plt.scatter(t[order][triaged & good_times], z[order][triaged & good_times], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# plt.legend(markerscale=2.5)

# tt = np.arange(0, 100 * (t.max() // 100) , 100)
# plt.xticks(tt, t_start + tt)
# plt.xlabel("time (s)")
# plt.ylabel("depth (um)")
# plt.title(f"time vs. y.  {kept.sum()} sorted spikes.")
# fig.savefig(dsout / "scatterplots" / "initclust_scatter_sorted_t_v_y.png")

# %% [markdown]
# ## Deconv 1

# %%
spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    np.load(dsout / "clust" / "merge_st.npy"),
    geom.shape[0],
    raw_data_bin,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    max_shift=0,
    min_n_spikes=5,
)

# %%
assert (order == np.arange(len(order))).all()

if (dsout / "deconv1").exists():
    shutil.rmtree(dsout / "deconv1")

# %%
merge_order = np.load(dsout / "clust" / "merge_order.npy")
sub_h5 = next((dsout / "sub").glob("*.h5"))
deconv_dict = deconvolve.deconv(
    raw_data_bin,
    dsout / "deconv1",
    templates,
    t_start=0,
    t_end=None,
    sampling_rate=fs,
    n_sec_chunk=1,
    n_jobs=10,
    max_upsample=8,
    refractory_period_frames=refractory_period_frames,
    trough_offset=trough_offset,
    threshold=deconv_thresh,
    lambd=0,
    allowed_scale=0.1
)
np.save(dsout / "deconv1" / "deconv_spike_train.npy",  deconv_dict['deconv_spike_train'])

# %%
extract_deconv1_h5 = extract_deconv.extract_deconv(
    deconv_dict['templates_up'],
    deconv_dict['deconv_spike_train_upsampled'],
    dsout / "deconv1",
    raw_data_bin,
    subtraction_h5=sub_h5,
    save_cleaned_waveforms=False,
    save_denoised_waveforms=False,
    save_cleaned_tpca_projs=True,
    n_channels_extract=20,
    n_jobs=10,
    # device="cpu",
    do_reassignment=False,
    # scratch_dir=deconv_scratch_dir,
)

# %%
overwrite = True
# overwrite = False
rereg = False

with h5py.File(extract_deconv1_h5, "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if rereg:
            if overwrite and "z_reg" in h5:
                del h5["z_reg"]
                del h5["p"]

            # z_reg, dispmap = ibme.register_nonrigid(
            z_reg, p = ibme.register_rigid(
                maxptps,
                z_abs - z_abs.min(),
                (samples - samples.min()) / 20000,
                # robust_sigma=1,
                # corr_threshold=0.3,
                adaptive_mincorr_percentile=5,
                disp=300,
                denoise_sigma=0.1,
                # max_dt=100,
                prior_lambda=1,
                batch_size=64,
            )
        else:
            z_reg = z_abs
            p = np.zeros(np.ceil(t.max()).astype(int))
        z_reg -= np.median(z_reg - z_abs)
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
order = slice(None)
spike_train = deconv_dict["deconv_spike_train"]

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(geom.min() - 25, geom.max() + 25),
    c=5,
)
cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
cbar.solids.set(alpha=1)
cax.set_yticks([0, 1], labels=[3, 15])
axes[0].set_ylabel("depth (um)")
axes[1].set_yticks([])
axes[1].set_xlabel("log peak-to-peak amplitude")
axes[2].set_yticks([])
nunits = np.setdiff1d(np.unique(spike_train[:, 1]), [-1]).size
axes[1].set_title(f"Spatial view of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv1_scatter.png")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(500, 1000),
    c=5,
)
cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
cbar.solids.set(alpha=1)
cax.set_yticks([0, 1], labels=[3, 15])
axes[0].set_ylabel("depth (um)")
axes[1].set_yticks([])
axes[1].set_xlabel("log peak-to-peak amplitude")
axes[2].set_yticks([])
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv1_scatter_detail.png")

kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1_scatter_sorted_t_v_regy.png", dpi=300)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1_scatter_sorted_t_v_y.png")

# %%
# deconv3_spike_train = np.load(Path('/local/bincache/simulated_results/outputs_ensemble_static_5min') / f'deconv3_mt_{merge_threshold_end_new}' / 'deconv_spike_train.npy')
spike_times = spike_train[:,0]
spike_labels = spike_train[:,1]

sorting_deconv_1 = si.numpyextractors.NumpySorting.from_times_labels(
        times_list=spike_times.astype("int"),
        labels_list=spike_labels.astype("int"),
        sampling_frequency=fs,
    )

# %%
cmp_gt = si.compare_sorter_to_ground_truth(sort_gt, sorting_deconv_1, exhaustive_gt=True, match_score=.1)
# cmp_gt_deconv1 = si.compare_sorter_to_ground_truth(sort_gt, sorting_deconv1, exhaustive_gt=True, match_score=.1)
# cmp_gt_deconv2 = si.compare_sorter_to_ground_truth(sort_gt, sorting_deconv2, exhaustive_gt=True, match_score=.1)
# cmp_gt_merge_deconv2 = si.compare_sorter_to_ground_truth(sort_gt, sorting_merge_deconv2, exhaustive_gt=True, match_score=.1)
# sort_ks = si.read_kilosort('/media/cat/cole/kilosort_results_sim_5min/KS_output/data/')
# cmp_gt_ks = si.compare_sorter_to_ground_truth(sort_gt, sort_ks, exhaustive_gt=True, match_score=.1)

# %%
sort_gt.get_num_units()

# %%
folder = 'waveform_folder'
we = si.extract_waveforms(
    rec_gt,
    sort_gt,
    folder,
    ms_before=1.5,
    ms_after=2.,
    max_spikes_per_unit=500,
    overwrite=True,
    seed=0,
    # load_if_exists=True,
)
print(we)

# %%
snrs = si.compute_snrs(waveform_extractor=we)
ptps = we.get_all_templates().ptp(1).max(1)

# %%
fig, ax = plt.subplots(1,1, figsize=(8,20))
ax.imshow(cmp_gt.get_ordered_agreement_scores().to_numpy()[:150,:150])
ax.set_title('deconv 1')
# axes[1].imshow(cmp_gt_deconv2.get_ordered_agreement_scores().to_numpy())
# axes[1].set_title('deconv 2')
# axes[2].imshow(cmp_gt_merge_deconv2.get_ordered_agreement_scores().to_numpy())
# axes[2].set_title('deconv 2 merge')
# axes[3].imshow(cmp_gt.get_ordered_agreement_scores().to_numpy())
# axes[3].set_title('final deconv');
# axes[4].imshow(cmp_gt_ks.get_ordered_agreement_scores().to_numpy())
# axes[4].set_title('kilosort (default)');
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# %%
for name, cmp in [('deconv1', cmp_gt)]:
    well_detected_units = cmp.get_well_detected_units(well_detected_score = .8)
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].scatter(snrs.values(), cmp.get_performance()['precision'])
    axes[0].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[0].set_ylabel('precision')
    axes[0].set_xlabel('snr')

    axes[1].scatter(snrs.values(), cmp.get_performance()['recall'])
    axes[1].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[1].set_ylabel('recall')
    axes[1].set_xlabel('snr')

    axes[2].scatter(snrs.values(), cmp.get_performance()['accuracy'])
    axes[2].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[2].set_ylabel('accuracy')
    axes[2].set_xlabel('snr')

# %%

# %%

# %%
fig, ax = plt.subplots(1,1, figsize=(8,20))
ax.imshow(cmp_gt.get_ordered_agreement_scores().to_numpy()[:150,:150])
ax.set_title('deconv 1')
# axes[1].imshow(cmp_gt_deconv2.get_ordered_agreement_scores().to_numpy())
# axes[1].set_title('deconv 2')
# axes[2].imshow(cmp_gt_merge_deconv2.get_ordered_agreement_scores().to_numpy())
# axes[2].set_title('deconv 2 merge')
# axes[3].imshow(cmp_gt.get_ordered_agreement_scores().to_numpy())
# axes[3].set_title('final deconv');
# axes[4].imshow(cmp_gt_ks.get_ordered_agreement_scores().to_numpy())
# axes[4].set_title('kilosort (default)');
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# %%

for name, cmp in [('deconv1', cmp_gt)]:
    well_detected_units = cmp.get_well_detected_units(well_detected_score = .8)
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].scatter(snrs.values(), cmp.get_performance()['precision'])
    axes[0].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[0].set_ylabel('precision')
    axes[0].set_xlabel('snr')

    axes[1].scatter(snrs.values(), cmp.get_performance()['recall'])
    axes[1].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[1].set_ylabel('recall')
    axes[1].set_xlabel('snr')

    axes[2].scatter(snrs.values(), cmp.get_performance()['accuracy'])
    axes[2].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[2].set_ylabel('accuracy')
    axes[2].set_xlabel('snr')

# %%

# %%
fig, ax = plt.subplots(1,1, figsize=(8,20))
ax.imshow(cmp_gt.get_ordered_agreement_scores().to_numpy())
ax.set_title('deconv 1')
# axes[1].imshow(cmp_gt_deconv2.get_ordered_agreement_scores().to_numpy())
# axes[1].set_title('deconv 2')
# axes[2].imshow(cmp_gt_merge_deconv2.get_ordered_agreement_scores().to_numpy())
# axes[2].set_title('deconv 2 merge')
# axes[3].imshow(cmp_gt.get_ordered_agreement_scores().to_numpy())
# axes[3].set_title('final deconv');
# axes[4].imshow(cmp_gt_ks.get_ordered_agreement_scores().to_numpy())
# axes[4].set_title('kilosort (default)');
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# %%

for name, cmp in [('deconv1', cmp_gt)]:
    well_detected_units = cmp.get_well_detected_units(well_detected_score = .8)
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].scatter(snrs.values(), cmp.get_performance()['precision'])
    axes[0].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[0].set_ylabel('precision')
    axes[0].set_xlabel('snr')

    axes[1].scatter(snrs.values(), cmp.get_performance()['recall'])
    axes[1].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[1].set_ylabel('recall')
    axes[1].set_xlabel('snr')

    axes[2].scatter(snrs.values(), cmp.get_performance()['accuracy'])
    axes[2].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[2].set_ylabel('accuracy')
    axes[2].set_xlabel('snr')

# %%

# %%

# %%

# %%
st = spike_index.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
print(f"{good_times.sum()=}")
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split_ensemble(
    st,
    geom.shape[0],
    raw_data_bin,
    sub_h5,
    geom,
    dsout / "clust",
    n_workers=10,
    merge_resid_threshold=merge_thresh_early,
    relocated=do_reloc,
    trough_offset=trough_offset,
    ensemble_percent=.8,
    num_ensemble=2,
    spike_length_samples=spike_length_samples,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
        split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=15,
                    # min_samples=5,
                    cluster_selection_epsilon=20.0,
                ),
            ),
        ),
    )
)

# %%
st = spike_train.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
print(f"{good_times.sum()=}")
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split_ensemble(
    st,
    geom.shape[0],
    raw_data_bin,
    extract_deconv1_h5,
    geom,
    dsout / "deconv1clust",
    n_workers=10,
    merge_resid_threshold=merge_thresh_early,
    relocated=do_reloc,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
        split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=15,
                    # min_samples=5,
                    cluster_selection_epsilon=20.0,
                ),
            ),
        ),
    )
)

# %%
(spike_train[:, 1] >= 0).sum(), (spike_train[good_times, 1]>=0).mean()

# %%
# for k in ("split", "merge"):
#     visst = np.load(dsout / "deconv1clust" / f"{k}_st.npy")
#     vissort = Sorting(
#         dsroot / "traces_cached_seg0.raw",
#         geom,
#         visst[:, 0],
#         spike_train_utils.make_labels_contiguous(visst[:, 1]),
#         name="Deconv1Clust" + k.capitalize(),
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
#     od = dsout / "summaries"
#     vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
#     # with h5py.File(extract_deconv1_h5) as h5:
#     #     vissort.make_unit_summaries(
#     #         out_folder=od / f"{vissort.name_lo}_cleaned",
#     #         stored_maxchans=h5["spike_index"][:, 1],
#     #         stored_order=order,
#     #         stored_channel_index=h5["channel_index"][:],
#     #         stored_tpca_projs=h5["cleaned_tpca_projs"],
#     #         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#     #         show_scatter=False,
#     #         relocated=False,
#     #         n_jobs=1,
#     #     )

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(geom.min() - 25, geom.max() + 25),
    c=5,
)
cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
cbar.solids.set(alpha=1)
cax.set_yticks([0, 1], labels=[3, 15])
axes[0].set_ylabel("depth (um)")
axes[1].set_yticks([])
axes[1].set_xlabel("log peak-to-peak amplitude")
axes[2].set_yticks([])
nunits = np.setdiff1d(np.unique(spike_train[:, 1]), [-1]).size
axes[1].set_title(f"Spatial view of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter.png")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(500, 1000),
    c=5,
)
cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
cbar.solids.set(alpha=1)
cax.set_yticks([0, 1], labels=[3, 15])
axes[0].set_ylabel("depth (um)")
axes[1].set_yticks([])
axes[1].set_xlabel("log peak-to-peak amplitude")
axes[2].set_yticks([])
axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter_detail.png")

kept = spike_train[:, 1] >= 0
triaged = spike_train[:, 1] < 0

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"time vs. registered y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter_sorted_t_v_regy.png", dpi=300)

fig = plt.figure(figsize=(8, 6))
plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
plt.legend(markerscale=2.5)

tt = np.arange(0, 100 * (t.max() // 100) , 100)
plt.xticks(tt, t_start + tt)
plt.xlabel("time (s)")
plt.ylabel("depth (um)")
plt.title(f"time vs. y.  {kept.sum()} sorted spikes.")
fig.savefig(dsout / "scatterplots" / "deconv1clust_scatter_sorted_t_v_y.png")

# %% [markdown]
# ## Deconv 2

# %%
spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    np.load(dsout / "deconv1clust" / "merge_st.npy"),
    geom.shape[0],
    raw_data_bin,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    max_shift=0,
    min_n_spikes=5,
)

# %%
assert (order == np.arange(len(order))).all()

if (dsout / "deconv2").exists():
    shutil.rmtree(dsout / "deconv2")

# %%
# merge_order = np.load(dsout / "deconv1clust" / "merge_order.npy")
# with h5py.File(extract_deconv1_h5) as h5:
#     z = h5["localizations"][:, 2][merge_order]
#     p = h5["p"][:]
# superres2 = drifty_deconv.superres_deconv(
#     dsroot / "traces_cached_seg0.raw",
#     geom,
#     z,
#     p,
#     spike_train=spike_train,
#     reference_displacement=p_mode,
#     bin_size_um=pitch / 2,
#     pfs=fs,
#     n_jobs=8,
#     deconv_dir=dsout / "deconv2",
#     trough_offset=trough_offset,
#     spike_length_samples=spike_length_samples,
#     threshold=deconv_thresh,
# )
deconv_dict_2 = deconvolve.deconv(
    raw_data_bin,
    dsout / "deconv2",
    templates,
    t_start=0,
    t_end=None,
    sampling_rate=fs,
    n_sec_chunk=1,
    n_jobs=8,
    max_upsample=8,
    refractory_period_frames=refractory_period_frames,
    trough_offset=trough_offset,
    threshold=deconv_thresh,
    lambd=0,
    allowed_scale=0.1
)
np.save(dsout / "deconv2" / "deconv_spike_train.npy",  deconv_dict_2['deconv_spike_train'])

# %%
deconv_dict_2['deconv_spike_train'].shape

# %%
# extract_deconv2_h5, extract_deconv2_extra = drifty_deconv.extract_superres_shifted_deconv(
#     superres2,
#     save_cleaned_waveforms=True,
#     save_cleaned_tpca_projs=True,
#     save_residual=False,
#     sampling_rate=fs,
#     subtraction_h5=extract_deconv1_h5,
#     nn_denoise=False,
#     geom=geom,
#     n_jobs=8,
#     # save_reassignment_residuals=True,
#     # pairs_method="radius",
#     max_resid_dist=20,
#     do_reassignment_tpca=True,
#     do_reassignment=False,
#     n_sec_train_feats=80,
#     tpca_weighted=tpca_weighted,
# )

extract_deconv2_h5 = extract_deconv.extract_deconv(
    deconv_dict_2['templates_up'],
    deconv_dict_2['deconv_spike_train_upsampled'],
    dsout / "deconv2",
    raw_data_bin,
    subtraction_h5=sub_h5,
    save_cleaned_waveforms=False,
    save_denoised_waveforms=False,
    save_cleaned_tpca_projs=True,
    n_channels_extract=20,
    n_jobs=8,
    device="cpu",
    do_reassignment=False,
    # scratch_dir=deconv_scratch_dir,
)

# %%
overwrite = True
# overwrite = False
rereg = False

with h5py.File(extract_deconv2_h5, "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if rereg:
            if overwrite and "z_reg" in h5:
                del h5["z_reg"]
                del h5["p"]

            # z_reg, dispmap = ibme.register_nonrigid(
            z_reg, p = ibme.register_rigid(
                maxptps,
                z_abs - z_abs.min(),
                (samples - samples.min()) / 20000,
                # robust_sigma=1,
                # corr_threshold=0.3,
                adaptive_mincorr_percentile=5,
                disp=300,
                denoise_sigma=0.1,
                # max_dt=100,
                prior_lambda=1,
                batch_size=64,
            )
        else:
            z_reg = z_abs
            p = np.zeros(np.ceil(t.max()).astype(int))
            # *_, tt = ibme.fast_raster(
            #     maxptp, z_abs - z_abs.min(), t
            # )
            # z_reg = ibme.warp_rigid(z_abs, t, tt, p)
        z_reg -= np.median(z_reg - z_abs)
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
order = slice(None)
spike_train = deconv_dict_2["deconv_spike_train"]

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(geom.min() - 25, geom.max() + 25),
    c=5,
)
cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
cbar.solids.set(alpha=1)
cax.set_yticks([0, 1], labels=[3, 15])
axes[0].set_ylabel("depth (um)")
axes[1].set_yticks([])
axes[1].set_xlabel("log peak-to-peak amplitude")
axes[2].set_yticks([])
nunits = np.setdiff1d(np.unique(spike_train[:, 1]), [-1]).size
axes[1].set_title(f"Spatial view of clustered and triaged spikes. {nunits} units.")
fig.savefig(dsout / "scatterplots" / "deconv2_scatter.png")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
cluster_viz_index.array_scatter(
    spike_train[:, 1],
    geom,
    x[order],
    z_reg[order],
    maxptp[order],
    axes=axes,
    zlim=(500, 1000),
    c=5,
)
# cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
# cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
# cbar.solids.set(alpha=1)
# cax.set_yticks([0, 1], labels=[3, 15])
# axes[0].set_ylabel("depth (um)")
# axes[1].set_yticks([])
# axes[1].set_xlabel("log peak-to-peak amplitude")
# axes[2].set_yticks([])
# axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
# fig.savefig(dsout / "scatterplots" / "deconv2_scatter_detail.png")

# kept = spike_train[:, 1] >= 0
# triaged = spike_train[:, 1] < 0

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# plt.legend(markerscale=2.5)

# tt = np.arange(0, 100 * (t.max() // 100) , 100)
# plt.xticks(tt, t_start + tt)
# plt.xlabel("time (s)")
# plt.ylabel("depth (um)")
# plt.title(f"{bird}: time vs. registered y.  {kept.sum()} sorted spikes.")
# fig.savefig(dsout / "scatterplots" / "deconv2_scatter_sorted_t_v_regy.png", dpi=300)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# plt.legend(markerscale=2.5)

# tt = np.arange(0, 100 * (t.max() // 100) , 100)
# plt.xticks(tt, t_start + tt)
# plt.xlabel("time (s)")
# plt.ylabel("depth (um)")
# plt.title(f"{bird}: time vs. y.  {kept.sum()} sorted spikes.")
# fig.savefig(dsout / "scatterplots" / "deconv2_scatter_sorted_t_v_y.png")

# %%
merge_thresh_end

# %%
merge_threshold_end_new = merge_thresh_end
spike_train = np.load(dsout / 'deconv2' / 'deconv_spike_train.npy')

# %%
st = spike_train.copy()
# st[:, 1] = newms.registered_maxchan(st, p, geom, pfs=fs)
good_times = np.isin((st[:, 0]) // fs, np.flatnonzero(p_good))
print(f"{good_times.sum()=}")
st[~good_times, 1] = -1
spike_train, templates, order = newms.new_merge_split(
    st,
    geom.shape[0],
    raw_data_bin,
    extract_deconv2_h5,
    geom,
    dsout / f"deconv2clust_mt_{merge_threshold_end_new}",
    n_workers=1,
    merge_resid_threshold=merge_threshold_end_new,
    relocated=do_reloc,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    split_kwargs=dict(
        split_steps=(
            before_deconv_merge_split.herding_split,
        ),
        recursive_steps=(True,),
         split_step_kwargs=(
            dict(
                hdbscan_kwargs=dict(
                    min_cluster_size=25,
                    # min_samples=5,
                    cluster_selection_epsilon=20.0,
                ),
            ),
        ),
    )
)

# %%
(spike_train[:, 1] >= 0).sum(), (spike_train[good_times, 1]>=0).mean()

# %%
# for k in ("split", "merge"):
#     visst = np.load(dsout / "deconv2clust" / f"{k}_st.npy")
#     vissort = Sorting(
#         dsroot / "traces_cached_seg0.raw",
#         geom,
#         visst[:, 0],
#         spike_train_utils.make_labels_contiguous(visst[:, 1]),
#         name="Deconv2Clust" + k.capitalize(),
#         trough_offset=trough_offset,
#         spike_length_samples=spike_length_samples,
#     )
#     od = dsout / "summaries"
#     vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
#     # with h5py.File(extract_deconv2_h5) as h5:
#     #     vissort.make_unit_summaries(
#     #         out_folder=od / f"{vissort.name_lo}_cleaned",
#     #         stored_maxchans=h5["spike_index"][:, 1],
#     #         stored_order=order,
#     #         stored_channel_index=h5["channel_index"][:],
#     #         stored_tpca_projs=h5["cleaned_tpca_projs"],
#     #         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#     #         show_scatter=False,
#     #         relocated=False,
#     #         n_jobs=1,
#     #     )

# %%
# fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
# cluster_viz_index.array_scatter(
#     spike_train[:, 1],
#     geom,
#     x[order],
#     z_reg[order],
#     maxptp[order],
#     axes=axes,
#     zlim=(geom.min() - 25, geom.max() + 25),
#     c=5,
# )
# cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
# cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
# cbar.solids.set(alpha=1)
# cax.set_yticks([0, 1], labels=[3, 15])
# axes[0].set_ylabel("depth (um)")
# axes[1].set_yticks([])
# axes[1].set_xlabel("log peak-to-peak amplitude")
# axes[2].set_yticks([])
# nunits = np.setdiff1d(np.unique(spike_train[:, 1]), [-1]).size
# axes[1].set_title(f"Spatial view of clustered and triaged spikes. {nunits} units.")
# fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter.png")
# plt.close(fig)

# fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
# cluster_viz_index.array_scatter(
#     spike_train[:, 1],
#     geom,
#     x[order],
#     z_reg[order],
#     maxptp[order],
#     axes=axes,
#     zlim=(0, 100),
#     c=5,
# );

# # cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
# # cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
# # cbar.solids.set(alpha=1)
# # cax.set_yticks([0, 1], labels=[3, 15])
# # axes[0].set_ylabel("depth (um)")
# # axes[1].set_yticks([])
# # axes[1].set_xlabel("log peak-to-peak amplitude")
# # axes[2].set_yticks([])
# # axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
# # fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter_detail.png")

# # kept = spike_train[:, 1] >= 0
# # triaged = spike_train[:, 1] < 0

# # fig = plt.figure(figsize=(8, 6))
# # plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# # plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# # plt.legend(markerscale=2.5)

# # tt = np.arange(0, 100 * (t.max() // 100) , 100)
# # plt.xticks(tt, t_start + tt)
# # plt.xlabel("time (s)")
# # plt.ylabel("depth (um)")
# # plt.title(f"{bird}: time vs. registered y.  {kept.sum()} sorted spikes.")
# # fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter_sorted_t_v_regy.png", dpi=300)

# # fig = plt.figure(figsize=(8, 6))
# # plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# # plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# # plt.legend(markerscale=2.5)

# # tt = np.arange(0, 100 * (t.max() // 100) , 100)
# # plt.xticks(tt, t_start + tt)
# # plt.xlabel("time (s)")
# # plt.ylabel("depth (um)")
# # plt.title(f"{bird}: time vs. y.  {kept.sum()} sorted spikes.")
# # fig.savefig(dsout / "scatterplots" / "deconv2clust_scatter_sorted_t_v_y.png")

# %% [markdown]
# ## Deconv 3

# %%
spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
    np.load(dsout / f"deconv2clust_mt_{merge_threshold_end_new}" / "merge_st.npy"),
    geom.shape[0],
    raw_data_bin,
    trough_offset=trough_offset,
    spike_length_samples=spike_length_samples,
    max_shift=0,
    min_n_spikes=5,
)

# %%
assert (order == np.arange(len(order))).all()

# %%
if (dsout / "deconv3").exists():
    shutil.rmtree(dsout / "deconv3")

# %%
# merge_order = np.load(dsout / "deconv2clust" / "merge_order.npy")
# with h5py.File(extract_deconv2_h5) as h5: 
#     z = h5["localizations"][:, 2][merge_order]
#     p = h5["p"][:]
# dsrooterres3 = drifty_deconv.superres_deconv(
#     dsout / "sippx" / "traces_cached_seg0.raw",
#     geom,
#     z,
#     p,
#     spike_train=spike_train,
#     reference_displacement=p_mode,
#     bin_size_um=pitch / 2,
#     pfs=fs,
#     n_jobs=8,
#     deconv_dir=dsout / "deconv3",
#     trough_offset=trough_offset,
#     spike_length_samples=spike_length_samples,
#     threshold=deconv_thresh,
# )

deconv_dict_3 = deconvolve.deconv(
    raw_data_bin,
    dsout / f"deconv3_mt_{merge_threshold_end_new}",
    templates,
    t_start=0,
    t_end=None,
    sampling_rate=fs,
    n_sec_chunk=1,
    n_jobs=8,
    max_upsample=8,
    refractory_period_frames=refractory_period_frames,
    trough_offset=trough_offset,
    threshold=deconv_thresh,
    lambd=0,
    allowed_scale=0.1
)
np.save(dsout / f"deconv3_mt_{merge_threshold_end_new}" / "deconv_spike_train.npy",  deconv_dict_3['deconv_spike_train'])

# %%
# extract_deconv3_h5, extract_deconv3_extra = drifty_deconv.extract_superres_shifted_deconv(
#     superres3,
#     save_cleaned_waveforms=True,
#     save_cleaned_tpca_projs=True,
#     save_residual=False,
#     sampling_rate=fs,
#     subtraction_h5=extract_deconv2_h5,
#     nn_denoise=False,
#     geom=geom,
#     n_jobs=1,
#     # save_reassignment_residuals=True,
#     # pairs_method="radius",
#     max_resid_dist=20,
#     do_reassignment_tpca=True,
#     do_reassignment=False,
#     n_sec_train_feats=80,
#     tpca_weighted=tpca_weighted,
# )

extract_deconv3_h5 = extract_deconv.extract_deconv(
    deconv_dict_3['templates_up'],
    deconv_dict_3['deconv_spike_train_upsampled'],
    dsout / f"deconv3_mt_{merge_threshold_end_new}",
    raw_data_bin,
    subtraction_h5=sub_h5,
    save_cleaned_waveforms=False,
    save_denoised_waveforms=False,
    save_cleaned_tpca_projs=True,
    n_channels_extract=20,
    n_jobs=8,
    device="cpu",
    do_reassignment=False,
    # scratch_dir=deconv_scratch_dir,
)

# %%
overwrite = True
# overwrite = False
rereg = False

with h5py.File(extract_deconv3_h5, "r+" if overwrite else "r") as h5:
    if "z_reg" in h5:
        print("already done, skip")

    samples = h5["spike_index"][:, 0] #- h5["start_sample"][()]
    x, y, z_abs, alpha, _ = h5["localizations"][:].T
    maxptp = maxptps = h5["maxptps"][:]
    spike_index = h5["spike_index"][:]
    maxchans = spike_index[:, 1]
    t = spike_index[:, 0] / fs
    z = z_abs
    
    
    if "z_reg" in h5 and not overwrite:
        z_reg = h5["z_reg"][:]
        pap = p = h5["p"][:]
    else:
        if rereg:
            if overwrite and "z_reg" in h5:
                del h5["z_reg"]
                del h5["p"]

            # z_reg, dispmap = ibme.register_nonrigid(
            z_reg, p = ibme.register_rigid(
                maxptps,
                z_abs - z_abs.min(),
                (samples - samples.min()) / 20000,
                # robust_sigma=1,
                # corr_threshold=0.3,
                adaptive_mincorr_percentile=5,
                disp=300,
                denoise_sigma=0.1,
                # max_dt=100,
                prior_lambda=1,
                batch_size=64,
            )
        else:
            z_reg = z_abs
            p = np.zeros(np.ceil(t.max()).astype(int))
            # *_, tt = ibme.fast_raster(
            #     maxptp, z_abs - z_abs.min(), t
            # )
            # z_reg = ibme.warp_rigid(z_abs, t, tt, p)
        z_reg -= np.median(z_reg - z_abs)
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("p", data=p)

# %%
order = slice(None)
spike_train = deconv_dict_3["deconv_spike_train"]

# %%
# fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
# cluster_viz_index.array_scatter(
#     spike_train[:, 1],
#     geom,
#     x[order],
#     z_reg[order],
#     maxptp[order],
#     axes=axes,
#     zlim=(geom.min() - 25, geom.max() + 25),
#     c=5,
# )
# cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
# cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
# cbar.solids.set(alpha=1)
# cax.set_yticks([0, 1], labels=[3, 15])
# axes[0].set_ylabel("depth (um)")
# axes[1].set_yticks([])
# axes[1].set_xlabel("log peak-to-peak amplitude")
# axes[2].set_yticks([])
# nunits = np.setdiff1d(np.unique(spike_train[:, 1]), [-1]).size
# axes[1].set_title(f"Spatial view of clustered and triaged spikes. {nunits} units.")
# fig.savefig(dsout / "scatterplots" / "deconv3_scatter.png")
# plt.close(fig)

# fig, axes = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw=dict(wspace=0.1))
# cluster_viz_index.array_scatter(
#     spike_train[:, 1],
#     geom,
#     x[order],
#     z_reg[order],
#     maxptp[order],
#     axes=axes,
#     zlim=(0, 400),
#     c=5,
# )
# cax = axes[2].inset_axes([0.05, 0.02, 0.075, 0.2])
# cbar = plt.colorbar(axes[0].collections[0], ax=axes[1], cax=cax, label="amplitude")
# cbar.solids.set(alpha=1)
# cax.set_yticks([0, 1], labels=[3, 15])
# axes[0].set_ylabel("depth (um)")
# axes[1].set_yticks([])
# axes[1].set_xlabel("log peak-to-peak amplitude")
# axes[2].set_yticks([])
# axes[1].set_title(f"Spatial view (zoom) of clustered and triaged spikes. {nunits} units.")
# fig.savefig(dsout / "scatterplots" / "deconv3_scatter_detail.png")

# kept = spike_train[:, 1] >= 0
# triaged = spike_train[:, 1] < 0

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(t[order][triaged], z_reg[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# plt.scatter(t[order][kept], z_reg[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# plt.legend(markerscale=2.5)

# tt = np.arange(0, 100 * (t.max() // 100) , 100)
# plt.xticks(tt, t_start + tt)
# plt.xlabel("time (s)")
# plt.ylabel("depth (um)")
# plt.title(f"time vs. registered y.  {kept.sum()} sorted spikes.")
# fig.savefig(dsout / "scatterplots" / "deconv3_scatter_sorted_t_v_regy.png", dpi=300)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(t[order][triaged], z[order][triaged], color="gray", s=5, alpha=0.5, marker=".", linewidths=0, label="triaged")
# plt.scatter(t[order][kept], z[order][kept], c=spike_train[kept, 1], cmap=cc.m_glasbey, s=5, alpha=0.5, marker=".", linewidths=0)
# plt.legend(markerscale=2.5)

# tt = np.arange(0, 100 * (t.max() // 100) , 100)
# plt.xticks(tt, t_start + tt)
# plt.xlabel("time (s)")
# plt.ylabel("depth (um)")
# plt.title(f"time vs. y.  {kept.sum()} sorted spikes.")
# # fig.savefig(dsout / "scatterplots" / "deconv3_scatter_sorted_t_v_y.png")
# plt.close(fig)

# %%
# M3_dv3 = h5py.File('/media/peter/2TB/sherry/IND106_shank1_Mar3/deconv3/deconv_results.h5', 'r')
# spike_train = np.array(M3_dv3["superres_deconv_spike_train"])
# order = slice(None)
# geom = np.load(dsout / "sippx" / "properties" / "location.npy")
# from spike_psvae import (
#     spike_train_utils)

# %%
from spike_psvae import (chunk_features)

# %%
# visst = spike_train.copy()
# # visst[:, 1] = newms.registered_maxchan(visst, p, geom, pfs=fs)
# good_times = np.isin((visst[:, 0]) // fs, np.flatnonzero(p_good))
# visst[~good_times, 1] = -1
# vissort = Sorting(
#     dsout / "sippx" / "traces_cached_seg0.raw",
#     geom,
#     visst[:, 0],
#     spike_train_utils.make_labels_contiguous(visst[:, 1]),
#     name="Deconv3Stable",
#     trough_offset=trough_offset,
#     spike_length_samples=spike_length_samples,
# )
# od = dsout / "summaries"
# vissort.make_unit_summaries(out_folder=od / f"{vissort.name_lo}_raw")
# with h5py.File(extract_deconv3_h5) as h5:
#     vissort.make_unit_summaries(
#         out_folder=od / f"{vissort.name_lo}_cleaned",
#         show_scatter=True,
#         relocated=False,
#         stored_maxchans=h5["spike_index"][:, 1],
#         stored_channel_index=h5["channel_index"][:],
#         stored_tpca_projs=h5["cleaned_tpca_projs"],
#         stored_tpca=chunk_features.TPCA.load_from_h5(h5, "cleaned"),
#         # stored_order=order,
#         n_jobs = 1
#     )

# %% [markdown]
# # ground-truth evaluation

# %%
merge_threshold_end_new =  3.0

# %%
deconv3_spike_train = np.load(dsout / f'deconv3_mt_{merge_threshold_end_new}' / 'deconv_spike_train.npy')
deconv3_spike_train.shape

# %%
# deconv3_spike_train = np.load(dsout / f'deconv3' / 'deconv_spike_train.npy')
# deconv3_spike_train.shape

# %%
sort_gt.get_num_units()

# %%
dsout

# %%

# %%
deconv3_spike_train = np.load(Path('/media/cat/cole/simulated_results/outputs_ensemble_static_5_min_sim_1') / f'deconv3_mt_{merge_threshold_end_new}' / 'deconv_spike_train.npy')
final_spike_times = deconv3_spike_train[:,0]
final_spike_labels = deconv3_spike_train[:,1]

sorting_deconv_final = si.numpyextractors.NumpySorting.from_times_labels(
        times_list=final_spike_times.astype("int"),
        labels_list=final_spike_labels.astype("int"),
        sampling_frequency=fs,
    )

# deconv1_h5 = h5py.File('/media/cat/cole/simulated_results/outputs_ensemble_static_1_min_sim/deconv1/deconv_results.h5', 'r')
deconv1_spike_train = np.load(dsout / 'deconv1' / 'deconv_spike_train.npy')
deconv1_spike_times = deconv1_spike_train[:,0]
deconv1_spike_labels = deconv1_spike_train[:,1]

sorting_deconv1 = si.numpyextractors.NumpySorting.from_times_labels(
        times_list=deconv1_spike_times.astype("int"),
        labels_list=deconv1_spike_labels.astype("int"),
        sampling_frequency=fs,
    )

# deconv2_h5 = h5py.File('/media/cat/cole/simulated_results/outputs_ensemble_static_1_min_sim/deconv2/deconv_results.h5', 'r')
deconv2_spike_train = np.load(dsout / 'deconv2' / 'deconv_spike_train.npy')
deconv2_spike_times = deconv2_spike_train[:,0]
deconv2_spike_labels = deconv2_spike_train[:,1]

sorting_deconv2 = si.numpyextractors.NumpySorting.from_times_labels(
        times_list=deconv2_spike_times.astype("int"),
        labels_list=deconv2_spike_labels.astype("int"),
        sampling_frequency=fs,
    )

# deconv2_h5 = h5py.File('/media/cat/cole/simulated_results/outputs_ensemble_static_1_min_sim/deconv2/deconv_results.h5', 'r')
deconv2_merge_spike_train = np.load(dsout / f'deconv2clust_mt_{merge_threshold_end_new}' / 'merge_st.npy')
deconv2_merge_spike_train = deconv2_merge_spike_train[np.where(deconv2_merge_spike_train[:,1]!=-1)] #remove outliers
deconv2_merge_spike_times = deconv2_merge_spike_train[:,0]
deconv2_merge_spike_labels = deconv2_merge_spike_train[:,1]

sorting_merge_deconv2 = si.numpyextractors.NumpySorting.from_times_labels(
        times_list=deconv2_merge_spike_times.astype("int"),
        labels_list=deconv2_merge_spike_labels.astype("int"),
        sampling_frequency=fs,
    )

# %%
folder = 'waveform_folder'
we = si.extract_waveforms(
    rec_gt,
    sort_gt,
    folder,
    ms_before=1.5,
    ms_after=2.,
    max_spikes_per_unit=500,
    overwrite=True,
    # load_if_exists=True,
)
print(we)

# %%
snrs = we.get_all_templates().ptp(1).max(1) #si.compute_snrs(waveform_extractor=we)
#ptps = snrs = si.compute_snrs(waveform_extractor=we)
# for name, cmp in [('deconv1', cmp_gt_deconv1),('deconv2', cmp_gt_deconv2),('final_deconv (rp: 10)', cmp_gt),('kilosort (default)', cmp_gt_ks)]:
#     well_detected_units = cmp.get_well_detected_units(well_detected_score = .8)
#     fig, axes = plt.subplots(1,3, figsize=(18,6))
#     axes[0].scatter(snrs.values(), cmp.get_performance()['precision'])
#     axes[0].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
#     axes[0].set_ylabel('precision')
#     axes[0].set_xlabel('snr')

#     axes[1].scatter(snrs.values(), cmp.get_performance()['recall'])
#     axes[1].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
#     axes[1].set_ylabel('recall')
#     axes[1].set_xlabel('snr')

#     axes[2].scatter(snrs.values(), cmp.get_performance()['accuracy'])
#     axes[2].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
#     axes[2].set_ylabel('accuracy')
#     axes[2].set_xlabel('snr')
    
    
    

# %%
we.

# %%
we.get_all_templates()

# %%
sorting_deconv_final.get_num_units()

# %%
cmp_gt = si.compare_sorter_to_ground_truth(sort_gt, sorting_deconv_final, exhaustive_gt=True, match_score=.1)
cmp_gt_deconv1 = si.compare_sorter_to_ground_truth(sort_gt, sorting_deconv1, exhaustive_gt=True, match_score=.1)
cmp_gt_deconv2 = si.compare_sorter_to_ground_truth(sort_gt, sorting_deconv2, exhaustive_gt=True, match_score=.1)
cmp_gt_merge_deconv2 = si.compare_sorter_to_ground_truth(sort_gt, sorting_merge_deconv2, exhaustive_gt=True, match_score=.1)
sort_ks = si.read_kilosort('/media/cat/cole/kilosort_results_sim_5min/KS_output/data/')
cmp_gt_ks = si.compare_sorter_to_ground_truth(sort_gt, sort_ks, exhaustive_gt=True, match_score=.1)

# %%
fig, axes = plt.subplots(5,1, figsize=(20,10))
axes[0].imshow(cmp_gt_deconv1.get_ordered_agreement_scores().to_numpy())
axes[0].set_title('deconv 1')
axes[1].imshow(cmp_gt_deconv2.get_ordered_agreement_scores().to_numpy())
axes[1].set_title('deconv 2')
axes[2].imshow(cmp_gt_merge_deconv2.get_ordered_agreement_scores().to_numpy())
axes[2].set_title('deconv 2 merge')
axes[3].imshow(cmp_gt.get_ordered_agreement_scores().to_numpy())
axes[3].set_title('final deconv');
axes[4].imshow(cmp_gt_ks.get_ordered_agreement_scores().to_numpy())
axes[4].set_title('kilosort (default)');
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# %%
print(cmp_gt.get_performance('pooled_with_average'))
well_detected_units = cmp_gt.get_well_detected_units(well_detected_score = .8)
bad_units = cmp_gt.get_bad_units()
print(f"well-detected (>.8): {len(well_detected_units)}")

# %%
print(cmp_gt_ks.get_performance('pooled_with_average'))
well_detected_units = cmp_gt_ks.get_well_detected_units(well_detected_score = .8)
bad_units = cmp_gt_ks.get_bad_units()
print(f"well-detected (>.8): {len(well_detected_units)}")

# %%
for name, cmp in [('deconv1', cmp_gt_deconv1),('deconv2', cmp_gt_deconv2),('final_deconv (rp: 10)', cmp_gt),('kilosort (default)', cmp_gt_ks)]:
    well_detected_units = cmp.get_well_detected_units(well_detected_score = .8)
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].scatter(snrs, cmp.get_performance()['precision'])
    axes[0].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[0].set_ylabel('precision')
    axes[0].set_xlabel('ptp')

    axes[1].scatter(snrs, cmp.get_performance()['recall'])
    axes[1].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[1].set_ylabel('recall')
    axes[1].set_xlabel('ptp')

    axes[2].scatter(snrs, cmp.get_performance()['accuracy'])
    axes[2].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
    axes[2].set_ylabel('accuracy')
    axes[2].set_xlabel('ptp')

# %%
# for name, cmp in [('deconv1', cmp_gt_deconv1),('deconv2', cmp_gt_deconv2),('final_deconv (rp: 10)', cmp_gt),('kilosort (default)', cmp_gt_ks)]:
#     well_detected_units = cmp.get_well_detected_units(well_detected_score = .8)
#     fig, axes = plt.subplots(1,3, figsize=(18,6))
#     axes[0].scatter(snrs.values(), cmp.get_performance()['precision'])
#     axes[0].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
#     axes[0].set_ylabel('precision')
#     axes[0].set_xlabel('snr')

#     axes[1].scatter(snrs.values(), cmp.get_performance()['recall'])
#     axes[1].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
#     axes[1].set_ylabel('recall')
#     axes[1].set_xlabel('snr')

#     axes[2].scatter(snrs.values(), cmp.get_performance()['accuracy'])
#     axes[2].set_title(f"{name}, total units: {len(sort_gt.get_unit_ids())}, num well-detected units: {len(well_detected_units)}")
#     axes[2].set_ylabel('accuracy')
#     axes[2].set_xlabel('snr')
    
    
    

# %%
sort_ks

# %%
sorting_deconv_final

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Phy export

# %%
# from spike_psvae.cluster_utils import read_waveforms, compare_two_sorters, make_sorting_from_labels_frames
# from spike_psvae.cluster_viz import plot_agreement_venn, plot_unit_similarities,plot_agreement_venn_better
# from spike_psvae.cluster_utils import get_closest_clusters_kilosort_hdbscan
# from spike_psvae.cluster_viz import plot_single_unit_summary
# from spike_psvae.cluster_viz import cluster_scatter, plot_waveforms_geom, plot_venn_agreement #plot_raw_waveforms_unit_geom,
# from spike_psvae.cluster_viz import array_scatter, plot_self_agreement, plot_single_unit_summary, plot_agreement_venn, plot_isi_distribution, plot_waveforms_geom_unit, plot_unit_similarities
# from spike_psvae.cluster_viz import plot_unit_similarity_heatmaps
# from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, remove_duplicate_units #run_weighted_triage_adaptive,
# from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
# from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan
# from sklearn import metrics
# import spikeinterface.extractors as se
# import spikeinterface as si

# %%
# if (dsout / "sisorting").exists():
#     (dsout / "sisorting").unlink()
# sorting = si.NumpySorting.from_times_labels(times, labels, sampling_frequency=fs)
# sorting = sorting.save(folder=dsout / "sisorting")
# sorting

# %%
# binrec = si.read_binary_folder(dsout / "sippx")
# binrec.annotate(is_filtered=True)
# binrec

# %%
# if (dsout / "siwfs").exists():
#     (dsout / "siwfs").unlink()

# %%
# we = si.extract_waveforms(binrec, sorting, dsout / "siwfs")

# %%
# from spikeinterface.exporters import export_to_phy

# %%
# if (dsout / "phy").exists():
#     (dsout / "phy").unlink()

# %%
# export_to_phy(we, dsout / "phy", n_jobs=8, chunk_size=30000)
