# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
import numpy as np
import h5py
from tqdm.auto import tqdm, trange
import scipy.io
import time
import torch

# %%
# templates = np.load("/mnt/3TB/charlie/data/high_snr_templates.npy")
# templates.shape

# %%
import subprocess

# %%
from spike_psvae import (
    simdata,
    subtract,
    ibme,
    denoise,
    template_reassignment,
    snr_templates,
    grab_and_localize,
    localize_index,
    cluster_viz,
    cluster_viz_index,
    deconvolve,
    relocalize_after_deconv,
    extract_deconv,
    residual,
)

# %%
import matplotlib.pyplot as plt
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
base_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/")
# base_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/")

# %%
in_dir = next(base_dir.glob("*input"))
in_dir, in_dir.exists()

# %%
out_dir = next(base_dir.glob("*output"))
# out_dir.mkdir(exist_ok=True)
out_dir, out_dir.exists()

# %%
# %ls {out_dir}

# %%
sub_dir = next(base_dir.glob("*subtraction"))
# sub_dir.mkdir(exist_ok=True)
sub_dir, sub_dir.exists()

# %%
ks_dir = next(base_dir.glob("*kilosort"))
# ks_dir.mkdir(exist_ok=True)
ks_dir, ks_dir.exists()

# %%
deconv_dir = next(base_dir.glob("*deconv"))
# deconv_dir.mkdir(exist_ok=True)
deconv_dir, deconv_dir.exists()

# %%
vis_dir = base_dir / "figs"
vis_dir.mkdir(exist_ok=True)

# %%
# in_bins = list(sorted(Path("/mnt/3TB/charlie/hybrid_1min_input/").glob("*.bin")))
in_bins = []
while True:
    in_bins = list(sorted(Path(in_dir).glob("*.bin")))
    print(len(in_bins), end=", ")
    if len(in_bins) >= 10:
        break
    time.sleep(10 * 60)
print("ok!")
in_bins

# %%
an_h5 = Path("/mnt/3TB/charlie/steps_5min/yeslb/subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5")
if not an_h5.exists():
    an_h5 = next(next(sub_dir.glob("*")).glob("sub*h5"))

# %%
with h5py.File(an_h5) as h5:
    geom = h5["geom"][:]

# %%
write_channel_index = subtract.make_contiguous_channel_index(384)
loc_channel_index = subtract.make_contiguous_channel_index(384, n_neighbors=20)

# %% [markdown] tags=[]
# # make hybrid binary

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    
    meta = in_bin.parent / f"{in_bin.stem}.meta"
    out_bin = out_dir / f"{in_bin.stem}.bin"
    if (out_dir / f"{in_bin.stem}.meta").exists():
        (out_dir / f"{in_bin.stem}.meta").unlink()
    (out_dir / f"{in_bin.stem}.meta").symlink_to(meta)
    gt_h5 = out_dir / f"{subject}_gt.h5"
    
    hybrid_raw, spike_train, spike_index, waveforms, choices, templates, cluster_maxchans = simdata.hybrid_recording(
        in_bin,
        templates,
        geom,
        loc_channel_index,
        write_channel_index,
        do_noise=False,
        mean_spike_rate=(0.1, 10),
        seed=i,
    )
    hybrid_raw.tofile(out_bin)
    
    with h5py.File(gt_h5, "w") as h5:
        h5.create_dataset("spike_train", data=spike_train)
        h5.create_dataset("spike_index", data=spike_index)
        h5.create_dataset("templates", data=templates)
        h5.create_dataset("write_channel_index", data=write_channel_index)
        h5.create_dataset("loc_channel_index", data=loc_channel_index)
        h5.create_dataset("geom", data=geom)
        h5.create_dataset("choices", data=choices)
        h5.create_dataset("cluster_maxchans", data=cluster_maxchans)

# %%

# %% [markdown] tags=[]
# # Vis to check

# %%
(vis_dir / "raw_data").mkdir(exist_ok=True)
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    out_bin = out_dir / f"{in_bin.stem}.bin"
    
    seg_in = np.fromfile(in_bin, dtype=np.float32, count=1_000 * 384, offset=28 * 4 * 30_000 * 384)
    seg_in = seg_in.reshape(1_000, 384)
    seg_out = np.fromfile(out_bin, dtype=np.float32, count=1_000 * 384, offset=28 * 4 * 30_000 * 384)
    seg_out = seg_out.reshape(1_000, 384)
    diff = seg_out - seg_in
    
    seg_in = np.clip(seg_in, -6, 6)
    seg_out = np.clip(seg_out, -6, 6)
    diff = np.clip(diff, -6, 6)
    # plt.imshow(seg_in.T, aspect=0.5 * diff.shape[0] / diff.shape[1])
    # plt.colorbar()
    # plt.show()
    
    mos = "a..\nb.x\nc.."
    fig, axes = plt.subplot_mosaic(mos, gridspec_kw=dict(width_ratios=[5, 0.0, 0.1], wspace=0.05, hspace=0.01), figsize=(4, 6))
    vmin = min(seg_in.min(), seg_out.min(), diff.min())
    vmax = max(seg_in.max(), seg_out.max(), diff.max())
    vmin = min(vmin, -vmax)
    vmax = max(-vmin, vmax)
    
    axes["a"].imshow(seg_in.T, aspect=0.5 * diff.shape[0] / diff.shape[1], vmin=vmin, vmax=vmax, cmap=plt.cm.RdGy)
    axes["b"].imshow(seg_out.T, aspect=0.5 * diff.shape[0] / diff.shape[1], vmin=vmin, vmax=vmax, cmap=plt.cm.RdGy)
    i = axes["c"].imshow((seg_out - seg_in).T, aspect=0.5 * diff.shape[0] / diff.shape[1], vmin=vmin, vmax=vmax, cmap=plt.cm.RdGy)
    plt.colorbar(i, cax=axes["x"])
    
    axes["a"].set_title(f"{subject}")
    axes["a"].set_ylabel("original")
    axes["b"].set_ylabel("hybrid")
    axes["c"].set_ylabel("diff")
    axes["a"].set_xticks([])
    axes["b"].set_xticks([])
    axes["c"].set_xlabel("time (samples)")
    
    fig.savefig(vis_dir / "raw_data" / f"{subject}_raw_hybrid_diff.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)



# %% tags=[]
(vis_dir / "unmatched_gt_scatter").mkdir(exist_ok=True)
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    
    meta = in_bin.parent / f"{in_bin.stem}.meta"
    out_bin = out_dir / f"{in_bin.stem}.bin"
    gt_h5 = out_dir / f"{subject}_gt.h5"

    with h5py.File(gt_h5, "r") as gt_h5:
        gt_spike_train = gt_h5["spike_train"][:]
        gt_spike_index = gt_h5["spike_index"][:]
        templates = gt_h5["templates"][:]
        geom = gt_h5["geom"][:]
        cluster_maxchans = gt_h5["cluster_maxchans"][:]
    
    gt_relocalizations, gt_remaxptp = grab_and_localize.grab_and_localize(
        gt_spike_index,
        out_bin,
        geom,
        loc_radius=100,
        nn_denoise=True,
        enforce_decrease=False,
        tpca=None,
        chunk_size=30_000,
        n_jobs=6,
    )

    gt_template_maxchans = templates.ptp(1).argmax(1)
    gt_template_locs = localize_index.localize_ptps_index(
        templates.ptp(1),
        geom,
        gt_template_maxchans,
        np.stack([np.arange(len(geom))] * len(geom), axis=0),
        n_channels=20,
        n_workers=None,
        pbar=True,
    )
    
    fig, (aa, ab) = plt.subplots(1, 2, figsize=(6, 6), sharey=True)
    
    cluster_viz_index.cluster_scatter(
        gt_relocalizations[:, 0], gt_relocalizations[:, 3], gt_spike_train[:, 1], ax=aa
    )
    cluster_viz_index.cluster_scatter(
        np.log(gt_remaxptp), gt_relocalizations[:, 3], gt_spike_train[:, 1], ax=ab
    )
    
    aa.set_ylabel("z")
    aa.set_xlabel("x")
    ab.set_xlabel("log ptp")
    
    aa.scatter(*geom.T, marker="s", s=2, color="orange")
    
    aa.scatter(gt_template_locs[0], gt_template_locs[3], color="k", marker="x", s=10)
    ab.scatter(np.log(templates.ptp(1).max(1)), gt_template_locs[3], color="k", marker="x", s=10)
    
    fig.suptitle(f"{subject} template locs + re-localizations", y=0.95)
    fig.tight_layout()
    fig.savefig(vis_dir / "unmatched_gt_scatter" / f"{subject}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %%
# make data for kilosort
# S2V_AP = 2.34375e-06
scaleproc = 200
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    out_bin = out_dir / f"{in_bin.stem}.bin"

    ks_sub_dir = ks_dir / subject
    ks_sub_dir.mkdir(exist_ok=True)
    ks_bin = ks_sub_dir / f"{in_bin.stem}.bin"
    ks_chanmap = ks_sub_dir / f"chanMap.mat"
    
    # write int16 binary since KS needs that
    # using S2V_AP really helps preserve the dynamic range
    (np.fromfile(out_bin, dtype=np.float32) * scaleproc).astype(np.int16).tofile(ks_bin)
    import gc; gc.collect()

    # we need a chanMap
    chanMap = dict(
        chanMap=np.arange(1, 385, dtype=np.float64),
        chanMap0ind=np.arange(384, dtype=np.float64),
        connected=np.ones(384, dtype=np.float64),
        fs=np.array([30000]),
        kcoords=np.ones(384, dtype=np.float64),
        shankInd=np.ones(384, dtype=np.float64),
        xcoords=geom[:, 0].astype(np.float64),
        ycoords=geom[:, 1].astype(np.float64),
    )
    scipy.io.savemat(ks_chanmap, chanMap)


# %%

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    out_bin = out_dir / f"{in_bin.stem}.bin"
    
    subject_sub_dir = sub_dir / subject
    subject_sub_dir.mkdir(exist_ok=True)

    sub_h5 = subtract.subtraction(
        out_bin,
        subject_sub_dir,
        n_sec_pca=20,
        # t_start=10,
        # t_end=1010,
        sampling_rate=30_000,
        thresholds=[12, 10, 8, 6, 5, 4],
        denoise_detect=True,
        neighborhood_kind="firstchan",
        # extract_box_radius=200,
        # dedup_spatial_radius=70,
        enforce_decrease_kind="radial",
        n_jobs=2,
        save_residual=True,
        save_waveforms=True,
        do_clean=True,
        localization_kind="logbarrier",
        localize_radius=100,
        loc_workers=4,
        # overwrite=True,
        random_seed=0,
    )

# %%
import gc; gc.collect()
import torch; torch.cuda.empty_cache()

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    out_bin = out_dir / f"{in_bin.stem}.bin"
    
    subject_sub_dir = sub_dir / subject
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    
    with h5py.File(subject_sub_h5, "r+") as h5:
        samples = h5["spike_index"][:, 0] - h5["start_sample"][()]
        z_abs = h5["localizations"][:, 2]
        maxptps = h5["maxptps"]

        z_reg, dispmap = ibme.register_nonrigid(
            maxptps,
            z_abs,
            samples / 30000,
            robust_sigma=1,
            corr_threshold=0.1,
            disp=200,
            denoise_sigma=0.1,
            rigid_init=False,
            n_windows=10,
            widthmul=0.5,
        )
        z_reg -= (z_reg - z_abs).mean()
        dispmap -= dispmap.mean()
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("dispmap", data=dispmap)

# %%
import gc; gc.collect()
import torch; torch.cuda.empty_cache()

# %%

# %% tags=[]
# cluster + deconv in one go for better cache behavior
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    print(subject)
    # if subject != "SWC_054":
        # continue
    out_bin = out_dir / f"{in_bin.stem}.bin"
    
    subject_sub_dir = sub_dir / subject
    subject_sub_h5 = next(subject_sub_dir.glob("sub*.h5"))
    subject_res_bin = next(subject_sub_dir.glob("res*.bin"))
    (deconv_dir / subject).mkdir(exist_ok=True, parents=True)
    
    if not (subject_sub_dir / "aligned_spike_index.npy").exists():
        res = subprocess.run(
            [
                "python", "../scripts/duster.py", out_bin, subject_res_bin, subject_sub_h5, subject_sub_dir,
                "--inmem",
            ]
        )
        print(subject, res.returncode)
        print(res)
    else:
        print(subject_sub_dir / "aligned_spike_index.npy", "exists, skipping", subject)

    # res = subprocess.run(
    #     [
    #         "python",
    #         "../scripts/run_deconv_merge_split.py",
    #         subject_sub_dir /"aligned_spike_index.npy",
    #         subject_sub_dir /"labels.npy",
    #         out_bin,
    #         subject_sub_dir,
    #         deconv_dir / subject,
    #     ]
    # )


# %%
1

# %%
# # print helpful stuff for copypasting kilosort sbatch commands
# for i, in_bin in enumerate(tqdm(in_bins)):
#     subject = in_bin.stem.split(".")[0]
#     out_bin = out_dir / f"{in_bin.stem}.bin"

#     ks_sub_dir = ks_dir / subject
#     ks_sub_dir.mkdir(exist_ok=True)
#     ks_bin = ks_sub_dir / f"{in_bin.stem}.bin"
    # ks_chanmap = ks_sub_dir / f"chanMap.mat"
    
#     print(
#         "kilosort2.5 "
#         f"-d {ks_sub_dir} "
#         "-t /tmp/hiddd "
#         "-c CONFIG_M "
#         f"-m {ks_chanmap} "
#         "-n 384 -s 0 -e Inf"
#     )

# %%
# %ll /share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_deconv/CSHL049/

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":
    if subject != "NYU-12":
        continue
    
    out_bin = out_dir / f"{in_bin.stem}.bin"
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_res_bin = next((sub_dir / subject).glob("res*.bin"))
    subject_deconv_dir = deconv_dir / subject
    subject_deconv_dir.mkdir(exist_ok=True, parents=True)
    # if (subject_deconv_dir / "spike_train.npy").exists():
    #     print("done, skip")
    #     continue
    
    # if (subject_deconv_dir / "spike_train.npy").exists():
    #     print("done, skipping")
    #     continue
    
    with h5py.File(subject_sub_h5) as h5:
        geom = h5["geom"][:]
        se, ss = h5["end_sample"][()], h5["start_sample"][()]
        ds = se - ss
        print(ds, ss, se)
    
    aligned_spike_index = np.load(sub_dir / subject / "aligned_spike_index.npy")
    print("asi", aligned_spike_index[:, 0].min(), aligned_spike_index[:, 0].max())
    final_labels = np.load(sub_dir / subject / "labels.npy")
    which = (final_labels >= 0) & (aligned_spike_index[:, 0] > 70) & (aligned_spike_index[:,0] < (ds - 79))
    print("asi2", aligned_spike_index[which, 0].min(), aligned_spike_index[which, 0].max())


    template_spike_train = np.c_[aligned_spike_index[which, 0], final_labels[which]]
    print("Nunits", template_spike_train[:, 1].max() + 1)
    
    np.save(subject_deconv_dir / "geom.npy", geom)
    fname_templates_up,fname_spike_train_up,template_path,fname_spike_train = deconvolve.deconvolution(
        aligned_spike_index[which],
        final_labels[which],
        subject_deconv_dir,
        out_bin,
        subject_res_bin,
        template_spike_train,
        subject_deconv_dir / "geom.npy",
        threshold=40,
        max_upsample=8,
        vis_su_threshold=1.0,
        approx_rank=5,
        multi_processing=True,
        cleaned_temps=True,
        n_processors=8,
        sampling_rate=30000,
        deconv_max_iters=1000,
        t_start=0,
        t_end=None,
        n_sec_chunk=1,
        verbose=False,
    )


# %%
tup = np.load(subject_deconv_dir / "templates_up.npy")

# %%
tup[np.arange(len(tup)),:,tup.ptp(1).argmax(1)].argmin(1)

# %%
for in_bin in in_bins:
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":
    if subject != "NYU-12":
        continue
    print(subject)
    
    out_bin = out_dir / f"{in_bin.stem}.bin"
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_res_bin = next((sub_dir / subject).glob("res*.bin"))
    subject_deconv_dir = deconv_dir / subject
    subject_deconv_dir.mkdir(exist_ok=True, parents=True)
    # print("hi")
    # if (subject_deconv_dir / "denoised_wfs.h5").exists() and (subject_deconv_dir / "collision_subtracted_wfs.h5").exists():
    #     print("wfs done")
    # if (subject_deconv_dir / "localization_results.npy").exists():
    #     print("locs done")
    # if (subject_deconv_dir / "z_reg.npy").exists():
    #     print("z_reg done")
    extract_deconv.extract_deconv(
        subject_deconv_dir / "templates_up.npy",
        subject_deconv_dir / "spike_train_up.npy",
        subject_deconv_dir,
        out_bin,
        subtraction_h5=subject_sub_h5,
        # save_denoised_waveforms=True,
        n_channels_extract=40,
        n_jobs=13,
        # device="cpu",
        scratch_dir="/tmp/hibbb"
    )

# %%
with h5py.File(subject_deconv_dir / "deconv_results.h5") as f:
    for k in f:
        print(k, f[k].shape)

# %%
with h5py.File(subject_deconv_dir / "deconv_results.h5") as f:
    cc = f["cleaned_waveforms"][:2048]
    tu = f["templates_up"][:]
    tl = f["templates_loc"][:]
    # dd = f["denoised_waveforms"][:2048]

# %%
with h5py.File(subject_deconv_dir / "deconv_results.h5") as f:
    tmc = f["templates_up_maxchans"][:]
    ci = f["channel_index"][:]

# %%
ci

# %%
ccp = cc.ptp(1).max(1)
# ddp = dd.ptp(1).max(1)
tup = tu.ptp(1).max(1)
tlp = tl.ptp(1).max(1)

# %%
ccp.min(), ccp.max()

# %%
# ddp.min(), ddp.max()

# %%
plt.hist(cc[np.arange(len(cc)),:,cc.ptp(1).argmax(1)].argmin(1), bins=np.arange(121));

# %%
# dd[np.arange(len(dd)),:,dd.ptp(1).argmax(1)].argmin(1)

# %%
(tup == tlp).all()

# %%
tlp.min(), tlp.max()

# %%
fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True)
rr = np.memmap(out_bin, dtype=np.float32).reshape(-1,384)[:1000]
re = np.memmap(subject_deconv_dir / "residual.bin", dtype=np.float32).reshape(-1,384)[:1000]
aa.imshow(rr)
ab.imshow(re)
ac.imshow(rr - re)
plt.show()

# %%
# fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True)
# rr = np.memmap(out_bin, dtype=np.float32).reshape(-1,384)[29500:30500]
# re = np.memmap(subject_deconv_dir / "residual.bin", dtype=np.float32).reshape(-1,384)[29500:30500]
# aa.imshow(rr)
# ab.imshow(re)
# ac.imshow(rr - re)
# aa.set_title("raw")
# ab.set_title("resid")
# ac.set_title("diff")
# plt.show()

# %%
with h5py.File(subject_deconv_dir / "deconv_results.h5") as f:
    locs = f["localizations"][:]
    x = locs[:, 0]
    z = locs[:, 3]
    maxptps = f["maxptps"][:]
    # nmaxptps = maxptps - maxptps.min()
    # nmaxptps *= 0.74 / nmaxptps.max()
    # nmaxptps += 0.25
    # plt.figure(figsize=(5, 25))
    # plt.scatter(x, z, c=maxptps, alpha=nmaxptps, s=1)
    # plt.show()

# %%
plt.hist(maxptps,bins=32);

# %%
labels = np.load(subject_deconv_dir / "spike_train.npy")[:, 1]

cluster_viz.array_scatter(
    labels,
    geom,
    x,
    z,
    maxptps,
    zlim=(-50, 3900),
    axes=None,
    # annotate=False,
    do_ellipse=False,
)

# %%
lold = np.load(subject_deconv_dir / "localization_results.npy")

# %%
labold = np.load(subject_deconv_dir / "spike_labels.npy")

# %%
# %ll {subject_deconv_dir}

# %%
lold.shape,labold.shape

# %%
cluster_viz.array_scatter(
    labels[:len(labold)],
    geom,
    lold[:,0],
    lold[:,1],
    lold[:,4],
    zlim=(-50, 3900),
    axes=None,
    # annotate=False,
    do_ellipse=False,
)

# %%
import torch

def run_relocalize(
    spike_train_npy, deconv_templates_up_npy, h5_subtract, residual_path,
    output_directory, standardized_bin
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = denoise.SingleChanDenoiser()
    denoiser.load()
    denoiser.to(device)
    output_directory = Path(output_directory)

    with h5py.File(h5_subtract) as f:
        geom_array = f["geom"][:]
    np.save(output_directory / "geom.npy", geom_array)

    deconv_spike_train_up = np.load(spike_train_npy)
    deconv_templates_up = np.load(deconv_templates_up_npy)

    n_spikes = deconv_spike_train_up.shape[0]
    print(f"number of deconv spikes: {n_spikes}")
    print(f"deconv templates shape: {deconv_templates_up.shape}")
    
    res_path = residual.run_residual(
        deconv_templates_up_npy,
        spike_train_npy,
        output_directory,
        standardized_bin,
        output_directory / "geom.npy",
    )

    # 42/60 issue :
    # deconvolve.read_waveforms used in this function reads at t-60:t+60
    # and pass wfs through denoising pipeline

    # Save all wfs in first_outdir
    n_chans_to_extract = 40
    
    if True:#not ((output_directory / "denoised_wfs.h5").exists() and (output_directory / "collision_subtracted_wfs.h5").exists()):
        (
            fname_spike_index,
            fname_spike_labels,
            fname_subtracted,
            cleaned_wfs_h5,
            denoised_wfs_h5,
        ) = relocalize_after_deconv.extract_deconv_wfs(
            h5_subtract,
            res_path,
            geom_array,
            deconv_spike_train_up,
            deconv_templates_up,
            output_directory,
            denoiser,
            device,
            n_chans_to_extract=n_chans_to_extract,
        )
    else:
        print("locs done, skip")
        fname_spike_index = output_directory / "spike_index.npy"
        fname_spike_labels = output_directory / "spike_labels.npy"
        denoised_wfs_h5 = output_directory / "denoised_wfs.h5"
        cleaned_wfs_h5 = output_directory / "collision_subtracted_wfs.h5"

    # Relocalize Waveforms

    deconv_spike_index = np.load(fname_spike_index)
    # assert deconv_spike_index.shape[0] == n_spikes
    print(f"number of deconv spikes: {deconv_spike_index.shape[0]}")
    if True:#not (output_directory / "localization_results.npy").exists():
        relocalize_after_deconv.relocalize_extracted_wfs(
            denoised_wfs_h5,
            deconv_spike_train_up,
            deconv_spike_index,
            geom_array,
            output_directory,
            n_workers=12,
            batch_size=512,
        )
    else:
        print("locs done, skip")

    localization_results_path = output_directory / "localization_results.npy"
    maxptpss = np.load(localization_results_path)[:, 4]
    z_absss = np.load(localization_results_path)[:, 1]
    times = deconv_spike_train_up[:, 0].copy() / 30000

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
    np.save(output_directory / "z_reg.npy", z_reg)
    np.save(output_directory / "ptps.npy", maxptpss)


# %%
for in_bin in in_bins:
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":
    # # if subject != "NYU-12":
        # continue
    
    out_bin = out_dir / f"{in_bin.stem}.bin"
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_res_bin = next((sub_dir / subject).glob("res*.bin"))
    subject_deconv_dir = deconv_dir / subject
    subject_deconv_dir.mkdir(exist_ok=True, parents=True)
    print("hi")
    if (subject_deconv_dir / "denoised_wfs.h5").exists() and (subject_deconv_dir / "collision_subtracted_wfs.h5").exists():
        print("wfs done")
    if (subject_deconv_dir / "localization_results.npy").exists():
        print("locs done")
    if (subject_deconv_dir / "z_reg.npy").exists():
        print("z_reg done")


# %% tags=[]
def job(in_bin):
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":
    print("subject", flush=True)
    if subject != "NYU-12":
        return
    print("run", flush=True)


    out_bin = out_dir / f"{in_bin.stem}.bin"
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_res_bin = next((sub_dir / subject).glob("res*.bin"))
    subject_deconv_dir = deconv_dir / subject
    subject_deconv_dir.mkdir(exist_ok=True, parents=True)
    # if (subject_deconv_dir / "z_reg.npy").exists():
    #     print("done, skip")
    #     return

    run_relocalize(
        subject_deconv_dir / "spike_train_up.npy",
        subject_deconv_dir / "templates_up.npy",
        subject_sub_h5,
        None,
        subject_deconv_dir,
        out_bin,
    )

    
from joblib import Parallel, delayed
with Parallel(1) as p:
    res = list(p(delayed(job)(in_bin) for in_bin in tqdm(in_bins)))

# %%
1

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    out_bin = out_dir / f"{in_bin.stem}.bin"
    
    subject_sub_dir = sub_dir / subject
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_res_bin = next((sub_dir / subject).glob("res*.bin"))
    
    final_labels = np.load(subject_sub_dir / "labels.npy")
    
    with h5py.File(subject_sub_h5) as h5:
        templates, snrs = snr_templates.get_templates(
            np.c_[
                h5["spike_index"][:, 0],
                final_labels,
            ],
            h5["geom"][:],
            out_bin,
            subject_res_bin,
            subtracted_waveforms=h5["subtracted_waveforms"],
            subtracted_max_channels=h5["spike_index"][:, 1],
            extract_channel_index=h5["channel_index"][:],
            do_tpca=True,
        )
        
    np.save(subject_sub_dir / "cleaned_templates.npy", templates)
    np.save(subject_sub_dir / "snrs.npy", snrs)
    
    reassignments = template_reassignment.template_reassignment(
        subject_sub_h5,
        subject_res_bin,
        templates,
        metric="cosine",
        batch_size=512,
        n_jobs=20,
    )
    np.save(subject_sub_dir / "reassignments.npy", snrs)


# %%
# %ll {sub_dir / subject}

# %%
upsample 8 8
unit_up_factor (265,)
up_factor 8
self.up_up_map (2120,)

# %%
u = np.unique(final_labels[final_labels >= 0])

# %%
u

# %%
u.size

# %%
1

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    print(subject, end=", ")

# %%
fname_templates_up

# %%
# 
