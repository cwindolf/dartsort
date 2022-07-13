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

# %%
from pathlib import Path
import numpy as np
import h5py
from tqdm.auto import tqdm, trange
import scipy.io
import time
import torch
import shutil

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
    extract_deconv,
    residual,
    pre_deconv_merge_split,
    after_deconv_merge_split,
    cluster_utils,
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

# %%
import time
class timer:
    def __init__(self, name="timer"):
        self.name = name
        print("start", name, "...")
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


# %%
# base_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/")
base_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/")

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
# %ll {sub_dir}/CSHL051

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

# %%
for meta in Path(in_dir).glob("*.meta"):
    try:
        (out_dir / meta.name).unlink()
    except FileNotFoundError:
        pass
    shutil.copy(meta, out_dir / meta.name)


# %% [markdown] tags=[]
# # make hybrid binary

# %%
1

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

# %% [markdown]
# # KS export

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

# %% [markdown]
# # subtraction

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    if subject not in ("DY_018", "CSHL051"):
        continue
    print(subject)
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
        save_residual=True,
        save_waveforms=True,
        do_clean=True,
        localization_kind="logbarrier",
        localize_radius=100,
        loc_workers=3,
        # overwrite=True,
        random_seed=0,
        n_jobs=8,
    )

# %%
import gc; gc.collect()
import torch; torch.cuda.empty_cache()

# %% [markdown]
# # registration

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    print(subject)
    out_bin = out_dir / f"{in_bin.stem}.bin"
    
    subject_sub_dir = sub_dir / subject
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    
    with h5py.File(subject_sub_h5, "r+") as h5:
        if "z_reg" in h5:
            print("already done, skip")
            continue
        
        samples = h5["spike_index"][:, 0] - h5["start_sample"][()]
        z_abs = h5["localizations"][:, 2]
        maxptps = h5["maxptps"]

        z_reg, dispmap = ibme.register_nonrigid(
            maxptps,
            z_abs,
            samples / 30000,
            robust_sigma=1,
            corr_threshold=0.6,
            disp=1500,
            denoise_sigma=0.1,
            rigid_init=False,
            n_windows=10,
        )
        z_reg -= (z_reg - z_abs).mean()
        dispmap -= dispmap.mean()
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("dispmap", data=dispmap)

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
        print(subject)
        print("-" * len(subject))
        for k in h5:
            print(" - ", k, h5[k].shape)

# %% [markdown]
# # duster

# %%
# %rm -rf /local/duster/*

# %%
1

# %%
1

# %%

# %% tags=[]
# cluster + deconv in one go for better cache behavior
just_do_it = False
just_do_it = True

# from joblib import Parallel, delayed
# for res in tqdm(Parallel(5)(delayed(job)(in_bin) for in_bin in in_bins)):
#     print(res)
Path("/local/duster").mkdir(exist_ok=True)
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    print(subject, flush=True)
    if subject not in ("DY_018",):
        continue
    # if subject != "SWC_054":
        # continue
    out_bin = out_dir / f"{in_bin.stem}.bin"
    
    subject_sub_dir = sub_dir / subject
    subject_sub_h5 = next(subject_sub_dir.glob("sub*.h5"))
    subject_res_bin = next(subject_sub_dir.glob("res*.bin"))
    (deconv_dir / subject).mkdir(exist_ok=True, parents=True)
    clust_plotdir = vis_dir / f"{subject}_clust_merge_split"
    clust_plotdir.mkdir(exist_ok=True)
    
    if just_do_it or not (subject_sub_dir / "aligned_spike_index.npy").exists():
        # print("
        res = subprocess.run(
            [
                "python",
                "../scripts/duster.py",
                out_bin,
                subject_res_bin,
                subject_sub_h5,
                subject_sub_dir,
                # "--inmem",
                "--merge_dipscore=0.5",
                "--doplot",
                f"--plotdir={clust_plotdir}",
                "--tmpdir=/local/duster",
            ]
        )
        print(subject, res.returncode)
        print(res)
    else:
        print(subject_sub_dir / "aligned_spike_index.npy", "exists, skipping", subject)
    



# %%
print(1)

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
#         "-t /local/hiddd "
#         "-c CONFIG_M "
#         f"-m {ks_chanmap} "
#         "-n 384 -s 0 -e Inf"
#     )

# %%
1

# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":
    if subject not in ("CSHL051", "DY_018"):
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
    
    aligned_spike_index = Path(sub_dir / subject / "aligned_spike_index.npy")
    if not aligned_spike_index.exists():
        print("no asi, skip")
        continue
    aligned_spike_index = np.load(aligned_spike_index)
    
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
# %rm -rf /local/deconv*

# %% tags=[]
for in_bin in in_bins:
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":

    if subject not in ("CSHL051", "DY_018"):
        continue
    print(subject)
    
    out_bin = out_dir / f"{in_bin.stem}.bin"
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_res_bin = next((sub_dir / subject).glob("res*.bin"))
    subject_deconv_dir = deconv_dir / subject
    subject_deconv_dir.mkdir(exist_ok=True, parents=True)
    
    deconv_scratch_dir = Path("/local/deconv")
    deconv_scratch_dir.mkdir(exist_ok=True)
    
    if not (subject_deconv_dir / "deconv_results.h5").exists():
        print("run extract")
        deconv_h5, deconv_residual_path = extract_deconv.extract_deconv(
            subject_deconv_dir / "templates_up.npy",
            subject_deconv_dir / "spike_train_up.npy",
            deconv_scratch_dir, # subject_deconv_dir,
            out_bin,
            subtraction_h5=subject_sub_h5,
            save_cleaned_waveforms=True,
            save_denoised_waveforms=True,
            n_channels_extract=20,
            n_jobs=13,
            # device="cpu",
            scratch_dir="/local/hibbb"
        )
        with timer("copying deconv h5 to output"):
            shutil.copy(deconv_h5, subject_deconv_dir / "deconv_results.h5")
            
    with h5py.File(subject_deconv_dir / "deconv_results.h5") as f:
        # for k in f:
        #     print(k, f[k].shape)
        maxptps = f["maxptps"][:]
        firstchans = f["first_channels"][:]
        x, y, z_rel, z_abs, alpha = f["localizations"][:].T
    
    spike_train_deconv = np.load(subject_deconv_dir / "spike_train.npy")
    labels_deconv = spike_train_deconv[:, 1]
    assert labels_deconv.shape == maxptps.shape
    templates = np.load(subject_deconv_dir / "templates.npy")
    assert templates.shape[0] == labels_deconv.max() + 1
    
    do_split = not (subject_deconv_dir / "postdeconv_split_labels.npy").exists()
    do_merge = True or not (subject_deconv_dir / "postdeconv_merge_labels.npy").exists()
    do_anything = do_split or do_merge
    
    if do_anything:
        if not (deconv_scratch_dir / "deconv_results.h5").exists():
            print("scratch deconv h5 does not exist")
            with timer("copying to scratch"):
                shutil.copy(subject_deconv_dir / "deconv_results.h5", deconv_scratch_dir / "deconv_results.h5")
        else:
            print("Scratch h5 already there")
        
        if not (deconv_scratch_dir / "raw.bin").exists():
            print("scratch bin does not exist")
            with timer("copying to scratch"):
                shutil.copy(out_bin, deconv_scratch_dir / "raw.bin")
        else:
            print("Scratch bin already there")
    
    if do_split:
        print("run split")
        split_labels = after_deconv_merge_split.split(
            labels_deconv,
            templates,
            firstchans,
            deconv_scratch_dir / "deconv_results.h5",
            wfs_key="denoised_waveforms",
        )
        spike_train_split = spike_train_deconv.copy().astype(np.int64)
        spike_train_split[:, 1] = split_labels
        print("A", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        after_deconv_merge_split.clean_big_clusters(
            templates, spike_train_split, deconv_scratch_dir / "raw.bin", geom
        )
        spike_train_split[:, 1] = cluster_utils.make_labels_contiguous(spike_train_split[:, 1])
        print("B", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        # split_templates = pre_deconv_merge_split.get_templates(
        #     deconv_scratch_dir / "raw.bin",
        #     geom,
        #     split_labels.max() + 1,
        #     spike_train_split,
        #     spike_train_split[:, 1],
        # )
        split_templates, snrs = snr_templates.get_templates(
            spike_train_split,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        
        
        print("first temps")
        maxchans = split_templates.ptp(1).argmax(1)
        mcts = split_templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        spike_train_split = spike_train_split.astype(np.int64)
        
        (
            template_shifts,
            template_maxchans,
            spike_train_split,
            idx_not_aligned,
        ) = pre_deconv_merge_split.align_spikes_by_templates(
            spike_train_split[:, 1], split_templates, spike_train_split, shift_max=3
        )
        print("C", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        # spike_index_dtype = np.dtype([('sample', np.int64), ('channel', np.int64)])
        spike_train_split = spike_train_split.astype(np.int64)
        # spike_train_split.view(spike_index_dtype).sort(order='sample')
        spike_train_split = spike_train_split[np.argsort(spike_train_split[:, 0])]
        print("D", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        print("shifts", template_shifts)
        
        templates, snrs = snr_templates.get_templates(
            spike_train_split,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        print("E", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        print("second temps / before remove oversplit")
        maxchans = templates.ptp(1).argmax(1)
        mcts = templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        
        spike_train_split, split_templates = after_deconv_merge_split.remove_oversplits(
            templates, spike_train_split
        )
        print("F", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        spike_train_split[:, 1] = cluster_utils.make_labels_contiguous(spike_train_split[:, 1])
        print("E", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        print("after remove oversplit")
        maxchans = split_templates.ptp(1).argmax(1)
        mcts = split_templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        
        split_times, split_labels = spike_train_split.T
        np.save(subject_deconv_dir / "postdeconv_split_labels.npy", spike_train_split[:, 1])
        np.save(subject_deconv_dir / "postdeconv_split_times.npy", spike_train_split[:, 0])
    else:
        split_labels = np.load(subject_deconv_dir / "postdeconv_split_labels.npy")
        split_times = np.load(subject_deconv_dir / "postdeconv_split_times.npy")
    
    if do_merge:
        print("run merge")
        # fill -- make split labels contiguous
        split_labels = cluster_utils.make_labels_contiguous(split_labels, in_place=True)
        spike_train_split = np.c_[split_times, split_labels]
        
        # get split templates
        # split_templates = deconvolve.get_templates(
        #     deconv_scratch_dir / "raw.bin",
        #     spike_train_deconv,          # asks for spike index but only uses first axis
        #     split_labels,
        #     geom,
        #     n_times=121,
        #     n_samples=250,
        #     trough_offset=42,
        # )
        
        split_templates, snrs = snr_templates.get_templates(
            spike_train_split,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        
        maxchans = split_templates.ptp(1).argmax(1)
        mcts = split_templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        
        print(split_labels.max(), np.unique(split_labels), np.unique(split_labels).size, split_templates.shape)
        

        merge_labels = after_deconv_merge_split.merge(
            split_labels,
            split_templates,
            deconv_scratch_dir / "deconv_results.h5",
            x,
            z_abs,
            maxptps,
            firstchans,
            geom,
            wfs_key="cleaned_waveforms",
        )

        spike_train_merge = spike_train_split.copy()
        spike_train_merge[:, 1] = merge_labels
        
        after_deconv_merge_split.clean_big_clusters(
            templates, spike_train_merge, deconv_scratch_dir / "raw.bin", geom
        )
        
        spike_train_merge[:, 1] = cluster_utils.make_labels_contiguous(spike_train_merge[:, 1])
        templates, snrs = snr_templates.get_templates(
            spike_train_merge,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        
        spike_train_merge, merge_templates = after_deconv_merge_split.remove_oversplits(
            templates, spike_train_merge
        )
        
        np.save(subject_deconv_dir / "postdeconv_merge_labels.npy", spike_train_merge[:, 1])
    
    if do_anything:
        print("scratch h5 exists?", (deconv_scratch_dir / "deconv_results.h5").exists())
        if (deconv_scratch_dir / "deconv_results.h5").exists():
            (deconv_scratch_dir / "deconv_results.h5").unlink()
        print("scratch bin exists?", (deconv_scratch_dir / "raw.bin").exists())
        if (deconv_scratch_dir / "raw.bin").exists():
            (deconv_scratch_dir / "raw.bin").unlink()


# %%
for i, in_bin in enumerate(tqdm(in_bins)):
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":
    if subject not in ("CSHL051", "DY_018"):
        continue
    
    out_bin = out_dir / f"{in_bin.stem}.bin"
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_deconv_dir = deconv_dir / subject
    second_deconv_dir = deconv_dir / subject / "deconv2"
    second_deconv_dir.mkdir(exist_ok=True)
    
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
    
    times = np.load(subject_deconv_dir / "postdeconv_split_times.npy")
    labels = np.load(subject_deconv_dir / "postdeconv_merge_labels.npy")
    which = (labels >= 0) & (times > 70) & (times < (ds - 79))

    template_spike_train = np.c_[times[which], labels[which]]
    print("Nunits", template_spike_train[:, 1].max() + 1)
    
    fname_templates_up,fname_spike_train_up,template_path,fname_spike_train = deconvolve.deconvolution(
        np.c_[times[which], np.full_like(times[which], -1)],
        labels[which],
        second_deconv_dir,
        out_bin,
        None,
        template_spike_train,
        subject_deconv_dir / "geom.npy",
        threshold=40,
        max_upsample=8,
        vis_su_threshold=1.0,
        approx_rank=5,
        multi_processing=True,
        cleaned_temps=True,
        n_processors=12,
        sampling_rate=30000,
        deconv_max_iters=1000,
        t_start=0,
        t_end=None,
        n_sec_chunk=1,
        verbose=False,
    )

# %%
for in_bin in in_bins:
    subject = in_bin.stem.split(".")[0]
    # if subject != "DY_018":
    # if subject_deconv_dir:
    #     del subject_deconv_dir

    if subject not in ("CSHL051", "DY_018"):
        continue
    print(subject)

    out_bin = out_dir / f"{in_bin.stem}.bin"
    subject_sub_h5 = next((sub_dir / subject).glob("sub*.h5"))
    subject_res_bin = next((sub_dir / subject).glob("res*.bin"))
    subj_dc2_dir = deconv_dir / subject / "deconv2"
    subj_dc2_dir.mkdir(exist_ok=True, parents=True)
    
    deconv_scratch_dir = Path("/local/deconv")
    deconv_scratch_dir.mkdir(exist_ok=True)
    
    do_extract = True or not (subj_dc2_dir / "deconv_results.h5").exists()
    if do_extract:
        print("run extract")
        deconv_h5, deconv_residual_path = extract_deconv.extract_deconv(
            subj_dc2_dir / "templates_up.npy",
            subj_dc2_dir / "spike_train_up.npy",
            deconv_scratch_dir, # subj_dc2_dir,
            out_bin,
            subtraction_h5=subject_sub_h5,
            save_cleaned_waveforms=True,
            save_denoised_waveforms=True,
            n_channels_extract=20,
            n_jobs=13,
            # device="cpu",
            scratch_dir=deconv_scratch_dir,
        )
        with timer("copying deconv h5 to output"):
            shutil.copy(deconv_h5, subj_dc2_dir / "deconv_results.h5")
            
    with h5py.File(subj_dc2_dir / "deconv_results.h5") as f:
        # for k in f:
        #     print(k, f[k].shape)
        maxptps = f["maxptps"][:]
        firstchans = f["first_channels"][:]
        x, y, z_rel, z_abs, alpha = f["localizations"][:].T
    
    spike_train_deconv = np.load(subj_dc2_dir / "spike_train.npy")
    labels_deconv = spike_train_deconv[:, 1]
    assert labels_deconv.shape == maxptps.shape
    templates = np.load(subj_dc2_dir / "templates.npy")
    assert templates.shape[0] == labels_deconv.max() + 1
    assert spike_train_deconv[:, 0].shape == firstchans.shape
    
    do_split = True or not (subj_dc2_dir / "postdeconv_split_labels.npy").exists()
    do_merge = True or not (subj_dc2_dir / "postdeconv_merge_labels.npy").exists()
    do_anything = do_split or do_merge
    
    if do_anything:
        if not (deconv_scratch_dir / "deconv_results.h5").exists():
            print("scratch deconv h5 does not exist")
            with timer("copying to scratch"):
                shutil.copy(subj_dc2_dir / "deconv_results.h5", deconv_scratch_dir / "deconv_results.h5")
        else:
            print("Scratch h5 already there")
        
        if not (deconv_scratch_dir / "raw.bin").exists():
            print("scratch bin does not exist")
            with timer("copying to scratch"):
                shutil.copy(out_bin, deconv_scratch_dir / "raw.bin")
        else:
            print("Scratch bin already there")
    
    if do_split:
        print("run split")
        split_labels = after_deconv_merge_split.split(
            labels_deconv,
            templates,
            firstchans,
            deconv_scratch_dir / "deconv_results.h5",
            wfs_key="denoised_waveforms",
        )
        spike_train_split = spike_train_deconv.copy().astype(np.int64)
        spike_train_split[:, 1] = split_labels
        print("A", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        after_deconv_merge_split.clean_big_clusters(
            templates, spike_train_split, deconv_scratch_dir / "raw.bin", geom
        )
        spike_train_split[:, 1] = cluster_utils.make_labels_contiguous(spike_train_split[:, 1])
        print("B", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        # split_templates = pre_deconv_merge_split.get_templates(
        #     deconv_scratch_dir / "raw.bin",
        #     geom,
        #     split_labels.max() + 1,
        #     spike_train_split,
        #     spike_train_split[:, 1],
        # )
        split_templates, snrs = snr_templates.get_templates(
            spike_train_split,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        
        
        print("first temps")
        maxchans = split_templates.ptp(1).argmax(1)
        mcts = split_templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        spike_train_split = spike_train_split.astype(np.int64)
        
        (
            template_shifts,
            template_maxchans,
            spike_train_split,
            idx_not_aligned,
        ) = pre_deconv_merge_split.align_spikes_by_templates(
            spike_train_split[:, 1], split_templates, spike_train_split, shift_max=3
        )
        print("C", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        # spike_index_dtype = np.dtype([('sample', np.int64), ('channel', np.int64)])
        spike_train_split = spike_train_split.astype(np.int64)
        # spike_train_split.view(spike_index_dtype).sort(order='sample')
        spike_train_split = spike_train_split[np.argsort(spike_train_split[:, 0])]
        print("D", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        print("shifts", template_shifts)
        
        templates, snrs = snr_templates.get_templates(
            spike_train_split,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        print("E", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        print("second temps / before remove oversplit")
        maxchans = templates.ptp(1).argmax(1)
        mcts = templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        
        spike_train_split, split_templates = after_deconv_merge_split.remove_oversplits(
            templates, spike_train_split
        )
        print("F", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        spike_train_split[:, 1] = cluster_utils.make_labels_contiguous(spike_train_split[:, 1])
        print("E", spike_train_split[:20], spike_train_split[-20:])
        print((spike_train_split[1:,0] >= spike_train_split[:-1,0]).all())
        
        print("after remove oversplit")
        maxchans = split_templates.ptp(1).argmax(1)
        mcts = split_templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        
        split_times, split_labels = spike_train_split.T
        np.save(subj_dc2_dir / "postdeconv_split_labels.npy", spike_train_split[:, 1])
        np.save(subj_dc2_dir / "postdeconv_split_times.npy", spike_train_split[:, 0])
    else:
        split_labels = np.load(subj_dc2_dir / "postdeconv_split_labels.npy")
        split_times = np.load(subj_dc2_dir / "postdeconv_split_times.npy")
    
    if do_merge:
        print("run merge")
        # fill -- make split labels contiguous
        split_labels = cluster_utils.make_labels_contiguous(split_labels, in_place=True)
        spike_train_split = np.c_[split_times, split_labels]
        
        # get split templates
        # split_templates = deconvolve.get_templates(
        #     deconv_scratch_dir / "raw.bin",
        #     spike_train_deconv,          # asks for spike index but only uses first axis
        #     split_labels,
        #     geom,
        #     n_times=121,
        #     n_samples=250,
        #     trough_offset=42,
        # )
        
        split_templates, snrs = snr_templates.get_templates(
            spike_train_split,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        
        maxchans = split_templates.ptp(1).argmax(1)
        mcts = split_templates[np.arange(len(maxchans)), :, maxchans]
        print(mcts.argmin(1))
        
        print(split_labels.max(), np.unique(split_labels), np.unique(split_labels).size, split_templates.shape)

        merge_labels = after_deconv_merge_split.merge(
            split_labels,
            split_templates,
            deconv_scratch_dir / "deconv_results.h5",
            x,
            z_abs,
            maxptps,
            firstchans,
            geom,
            wfs_key="cleaned_waveforms",
        )

        spike_train_merge = spike_train_split.copy()
        spike_train_merge[:, 1] = merge_labels
        
        after_deconv_merge_split.clean_big_clusters(
            templates, spike_train_merge, deconv_scratch_dir / "raw.bin", geom
        )
        
        spike_train_merge[:, 1] = cluster_utils.make_labels_contiguous(spike_train_merge[:, 1])
        templates, snrs = snr_templates.get_templates(
            spike_train_merge,
            geom,
            deconv_scratch_dir / "raw.bin",
            max_spikes_per_unit=250,
        )
        
        spike_train_merge, merge_templates = after_deconv_merge_split.remove_oversplits(
            templates, spike_train_merge
        )
        
        np.save(subj_dc2_dir / "postdeconv_merge_labels.npy", spike_train_merge[:, 1])
    
    if do_anything:
        print("scratch h5 exists?", (deconv_scratch_dir / "deconv_results.h5").exists())
        if (deconv_scratch_dir / "deconv_results.h5").exists():
            (deconv_scratch_dir / "deconv_results.h5").unlink()
        print("scratch bin exists?", (deconv_scratch_dir / "raw.bin").exists())
        if (deconv_scratch_dir / "raw.bin").exists():
            (deconv_scratch_dir / "raw.bin").unlink()

# %%

# %%

# %%

# %%
split_times

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
# 
