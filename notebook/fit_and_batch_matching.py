# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python [conda env:a]
#     language: python
#     name: conda-env-a-py
# ---

# %% [markdown]
# This example notebook shows how to break dartsort into stages:
#  - Fitting the template matching model
#  - Running template matching
#  - Gathering outputs
#
# This pipeline is intended for users with very long recordings who want more control over the matching step.
#
# Since the matching step is the only one which passes over the whole recording, long recordings can benefit
# from additional parallelism there beyond *dartsort*'s built-in single- or multi-GPU parallelism.
# In this notebook, the matching step is run "chunked", but just in a for loop without parallelism.
# This keeps it simple here, but shows all of the boilerplate for chunking and gathering results.
# Users should swap out the for loop with something for their platform (SLURM script, Dask situation, ...).
#
#
# This is a translation of the [ibl_dandi_example](ibl_dandi_example.py) notebook, which works off of a DANDI
# recording.

# %%
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
from dandi.dandiapi import DandiAPIClient

import dartsort
import dartsort.vis as dartvis

dartvis.set_plt_style()

# %% [markdown]
# ### Paths / config

# %%
# update this for your machine!
experiment_path = dartsort.ensure_path("~/scratch/ibl-dandi-example/", mkdir=True)

# %%
ds_dir = experiment_path / "dartsort-chunked"

# %%
# IBL BWM dandiset, example recording
dandiset_id = "000409"
asset_path = "sub-CSH-ZAD-026/sub-CSH-ZAD-026_ses-15763234-d21e-491f-a01b-1238eb96d389_desc-raw_ecephys.nwb"

# %% [markdown]
# ### Preprocess and save a 10 minute snippet of data

# %% [markdown]
# Pre-saving the preprocessing like this is optional. You can also set the
# preprocessing flag of the `DARTsortUserConfig()` to the same strategy (`"ibllike"`)
# or one of the others. In that case, you can also set `copy_recording_to_tmpdir` to
# reduce preprocessing overhead. Here, I'm pre-saving it just to visualize and in
# case the user wants to experiment with the cached preprocessed recording.

# %%
preprocessing_path = experiment_path / "preprocessed_recording"
if preprocessing_path.exists():
    rec = si.read_binary_folder(preprocessing_path)
else:
    si.set_global_job_kwargs(n_jobs=8, pool_engine="thread")

    # stream from DANDI with SpikeInterface
    # start by finding the s3 asset URL
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(asset_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    rec0 = si.read_nwb_recording(
        s3_url,
        stream_mode="fsspec",
        electrical_series_path="acquisition/ElectricalSeriesProbe00AP",
    )
    print("Original recording:")
    print(rec0)

    # slice 10 minutes near the end
    # TODO better IBL sampling rate handling
    rec = rec0
    end_time = rec.get_end_time()
    start_frame = rec.time_to_sample_index(end_time - 30 * 60)
    end_frame = start_frame + 10 * 60 * 30_000
    rec = rec.frame_slice(start_frame, end_frame)
    rec.reset_times()
    rec._sampling_frequency = 30_000

    rec = dartsort.preprocess(rec, strategy="ibllike")
    rec = rec.save(folder=preprocessing_path, format="binary")

# %%
plt.imshow(
    rec.get_traces(0, 10_000, 11_000).T,
    aspect="auto",
    vmin=-5,
    vmax=5,
    interpolation="nearest",
)
plt.colorbar(shrink=0.5, label="standardized voltage")
plt.xlabel("time (samples")
plt.ylabel("channels")
# %% [markdown]
# # Configuration

# %%
# use defaults, except for the pipeline change
cfg = dartsort.DARTsortUserConfig()

# %%
# process into an internal config object for later use
internal_cfg = dartsort.to_internal_config(cfg)

# %% [markdown]
# # Initial spike detection and motion tracking model fits

# %%
dartsort.dartsort(rec, ds_dir, cfg=dataclasses.replace(cfg, fit_motion_models_only=True))

# %% [markdown]
# # Chunked spike detection for motion tracking
#
# While dartsort itself only needs to detect spikes on a subsample of the recording
# in order to do clustering, motion tracking does require spikes from the whole
# recording. dartsort uses a cheap but good spike detection and localization pass for
# this, which is very fast, but users at the thousand hour scale still might want to
# parallelize it.
#
# This runs that cheap spike detection for motion tracking in 1-minute batches for demo
# purposes.
#
# Here, the `chunk_starts_samples` argument slices the recording for us.
# This is better in this case than using `.frame_slice` or `.time_slice` on the recording,
# since this way the sample indices (spike times) are relative to the full recording without extra bookkeeping.
#
# If the initial detection already covered the whole recording, motion tracking uses its
# spikes and this step is skipped.

# %%
subtraction_sorting = dartsort.load(ds_dir / "subtraction.h5")
need_motion_peaks = cfg.do_motion_estimation and dartsort.motion_needs_peaks(
    internal_cfg, rec, subtraction_sorting
)
need_motion_peaks

# %%
chunk_length_samples = internal_cfg.motion_estimation_cfg.threshold_cfg.chunk_length_samples
batch_length_samples = 60 * chunk_length_samples
duration_samples = rec.get_num_samples()

if need_motion_peaks:
    for batch_ix, batch_start_samples in enumerate(range(0, duration_samples, batch_length_samples)):
        batch_end_samples = min(duration_samples, batch_start_samples + batch_length_samples)
        res = dartsort.detect_for_motion(
            recording=rec,
            output_dir=ds_dir,
            previous_sorting=subtraction_sorting,
            hdf5_filename=f"motionthreshold_batch{batch_ix:05}.h5",
            chunk_starts_samples=np.arange(batch_start_samples, batch_end_samples, chunk_length_samples),
            motion_cfg=internal_cfg.motion_estimation_cfg,
            computation_cfg=internal_cfg.motion_computation_cfg,
            sampling_cfg=internal_cfg.peeler_sampling_cfg,
            waveform_cfg=internal_cfg.waveform_cfg,
        )
        del res

# %% [markdown]
# Combine the output of those detections into one sorting, and check that together the
# batches covered the recording.

# %%
if need_motion_peaks and not (ds_dir / "motionthreshold.h5").exists():
    motion_chunk_h5s = sorted(ds_dir.glob("motionthreshold_batch*.h5"))
    dartsort.virtual_concatenate_sorting_h5s(motion_chunk_h5s, ds_dir / "motionthreshold.h5")
    motion_sorting = dartsort.load(ds_dir / "motionthreshold.h5")
    assert np.array_equal(
        motion_sorting.load_dataset("chunk_starts_samples"),
        np.arange(0, duration_samples, chunk_length_samples),
    )
    print(motion_sorting)

# %%
# !ls -lah {ds_dir}

# %% [markdown]
# ^ Notice all of the chunked motion files! But motionthreshold.h5 itself is tiny, because the concatenation is virtual.

# %% [markdown]
# # Resume: estimate motion, cluster, learn templates and the matching step classifier

# %%
dartsort.dartsort(rec, ds_dir, cfg=dataclasses.replace(cfg, fit_matching_models_only=True))

# %%
# !ls {ds_dir}

# %%
# !ls {ds_dir}/matching1_models

# %% [markdown]
# ^ This was our goal: get a motion estimate, template data, and the featurizers and classifiers that template matching needs.

# %% [markdown]
# # Chunked template matching
#
# This runs template matching in 1-minute batches, with the same loop over chunks
# as the motion tracking's spike detection above.
#

# %%
duration_samples = rec.get_num_samples()
for batch_ix, batch_start_samples in enumerate(range(0, duration_samples, 60 * 30_000)):
    batch_end_samples = min(duration_samples, batch_start_samples + 60 * 30_000)
    res = dartsort.match(
        recording=rec,
        output_dir=ds_dir,
        model_subdir="matching1_models",
        hdf5_filename=f"matching1_batch{batch_ix:05}.h5",
        chunk_starts_samples=np.arange(batch_start_samples, batch_end_samples, 30_000),
        load_simple_features=False,  # ignoring the output, so don't load up extra stuff
        featurization_cfg=internal_cfg.matching_classifier_featurization_cfg,
        motion=dartsort.try_load_motion_info(ds_dir),
    )
    del res

# %% [markdown]
# # Gather results and postprocess
#
# In this section, I'm grabbing all of the matching outputs and concatenating them.
# Then, I'm running *dartsort*'s standard postprocessing: a final merge step and a spike-train
# cleaning step (duplicate detection removal). This is controlled by the `DARTsortUserConfig` flag `postprocessing`,
# which is `"agglomerate_and_clean"` by default.
#
# Users can change this. If you don't want to merge, set `postprocessing="clean"` in the configuration
# section above. In that case, you could run the cleaning step on each chunk before concatenating: the
# merge needs to agree across the batched matching output, but the output of cleaning is the same when
# run batchwise or on the full result.
#
# Here, extra features (amplitudes, localizations) are loaded by the default behavior of `dartsort.load`,
# which can be turned off for users who don't need them.

# %%
sorting = dartsort.concatenate_sortings(
    [dartsort.load(h5) for h5 in sorted(ds_dir.glob("matching1_batch*.h5"))]
)

# %%
sorting = dartsort.cluster(
    rec,
    sorting,
    motion=dartsort.try_load_motion_info(ds_dir),
    clustering_cfg=None,
    refinement_cfgs=internal_cfg.final_refinement_cfgs,
)

# %%
# if you want to keep this...
sorting.save(ds_dir / "dartsort_sorting.npz")

# %% [markdown]
# # Visualize

# %%
dartvis.scatter_spike_features(sorting=sorting);

# %%
