# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python [conda env:dart14]
#     language: python
#     name: conda-env-dart14-py
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
import matplotlib.pyplot as plt

# %%
# dandi-related
from dandi.dandiapi import DandiAPIClient
import fsspec

# %%
import numpy as np
import spikeinterface.full as si
import dartsort
import dartsort.vis as dartvis
dartvis.set_plt_style()

# %% [markdown]
# ### Paths / config

# %%
# update this for your machine!
experiment_path = dartsort.ensure_path(
    "~/scratch/ibl-dandi-example/", mkdir=True
)

# %%
ds_dir = experiment_path / "dartsort"

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
    #TODO better IBL sampling rate handling
    rec = rec0
    end_time = rec.get_end_time()
    start_frame = rec.time_to_sample_index(end_time - 30 * 60)
    end_frame = start_frame + 10 * 60 * 30_000
    rec = rec.frame_slice(start_frame, end_frame)
    rec.reset_times()
    rec._sampling_frequency = 30_000

    rec = dartsort.preprocess(rec, strategy="ibllike")
    rec = rec.save_to_folder(preprocessing_path)

# %%
plt.imshow(
    rec.get_traces(0, 10_000, 11_000).T,
    aspect='auto',
    vmin=-5,
    vmax=5,
    interpolation='nearest',
)
plt.colorbar(shrink=0.5, label='standardized voltage')
plt.xlabel('time (samples')
plt.ylabel('channels');

# %% [markdown]
# # Fit models

# %%
dartsort.dartsort(rec, ds_dir, cfg=dartsort.DARTsortUserConfig(fit_matching_models_only=True))

# %%
# !ls {ds_dir}

# %%
# !ls {ds_dir}/matching1_models

# %% [markdown]
# ^ This was our goal: get a motion estimate, template data, and the featurizers and classifiers that template matching needs.

# %% [markdown]
# # Chunked template matching
#
# This runs template matching in 1-minute batches.
#
# Here, `chunk_starts_samples` slices the recording for us.
# This might be better than using `.frame_slice` or `.time_slice` on the recording,
# since this way the sample indices (spike times) are relative to the full recording without extra bookkeeping.
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
        featurization_cfg=dartsort.default_matching_streaming_classifier_cfg,
        motion=dartsort.try_load_motion_info(ds_dir),
    )
    del res

# %% [markdown]
# # Gather results and postprocess
#
# In this section, I'm grabbing all of the matching outputs and concatenating them.
# Then, I'm running *dartsort*'s standard postprocessing: a final merge step and a spike-train
# cleaning step (duplicate detection removal).
#
# Users can change this. If you don't want to merge, remove that config from the list. In that case, you could
# run the cleaning step on each chunk before concatenating: the merge needs to agree across the batched matching
# output, but the output of cleaning is the same when run batchwise or on the full result.
#
# Here, extra features (amplitudes, localizations) are loaded by the default behavior of `dartsort.load`,
# which can be turned off for users who don't need them.

# %%
sorting = dartsort.concatenate_sortings([dartsort.load(h5) for h5 in sorted(ds_dir.glob("matching1_batch*.h5"))])

# %%
sorting = dartsort.cluster(
    rec,
    sorting,
    motion=dartsort.try_load_motion_info(ds_dir),
    clustering_cfg=None,
    refinement_cfgs=[dartsort.default_agglomerate_cfg, dartsort.default_clean_cfg],
)

# %%
# if you want to keep this...
sorting.save(ds_dir / 'dartsort_sorting.npz')

# %% [markdown]
# # Visualize

# %%
dartvis.scatter_spike_features(sorting=sorting);

# %%
