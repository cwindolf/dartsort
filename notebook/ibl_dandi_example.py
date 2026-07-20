# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python [conda env:dart14]
#     language: python
#     name: conda-env-dart14-py
# ---

# %%
import matplotlib.pyplot as plt

# %%
# dandi-related
from dandi.dandiapi import DandiAPIClient
import fsspec

# %%
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
# ## Run *dartsort*

# %%
ds_dir = experiment_path / "dartsort"
ds_res = dartsort.dartsort(rec, ds_dir)

# %% [markdown]
# # Visualize

# %%
dartvis.scatter_spike_features(sorting=ds_res["sorting"]);

# %%
