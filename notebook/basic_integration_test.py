# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:a]
#     language: python
#     name: conda-env-a-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from spike_psvae import subtract
from one.api import ONE
from pathlib import Path

# %%
import spikeinterface.full as si

# %%
one = ONE()

# %%
sdsc_base_path = Path("/mnt/sdceph/users/ibl/data")


# %%
def eid2sdscpath(eid):
    pids, probes = one.eid2pid(eid)
    alyx_base_path = one.eid2path(eid)
    paths = {}
    for pid, probe in zip(pids, probes):
        rel_path = one.list_datasets(eid, f"raw_ephys_data/{probe}*ap.cbin")
        assert len(rel_path) == 1
        rel_path = Path(rel_path[0])
        searchdir = sdsc_base_path / alyx_base_path.relative_to(one.cache_dir) / rel_path.parent
        pattern = Path(rel_path.name).with_suffix(f".*.cbin")
        glob = list(searchdir.glob(str(pattern)))
        assert len(glob) == 1
        paths[probe] = pid, glob[0]
        assert paths[probe][1].exists()
    return paths


# %%
pid = "7d999a68-0215-4e45-8e6c-879c6ca2b771"

# %%
eid, probe = one.pid2eid(pid)
paths = eid2sdscpath(eid)
print(paths)
pid, cbin_path = paths[probe]
ch_path = list(cbin_path.parent.glob("*ap*.ch"))
assert len(ch_path) == 1
ch_path = ch_path[0]
meta_path = list(cbin_path.parent.glob("*ap*.meta"))
assert len(meta_path) == 1
meta_path = meta_path[0]

print("-" * 50)
rec = si.read_cbin_ibl(str(cbin_path.parent), cbin_file=str(cbin_path), ch_file=str(ch_path), meta_file=str(meta_path))
print(rec)
fs = int(rec.get_sampling_frequency())

# %%
rec = si.highpass_filter(rec)
rec = si.phase_shift(rec)
bad_channel_ids, channel_labels = si.detect_bad_channels(rec, num_random_chunks=100)
print(f"{bad_channel_ids=}")
rec = si.interpolate_bad_channels(rec, bad_channel_ids)
rec = si.highpass_spatial_filter(rec)
# we had been working with this before -- should switch to MAD,
# but we need to rethink the thresholds
rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100, margin_frames=250*fs)

# %%
rec = rec.frame_slice(start_frame=rec.get_num_samples() // 2, end_frame=rec.get_num_samples() // 2 + 30 * 30000)

rec

rec_chansliced = rec.channel_slice(channel_ids=rec.channel_ids[[11, 13, 21, 65, 4, 2]])

rec_chansliced

geom = rec_chansliced.get_channel_locations()
geom

rec_ord = si.depth_order(rec_chansliced)
rec_ord

geom = rec_ord.get_channel_locations()
geom



# !rm -rf /tmp/testfolder

rec_ord.("/tmp/testfolder", n_jobs=1)

# !ls /tmp/testfolder

xcoords = geom[:, 0]
ycoords = geom[:, 1]
kcoords = np.zeros_like(geom[:, 0])
chanMap = np.arange(len(geom))

chanMap

from scipy.io import savemat

savemat("/tmp/chanMap.mat", dict(xcoords=xcoords, ycoords=ycoords, cm))

# %%
sub_h5 = subtract.subtraction(
    rec,
    out_folder="/tmp/test",
    thresholds=[12, 10, 8, 6, 5],
    n_sec_pca=10,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    n_jobs=1,
    loc_workers=1,
    overwrite=False,
    n_sec_chunk=1,
    device="cuda:1",
)

# %%
