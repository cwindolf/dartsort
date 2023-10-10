# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from pathlib import Path
from spike_psvae.deconvolve import MatchPursuitObjectiveUpsample

# %%
templates_cleaned = np.load("/mnt/3TB/charlie/re_res_5min/CSH_ZAD_026_snip/deconv1/templates.npy")

# %%
geom = np.load("/mnt/3TB/charlie/re_res_5min/CSH_ZAD_026_snip/deconv1/geom.npy")

# %%
templates_cleaned.shape, templates_cleaned.dtype

# %%
deconv_dir = Path("./tmp")
deconv_dir.mkdir(exist_ok=True)
deconv_dir

# %%
tptp = templates_cleaned.ptp(1)
tptp.shape

# %%
tc = templates_cleaned[(tptp[:, None, :] * np.ones(121)[None, :, None]) > 1]

# %%
tc.shape

# %%
# Find deconv threshold
min_th = 50
for i in range(templates_cleaned.shape[0]):
    chans = templates_cleaned[i].ptp(0) >=1
    templates_cleaned[i, :, ~chans] = 0
    l2norm = 0.9*(templates_cleaned[i, :, chans]**2).sum()
    if l2norm<= min_th and l2norm>0:
        min_th = l2norm

# %%
min_th

# %%
from spike_psvae import localize_index

# %%
x, y, z_rel, z_abs, alpha = localize_index.localize_ptps_index(
    templates_cleaned.ptp(1),
    geom,
    templates_cleaned.ptp(1).argmax(1),
    np.arange(384)[None, :] * np.ones(384, dtype=int)[:, None],
    n_channels=20,
    radius=None,
    n_workers=None,
    pbar=True,
    logbarrier=True,
)

# %%
import matplotlib.pyplot as plt
plt.scatter(x, z_abs)

# %%
from spike_psvae.pre_deconv_merge_split import get_proposed_pairs, get_x_z_templates
dist_argsort, dist_template = get_proposed_pairs(
    templates_cleaned.shape[0],
    templates_cleaned,
    np.c_[x, z_abs],
    n_temp = 20,
) 

# %%
plt.imshow(dist_template, aspect="auto")

# %%
dist_template[423]

# %%
dist_argsort

# %%
dist_argsort[423]

# %%
tmc = templates_cleaned.ptp(1).argmax(1)

# %%
tmc[[423,424]]

# %%
close_templates = templates_cleaned[dist_argsort[423]]
close_templates.shape

# %%
from spike_psvae import cluster_viz_index

# %%

# %%
plt.plot(close_templates[np.arange(close_templates.shape[0]), :, close_templates.ptp(1).argmax(1)].T);

# %%
close_templates.min(), close_templates.max()

# %%
temp_concatenated = np.zeros((30000, templates_cleaned.shape[2]), dtype=np.float32)
temp_concatenated[15000:templates_cleaned.shape[1]+15000] = templates_cleaned[i]
print(f"{temp_concatenated.shape=} {temp_concatenated.dtype=} {temp_concatenated.min()=} {temp_concatenated.max()=}")

# Create bin file with templates
f_out =  deconv_dir / "template_bin.bin"
with open(f_out, 'wb') as f:
    temp_concatenated.tofile(f)

# %%
deconv_dir / "template_bin.bin"

# %%
f_out

# %%
# %ll {deconv_dir}

# %%
# Loop over templates to find residuals 
# %rm -rf {deconv_dir}/*

f_out =  deconv_dir / "template_bin.bin"
with open(f_out, 'wb') as f:
    temp_concatenated.tofile(f)

max_upsample = 8
trough_offset = 42

max_values = []
units = []
units_matched = []
for i in [423]:
    print("STARTING UNIT " + str(i))

    # templates_cleaned_amputated = templates_cleaned[dist_argsort[i][:10]]
    templates_cleaned_amputated = np.delete(templates_cleaned, i, axis=0)
    # print(f"{templates_cleaned_amputated.shape=} {templates_cleaned_amputated.dtype=}")
#     templates_cleaned_amputated = templates_cleaned_amputated[400:440]
    close_templates = templates_cleaned[dist_argsort[423][:5]]
    close_templates.shape

    mp_object = MatchPursuitObjectiveUpsample(
        templates=close_templates,
        # templates=templates_cleaned_amputated,
        deconv_dir=deconv_dir,
        standardized_bin=f_out,
        t_start=0,
        t_end=None,
        n_sec_chunk=1,
        sampling_rate=30000,
        max_iter=1,
        upsample=8,
        threshold=min_th,
        conv_approx_rank=5,
        n_processors=6,
        multi_processing=False,
        verbose=False,
        lambd=0, #0.001,
        allowed_scale=0.1
    )
    mp_object.load_saved_state()

    fnames_out = []
    batch_ids = []
    for batch_id in range(mp_object.n_batches):
        fname_temp = deconv_dir /f"seg_{str(batch_id).zfill(6)}_deconv.npz"
        fnames_out.append(fname_temp)
        batch_ids.append(batch_id)

    mp_object.run(batch_ids, fnames_out) 
    
    deconv_st = []
    deconv_scalings = []
    print("gathering deconvolution results")
    for batch_id in range(mp_object.n_batches):
        fname_out = deconv_dir / "seg_{}_deconv.npz".format(str(batch_id).zfill(6))
        with np.load(fname_out) as d:
            deconv_st.append(d["spike_train"])
            deconv_scalings.append(d["scalings"])
    deconv_st = np.concatenate(deconv_st, axis=0)
    deconv_scalings = np.concatenate(deconv_scalings, axis=0)

# %%
deconv_st

# %%
