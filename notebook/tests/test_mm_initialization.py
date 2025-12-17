# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python [conda env:dartsort]
#     language: python
#     name: conda-env-dartsort-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dartsort.util.testing_util import mixture_testing_util
from dartsort.cluster.gmm import mixture
import dartsort

# %%
mix_data = mixture_testing_util.simulate_moppca(
    K=1,
    Nper=4096 * 10,
    t_cov="eye",
    t_w="smooth",
    # t_missing=None,
    # t_missing="random_no_extrap",
    t_missing="random",
    n_missing=2,
    M=0,
)
mix_data.keys()

# %%
M = mix_data['M']
mu = mix_data['mu'][0]
W = mix_data['W'][0].permute(2, 0, 1)

# %%
neighb_cov, erp, train_data, val_data, full_data, noise = mixture.get_truncated_datasets(
    sorting=mix_data['init_sorting'],
    motion_est=None,
    refinement_cfg=dartsort.RefinementConfig(
        # interpolation_method='kriging',
        # kernel_name='thinplate',
        # extrapolation_method='nearest',
        # extrapolation_kernel='nerest',
        extrapolation_method='kriging',
        extrapolation_kernel='thinplate',
        interpolation_sigma=5.0,
        kriging_poly_degree=0,
        smoothing_lambda=0.01,
    ),
    rg=0,
    noise=mix_data["noise"],
    stable_data=mix_data["data"],
    device=None,
)

# %%
chans = neighb_cov.obs_ix[full_data.neighborhood_ids]
x0 = torch.full((*mix_data['x'].shape[:2], mix_data['n_channels']), torch.nan)
x0.scatter_(dim=2, index=chans[:, None, :].broadcast_to(mix_data['x'].shape), src=mix_data['x']);

# %%
x0.shape, chans.shape, mix_data['x'].shape

# %%
x = erp.interp_to_chans(
    full_data.x.view(mix_data['x'].shape),
    neighborhood_ids=full_data.neighborhood_ids,
    target_channels=torch.arange(mix_data['n_channels']),
)

# %%
fig, (aa, ab, ac) = plt.subplots(ncols=3, layout='constrained', figsize=(12, 6))

nm = 0
nn = 50
im = aa.imshow(x[nm:nn].reshape(nn, -1), aspect='auto', interpolation="none")
plt.colorbar(im, ax=aa, shrink=0.2)
im = ab.imshow(x0[nm:nn].reshape(nn, -1), aspect='auto', interpolation="none")
plt.colorbar(im, ax=ab, shrink=0.2);
im = ac.imshow(mix_data['x'][nm:nn].reshape(nn, -1), aspect='auto', interpolation="none")
plt.colorbar(im, ax=ac, shrink=0.2);

for ax in (aa, ab, ac):
    for j in range(1, mix_data['x'].shape[1]):
        ax.axvline(j * x.shape[2] - 0.5, c='r', lw=1.0)

# %%
mean, basis = mixture._initialize_single(
    x=x,
    chans=torch.arange(mix_data['n_channels']),
    noise=mix_data['noise'],
    rank=mix_data['M'],
    prior_pseudocount=0.0,
    # mean=mu,
)

# %%
fig, (aa, ab, ac, ad) = plt.subplots(ncols=4, figsize=(8, 3), layout='constrained')

vm = mu.abs().max()

aa.imshow(mu, vmin=-vm, vmax=vm, cmap='PRGn')
im = ab.imshow(mean, vmin=-vm, vmax=vm, cmap='PRGn')
plt.colorbar(im, ax=ab, shrink=0.1)
vdm = 1.0
im = ac.imshow(mu - mean, vmin=-vdm, vmax=vdm, cmap='seismic')
plt.colorbar(im, ax=ac, shrink=0.1)

ad.hist((mu - mean).view(-1), bins=32, histtype='step');
ad.hist((mu.abs() - mean.abs()).view(-1), bins=32, histtype='step');

# %%
fig, (aa, ab, ac, ad) = plt.subplots(ncols=4, layout='constrained', figsize=(8, 3))

shp = (M, -1)
WTW_0 = W.view(shp).T @ W.view(shp)
WTW_1 = basis.view(shp).T @ basis.view(shp)
vm = WTW_0.abs().max()

aa.imshow(WTW_0, vmin=-vm, vmax=vm, aspect='auto', interpolation='none')
im = ab.imshow(WTW_1, vmin=-vm, vmax=vm, aspect='auto', interpolation='none')
plt.colorbar(im, ax=ab, shrink=0.1)
im = ac.imshow(WTW_0 - WTW_1, vmin=-0.5, vmax=0.5, aspect='auto', interpolation='none')
plt.colorbar(im, ax=ac, shrink=0.1)

ad.hist((WTW_0 - WTW_1).view(-1), bins=32, histtype='step');
ad.hist((WTW_0.abs() - WTW_1.abs()).view(-1), bins=32, histtype='step');

# %%
