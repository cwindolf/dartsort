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
import torch
from torch import nn
import h5py
from torch.utils.tensorboard import SummaryWriter
import time

# %%
from spike_psvae.psvae import PSVAE
from spike_psvae.data_utils import SpikeHDF5Dataset

# %%
input_h5 = "/mnt/3TB/charlie/features/wfs_locs.h5"
with h5py.File(input_h5, "r") as f:
    for k in f.keys():
        print(k.ljust(20), f[k].dtype, f[k].shape)

# %%
torch.cuda.is_available()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
in_chan = 20
in_w = 121
in_dim = in_w * in_chan
in_dim

# %%
hidden_dim = 512

# %%
# load up data
y_keys = ["alpha", "x", "y", "z_rel"] # , "spread"]
dataset = SpikeHDF5Dataset(input_h5, "denoised_waveforms", y_keys)

# %%
len(dataset)

# %%
n_sup_latents = len(y_keys)
n_unsup_latents = 10
n_latents = n_sup_latents + n_unsup_latents

# %%
# vanilla encoders and decoders
vanilla_enc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.LeakyReLU(),
    nn.Linear(hidden_dim, n_latents),
    nn.BatchNorm1d(n_latents),
    nn.LeakyReLU(),
)
vanilla_dec = nn.Sequential(
    nn.Linear(n_latents, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.LeakyReLU(),
    nn.Linear(hidden_dim, in_dim),
    # no output activation for now. data range is not standard
    # and I'm not sure how best to preprocess.
    # probably letting PCA do it is the answer?
    # nn.Sigmoid(),
    nn.Unflatten(1, (in_w, in_chan)),
)

# %%
psvae = PSVAE(vanilla_enc, vanilla_dec, n_sup_latents, n_unsup_latents)
optimizer = torch.optim.Adam(psvae.parameters(), lr=1e-3)

# %%
batch_size = 8
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
writer = SummaryWriter(
    log_dir="/mnt/3TB/charlie/features/runs/morestats",
    # comment="initial",
)
# _, (x, y) = next(enumerate(loader))
# writer.add_graph(psvae, x)

# %%
psvae.to(device)

# %%
global_step = 0
n_epochs = 50
for e in range(n_epochs):
    tic = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        # print(x.shape, y.shape)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        recon_x, y_hat, mu, logvar = psvae(x)
        loss, loss_dict = psvae.loss(x, y, recon_x, y_hat, mu, logvar)

        loss.backward()
        optimizer.step()

        if not batch_idx % 1000:
            print(e, batch_idx, loss.item(), flush=True)


            # -- Losses
            writer.add_scalar("Loss/loss", loss.cpu(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/{k}", v.cpu(), global_step)
                
            # -- Images
            x_ = x.cpu()
            recon_x_ = recon_x.cpu()
            im = torch.hstack((x_, recon_x_, x_ - recon_x_))
            im = im - im.min()
            im *= 255. / im.max()
            writer.add_images(
                "x,recon_x,residual",
                im.to(torch.uint8).view(*im.shape, 1),
                global_step,
                dataformats="NHWC",
            )
            
            # -- Stats
            y_hat_ = y_hat.cpu()
            y_ = y.cpu()
            y_mses = (y_hat_ - y_).pow(2).mean(axis=0)
            for y_key, y_mse in zip(y_keys, y_mses):
                writer.add_scalar(f"Stat/{y_key}_mse", y_mse, global_step)
            
            if np.isnan(loss.item()):
                break
        
        global_step += 1
    print("epoch", e, "took", (time.time() - tic) / 60, "min")

# %%
print("done")

# %%
