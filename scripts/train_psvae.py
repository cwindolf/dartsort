import argparse
import h5py
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from spike_psvae.psvae import PSVAE
from spike_psvae.torch_utils import SpikeHDF5Dataset
from spike_psvae import stacks


ap = argparse.ArgumentParser()

ap.add_argument("input_h5", default="/mnt/3TB/charlie/features/wfs_locs.h5")
ap.add_argument("alpha", default=1.0)
ap.add_argument(
    "supervised_keys",
    default=["alpha", "x", "y", "z_rel"],
    type=lambda x: x.split(","),
)
ap.add_argument(
    "hidden_dims",
    default=[512],
    type=lambda x: list(map(int, x.split(","))),
)
ap.add_argument("unsupervised_latents", default=10)
ap.add_argument("log_interval", default=1000)
ap.add_argument("batch_size", default=8)
ap.add_argument("run_name", required=True, type=str)

args = ap.parse_args()


print("data from", args.input_h5)
with h5py.File(args.input_h5, "r") as f:
    for k in f.keys():
        print(k.ljust(20), f[k].dtype, f[k].shape)
    N, in_w, in_chan = f["denoised_waveforms"].shape
    in_shape = (in_w, in_chan)
    in_dim = in_w * in_chan
    print("x dim:", in_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%


# vanilla encoders and decoders
n_latents = len(args.supervised_keys) + args.unsupervised_latents
encoder = stacks.linear_encoder(in_dim, *args.hidden_dims, n_latents)
decoder = stacks.linear_decoder(n_latents, *args.hidden_dims[::-1], in_shape)

# %%
psvae = PSVAE(
    encoder, decoder, len(args.supervised_keys), args.unsupervised_latents
)
optimizer = torch.optim.Adam(psvae.parameters(), lr=1e-3)

# %%
dataset = SpikeHDF5Dataset(
    args.input_h5,
    "denoised_waveforms",
    args.supervised_keys,
)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1,
)

# %%
writer = SummaryWriter(
    log_dir=f"/mnt/3TB/charlie/features/runs/{args.run_name}",
)

psvae.to(device)
global_step = 0
n_epochs = 50
log_tic = time.time()
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

        if not batch_idx % args.log_interval:
            gsps = (time.time() - log_tic) / args.log_interval
            print(
                "Epoch",
                e,
                "batch",
                batch_idx,
                "loss",
                loss.item(),
                "global steps per sec",
                gsps,
                flush=True,
            )

            # -- Losses
            writer.add_scalar("Loss/loss", loss.cpu(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/{k}", v.cpu(), global_step)

            # -- Images
            x_ = x.cpu()
            recon_x_ = recon_x.cpu()
            im = torch.hstack((x_, recon_x_, x_ - recon_x_))
            im = im - im.min()
            im *= 255.0 / im.max()
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
            for y_key, y_mse in zip(args.supervised_keys, y_mses):
                writer.add_scalar(f"Stat/{y_key}_mse", y_mse, global_step)

            if np.isnan(loss.item()):
                break

        global_step += 1
    print("epoch", e, "took", (time.time() - tic) / 60, "min")

# %%
print("done")
