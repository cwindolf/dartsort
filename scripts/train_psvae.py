import argparse
import h5py
import numpy as np
import time
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from spike_psvae.psvae import PSVAE
from spike_psvae.data_utils import (
    SpikeHDF5Dataset,
    ContiguousRandomBatchSampler,
    LocalizingHDF5Dataset,
)
from spike_psvae import stacks


ap = argparse.ArgumentParser()

ap.add_argument("--input_h5", default="data/wfs_locs_b.h5", required=False)
ap.add_argument(
    "--waveforms_key", default="denoised_waveforms", required=False
)
ap.add_argument("--alpha", type=float, default=1.0, required=False)
ap.add_argument(
    "--supervised_keys",
    default=["alpha", "x", "y", "z_rel"],
    required=False,
    type=lambda x: x.split(","),
)
ap.add_argument("--y_min", default=None, required=False, type=float)
ap.add_argument(
    "--netspec",
    default="linear:512,512:256",
    required=False,
)
ap.add_argument("--unsupervised_latents", type=int, default=10, required=False)
ap.add_argument("--log_interval", type=int, default=1000, required=False)
ap.add_argument("--batch_size", type=int, default=8, required=False)
ap.add_argument("--run_name", type=str)
ap.add_argument("--nobatchnorm", action="store_true")
ap.add_argument("--num_data_workers", default=0, type=int)
ap.add_argument("--localize", action="store_true")
ap.add_argument(
    "--rundir", default=Path("/mnt/3TB/charlie/features/runs"), type=Path
)

args = ap.parse_args()


print("data from", args.input_h5)
print("netspec:", args.netspec)
with h5py.File(args.input_h5, "r") as f:
    for k in f.keys():
        print(k.ljust(20), f[k].dtype, f[k].shape)
    N, in_w, in_chan = f[args.waveforms_key].shape
    if args.localize:
        in_chan = 20
    in_shape = in_w, in_chan
    print("x shape:", in_shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%


# vanilla encoders and decoders
supervised_latents = len(args.supervised_keys)
n_latents = supervised_latents + args.unsupervised_latents
encoder, decoder = stacks.netspec(args.netspec, in_shape, not args.nobatchnorm)

# %%
psvae = PSVAE(encoder, decoder, supervised_latents, args.unsupervised_latents)
print(psvae)
optimizer = torch.optim.Adam(psvae.parameters(), lr=1e-3)

# %%
# data / batching managers
if not args.localize:
    dataset = SpikeHDF5Dataset(
        args.input_h5,
        args.waveforms_key,
        args.supervised_keys,
        y_min=args.y_min,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_data_workers,
        batch_sampler=ContiguousRandomBatchSampler(dataset, args.batch_size),
    )
else:
    with h5py.File(args.input_h5, "r") as f:
        dataset = LocalizingHDF5Dataset(
            f[args.waveforms_key][:],
            f["geom"][:],
            args.supervised_keys,
            y_min=args.y_min,
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_data_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

# %%
writer = SummaryWriter(
    log_dir=args.rundir / args.run_name,
)

psvae.to(device)
global_step = 0
n_epochs = 50
log_tic = time.time()
for e in range(n_epochs):
    tic = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        recon_x, y_hat, mu, logvar = psvae(x)
        loss, loss_dict = psvae.loss(x, y, recon_x, y_hat, mu, logvar)

        loss.backward()
        optimizer.step()

        if not batch_idx % args.log_interval:
            gsps = args.log_interval / (time.time() - log_tic)
            print(
                f"Epoch {e}, batch {batch_idx}. Loss {loss.item()}, "
                f"Global steps per sec: {gsps}",
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
            im -= im.min()
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

            log_tic = time.time()

        global_step += 1
    print("epoch", e, "took", (time.time() - tic) / 60, "min")

# %%
print("done")
