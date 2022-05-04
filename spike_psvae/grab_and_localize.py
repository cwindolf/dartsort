import numpy as np
import torch

from collections import namedtuple
from multiprocessing.pool import Pool
from spike_psvae import denoise, localize_index, subtract


def grab_and_localize(
    spike_index,
    binary_file,
    geom,
    loc_radius=100,
    nn_denoise=True,
    enforce_decrease=True,
    tpca=None,
    chunk_size=30_000,
    n_jobs=1,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- make channel index
    channel_index = subtract.make_channel_index(
        geom, loc_radius, distance_order=False, p=1
    )

    # -- construct pool and job arguments, and run jobs
    localizations = np.empty((spike_index.shape[0], 5))
    with Pool(
        n_jobs,
        initializer=_job_init,
        initargs=(
            geom,
            nn_denoise,
            enforce_decrease,
            tpca,
            device,
            channel_index,
            chunk_size,
            spike_index,
            binary_file,
        ),
    ) as pool:
        for result in pool.imap(_job, range(0, spike_index[:, 0].max())):
            localizations[result.indices] = result.localizations

    return localizations


JobResult = namedtuple(
    "JobResult",
    [
        "indices",
        "localizations",
    ],
)


def _job(batch_start):
    # this is set by _job_init below in the worker
    p = _job.data

    # -- which spikes?
    which = np.flatnonzero(
        (batch_start <= p.spike_index[:, 0])
        & (p.spike_index[:, 0] < batch_start + p.chunk_size)
    )
    spike_index = p.spike_index[which]

    # -- load waveforms
    rec = subtract.read_data(
        p.binary_file,
        np.float32,
        batch_start,
        batch_start + p.chunk_size,
        len(p.geom),
    )
    waveforms = subtract.read_waveforms(
        rec,
        spike_index,
        extract_channel_index=p.channel_index,
        buffer=-batch_start,
    )

    # -- denoise
    waveforms = subtract.full_denoising(
        waveforms,
        spike_index[:, 1],
        p.channel_index,
        radial_parents=p.radial_parents,
        do_enforce_decrease=p.radial_parents is not None,
        tpca=p.tpca,
        device=p.device,
        denoiser=p.denoiser,
    )

    # -- localize and return
    ptps = waveforms.ptp(1)
    del waveforms
    x, y, z_rel, z_abs, alpha = localize_index.localize_ptps_index(
        ptps, p.geom, spike_index[:, 1], p.channel_index, pbar=False
    )

    return JobResult(which, np.c_[x, y, z_rel, z_abs, alpha])


JobData = namedtuple(
    "JobData",
    [
        "denoiser",
        "radial_parents",
        "tpca",
        "geom",
        "channel_index",
        "chunk_size",
        "spike_index",
        "binary_file",
        "device",
    ],
)


def _job_init(
    geom,
    nn_denoise,
    enforce_decrease,
    tpca,
    device,
    channel_index,
    chunk_size,
    spike_index,
    binary_file,
):
    denoiser = radial_parents = None
    if nn_denoise:
        denoiser = denoise.SingleChanDenoiser().load().to(device)
    if enforce_decrease:
        radial_parents = denoise.make_radial_order_parents(
            geom, channel_index, n_jumps_per_growth=1, n_jumps_parent=3
        )
    _job.data = JobData(
        denoiser,
        radial_parents,
        tpca,
        geom,
        channel_index,
        chunk_size,
        spike_index,
        binary_file,
        device,
    )
