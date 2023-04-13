import multiprocessing
import numpy as np
import torch

from collections import namedtuple
from .multiprocessing_utils import get_pool, MockQueue
from pathlib import Path
from tqdm.auto import tqdm

from . import denoise, localize_index, subtract, spikeio


def grab_and_localize(
    spike_index,
    binary_file,
    geom,
    trough_offset=42,
    loc_radius=100,
    nn_denoise=True,
    enforce_decrease=True,
    tpca=None,
    chunk_size=30_000,
    start_time=0,
    n_jobs=1,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- figure out length of recording
    # TODO make this a function in subtract
    std_size = Path(binary_file).stat().st_size
    assert not std_size % np.dtype(np.float32).itemsize
    std_size = std_size // np.dtype(np.float32).itemsize
    assert not std_size % len(geom)
    len_data_samples = std_size // len(geom)

    # -- make channel index
    channel_index = subtract.make_channel_index(
        geom, loc_radius, distance_order=False, p=1
    )
    # channel_index = subtract.make_contiguous_channel_index(len(geom), 10)

    # -- construct pool and job arguments, and run jobs
    localizations = np.empty((spike_index.shape[0], 5))
    maxptp = np.empty(spike_index.shape[0])
    starts = range(
        start_time,
        spike_index[:, 0].max(),
        chunk_size,
    )
    
    Executor, context = get_pool(n_jobs)
    manager = context.Manager() if n_jobs > 1 else None
    id_queue = manager.Queue() if n_jobs > 1 else MockQueue()

    n_jobs = n_jobs or 1
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() - 1

    for id in range(n_jobs):
        id_queue.put(id)
        
    with Executor(
        max_workers=n_jobs,
        initializer=_job_init,
        initargs=(
            id_queue,
            geom,
            nn_denoise,
            enforce_decrease,
            tpca,
            device,
            channel_index,
            chunk_size,
            spike_index,
            binary_file,
            len_data_samples,
            trough_offset,
        ),
        mp_context=context,
    ) as pool:
        for result in tqdm(
            pool.map(_job, starts),
            desc="grab and localize",
            smoothing=0,
            total=len(starts),
        ):
            localizations[result.indices] = result.localizations
            maxptp[result.indices] = result.maxptp

    return localizations, maxptp


JobResult = namedtuple(
    "JobResult",
    [
        "indices",
        "localizations",
        "maxptp",
    ],
)


def _job(batch_start):
    # p is set by _job_init below
    p = _job.data

    # -- which spikes?
    which = np.flatnonzero(
        (batch_start <= p.spike_index[:, 0])
        & (p.spike_index[:, 0] < batch_start + p.chunk_size)
    )
    spike_index = p.spike_index[which]

    # -- load waveforms
    rec = spikeio.read_data(
        p.binary_file,
        np.float32,
        max(0, batch_start - 42),
        min(p.len_data_samples, batch_start + p.chunk_size + 79),
        len(p.geom),
    )
    waveforms = spikeio.read_waveforms_in_memory(
        rec,
        spike_index,
        spike_length_samples=121,
        channel_index=p.channel_index,
        trough_offset=p.trough_offset,
        buffer=-batch_start + 42 * (batch_start > 0),
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
    waveforms = waveforms.detach().cpu().numpy()

    # -- localize and return
    ptps = waveforms.ptp(1)
    del waveforms
    x, y, z_rel, z_abs, alpha = localize_index.localize_ptps_index(
        ptps, p.geom, spike_index[:, 1], p.channel_index, pbar=False
    )

    return JobResult(
        which, np.c_[x, y, z_rel, z_abs, alpha], np.nanmax(ptps, axis=1)
    )


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
        "len_data_samples",
        "trough_offset",
    ],
)


def _job_init(
    id_queue,
    geom,
    nn_denoise,
    enforce_decrease,
    tpca,
    device,
    channel_index,
    chunk_size,
    spike_index,
    binary_file,
    len_data_samples,
    trough_offset,
):
    
    rank = id_queue.get()

    torch.set_grad_enabled(False)
    if torch.device(device).type == "cuda":
        print("num gpus:", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device = torch.device(
                "cuda", index=rank % torch.cuda.device_count()
            )
            print(
                f"Worker {rank} using GPU {rank % torch.cuda.device_count()} "
                f"out of {torch.cuda.device_count()} available."
            )
        torch.cuda._lazy_init()
    
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
        len_data_samples,
        trough_offset,
    )
