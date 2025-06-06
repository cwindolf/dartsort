"""Helper functions for localizing things other than torch Tensors
"""
import threading

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..util.multiprocessing_util import get_pool
from ..util.spiketorch import ptp
from .localize_torch import localize_amplitude_vectors, vmap_point_source_find_alpha


def localize_waveforms(
    waveforms,
    geom,
    main_channels=None,
    channel_index=None,
    radius=None,
    n_channels_subset=None,
):
    C, dim = geom.shape
    amp_vecs = ptp(waveforms, 1)
    if main_channels is None:
        main_channels = amp_vecs.argmax(1)

    return localize_amplitude_vectors(
        amp_vecs,
        geom,
        main_channels,
        channel_index=channel_index,
        radius=radius,
        n_channels_subset=n_channels_subset,
    )


def localize_hdf5(
    hdf5_filename,
    radius=None,
    n_channels_subset=None,
    output_dataset_name="point_source_localizations",
    main_channels_dataset_name="channels",
    amplitude_vectors_dataset_name="denoised_peak_amplitude_vectors",
    channel_index_dataset_name="channel_index",
    geometry_dataset_name="geom",
    spikes_per_batch=10 * 1024,
    overwrite=False,
    show_progress=True,
    device=None,
    localization_model="pointsource",
    logbarrier=True,
    n_jobs=0,
):
    """Run localization on a HDF5 file with stored amplitude vectors

    When this is run after a peeling pipeline which includes an AmplitudeVector
    feature, that's equivalent to having had a PointSourceLocalization feature.
    But, it's faster to do this after the fact.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    done, do_delete, next_batch_start = check_resume_or_overwrite(
        hdf5_filename,
        output_dataset_name,
        main_channels_dataset_name,
        overwrite=overwrite,
    )
    if done:
        return

    try:
        n_jobs, Executor, context, queue = get_pool(
            n_jobs, with_rank_queue=True, rank_queue_empty=True, cls="ProcessPoolExecutor"
        )
        with Executor(
            max_workers=n_jobs,
            mp_context=context,
            initializer=_h5_localize_init,
            initargs=(
                queue,
                hdf5_filename,
                main_channels_dataset_name,
                amplitude_vectors_dataset_name,
                channel_index_dataset_name,
                geometry_dataset_name,
                radius,
                n_channels_subset,
                localization_model,
                device,
                spikes_per_batch,
                logbarrier,
            ),
        ) as pool:
            with h5py.File(hdf5_filename, "r+", locking=False) as h5:
                n_spikes = h5[main_channels_dataset_name].shape[0]
                if do_delete:
                    del h5[output_dataset_name]
                localizations_dataset = h5.require_dataset(
                    output_dataset_name,
                    shape=(n_spikes, 4),
                    dtype=np.float64,
                    fillvalue=np.nan,
                    exact=True,
                )

                # workers are blocked waiting for their ranks, which
                # allows us to put the h5 in swmr before they open it
                if not h5.swmr_mode:
                    h5.swmr_mode = True
                for rank in range(n_jobs):
                    queue.put(rank)

                batches = range(next_batch_start, n_spikes, spikes_per_batch)
                results = pool.map(_h5_localize_job, batches)
                if show_progress:
                    results = tqdm(results, total=len(batches), desc="Localization")
                for start_ix, end_ix, xyza_batch in results:
                    localizations_dataset[start_ix:end_ix] = xyza_batch
                pool.shutdown()
    finally:
        if hasattr(_loc_context, 'ctx'):
            del _loc_context.ctx


def check_resume_or_overwrite(
    hdf5_filename,
    output_dataset_name,
    main_channels_dataset_name,
    overwrite=False,
):
    done = False
    do_delete = False
    next_batch_start = 0
    with h5py.File(hdf5_filename, "r", locking=False) as h5:
        if output_dataset_name in h5:
            n_spikes = h5[main_channels_dataset_name].shape[0]
            shape = h5[output_dataset_name].shape
            if overwrite and shape != (n_spikes, 4):
                do_delete = True
            elif shape != (n_spikes, 4):
                raise ValueError(
                    f"The {output_dataset_name} dataset in {hdf5_filename} "
                    f"has unexpected shape {shape}, where we expected {(n_spikes, 4)}."
                )
            else:
                # else, we are resuming
                nan_ix = np.flatnonzero(np.isnan(h5[output_dataset_name][:, 0]))
                if nan_ix.size:
                    next_batch_start = nan_ix[0]
                else:
                    done = True
    return done, do_delete, next_batch_start


class H5LocalizationContext:
    def __init__(
        self,
        hdf5_filename,
        main_channels_dataset_name,
        amplitude_vectors_dataset_name,
        channel_index_dataset_name,
        geometry_dataset_name,
        device,
        spikes_per_batch,
        radius,
        n_channels_subset,
        localization_model,
        logbarrier,
    ):
        h5 = h5py.File(hdf5_filename, "r", swmr=True, locking=False)
        channels = h5[main_channels_dataset_name][:]
        amp_vecs = h5[amplitude_vectors_dataset_name]
        channel_index = h5[channel_index_dataset_name][:]
        geom = h5[geometry_dataset_name][:]

        assert geom.shape == (channel_index.shape[0], 2)
        assert amp_vecs.shape == (*channels.shape, channel_index.shape[1])

        self.geom = torch.tensor(geom, device=device)
        self.channel_index = torch.tensor(channel_index, device=device)
        self.channels = torch.tensor(channels, device=device)
        self.amp_vecs = amp_vecs
        self.device = device
        self.n_spikes = channels.shape[0]
        self.spikes_per_batch = spikes_per_batch
        self.radius = radius
        self.n_channels_subset = n_channels_subset
        self.localization_model = localization_model
        self.logbarrier = logbarrier


_loc_context = threading.local()


def _h5_localize_init(
    rank_queue,
    hdf5_filename,
    main_channels_dataset_name,
    amplitude_vectors_dataset_name,
    channel_index_dataset_name,
    geometry_dataset_name,
    radius,
    n_channels_subset,
    localization_model,
    device,
    spikes_per_batch,
    logbarrier,
):
    my_rank = rank_queue.get()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda", index=my_rank % torch.cuda.device_count())

    _loc_context.ctx = H5LocalizationContext(
        hdf5_filename,
        main_channels_dataset_name,
        amplitude_vectors_dataset_name,
        channel_index_dataset_name,
        geometry_dataset_name,
        device,
        spikes_per_batch,
        radius,
        n_channels_subset,
        localization_model,
        logbarrier,
    )


def _h5_localize_job(start_ix):
    p = _loc_context.ctx

    end_ix = min(p.n_spikes, start_ix + p.spikes_per_batch)
    channels_batch = p.channels[start_ix:end_ix]
    amp_vecs_batch = torch.tensor(p.amp_vecs[start_ix:end_ix], device=p.device)

    with torch.enable_grad():
        locs = localize_amplitude_vectors(
            amp_vecs_batch,
            p.geom,
            channels_batch,
            channel_index=p.channel_index,
            radius=p.radius,
            n_channels_subset=p.n_channels_subset,
            model=p.localization_model,
            logbarrier=p.logbarrier,
        )
    xyza_batch = np.c_[
        locs["x"].cpu().numpy(),
        locs["y"].cpu().numpy(),
        locs["z_abs"].cpu().numpy(),
        locs["alpha"].cpu().numpy(),
    ]
    return start_ix, end_ix, xyza_batch


def point_source_mse(locs, amp_vecs, channels, channel_index, geom):
    is_torch = torch.is_tensor(locs)
    locs = torch.asarray(locs)
    amp_vecs = torch.asarray(amp_vecs)
    channels = torch.asarray(channels)
    channel_index = torch.asarray(channel_index)
    pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=np.nan)
    geom = torch.asarray(geom)
    pgeom = torch.asarray(pgeom)

    neighborhoods = channel_index[channels]
    invalid = neighborhoods == len(geom)
    dxz = pgeom[neighborhoods]

    if locs.shape[1] == 3:
        channel_mask = torch.logical_not(invalid).to(torch.float)
        alpha = vmap_point_source_find_alpha(
            torch.asarray(amp_vecs).nan_to_num(),
            channel_mask,
            *torch.asarray(locs).T,
            torch.asarray(dxz).nan_to_num(),
        )
        alpha = alpha.numpy()
    else:
        alpha = locs[:, 3]
        locs = locs[:, :3]

    x, y, z = locs.T

    dxz[:, :, 0] -= x[:, None]
    dxz[:, :, 1] -= z[:, None]
    dxz.square_()
    pred = dxz.sum(2)
    pred += y[:, None] ** 2
    pred.sqrt_()
    pred[invalid] = 1.0
    pred.reciprocal_()
    pred *= alpha[:, None]

    mse = amp_vecs - pred
    mse[invalid] = 0.0
    mse.square_()
    nnz = channel_index.shape[1] - invalid.sum(1)
    mse = mse.sum(1) / nnz

    if not is_torch:
        mse = mse.numpy()

    return mse
