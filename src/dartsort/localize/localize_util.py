"""Helper functions for localizing things other than torch Tensors
"""

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..util.multiprocessing_util import get_pool
from ..util.py_util import delay_keyboard_interrupt
from .localize_torch import localize_amplitude_vectors
from ..util.spiketorch import ptp


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
            n_jobs, with_rank_queue=True, rank_queue_empty=True
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
            ),
        ) as pool:
            with h5py.File(hdf5_filename, "r+") as h5:
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
                    with delay_keyboard_interrupt:
                        localizations_dataset[start_ix:end_ix] = xyza_batch
    finally:
        global _loc_context
        if _loc_context is not None:
            _loc_context = None


def check_resume_or_overwrite(
    hdf5_filename,
    output_dataset_name,
    main_channels_dataset_name,
    overwrite=False,
):
    done = False
    do_delete = False
    next_batch_start = 0
    with h5py.File(hdf5_filename, "r") as h5:
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
    ):
        h5 = h5py.File(hdf5_filename, "r", swmr=True)
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


global _loc_context
_loc_context = None


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
):
    global _loc_context

    my_rank = rank_queue.get()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        if torch.cuda.device_count() > 1:
            device = torch.device(
                "cuda", index=my_rank % torch.cuda.device_count()
            )

    _loc_context = H5LocalizationContext(
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
    )


def _h5_localize_job(start_ix):
    global _loc_context
    p = _loc_context

    end_ix = min(p.n_spikes, start_ix + p.spikes_per_batch)
    channels_batch = p.channels[start_ix:end_ix]
    amp_vecs_batch = torch.tensor(p.amp_vecs[start_ix:end_ix], device=p.device)

    locs = localize_amplitude_vectors(
        amp_vecs_batch,
        p.geom,
        channels_batch,
        channel_index=p.channel_index,
        radius=p.radius,
        n_channels_subset=p.n_channels_subset,
        model=p.localization_model,
    )
    xyza_batch = np.c_[
        locs["x"].cpu().numpy(),
        locs["y"].cpu().numpy(),
        locs["z_abs"].cpu().numpy(),
        locs["alpha"].cpu().numpy(),
    ]
    return start_ix, end_ix, xyza_batch
