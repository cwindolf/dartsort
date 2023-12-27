"""Helper functions for localizing things other than torch Tensors
"""
import h5py
import numpy as np
import torch
from tqdm.auto import trange

from .localize_torch import localize_amplitude_vectors


def localize_waveforms(
    waveforms,
    geom,
    main_channels=None,
    channel_index=None,
    radius=None,
    n_channels_subset=None,
):
    C, dim = geom.shape
    amp_vecs = waveforms.ptp(1)
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
    amplitude_vectors_dataset_name="denoised_amplitude_vectors",
    channel_index_dataset_name="channel_index",
    geometry_dataset_name="geom",
    spikes_per_batch=10_000,
    overwrite=False,
    show_progress=True,
    device=None,
    localization_model="pointsource",
):
    """Run localization on a HDF5 file with stored amplitude vectors

    When this is run after a peeling pipeline which includes an AmplitudeVector
    feature, that's equivalent to having had a PointSourceLocalization feature.
    But, it's faster to do this after the fact.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with h5py.File(hdf5_filename, "r+") as h5:
        channels = h5[main_channels_dataset_name][:]
        amp_vecs = h5[amplitude_vectors_dataset_name]
        channel_index = h5[channel_index_dataset_name][:]
        geom = h5[geometry_dataset_name][:]

        n_spikes = channels.shape[0]
        assert geom.shape == (channel_index.shape[0], 2)
        assert amp_vecs.shape == (*channels.shape, channel_index.shape[1])
        if output_dataset_name in h5:
            shape = h5[output_dataset_name].shape
            if overwrite:
                del h5[output_dataset_name]
            elif shape != (n_spikes, 4):
                raise ValueError(
                    f"The {output_dataset_name} dataset in {hdf5_filename} "
                    f"has unexpected shape {shape}, where we expected {(n_spikes, 4)}."
                )
            else:
                # else, we are resuming
                return

        geom = torch.tensor(geom, device=device)
        channel_index = torch.tensor(channel_index, device=device)
        channels = torch.tensor(channels, device=device)

        localizations_dataset = h5.create_dataset(
            output_dataset_name,
            shape=(n_spikes, 4),
            dtype=np.float64,
        )

        batches = range(0, n_spikes, spikes_per_batch)
        if show_progress:
            batches = trange(0, n_spikes, spikes_per_batch, desc="Localization")
        for start_ix in batches:
            end_ix = min(n_spikes, start_ix + spikes_per_batch)

            channels_batch = channels[start_ix:end_ix]
            amp_vecs_batch = torch.tensor(amp_vecs[start_ix:end_ix], device=device)

            locs = localize_amplitude_vectors(
                amp_vecs_batch,
                geom,
                channels_batch,
                channel_index=channel_index,
                radius=radius,
                n_channels_subset=n_channels_subset,
                model=localization_model,
            )
            xyza_batch = np.c_[
                locs["x"].cpu().numpy(),
                locs["y"].cpu().numpy(),
                locs["z_abs"].cpu().numpy(),
                locs["alpha"].cpu().numpy(),
            ]
            localizations_dataset[start_ix:end_ix] = xyza_batch
