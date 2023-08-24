import numpy as np
import os
from pathlib import Path
import h5py
from sklearn.decomposition import PCA
from spike_psvae import (
    subtract,
    localization,
    deconvolve,
)
from tqdm.auto import tqdm


def extract_deconv_wfs(
    h5_subtract,
    residual_path,
    geom_array,
    deconv_spike_train_up,
    deconv_templates_up,
    output_directory,
    denoiser,
    device,
    batch_size=1024,
    n_chans_to_extract=40,
    rank_tpca=8,
):

    # get waveforms dir
    subtracted_waveforms_dir = os.path.join(
        output_directory, "subtracted_waveforms"
    )
    collision_subtracted_waveforms_dir = os.path.join(
        output_directory, "collision_subtracted_waveforms"
    )
    denoised_waveforms_dir = os.path.join(
        output_directory, "denoised_waveforms"
    )
    Path(subtracted_waveforms_dir).mkdir(exist_ok=True, parents=True)
    Path(collision_subtracted_waveforms_dir).mkdir(exist_ok=True, parents=True)
    Path(denoised_waveforms_dir).mkdir(exist_ok=True, parents=True)

    n_spikes = deconv_spike_train_up.shape[0]
    batch_id = 0
    skipped_count = 0
    deconv_spike_index = np.zeros((n_spikes, 2)).astype(int)
    deconv_labels = np.zeros(n_spikes).astype(int)

    extract_channel_index = []
    for c in range(len(geom_array)):
        low = max(0, c - n_chans_to_extract // 2)
        low = min(len(geom_array) - n_chans_to_extract, low)
        extract_channel_index.append(np.arange(low, low + n_chans_to_extract))
    extract_channel_index = np.array(extract_channel_index)

    # load tPCA
    with h5py.File(h5_subtract, "r") as f:
        tpca_components = f["tpca_components"][:]
        tpca_mean = f["tpca_mean"][:]

    tpca = PCA(rank_tpca)
    tpca.components_ = tpca_components
    tpca.mean_ = tpca_mean

    for start in tqdm(range(0, n_spikes, batch_size)):
        end = start + batch_size

        batch_deconv_spike_train_up = deconv_spike_train_up[start:end]
        batch_t, batch_template_idx = (
            batch_deconv_spike_train_up[:, 0],
            batch_deconv_spike_train_up[:, 1],
        )
        batch_subtracted_wfs = deconv_templates_up[batch_template_idx]
        batch_mcs = batch_subtracted_wfs.ptp(1).argmax(1)
        batch_extract_channel_index = extract_channel_index[batch_mcs]

        batch_subtracted_wfs = np.array(
            list(
                map(
                    lambda x, idx: x[:, idx],
                    batch_subtracted_wfs,
                    batch_extract_channel_index,
                )
            )
        )

        deconv_spike_index[start:end, 0] = batch_t
        deconv_spike_index[start:end, 1] = batch_mcs
        deconv_labels[start:end] = batch_template_idx

        # load residual batch
        # THIS FUNCTION LOADS AT -60 + 60
        # NEED batch_deconv_spike_train_up[:,0] to indicate trough+18
        residual_batch, skipped_idx = deconvolve.read_waveforms(
            batch_t, residual_path, geom_array
        )
        residual_batch = np.array(
            list(
                map(
                    lambda x, idx: x[:, idx],
                    residual_batch,
                    batch_extract_channel_index,
                )
            )
        )
        kept_idx = np.ones(batch_subtracted_wfs.shape[0]).astype(bool)
        kept_idx[skipped_idx] = False
        skipped_count += len(skipped_idx)

        batch_collision_subtracted_wfs = (
            batch_subtracted_wfs[kept_idx] + residual_batch
        )

        relative_batch_mcs = np.where(
            batch_extract_channel_index - batch_mcs[:, None] == 0
        )[1]
        aligned_wfs = temporal_align(
            batch_collision_subtracted_wfs, relative_batch_mcs
        )

        batch_denoised_wfs = subtract.full_denoising(
            aligned_wfs,
            batch_mcs,
            extract_channel_index,
            None,
            probe="np1",
            tpca=tpca,
            device=device,
            denoiser=denoiser,
        )

        np.save(
            os.path.join(
                subtracted_waveforms_dir,
                f"subtracted_{str(batch_id).zfill(6)}.npy",
            ),
            batch_subtracted_wfs.astype(np.float32),
        )
        np.save(
            os.path.join(
                collision_subtracted_waveforms_dir,
                f"collision_subtracted_{str(batch_id).zfill(6)}.npy",
            ),
            batch_collision_subtracted_wfs.astype(np.float32),
        )
        np.save(
            os.path.join(
                denoised_waveforms_dir,
                f"denoised_{str(batch_id).zfill(6)}.npy",
            ),
            batch_denoised_wfs.astype(np.float32),
        )
        batch_id += 1

    fname_spike_index = os.path.join(output_directory, "spike_index.npy")
    fname_spike_labels = os.path.join(output_directory, "spike_labels.npy")

    np.save(fname_spike_index, deconv_spike_index)
    np.save(fname_spike_labels, deconv_labels)

    # Merge wfs in h5 fils
    shape = (n_spikes - skipped_count, 121, n_chans_to_extract)

    fname_subtracted = os.path.join(output_directory, "subtracted_wfs.h5")
    fname_cleaned = os.path.join(
        output_directory, "collision_subtracted_wfs.h5"
    )
    fname_denoised = os.path.join(output_directory, "denoised_wfs.h5")

    merge_files_h5(
        subtracted_waveforms_dir, fname_subtracted, "wfs", shape, channel_index=extract_channel_index, delete=True
    )
    Path(fname_subtracted).unlink()
    merge_files_h5(
        collision_subtracted_waveforms_dir,
        fname_cleaned,
        "wfs",
        shape,
        channel_index=extract_channel_index,
        delete=True,
    )
    merge_files_h5(
        denoised_waveforms_dir, fname_denoised, "wfs", shape, channel_index=extract_channel_index, delete=True
    )

    return (
        fname_spike_index,
        fname_spike_labels,
        fname_subtracted,
        fname_cleaned,
        fname_denoised,
    )


def merge_files_h5(
    filtered_location, output_h5, dataset_name, shape, channel_index=None, delete=False
):
    with h5py.File(output_h5, "w") as out:
        if channel_index is not None:
            out.create_dataset("channel_index", data=channel_index)
        wfs = out.create_dataset(dataset_name, shape=shape, dtype=np.float32)
        filenames = os.listdir(filtered_location)
        filenames_sorted = sorted(filenames)
        i = 0
        for fname in tqdm(filenames_sorted):
            if ".ipynb" in fname or ".bin" in fname:
                continue
            res = np.load(os.path.join(filtered_location, fname)).astype(
                "float32"
            )
            n_new = res.shape[0]
            wfs[i : i + n_new] = res
            i += n_new

            if delete:
                Path(os.path.join(filtered_location, fname)).unlink()


def relocalize_extracted_wfs(
    denoised_wfs_h5,
    deconv_spike_train_up,
    deconv_spike_index,
    geom_array,
    output_directory,
    n_workers=8,
    batch_size=16384,
    fs=30000,
):

    h5 = h5py.File(denoised_wfs_h5)
    extract_channel_index = h5["channel_index"][:]
    denoised_wfs = h5["wfs"]
    # times = deconv_spike_train_up[:, 0].copy() / fs
    n_spikes = denoised_wfs.shape[0]
    xss = []
    yss = []
    z_relss = []
    z_absss = []
    alphass = []
    maxptpss = []
    mcss = []
    fcss = []

    for start in tqdm(range(0, n_spikes, batch_size)):
        end = start + batch_size
        ptps = denoised_wfs[start:end].copy().ptp(1)
        batch_mcs = deconv_spike_index[start:end, 1].copy()
        batch_fcs = extract_channel_index[batch_mcs][:, 0]
        xs, ys, z_rels, z_abss, alphas, _ = localization.localize_ptps(
            ptps, geom_array, batch_fcs, batch_mcs, n_workers=n_workers, pbar=False
        )
        xss.append(xs)
        yss.append(ys)
        z_relss.append(z_rels)
        z_absss.append(z_abss)
        alphass.append(alphas)
        maxptpss.append(ptps.max(1))
        mcss.append(batch_mcs)
        fcss.append(batch_fcs)

    xss = np.concatenate(xss)
    yss = np.concatenate(yss)
    z_relss = np.concatenate(z_relss)
    z_absss = np.concatenate(z_absss)
    alphass = np.concatenate(alphass)
    maxptpss = np.concatenate(maxptpss)
    mcss = np.concatenate(mcss)
    fcss = np.concatenate(fcss)

    localization_results = np.zeros((n_spikes, 7))
    localization_results[:, 0] = xss
    localization_results[:, 1] = z_absss
    localization_results[:, 2] = yss
    localization_results[:, 3] = alphass
    localization_results[:, 4] = maxptpss
    localization_results[:, 5] = fcss
    localization_results[:, 6] = mcss

    np.save(
        os.path.join(output_directory, "localization_results.npy"),
        localization_results,
    )


def temporal_align(waveforms, maxchans, offset=42):
    N, T, C = waveforms.shape
    offsets = waveforms[np.arange(N), :, maxchans].argmin(1)
    rolls = offset - offsets
    out = np.empty_like(waveforms)
    pads = [(0, 0), (0, 0)]
    for i, roll in enumerate(rolls):
        if roll > 0:
            pads[0] = (roll, 0)
            start, end = 0, T
        elif roll < 0:
            pads[0] = (0, -roll)
            start, end = -roll, T - roll
        else:
            out[i] = waveforms[i]
            continue

        pwf = np.pad(waveforms[i], pads, mode="linear_ramp")
        out[i] = pwf[start:end, :]

    return out
