import h5py
import numpy as np
import torch
import multiprocessing
from collections import namedtuple
from multiprocessing.pool import Pool
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from spike_psvae import denoise, subtract, localize_index, spikeio


def extract_deconv(
    templates_up_path,
    spike_train_up_path,
    output_directory,
    standardized_bin,
    scalings_path=None,
    channel_index=None,
    n_channels_extract=20,
    subtraction_h5=None,
    save_residual=True,
    save_cleaned_waveforms=True,
    save_denoised_waveforms=False,
    localize=True,
    n_sec_chunk=1,
    n_jobs=1,
    sampling_rate=30_000,
    device=None,
    trough_offset=42,
    overwrite=True,
    scratch_dir=None,
    pbar=True,
):
    standardized_bin = Path(standardized_bin)
    output_directory = Path(output_directory)
    scratch_dir = output_directory
    if scratch_dir is not None:
        scratch_dir = Path(scratch_dir)

    templates_up = np.load(templates_up_path)
    n_templates, spike_length_samples, n_chans = templates_up.shape
    spike_train_up = np.load(spike_train_up_path)
    n_spikes = len(spike_train_up)

    if scalings_path is not None:
        scalings = np.load(scalings_path)
        assert scalings.shape == (n_spikes,)
    else:
        scalings = np.ones(n_spikes, dtype=templates_up.dtype)

    std_size = standardized_bin.stat().st_size
    assert not std_size % np.dtype(np.float32).itemsize
    std_size = std_size // np.dtype(np.float32).itemsize
    assert not std_size % n_chans
    T_samples = std_size // n_chans

    # discard un-loadable spikes

    # load TPCA if necessary
    if save_denoised_waveforms or localize:
        if subtraction_h5 is None:
            raise ValueError(
                "subtraction_h5 is needed to load TPCA when computing "
                "denoised waveforms or localizing."
            )
        subtraction_h5 = Path(subtraction_h5)

    # paths for output
    if save_residual:
        residual_path = output_directory / "residual.bin"
        if overwrite:
            if residual_path.exists():
                residual_path.unlink()
        resid = open(residual_path, "a")
    out_h5 = output_directory / "deconv_results.h5"
    if overwrite:
        if out_h5.exists():
            out_h5.unlink()
    temp_dir = scratch_dir / "temp_batch_results"
    temp_dir.mkdir(exist_ok=True, parents=True)

    # are we resuming?
    last_batch_end = 0
    if out_h5.exists():
        with h5py.File(out_h5, "r") as h5:
            last_batch_end = h5["last_batch_end"][()]

    # determine jobs to run
    batch_length = n_sec_chunk * sampling_rate
    start_samples = range(last_batch_end, T_samples, batch_length)
    if not len(start_samples):
        print("Extraction already done")
        return out_h5, residual_path

    # build spike index from templates and spike train
    templates_up_maxchans = templates_up.ptp(1).argmax(1)
    up_maxchans = templates_up_maxchans[spike_train_up[:, 1]]
    spike_index_up = np.c_[spike_train_up[:, 0], up_maxchans]

    # a firstchans-style channel index
    if channel_index is None:
        channel_index = subtract.make_contiguous_channel_index(
            n_chans, n_neighbors=n_channels_extract
        )

    # templates on few channels
    templates_loc = np.empty(
        (n_templates, spike_length_samples, n_channels_extract),
        dtype=templates_up.dtype,
    )
    for i in range(n_templates):
        templates_loc[i] = templates_up[i][
            :, channel_index[templates_up_maxchans[i]]
        ]

    with h5py.File(out_h5, "a") as h5:
        if last_batch_end > 0:
            if save_cleaned_waveforms:
                cleaned_wfs = h5["cleaned_waveforms"]
            if save_denoised_waveforms:
                denoised_wfs = h5["denoised_waveforms"]
            if localize:
                localizations = h5["localizations"]
                maxptps = h5["maxptps"]
        else:
            h5.create_dataset("templates_up", data=templates_up)
            h5.create_dataset(
                "templates_up_maxchans", data=templates_up_maxchans
            )
            h5.create_dataset("channel_index", data=channel_index)
            h5.create_dataset("templates_loc", data=templates_loc)
            h5.create_dataset("spike_train_up", data=spike_train_up)
            h5.create_dataset("scalings", data=scalings)
            h5.create_dataset("spike_index_up", data=spike_index_up)
            h5.create_dataset(
                "first_channels",
                data=channel_index[:, 0][spike_index_up[:, 1]],
            )
            h5.create_dataset("last_batch_end", data=0)

            if save_cleaned_waveforms:
                cleaned_wfs = h5.create_dataset(
                    "cleaned_waveforms",
                    shape=(n_spikes, spike_length_samples, n_channels_extract),
                    dtype=np.float32,
                )
            if save_denoised_waveforms:
                denoised_wfs = h5.create_dataset(
                    "denoised_waveforms",
                    shape=(n_spikes, spike_length_samples, n_channels_extract),
                    dtype=np.float32,
                )
            if localize:
                localizations = h5.create_dataset(
                    "localizations", shape=(n_spikes, 5), dtype=np.float64
                )
                maxptps = h5.create_dataset(
                    "maxptps", shape=(n_spikes,), dtype=np.float64
                )

        ctx = multiprocessing.get_context("spawn")
        if n_jobs is not None and n_jobs > 1:
            print("Initializing threads", end="")
        with Pool(
            n_jobs,
            initializer=_extract_deconv_init,
            initargs=(
                subtraction_h5,
                device,
                channel_index,
                batch_length,
                save_cleaned_waveforms,
                save_denoised_waveforms,
                save_residual,
                localize,
                trough_offset,
                spike_index_up,
                spike_train_up,
                standardized_bin,
                spike_length_samples,
                templates_up,
                up_maxchans,
                temp_dir,
                T_samples,
                templates_loc,
                scalings,
                n_chans,
            ),
            context=ctx,
        ) as pool:
            if n_jobs is not None and n_jobs > 1:
                print(" Ok.", flush=True)
            for result in xqdm(
                pool.imap(_extract_deconv_worker, start_samples),
                desc="extract deconv",
                smoothing=0,
                total=len(start_samples),
                pbar=pbar,
            ):
                h5["last_batch_end"][()] = result.last_batch_end

                if save_residual:
                    np.load(result.resid_path).tofile(resid)
                    Path(result.resid_path).unlink()

                if save_cleaned_waveforms:
                    cleaned_wfs[result.inds] = np.load(result.cleaned_path)
                    Path(result.cleaned_path).unlink()

                if save_denoised_waveforms:
                    denoised_wfs[result.inds] = np.load(result.denoised_path)
                    Path(result.denoised_path).unlink()

                if localize:
                    localizations[result.inds] = np.load(result.locs_path)
                    maxptps[result.inds] = np.load(result.maxptps_path)
                    Path(result.locs_path).unlink()
                    Path(result.maxptps_path).unlink()

    if save_residual:
        resid.close()

    return out_h5, residual_path


JobResult = namedtuple(
    "JobResult",
    [
        "last_batch_end",
        "resid_path",
        "cleaned_path",
        "denoised_path",
        "locs_path",
        "maxptps_path",
        "inds",
    ],
)


def _extract_deconv_worker(start_sample):
    # an easy name to extract the params set by _extract_deconv_init
    p = _extract_deconv_worker
    end_sample = min(p.T_samples, start_sample + p.batch_length)
    batch_str = f"{start_sample:012d}"
    which_spikes = np.flatnonzero(
        (p.spike_index_up[:, 0] >= start_sample)
        & (p.spike_index_up[:, 0] < end_sample)
    )
    spike_index = p.spike_index_up[which_spikes]
    spike_train = p.spike_train_up[which_spikes]
    scalings = p.scalings[which_spikes]

    # load raw recording with padding
    at_start = start_sample == 0
    at_end = end_sample == p.T_samples
    buffer_left = p.trough_offset * (not at_start)
    buffer_right = (p.spike_length_samples - p.trough_offset) * (not at_end)
    resid = subtract.read_data(
        p.standardized_bin,
        np.float32,
        start_sample - buffer_left,
        end_sample + buffer_right,
        p.n_chans,
    )
    # pad left/right if necessary
    # note, for spikes which end up loading in this buffer,
    # the localizations will be junk.
    pad_left = p.trough_offset * at_start
    pad_right = (p.spike_length_samples - p.trough_offset) * at_end
    if pad_left > 0 or pad_right > 0:
        resid = np.pad(resid, [(pad_left, pad_right), (0, 0)])

    # subtract templates in-place
    rel_times = np.arange(
        -start_sample + buffer_left + pad_left - p.trough_offset,
        -start_sample
        + buffer_left
        + pad_left
        + p.spike_length_samples
        - p.trough_offset,
    )
    for i in range(len(spike_index)):
        resid[spike_index[i, 0] + rel_times] -= (
            scalings[i] * p.templates_up[spike_train[i, 1]]
        )

    if p.save_residual:
        np.save(
            p.temp_dir / f"resid_{batch_str}.npy",
            resid[
                buffer_left + pad_left : len(resid) - buffer_right - pad_right
            ],
        )

    # -- load collision-cleaned waveforms
    if p.save_cleaned_waveforms or p.save_denoised_waveforms or p.localize:
        # initialize by reading from residual
        waveforms = spikeio.read_waveforms_in_memory(
            resid,
            spike_index,
            spike_length_samples=p.spike_length_samples,
            channel_index=p.channel_index,
            buffer=-start_sample + buffer_left + pad_left,
        )
        # now add the templates
        waveforms += (
            scalings[:, None, None] * p.templates_loc[spike_train[:, 1]]
        )
        if p.save_cleaned_waveforms:
            np.save(p.temp_dir / f"cleaned_{batch_str}.npy", waveforms)

    # -- denoise them
    if p.save_denoised_waveforms or p.localize:
        relative_batch_mcs = np.where(
            p.channel_index[spike_index[:, 1]] - spike_index[:, 1][:, None]
            == 0
        )[1]
        waveforms = temporal_align(waveforms, relative_batch_mcs)

        waveforms = subtract.full_denoising(
            waveforms,
            spike_index[:, 1],
            p.channel_index,
            probe="np1",
            tpca=p.tpca,
            device=p.device,
            denoiser=p.denoiser,
        )
        if p.save_denoised_waveforms:
            np.save(p.temp_dir / f"denoised_{batch_str}.npy", waveforms)

    # -- localize and done
    if p.localize:
        ptps = waveforms.ptp(1)
        maxptps = ptps.max(1)
        x, y, z_rel, z_abs, alpha = localize_index.localize_ptps_index(
            ptps,
            p.geom,
            spike_index[:, 1],
            p.channel_index,
            pbar=False,
        )
        np.save(p.temp_dir / f"maxptps_{batch_str}.npy", maxptps)
        np.save(
            p.temp_dir / f"locs_{batch_str}.npy",
            np.c_[x, y, z_rel, z_abs, alpha],
        )

    return JobResult(
        end_sample,
        p.temp_dir / f"resid_{batch_str}.npy",
        p.temp_dir / f"cleaned_{batch_str}.npy",
        p.temp_dir / f"denoised_{batch_str}.npy",
        p.temp_dir / f"locs_{batch_str}.npy",
        p.temp_dir / f"maxptps_{batch_str}.npy",
        which_spikes,
    )


def temporal_align(waveforms, maxchans, offset=42):
    N, T, C = waveforms.shape
    offsets = np.abs(waveforms[np.arange(N), :, maxchans]).argmax(1)
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


def _extract_deconv_init(
    subtraction_h5,
    device,
    channel_index,
    batch_length,
    save_cleaned_waveforms,
    save_denoised_waveforms,
    save_residual,
    localize,
    trough_offset,
    spike_index_up,
    spike_train_up,
    standardized_bin,
    spike_length_samples,
    templates_up,
    template_maxchans,
    temp_dir,
    T_samples,
    templates_loc,
    scalings,
    n_chans,
):
    geom = tpca = None
    if subtraction_h5 is not None:
        with h5py.File(subtraction_h5) as h5:
            tpca_mean = h5["tpca_mean"][:]
            tpca_components = h5["tpca_components"][:]
            geom = h5["geom"][:]
        tpca = PCA(tpca_components.shape[0])
        tpca.mean_ = tpca_mean
        tpca.components_ = tpca_components

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = denoise.SingleChanDenoiser().load().to(device)

    _extract_deconv_worker.geom = geom
    _extract_deconv_worker.tpca = tpca
    _extract_deconv_worker.device = device
    _extract_deconv_worker.denoiser = denoiser
    _extract_deconv_worker.channel_index = channel_index
    _extract_deconv_worker.save_cleaned_waveforms = save_cleaned_waveforms
    _extract_deconv_worker.save_denoised_waveforms = save_denoised_waveforms
    _extract_deconv_worker.save_residual = save_residual
    _extract_deconv_worker.localize = localize
    _extract_deconv_worker.trough_offset = trough_offset
    _extract_deconv_worker.spike_index_up = spike_index_up
    _extract_deconv_worker.spike_train_up = spike_train_up
    _extract_deconv_worker.standardized_bin = standardized_bin
    _extract_deconv_worker.spike_length_samples = spike_length_samples
    _extract_deconv_worker.templates_up = templates_up
    _extract_deconv_worker.templates_loc = templates_loc
    _extract_deconv_worker.scalings = scalings
    _extract_deconv_worker.temp_dir = temp_dir
    _extract_deconv_worker.T_samples = T_samples
    _extract_deconv_worker.batch_length = batch_length
    _extract_deconv_worker.n_chans = n_chans
    print(".", end="", flush=True)


def xqdm(it, pbar=True, **kwargs):
    if pbar:
        return tqdm(it, **kwargs)
    else:
        return it
