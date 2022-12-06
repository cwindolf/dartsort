from pathlib import Path
import contextlib
import gc
import h5py
import numpy as np
import signal
import time
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import logging
import pandas as pd
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from copy import copy

try:
    from spikeglx import _geometry_from_meta, read_meta_data
except ImportError:
    try:
        from ibllib.io.spikeglx import _geometry_from_meta, read_meta_data
    except ImportError:
        raise ImportError("Can't find spikeglx...")

from . import denoise, detect, localize_index, subtraction_feats
from .multiprocessing_utils import MockPoolExecutor, MockQueue
from .spikeio import get_binary_length, read_data, read_waveforms_in_memory
from .waveform_utils import make_channel_index, make_contiguous_channel_index

_logger = logging.getLogger(__name__)


default_extra_feats = [
    subtraction_feats.MaxPTP(),
    subtraction_feats.TroughDepth(),
    subtraction_feats.PeakHeight(),
]


def subtraction(
    standardized_bin,
    out_folder,
    geom=None,
    # should we start over?
    overwrite=False,
    # waveform args
    spike_length_samples=121,
    trough_offset=42,
    # tpca args
    tpca_rank=8,
    n_sec_pca=10,
    # time / input binary details
    n_sec_chunk=1,
    t_start=0,
    t_end=None,
    sampling_rate=30_000,
    nsync=0,
    binary_dtype=np.float32,
    # detection
    thresholds=[12, 10, 8, 6, 5, 4],
    peak_sign="neg",
    nn_detect=False,
    denoise_detect=False,
    do_nn_denoise=True,
    # waveform extraction channels
    neighborhood_kind="firstchan",
    extract_box_radius=200,
    extract_firstchan_n_channels=40,
    box_norm_p=np.inf,
    dedup_spatial_radius=70,
    enforce_decrease_kind="columns",
    # what to save?
    save_residual=False,
    save_subtracted_waveforms=False,
    save_cleaned_waveforms=False,
    save_denoised_waveforms=False,
    save_subtracted_tpca_projs=True,
    save_cleaned_tpca_projs=True,
    save_denoised_tpca_projs=True,
    # localization args
    # set this to None or "none" to turn off
    localization_kind="logbarrier",
    localize_radius=100,
    localize_firstchan_n_channels=20,
    loc_workers=4,
    # want to compute any other features of the waveforms?
    extra_features="default",
    # misc kwargs
    random_seed=0,
    denoiser_init_kwargs={},
    denoiser_weights_path=None,
    dtype=np.float32,
    n_jobs=1,
    device=None,
):
    """Subtraction-based waveform extraction

    Runs subtraction pipeline, and optionally also the localization.

    Loads data from a binary file (standardized_bin), and loads geometry
    from the associated meta file if `geom=None`.

    Results are saved to `out_folder` in the following format:
     - residual_[dataset name]_[time region].bin
        - a binary file like the input binary
     - subtraction_[dataset name]_[time region].h5
        - An HDF5 file containing all of the resulting data.
          In detail, if N is the number of discovered waveforms,
          n_channels is the number of channels in the probe,
          T is `spike_len_samples`, C is the number of channels
          of the extracted waveforms (determined from `extract_box_radius`),
          then this HDF5 file contains the following datasets:

            geom : (n_channels, 2)
            start_sample : scalar
            end_sample : scalar
                First and last sample of time region considered
                (controlled by arguments `t_start`, `t_end`)
            channel_index : (n_channels, C)
                Array of channel indices. channel_index[c] contains the
                channels that a waveform with max channel `c` was extracted
                on.
            tpca_mean, tpca_components : arrays
                The fitted temporal PCA parameters.
            spike_index : (N, 2)
                The columns are (sample, max channel)
            subtracted_waveforms : (N, T, C)
                Waveforms that were subtracted
            cleaned_waveforms : (N, T, C)
                Final denoised waveforms, only computed/saved if
                `do_clean=True`
            localizations : (N, 5)
                Only computed/saved if `localization_kind` is `"logbarrier"`
                or `"original"`
                The columsn are: x, y, z, alpha, z relative to max channel
            maxptps : (N,)
                Only computed/saved if `localization_kind="logbarrier"`

    Returns
    -------
    out_h5 : path to output hdf5 file
    residual : path to residual if save_residual
    """
    if extra_features == "default":
        extra_features = copy(default_extra_feats)

    if neighborhood_kind not in ("firstchan", "box", "circle"):
        raise ValueError(
            "Neighborhood kind", neighborhood_kind, "not understood."
        )
    if enforce_decrease_kind not in ("columns", "radial", "none"):
        raise ValueError(
            "Enforce decrease method", enforce_decrease_kind, "not understood."
        )
    if peak_sign not in ("neg", "both"):
        raise ValueError("peak_sign", peak_sign, "not understood.")

    if neighborhood_kind == "circle":
        neighborhood_kind = "box"
        box_norm_p = 2

    standardized_bin = Path(standardized_bin)
    stem = standardized_bin.stem
    batch_len_samples = n_sec_chunk * sampling_rate

    # prepare output dir
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True)
    batch_data_folder = out_folder / f"batches_{stem}"
    batch_data_folder.mkdir(exist_ok=True)
    out_h5 = out_folder / f"subtraction_{stem}_t_{t_start}_{t_end}.h5"
    if save_residual:
        residual_bin = out_folder / f"residual_{stem}_t_{t_start}_{t_end}.bin"
    try:
        # this is to check if another program is using our h5, in which
        # case we should crash early rather than late.
        if out_h5.exists():
            with h5py.File(out_h5, "r+") as _:
                pass
            del _
    except BlockingIOError as e:
        raise ValueError(
            f"Output HDF5 {out_h5} is currently in use by another program. "
            "Maybe a Jupyter notebook that's still running?"
        ) from e

    # pick device if it's not supplied
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda._lazy_init()
            torch.set_grad_enabled(False)
    else:
        device = torch.device(device)

    # if no geometry is supplied, try to load it from meta file
    if geom is None:
        geom = read_geom_from_meta(standardized_bin)
        if geom is None:
            raise ValueError(
                "Either pass `geom` or put meta file in folder with binary."
            )
    n_channels = geom.shape[0]

    # TODO: read this from meta.
    # right now it's just used to load NN detector and pick the enforce
    # decrease method if enforce_decrease_kind=="columns"
    probe = {4: "np1", 2: "np2"}.get(np.unique(geom[:, 0]).size, None)

    # figure out if we will use a NN detector, and if so which
    nn_detector_path = None
    if nn_detect:
        nn_detector_path = (
            Path(__file__).parent.parent / f"pretrained/detect_{probe}.pt"
        )
        print("Using pretrained detector for", probe, "from", nn_detector_path)
        detection_kind = "voltage->NN"
    elif denoise_detect:
        print("Using denoising NN detection")
        detection_kind = "denoised PTP"
    else:
        print("Using voltage detection")
        detection_kind = "voltage"

    # figure out length of data
    T_samples, T_sec = get_binary_length(
        standardized_bin,
        n_channels,
        sampling_rate,
        nsync=nsync,
        dtype=binary_dtype,
    )
    assert t_start >= 0 and (t_end is None or t_end <= T_sec)
    start_sample = int(np.floor(t_start * sampling_rate))
    end_sample = (
        T_samples if t_end is None else int(np.floor(t_end * sampling_rate))
    )
    portion_len_s = (end_sample - start_sample) / sampling_rate
    print(
        f"Running subtraction. Total recording length is {T_sec:0.2f} "
        f"s, running on portion of length {portion_len_s:0.2f} s. "
        f"Using {detection_kind} detection with thresholds: {thresholds}."
    )

    # compute helper data structures
    # channel indexes for extraction, NN detection, deduplication
    dedup_channel_index = make_channel_index(
        geom, dedup_spatial_radius, steps=2
    )
    nn_channel_index = make_channel_index(geom, dedup_spatial_radius, steps=1)
    if neighborhood_kind == "box":
        extract_channel_index = make_channel_index(
            geom, extract_box_radius, distance_order=False, p=box_norm_p
        )
    elif neighborhood_kind == "firstchan":
        extract_channel_index = make_contiguous_channel_index(
            n_channels, n_neighbors=extract_firstchan_n_channels
        )
    else:
        assert False

    # helper data structure for radial enforce decrease
    do_enforce_decrease = True
    radial_parents = None
    if enforce_decrease_kind == "radial":
        radial_parents = denoise.make_radial_order_parents(
            geom, extract_channel_index, n_jumps_per_growth=1, n_jumps_parent=3
        )
    elif enforce_decrease_kind == "columns":
        pass
    else:
        print("Skipping enforce decrease.")
        do_enforce_decrease = False

    # check localization arg
    if localization_kind in ("original", "logbarrier"):
        print("Using", localization_kind, "localization")
        extra_features += [
            subtraction_feats.Localization(
                geom,
                extract_channel_index,
                loc_n_chans=localize_firstchan_n_channels
                if neighborhood_kind == "firstchan"
                else None,
                loc_radius=localize_radius
                if neighborhood_kind != "firstchan"
                else None,
                localization_kind=localization_kind,
            )
        ]
    else:
        print("No localization")

    # see if we are asked to save any waveforms
    wf_bools = (
        save_subtracted_waveforms,
        save_cleaned_waveforms,
        save_denoised_waveforms,
    )
    wf_names = ("subtracted", "cleaned", "denoised")
    for do_save, kind in zip(wf_bools, wf_names):
        if do_save:
            extra_features += [
                subtraction_feats.Waveform(
                    which_waveforms=kind,
                )
            ]

    # see if we are asked to save tpca projs for
    # collision-cleaned or denoised waveforms
    subtracted_tpca_feat = subtraction_feats.TPCA(
        tpca_rank,
        extract_channel_index,
        which_waveforms="subtracted",
        random_state=random_seed,
    )
    if save_subtracted_tpca_projs:
        extra_features += [subtracted_tpca_feat]
    if save_cleaned_tpca_projs:
        extra_features += [
            subtraction_feats.TPCA(
                tpca_rank,
                extract_channel_index,
                which_waveforms="cleaned",
                random_state=random_seed,
            )
        ]
    do_clean = False
    fit_feats = []
    if save_denoised_tpca_projs or localization_kind in (
        "original",
        "logbarrier",
    ):
        denoised_tpca_feat = subtraction_feats.TPCA(
            tpca_rank,
            extract_channel_index,
            which_waveforms="denoised",
            random_state=random_seed,
        )
        do_clean = True
        if save_denoised_tpca_projs:
            extra_features += [denoised_tpca_feat]
        else:
            fit_feats += [denoised_tpca_feat]

    # temporal PCA for subtracted waveforms
    # try to load old TPCA if it's around
    if not overwrite and out_h5.exists():
        with h5py.File(out_h5, "r") as output_h5:
            subtracted_tpca_feat.from_h5(output_h5)

    # otherwise, train it
    # TODO: ideally would run this on another process,
    # because this is the only time the main thread uses
    # GPU, and we could avoid initializing torch runtime.
    if subtracted_tpca_feat.needs_fit:
        with timer("Training TPCA..."), torch.no_grad():
            train_featurizers(
                standardized_bin,
                spike_length_samples,
                extract_channel_index,
                geom,
                radial_parents,
                T_samples,
                sampling_rate,
                dedup_channel_index,
                thresholds,
                nn_detector_path,
                denoise_detect,
                nn_channel_index,
                extra_features=[subtracted_tpca_feat],
                subtracted_tpca=None,
                nsync=nsync,
                peak_sign=peak_sign,
                do_nn_denoise=do_nn_denoise,
                do_enforce_decrease=do_enforce_decrease,
                probe=probe,
                denoiser_init_kwargs=denoiser_init_kwargs,
                denoiser_weights_path=denoiser_weights_path,
                n_sec_pca=n_sec_pca,
                random_seed=random_seed,
                device=device,
                binary_dtype=binary_dtype,
                dtype=dtype,
            )

        # try to free up some memory on GPU that might have been used above
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    # train featurizers
    if any(f.needs_fit for f in extra_features + fit_feats):
        # try to load old featurizers
        if not overwrite and out_h5.exists():
            with h5py.File(out_h5, "r") as output_h5:
                for f in extra_features:
                    if f.needs_fit:
                        f.from_h5(output_h5)

        # train any which couldn't load
        if any(f.needs_fit for f in extra_features + fit_feats):
            train_featurizers(
                standardized_bin,
                spike_length_samples,
                extract_channel_index,
                geom,
                radial_parents,
                T_samples,
                sampling_rate,
                dedup_channel_index,
                thresholds,
                nn_detector_path,
                denoise_detect,
                nn_channel_index,
                subtracted_tpca=subtracted_tpca_feat.tpca,
                extra_features=extra_features + fit_feats,
                nsync=nsync,
                peak_sign=peak_sign,
                do_nn_denoise=do_nn_denoise,
                do_enforce_decrease=do_enforce_decrease,
                probe=probe,
                denoiser_init_kwargs=denoiser_init_kwargs,
                denoiser_weights_path=denoiser_weights_path,
                n_sec_pca=n_sec_pca,
                random_seed=random_seed,
                device=device,
                binary_dtype=binary_dtype,
                dtype=dtype,
            )

    # if we're on GPU, we can't use processes, since each process will
    # have it's own torch runtime and those will use all the memory
    if device.type == "cuda":
        pass
    else:
        if loc_workers > 1:
            print(
                "Setting number of localization workers to 1. (Since "
                "you're on CPU, use a large n_jobs for parallelism.)"
            )
            loc_workers = 1

    # parallel batches
    jobs = list(
        enumerate(
            range(
                start_sample,
                end_sample,
                batch_len_samples,
            )
        )
    )

    # -- initialize storage
    with get_output_h5(
        out_h5,
        start_sample,
        end_sample,
        geom,
        extract_channel_index,
        sampling_rate,
        extra_features,
        overwrite=overwrite,
        dtype=dtype,
    ) as (output_h5, last_sample):

        spike_index = output_h5["spike_index"]
        feature_dsets = [output_h5[f.name] for f in extra_features]
        N = len(spike_index)

        # if we're resuming, filter out jobs we already did
        jobs = [
            (batch_id, start)
            for batch_id, start in jobs
            if start >= last_sample
        ]
        n_batches = len(jobs)

        # residual binary file -- append if we're resuming
        if save_residual:
            residual_mode = "ab" if last_sample > 0 else "wb"
            residual = open(residual_bin, mode=residual_mode)

        # now run subtraction in parallel
        jobs = (
            (
                batch_data_folder,
                s_start,
                batch_len_samples,
                standardized_bin,
                thresholds,
                subtracted_tpca_feat.tpca,
                denoised_tpca_feat.tpca if do_clean else None,
                trough_offset,
                dedup_channel_index,
                spike_length_samples,
                extract_channel_index,
                start_sample,
                end_sample,
                do_clean,
                save_residual,
                radial_parents,
                geom,
                do_enforce_decrease,
                probe,
                peak_sign,
                nsync,
                binary_dtype,
                dtype,
                extra_features,
            )
            for batch_id, s_start in jobs
        )

        # no-threading/multiprocessing execution for debugging if n_jobs == 0
        Executor = ProcessPoolExecutor if n_jobs else MockPoolExecutor
        context = multiprocessing.get_context("spawn")
        manager = context.Manager() if n_jobs else None
        id_queue = manager.Queue() if n_jobs else MockQueue()

        n_jobs = n_jobs or 1
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() - 1

        for id in range(n_jobs):
            id_queue.put(id)

        with Executor(
            max_workers=n_jobs,
            mp_context=context,
            initializer=_subtraction_batch_init,
            initargs=(
                device,
                nn_detector_path,
                nn_channel_index,
                denoise_detect,
                do_nn_denoise,
                id_queue,
                denoiser_init_kwargs,
                denoiser_weights_path,
            ),
        ) as pool:
            for result in tqdm(
                pool.map(_subtraction_batch, jobs),
                total=n_batches,
                desc="Batches",
                smoothing=0,
            ):
                with noint:
                    N_new = result.N_new

                    # write new residual
                    if save_residual:
                        np.load(result.residual).tofile(residual)
                        Path(result.residual).unlink()

                    # grow arrays as necessary and write results
                    if N_new > 0:
                        spike_index.resize(N + N_new, axis=0)
                        spike_index[N:] = np.load(result.spike_index)
                        Path(result.spike_index).unlink()
                        for f, dset in zip(extra_features, feature_dsets):
                            dset.resize(N + N_new, axis=0)
                            fnpy = (
                                batch_data_folder / f"{result.prefix}{f.name}.npy"
                            )
                            dset[N:] = np.load(fnpy)
                            Path(fnpy).unlink()
                        # update spike count
                        N += N_new

    # -- done!
    if save_residual:
        residual.close()
    print("Done. Detected", N, "spikes")
    print("Results written to:")
    if save_residual:
        print(residual_bin)
    print(out_h5)
    try:
        batch_data_folder.rmdir()
    except OSError as e:
        print(e)
    return out_h5


# -- subtraction routines


# the return type for `subtraction_batch` below
SubtractionBatchResult = namedtuple(
    "SubtractionBatchResult",
    ["N_new", "s_start", "s_end", "spike_index", "residual", "prefix"],
)


# Parallelism helpers
def _subtraction_batch(args):
    return subtraction_batch(
        *args,
        _subtraction_batch.device,
        _subtraction_batch.denoiser,
        _subtraction_batch.detector,
        _subtraction_batch.dn_detector,
    )


def _subtraction_batch_init(
    device,
    nn_detector_path,
    nn_channel_index,
    denoise_detect,
    do_nn_denoise,
    id_queue,
    denoiser_init_kwargs,
    denoiser_weights_path,
):
    """Thread/process initializer -- loads up neural nets"""
    rank = id_queue.get()

    torch.set_grad_enabled(False)
    if device.type == "cuda":
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
    _subtraction_batch.device = device

    time.sleep(rank)
    print(f"Worker {rank} init", flush=True)

    denoiser = None
    if do_nn_denoise:
        denoiser = denoise.SingleChanDenoiser(**denoiser_init_kwargs)
        if denoiser_weights_path is not None:
            denoiser.load(fname_model=denoiser_weights_path)
        else:
            denoiser.load()
        denoiser.to(device)
    _subtraction_batch.denoiser = denoiser

    detector = None
    if nn_detector_path:
        detector = detect.Detect(nn_channel_index)
        detector.load(nn_detector_path)
        detector.to(device)
    _subtraction_batch.detector = detector

    dn_detector = None
    if denoise_detect:
        dn_detector = detect.DenoiserDetect(denoiser)
        dn_detector.to(device)
    _subtraction_batch.dn_detector = dn_detector


def subtraction_batch(
    batch_data_folder,
    s_start,
    batch_len_samples,
    standardized_bin,
    thresholds,
    subtracted_tpca,
    denoised_tpca,
    trough_offset,
    dedup_channel_index,
    spike_length_samples,
    extract_channel_index,
    start_sample,
    end_sample,
    do_clean,
    save_residual,
    radial_parents,
    geom,
    do_enforce_decrease,
    probe,
    peak_sign,
    nsync,
    binary_dtype,
    dtype,
    extra_features,
    device,
    denoiser,
    detector,
    dn_detector,
):
    """Runs subtraction on a batch

    This function handles the logic of loading data from disk
    (padding it with a buffer where necessary), running the loop
    over thresholds for `detect_and_subtract`, handling spikes
    that were in the buffer, and applying the denoising pipeline.

    A note on buffer logic:
     - We load a buffer of twice the spike length.
     - The outer buffer of size spike length is to ensure that
       spikes inside the inner buffer of size spike length can be
       loaded
     - We subtract spikes inside the inner buffer in `detect_and_subtract`
       to ensure consistency of the residual across batches.

    Arguments
    ---------
    batch_data_folder : string
        Where temporary results are being stored
    s_start : int
        The batch's starting time in samples
    batch_len_samples : int
        The length of a batch in samples
    standardized_bin : int
        The path to the standardized binary file
    thresholds : list of int
        Voltage thresholds for subtraction
    tpca : sklearn PCA object or None
        A pre-trained temporal PCA (or None in which case no PCA
        is applied)
    trough_offset : int
        42 in practice, the alignment of the max channel's trough
        in the extracted waveforms
    dedup_channel_index : int array (num_channels, num_neighbors)
        Spatial neighbor structure for deduplication
    spike_length_samples : int
        121 in practice, temporal length of extracted waveforms
    extract_channel_index : int array (num_channels, extract_channels)
        Channel neighborhoods for extracted waveforms
    device : string or torch.device
    start_sample, end_sample : int
        Temporal boundary of the region of the recording being
        considered (in samples)
    radial_parents
        Helper data structure for enforce_decrease
    localization_kind : str
        How should we run localization?
    loc_workers : int
        on how many threads?
    geom : array
        The probe geometry
    denoiser, detector : torch nns or None
    probe : string or None

    Returns
    -------
    res : SubtractionBatchResult
    """
    # we use a double buffer: inner buffer of length spike_length,
    # outer buffer of length spike_length
    #  - detections are restricted to the inner buffer
    #  - outer buffer allows detections on the border of the inner
    #    buffer to be loaded
    #  - using the inner buffer allows for subtracted residuals to be
    #    consistent (more or less) across batches
    #  - only the spikes in the center (i.e. not in either buffer)
    #    will be returned to the caller
    buffer = 2 * spike_length_samples

    # load raw data with buffer
    s_end = min(end_sample, s_start + batch_len_samples)
    n_channels = len(dedup_channel_index)
    load_start = max(start_sample, s_start - buffer)
    load_end = min(end_sample, s_end + buffer)
    residual = read_data(
        standardized_bin,
        binary_dtype,
        load_start,
        load_end,
        n_channels,
        nsync,
        out_dtype=dtype,
    )
    prefix = f"{s_start:10d}_"

    # 0 padding if we were at the edge of the data
    pad_left = pad_right = 0
    if load_start == start_sample:
        pad_left = buffer
    if load_end == end_sample:
        pad_right = buffer - (end_sample - s_end)
    if pad_left != 0 or pad_right != 0:
        residual = np.pad(
            residual, [(pad_left, pad_right), (0, 0)], mode="edge"
        )

    # now, no matter where we were, the data has the following shape
    assert residual.shape == (2 * buffer + s_end - s_start, n_channels)

    # main subtraction loop
    subtracted_wfs = []
    spike_index = []
    for threshold in thresholds:
        subwfs, subpcs, residual, spind = detect_and_subtract(
            residual,
            threshold,
            radial_parents,
            subtracted_tpca,
            dedup_channel_index,
            extract_channel_index,
            peak_sign=peak_sign,
            detector=detector,
            denoiser=denoiser,
            denoiser_detector=dn_detector,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            device=device,
            do_enforce_decrease=do_enforce_decrease,
            probe=probe,
        )
        if len(spind):
            subtracted_wfs.append(subwfs)
            spike_index.append(spind)

    # at this point, trough times in the spike index are relative
    # to the full buffer of length 2 * spike length

    # strip buffer from residual and remove spikes in buffer
    residual_singlebuf = residual[spike_length_samples:-spike_length_samples]
    residual = residual[buffer:-buffer]
    if batch_data_folder is not None and save_residual:
        np.save(batch_data_folder / f"{prefix}res.npy", residual)

    # return early if there were no spikes
    if batch_data_folder is None and not spike_index:
        # this return is used by `train_pca` as an early exit
        return spike_index, subtracted_wfs, residual_singlebuf
    elif not spike_index:
        return SubtractionBatchResult(
            N_new=0,
            s_start=s_start,
            s_end=s_end,
            spike_index=None,
            residual=batch_data_folder / f"{prefix}res.npy",
            prefix=prefix,
        )

    subtracted_wfs = np.concatenate(subtracted_wfs, axis=0)
    spike_index = np.concatenate(spike_index, axis=0)

    # sort so time increases
    sort = np.argsort(spike_index[:, 0])
    subtracted_wfs = subtracted_wfs[sort]
    spike_index = spike_index[sort]

    # get rid of spikes in the buffer
    # also, get rid of spikes too close to the beginning/end
    # of the recording if we are in the first or last batch
    spike_time_min = 0
    if s_start == start_sample:
        spike_time_min = trough_offset
    spike_time_max = s_end - s_start
    if load_end == end_sample:
        spike_time_max -= spike_length_samples - trough_offset

    minix = np.searchsorted(spike_index[:, 0], spike_time_min, side="left")
    maxix = np.searchsorted(spike_index[:, 0], spike_time_max, side="right")
    spike_index = spike_index[minix:maxix]
    subtracted_wfs = subtracted_wfs[minix:maxix]

    # if caller passes None for the output folder, just return
    # the results now (eg this is used by train_pca)
    if batch_data_folder is None:
        return spike_index, subtracted_wfs, residual_singlebuf

    # get cleaned waveforms
    cleaned_wfs = denoised_wfs = None
    if not spike_index.size:
        cleaned_wfs = denoised_wfs = np.empty_like(subtracted_wfs)
    if do_clean:
        cleaned_wfs = read_waveforms_in_memory(
            residual_singlebuf,
            spike_index,
            spike_length_samples,
            extract_channel_index,
            trough_offset=trough_offset,
            buffer=spike_length_samples,
        )
        cleaned_wfs += subtracted_wfs
        denoised_wfs = full_denoising(
            cleaned_wfs,
            spike_index[:, 1],
            extract_channel_index,
            radial_parents,
            do_enforce_decrease=do_enforce_decrease,
            probe=probe,
            # tpca=subtracted_tpca,
            tpca=denoised_tpca,
            device=device,
            denoiser=denoiser,
        )

    # times relative to batch start
    # recall, these times were aligned to the double buffer, so we don't
    # need to adjust them according to the buffer at all.
    spike_index[:, 0] += s_start

    # save the results to disk to avoid memory issues
    N_new = len(spike_index)
    np.save(batch_data_folder / f"{prefix}si.npy", spike_index)

    # compute and save features
    for f in extra_features:
        np.save(
            batch_data_folder / f"{prefix}{f.name}.npy",
            f.transform(
                spike_index[:, 1],
                subtracted_wfs=subtracted_wfs,
                cleaned_wfs=cleaned_wfs,
                denoised_wfs=denoised_wfs,
            ),
        )

    res = SubtractionBatchResult(
        N_new=N_new,
        s_start=s_start,
        s_end=s_end,
        spike_index=batch_data_folder / f"{prefix}si.npy",
        residual=batch_data_folder / f"{prefix}res.npy",
        prefix=prefix,
    )

    return res


# -- temporal PCA


def train_featurizers(
    standardized_bin,
    spike_length_samples,
    extract_channel_index,
    geom,
    radial_parents,
    len_recording_samples,
    sampling_rate,
    dedup_channel_index,
    thresholds,
    nn_detector_path,
    denoise_detect,
    nn_channel_index,
    extra_features=None,
    subtracted_tpca=None,
    peak_sign="neg",
    do_nn_denoise=True,
    do_enforce_decrease=True,
    probe=None,
    n_sec_pca=10,
    nsync=0,
    random_seed=0,
    device="cpu",
    denoiser_init_kwargs={},
    denoiser_weights_path=None,
    trough_offset=42,
    binary_dtype=np.float32,
    dtype=np.float32,
):
    """Pre-train temporal PCA

    Extracts several random seconds of data by subtraction
    with no PCA, and trains a temporal PCA on the resulting
    waveforms.

    This same function is used to fit the subtraction TPCA and the
    collision-cleaned TPCA.
    """
    n_seconds = len_recording_samples // sampling_rate
    starts = sampling_rate * np.random.default_rng(random_seed).choice(
        n_seconds, size=min(n_sec_pca, n_seconds), replace=False
    )

    denoiser = None
    if do_nn_denoise:
        denoiser = denoise.SingleChanDenoiser(**denoiser_init_kwargs)
        if denoiser_weights_path is not None:
            denoiser.load(fname_model=denoiser_weights_path)
        else:
            denoiser.load()
        denoiser.to(device)

    detector = None
    if nn_detector_path:
        detector = detect.Detect(nn_channel_index)
        detector.load(nn_detector_path)
        detector.to(device)

    dn_detector = None
    if denoise_detect:
        assert do_nn_denoise
        dn_detector = detect.DenoiserDetect(denoiser)
        dn_detector.to(device)

    # do a mini-subtraction with no PCA, just NN denoise and enforce_decrease
    spike_indices = []
    waveforms = []
    residuals = []
    for s_start in tqdm(starts, "PCA training subtraction"):
        spind, wfs, residual_singlebuf = subtraction_batch(
            batch_data_folder=None,
            s_start=s_start,
            batch_len_samples=sampling_rate,
            standardized_bin=standardized_bin,
            thresholds=thresholds,
            subtracted_tpca=subtracted_tpca,
            denoised_tpca=None,
            trough_offset=trough_offset,
            dedup_channel_index=dedup_channel_index,
            spike_length_samples=spike_length_samples,
            extract_channel_index=extract_channel_index,
            start_sample=0,
            end_sample=len_recording_samples,
            do_clean=False,
            save_residual=False,
            radial_parents=radial_parents,
            geom=geom,
            do_enforce_decrease=do_enforce_decrease,
            probe=probe,
            peak_sign=peak_sign,
            nsync=nsync,
            binary_dtype=binary_dtype,
            dtype=dtype,
            extra_features=[],
            device=device,
            denoiser=denoiser,
            detector=detector,
            dn_detector=dn_detector,
        )
        spike_indices.append(spind)
        waveforms.append(wfs)
        residuals.append(residual_singlebuf)

    try:
        # this can raise value error
        spike_index = np.concatenate(spike_indices, axis=0)
        waveforms = np.concatenate(waveforms, axis=0)

        # but we can also be here...
        if waveforms.size == 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"No waveforms found in the whole {n_sec_pca} training "
            "batches for TPCA, so we could not train it. Maybe you "
            "can increase n_sec_pca, but also maybe there are data "
            "or threshold issues?"
        )

    N, T, C = waveforms.shape
    print("Fitting PCA on", N, "waveforms from mini-subtraction")

    # get subtracted or collision-cleaned waveforms
    if subtracted_tpca is None:
        # we're fitting TPCA to subtracted wfs so we won't need denoised wfs
        cleaned_waveforms = denoised_waveforms = None
    else:
        # otherwise, we are fitting featurizers, so compute
        # denoised versions
        cleaned_waveforms = np.concatenate(
            [
                read_waveforms_in_memory(
                    res,
                    spind,
                    spike_length_samples,
                    extract_channel_index,
                    trough_offset=trough_offset,
                    buffer=spike_length_samples,
                )
                for res, spind in zip(residuals, spike_indices)
            ],
            axis=0,
        )
        cleaned_waveforms += waveforms
        denoised_waveforms = full_denoising(
            cleaned_waveforms,
            spike_index[:, 1],
            extract_channel_index,
            radial_parents,
            do_enforce_decrease=do_enforce_decrease,
            probe=probe,
            tpca=None,
            device=device,
            denoiser=denoiser,
        )

    # train extra featurizers if necessary
    extra_features = [] if extra_features is None else extra_features
    if not any(f.needs_fit for f in extra_features):
        return

    for f in extra_features:
        f.fit(
            spike_index[:, 1],
            subtracted_wfs=waveforms,
            cleaned_wfs=cleaned_waveforms,
            denoised_wfs=denoised_waveforms,
        )


# -- denoising / detection helpers


def detect_and_subtract(
    raw,
    threshold,
    radial_parents,
    tpca,
    dedup_channel_index,
    extract_channel_index,
    *,
    peak_sign="neg",
    detector=None,
    denoiser=None,
    denoiser_detector=None,
    nn_switch_threshold=4,
    trough_offset=42,
    spike_length_samples=121,
    device="cpu",
    do_enforce_decrease=True,
    probe=None,
):
    """Detect and subtract

    For a fixed voltage threshold, detect spikes, denoise them,
    and subtract them from the recording.

    This function is the core of the subtraction routine.

    Returns
    -------
    waveforms, subtracted_raw, spike_index
    """
    device = torch.device(device)

    kwargs = dict(nn_detector=None, nn_denoiser=None, denoiser_detector=None)
    kwargs["denoiser_detector"] = denoiser_detector
    if detector is not None and threshold <= nn_switch_threshold:
        kwargs["nn_detector"] = detector
        kwargs["nn_denoiser"] = denoiser

    start = spike_length_samples
    end = -spike_length_samples
    if denoiser_detector is not None:
        start = start - 42
        end = end + 79

    # the full buffer has length 2 * spike len on both sides,
    # but this spike index only contains the spikes inside
    # the inner buffer of length spike len
    # times are relative to the *inner* buffer
    spike_index = detect.detect_and_deduplicate(
        raw[start:end].copy(),
        threshold,
        dedup_channel_index,
        buffer_size=spike_length_samples,
        device=device,
        peak_sign=peak_sign,
        spike_length_samples=spike_length_samples,
        **kwargs,
    )
    if not spike_index.size:
        return np.empty(0), np.empty(0), raw, np.empty(0)

    # -- read waveforms
    padded_raw = np.pad(raw, [(0, 0), (0, 1)], constant_values=np.nan)
    # get times relative to trough + buffer
    # currently, times are trough times relative to spike_length_samples,
    # but they also *start* there
    # thus, they are actually relative to the full buffer
    # of length 2 * spike_length_samples
    time_range = np.arange(
        2 * spike_length_samples - trough_offset,
        3 * spike_length_samples - trough_offset,
    )
    time_ix = spike_index[:, 0, None] + time_range[None, :]
    chan_ix = extract_channel_index[spike_index[:, 1]]
    waveforms = padded_raw[time_ix[:, :, None], chan_ix[:, None, :]]

    # -- denoising
    waveforms, tpca_proj = full_denoising(
        waveforms,
        spike_index[:, 1],
        extract_channel_index,
        radial_parents,
        do_enforce_decrease=do_enforce_decrease,
        probe=probe,
        tpca=tpca,
        device=device,
        denoiser=denoiser,
        return_tpca_embedding=True,
    )

    # -- the actual subtraction
    # have to use subtract.at since -= will only subtract once in the overlaps,
    # subtract.at will subtract multiple times where waveforms overlap
    np.subtract.at(
        padded_raw,
        (time_ix[:, :, None], chan_ix[:, None, :]),
        waveforms,
    )
    # remove the NaN padding
    subtracted_raw = padded_raw[:, :-1]

    return waveforms, tpca_proj, subtracted_raw, spike_index


@torch.no_grad()
def full_denoising(
    waveforms,
    maxchans,
    extract_channel_index,
    radial_parents=None,
    do_enforce_decrease=True,
    probe=None,
    tpca=None,
    device=None,
    denoiser=None,
    batch_size=1024,
    align=False,
    return_tpca_embedding=False,
):
    """Denoising pipeline: neural net denoise, temporal PCA, enforce_decrease"""
    num_channels = len(extract_channel_index)
    N, T, C = waveforms.shape
    assert not align  # still working on that

    # in new pipeline, some channels are off the edge of the probe
    # those are filled with NaNs, which will blow up PCA. so, here
    # we grab just the non-NaN channels.

    in_probe_channel_index = extract_channel_index < num_channels
    in_probe_index = in_probe_channel_index[maxchans]
    waveforms = waveforms.transpose(0, 2, 1)
    wfs_in_probe = waveforms[in_probe_index]

    # Apply NN denoiser (skip if None) #doesn't matter if wf on channels or everywhere
    if denoiser is not None:
        results = []
        for bs in range(0, wfs_in_probe.shape[0], batch_size):
            be = min(bs + batch_size, N * C)
            results.append(
                denoiser(
                    torch.as_tensor(
                        wfs_in_probe[bs:be], device=device, dtype=torch.float
                    )
                )
                .cpu()
                .numpy()
            )
        wfs_in_probe = np.concatenate(results, axis=0)
        del results

    # everyone to numpy now, if we were torch
    if torch.is_tensor(waveforms):
        waveforms = np.array(waveforms.cpu())

    # Temporal PCA while we are still transposed
    if tpca is not None:
        wfs_in_probe = tpca.inverse_transform(tpca.transform(wfs_in_probe))

    # back to original shape
    waveforms[in_probe_index] = wfs_in_probe
    waveforms = waveforms.transpose(0, 2, 1)

    # enforce decrease
    if do_enforce_decrease:
        if radial_parents is None and probe is not None:
            rel_maxchans = maxchans - extract_channel_index[maxchans, 0]
            if probe == "np1":
                for i in range(N):
                    denoise.enforce_decrease_np1(
                        waveforms[i], max_chan=rel_maxchans[i], in_place=True
                    )
            elif probe == "np2":
                for i in range(N):
                    denoise.enforce_decrease(
                        waveforms[i], max_chan=rel_maxchans[i], in_place=True
                    )
            else:
                assert False
        elif radial_parents is not None:
            denoise.enforce_decrease_shells(
                waveforms, maxchans, radial_parents, in_place=True
            )
        else:
            # no enforce decrease
            pass

    if return_tpca_embedding and tpca is not None:
        tpca_embeddings = np.empty(
            (N, C, tpca.n_components), dtype=waveforms.dtype
        )
        # run tpca only on channels that matter!
        tpca_embeddings[in_probe_index] = tpca.transform(
            waveforms.transpose(0, 2, 1)[in_probe_index]
        )
        return waveforms, tpca_embeddings.transpose(0, 2, 1)
    elif return_tpca_embedding:
        return waveforms, None

    return waveforms


# -- HDF5 initialization / resuming old job logic


@contextlib.contextmanager
def get_output_h5(
    out_h5,
    start_sample,
    end_sample,
    geom,
    extract_channel_index,
    sampling_rate,
    extra_features,
    overwrite=False,
    chunk_len=1024,
    dtype=np.float32,
):
    h5_exists = Path(out_h5).exists()
    last_sample = 0
    if h5_exists and not overwrite:
        output_h5 = h5py.File(out_h5, "r+")
        h5_exists = True
        if len(output_h5["spike_index"]) > 0:
            last_sample = output_h5["spike_index"][-1, 0]
    else:
        if overwrite and h5_exists:
            print("Overwriting previous results, if any.")
            Path(out_h5).unlink()
        output_h5 = h5py.File(out_h5, "w")

        # initialize datasets
        output_h5.create_dataset("fs", data=sampling_rate)
        output_h5.create_dataset("geom", data=geom)
        output_h5.create_dataset("start_sample", data=start_sample)
        output_h5.create_dataset("end_sample", data=end_sample)
        output_h5.create_dataset("channel_index", data=extract_channel_index)

        # resizable datasets so we don't fill up space
        output_h5.create_dataset(
            "spike_index",
            shape=(0, 2),
            chunks=(chunk_len, 2),
            maxshape=(None, 2),
            dtype=np.int64,
        )

        for f in extra_features:
            output_h5.create_dataset(
                f.name,
                shape=(0, *f.out_shape),
                chunks=(chunk_len, *f.out_shape),
                maxshape=(None, *f.out_shape),
                dtype=f.dtype,
            )
            f.to_h5(output_h5)

    done_percent = (
        100 * (last_sample - start_sample) / (end_sample - start_sample)
    )
    if h5_exists and not overwrite:
        print(f"Resuming previous job, which was {done_percent:.0f}% done")
    elif h5_exists and overwrite:
        last_sample = 0
    else:
        print("No previous output found, starting from scratch.")

    # try/finally ensures we close `output_h5` if job is interrupted
    # docs.python.org/3/library/contextlib.html#contextlib.contextmanager
    try:
        yield output_h5, last_sample
    finally:
        output_h5.close()


def tpca_from_h5(h5):
    tpca = None
    if "tpca_mean" in h5:
        tpca_mean = h5["tpca_mean"][:]
        tpca_components = h5["tpca_components"][:]
        if (tpca_mean == 0).all():
            print("H5 exists but TPCA params == 0, re-fit.")
        else:
            tpca = PCA(tpca_components.shape[0])
            tpca.mean_ = tpca_mean
            tpca.components_ = tpca_components
            print("Loaded TPCA from h5")
    return tpca


# -- data loading helpers


def read_geom_from_meta(bin_file):
    meta = Path(bin_file.parent) / (bin_file.stem + ".meta")
    if not meta.exists():
        raise ValueError("Expected", meta, "to exist.")
    header = _geometry_from_meta(read_meta_data(meta))
    geom = np.c_[header["x"], header["y"]]
    return geom


def subtract_and_localize_numpy(
    raw,
    geom,
    extract_radius=200.0,
    loc_radius=100.0,
    dedup_spatial_radius=70.0,
    thresholds=[12, 10, 8, 6, 5],
    radial_parents=None,
    tpca=None,
    device=None,
    probe="np1",
    trough_offset=42,
    spike_length_samples=121,
    loc_workers=1,
):
    # we will run in this buffer and return it after subtraction
    residual = raw.copy()

    # probe geometry helper structures
    dedup_channel_index = make_channel_index(
        geom, dedup_spatial_radius, steps=2
    )
    extract_channel_index = make_channel_index(
        geom, extract_radius, distance_order=False
    )
    # use radius-based localization neighborhood
    loc_n_chans = None

    if radial_parents is None:
        # this can be slow to compute, so could be worth pre-computing it
        radial_parents = denoise.make_radial_order_parents(
            geom, extract_channel_index, n_jumps_per_growth=1, n_jumps_parent=3
        )

    # load neural nets
    if device is None:
        device = "cuda" if torch.cuda.is_available else "cpu"
    device = torch.device(device)
    denoiser = denoise.SingleChanDenoiser()
    denoiser.load()
    denoiser.to(device)
    dn_detector = detect.DenoiserDetect(denoiser)
    dn_detector.to(device)

    subtracted_wfs = []
    spike_index = []
    for threshold in thresholds:
        subwfs, tcpca_proj, residual, spind = detect_and_subtract(
            residual,
            threshold,
            radial_parents,
            tpca,
            dedup_channel_index,
            extract_channel_index,
            detector=None,
            denoiser=denoiser,
            denoiser_detector=dn_detector,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            device=device,
            probe=probe,
        )
        _logger.debug(
            f"Detected and subtracted {spind.shape[0]} spikes "
            "with threshold {threshold} on {thresholds}"
        )
        if len(spind):
            subtracted_wfs.append(subwfs)
            spike_index.append(spind)

    subtracted_wfs = np.concatenate(subtracted_wfs, axis=0)
    spike_index = np.concatenate(spike_index, axis=0)
    _logger.debug(
        f"Detected and subtracted {spike_index.shape[0]} spikes Total"
    )

    # sort so time increases
    sort = np.argsort(spike_index[:, 0])
    subtracted_wfs = subtracted_wfs[sort]
    spike_index = spike_index[sort]

    _logger.debug(f"Denoising waveforms...")
    # "collision-cleaned" wfs

    cleaned_wfs = read_waveforms_in_memory(
        residual,
        spike_index,
        spike_length_samples,
        extract_channel_index,
        trough_offset=trough_offset,
        buffer=0,
    )

    cleaned_wfs = full_denoising(
        cleaned_wfs + subtracted_wfs,
        spike_index[:, 1],
        extract_channel_index,
        radial_parents,
        probe=probe,
        tpca=tpca,
        device=device,
        denoiser=denoiser,
    )

    # localize
    _logger.debug(f"Localisation...")
    locptps = cleaned_wfs.ptp(1)
    xs, ys, z_rels, z_abss, alphas = localize_index.localize_ptps_index(
        locptps,
        geom,
        spike_index[:, 1],
        extract_channel_index,
        n_channels=loc_n_chans,
        radius=loc_radius,
        n_workers=loc_workers,
        pbar=False,
    )
    df_localisation = pd.DataFrame(
        data=np.c_[
            spike_index[:, 0] + spike_length_samples * 2,
            spike_index[:, 1],
            xs,
            ys,
            z_abss,
            alphas,
        ],
        columns=["sample", "trace", "x", "y", "z", "alpha"],
    )
    return df_localisation, cleaned_wfs


# -- utils


class timer:
    def __init__(self, name="timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


class NoKeyboardInterrupt:
    """A context manager that we use to avoid ending up in invalid states."""

    def handler(self, *sig):
        if self.sig:
            signal.signal(signal.SIGINT, self.old_handler)
            sig, self.sig = self.sig, None
            self.old_handler(*sig)
        self.sig = sig

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
        self.sig = None

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.sig:
            self.old_handler(*self.sig)


noint = NoKeyboardInterrupt()
