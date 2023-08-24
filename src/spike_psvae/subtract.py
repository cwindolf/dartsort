import contextlib
import gc
import logging
import multiprocessing
import time
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import spikeinterface.core as sc
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from . import chunk_features, denoise, detect, localize_index
from .multiprocessing_utils import MockQueue, ProcessPoolExecutor, get_pool
from .py_utils import noint, timer
from .spikeio import read_waveforms_in_memory
from .waveform_utils import make_channel_index, make_contiguous_channel_index

try:
    from dartsort.transform.enforce_decrease import EnforceDecrease
except ImportError:
    pass

_logger = logging.getLogger(__name__)


default_extra_feats = [
    chunk_features.MaxPTP,
    chunk_features.TroughDepth,
    chunk_features.PeakHeight,
]


def subtraction(
    recording,
    out_folder,
    out_filename="subtraction.h5",
    # should we start over?
    overwrite=False,
    # waveform args
    trough_offset=42,
    spike_length_samples=121,
    # tpca args
    tpca_rank=8,
    n_sec_pca=40,
    pca_t_start=0,
    pca_t_end=None,
    # time / input binary details
    n_sec_chunk=1,
    # detection
    thresholds=[12, 10, 8, 6, 5, 4],
    peak_sign="both",
    nn_detect=False,
    denoise_detect=False,
    do_nn_denoise=True,
    residnorm_decrease=np.sqrt(10.0),
    # waveform extraction channels
    neighborhood_kind="circle",
    extract_box_radius=200,
    extract_firstchan_n_channels=40,
    box_norm_p=np.inf,
    dedup_spatial_radius=70,
    enforce_decrease_kind="radial",
    do_phaseshift=False,
    ci_graph_all_maxCH_uniq=None,
    maxCH_neighbor=None,
    # what to save?
    save_residual=False,
    save_subtracted_waveforms=False,
    save_cleaned_waveforms=False,
    save_denoised_waveforms=False,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=True,
    save_denoised_tpca_projs=False,
    save_denoised_ptp_vectors=True,
    # we will save spatiotemporal PCA embeds if this is >0
    save_cleaned_pca_projs_on_n_channels=None,
    save_cleaned_pca_projs_rank=5,
    # localization args
    # set this to None or "none" to turn off
    localization_model="pointsource",
    localization_kind="logbarrier",
    localize_radius=100,
    localize_firstchan_n_channels=20,
    loc_workers=4,
    loc_feature="ptp",
    loc_ptp_precision_decimals=None,
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
    # validate and process args
    if neighborhood_kind not in ("firstchan", "box", "circle"):
        raise ValueError(
            "Neighborhood kind", neighborhood_kind, "not understood."
        )
    if enforce_decrease_kind not in ("columns", "radial", "none", "new"):
        raise ValueError(
            "Enforce decrease method", enforce_decrease_kind, "not understood."
        )
    if peak_sign not in ("neg", "both"):
        raise ValueError("peak_sign", peak_sign, "not understood.")

    if neighborhood_kind == "circle":
        neighborhood_kind = "box"
        box_norm_p = 2

    batch_len_samples = int(
        np.floor(n_sec_chunk * recording.get_sampling_frequency())
    )

    print(device)

    # prepare output dir
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True)
    batch_data_folder = out_folder / f"subtraction_batches"
    batch_data_folder.mkdir(exist_ok=True)
    assert out_filename.endswith(".h5"), "Nice try."
    out_h5 = out_folder / out_filename
    if save_residual:
        residual_bin = out_folder / f"residual.bin"
    try:
        # this is to check if another program is using our h5, in which
        # case we should crash early rather than late.
        if out_h5.exists():
            with h5py.File(out_h5, "r+") as _:
                pass
            del _
            gc.collect()
    except BlockingIOError as e:
        raise ValueError(
            f"Output HDF5 {out_h5} is currently in use by another program. "
            "Maybe a Jupyter notebook that's still running?"
        ) from e

    # pick torch device if it's not supplied
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda._lazy_init()
    else:
        device = torch.device(device)
    torch.set_grad_enabled(False)

    # figure out if we will use a NN detector, and if so which
    nn_detector_path = None
    if nn_detect:
        raise NotImplementedError(
            "Need to find out how to get Neuropixels version from SI."
        )
        # nn_detector_path = (
        #     Path(__file__).parent.parent / f"pretrained/detect_{probe}.pt"
        # )
        # print("Using pretrained detector for", probe, "from", nn_detector_path)
        detection_kind = "voltage->NN"
    elif denoise_detect:
        print("Using denoising NN detection")
        detection_kind = "denoised PTP"
    else:
        print("Using voltage detection")
        detection_kind = "voltage"

    print(
        f"Running subtraction on: {recording}. "
        f"Using {detection_kind} detection with thresholds: {thresholds}."
    )

    # compute helper data structures
    # channel indexes for extraction, NN detection, deduplication
    geom = recording.get_channel_locations()
    print(f"{geom.shape=}")
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
            recording.get_num_channels(),
            n_neighbors=extract_firstchan_n_channels,
        )
    else:
        assert False

    # handle ChunkFeature pipeline
    do_clean = (
        save_denoised_tpca_projs
        or save_denoised_ptp_vectors
        or localization_kind
        in (
            "original",
            "logbarrier",
        )
    )
    if extra_features == "default":
        feat_wfs = "denoised" if do_clean else "subtracted"
        extra_features = [
            F(which_waveforms=feat_wfs) for F in default_extra_feats
        ]
    if save_denoised_ptp_vectors:
        extra_features += [
            chunk_features.PTPVector(which_waveforms="denoised")
        ]
    if save_cleaned_pca_projs_on_n_channels:
        extra_features += [
            chunk_features.STPCA(
                channel_index=extract_channel_index,
                which_waveforms="cleaned",
                rank=save_cleaned_pca_projs_rank,
                n_channels=save_cleaned_pca_projs_on_n_channels,
                geom=geom,
            )
        ]

    # helper data structure for radial enforce decrease
    do_enforce_decrease = True
    radial_parents = None
    enfdec = None
    if enforce_decrease_kind == "radial":
        radial_parents = denoise.make_radial_order_parents(
            geom, extract_channel_index, n_jumps_per_growth=1, n_jumps_parent=3
        )
    elif enforce_decrease_kind == "columns":
        pass
    elif enforce_decrease_kind == "new":
        enfdec = EnforceDecrease(
            channel_index=extract_channel_index, geom=geom
        )
    else:
        print("Skipping enforce decrease.")
        do_enforce_decrease = False

    maxCH_neighbor = None
    ci_graph_all_maxCH_uniq = None
    if do_phaseshift == True:
        ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(
            extract_channel_index, geom, device
        )
        ci_graph_all_maxCH_uniq = denoise.make_ci_graph_all_maxCH(
            ci_graph_on_probe, maxCH_neighbor, device
        )
    else:
        print("No phase-shift.")
        do_phaseshift = False

    # check localization arg
    if localization_model not in ("pointsource", "CoM", "dipole"):
        raise ValueError(f"Unknown localization model: {localization_model}")
    if localization_kind in ("original", "logbarrier"):
        print("Using", localization_kind, "localization")
        if not isinstance(loc_feature, (list, tuple)):
            loc_feature = (loc_feature,)
        for lf in loc_feature:
            extra_features += [
                chunk_features.Localization(
                    geom,
                    extract_channel_index,
                    loc_n_chans=localize_firstchan_n_channels
                    if neighborhood_kind == "firstchan"
                    else None,
                    loc_radius=localize_radius
                    if neighborhood_kind != "firstchan"
                    else None,
                    localization_kind=localization_kind,
                    localization_model=localization_model,
                    feature=lf,
                    ptp_precision_decimals=loc_ptp_precision_decimals,
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
                chunk_features.Waveform(
                    which_waveforms=kind,
                )
            ]

    # see if we are asked to save tpca projs for
    # collision-cleaned or denoised waveforms
    subtracted_tpca_feat = chunk_features.TPCA(
        tpca_rank,
        extract_channel_index,
        which_waveforms="subtracted",
        random_state=random_seed,
    )
    if save_subtracted_tpca_projs:
        extra_features += [subtracted_tpca_feat]
    if save_cleaned_tpca_projs:
        extra_features += [
            chunk_features.TPCA(
                tpca_rank,
                extract_channel_index,
                which_waveforms="cleaned",
                random_state=random_seed,
            )
        ]
    fit_feats = []
    if do_clean:
        denoised_tpca_feat = chunk_features.TPCA(
            tpca_rank,
            extract_channel_index,
            which_waveforms="denoised",
            random_state=random_seed,
        )
        if save_denoised_tpca_projs:
            extra_features += [denoised_tpca_feat]
        else:
            fit_feats += [denoised_tpca_feat]

    # try to load feats from h5
    if not overwrite and out_h5.exists():
        with h5py.File(out_h5, "r") as output_h5:
            for feat in [subtracted_tpca_feat] + fit_feats:
                feat.from_h5(output_h5)
        del output_h5
        gc.collect()

    # otherwise, train it
    # TODO: ideally would run this on another process,
    # because this is the only time the main thread uses
    # GPU, and we could avoid initializing torch runtime.
    if subtracted_tpca_feat.needs_fit:
        with timer("Training TPCA..."):
            train_featurizers(
                recording,
                extract_channel_index,
                geom,
                radial_parents,
                enfdec,
                dedup_channel_index,
                thresholds,
                nn_detector_path=nn_detector_path,
                denoise_detect=denoise_detect,
                nn_channel_index=nn_channel_index,
                extra_features=[subtracted_tpca_feat],
                subtracted_tpca=None,
                peak_sign=peak_sign,
                do_nn_denoise=do_nn_denoise,
                residnorm_decrease=residnorm_decrease,
                do_enforce_decrease=do_enforce_decrease,
                do_phaseshift=do_phaseshift,
                ci_graph_all_maxCH_uniq=ci_graph_all_maxCH_uniq,
                maxCH_neighbor=maxCH_neighbor,
                denoiser_init_kwargs=denoiser_init_kwargs,
                denoiser_weights_path=denoiser_weights_path,
                n_sec_pca=n_sec_pca,
                pca_t_start=pca_t_start,
                pca_t_end=pca_t_end,
                random_seed=random_seed,
                device="cpu",
                trough_offset=trough_offset,
                spike_length_samples=spike_length_samples,
                dtype=dtype,
            )

    # train featurizers
    if any(f.needs_fit for f in extra_features + fit_feats):
        # try to load old featurizers
        if not overwrite and out_h5.exists():
            with h5py.File(out_h5, "r") as output_h5:
                for f in extra_features + fit_feats:
                    if f.needs_fit:
                        f.from_h5(output_h5)

        # train any which couldn't load
        if any(f.needs_fit for f in extra_features + fit_feats):
            train_featurizers(
                recording,
                extract_channel_index,
                geom,
                radial_parents,
                enfdec,
                dedup_channel_index,
                thresholds,
                nn_detector_path,
                denoise_detect,
                nn_channel_index,
                subtracted_tpca=subtracted_tpca_feat,
                extra_features=extra_features + fit_feats,
                peak_sign=peak_sign,
                do_nn_denoise=do_nn_denoise,
                residnorm_decrease=residnorm_decrease,
                do_enforce_decrease=do_enforce_decrease,
                do_phaseshift=do_phaseshift,
                ci_graph_all_maxCH_uniq=ci_graph_all_maxCH_uniq,
                maxCH_neighbor=maxCH_neighbor,
                denoiser_init_kwargs=denoiser_init_kwargs,
                denoiser_weights_path=denoiser_weights_path,
                n_sec_pca=n_sec_pca,
                pca_t_start=pca_t_start,
                pca_t_end=pca_t_end,
                random_seed=random_seed,
                device="cpu",
                dtype=dtype,
                trough_offset=trough_offset,
                spike_length_samples=spike_length_samples,
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
                0,
                recording.get_num_samples(),
                batch_len_samples,
            )
        )
    )

    # -- initialize storage
    with get_output_h5(
        out_h5,
        recording,
        extract_channel_index,
        extra_features,
        fit_features=[subtracted_tpca_feat] + fit_feats,
        overwrite=overwrite,
        dtype=dtype,
    ) as (output_h5, last_sample):
        # residual binary file -- append if we're resuming
        if save_residual:
            residual_mode = "ab" if last_sample > 0 else "wb"
            residual = open(residual_bin, mode=residual_mode)

        extra_features = [ef.to("cpu") for ef in extra_features]

        # no-threading/multiprocessing execution for debugging if n_jobs == 0
        Executor, context = get_pool(n_jobs, cls=ProcessPoolExecutor)
        manager = context.Manager() if n_jobs > 1 else None
        id_queue = manager.Queue() if n_jobs > 1 else MockQueue()

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
                recording.to_dict(),
                extra_features,
                subtracted_tpca_feat,
                denoised_tpca_feat if do_clean else None,
                enfdec,
            ),
        ) as pool:
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

            if n_batches > 0:
                jobs = (
                    (
                        batch_data_folder,
                        batch_len_samples,
                        s_start,
                        thresholds,
                        dedup_channel_index,
                        trough_offset,
                        spike_length_samples,
                        extract_channel_index,
                        do_clean,
                        residnorm_decrease,
                        save_residual,
                        radial_parents,
                        geom,
                        do_enforce_decrease,
                        do_phaseshift,
                        ci_graph_all_maxCH_uniq,
                        maxCH_neighbor,
                        peak_sign,
                        dtype,
                    )
                    for batch_id, s_start in jobs
                )

                count = sum(
                    s < last_sample
                    for s in range(
                        0,
                        recording.get_num_samples(),
                        batch_len_samples,
                    )
                )

                # now run subtraction in parallel
                pbar = tqdm(
                    pool.map(_subtraction_batch, jobs),
                    total=n_batches,
                    desc="Batches",
                    smoothing=0.01,
                )
                for result in pbar:
                    with noint:
                        N_new = result.N_new

                        # write new residual
                        if save_residual:
                            np.load(result.residual).tofile(residual)
                            Path(result.residual).unlink()

                        if result.spike_index is None:
                            continue

                        # grow arrays as necessary and write results
                        if N_new > 0:
                            spike_index.resize(N + N_new, axis=0)
                            spike_index[N:] = np.load(result.spike_index)
                        if Path(result.spike_index).exists():
                            Path(result.spike_index).unlink()
                        for f, dset in zip(extra_features, feature_dsets):
                            fnpy = (
                                batch_data_folder
                                / f"{result.prefix}{f.name}.npy"
                            )
                            if N_new > 0:
                                dset.resize(N + N_new, axis=0)
                                dset[N:] = np.load(fnpy)
                            if Path(fnpy).exists():
                                Path(fnpy).unlink()
                        # update spike count
                        N += N_new

                    count += 1
                    pbar.set_description(
                        f"{n_sec_chunk}s/it [spk/it={N / count:0.1f}]"
                    )

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


def subtraction_binary(
    standardized_bin,
    *args,
    geom=None,
    t_start=0,
    t_end=None,
    sampling_rate=30_000,
    nsync=0,
    binary_dtype=np.float32,
    time_axis=0,
    **kwargs,
):
    """Wrapper around `subtraction` to provide the old binary file API"""
    # if no geometry is supplied, try to load it from meta file
    if geom is None:
        geom = read_geom_from_meta(standardized_bin)
        if geom is None:
            raise ValueError(
                "Either pass `geom` or put meta file in folder with binary."
            )
    n_channels = geom.shape[0]

    recording = sc.read_binary(
        standardized_bin,
        sampling_rate,
        n_channels,
        binary_dtype,
        time_axis=time_axis,
        is_filtered=True,
    )

    # set geometry
    recording.set_dummy_probe_from_locations(
        geom, shape_params=dict(radius=10)
    )

    if nsync > 0:
        recording = recording.channel_slice(
            channel_ids=recording.get_channel_ids()[:-nsync]
        )

    T_samples = recording.get_num_samples()
    T_sec = T_samples / recording.get_sampling_frequency()
    assert t_start >= 0 and (t_end is None or t_end <= T_sec)
    start_sample = int(np.floor(t_start * sampling_rate))
    end_sample = (
        T_samples if t_end is None else int(np.floor(t_end * sampling_rate))
    )
    if start_sample > 0 or end_sample < T_samples:
        recording = recording.frame_slice(
            start_frame=start_sample, end_frame=end_sample
        )

    return subtraction(recording, *args, **kwargs)


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
        _subtraction_batch.extra_features,
        _subtraction_batch.subtracted_tpca,
        _subtraction_batch.denoised_tpca,
        _subtraction_batch.recording,
        _subtraction_batch.device,
        _subtraction_batch.denoiser,
        _subtraction_batch.detector,
        _subtraction_batch.dn_detector,
        _subtraction_batch.enfdec,
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
    recording_dict,
    extra_features,
    subtracted_tpca,
    denoised_tpca,
    enfdec,
):
    """Thread/process initializer -- loads up neural nets"""
    rank = id_queue.get()

    torch.set_grad_enabled(False)
    if device.type == "cuda" and device.index is None:
        if not rank:
            print("num gpus:", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device = torch.device(
                "cuda", index=rank % torch.cuda.device_count()
            )
            print(
                f"Worker {rank} using GPU {rank % torch.cuda.device_count()} "
                f"out of {torch.cuda.device_count()} available."
            )
    elif device.type == "cuda" and device.index is not None and not rank:
        print(
            f"All workers will live on {device} since a specific GPU was chosen"
        )
    _subtraction_batch.device = device

    time.sleep(rank / 20)
    print(f"Worker {rank} init", flush=True)

    denoiser = None
    if do_nn_denoise:
        denoiser = denoise.SingleChanDenoiser(**denoiser_init_kwargs)
        if denoiser_weights_path is not None:
            denoiser.load(fname_model=denoiser_weights_path)
        else:
            denoiser.load()
        denoiser.requires_grad_(False)
        denoiser.to(device)
    _subtraction_batch.denoiser = denoiser

    detector = None
    if nn_detector_path:
        detector = detect.Detect(nn_channel_index)
        detector.load(nn_detector_path)
        detector.requires_grad_(False)
        detector.to(device)
    _subtraction_batch.detector = detector

    dn_detector = None
    if denoise_detect:
        dn_detector = detect.DenoiserDetect(denoiser)
        dn_detector.requires_grad_(False)
        dn_detector.to(device)
    _subtraction_batch.dn_detector = dn_detector

    _subtraction_batch.extra_features = [
        ef.to(device) for ef in extra_features
    ]
    _subtraction_batch.subtracted_tpca = subtracted_tpca.to(device)
    if denoised_tpca is not None:
        denoised_tpca = denoised_tpca.to(device)
    _subtraction_batch.denoised_tpca = denoised_tpca

    _subtraction_batch.enfdec = enfdec
    if enfdec is not None:
        _subtraction_batch.enfdec.to(device)

    # this is a hack to fix ibl streaming in parallel
    stack = [recording_dict]
    for d in stack:
        for k, v in d.items():
            if isinstance(v, dict):
                if (
                    "class" in v
                    and "IblStreamingRecordingExtractor" in v["class"]
                ):
                    v["kwargs"]["cache_folder"] = (
                        Path(v["kwargs"]["cache_folder"]) / f"cache{rank}"
                    )
                else:
                    stack.append(v)

    _subtraction_batch.recording = sc.BaseRecording.from_dict(recording_dict)


def subtraction_batch(
    batch_data_folder,
    batch_len_samples,
    s_start,
    thresholds,
    dedup_channel_index,
    trough_offset,
    spike_length_samples,
    extract_channel_index,
    do_clean,
    residnorm_decrease,
    save_residual,
    radial_parents,
    geom,
    do_enforce_decrease,
    do_phaseshift,
    ci_graph_all_maxCH_uniq,
    maxCH_neighbor,
    peak_sign,
    dtype,
    extra_features,
    subtracted_tpca,
    denoised_tpca,
    recording,
    device,
    denoiser,
    detector,
    dn_detector,
    enfdec,
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
    s_end = min(recording.get_num_samples(), s_start + batch_len_samples)
    n_channels = len(dedup_channel_index)
    load_start = max(0, s_start - buffer)
    load_end = min(recording.get_num_samples(), s_end + buffer)
    residual = recording.get_traces(start_frame=load_start, end_frame=load_end)
    residual = residual.astype(dtype)
    assert np.isfinite(residual).all()
    prefix = f"{s_start:010d}_"

    # 0 padding if we were at the edge of the data
    pad_left = pad_right = 0
    if load_start == 0:
        pad_left = buffer
    if load_end == recording.get_num_samples():
        pad_right = buffer - (recording.get_num_samples() - s_end)
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
            enfdec,
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
            do_phaseshift=do_phaseshift,
            ci_graph_all_maxCH_uniq=ci_graph_all_maxCH_uniq,
            maxCH_neighbor=maxCH_neighbor,
            geom=geom,
            residnorm_decrease=residnorm_decrease,
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
        np.save(batch_data_folder / f"{prefix}res.npy", residual.cpu().numpy())

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

    subtracted_wfs = torch.cat(subtracted_wfs, dim=0)
    spike_index = np.concatenate(spike_index, axis=0)

    # sort so time increases
    sort = np.argsort(spike_index[:, 0], kind="stable")
    subtracted_wfs = subtracted_wfs[sort]
    spike_index = spike_index[sort]

    # get rid of spikes in the buffer
    # also, get rid of spikes too close to the beginning/end
    # of the recording if we are in the first or last batch
    spike_time_min = 0
    if s_start == 0:
        spike_time_min = trough_offset
    spike_time_max = s_end - s_start
    if load_end == recording.get_num_samples():
        spike_time_max -= spike_length_samples - trough_offset

    minix = np.searchsorted(spike_index[:, 0], spike_time_min, side="left")
    maxix = np.searchsorted(spike_index[:, 0], spike_time_max, side="right")
    spike_index = spike_index[minix:maxix]
    subtracted_wfs = subtracted_wfs[minix:maxix]

    # if caller passes None for the output folder, just return
    # the results now (eg this is used by train_pca)
    if batch_data_folder is None:
        return spike_index, subtracted_wfs, residual_singlebuf

    if not np.prod(spike_index.shape):
        return SubtractionBatchResult(
            N_new=0,
            s_start=s_start,
            s_end=s_end,
            spike_index=None,
            residual=batch_data_folder / f"{prefix}res.npy",
            prefix=prefix,
        )

    # compute and save features for subtracted wfs
    for f in extra_features:
        feat = f.transform(
            spike_index[:, 1],
            subtracted_wfs=subtracted_wfs,
        )
        if feat is not None:
            if torch.is_tensor(feat):
                feat = feat.cpu().numpy()
            np.save(
                batch_data_folder / f"{prefix}{f.name}.npy",
                feat,
            )

    # get cleaned waveforms
    cleaned_wfs = denoised_wfs = None
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
        del subtracted_wfs

        # compute and save features for subtracted wfs
        for f in extra_features:
            feat = f.transform(
                spike_index[:, 1],
                cleaned_wfs=cleaned_wfs,
                denoised_wfs=None,
            )
            if feat is not None:
                if torch.is_tensor(feat):
                    feat = feat.cpu().numpy()
                np.save(
                    batch_data_folder / f"{prefix}{f.name}.npy",
                    feat,
                )

        denoised_wfs = full_denoising(
            cleaned_wfs,
            spike_index[:, 1],
            extract_channel_index,
            radial_parents,
            enfdec,
            do_enforce_decrease=do_enforce_decrease,
            do_phaseshift=do_phaseshift,
            ci_graph_all_maxCH_uniq=ci_graph_all_maxCH_uniq,
            maxCH_neighbor=maxCH_neighbor,
            geom=geom,
            # tpca=subtracted_tpca,
            tpca=denoised_tpca,
            device=device,
            denoiser=denoiser,
        )
        del cleaned_wfs

        # compute and save features for subtracted wfs
        for f in extra_features:
            feat = f.transform(
                spike_index[:, 1],
                denoised_wfs=denoised_wfs,
            )
            if feat is not None:
                if torch.is_tensor(feat):
                    feat = feat.cpu().numpy()
                np.save(
                    batch_data_folder / f"{prefix}{f.name}.npy",
                    feat,
                )

    # times relative to batch start
    # recall, these times were aligned to the double buffer, so we don't
    # need to adjust them according to the buffer at all.
    spike_index[:, 0] += s_start

    # save the results to disk to avoid memory issues
    N_new = len(spike_index)
    np.save(batch_data_folder / f"{prefix}si.npy", spike_index)

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
    recording,
    extract_channel_index,
    geom,
    radial_parents,
    enfdec,
    dedup_channel_index,
    thresholds,
    nn_detector_path=None,
    denoise_detect=False,
    nn_channel_index=None,
    extra_features=None,
    subtracted_tpca=None,
    peak_sign="neg",
    do_nn_denoise=True,
    residnorm_decrease=False,
    do_enforce_decrease=True,
    do_phaseshift=False,
    ci_graph_all_maxCH_uniq=None,
    maxCH_neighbor=None,
    n_sec_pca=10,
    pca_t_start=0,
    pca_t_end=None,
    random_seed=0,
    device="cpu",
    denoiser_init_kwargs={},
    denoiser_weights_path=None,
    trough_offset=42,
    spike_length_samples=121,
    dtype=np.float32,
):
    """Pre-train temporal PCA

    Extracts several random seconds of data by subtraction
    with no PCA, and trains a temporal PCA on the resulting
    waveforms.

    This same function is used to fit the subtraction TPCA and the
    collision-cleaned TPCA.
    """
    fs = int(np.floor(recording.get_sampling_frequency()))
    n_seconds = recording.get_num_samples() // fs
    second_starts = fs * np.arange(n_seconds)
    second_starts = second_starts[second_starts >= fs * pca_t_start]
    if pca_t_end is not None:
        second_starts = second_starts[second_starts < fs * pca_t_end]
    starts = np.random.default_rng(random_seed).choice(
        second_starts, size=min(n_sec_pca, len(second_starts)), replace=False
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
    n_empty_batches = 0
    print("zzz")

    for s_start in tqdm(starts, "PCA training subtraction"):
        spind, wfs, residual_singlebuf = subtraction_batch(
            batch_data_folder=None,
            batch_len_samples=fs,
            s_start=s_start,
            thresholds=thresholds,
            dedup_channel_index=dedup_channel_index,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            extract_channel_index=extract_channel_index,
            do_clean=False,
            residnorm_decrease=residnorm_decrease,
            save_residual=False,
            radial_parents=radial_parents,
            geom=geom,
            do_enforce_decrease=do_enforce_decrease,
            do_phaseshift=do_phaseshift,
            ci_graph_all_maxCH_uniq=ci_graph_all_maxCH_uniq,
            maxCH_neighbor=maxCH_neighbor,
            peak_sign=peak_sign,
            dtype=dtype,
            extra_features=[],
            subtracted_tpca=subtracted_tpca.to(device)
            if subtracted_tpca is not None
            else None,
            denoised_tpca=None,
            recording=recording,
            device=device,
            denoiser=denoiser,
            detector=detector,
            dn_detector=dn_detector,
            enfdec=enfdec,
        )
        if (
            (torch.is_tensor(spind) and not spind.numel())
            or not len(spind)
            or not spind.size
        ):
            n_empty_batches += 1
            continue
        spike_indices.append(spind)
        if torch.is_tensor(wfs):
            wfs = wfs.cpu().numpy()
        waveforms.append(wfs)
        residuals.append(residual_singlebuf.cpu().numpy())
    if n_empty_batches:
        print("Got", n_empty_batches, "empty batches")
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
        # print(np.shape(cleaned_waveforms))
        denoised_waveforms = full_denoising(
            cleaned_waveforms,
            spike_index[:, 1],
            extract_channel_index,
            radial_parents,
            enfdec=enfdec,
            do_enforce_decrease=do_enforce_decrease,
            do_phaseshift=do_phaseshift,
            ci_graph_all_maxCH_uniq=ci_graph_all_maxCH_uniq,
            maxCH_neighbor=maxCH_neighbor,
            geom=geom,
            tpca=None,
            device=device,
            denoiser=denoiser,
        )
    if denoised_waveforms != None:
        denoised_waveforms = denoised_waveforms.to(device)

    # train extra featurizers if necessary
    extra_features = [] if extra_features is None else extra_features
    for f in extra_features:
        if f.needs_fit:
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
    enfdec,
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
    do_phaseshift=False,
    ci_graph_all_maxCH_uniq=None,
    maxCH_neighbor=None,
    geom=None,
    residnorm_decrease=False,
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

    raw = torch.as_tensor(raw, device=device)

    # the full buffer has length 2 * spike len on both sides,
    # but this spike index only contains the spikes inside
    # the inner buffer of length spike len
    # times are relative to the *inner* buffer
    spike_index = detect.detect_and_deduplicate(
        raw[start:end],
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
    padded_raw = F.pad(raw, (0, 1), value=torch.nan)
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
    if residnorm_decrease:
        resids = waveforms.clone()
    # print(np.shape(waveforms))
    # -- denoising
    waveforms, tpca_proj = full_denoising(
        waveforms,
        spike_index[:, 1],
        extract_channel_index,
        radial_parents,
        enfdec=enfdec,
        do_enforce_decrease=do_enforce_decrease,
        do_phaseshift=do_phaseshift,
        ci_graph_all_maxCH_uniq=ci_graph_all_maxCH_uniq,
        maxCH_neighbor=maxCH_neighbor,
        geom=geom,
        tpca=tpca,
        device=device,
        denoiser=denoiser,
        return_tpca_embedding=True,
    )

    waveforms = waveforms.to(device)

    # test residual norm decrease
    if residnorm_decrease:
        residthresh = 0.0
        if isinstance(residnorm_decrease, (int, float)):
            residthresh = residnorm_decrease
        residnorms0 = torch.linalg.norm(torch.nan_to_num(resids), dim=(1, 2))
        resids -= waveforms
        residnorms1 = torch.linalg.norm(torch.nan_to_num(resids), dim=(1, 2))
        decreased = residnorms1 + residthresh < residnorms0

        waveforms = waveforms[decreased]
        decreased_np = decreased.cpu().numpy()
        # print(f"{threshold=} {len(decreased_np)=} {decreased_np.mean()=}")
        spike_index = spike_index[decreased_np]
        time_ix = spike_index[:, 0, None] + time_range[None, :]
        chan_ix = extract_channel_index[spike_index[:, 1]]
        if tpca_proj is not None:
            tpca_proj = tpca_proj[decreased_np]

    # -- the actual subtraction
    # have to use subtract.at since -= will only subtract once in the overlaps,
    # subtract.at will subtract multiple times where waveforms overlap
    # np.subtract.at(
    #     padded_raw,
    #     (time_ix[:, :, None], chan_ix[:, None, :]),
    #     waveforms,
    # )
    # this is the torch equivalent. just have to do all the broadcasting manually.
    padded_raw.reshape(-1).scatter_add_(
        0,
        torch.as_tensor(
            np.ravel_multi_index(
                np.broadcast_arrays(time_ix[:, :, None], chan_ix[:, None, :]),
                padded_raw.shape,
            ),
            device=padded_raw.device,
        ).reshape(-1),
        -waveforms.reshape(-1),
    )

    # remove the NaN padding
    subtracted_raw = padded_raw[:, :-1]

    return waveforms, tpca_proj, subtracted_raw, spike_index


def full_denoising(
    waveforms,
    maxchans,
    extract_channel_index,
    radial_parents=None,
    enfdec=None,
    do_enforce_decrease=True,
    do_phaseshift=False,
    ci_graph_all_maxCH_uniq=None,
    maxCH_neighbor=None,
    geom=None,
    probe=None,
    tpca=None,
    device=None,
    denoiser=None,
    batch_size=128,
    align=False,
    return_tpca_embedding=False,
):
    """Denoising pipeline: neural net denoise, temporal PCA, enforce_decrease"""
    num_channels = len(extract_channel_index)
    N, T, C = waveforms.shape
    assert not align  # still working on that

    if do_phaseshift:
        if device == "cuda":
            torch.cuda.empty_cache()
        maxCH_neighbor = maxCH_neighbor.to(device)
        # if geom is None:
        #     raise ValueError('Phase-shift denoising needs geom input!')
        if ci_graph_all_maxCH_uniq is None:
            raise ValueError("Needs channel graph for neighbor searching!")
        # ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(extract_channel_index, geom, device = device)
        ci_graph_all_maxCH_uniq = ci_graph_all_maxCH_uniq.to(device)
        waveforms = torch.as_tensor(
            waveforms, device=device, dtype=torch.float
        )
        maxchans = torch.tensor(maxchans, device=device)

        if device == "cuda":
            waveforms = denoise.multichan_phase_shift_denoise_preshift(
                waveforms,
                ci_graph_all_maxCH_uniq,
                maxCH_neighbor,
                denoiser,
                maxchans,
                device,
            )
        else:
            for bs in range(0, N, batch_size):
                torch.cuda.empty_cache()
                be = min(bs + batch_size, N)
                bs = torch.as_tensor(bs, device=device)
                be = torch.as_tensor(be, device=device)
                waveforms[
                    bs:be, :, :
                ] = denoise.multichan_phase_shift_denoise_preshift(
                    waveforms[bs:be, :, :],
                    ci_graph_all_maxCH_uniq,
                    maxCH_neighbor,
                    denoiser,
                    maxchans[bs:be],
                    device,
                )

        # waveforms = torch.as_tensor(waveforms, device=device, dtype=torch.float)
        in_probe_channel_index = (
            torch.as_tensor(extract_channel_index, device=device)
            < num_channels
        )
        in_probe_index = in_probe_channel_index[maxchans]

        waveforms = waveforms.permute(0, 2, 1)
        wfs_in_probe = waveforms[in_probe_index]
    else:
        waveforms = torch.as_tensor(
            waveforms, device=device, dtype=torch.float
        )

        if not waveforms.numel():
            if return_tpca_embedding:
                embed = np.full(
                    (0, C, tpca.n_components), np.nan, dtype=tpca.dtype
                )
                return waveforms, embed
            return waveforms

        # in new pipeline, some channels are off the edge of the probe
        # those are filled with NaNs, which will blow up PCA. so, here
        # we grab just the non-NaN channels.
        in_probe_channel_index = (
            torch.as_tensor(extract_channel_index, device=device)
            < num_channels
        )
        in_probe_index = in_probe_channel_index[maxchans]
        waveforms = waveforms.permute(0, 2, 1)
        wfs_in_probe = waveforms[in_probe_index]

        # Apply NN denoiser (skip if None) #doesn't matter if wf on channels or everywhere
        if denoiser is not None:
            for bs in range(0, wfs_in_probe.shape[0], batch_size):
                be = min(bs + batch_size, N * C)
                wfs_in_probe[bs:be] = denoiser(wfs_in_probe[bs:be])

    # Temporal PCA while we are still transposed
    if tpca is not None:
        tpca = tpca.to(device)
        tpca_embeds = tpca.raw_transform(wfs_in_probe)
        wfs_in_probe = tpca.raw_inverse_transform(tpca_embeds)
        if not return_tpca_embedding:
            del tpca_embeds

    # back to original shape
    waveforms[in_probe_index] = wfs_in_probe
    waveforms = waveforms.permute(0, 2, 1)

    # enforce decrease
    if do_enforce_decrease:
        if enfdec is not None:
            waveforms = enfdec(waveforms, maxchans)
        elif radial_parents is not None:
            denoise.enforce_decrease_shells(
                waveforms, maxchans, radial_parents, in_place=True
            )
        else:
            # no enforce decrease
            pass

    if return_tpca_embedding and tpca is not None:
        tpca_embeddings = np.full(
            (N, C, tpca.n_components), np.nan, dtype=tpca.dtype
        )
        # run tpca only on channels that matter!
        tpca_embeddings[in_probe_index.cpu()] = tpca_embeds.cpu()
        return waveforms, tpca_embeddings.transpose(0, 2, 1)
    elif return_tpca_embedding:
        return waveforms, None

    return waveforms


# -- HDF5 initialization / resuming old job logic


@contextlib.contextmanager
def get_output_h5(
    out_h5,
    recording,
    extract_channel_index,
    extra_features,
    fit_features=None,
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
        output_h5.create_dataset("fs", data=recording.get_sampling_frequency())
        output_h5.create_dataset(
            "geom", data=recording.get_channel_locations()
        )
        output_h5.create_dataset("start_time", data=recording.get_times()[0])
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
        if fit_features is not None:
            for f in fit_features:
                # f.to('cpu')
                f.to_h5(output_h5)

    done_percent = 100 * last_sample / recording.get_num_samples()
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


# -- data loading helpers


def read_geom_from_meta(bin_file):
    try:
        from spikeglx import _geometry_from_meta, read_meta_data
    except ImportError:
        try:
            from ibllib.io.spikeglx import _geometry_from_meta, read_meta_data
        except ImportError:
            raise ImportError("Can't find spikeglx...")

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
        # we will run in this buffer and return it after subtraction
    raw = torch.from_numpy(raw)
    raw.to(device)
    residual = raw.clone()
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
            geom=geom,
        )
        _logger.debug(
            f"Detected and subtracted {spind.shape[0]} spikes "
            "with threshold {threshold} on {thresholds}"
        )
        if len(spind):
            subtracted_wfs.append(subwfs)
            spike_index.append(spind)

    subtracted_wfs = torch.cat(subtracted_wfs, dim=0)
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
        geom=geom,
        probe=probe,
        tpca=tpca,
        device=device,
        denoiser=denoiser,
    )

    # localize
    _logger.debug(f"Localisation...")
    ptp = chunk_features.PTPVector(which_waveforms="denoised").transform(
        spike_index[:, 1],
        denoised_wfs=cleaned_wfs,
    )
    xs, ys, z_rels, z_abss, alphas = localize_index.localize_ptps_index(
        ptp,
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
    return df_localisation, cleaned_wfs.to("cpu")
