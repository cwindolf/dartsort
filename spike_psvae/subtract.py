import concurrent.futures
import contextlib
import h5py
import numpy as np
import signal
import time
import torch

from collections import namedtuple
from ibllib.io.spikeglx import _geometry_from_meta, read_meta_data
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from . import denoise, detect, localize_index


def subtraction(
    standardized_bin,
    out_folder,
    geom=None,
    tpca_rank=8,
    n_sec_chunk=1,
    n_sec_pca=10,
    t_start=0,
    t_end=None,
    sampling_rate=30_000,
    thresholds=[12, 10, 8, 6, 5, 4],
    nn_detect=False,
    denoise_detect=False,
    neighborhood_kind="firstchan",
    extract_box_radius=200,
    extract_firstchan_n_channels=40,
    spike_length_samples=121,
    trough_offset=42,
    dedup_spatial_radius=70,
    enforce_decrease_kind="columns",
    n_jobs=1,
    device=None,
    save_residual=True,
    save_waveforms=True,
    do_clean=True,
    do_localize=True,
    localize_radius=100,
    localize_firstchan_n_channels=20,
    loc_workers=4,
    overwrite=False,
    random_seed=0,
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
                Only computed/saved if `do_localize=True`
                The columsn are: x, y, z, alpha, z relative to max channel
            maxptps : (N,)
                Only computed/saved if `do_localize=True`
    """
    if neighborhood_kind not in ("firstchan", "box"):
        raise ValueError(
            "Neighborhood kind", neighborhood_kind, "not understood."
        )
    if enforce_decrease_kind not in ("columns", "radial"):
        raise ValueError(
            "Enforce decrease method", enforce_decrease_kind, "not understood."
        )

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
        if out_h5.exists():
            with h5py.File(out_h5, "r+") as _d:
                pass
            del _d
    except BlockingIOError as e:
        raise ValueError(
            f"Output HDF5 {out_h5} is currently in use by another program. "
            "Maybe a Jupyter notebook that's still running?"
        ) from e

    # pick device if it's not supplied
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if no geometry is supplied, try to load it from meta file
    if geom is None:
        geom = read_geom_from_meta(standardized_bin)
        if geom is None:
            raise ValueError(
                "Either pass `geom` or put meta file in folder with binary."
            )
    n_channels = geom.shape[0]
    ncols = len(np.unique(geom[:, 0]))

    # TODO: read this from meta.
    # right now it's just used to load NN detector and pick the enforce
    # decrease method if enforce_decrease_kind=="columns"
    probe = {4: "np1", 2: "np2"}.get(ncols, None)

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
    std_size = standardized_bin.stat().st_size
    assert not std_size % np.dtype(np.float32).itemsize
    std_size = std_size // np.dtype(np.float32).itemsize
    assert not std_size % n_channels
    T_samples = std_size // n_channels

    # time logic -- what region are we going to load
    T_sec = T_samples / sampling_rate
    assert t_start >= 0 and (t_end is None or t_end <= T_sec)
    start_sample = int(np.floor(t_start * sampling_rate))
    end_sample = (
        T_samples if t_end is None else int(np.floor(t_end * sampling_rate))
    )
    portion_len_s = (end_sample - start_sample) / 30000
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
            geom, extract_box_radius, distance_order=False, p=1
        )
        # use radius-based localization neighborhood
        loc_n_chans = None
        loc_radius = localize_radius
    elif neighborhood_kind == "firstchan":
        extract_channel_index = []
        for c in range(n_channels):
            low = max(0, c - extract_firstchan_n_channels // 2)
            low = min(n_channels - extract_firstchan_n_channels, low)
            extract_channel_index.append(
                np.arange(low, low + extract_firstchan_n_channels)
            )
        extract_channel_index = np.array(extract_channel_index)

        # use old firstchan style localization neighborhood
        loc_n_chans = localize_firstchan_n_channels
        loc_radius = None
    else:
        assert False

    # helper data structure for radial enforce decrease
    if enforce_decrease_kind == "radial":
        radial_parents = denoise.make_radial_order_parents(
            geom, extract_channel_index, n_jumps_per_growth=1, n_jumps_parent=3
        )
    elif enforce_decrease_kind == "columns":
        radial_parents = None
    else:
        assert False

    # pre-fit temporal PCA
    tpca = None
    if n_sec_pca is not None:
        # try to load old TPCA if it's around
        if not overwrite and out_h5.exists():
            with h5py.File(out_h5, "r") as output_h5:
                if "tpca_mean" in output_h5:
                    tpca_mean = output_h5["tpca_mean"][:]
                    tpca_components = output_h5["tpca_components"][:]
                    print("Loading TPCA from h5")
                    tpca = PCA(tpca_components.shape[0])
                    tpca.mean_ = tpca_mean
                    tpca.components_ = tpca_components

        # otherwise, train it
        if tpca is None:
            with timer("Training TPCA"):
                tpca = train_pca(
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
                    probe=probe,
                    standardized_dtype=np.float32,
                    n_sec_pca=n_sec_pca,
                    rank=tpca_rank,
                    random_seed=random_seed,
                    device=device,
                )

    # if we're on GPU, we can't use processes, since each process will
    # have it's own torch runtime and those will use all the memory
    if device.type == "cuda":
        Pool = concurrent.futures.ThreadPoolExecutor
    else:
        if loc_workers > 1:
            print(
                "Setting number of localization workers to 1. (Since "
                "you're on CPU, use a large n_jobs for parallelism.)"
            )
            loc_workers = 1
        Pool = concurrent.futures.ThreadPoolExecutor

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
    n_batches = len(jobs)

    # -- initialize storage
    with get_output_h5(
        out_h5,
        start_sample,
        end_sample,
        geom,
        extract_channel_index,
        tpca,
        neighborhood_kind,
        spike_length_samples,
        save_waveforms=save_waveforms,
        do_clean=do_clean,
        do_localize=do_localize,
        overwrite=overwrite,
    ) as (output_h5, last_sample):

        spike_index = output_h5["spike_index"]
        if neighborhood_kind == "firstchan":
            firstchans = output_h5["first_channels"]
        if save_waveforms:
            subtracted_wfs = output_h5["subtracted_waveforms"]
            if do_clean:
                cleaned_wfs = output_h5["cleaned_waveforms"]
        if do_localize:
            locs = output_h5["localizations"]
            maxptps = output_h5["maxptps"]
        N = len(spike_index)

        # if we're resuming, filter out jobs we already did
        jobs = (
            (batch_id, start)
            for batch_id, start in jobs
            if start >= last_sample
        )

        # residual binary file -- append if we're resuming
        if save_residual:
            residual_mode = "ab" if last_sample > 0 else "ab"
            residual = open(residual_bin, mode=residual_mode)

        # now run subtraction in parallel
        jobs = (
            (
                batch_id,
                batch_data_folder,
                s_start,
                batch_len_samples,
                standardized_bin,
                thresholds,
                tpca,
                trough_offset,
                dedup_channel_index,
                spike_length_samples,
                extract_channel_index,
                device,
                start_sample,
                end_sample,
                do_clean,
                radial_parents,
                do_localize,
                loc_workers,
                geom,
                probe,
                loc_n_chans,
                loc_radius,
            )
            for batch_id, s_start in jobs
        )

        with Pool(
            n_jobs,
            initializer=_subtraction_batch_init,
            initargs=(
                device,
                nn_detector_path,
                nn_channel_index,
                denoise_detect,
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

                    # grow arrays as necessary
                    if save_waveforms:
                        subtracted_wfs.resize(N + N_new, axis=0)
                        if do_clean:
                            cleaned_wfs.resize(N + N_new, axis=0)
                    spike_index.resize(N + N_new, axis=0)
                    if do_localize:
                        locs.resize(N + N_new, axis=0)
                        maxptps.resize(N + N_new, axis=0)
                    if neighborhood_kind == "firstchan":
                        firstchans.resize(N + N_new, axis=0)

                    # write results
                    if save_waveforms:
                        subtracted_wfs[N:] = np.load(result.subtracted_wfs)
                        if do_clean:
                            cleaned_wfs[N:] = np.load(result.cleaned_wfs)
                    spike_index[N:] = np.load(result.spike_index)
                    if do_localize:
                        locs[N:] = np.load(result.localizations)
                        maxptps[N:] = np.load(result.maxptps)
                    if neighborhood_kind == "firstchan":
                        firstchans[N:] = extract_channel_index[
                            np.load(result.spike_index)[:, 1],
                            0,
                        ]

                    # delete original files
                    Path(result.residual).unlink()
                    Path(result.subtracted_wfs).unlink()
                    if do_clean:
                        Path(result.cleaned_wfs).unlink()
                    Path(result.spike_index).unlink()
                    if do_localize:
                        Path(result.localizations).unlink()
                        Path(result.maxptps).unlink()

                    # update spike count
                    N += N_new

    # -- done!
    batch_data_folder.rmdir()
    if save_residual:
        residual.close()
    print("Done. Detected", N, "spikes")
    print("Results written to:")
    if save_residual:
        print(residual_bin)
    print(out_h5)
    return out_h5


# -- subtraction routines


# the return type for `subtraction_batch` below
SubtractionBatchResult = namedtuple(
    "SubtractionBatchResult",
    [
        "N_new",
        "s_start",
        "s_end",
        "residual",
        "subtracted_wfs",
        "cleaned_wfs",
        "spike_index",
        "batch_id",
        "localizations",
        "maxptps",
    ],
)


# Parallelism helpers
def _subtraction_batch(args):
    return subtraction_batch(
        *args,
        _subtraction_batch.denoiser,
        _subtraction_batch.detector,
        _subtraction_batch.dn_detector,
    )


def _subtraction_batch_init(
    device, nn_detector_path, nn_channel_index, denoise_detect
):
    """Thread/process initializer -- loads up neural nets"""
    denoiser = denoise.SingleChanDenoiser()
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
    batch_id,
    batch_data_folder,
    s_start,
    batch_len_samples,
    standardized_bin,
    thresholds,
    tpca,
    trough_offset,
    dedup_channel_index,
    spike_length_samples,
    extract_channel_index,
    device,
    start_sample,
    end_sample,
    do_clean,
    radial_parents,
    do_localize,
    loc_workers,
    geom,
    probe,
    loc_n_chans,
    loc_radius,
    denoiser,
    detector,
    dn_detector,
):
    """Runs subtraction on a batch

    This function handles the logic of loading data from disk
    (padding it with a buffer where necessary), running the loop
    over thresholds for `detect_and_subtract`, handling spikes
    that were in the buffer, and applying the denoising pipeline.

    Arguments
    ---------
    batch_id : int
        Used when naming temporary batch result files saved to
        `batch_data_folder`. (Not used otherwise -- in particular
        this does not determine what data is loaded or processed.)
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
    do_localize : bool
        Should we run localization?
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
    # load raw data with buffer
    s_end = min(end_sample, s_start + batch_len_samples)
    buffer = 2 * spike_length_samples
    n_channels = len(dedup_channel_index)
    load_start = max(start_sample, s_start - buffer)
    load_end = min(end_sample, s_end + buffer)
    residual = read_data(
        standardized_bin, np.float32, load_start, load_end, n_channels
    )

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
    assert residual.shape == (2 * buffer + s_end - s_start, n_channels)

    # main subtraction loop
    subtracted_wfs = []
    spike_index = []
    for threshold in thresholds:
        subwfs, residual, spind = detect_and_subtract(
            residual,
            threshold,
            radial_parents,
            tpca,
            dedup_channel_index,
            extract_channel_index,
            detector=detector,
            denoiser=denoiser,
            denoiser_detector=dn_detector,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            device=device,
            probe=probe,
        )
        if len(spind):
            subtracted_wfs.append(subwfs)
            spike_index.append(spind)

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
    spike_time_max = s_end - s_start - 2 * buffer
    if load_end == end_sample:
        spike_time_max -= spike_length_samples - trough_offset

    minix = np.searchsorted(spike_index[:, 0], spike_time_min, side="right")
    maxix = -1 + np.searchsorted(
        spike_index[:, 0], spike_time_max, side="left"
    )
    spike_index = spike_index[minix:maxix]
    subtracted_wfs = subtracted_wfs[minix:maxix]

    # get cleaned waveforms
    cleaned_wfs = None
    if do_clean:
        cleaned_wfs = read_waveforms(
            residual,
            spike_index,
            spike_length_samples,
            extract_channel_index,
            trough_offset=trough_offset,
            buffer=buffer,
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

    # strip buffer from residual and remove spikes in buffer
    residual = residual[buffer:-buffer]

    # if caller passes None for the output folder, just return
    # the results now (eg this is used by train_pca)
    if batch_data_folder is None:
        return spike_index, subtracted_wfs

    # time relative to batch start
    spike_index[:, 0] += s_start

    # save the results to disk to avoid memory issues
    N_new = len(spike_index)
    np.save(batch_data_folder / f"{batch_id:08d}_res.npy", residual)
    np.save(batch_data_folder / f"{batch_id:08d}_sub.npy", subtracted_wfs)
    np.save(batch_data_folder / f"{batch_id:08d}_si.npy", spike_index)

    clean_file = None
    if do_clean:
        clean_file = batch_data_folder / f"{batch_id:08d}_clean.npy"
        np.save(clean_file, cleaned_wfs)

    localizations_file = maxptps_file = None
    if do_localize:
        locwfs = cleaned_wfs if do_clean else subtracted_wfs
        locptps = locwfs.ptp(1)
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
        localizations_file = batch_data_folder / f"{batch_id:08d}_loc.npy"
        np.save(localizations_file, np.c_[xs, ys, z_abss, alphas, z_rels])
        maxptps_file = batch_data_folder / f"{batch_id:08d}_maxptp.npy"
        np.save(maxptps_file, np.nanmax(locptps, axis=1))

    res = SubtractionBatchResult(
        N_new=N_new,
        s_start=s_start,
        s_end=s_end,
        residual=batch_data_folder / f"{batch_id:08d}_res.npy",
        subtracted_wfs=batch_data_folder / f"{batch_id:08d}_sub.npy",
        cleaned_wfs=clean_file,
        spike_index=batch_data_folder / f"{batch_id:08d}_si.npy",
        batch_id=batch_id,
        localizations=localizations_file,
        maxptps=maxptps_file,
    )

    return res


# -- temporal PCA


def train_pca(
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
    probe=None,
    standardized_dtype=np.float32,
    n_sec_pca=10,
    rank=7,
    random_seed=0,
    device="cpu",
):
    """Pre-train temporal PCA

    Extracts several random seconds of data by subtraction
    with no PCA, and trains a temporal PCA on the resulting
    waveforms.
    """
    n_seconds = len_recording_samples // sampling_rate
    starts = sampling_rate * np.random.default_rng(random_seed).choice(
        n_seconds, size=min(n_sec_pca, n_seconds), replace=False
    )

    denoiser = denoise.SingleChanDenoiser().load().to(device)

    detector = None
    if nn_detector_path:
        detector = detect.Detect(nn_channel_index)
        detector.load(nn_detector_path)
        detector.to(device)

    dn_detector = None
    if denoise_detect:
        dn_detector = detect.DenoiserDetect(denoiser)
        dn_detector.to(device)

    # do a mini-subtraction with no PCA, just NN denoise and enforce_decrease
    spike_index = []
    waveforms = []
    for s_start in tqdm(starts, "PCA training subtraction"):
        spind, wfs = subtraction_batch(
            0,
            None,
            s_start,
            sampling_rate,
            standardized_bin,
            thresholds,
            None,
            42,
            dedup_channel_index,
            spike_length_samples,
            extract_channel_index,
            device,
            0,
            len_recording_samples,
            False,
            radial_parents,
            False,
            None,
            None,
            probe,
            None,
            None,
            denoiser,
            detector,
            dn_detector,
        )
        spike_index.append(spind)
        waveforms.append(wfs)
    spike_index = np.concatenate(spike_index, axis=0)
    waveforms = np.concatenate(waveforms, axis=0)
    N, T, C = waveforms.shape
    print("Fitting PCA on", N, "waveforms from mini-subtraction")

    # fit TPCA
    tpca = PCA(rank, random_state=random_seed)
    # extract waveforms for real channels
    in_probe_index = extract_channel_index < extract_channel_index.shape[0]
    wfs_in_probe = waveforms.transpose(0, 2, 1)
    wfs_in_probe = wfs_in_probe[in_probe_index[spike_index[:, 1]]]
    tpca.fit(wfs_in_probe)

    return tpca


# -- denoising / detection helpers


def detect_and_subtract(
    raw,
    threshold,
    radial_parents,
    tpca,
    dedup_channel_index,
    extract_channel_index,
    detector=None,
    denoiser=None,
    denoiser_detector=None,
    nn_switch_threshold=4,
    trough_offset=42,
    spike_length_samples=121,
    device="cpu",
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

    spike_index = detect.detect_and_deduplicate(
        raw[start:end].copy(),
        threshold,
        dedup_channel_index,
        spike_length_samples,
        device=device,
        **kwargs,
    )
    # print(threshold, len(spike_index), flush=True)
    if not len(spike_index):
        return [], raw, []

    # -- read waveforms
    padded_raw = np.pad(raw, [(0, 0), (0, 1)], constant_values=np.nan)
    # times relative to trough + buffer
    time_range = np.arange(
        2 * spike_length_samples - trough_offset,
        3 * spike_length_samples - trough_offset,
    )
    time_ix = spike_index[:, 0, None] + time_range[None, :]
    chan_ix = extract_channel_index[spike_index[:, 1]]
    waveforms = padded_raw[time_ix[:, :, None], chan_ix[:, None, :]]

    # -- denoising
    waveforms = full_denoising(
        waveforms,
        spike_index[:, 1],
        extract_channel_index,
        radial_parents,
        probe=probe,
        tpca=tpca,
        device=device,
        denoiser=denoiser,
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

    return waveforms, subtracted_raw, spike_index


@torch.no_grad()
def full_denoising(
    waveforms,
    maxchans,
    extract_channel_index,
    radial_parents,
    probe=None,
    tpca=None,
    device=None,
    denoiser=None,
    batch_size=1024,
    align=False,
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

    # Apply NN denoiser (skip if None)
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
    if radial_parents is None and probe is not None:
        rel_maxchans = maxchans - extract_channel_index[maxchans, 0]
        if probe == "np1":
            for i in range(N):
                denoise.enforce_decrease_np1(
                    waveforms[i], max_chan=rel_maxchans[i], in_place=True
                )
        elif probe == "np2":
            for i in range(N):
                denoise.enforce_decrease_np1(
                    waveforms[i], max_chan=rel_maxchans[i], in_place=True
                )
        else:
            assert False
    elif radial_parents is not None:
        denoise.enforce_decrease_shells(
            waveforms, maxchans, radial_parents, in_place=True
        )
    else:
        assert False

    return waveforms


def read_waveforms(
    recording,
    spike_index,
    spike_length_samples,
    extract_channel_index,
    trough_offset=42,
    buffer=0,
):
    """Load waveforms from an array in memory"""
    # pad with NaN to fill resulting waveforms with NaN when
    # channel is outside probe
    padded_recording = np.pad(
        recording, [(0, 0), (0, 1)], constant_values=np.nan
    )
    # times relative to trough + buffer
    time_range = np.arange(
        buffer - trough_offset,
        buffer + spike_length_samples - trough_offset,
    )
    time_ix = spike_index[:, 0, None] + time_range[None, :]
    chan_ix = extract_channel_index[spike_index[:, 1]]
    waveforms = padded_recording[time_ix[:, :, None], chan_ix[:, None, :]]

    return waveforms


# -- HDF5 initialization / resuming old job logic


@contextlib.contextmanager
def get_output_h5(
    out_h5,
    start_sample,
    end_sample,
    geom,
    extract_channel_index,
    tpca,
    neighborhood_kind,
    spike_length_samples,
    do_clean=True,
    do_localize=True,
    save_waveforms=True,
    overwrite=False,
    chunk_len=4096,
):
    h5_exists = Path(out_h5).exists()
    last_sample = 0
    if h5_exists and not overwrite:
        output_h5 = h5py.File(out_h5, "r+")
        h5_exists = True
        if len(output_h5["spike_index"]) > 0:
            last_sample = output_h5["spike_index"][-1, 0]
    else:
        if overwrite:
            print("Overwriting previous results, if any.")
        output_h5 = h5py.File(out_h5, "w")

        # initialize datasets
        output_h5.create_dataset("geom", data=geom)
        output_h5.create_dataset("start_sample", data=start_sample)
        output_h5.create_dataset("end_sample", data=end_sample)
        output_h5.create_dataset("channel_index", data=extract_channel_index)
        if tpca is not None:
            output_h5.create_dataset("tpca_mean", data=tpca.mean_)
            output_h5.create_dataset("tpca_components", data=tpca.components_)

        # resizable datasets so we don't fill up space
        extract_channels = extract_channel_index.shape[1]
        if save_waveforms:
            output_h5.create_dataset(
                "subtracted_waveforms",
                shape=(0, spike_length_samples, extract_channels),
                chunks=(chunk_len, spike_length_samples, extract_channels),
                maxshape=(None, spike_length_samples, extract_channels),
                dtype=np.float32,
            )
        output_h5.create_dataset(
            "spike_index",
            shape=(0, 2),
            chunks=(chunk_len, 2),
            maxshape=(None, 2),
            dtype=np.int64,
        )
        if neighborhood_kind == "firstchan":
            output_h5.create_dataset(
                "first_channels",
                shape=(0,),
                chunks=(chunk_len,),
                maxshape=(None,),
                dtype=np.int64,
            )
        if save_waveforms and do_clean:
            output_h5.create_dataset(
                "cleaned_waveforms",
                shape=(0, spike_length_samples, extract_channels),
                chunks=(chunk_len, spike_length_samples, extract_channels),
                maxshape=(None, spike_length_samples, extract_channels),
                dtype=np.float32,
            )
        if do_localize:
            output_h5.create_dataset(
                "localizations",
                shape=(0, 5),
                chunks=(chunk_len, 5),
                maxshape=(None, 5),
                dtype=np.float32,
            )
            output_h5.create_dataset(
                "maxptps",
                shape=(0,),
                chunks=(chunk_len,),
                maxshape=(None,),
                dtype=np.float32,
            )

    done_percent = (
        100 * (last_sample - start_sample) / (end_sample - start_sample)
    )
    if h5_exists and not overwrite:
        print(f"Resuming previous job, which was {done_percent:.0f}% done")
    elif h5_exists and overwrite:
        print("Overwriting previous results.")
        last_sample = 0
    else:
        print("No previous output found, starting from scratch.")

    yield output_h5, last_sample

    output_h5.close()


# -- channels / geometry helpers


def n_steps_neigh_channels(neighbors_matrix, steps):
    """Compute a neighbors matrix by considering neighbors of neighbors

    Parameters
    ----------
    neighbors_matrix: numpy.ndarray
        Neighbors matrix
    steps: int
        Number of steps to still consider channels as neighbors

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    # Compute neighbors of neighbors via matrix powers
    output = np.eye(neighbors_matrix.shape[0]) + neighbors_matrix
    return np.linalg.matrix_power(output, steps) > 0


def order_channels_by_distance(reference, channels, geom):
    """Order channels by distance using certain channel as reference
    Parameters
    ----------
    reference: int
        Reference channel
    channels: np.ndarray
        Channels to order
    geom
        Geometry matrix
    Returns
    -------
    numpy.ndarray
        1D array with the channels ordered by distance using the reference
        channels
    numpy.ndarray
        1D array with the indexes for the ordered channels
    """
    coord_main = geom[reference]
    coord_others = geom[channels]
    idx = np.argsort(np.sum(np.square(coord_others - coord_main), axis=1))
    return channels[idx], idx


def make_channel_index(geom, radius, steps=1, distance_order=True, p=2):
    """
    Compute an array whose whose ith row contains the ordered
    (by distance) neighbors for the ith channel
    """
    C = geom.shape[0]

    # get neighbors matrix
    neighbors = squareform(pdist(geom, metric="minkowski", p=p)) <= radius
    neighbors = n_steps_neigh_channels(neighbors, steps=steps)

    # max number of neighbors for all channels
    n_neighbors = np.max(np.sum(neighbors, 0))

    # initialize channel index
    # entries for channels which don't have as many neighbors as
    # others will be filled with the total number of channels
    # (an invalid index into the recording, but this behavior
    # is useful e.g. in the spatial max pooling for deduplication)
    channel_index = np.full((C, n_neighbors), C, dtype=int)

    # fill every row in the matrix (one per channel)
    for current in range(C):
        # indexes of current channel neighbors
        ch_idx = np.flatnonzero(neighbors[current])

        # sort them by distance
        if distance_order:
            ch_idx, _ = order_channels_by_distance(current, ch_idx, geom)

        # fill entries with the sorted neighbor indexes
        channel_index[current, : ch_idx.shape[0]] = ch_idx

    return channel_index


# -- data loading helpers


def read_geom_from_meta(bin_file):
    meta = Path(bin_file.parent) / (bin_file.stem + ".meta")
    if not meta.exists():
        raise ValueError("Expected", meta, "to exist.")
    header = _geometry_from_meta(read_meta_data(meta))
    geom = np.c_[header["x"], header["y"]]
    return geom


def read_data(bin_file, dtype, s_start, s_end, n_channels):
    """Read a chunk of a binary file"""
    offset = s_start * np.dtype(dtype).itemsize * n_channels
    with open(bin_file, "rb") as fin:
        data = np.fromfile(
            fin,
            dtype=dtype,
            count=(s_end - s_start) * n_channels,
            offset=offset,
        )
    data = data.reshape(-1, n_channels)
    return data


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


class _noint:
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


noint = _noint()
