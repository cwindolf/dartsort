import h5py
import numpy as np
import time
import torch
import itertools
import multiprocessing

from collections import namedtuple
from ibllib.io.spikeglx import _geometry_from_meta, read_meta_data
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm.auto import tqdm, trange

from . import denoise, voltage_detect, waveform_utils


# -- subtraction routines
# the main function is `subtraction(...)`, which uses `subtraction_batch(...)`
# as a helper for parallelism
# however, the actual logic of subtraction is in `detect_and_subtract(...)`


SubtractionBatchResult = namedtuple(
    "SubtractionBatchResult",
    [
        "N_new",
        "s_start",
        "s_end",
        "residual",
        "subtracted_wfs",
        "cleaned_wfs",
        "firstchans",
        "spike_index",
        "batch_id",
    ],
)


def subtraction_batch(*args):
    if len(args) == 1:
        args = args[0]
    (
        batch_id,
        batch_data_folder,
        s_start,
        batch_len_samples,
        T_samples,
        standardized_bin,
        thresholds,
        tpca,
        trough_offset,
        channel_index,
        spike_length_samples,
        extract_channels,
        device,
        start_sample,
        end_sample,
        do_clean,
    ) = args

    # load denoiser (load here so that we load only once per batch)
    denoiser = denoise.SingleChanDenoiser()
    denoiser.load()
    denoiser.to(device)

    # load raw data with buffer
    s_end = min(end_sample, s_start + batch_len_samples)
    buffer = 2 * spike_length_samples
    n_channels = len(channel_index)
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
    firstchans = []
    for threshold in thresholds:
        # old_resid = residual.copy()
        subwfs, spind, fcs = detect_and_subtract(
            residual,
            threshold,
            tpca,
            denoiser,
            trough_offset,
            channel_index,
            spike_length_samples,
            extract_channels,
            device,
        )

        if len(spind):
            # assert (np.abs(residual - old_resid) > 0).any()
            subtracted_wfs.append(subwfs)
            spike_index.append(spind)
            firstchans.append(fcs)
    subtracted_wfs = np.concatenate(subtracted_wfs, axis=0)
    spike_index = np.concatenate(spike_index, axis=0)
    firstchans = np.concatenate(firstchans, axis=0)

    # sort so that time is increasing
    sort = np.argsort(spike_index[:, 0])
    subtracted_wfs = subtracted_wfs[sort]
    spike_index = spike_index[sort]
    firstchans = firstchans[sort]

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
    firstchans = firstchans[minix:maxix]
    subtracted_wfs = subtracted_wfs[minix:maxix]

    # get cleaned waveforms
    cleaned_wfs = None
    if do_clean:
        cleaned_wfs, _ = read_waveforms(
            residual,
            spike_index,
            spike_length_samples,
            extract_channels,
            trough_offset=trough_offset,
            buffer=buffer,
        )
        # assert (f == firstchans).all()
        cleaned_wfs = full_denoising(
            cleaned_wfs + subtracted_wfs,
            spike_index[:, 1] - firstchans,
            tpca=tpca,
            device=device,
            denoiser=denoiser,
        )

    # strip buffer from residual and remove spikes in buffer
    residual = residual[buffer:-buffer]

    # time relative to batch start
    spike_index[:, 0] += s_start

    # write temp files
    if batch_data_folder is None:
        return subtracted_wfs

    N_new = len(spike_index)
    np.save(batch_data_folder / f"{batch_id:08d}_res.npy", residual)
    np.save(batch_data_folder / f"{batch_id:08d}_sub.npy", subtracted_wfs)
    np.save(batch_data_folder / f"{batch_id:08d}_fc.npy", firstchans)
    np.save(batch_data_folder / f"{batch_id:08d}_si.npy", spike_index)

    clean_file = None
    if do_clean:
        clean_file = batch_data_folder / f"{batch_id:08d}_clean.npy"
        np.save(clean_file, cleaned_wfs)

    res = SubtractionBatchResult(
        N_new=N_new,
        s_start=s_start,
        s_end=s_end,
        residual=batch_data_folder / f"{batch_id:08d}_res.npy",
        subtracted_wfs=batch_data_folder / f"{batch_id:08d}_sub.npy",
        cleaned_wfs=clean_file,
        firstchans=batch_data_folder / f"{batch_id:08d}_fc.npy",
        spike_index=batch_data_folder / f"{batch_id:08d}_si.npy",
        batch_id=batch_id,
    )

    return res


def subtraction(
    standardized_bin,
    out_folder,
    geom=None,
    spatial_radius=70,
    tpca_rank=7,
    n_sec_chunk=1,
    n_sec_pca=10,
    t_start=0,
    t_end=None,
    sampling_rate=30_000,
    thresholds=[12, 10, 8, 6, 5, 4],
    extract_channels=40,
    spike_length_samples=121,
    trough_offset=42,
    n_jobs=1,
    device=None,
    do_clean=True,
    random_seed=0,
):
    standardized_bin = Path(standardized_bin)
    stem = standardized_bin.stem
    batch_len_samples = n_sec_chunk * sampling_rate

    # prepare output dir
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True)
    batch_data_folder = out_folder / f"batches_{stem}"
    batch_data_folder.mkdir(exist_ok=True)
    out_h5 = out_folder / f"subtraction_{stem}_t_{t_start}_{t_end}.h5"
    residual_bin = out_folder / f"residual_{stem}_t_{t_start}_{t_end}.bin"
    if residual_bin.exists():
        print("Output residual exists, it will be overwritten.")
    try:
        h5py.File(out_h5, "w")
    except BlockingIOError as e:
        raise ValueError(
            f"Output HDF5 {out_h5} is currently in use by another program. "
            "Maybe a Jupyter notebook that's still running?"
        ) from e

    # pick device if it's None
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # if no geometry is supplied, try to load it from meta file
    if geom is None:
        metas = list(standardized_bin.parent.glob("*.meta"))
        if metas:
            assert len(metas) == 1
            header = _geometry_from_meta(read_meta_data(metas[0]))
            geom = np.c_[header["x"], header["y"]]
        else:
            raise ValueError(
                "Either pass `geom` or put meta file in folder with binary."
            )
    n_channels = geom.shape[0]

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
    print(
        "Running subtraction on ",
        T_sec,
        "seconds long recording with thresholds",
        thresholds,
    )

    # compute helper data structures
    channel_index = make_channel_index(geom, spatial_radius, steps=2)

    # pre-fit temporal PCA
    with timer("Training TPCA"):
        tpca = train_pca(
            standardized_bin,
            spike_length_samples,
            extract_channels,
            geom,
            T_samples,
            sampling_rate,
            channel_index,
            thresholds,
            standardized_dtype=np.float32,
            n_sec_pca=n_sec_pca,
            rank=tpca_rank,
            random_seed=random_seed,
        )

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
    # residual binary file
    residual = open(residual_bin, mode="wb")

    # everybody else in hdf5
    output_h5 = h5py.File(out_h5, "w")
    output_h5.create_dataset("geom", data=geom)
    output_h5.create_dataset("start_sample", data=start_sample)
    output_h5.create_dataset("end_sample", data=end_sample)
    output_h5.create_dataset("tpca_mean", data=tpca.mean_)
    output_h5.create_dataset("tpca_components", data=tpca.components_)

    # resizable datasets so we don't fill up space
    subtracted_wfs = output_h5.create_dataset(
        "subtracted_waveforms",
        shape=(1, spike_length_samples, extract_channels),
        chunks=(4096, spike_length_samples, extract_channels),
        maxshape=(None, spike_length_samples, extract_channels),
        dtype=np.float32,
    )
    firstchans = output_h5.create_dataset(
        "first_channels",
        shape=(1,),
        chunks=(4096,),
        maxshape=(None,),
        dtype=np.int32,
    )
    spike_index = output_h5.create_dataset(
        "spike_index",
        shape=(1, 2),
        chunks=(4096, 2),
        maxshape=(None, 2),
        dtype=np.int64,
    )
    if do_clean:
        cleaned_wfs = output_h5.create_dataset(
            "cleaned_waveforms",
            shape=(1, spike_length_samples, extract_channels),
            chunks=(4096, spike_length_samples, extract_channels),
            maxshape=(None, spike_length_samples, extract_channels),
            dtype=np.float32,
        )

    # now run subtraction in parallel
    N = 0
    jobs = (
        (
            batch_id,
            batch_data_folder,
            s_start,
            batch_len_samples,
            T_samples,
            standardized_bin,
            thresholds,
            tpca,
            trough_offset,
            channel_index,
            spike_length_samples,
            extract_channels,
            device,
            start_sample,
            end_sample,
            do_clean,
        )
        for batch_id, s_start in jobs
    )

    with multiprocessing.pool.ThreadPool(n_jobs) as pool:
        for result in tqdm(
            pool.imap(subtraction_batch, jobs), total=n_batches, desc="Batches"
        ):
            N_new = result.N_new

            # write new residual
            np.load(result.residual).tofile(residual)

            # grow arrays as necessary
            subtracted_wfs.resize(N + N_new, axis=0)
            if do_clean:
                cleaned_wfs.resize(N + N_new, axis=0)
            firstchans.resize(N + N_new, axis=0)
            spike_index.resize(N + N_new, axis=0)

            # write results
            subtracted_wfs[N : N + N_new] = np.load(result.subtracted_wfs)
            if do_clean:
                cleaned_wfs[N : N + N_new] = np.load(result.cleaned_wfs)
            firstchans[N : N + N_new] = np.load(result.firstchans)
            spike_index[N : N + N_new] = np.load(result.spike_index)

            # delete original files
            Path(result.residual).unlink()
            Path(result.subtracted_wfs).unlink()
            if do_clean:
                Path(result.cleaned_wfs).unlink()
            Path(result.firstchans).unlink()
            Path(result.spike_index).unlink()

            # update my state
            N += N_new

    # -- done!
    residual.close()
    output_h5.close()
    print("Done. Detected", N, "spikes")
    print("Results written to:")
    print(residual_bin)
    print(out_h5)


# -- temporal PCA


def train_pca(
    standardized_bin,
    spike_length_samples,
    extract_channels,
    geom,
    len_recording_samples,
    sampling_rate,
    channel_index,
    thresholds,
    standardized_dtype=np.float32,
    n_sec_pca=10,
    rank=7,
    random_seed=0,
):
    s_start = len_recording_samples // 2 - sampling_rate * n_sec_pca // 2
    s_end = len_recording_samples // 2 + sampling_rate * n_sec_pca // 2
    if s_start < 0 or s_end > len_recording_samples:
        raise ValueError(f"n_sec_pca={n_sec_pca} was too big for this data.")

    # do a mini-subtraction with no PCA, just NN denoise and enforce_decrease
    waveforms = subtraction_batch(
        0,
        None,
        s_start,
        s_end - s_start,
        len_recording_samples,
        standardized_bin,
        thresholds,
        None,
        42,
        channel_index,
        spike_length_samples,
        extract_channels,
        "cpu",
        s_start,
        s_end,
        False,
    )
    N, T, C = waveforms.shape
    print("Fitting PCA on", N, "waveforms from mini-subtraction")

    # fit TPCA
    tpca = PCA(rank, random_state=random_seed)
    tpca.fit(waveforms.transpose(0, 2, 1).reshape(N * C, T))

    return tpca


# -- denoising / detection helpers


def detect_and_subtract(
    raw,
    threshold,
    tpca,
    denoiser,
    trough_offset,
    channel_index,
    spike_length_samples,
    extract_channels,
    device,
):
    """This subtracts from raw in place, leaving the residual behind"""
    spike_index, energy = voltage_detect.detect_and_deduplicate(
        raw[spike_length_samples:-spike_length_samples],
        threshold,
        channel_index,
        spike_length_samples,
        device,
    )
    # it would be nice to go in order, but we would need to
    # combine the reading and subtraction steps together
    # subtraction_order = np.argsort(energy)[::-1]
    subtracted_wfs, firstchans = read_waveforms(
        raw,
        spike_index,
        spike_length_samples,
        extract_channels,
        trough_offset=trough_offset,
        buffer=2 * spike_length_samples,
    )

    # denoising
    subtracted_wfs = full_denoising(
        subtracted_wfs,
        spike_index[:, 1] - firstchans,
        tpca,
        device,
        denoiser,
    )

    # the actual subtraction
    for wf, (t, mc), fc in zip(subtracted_wfs, spike_index, firstchans):
        raw[
            t
            - trough_offset
            + 2 * spike_length_samples : t
            - trough_offset
            + spike_length_samples
            + 2 * spike_length_samples,
            fc : fc + extract_channels,
        ] -= wf

    return subtracted_wfs, spike_index, firstchans


def read_waveforms(
    recording,
    spike_index,
    spike_length_samples,
    extract_channels,
    trough_offset=42,
    buffer=0,
):
    n_channels = recording.shape[1]

    # how many channels down from max channel?
    chans_down = extract_channels // 2
    chans_down -= chans_down % 2

    # allocate output storage
    waveforms = np.empty(
        (len(spike_index), spike_length_samples, extract_channels),
        dtype=recording.dtype,
    )
    firstchans = np.empty(len(spike_index), dtype=np.int32)

    # extraction loop
    for i in range(len(spike_index)):
        t, mc = spike_index[i]

        # what will be the first extracted channel?
        mc_idx = mc - mc % 2
        fc = mc_idx - chans_down
        fc = max(fc, 0)
        fc = min(fc, n_channels - extract_channels)

        waveforms[i] = recording[
            t
            - trough_offset
            + buffer : t
            - trough_offset
            + spike_length_samples
            + buffer,
            fc : fc + extract_channels,
        ]
        firstchans[i] = fc

    return waveforms, firstchans


@torch.inference_mode()
def full_denoising(
    waveforms,
    maxchans,
    tpca=None,
    device=None,
    denoiser=None,
    batch_size=1024,
    align=False,
):
    N, T, C = waveforms.shape

    # temporal align
    if align:
        waveforms, rolls = denoise.temporal_align(waveforms, maxchans=maxchans)

    # Apply NN denoiser (skip if None)
    waveforms = waveforms.transpose(0, 2, 1).reshape(N * C, T)
    if denoiser is not None:
        for bs in range(0, N * C, batch_size):
            be = min(bs + batch_size, N * C)
            o = waveforms[bs:be].copy()
            waveforms[bs:be] = (
                denoiser(
                    torch.tensor(
                        waveforms[bs:be], device=device, dtype=torch.float
                    )
                )
                .cpu()
                .numpy()
            )

    # Temporal PCA while we are still transposed
    if tpca is not None:
        waveforms = tpca.inverse_transform(tpca.transform(waveforms))

    # Un-transpose, enforce temporal decrease
    waveforms = waveforms.reshape(N, C, T).transpose(0, 2, 1)
    for wf in waveforms:
        denoise.enforce_decrease(wf, in_place=True)

    # un-temporal align
    if align:
        waveforms = denoise.invert_temporal_align(waveforms, rolls)

    return waveforms


# -- you may want to clean waveforms after the fact


def clean_waveforms(
    h5_path,
    tpca_rank=7,
    num_channels=20,
    trough_offset=42,
    batch_len_s=10,
    n_workers=1,
):
    @delayed
    def job(s_start):
        with h5py.File(h5_path, "r", swmr=True) as h5:
            s_end = s_start + batch_len_s * 30000
            t = h5["spike_index"][:, 0]

            # NB times are sorted
            which = np.flatnonzero((t >= s_start) & (t < s_end))
            bs = which[0]
            be = which[-1] + 1
            # cleaned_batch = batch_cleaned_waveforms(
            #     h5["residual"],
            #     h5["subtracted_waveforms"][bs:be],
            #     h5["spike_index"][bs:be] - [[h5["start_sample"][()], 0]],
            #     h5["first_channels"][bs:be],
            #     denoiser,
            #     tpca_rank,
            #     trough_offset,
            #     0,
            # )

            cleaned_batch = denoise.cleaned_waveforms(
                h5["subtracted_waveforms"][bs:be],
                h5["spike_index"][bs:be],
                h5["first_channels"][bs:be],
                h5["residual"],
                s_start=h5["start_sample"][()],
                pbar=False,
            )

            (
                cleaned_batch,
                firstchans_std,
                maxchans_std,
                chans_down,
            ) = waveform_utils.relativize_waveforms(
                cleaned_batch,
                h5["first_channels"][bs:be],
                None,
                h5["geom"][:],
                feat_chans=num_channels,
            )
            # print("m>f", (maxchans_std >= firstchans_std).all())
            # print("m=f+a", np.abs(maxchans_std - (firstchans_std + cleaned_batch.ptp(1).argmax(1))).max())
            # print(cleaned_batch.shape, be - bs)
            # print("mc", cleaned_batch[np.arange(be - bs), :, maxchans_std - firstchans_std]
            return bs, be, cleaned_batch, firstchans_std, maxchans_std

    with h5py.File(h5_path, "r+") as oh5:
        N, T, C = oh5["subtracted_waveforms"].shape
        cleaned_wfs = oh5.create_dataset(
            "cleaned_waveforms",
            shape=(N, T, num_channels),
            dtype=oh5["subtracted_waveforms"].dtype,
        )
        cmaxchans = oh5.create_dataset(
            "cleaned_max_channels",
            shape=N,
            dtype=np.int32,
        )
        cfirstchans = oh5.create_dataset(
            "cleaned_first_channels",
            shape=N,
            dtype=np.int32,
        )
        oh5.swmr_mode = True

        jobs = trange(
            oh5["start_sample"][()],
            oh5["end_sample"][()],
            batch_len_s * 30000,
            desc="Cleaning batches",
        )
        for batch in grouper(n_workers, jobs):
            for res in Parallel(n_workers)(job(bs) for bs in batch):
                bs, be, cleaned_batch, firstchans_std, maxchans_std = res
                cleaned_wfs[bs:be] = cleaned_batch
                cfirstchans[bs:be] = firstchans_std
                cmaxchans[bs:be] = maxchans_std


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


def make_channel_index(geom, radius, steps=1):
    """
    Compute an array whose whose ith row contains the ordered
    (by distance) neighbors for the ith channel
    """
    C = geom.shape[0]

    # get neighbors matrix
    neighbors = squareform(pdist(geom)) <= radius
    neighbors = n_steps_neigh_channels(neighbors, steps=steps)

    # max number of neighbors for all channels
    n_neighbors = np.max(np.sum(neighbors, 0))

    # FIXME: we are using C as a dummy value which is confusing, it may
    # be better to use something else, maybe np.nan
    # initialize channel index, initially with a dummy C value (a channel)
    # that does not exists
    channel_index = np.full((C, n_neighbors), C, dtype=np.int32)

    # fill every row in the matrix (one per channel)
    for current in range(C):
        # indexes of current channel neighbors
        neighbor_channels = np.where(neighbors[current])[0]

        # sort them by distance
        ch_idx, _ = order_channels_by_distance(
            current, neighbor_channels, geom
        )

        # fill entries with the sorted neighbor indexes
        channel_index[current, : ch_idx.shape[0]] = ch_idx

    return channel_index


# -- data loading helpers


def read_data(bin_file, dtype, s_start, s_end, n_channels):
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


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class timer:
    def __init__(self, name="timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")
