import h5py
import numpy as np
import torch

from collections import namedtuple
from ibllib.io.spikeglx import _geometry_from_meta, read_meta_data
from joblib import Parallel, delayed
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm.auto import trange

from . import denoise, voltage_detect


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
    ],
)


def subtraction_batch(
    s_start,
    batch_len_samples,
    T_samples,
    standardized_bin,
    thresholds,
    tpca_rank,
    trough_offset,
    channel_index,
    spike_length_samples,
    extract_channels,
    device,
):
    s_end = min(T_samples, s_start + batch_len_samples)

    # load denoiser (load here so that we load only once per batch)
    denoiser = denoise.SingleChanDenoiser()
    denoiser.load()
    denoiser.to(device)

    # load raw data with buffer
    buffer = spike_length_samples
    load_start = max(0, s_start - buffer)
    load_end = min(T_samples, s_end + buffer)
    residual = read_data(
        standardized_bin, np.float32, s_start, s_end, len(channel_index)
    )

    # 0 padding if we were at the edge of the data
    pad_left = pad_right = 0
    if load_start == 0:
        pad_left = buffer
    if load_end == T_samples:
        pad_right = buffer - (T_samples - s_end)
    if pad_left != 0 or pad_right != 0:
        residual = np.pad(residual, [(pad_left, pad_right), (0, 0)])

    # main subtraction loop
    subtracted_wfs = []
    spike_index = []
    firstchans = []
    for threshold in thresholds:
        subwfs, spind, fcs = detect_and_subtract(
            residual,
            threshold,
            tpca_rank,
            denoiser,
            trough_offset,
            channel_index,
            spike_length_samples,
            extract_channels,
            device,
        )
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

    # strip buffer from residual
    residual = residual[buffer:-buffer]

    # get cleaned waveforms
    cleaned_wfs = batch_cleaned_waveforms(
        residual,
        subtracted_wfs,
        spike_index,
        firstchans,
        denoiser,
        tpca_rank,
        trough_offset,
    )

    # time relative to batch start
    spike_index[:, 0] += s_start

    return SubtractionBatchResult(
        N_new=len(spike_index),
        s_start=s_start,
        s_end=s_end,
        residual=residual,
        subtracted_wfs=subtracted_wfs,
        cleaned_wfs=cleaned_wfs,
        firstchans=firstchans,
        spike_index=spike_index,
    )


def subtraction(
    standardized_bin,
    output_h5,
    geom=None,
    spatial_radius=70,
    tpca_rank=7,
    n_sec_chunk=1,
    sampling_rate=30_000,
    thresholds=[12, 10, 8, 6, 5, 4],
    extract_channels=40,
    spike_length_samples=121,
    trough_offset=42,
    n_jobs=1,
    device=None,
):
    standardized_bin = Path(standardized_bin)
    output_h5 = Path(output_h5)
    out_h5 = h5py.File(output_h5, "w", libver="latest")
    batch_len_samples = n_sec_chunk * sampling_rate

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

    # figure out length of data
    std = np.memmap(standardized_bin, mode="r", dtype=np.float32)
    std = std.reshape(-1, geom.shape[0])
    T_samples, n_channels = std.shape
    del std

    # compute helper data structures
    channel_index = make_channel_index(geom, spatial_radius, steps=2)

    # initialize resizable output datasets for waveforms etc
    residual = out_h5.create_dataset(
        "residual", shape=(T_samples, n_channels), dtype=np.float32
    )
    subtracted_wfs = out_h5.create_dataset(
        "subtracted_waveforms",
        chunks=(1024, spike_length_samples, extract_channels),
        shape=(1, spike_length_samples, extract_channels),
        maxshape=(None, spike_length_samples, extract_channels),
        dtype=np.float32,
    )
    cleaned_wfs = out_h5.create_dataset(
        "cleaned_waveforms",
        chunks=(1024, spike_length_samples, extract_channels),
        shape=(1, spike_length_samples, extract_channels),
        maxshape=(None, spike_length_samples, extract_channels),
        dtype=np.float32,
    )
    firstchans = out_h5.create_dataset(
        "first_channels",
        chunks=(1024,),
        shape=(1,),
        maxshape=(None,),
        dtype=np.int32,
    )
    spike_index = out_h5.create_dataset(
        "spike_index",
        chunks=(1024, 2),
        shape=(1, 2),
        maxshape=(None, 2),
        dtype=np.int64,
    )

    # now run subtraction in parallel
    N = 0  # how many have we detected so far?
    for result in Parallel(n_jobs)(
        delayed(subtraction_batch)(
            s_start,
            batch_len_samples,
            T_samples,
            standardized_bin,
            thresholds,
            tpca_rank,
            trough_offset,
            channel_index,
            spike_length_samples,
            extract_channels,
            device,
        )
        for s_start in trange(
            0, T_samples, batch_len_samples, desc="Subtracting batches"
        )
    ):
        # grow arrays as necessary
        N_new = result.N_new
        subtracted_wfs.resize(N + N_new, axis=0)
        cleaned_wfs.resize(N + N_new, axis=0)
        firstchans.resize(N + N_new, axis=0)
        spike_index.resize(N + N_new, axis=0)

        # write results
        residual[result.s_start : result.s_end] = result.residual
        subtracted_wfs[N : N + N_new] = result.subtracted_wfs
        cleaned_wfs[N : N + N_new] = result.cleaned_wfs
        firstchans[N : N + N_new] = result.firstchans
        spike_index[N : N + N_new] = result.spike_index

        N += N_new


# -- denoising / detection helpers


@torch.inference_mode()
def full_denoising(waveforms, tpca_rank, denoiser=None):
    N, T, C = waveforms.shape

    # Apply NN denoiser (skip if None)
    waveforms = waveforms.transpose(0, 2, 1).reshape(N * C, T)
    if denoiser is not None:
        waveforms = denoiser(torch.tensor(waveforms, device=denoiser.device))
        waveforms = waveforms.cpu().numpy()

    # Temporal PCA while we are still transposed
    tpca = PCA(tpca_rank)
    waveforms = tpca.fit_transform(waveforms)
    waveforms = tpca.inverse_transform(waveforms)

    # Un-transpose, enforce temporal decrease
    waveforms = waveforms.reshape(N, C, T).transpose(0, 2, 1)
    for wf in waveforms:
        denoise.enforce_decrease(wf, in_place=True)

    return waveforms


def detect_and_subtract(
    raw,
    threshold,
    tpca_rank,
    denoiser,
    trough_offset,
    channel_index,
    spike_length_samples,
    extract_channels,
    device,
):
    """This subtracts from raw in place, leaving the residual behind"""
    spike_index, energy = voltage_detect.detect_and_deduplicate(
        raw, threshold, channel_index, spike_length_samples, device
    )
    # it would be nice to go in order but we would need to fit the
    # TPCA to something other than the subtracted waveforms (could
    # probably just fit it to denoised raw)
    # subtraction_order = np.argsort(energy)[::-1]

    # how many channels down from max channel?
    chans_down = extract_channels // 2
    chans_down -= chans_down % 2

    # allocate output storage
    subtracted_wfs = np.empty(
        (len(spike_index), spike_length_samples, extract_channels),
        dtype=raw.dtype,
    )
    firstchans = np.empty(len(spike_index), dtype=np.int32)

    # extraction loop
    for i in range(len(spike_index)):
        t, mc = spike_index[i]
        mc_idx = mc - mc % 2
        fc = mc_idx - chans_down

        subtracted_wfs[i] = raw[
            t - trough_offset : t - trough_offset + spike_length_samples,
            fc : fc + extract_channels,
        ]
        firstchans[i] = fc

    # denoising
    subtracted_wfs = full_denoising(subtracted_wfs, tpca_rank, denoiser)

    # the actual subtraction
    for wf, (t, mc), fc in zip(subtracted_wfs, spike_index, firstchans):
        raw[
            t - trough_offset : t - trough_offset + spike_length_samples,
            fc : fc + extract_channels,
        ] -= wf

    return subtracted_wfs, spike_index, firstchans


def batch_cleaned_waveforms(
    residual,
    subtracted_wfs,
    spike_index,
    firstchans,
    denoiser,
    tpca_rank,
    trough_offset,
):
    N, T, C = subtracted_wfs.shape

    # Add residuals to subtracted wfs
    cleaned_waveforms = subtracted_wfs.copy()
    for n, ((t, mc), fc) in enumerate(zip(spike_index, firstchans)):
        cleaned_waveforms[n] += residual[
            t - trough_offset : t - trough_offset + T, fc : fc + C
        ]

    # Denoise and return
    return full_denoising(cleaned_waveforms, tpca_rank, denoiser)


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
    C = neighbors_matrix.shape[0]

    # each channel is its own neighbor (diagonal of trues)
    output = np.eye(C, dtype="bool")

    # for every step
    for _ in range(steps):

        # go trough every channel
        for current in range(C):
            # neighbors of the current channel
            neighbors_current = output[current]
            # get the neighbors of all the neighbors of the current channel
            neighbors_of_neighbors = neighbors_matrix[neighbors_current]
            # sub over rows and convert to bool, this will turn to true entries
            # where at least one of the neighbors has each channel as its
            # neighbor
            is_neighbor_of_neighbor = np.sum(
                neighbors_of_neighbors, axis=0
            ).astype("bool")
            # set the channels that are neighbors to true
            output[current][is_neighbor_of_neighbor] = True

    return output


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
    offset = s_start * dtype.itemsize * n_channels
    with open(bin_file, "rb") as fin:
        data = np.fromfile(
            fin,
            dtype=dtype,
            count=(s_end - s_start) * n_channels,
            offset=offset,
        )
    data = data.reshape(-1, n_channels)
    return data
