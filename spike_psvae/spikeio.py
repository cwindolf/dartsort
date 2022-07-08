"""A library for quickly reading spike data from .bin files."""
from pathlib import Path
import numpy as np


def get_binary_length(input_bin, n_channels, sampling_rate, nsync=0, dtype=np.float32):
    """How long is this binary file in samples and seconds?"""
    bin_size = Path(input_bin).stat().st_size
    assert not bin_size % np.dtype(dtype).itemsize
    bin_size = bin_size // np.dtype(dtype).itemsize
    assert not bin_size % (nsync + n_channels)
    T_samples = bin_size // (nsync + n_channels)
    T_sec = T_samples / sampling_rate
    return T_samples, T_sec


def read_data(bin_file, dtype, s_start, s_end, n_channels, nsync=0):
    """Read a chunk of a binary file

    Reads a temporal chunk on all channels: so, this is for loading a
    T x num_channels_total chunk.

    Arguments
    ---------
    bin_file : string or Path
    dtype : numpy datatype
        The type of data stored in bin_file (and the output type)
    s_start, s_end : int
        Start and end samples of region to load
    n_channels : int
        Number of channels saved in this binary file.

    Returns
    -------
    data : np.array of shape (s_end - s_start, n_channels)
    """
    offset = s_start * np.dtype(dtype).itemsize * (n_channels + nsync)
    with open(bin_file, "rb") as fin:
        data = np.fromfile(
            fin,
            dtype=dtype,
            count=(s_end - s_start) * (n_channels + nsync),
            offset=offset,
        )
    data = data.reshape(-1, n_channels + nsync)[:, :n_channels]
    return data


# TODO: read_waveforms, read_maxchan_traces, read_local_waveforms
