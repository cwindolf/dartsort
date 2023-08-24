# %%
"""A library for quickly reading spike data from .bin files."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from os import SEEK_SET


# %%
def get_binary_length_samples(
    input_bin, n_channels, nsync=0, dtype=np.float32
):
    """How long is this binary file in samples?"""
    bin_size = Path(input_bin).stat().st_size
    assert not bin_size % np.dtype(dtype).itemsize
    bin_size = bin_size // np.dtype(dtype).itemsize
    assert not bin_size % (nsync + n_channels)
    T_samples = bin_size // (nsync + n_channels)
    return T_samples


# %%
def get_binary_length(
    input_bin, n_channels, sampling_rate, nsync=0, dtype=np.float32
):
    """How long is this binary file in samples and seconds?"""
    bin_size = Path(input_bin).stat().st_size
    if bin_size % np.dtype(dtype).itemsize:
        raise ValueError(f"Binary file cannot be of dtype {dtype}.")
    bin_size = bin_size // np.dtype(dtype).itemsize
    if bin_size % (nsync + n_channels):
        raise ValueError(
            f"Binary file cannot have {n_channels} channels with {nsync} sync chans."
        )
    T_samples = bin_size // (nsync + n_channels)
    T_sec = T_samples / sampling_rate
    return T_samples, T_sec


# %%
def read_data(
    bin_file, dtype, s_start, s_end, n_channels, nsync=0, out_dtype=None
):
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
    out_dtype = dtype if out_dtype is None else out_dtype
    offset = np.int64(s_start) * np.dtype(dtype).itemsize * (n_channels + nsync)
    with open(bin_file, "rb") as fin:
        data = np.fromfile(
            fin,
            dtype=dtype,
            count=(s_end - s_start) * (n_channels + nsync),
            offset=offset,
        )
    data = data.reshape(-1, n_channels + nsync)[:, :n_channels]
    data = data.astype(out_dtype)
    return data


# %%
def read_waveforms_in_memory(
    array,
    spike_index,
    spike_length_samples,
    channel_index,
    trough_offset=42,
    buffer=0,
):
    """Load waveforms from an array in memory"""
    # pad with NaN to fill resulting waveforms with NaN when
    # channel is outside probe
    if torch.is_tensor(array):
        padded_array = F.pad(array, (0, 1), value=np.nan)
    else:
        padded_array = np.pad(array, [(0, 0), (0, 1)], constant_values=np.nan)
    # times relative to trough + buffer
    time_range = np.arange(
        buffer - trough_offset,
        buffer + spike_length_samples - trough_offset,
    )
    time_ix = spike_index[:, 0, None] + time_range[None, :]
    chan_ix = channel_index[spike_index[:, 1]]
    waveforms = padded_array[time_ix[:, :, None], chan_ix[:, None, :]]
    return waveforms


# %%
def read_waveforms(
    trough_times,
    bin_file,
    n_channels,
    channel_index=None,
    max_channels=None,
    channels=None,
    trough_offset=42,
    spike_length_samples=121,
    dtype=np.float32,
    dtype_output=np.float32,
    fill_value=np.nan,
    buffer=None,
):
    """Read waveforms from binary file

    Load either waveforms on the full probe, or on a subset of channels
    if max_channels and channel_index are not None.

    This one figures out in advance which reads will be impossible,
    avoiding our usual try/except.

    Arguments
    ---------
    trough_times : int array
    bin_file : str or Path
        Path to binary file of dtype `dtype` with `n_channels` channels.
    n_channels : int
        Number of channels on the probe.
    channel_index : None or array
        A channel index as created by one of the functions in `subtract`
    max_channels : None or int array
        The detection channels for the spikes, used to look up the
        channels subset to load in `channel_index`
    channels : None or int array
        Just read data on these channels. (Don't use this argument and
        channel_index together.) This can be a 1d set of channels,
        or a 2d array giving an arbitrary set of channels to load
        for each spike.
    spike_length_samples, trough_offset : int
    dtype : numpy dtype
        dtype stored in bin_file and returned from this function.
    fill_value : any value of dtype
        If a spike is loaded on a smaller channel neighborhood, this value
        will fill in the blank space in the array.

    Returns
    -------
    waveforms : (N,T,C) array
    skipped_ix : int array
        Which indices could not be loaded, if any.
    """
    T_samples = get_binary_length_samples(bin_file, n_channels, dtype=dtype)
    N = trough_times.shape[0]
    load_channels = n_channels
    load_ci = load_chans = False

    bin_file = Path(bin_file)
    assert bin_file.exists()

    if max_channels is not None:
        assert max_channels.shape == trough_times.shape
        if channel_index is None:
            raise ValueError(
                "If loading a subset of channels depending on the max "
                "channel, please supply `channel_index`."
            )
        if channels is not None:
            raise ValueError("Pass channel_index or channels, but not both.")

        load_channels = channel_index.shape[1]
        load_ci = True

    if channels is not None:
        channels = np.atleast_1d(channels)
        if channels.ndim == 1:
            channels = channels[None, :]
        assert channels.ndim == 2
        assert channels.shape[0] in (1, N)
        load_channels = channels.shape[1]
        load_chans = True

    # figure out which loads will be skipped in advance
    max_load_time = T_samples - spike_length_samples + trough_offset
    # this can be sped up with a searchsorted if times are sorted...
    skipped_idx = np.flatnonzero(
        (trough_times < trough_offset) | (trough_times > max_load_time)
    )
    kept_idx = np.setdiff1d(np.arange(N), skipped_idx)
    N_load = N - len(skipped_idx)

    # allocate output space
    if buffer is not None:
        waveforms = buffer[:N_load]
        assert waveforms.shape == (N_load, spike_length_samples, load_channels)
    else:
        waveforms = np.empty(
            (N_load, spike_length_samples, load_channels),
            dtype=dtype,
        )

    load_times = trough_times - trough_offset
    offsets = (
        load_times.astype(np.int64) * np.dtype(dtype).itemsize * n_channels
    )
    with open(bin_file, "rb") as fin:
        for i, spike_ix in enumerate(kept_idx):
            fin.seek(offsets[spike_ix], SEEK_SET)
            wf = np.fromfile(
                fin,
                dtype=dtype,
                count=spike_length_samples * n_channels,
            ).reshape(spike_length_samples, n_channels)

            if load_ci:
                if dtype==np.int16 or dtype==np.int32: #otherwise cannot fill with nan values
                    wf = wf.astype(np.float32)
                wf = np.pad(wf, [(0, 0), (0, 1)], constant_values=fill_value)
                wf = wf[:, channel_index[max_channels[spike_ix]]]
            elif load_chans:
                if dtype==np.int16 or dtype==np.int32:
                    wf = wf.astype(np.float32)
                wf = np.pad(wf, [(0, 0), (0, 1)], constant_values=fill_value)
                wf = wf[:, channels[spike_ix % channels.shape[0]]]

            waveforms[i] = wf

    return waveforms.astype(dtype_output), skipped_idx


# %%
def read_maxchan_traces(
    spike_index,
    bin_file,
    n_channels,
    dtype=np.float32,
    trough_offset=42,
    spike_length_samples=121,
):
    """Helper to read waveforms on their main channel."""
    maxchan_channel_index = np.arange(n_channels)[:, None]
    return read_waveforms(
        spike_index[:, 0],
        bin_file,
        n_channels,
        channel_index=maxchan_channel_index,
        max_channels=spike_index[:, 1],
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        dtype=dtype,
    )
