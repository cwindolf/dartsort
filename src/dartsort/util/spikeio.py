"""Utilities for reading waveforms from recordings

If you want to read many waveforms and store them in a file, also
consider peel.GrabAndLocalize, which can do this and optionally
also featurize the waveforms
"""

from os import SEEK_SET

import numpy as np


def read_full_waveforms(
    recording,
    times_samples,
    trough_offset_samples=42,
    spike_length_samples=121,
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert (
        times_samples.max()
        <= recording.get_num_samples()
        - (spike_length_samples - trough_offset_samples)
    )
    n_channels = recording.get_num_channels()
    n_channels = recording.get_num_channels()

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1
    ):
        # fast path. this is like 2x as fast as the read_traces for loop
        # below, but requires a recording on disk in a nice format
        binary_path = recording.get_binary_description()["file_paths"][0]
        return _read_full_waveforms_binary(
            binary_path,
            times_samples,
            n_channels=n_channels,
            dtype=recording.dtype,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )

    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples, n_channels), dtype=recording.dtype
    )
    read_times = times_samples - trough_offset_samples
    for i, t in enumerate(read_times):
        waveforms[i] = recording.get_traces(
            0, start_frame=t, end_frame=t + spike_length_samples
        )

    return waveforms


def _read_full_waveforms_binary(
    binary_path,
    times_samples,
    n_channels,
    dtype,
    trough_offset_samples=42,
    spike_length_samples=121,
):
    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples, n_channels), dtype=dtype
    )
    load_times = times_samples - trough_offset_samples
    offsets = load_times * np.dtype(dtype).itemsize * n_channels
    with open(binary_path, "rb") as binary:
        for i, offset in enumerate(offsets):
            binary.seek(offset, SEEK_SET)
            waveforms[i] = np.fromfile(
                binary,
                dtype=dtype,
                count=spike_length_samples * n_channels,
            ).reshape(spike_length_samples, n_channels)
    return waveforms


def read_subset_waveforms(
    recording,
    times_samples,
    load_channels,
    trough_offset_samples=42,
    spike_length_samples=121,
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert (
        times_samples.max()
        <= recording.get_num_samples()
        - (spike_length_samples - trough_offset_samples)
    )
    n_channels = recording.get_num_channels()
    assert load_channels.size <= n_channels

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1
    ):
        # fast path. this is like 2x as fast as the read_traces for loop
        # below, but requires a recording on disk in a nice format
        binary_path = recording.get_binary_description()["file_paths"][0]
        return _read_subset_waveforms_binary(
            binary_path,
            times_samples,
            load_channels=load_channels,
            n_channels=n_channels,
            dtype=recording.dtype,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )

    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples, load_channels.size), dtype=recording.dtype
    )
    read_times = times_samples - trough_offset_samples
    load_channel_ids = recording.channel_ids[load_channels]
    for i, t in enumerate(read_times):
        waveforms[i] = recording.get_traces(
            0, start_frame=t, end_frame=t + spike_length_samples, channel_ids=load_channel_ids
        )

    return waveforms


def _read_subset_waveforms_binary(
    binary_path,
    times_samples,
    n_channels,
    dtype,
    load_channels,
    trough_offset_samples=42,
    spike_length_samples=121,
):
    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples, load_channels.size), dtype=dtype
    )
    load_times = times_samples - trough_offset_samples
    offsets = load_times * np.dtype(dtype).itemsize * n_channels
    with open(binary_path, "rb") as binary:
        for i, offset in enumerate(offsets):
            binary.seek(offset, SEEK_SET)
            waveforms[i] = np.fromfile(
                binary,
                dtype=dtype,
                count=spike_length_samples * n_channels,
            ).reshape(spike_length_samples, n_channels)[:, load_channels]
    return waveforms


def read_waveforms_channel_index(
    recording,
    times_samples,
    channel_index,
    main_channels,
    trough_offset_samples=42,
    spike_length_samples=121,
    fill_value=np.nan,
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert times_samples.min() >= trough_offset_samples
    assert (
        times_samples.max()
        <= recording.get_num_samples()
        - (spike_length_samples - trough_offset_samples)
    )
    n_channels = recording.get_num_channels()

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1  # sic
    ):
        # fast path. this is like 2x as fast as the read_traces for loop
        # below, but requires a recording on disk in a nice format
        binary_path = recording.get_binary_description()["file_paths"][0]
        return _read_waveforms_binary_channel_index(
            binary_path,
            times_samples,
            channel_index=channel_index,
            main_channels=main_channels,
            n_channels=n_channels,
            dtype=recording.dtype,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            fill_value=fill_value,
        )

    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples, channel_index.shape[1]), dtype=recording.dtype
    )
    read_times = times_samples - trough_offset_samples
    for i, t in enumerate(read_times):
        chans = channel_index[main_channels[i]]
        good = chans < n_channels
        load_channel_ids = recording.channel_ids[chans[good]]
        waveforms[i, :, good] = recording.get_traces(
            0, start_frame=t, end_frame=t + spike_length_samples, channel_ids=load_channel_ids
        ).T

    return waveforms


def _read_waveforms_binary_channel_index(
    binary_path,
    times_samples,
    n_channels,
    dtype,
    channel_index,
    main_channels,
    trough_offset_samples=42,
    spike_length_samples=121,
    fill_value=np.nan,
):
    n_spikes = times_samples.size
    waveforms = np.full(
        (n_spikes, spike_length_samples, channel_index.shape[1]), fill_value, dtype=dtype
    )
    load_times = times_samples - trough_offset_samples
    offsets = load_times * np.dtype(dtype).itemsize * n_channels
    with open(binary_path, "rb") as binary:
        for i, offset in enumerate(offsets):
            binary.seek(offset, SEEK_SET)
            wf = np.fromfile(
                binary,
                dtype=dtype,
                count=spike_length_samples * n_channels,
            ).reshape(spike_length_samples, n_channels)
            chans = channel_index[main_channels[i]]
            good = chans < n_channels
            waveforms[i, :, good] = wf.T[chans[good]]
    return waveforms


def read_single_channel_waveforms(
    recording,
    times_samples,
    channels,
    trough_offset_samples=42,
    spike_length_samples=121,
    fill_value=np.nan,
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert times_samples.min() >= trough_offset_samples
    assert (
        times_samples.max()
        <= recording.get_num_samples()
        - (spike_length_samples - trough_offset_samples)
    )
    n_channels = recording.get_num_channels()
    assert channels.max() < n_channels
    assert channels.min() >= 0

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1  # sic
    ):
        # fast path. this is like 2x as fast as the read_traces for loop
        # below, but requires a recording on disk in a nice format
        binary_path = recording.get_binary_description()["file_paths"][0]
        return _read_single_channel_waveforms(
            binary_path,
            times_samples,
            channels=channels,
            n_channels=n_channels,
            dtype=recording.dtype,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            fill_value=fill_value,
        )

    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples), dtype=recording.dtype
    )
    read_times = times_samples - trough_offset_samples
    for i, (t, c) in enumerate(zip(read_times, channels)):
        waveforms[i, :] = recording.get_traces(
            0, start_frame=t, end_frame=t + spike_length_samples, channel_ids=c
        ).T

    return waveforms


def _read_single_channel_waveforms(
    binary_path,
    times_samples,
    channels,
    n_channels,
    dtype,
    trough_offset_samples=42,
    spike_length_samples=121,
    fill_value=np.nan,
):
    n_spikes = times_samples.size
    waveforms = np.full(
        (n_spikes, spike_length_samples), fill_value, dtype=dtype
    )
    load_times = times_samples - trough_offset_samples
    offsets = load_times * np.dtype(dtype).itemsize * n_channels
    with open(binary_path, "rb") as binary:
        for i, (offset, chan) in enumerate(zip(offsets, channels)):
            binary.seek(offset, SEEK_SET)
            wf = np.fromfile(
                binary,
                dtype=dtype,
                count=spike_length_samples * n_channels,
            ).reshape(spike_length_samples, n_channels)
            waveforms[i, :] = wf[:, chan]
    return waveforms
