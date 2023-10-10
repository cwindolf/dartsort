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
        < recording.get_num_samples()
        - (spike_length_samples - trough_offset_samples)
    )
    n_channels = recording.get_num_channels()
    n_channels = recording.get_num_channels()

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1
    ):
        # fast path (with spikeinterface typo). this is like 2x as fast
        # as the read_traces for loop below, but requires a recording on disk
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
        < recording.get_num_samples()
        - (spike_length_samples - trough_offset_samples)
    )
    n_channels = recording.get_num_channels()
    assert load_channels.size <= n_channels

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1
    ):
        # fast path (with spikeinterface typo). this is like 2x as fast
        # as the read_traces for loop below, but requires a recording on disk
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
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert times_samples.min() >= trough_offset_samples
    assert (
        times_samples.max()
        < recording.get_num_samples()
        - (spike_length_samples - trough_offset_samples)
    )
    n_channels = recording.get_num_channels()

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1  # sic
    ):
        # fast path (with spikeinterface typo). this is like 2x as fast
        # as the read_traces for loop below, but requires a recording on disk
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
        )

    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples, channel_index.shape[1]), dtype=recording.dtype
    )
    read_times = times_samples - trough_offset_samples
    for i, t in enumerate(read_times):
        load_channel_ids = recording.channel_ids[channel_index[main_channels[i]]]
        waveforms[i] = recording.get_traces(
            0, start_frame=t, end_frame=t + spike_length_samples, channel_ids=load_channel_ids
        )

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
):
    n_spikes = times_samples.size
    waveforms = np.empty(
        (n_spikes, spike_length_samples, channel_index.shape[1]), dtype=dtype
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
            waveforms[i] = wf[:, channel_index[main_channels[i]]]
    return waveforms
