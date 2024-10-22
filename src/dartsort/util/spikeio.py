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
    return_scaled=False,
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert times_samples.max() <= recording.get_num_samples() - (
        spike_length_samples - trough_offset_samples
    )
    n_channels = recording.get_num_channels()
    n_spikes = times_samples.size
    read_times = times_samples - trough_offset_samples

    if not return_scaled and recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_length=1
    ):
        # fast path. this is like 2x as fast as the read_traces for loop
        # below, but requires a recording on disk in a nice format
        binary_path = recording.get_binary_description()["file_paths"][0]
        return _read_full_waveforms_binary(
            binary_path,
            read_times,
            n_channels=n_channels,
            dtype=recording.dtype,
            spike_length_samples=spike_length_samples,
        )

    waveforms = np.empty(
        (n_spikes, spike_length_samples, n_channels), dtype=recording.dtype
    )
    for i, t in enumerate(read_times):
        waveforms[i] = recording.get_traces(
            0, start_frame=t, end_frame=t + spike_length_samples, return_scaled=return_scaled
        )

    return waveforms


def _read_full_waveforms_binary(
    binary_path,
    read_times_samples,
    n_channels,
    dtype,
    spike_length_samples=121,
):
    n_spikes = read_times_samples.size
    waveforms = np.empty((n_spikes, spike_length_samples, n_channels), dtype=dtype)
    offsets = read_times_samples * np.dtype(dtype).itemsize * n_channels
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
    assert times_samples.max() <= recording.get_num_samples() - (
        spike_length_samples - trough_offset_samples
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
            0,
            start_frame=t,
            end_frame=t + spike_length_samples,
            channel_ids=load_channel_ids,
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
    assert times_samples.max() <= recording.get_num_samples() - (
        spike_length_samples - trough_offset_samples
    )
    n_channels = recording.get_num_channels()

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_length=1
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
    waveforms = np.full(
        (n_spikes, spike_length_samples, channel_index.shape[1]),
        fill_value,
        dtype=recording.dtype,
    )
    read_times = times_samples - trough_offset_samples
    for i, t in enumerate(read_times):
        chans = channel_index[main_channels[i]]
        good = chans < n_channels
        load_channel_ids = recording.channel_ids[chans[good]]
        waveforms[i, :, good] = recording.get_traces(
            0,
            start_frame=t,
            end_frame=t + spike_length_samples,
            channel_ids=load_channel_ids,
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
        (n_spikes, spike_length_samples, channel_index.shape[1]),
        fill_value,
        dtype=dtype,
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
    assert times_samples.max() <= recording.get_num_samples() - (
        spike_length_samples - trough_offset_samples
    )
    n_channels = recording.get_num_channels()
    assert channels.max() < n_channels
    assert channels.min() >= 0

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_length=1
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
    waveforms = np.full(
        (n_spikes, spike_length_samples), fill_value, dtype=recording.dtype
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
    waveforms = np.full((n_spikes, spike_length_samples), fill_value, dtype=dtype)
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


def get_read_chunks(read_times, spike_length=121, max_chunk=512):
    chunk_ranges = np.zeros((len(read_times), 2), dtype=int)
    chunk_indices = []
    chunk_to_waveform_indexers = []

    nchunks = 0
    chunk_start_i = 0
    chunk_start_time = chunk_end_time = read_times[0]
    chunk_len = 1
    chunk_wf_starts = np.zeros(max_chunk, dtype=int)
    time_ix = np.arange(spike_length)[None, :]
    for i, t in enumerate(read_times[1:], start=1):
        if t - chunk_end_time > spike_length or chunk_len == max_chunk:
            # current chunk is done
            # finalize it
            chunk_ranges[nchunks, 0] = chunk_start_time
            chunk_ranges[nchunks, 1] = chunk_end_time + spike_length
            chunk_indices.append(range(chunk_start_i, i))
            chunk_to_waveform_indexers.append(chunk_wf_starts[:chunk_len, None].copy())
            nchunks += 1

            # start new chunk
            chunk_start_i = i
            chunk_start_time = chunk_end_time = t
            chunk_len = 1
        else:
            # grow current chunk
            print(f"grow {chunk_start_time=} {t=} {(t-chunk_start_time)=}")
            chunk_end_time = t
            chunk_wf_starts[chunk_len] = t - chunk_start_time
            chunk_len += 1

    # final finalize:
    chunk_ranges[nchunks, 0] = chunk_start_time
    chunk_ranges[nchunks, 1] = chunk_end_time + spike_length
    chunk_indices.append(range(chunk_start_i, i + 1))
    chunk_to_waveform_indexers.append(chunk_wf_starts[:chunk_len, None])
    nchunks += 1

    chunk_ranges = chunk_ranges[:nchunks]
    return chunk_ranges, chunk_indices, chunk_to_waveform_indexers, time_ix


def read_full_waveforms_chunked(
    recording,
    times_samples,
    trough_offset_samples=42,
    spike_length_samples=121,
    max_chunk=512,
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert times_samples.max() <= recording.get_num_samples() - (
        spike_length_samples - trough_offset_samples
    )
    n_channels = recording.get_num_channels()
    n_spikes = times_samples.size
    read_times = times_samples - trough_offset_samples

    chunk_ranges, chunk_indices, chunk_to_waveform_indexers, time_ix = get_read_chunks(
        read_times, spike_length=spike_length_samples, max_chunk=max_chunk
    )

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_lenght=1
    ):
        # fast path. this is like 2x as fast as the read_traces for loop
        # below, but requires a recording on disk in a nice format
        binary_path = recording.get_binary_description()["file_paths"][0]
        return _read_full_waveforms_binary_chunked(
            binary_path,
            read_times.size,
            chunk_ranges,
            chunk_indices,
            chunk_to_waveform_indexers,
            time_ix,
            n_channels=n_channels,
            dtype=recording.dtype,
        )

    waveforms = np.empty(
        (n_spikes, spike_length_samples, n_channels), dtype=recording.dtype
    )
    for (cs, ce), ix, tix in zip(
        chunk_ranges, chunk_indices, chunk_to_waveform_indexers
    ):
        waveforms[ix] = recording.get_traces(0, start_frame=cs, end_frame=ce)[
            tix + time_ix
        ]

    return waveforms


def _read_full_waveforms_binary_chunked(
    binary_path,
    n_spikes,
    chunk_ranges,
    chunk_indices,
    chunk_to_waveform_indexers,
    time_ix,
    n_channels,
    dtype,
):
    waveforms = np.empty((n_spikes, time_ix.size, n_channels), dtype=dtype)
    chunk_start_offsets = chunk_ranges[:, 0] * np.dtype(dtype).itemsize * n_channels
    with open(binary_path, "rb") as binary:
        for offset, (cs, ce), ix, tix in zip(
            chunk_start_offsets, chunk_ranges, chunk_indices, chunk_to_waveform_indexers
        ):
            binary.seek(offset, SEEK_SET)
            read_len_samples = ce - cs
            waveforms[ix] = np.fromfile(
                binary,
                dtype=dtype,
                count=read_len_samples * n_channels,
            ).reshape(read_len_samples, n_channels)[tix + time_ix]
    return waveforms


def read_waveforms_channel_index_chunked(
    recording,
    times_samples,
    channel_index,
    main_channels,
    trough_offset_samples=42,
    spike_length_samples=121,
    fill_value=np.nan,
    max_chunk=512,
):
    assert times_samples.ndim == 1
    assert times_samples.size > 0
    assert times_samples.dtype.kind == "i"
    assert times_samples.min() >= trough_offset_samples
    assert times_samples.max() <= recording.get_num_samples() - (
        spike_length_samples - trough_offset_samples
    )
    n_channels = recording.get_num_channels()
    n_spikes = times_samples.size
    read_times = times_samples - trough_offset_samples

    if recording.binary_compatible_with(
        file_offset=0, time_axis=0, file_paths_length=1
    ):
        # fast path. this is like 2x as fast as the read_traces for loop
        # below, but requires a recording on disk in a nice format
        chunk_ranges, chunk_indices, chunk_to_waveform_indexers, time_ix = (
            get_read_chunks(
                read_times, spike_length=spike_length_samples, max_chunk=max_chunk
            )
        )
        binary_path = recording.get_binary_description()["file_paths"][0]
        return _read_waveforms_binary_channel_index_chunked(
            binary_path,
            n_spikes,
            chunk_ranges,
            chunk_indices,
            chunk_to_waveform_indexers,
            time_ix,
            channel_index=channel_index,
            main_channels=main_channels,
            n_channels=n_channels,
            dtype=recording.dtype,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            fill_value=fill_value,
        )

    waveforms = np.full(
        (n_spikes, spike_length_samples, channel_index.shape[1]),
        fill_value,
        dtype=recording.dtype,
    )
    for i, t in enumerate(read_times):
        chans = channel_index[main_channels[i]]
        good = chans < n_channels
        load_channel_ids = recording.channel_ids[chans[good]]
        waveforms[i, :, good] = recording.get_traces(
            0,
            start_frame=t,
            end_frame=t + spike_length_samples,
            channel_ids=load_channel_ids,
        ).T

    return waveforms


def _read_waveforms_binary_channel_index_chunked(
    binary_path,
    n_spikes,
    chunk_ranges,
    chunk_indices,
    chunk_to_waveform_indexers,
    time_ix,
    n_channels,
    dtype,
    channel_index,
    main_channels,
    trough_offset_samples=42,
    spike_length_samples=121,
    fill_value=np.nan,
):
    waveforms = np.full(
        (n_spikes, spike_length_samples, channel_index.shape[1]),
        fill_value,
        dtype=dtype,
    )
    chunk_start_offsets = chunk_ranges[:, 0] * np.dtype(dtype).itemsize * n_channels
    with open(binary_path, "rb") as binary:
        for offset, (cs, ce), ix, tix in zip(
            chunk_start_offsets, chunk_ranges, chunk_indices, chunk_to_waveform_indexers
        ):
            binary.seek(offset, SEEK_SET)
            read_len_samples = ce - cs
            wfs = np.fromfile(
                binary,
                dtype=dtype,
                count=read_len_samples * n_channels,
            ).reshape(read_len_samples, n_channels)
            for i, tt in zip(ix, tix):
                chans = channel_index[main_channels[i]]
                good = chans < n_channels
                # print(f"{i=}")
                # print(f"{tt=}")
                waveforms[i, :, good] = wfs[
                    (tt + time_ix.ravel())[:, None], chans[good][None]
                ].T
    return waveforms
