import numpy as np
import signal as signal_
from scipy import signal
from scipy import ndimage
from scipy.stats import zscore
from tqdm.auto import trange
from neurodsp.utils import rms
from neurodsp import voltage
from pathlib import Path
import matplotlib.pyplot as plt

from spike_psvae.spikeio import get_binary_length, read_data


def run_preprocessing(
    raw_bin,
    out_bin,
    geom=None,
    fs=30_000,
    n_channels=384,
    extra_channels=0,
    ptp_len=0,
    bp=(300, 2000),
    order=3,
    decorr_iter=0,
    resample_to=None,
    lfp_destripe=False,
    csd=False,
    pad_for_filter=None,
    chunk_seconds=5,
    do_filter=True,
    standardize="perchan",
    in_dtype=np.int16,
    rmss=None,
    avg_depth=False,
    t_start=0,
    t_end=None,
    debug_imshow=False,
):
    """Just preprocessing. For spike detection, use method below.

    The pipeline is: bandpass, alternating zscore decorr, peak
    to peak.

    Parameters
    ----------
    in_key, out_key : str
        Dataset to load, dataset to write.
    ptp_len : int
        Length of peak to peak filter
    bp : pair of int
        Bandpass frequencies
    order : int
        Butterworth filter order (effectively doubled by filtfilt)
    decorr_iter : int
        Number of alternating zscore iterations
    pad_for_filter : int
        We'll draw extra big chunks to avoid boundary effects from
        the bandpass filter
    """
    assert standardize in (None, "none", "perchan", "global")
    T_samples, T_seconds = get_binary_length(raw_bin, n_channels + extra_channels, fs, dtype=in_dtype)
    print("T_samples", T_samples, "T_seconds", T_seconds)

    s_start = t_start * fs
    s_end = t_end * fs if t_end is not None else T_samples
    assert 0 <= s_start < s_end <= T_samples

    # preprocessed chunk factory
    get_chunk = make_chunk_preprocessor(
        raw_bin,
        fs,
        n_channels,
        extra_channels,
        T_samples,
        geom,
        ptp_len=ptp_len,
        bp=bp,
        order=order,
        decorr_iter=decorr_iter,
        pad_for_filter=pad_for_filter,
        dtype=in_dtype,
        out_dtype=np.float32,
        resample_to=resample_to,
        lfp_destripe=lfp_destripe,
        csd=csd,
        avg_depth=avg_depth,
    )

    # Create output
    if do_filter:
        rmss = []
        with open(out_bin, "wb") as out:
            for i, s in enumerate(trange(0, T_samples, fs * chunk_seconds, desc="filter")):
                chunk = get_chunk(s, min(T_samples, s + fs * chunk_seconds))

                with noint:
                    rmss.append(rms(chunk.T))
                    chunk.tofile(out)

                if debug_imshow and not i % debug_imshow:
                    fig = plt.figure()
                    plt.imshow(chunk.T, aspect=chunk.shape[1] / chunk.shape[0])
                    plt.show()
                    plt.close(fig)

        rmss = np.r_[rmss]

    if standardize in (None, "none"):
        return rmss

    rmss = np.median(rmss, axis=0)
    if standardize == "global":
        rmss[:] = np.median(rmss)

    Path(out_bin).rename(out_bin + ".tmp")
    with open(out_bin, "wb") as out:
        for s in trange(0, T_samples, fs * chunk_seconds, desc="stdize"):
            chunk = read_data(
                out_bin + ".tmp",
                np.float32,
                s,
                min(T_samples, s + fs * chunk_seconds),
                n_channels,
            )
            chunk = chunk / rmss
            chunk.tofile(out)


def run_standardize(
    in_bin,
    out_bin,
    rmss,
    fs=30_000,
    n_channels=384,
    chunk_seconds=5,
    standardize="perchan",
):
    """Just preprocessing. For spike detection, use method below.

    The pipeline is: bandpass, alternating zscore decorr, peak
    to peak.

    Parameters
    ----------
    in_key, out_key : str
        Dataset to load, dataset to write.
    ptp_len : int
        Length of peak to peak filter
    bp : pair of int
        Bandpass frequencies
    order : int
        Butterworth filter order (effectively doubled by filtfilt)
    decorr_iter : int
        Number of alternating zscore iterations
    pad_for_filter : int
        We'll draw extra big chunks to avoid boundary effects from
        the bandpass filter
    """
    assert standardize in (None, "none", "perchan", "global")
    T_samples, T_seconds = get_binary_length(in_bin, n_channels, fs)
    print("T_samples", T_samples, "T_seconds", T_seconds)

    rmss = np.median(rmss, axis=0)
    if standardize == "global":
        rmss[:] = np.median(rmss)

    with open(out_bin, "wb") as out:
        for s in trange(0, T_samples, fs * chunk_seconds, desc="stdize"):
            with noint:
                chunk = read_data(
                    in_bin,
                    np.float32,
                    s,
                    min(T_samples, s + fs * chunk_seconds),
                    n_channels,
                )
                chunk = chunk / rmss
                chunk.tofile(out)


# -- library


def design_butter_bandpass(order, freqs, fs):
    """Bandpas filter factory. Operates on the T axis of a TxD array."""
    sos = signal.butter(order, freqs, btype="bandpass", fs=fs, output="sos")

    def bandpass(x):
        return signal.sosfiltfilt(sos, x, axis=0)

    return bandpass


def ptp_filter(x, axis=0, length=25, mode="constant"):
    out = ndimage.maximum_filter1d(x, length, axis=axis, mode=mode)
    out -= ndimage.minimum_filter1d(x, length, axis=axis, mode=mode)
    return out


def make_chunk_preprocessor(
    raw_bin,
    fs,
    n_channels,
    extra_channels,
    n_samples,
    geom,
    ptp_len=0,
    bp=(300, 2000),
    order=3,
    decorr_iter=0,
    pad_for_filter=None,
    resample_to=None,
    lfp_destripe=False,
    avg_depth=False,
    csd=False,
    dtype=np.float32,
    out_dtype=np.float32,
):
    # build bandpass filter
    if bp is not None:
        bandpass = design_butter_bandpass(order, bp, fs)

    if pad_for_filter is None:
        pad_for_filter = fs

    def get_chunk(s_start, s_end):
        # load from h5
        chunk_start = s_start - pad_for_filter
        chunk_end = s_end + pad_for_filter
        load_start = max(0, chunk_start)
        load_end = min(n_samples, chunk_end)

        chunk = read_data(raw_bin, dtype, load_start, load_end, n_channels + extra_channels)
        if extra_channels is not None and extra_channels > 0:
            chunk = chunk[:, :-extra_channels]
        chunk = chunk.astype(out_dtype)
        pad_left = load_start - chunk_start
        pad_right = chunk_end - load_end
        if pad_left + pad_right > 0:
            # reflect padding is nice for filters
            chunk = np.pad(
                chunk, [(pad_left, pad_right), (0, 0)], mode="reflect"
            )

        # bandpass filter
        if bp is not None:
            chunk = bandpass(chunk)

        # we padded chunks to avoid edge effects in bandpass
        if pad_for_filter > 0:
            chunk = chunk[pad_for_filter:-pad_for_filter]

        if lfp_destripe:
            chunk_shape = chunk.shape
            chunk = voltage.destripe_lfp(chunk.T, fs).T
            assert chunk.shape == (chunk_shape[0], n_channels)

        if resample_to is not None:
            assert not fs % resample_to
            n_samples_out = chunk.shape[0] // (fs // resample_to)
            chunk = signal.resample(chunk, n_samples_out)

        if avg_depth:
            # average same depth channels
            unique_depths, same_depth_chans = np.unique(geom[:, 1], return_inverse=True)
            chunk_ = np.zeros((chunk.shape[0], unique_depths.size), dtype=chunk.dtype)
            np.add.at(chunk_, (slice(None), same_depth_chans), chunk)
            chunk = chunk_ * (unique_depths.size / same_depth_chans.size)

        if csd:
            chunk = np.pad(chunk, [(0, 0), (1, 1)], mode="edge")
            chunk = 2 * chunk[:, 1:-1] - chunk[:, 2:] - chunk[:, :-2]

        # z score iters for decorrelation
        for _ in range(decorr_iter):
            chunk = zscore(chunk, axis=1)
            chunk = zscore(chunk, axis=0)

        # peak to peak
        if ptp_len > 0:
            chunk = ptp_filter(chunk, length=ptp_len)

        return chunk.astype(out_dtype)

    return get_chunk


class _noint:
    def handler(self, *sig):
        if self.sig:
            signal_.signal(signal_.SIGINT, self.old_handler)
            sig, self.sig = self.sig, None
            self.old_handler(*sig)
        self.sig = sig

    def __enter__(self):
        self.old_handler = signal_.signal(signal_.SIGINT, self.handler)
        self.sig = None

    def __exit__(self, type, value, traceback):
        signal_.signal(signal_.SIGINT, self.old_handler)
        if self.sig:
            self.old_handler(*self.sig)


noint = _noint()
