import numpy as np


def peak_to_trough(waveforms, log=True):
    peak_vecs = waveforms.max(dim=1)
    trough_vecs = waveforms.min(dim=1)
    max_channels = np.nanargmax(peak_vecs - trough_vecs, axis=1, keepdims=True)
    peak = np.take_along_axis(peak_vecs, max_channels, axis=1)[:, 0]
    trough = np.take_along_axis(trough_vecs, max_channels, axis=1)[:, 0]
    if log:
        return np.log(peak) - np.log(trough)
    return peak / trough
