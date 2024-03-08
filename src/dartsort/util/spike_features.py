import numpy as np


def amplitude_feats(waveforms, log=True):
    peak = np.nanmax(waveforms, axis=(1, 2))
    trough = np.nanmin(waveforms, axis=(1, 2))
    if log:
        return np.log(peak) - np.log(trough)
    return peak / trough


def ptp(waveforms):
    return waveforms.ptp(dim=(1, 2))
