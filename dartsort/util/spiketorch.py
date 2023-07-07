def ptp(waveforms, dim=1):
    return waveforms.max(dim=dim).values - waveforms.min(dim=dim).values
