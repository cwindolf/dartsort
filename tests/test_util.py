import numpy as np
import spikeinterface.core as sc
from dartsort.util.data_util import DARTsortSorting


def no_overlap_recording_sorting(templates, fs=30000, trough_offset_samples=42, pad=0):
    n_templates, spike_length_samples, n_channels = templates.shape
    rec = templates.reshape(n_templates * spike_length_samples, n_channels)
    if pad > 0:
        rec = np.pad(rec, [(pad, pad), (0, 0)])
    geom = np.c_[np.zeros(n_channels), np.arange(n_channels)]
    rec = sc.NumpyRecording(rec, fs)
    rec.set_dummy_probe_from_locations(geom)
    depths = np.zeros(n_templates)
    locs = np.c_[np.zeros_like(depths), np.zeros_like(depths), depths]
    times = np.arange(n_templates) * spike_length_samples + trough_offset_samples
    times_seconds = times / fs
    sorting = DARTsortSorting(
        times + pad,
        np.zeros(n_templates),
        np.arange(n_templates),
        extra_features=dict(
            point_source_localizations=locs, times_seconds=times_seconds
        ),
    )
    return rec, sorting
