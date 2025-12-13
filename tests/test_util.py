import numpy as np
import spikeinterface.core as sc
from dartsort.util.data_util import DARTsortSorting


def no_overlap_recording_sorting(
    templates, fs=30000, trough_offset_samples=42, pad=242
):
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
        times_samples=times + pad,
        channels=np.zeros(n_templates, dtype=np.int64),
        labels=np.arange(n_templates),
        ephemeral_features=dict(
            point_source_localizations=locs, times_seconds=times_seconds
        ),
    )
    return rec, sorting


def rc2xy(row, col, version=1):
    """
    converts the row/col indices to um coordinates.
    :param row: row index on the probe
    :param col: col index on the probe
    :param version: neuropixel major version 1 or 2
    :return: dictionary with keys x and y
    """
    if version == 1:
        x = col * 16 + 11
        y = (row * 20) + 20
    elif np.floor(version) == 2:
        x = col * 32
        y = row * 15
    else:
        assert False
    return {"x": x, "y": y}


def dense_layout(version=1, nshank=1, NC=384):
    """Copied from ibl-neuropixel

    Returns a dense layout indices map for neuropixel, as used at IBL
    :param version: major version number: 1 or 2 or 2.4
    :return: dictionary with keys 'ind', 'col', 'row', 'x', 'y'
    """
    ch = {
        "ind": np.arange(NC),
        "row": np.floor(np.arange(NC) / 2),
        "shank": np.zeros(NC),
    }

    if version == 1:  # version 1 has a dense layout, checkerboard pattern
        ch.update({"col": np.tile(np.array([2, 0, 3, 1]), int(NC / 4))})
    elif (
        np.floor(version) == 2 and nshank == 1
    ):  # single shank NP1 has 2 columns in a dense patter
        ch.update({"col": np.tile(np.array([0, 1]), int(NC / 2))})
    elif (
        np.floor(version) == 2 and nshank == 4
    ):  # the 4 shank version default is rather complicated
        shank_row = np.tile(np.arange(NC / 16), (2, 1)).T[:, np.newaxis].flatten()
        shank_row = np.tile(shank_row, 8)
        shank_row += (
            np.tile(
                np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, np.newaxis], (1, int(NC / 8))
            ).flatten()
            * 24
        )
        ch.update(
            {
                "col": np.tile(np.array([0, 1]), int(NC / 2)),
                "shank": np.tile(
                    np.array([0, 1, 0, 1, 2, 3, 2, 3])[:, np.newaxis], (1, int(NC / 8))
                ).flatten(),
                "row": shank_row,
            }
        )
    # for all, get coordinates
    ch.update(rc2xy(ch["row"], ch["col"], version=version))
    return ch
