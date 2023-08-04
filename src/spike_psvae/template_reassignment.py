import h5py
import multiprocessing
import numpy as np

from collections import namedtuple
from multiprocessing.pool import Pool
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm


def template_reassignment(
    subtraction_h5,
    residual_binary_file,
    templates,
    metric="cosine",
    batch_size=512,
    n_jobs=20,
):
    """Compute collision-cleaned waveforms and reassign them to templates

    This is a sketch of a naive template matching scheme. Could be
    extended in a lot of directions! Maybe the general architecture
    of this function will be useful, also.

    Arguments
    ---------
    subtraction_h5 : string or path
        Path to HDF5 with subtraction result
    residual_binary_file : string or path
        Path to float32 residual binary
    templates : np.array
        NTC array of templates
    metric : string
        I'm using scipy.spatial.cdist, which supports lots of distances,
        but we can switch to something else if we want something else.
    batch_size, n_jobs : int
        Parallelism controls.

    Returns
    -------
    reassignments : array of int
        Which template was each wf assigned to?
    """
    # load up metadata for building jobs
    with h5py.File(subtraction_h5) as h5:
        geom = h5["geom"][:]
        channel_index = h5["channel_index"][:]
        Nwf = h5["spike_index"].shape[0]

    # templates on same # of chans as wfs
    Nt, T, n_channels = templates.shape
    assert n_channels == len(geom)
    template_maxchans = templates.ptp(1).argmax(1)

    # make a structure for quickly checking whether a waveform
    # should be compared to each template, so we don't do all
    # of the possible comparisons, but just the relevant subset
    n_channels = len(channel_index)
    neighbors = np.zeros((n_channels, n_channels), dtype=bool)
    neighbors[np.arange(n_channels)[:, None], channel_index] = 1
    mc_to_templates = neighbors[
        np.arange(n_channels)[:, None], template_maxchans[None, :]
    ]
    mc_to_templates = [np.flatnonzero(row) for row in mc_to_templates]

    # we will be using channel index to index into templates
    padded_templates = np.pad(
        templates, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
    )

    # construct jobs and allocate loop output
    jobs = range(0, Nwf, batch_size)
    reassignments = np.empty(Nwf, dtype=int)

    # run jobs in parallel
    ctx = multiprocessing.get_context("spawn")
    with Pool(
        n_jobs,
        initializer=_job_init,
        initargs=(
            subtraction_h5,
            residual_binary_file,
            padded_templates,
            template_maxchans,
            metric,
            batch_size,
            mc_to_templates,
        ),
        context=ctx,
    ) as pool:
        for batch_idx, batch_assignments in tqdm(
            pool.imap(_job, jobs),
            desc="grab and localize",
            smoothing=0,
            total=len(jobs),
        ):
            reassignments[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ] = batch_assignments

    return reassignments


def _job(batch_idx):
    p = _job.data
    T = p.padded_templates.shape[1]
    n_channels = len(p.channel_index)

    # which do we load?
    which = range(
        batch_idx * p.batch_size,
        min(len(p.spike_index), (batch_idx + 1) * p.batch_size),
    )

    # load waveforms
    waveforms = get_raw_local_waveforms(
        p.spike_index[which],
        p.residual_binary_file,
        p.channel_index,
        subtracted_waveforms=None,
    )
    waveforms += p.subtracted_wfs[which]

    # argmin assignment loop
    reassignments = np.empty(len(which), dtype=int)
    for ix, (trough_loc, maxchan) in enumerate(p.spike_index[which]):
        # which channels will we work on for this waveform
        channels = p.channel_index[maxchan]
        good_channels = channels < n_channels
        channels = channels[good_channels]

        # grab flattened waveform, add a dimension for `cdist`
        waveform = waveforms[ix, :, good_channels].ravel()[None]

        # grab templates on these channels and flatten
        which_templates = p.mc_to_templates[maxchan]
        local_templates = p.padded_templates[
            which_templates[:, None, None],
            np.arange(T)[None, :, None],
            channels[None, None, :],
        ].reshape(len(which_templates), -1)

        # get distances and assign
        distances = cdist(waveform, local_templates, metric=p.metric)
        reassignments.append(which_templates[distances.squeeze().argmin()])

    return batch_idx, reassignments


JobData = namedtuple(
    "JobData",
    [
        "channel_index",
        "spike_index",
        "template_maxchans",
        "padded_templates",
        "mc_to_templates",
        "metric",
        "subtracted_wfs",
        "residual_binary_file",
        "batch_size",
    ],
)


def _job_init(
    subtraction_h5,
    residual_binary_file,
    padded_templates,
    template_maxchans,
    metric,
    batch_size,
    mc_to_templates,
):
    h5 = h5py.File(subtraction_h5)
    channel_index = h5["channel_index"][:]
    spike_index = h5["spike_index"][:]

    _job.p = JobData(
        channel_index,
        spike_index,
        template_maxchans,
        padded_templates,
        mc_to_templates,
        metric,
        h5["subtracted_waveforms"],
        residual_binary_file,
        batch_size,
    )


def get_raw_local_waveforms(
    spike_index,
    binary_file,
    channel_index,
    subtracted_waveforms=None,
    trough_offset=42,
    spike_length_samples=121,
):
    n_channels = channel_index.shape[0]
    n_channels_out = channel_index.shape[1]

    # load from binary
    waveforms = np.empty(
        (len(spike_index), spike_length_samples, n_channels_out),
        dtype=np.float32,
    )
    for ix, (trough_loc, maxchan) in enumerate(spike_index):
        start = trough_loc - trough_offset
        wf = np.fromfile(
            binary_file,
            np.float32,
            count=spike_length_samples * n_channels,
            offset=np.dtype(np.float32).itemsize * start * n_channels,
        ).reshape(spike_length_samples, n_channels)
        wf = np.pad(wf, [(0, 0), (0, 1)], constant_values=np.nan)
        waveforms[ix] = wf[:, channel_index[maxchan]]

    return waveforms
