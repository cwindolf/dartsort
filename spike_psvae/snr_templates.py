import numpy as np
from tqdm.auto import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from . import denoise, spikeio, waveform_utils


def get_templates(
    spike_train,
    geom,
    raw_binary_file,
    unit_max_channels,
    max_spikes_per_unit=500,
    do_temporal_decrease=True,
    zero_radius_um=200,
    reducer=np.median,
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    sampling_frequency=30_000,
    return_raw_cleaned=False,
    do_tpca=True,
    tpca=None,
    tpca_rank=5,
    tpca_radius=75,
    tpca_n_wfs=50_000,
    pbar=True,
    seed=0,
    n_jobs=-1,
):
    """Get denoised templates

    This computes a weighted average of the raw template
    and the TPCA'd collision-cleaned template, based on
    the SNR = maxptp * sqrt(n). If snr > snr_threshold,
    the raw template is used. Otherwise, a convex combination
    with weight snr / snr_threshold is used.

    Enforce decrease is applied to waveforms before reducing.

    Arguments
    ---------
    spike_train : np.array, (n_spikes, 2)
        First column is spike trough time (samples), second column
        is cluster ID.
    geom : np.array (n_channels, 2)
    raw_binary_file, residual_binary_file : string or Path
    subtracted_waveforms : array, memmap, or h5 dataset
    n_templates : None or int
        If None, it will be set to max unit id + 1
    max_spikes_per_unit : int
        If a unit spikes more than this, this many will be sampled
        uniformly (separately for raw + cleaned wfs)
    reducer : e.g. np.mean or np.median
    snr_threshold : float
        Below this number, a weighted combo of raw and cleaned
        template will be computed (weight based on template snr).

    Returns
    -------
    templates : np.array (n_templates, spike_length_samples, geom.shape[0])
    snrs : np.array (n_templates,)
        The snrs of the original raw templates.
    if return_raw_cleaned, also returns raw_templates and denoised_templates,
    both arrays like templates.
    """
    # -- initialize output
    n_templates = spike_train[:, 1].max() + 1
    templates = np.zeros((n_templates, spike_length_samples, len(geom)))

    snr_by_channel = np.zeros((n_templates, len(geom)))
    rg = np.random.default_rng(seed)

    raw_templates = np.zeros_like(templates)
    denoised_templates = np.zeros_like(templates)
    extra = dict(
        raw_templates=raw_templates,
        denoised_templates=denoised_templates,
        snr_by_channel=snr_by_channel,
    )

    # -- fit TPCA to randomly sampled waveforms
    if do_tpca and tpca is None:
        max_channels = unit_max_channels[spike_train[:, 1]]
        tpca = waveform_utils.fit_tpca_bin(
            np.c_[spike_train[:, 0], max_channels],
            geom,
            raw_binary_file,
            tpca_rank=tpca_rank,
            tpca_n_wfs=tpca_n_wfs,
            spike_length_samples=spike_length_samples,
            spatial_radius=tpca_radius,
            seed=seed,
        )
        extra["tpca"] = tpca

    # -- main loop to make templates
    units = np.unique(spike_train[:, 1])
    units = units[units >= 0]

    # no-threading/multiprocessing execution for debugging if n_jobs == 0
    Executor = ProcessPoolExecutor if n_jobs else MockPoolExecutor
    context = multiprocessing.get_context("spawn")
    manager = context.Manager() if n_jobs else None
    id_queue = manager.Queue() if n_jobs else MockQueue()

    n_jobs = n_jobs or 1
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() - 1

    for id in range(n_jobs):
        id_queue.put(id)

    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=template_worker_init,
        initargs=(
            id_queue,
            seed,
            spike_train,
            geom,
            raw_binary_file,
            do_tpca,
            tpca,
            do_temporal_decrease,
            max_spikes_per_unit,
            reducer,
            trough_offset,
            spike_length_samples,
        ),
    ) as pool:
        for unit, raw_template, denoised_template, snr_by_chan in xqdm(
            pool.map(template_worker, units),
            total=len(units),
            desc="Cleaned templates",
            smoothing=0,
            pbar=pbar,
        ):
            raw_templates[unit] = raw_template
            denoised_templates[unit] = denoised_template
            snr_by_channel[unit] = snr_by_chan

    # SNR-weighted combination to create the template
    weights = denoised_weights(
        snr_by_channel, spike_length_samples, trough_offset, snr_threshold
    )
    templates = weights * raw_templates + (1 - weights) * denoised_templates
    extra["weights"] = weights

    # zero out far away channels
    if zero_radius_um is not None:
        zero_ci = waveform_utils.make_channel_index(
            geom, zero_radius_um, steps=1, distance_order=False, p=2
        )
        for i in range(len(templates)):
            mc = templates[i].ptp(0).argmax()
            far = ~np.isin(np.arange(len(geom)), zero_ci[mc])
            templates[i, :, far] = 0

    return templates, extra


def get_single_templates(
    spike_times,
    geom,
    raw_binary_file,
    tpca=None,
    max_spikes_per_unit=500,
    do_tpca=True,
    do_temporal_decrease=True,
    zero_radius_um=200,
    reducer=np.median,
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    sampling_frequency=30_000,
    return_raw_cleaned=False,
    tpca_rank=5,
    tpca_radius=75,
    pbar=True,
    tpca_n_wfs=50_000,
    seed=0,
):
    (
        raw_template,
        denoised_template,
        snr_by_channel,
    ) = get_raw_denoised_template_single(
        spike_times,
        geom,
        raw_binary_file,
        do_tpca=do_tpca,
        tpca=tpca,
        do_temporal_decrease=do_temporal_decrease,
        max_spikes_per_unit=max_spikes_per_unit,
        seed=seed,
        reducer=reducer,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

    # SNR-weighted combination to create the template
    weights = denoised_weights_single(
        snr_by_channel, spike_length_samples, trough_offset, snr_threshold
    )
    template = weights * raw_template + (1 - weights) * denoised_template

    # zero out far away channels
    if zero_radius_um is not None:
        zero_ci = waveform_utils.make_channel_index(
            geom, zero_radius_um, steps=1, distance_order=False, p=2
        )
        mc = template.ptp(0).argmax()
        far = ~np.isin(np.arange(len(geom)), zero_ci[mc])
        template[:, far] = 0

    return template


def get_raw_denoised_template_single(
    spike_times,
    geom,
    raw_binary_file,
    do_tpca=True,
    tpca=None,
    do_temporal_decrease=True,
    max_spikes_per_unit=500,
    seed=0,
    reducer=np.median,
    trough_offset=42,
    spike_length_samples=121,
):
    rg = np.random.default_rng(seed)
    choices = slice(None)
    if spike_times.shape[0] > max_spikes_per_unit:
        choices = rg.choice(
            spike_times.shape[0], max_spikes_per_unit, replace=False
        )
        choices.sort()
    waveforms, skipped_idx = spikeio.read_waveforms(
        spike_times[choices],
        raw_binary_file,
        len(geom),
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

    if do_temporal_decrease:
        denoise.enforce_temporal_decrease(waveforms, in_place=True)

    raw_template = reducer(waveforms, axis=0)
    raw_ptp = raw_template.ptp(0)
    snr_by_channel = raw_ptp * np.sqrt(len(waveforms))

    # denoise the waveforms
    if do_tpca and tpca is not None:
        nn, tt, cc = waveforms.shape
        waveforms = waveforms.transpose(0, 2, 1).reshape(nn * cc, tt)
        waveforms = tpca.inverse_transform(tpca.transform(waveforms))
        waveforms = waveforms.reshape(nn, cc, tt).transpose(0, 2, 1)

    # enforce decrease for both, using raw maxchan
    if do_temporal_decrease:
        denoise.enforce_temporal_decrease(waveforms, in_place=True)
    denoised_template = reducer(waveforms, axis=0)

    return raw_template, denoised_template, snr_by_channel


def denoised_weights(
    snrs,
    spike_length_samples,
    trough_offset,
    snr_threshold,
    a=12.0,
    b=12.0,
    d=6.0,
):
    # v shaped function for time weighting
    vt = np.abs(np.arange(spike_length_samples) - trough_offset, dtype=float)
    vt[trough_offset:] = vt[trough_offset:] / vt[trough_offset:].max()
    vt[:trough_offset] = vt[:trough_offset] / vt[:trough_offset].max()

    # snr weighting per channel
    sc = np.minimum(snrs, snr_threshold) / snr_threshold

    # pass it through a hand picked squashing function
    wtc = 1.0 / (1.0 + np.exp(d + a * vt[None, :, None] - b * sc[:, None, :]))

    return wtc


def denoised_weights_single(
    snrs,
    spike_length_samples,
    trough_offset,
    snr_threshold,
    a=12.0,
    b=12.0,
    d=6.0,
):
    # v shaped function for time weighting
    vt = np.abs(np.arange(spike_length_samples) - trough_offset, dtype=float)
    vt[trough_offset:] = vt[trough_offset:] / vt[trough_offset:].max()
    vt[:trough_offset] = vt[:trough_offset] / vt[:trough_offset].max()

    # snr weighting per channel
    sc = np.minimum(snrs, snr_threshold) / snr_threshold
    # pass it through a hand picked squashing function
    wtc = 1.0 / (1.0 + np.exp(d + a * vt[:, None] - b * sc[None, :]))

    return wtc


# -- parallelism helpers


def template_worker(unit):
    # parameters set by init below
    p = template_worker

    in_unit = np.flatnonzero(p.spike_train[:, 1] == unit)

    (
        raw_template,
        denoised_template,
        snr_by_channel,
    ) = get_raw_denoised_template_single(
        p.spike_train[in_unit, 0],
        p.geom,
        p.raw_binary_file,
        do_tpca=p.do_tpca,
        tpca=p.tpca,
        do_temporal_decrease=p.do_temporal_decrease,
        max_spikes_per_unit=p.max_spikes_per_unit,
        seed=p.rg.integers(np.iinfo(np.int64).max),
        reducer=p.reducer,
        trough_offset=p.trough_offset,
        spike_length_samples=p.spike_length_samples,
    )

    return unit, raw_template, denoised_template, snr_by_channel


def template_worker_init(
    id_queue,
    seed,
    spike_train,
    geom,
    raw_binary_file,
    do_tpca,
    tpca,
    do_temporal_decrease,
    max_spikes_per_unit,
    reducer,
    trough_offset,
    spike_length_samples,
):
    rank = id_queue.get()
    p = template_worker
    p.rg = np.random.default_rng(seed + rank)
    p.spike_train = spike_train
    p.geom = geom
    p.raw_binary_file = raw_binary_file
    p.do_tpca = do_tpca
    p.tpca = tpca
    p.do_temporal_decrease = do_temporal_decrease
    p.max_spikes_per_unit = max_spikes_per_unit
    p.reducer = reducer
    p.trough_offset = trough_offset
    p.spike_length_samples = spike_length_samples


class MockPoolExecutor:
    """A helper class for turning off concurrency when debugging."""

    def __init__(
        self,
        max_workers=None,
        mp_context=None,
        initializer=None,
        initargs=None,
    ):
        initializer(*initargs)
        self.map = map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class MockQueue:
    """Another helper class for turning off concurrency when debugging."""

    def __init__(self):
        self.q = []
        self.put = self.q.append
        self.get = lambda: self.q.pop(0)


def xqdm(iterator, pbar=True, **kwargs):
    if pbar:
        return tqdm(iterator, **kwargs)
    else:
        return iterator
