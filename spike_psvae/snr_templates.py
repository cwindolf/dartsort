import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tqdm.auto import tqdm

from . import denoise, spikeio, waveform_utils
from .multiprocessing_utils import MockPoolExecutor, MockQueue


def get_templates(
    spike_train,
    geom,
    raw_binary_file,
    unit_max_channels=None,
    max_spikes_per_unit=500,
    do_temporal_decrease=True,
    zero_radius_um=200,
    reducer=np.median,
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    do_tpca=True,
    tpca=None,
    tpca_centered=True,
    tpca_rank=5,
    tpca_radius=75,
    tpca_n_wfs=50_000,
    use_previous_max_channels=False,
    do_nn_denoise=False,
    denoiser_init_kwargs={},
    denoiser_weights_path=None,
    device=None,
    batch_size=1024,
    pbar=True,
    seed=0,
    n_jobs=-1,
    raw_only=False,
    dtype=np.float32,
    edge_behavior="saturate",
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
    extra : dict with other info
    """
    if do_tpca and unit_max_channels is None:
        # estimate max channels for each unit by computing raw templates
        unit_max_channels = (
            get_raw_templates(
                spike_train,
                geom,
                raw_binary_file,
                reducer=reducer,
                spike_length_samples=spike_length_samples,
                trough_offset=trough_offset,
                pbar=pbar,
                seed=seed,
                n_jobs=n_jobs,
                max_spikes_per_unit=max_spikes_per_unit,
            )
            .ptp(1)
            .argmax(1)
        )

    # -- initialize output
    n_templates = spike_train[:, 1].max() + 1
    templates = np.zeros((n_templates, spike_length_samples, len(geom)))

    raw_templates = np.zeros_like(templates)
    if raw_only:
        snr_by_channel = np.zeros((n_templates, len(geom)))
        extra = dict(snr_by_channel=snr_by_channel)
    else:
        snr_by_channel = np.zeros((n_templates, len(geom)))
        denoised_templates = np.zeros_like(templates)
        extra = dict(
            raw_templates=raw_templates,
            denoised_templates=denoised_templates,
            snr_by_channel=snr_by_channel,
        )

    # -- fit TPCA to randomly sampled waveforms
    if not raw_only and do_tpca and tpca is None:
        max_channels = unit_max_channels[spike_train[:, 1]]
        tpca = waveform_utils.fit_tpca_bin(
            np.c_[spike_train[:, 0], max_channels],
            geom,
            raw_binary_file,
            centered=tpca_centered,
            tpca_rank=tpca_rank,
            tpca_n_wfs=tpca_n_wfs,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            spatial_radius=tpca_radius,
            do_nn_denoise=do_nn_denoise,
            denoiser_init_kwargs=denoiser_init_kwargs,
            denoiser_weights_path=denoiser_weights_path,
            device=device,
            batch_size=batch_size,
            seed=seed,
            dtype=dtype,
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
            raw_only,
            do_nn_denoise,
            denoiser_init_kwargs,
            denoiser_weights_path,
            device,
            batch_size,
            dtype,
        ),
    ) as pool:
        for unit, raw_template, denoised_template, snr_by_chan in xqdm(
            pool.map(template_worker, units),
            total=len(units),
            desc="Raw templates" if raw_only else "Cleaned templates",
            smoothing=0,
            pbar=pbar,
        ):
            raw_templates[unit] = raw_template
            snr_by_channel[unit] = snr_by_chan
            if not raw_only:
                denoised_templates[unit] = denoised_template

    if raw_only:
        return raw_templates, extra

    # SNR-weighted combination to create the template
    weights = denoised_weights(
        snr_by_channel,
        spike_length_samples,
        trough_offset,
        snr_threshold,
        edge_behavior=edge_behavior,
    )
    templates = weights * raw_templates + (1 - weights) * denoised_templates
    extra["weights"] = weights

    # zero out far away channels
    if zero_radius_um is not None:
        zero_ci = waveform_utils.make_channel_index(
            geom, zero_radius_um, steps=1, distance_order=False, p=2
        )
        for i in range(len(templates)):
            if use_previous_max_channels:
                mc = int(unit_max_channels[i])
            else:
                mc = templates[i].ptp(0).argmax()
            far = ~np.isin(np.arange(len(geom)), zero_ci[mc])
            templates[i, :, far] = 0

    return templates, extra


def get_raw_templates(
    spike_train,
    geom,
    raw_binary_file,
    max_spikes_per_unit=250,
    reducer=np.median,
    spike_length_samples=121,
    trough_offset=42,
    pbar=True,
    seed=0,
    n_jobs=-1,
    dtype=np.float32,
):
    raw_templates, _ = get_templates(
        spike_train,
        geom,
        raw_binary_file,
        max_spikes_per_unit=max_spikes_per_unit,
        reducer=reducer,
        spike_length_samples=spike_length_samples,
        do_temporal_decrease=False,
        raw_only=True,
        trough_offset=trough_offset,
        do_tpca=False,
        pbar=True,
        seed=seed,
        n_jobs=n_jobs,
        dtype=dtype,
    )
    return raw_templates


def get_denoised_template_single(
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
    tpca_rank=5,
    tpca_radius=75,
    pbar=True,
    tpca_n_wfs=50_000,
    do_nn_denoise=False,
    denoiser_init_kwargs={},
    denoiser_weights_path=None,
    device=None,
    batch_size=1024,
    seed=0,
    edge_behavior="saturate",
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
        do_nn_denoise=do_nn_denoise,
        denoiser_init_kwargs=denoiser_init_kwargs,
        denoiser_weights_path=denoiser_weights_path,
        device=device,
        batch_size=batch_size,
    )

    # SNR-weighted combination to create the template
    weights = denoised_weights_single(
        snr_by_channel,
        spike_length_samples,
        trough_offset,
        snr_threshold,
        edge_behavior=edge_behavior,
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


get_single_templates = get_denoised_template_single


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
    do_nn_denoise=False,
    denoiser_init_kwargs={},
    denoiser_weights_path=None,
    device=None,
    batch_size=1024,
    dtype=np.float32,
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
        dtype=dtype,
    )

    if do_nn_denoise:
        N, T, C = waveforms.shape
        waveforms = waveforms.transpose(0, 2, 1).reshape(
            -1, spike_length_samples
        )
        # pick torch device if it's not supplied
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            if device.type == "cuda":
                torch.cuda._lazy_init()
        else:
            device = torch.device(device)
        torch.set_grad_enabled(False)

        denoiser = denoise.SingleChanDenoiser(**denoiser_init_kwargs)
        if denoiser_weights_path is not None:
            denoiser.load(fname_model=denoiser_weights_path)
        else:
            denoiser.load()
        denoiser.to(device)

        results = []
        for bs in range(0, waveforms.shape[0], batch_size):
            be = min(bs + batch_size, N * C)
            results.append(
                denoiser(
                    torch.as_tensor(
                        waveforms[bs:be], device=device, dtype=torch.float
                    )
                )
                .cpu()
                .numpy()
            )
        waveforms = np.concatenate(results, axis=0)
        del results
        waveforms = waveforms.reshape(N, C, T).transpose(0, 2, 1)

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


def get_raw_template_single(
    spike_times,
    raw_binary_file,
    n_channels,
    max_spikes_per_unit=250,
    reducer=np.median,
    trough_offset=42,
    spike_length_samples=121,
    seed=0,
    dtype=np.float32,
):
    choices = slice(None)
    if spike_times.shape[0] > max_spikes_per_unit:
        choices = np.random.default_rng(seed).choice(
            spike_times.shape[0], max_spikes_per_unit, replace=False
        )
        choices.sort()

    waveforms, skipped_idx = spikeio.read_waveforms(
        spike_times[choices],
        raw_binary_file,
        n_channels,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        dtype=dtype,
    )

    return reducer(waveforms, axis=0)


def denoised_weights(
    snrs,
    spike_length_samples,
    trough_offset,
    snr_threshold,
    a=12.0,
    b=12.0,
    d=6.0,
    edge_behavior="saturate",
):
    # v shaped function for time weighting
    vt = np.abs(np.arange(spike_length_samples) - trough_offset, dtype=float)
    if trough_offset < spike_length_samples:
        vt[trough_offset:] = vt[trough_offset:] / vt[trough_offset:].max()
    if trough_offset > 0:
        vt[:trough_offset] = vt[:trough_offset] / vt[:trough_offset].max()

    # snr weighting per channel
    if edge_behavior == "saturate":
        sc = np.minimum(snrs, snr_threshold) / snr_threshold
    if edge_behavior == "inf":
        sc = np.minimum(snrs, snr_threshold) / snr_threshold
        sc[sc >= 1.0] = np.inf
    elif edge_behavior == "raw":
        sc = snrs
    # should this be inf when > snr threshold? (note it gets -)

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
    edge_behavior="saturate",
):
    # v shaped function for time weighting
    vt = np.abs(np.arange(spike_length_samples) - trough_offset, dtype=float)
    if trough_offset < spike_length_samples:
        vt[trough_offset:] = vt[trough_offset:] / vt[trough_offset:].max()
    if trough_offset > 0:
        vt[:trough_offset] = vt[:trough_offset] / vt[:trough_offset].max()

    # snr weighting per channel
    if edge_behavior == "saturate":
        sc = np.minimum(snrs, snr_threshold) / snr_threshold
    if edge_behavior == "inf":
        sc = np.minimum(snrs, snr_threshold) / snr_threshold
        sc[sc >= 1.0] = np.inf
    elif edge_behavior == "raw":
        sc = snrs
    # pass it through a hand picked squashing function
    wtc = 1.0 / (1.0 + np.exp(d + a * vt[:, None] - b * sc[None, :]))

    return wtc


# -- parallelism helpers


def template_worker(unit):
    # parameters set by init below
    p = template_worker

    in_unit = np.flatnonzero(p.spike_train[:, 1] == unit)
    if p.raw_only:
        raw_template = get_raw_template_single(
            p.spike_train[in_unit, 0],
            p.raw_binary_file,
            p.geom.shape[0],
            max_spikes_per_unit=p.max_spikes_per_unit,
            reducer=p.reducer,
            trough_offset=p.trough_offset,
            spike_length_samples=p.spike_length_samples,
            seed=p.rg.integers(np.iinfo(np.int64).max),
            dtype=p.dtype,
        )
        denoised_template = snr_by_channel = None
    else:
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
            do_nn_denoise=p.do_nn_denoise,
            denoiser_init_kwargs=p.denoiser_init_kwargs,
            denoiser_weights_path=p.denoiser_weights_path,
            device=p.device,
            batch_size=p.batch_size,
            dtype=p.dtype,
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
    raw_only,
    do_nn_denoise,
    denoiser_init_kwargs,
    denoiser_weights_path,
    device,
    batch_size,
    dtype,
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
    p.raw_only = raw_only
    p.do_nn_denoise = do_nn_denoise
    p.denoiser_init_kwargs = denoiser_init_kwargs
    p.denoiser_weights_path = denoiser_weights_path
    p.device = device
    p.batch_size = batch_size
    p.dtype = dtype


def xqdm(iterator, pbar=True, **kwargs):
    if pbar:
        return tqdm(iterator, **kwargs)
    else:
        return iterator
