import h5py
import numpy as np
import torch
from collections import namedtuple
from multiprocessing.pool import Pool
from .multiprocessing_utils import get_pool
from pathlib import Path
import shutil
from tqdm.auto import tqdm
from spike_psvae import (
    denoise,
    subtract,
    spikeio,
    waveform_utils,
    chunk_features,
    reassignment,
)


def extract_deconv(
    templates_up,
    spike_train_up,
    output_directory,
    standardized_bin,
    scalings=None,
    geom=None,
    channel_index=None,
    n_channels_extract=20,
    extract_radius_um=100,
    subtraction_h5=None,
    tpca_rank=8,
    save_residual=False,
    save_cleaned_waveforms=False,
    save_cleaned_tpca_projs=True,
    save_denoised_waveforms=False,
    do_denoised_tpca=True,
    save_denoised_tpca_projs=False,
    save_outlier_scores=True,
    do_reassignment=True,
    reassignment_proposed_pairs_up=None,
    reassignment_tpca_rank=5,
    reassignment_tpca_spatial_radius=75,
    reassignment_tpca_n_wfs=50000,
    localize=True,
    loc_radius=100,
    n_sec_chunk=1,
    n_jobs=1,
    sampling_rate=30_000,
    n_sec_train_feats=40,
    device=None,
    trough_offset=42,
    overwrite=True,
    pbar=True,
    nn_denoise=True,
    seed=0,
):
    standardized_bin = Path(standardized_bin)
    output_directory = Path(output_directory)

    if isinstance(templates_up, (str, Path)):
        templates_up = np.load(templates_up)
    if isinstance(spike_train_up, (str, Path)):
        spike_train_up = np.load(spike_train_up)

    n_templates, spike_length_samples, n_chans = templates_up.shape
    n_spikes = len(spike_train_up)

    if scalings is not None:
        if isinstance(scalings, (str, Path)):
            scalings = np.load(scalings)
        assert scalings.shape == (n_spikes,)
    else:
        scalings = np.ones(n_spikes, dtype=templates_up.dtype)

    std_size = standardized_bin.stat().st_size
    assert not std_size % np.dtype(np.float32).itemsize
    std_size = std_size // np.dtype(np.float32).itemsize
    assert not std_size % n_chans
    T_samples = std_size // n_chans

    if geom is None and subtraction_h5 is not None:
        with h5py.File(subtraction_h5, "r") as h5:
            geom = h5["geom"][:]
    if geom is None:
        raise ValueError(
            "Please pass geom, or subtraction_h5 if that's easier."
        )

    # paths for output
    if save_residual:
        residual_path = output_directory / "residual.bin"
        if overwrite:
            if residual_path.exists():
                residual_path.unlink()
        resid = open(residual_path, "a")
    out_h5 = output_directory / "deconv_results.h5"
    if overwrite:
        if out_h5.exists():
            out_h5.unlink()
    temp_dir = output_directory / "temp_batch_results"
    temp_dir.mkdir(exist_ok=True, parents=True)

    # are we resuming?
    last_batch_end = 0
    if out_h5.exists():
        with h5py.File(out_h5, "r") as h5:
            last_batch_end = h5["last_batch_end"][()]

    # determine jobs to run
    batch_length = n_sec_chunk * sampling_rate
    start_samples = range(last_batch_end, T_samples, batch_length)
    if not len(start_samples):
        print("Extraction already done")
        return out_h5, residual_path

    # build spike index from templates and spike train
    templates_up_maxchans = templates_up.ptp(1).argmax(1)
    up_maxchans = templates_up_maxchans[spike_train_up[:, 1]]
    spike_index_up = np.c_[spike_train_up[:, 0], up_maxchans]

    # a firstchans-style channel index
    if channel_index is None and extract_radius_um is not None:
        channel_index = waveform_utils.make_channel_index(
            geom, extract_radius_um
        )
    elif channel_index is None and n_channels_extract is not None:
        channel_index = waveform_utils.make_contiguous_channel_index(
            n_chans, n_neighbors=n_channels_extract
        )
    else:
        assert channel_index is not None
    n_channels_extract = channel_index.shape[1]

    # templates on few channels
    templates_up_loc = np.full(
        (n_templates, spike_length_samples, n_channels_extract),
        np.nan,
        dtype=templates_up.dtype,
    )
    for i in range(n_templates):
        ci_chans = channel_index[templates_up_maxchans[i]]
        templates_up_loc[i, :, :] = np.pad(
            templates_up[i], [(0, 0), (0, 1)], constant_values=np.nan
        )[:, ci_chans]

    # precompute things for reassignment
    reassignment_tpca = reassignment_temps_up_loc = None
    if do_reassignment:
        save_outlier_scores = True

        # fit a TPCA to raw waveforms
        reassignment_tpca = waveform_utils.fit_tpca_bin(
            spike_index_up,
            geom,
            standardized_bin,
            tpca_rank=reassignment_tpca_rank,
            spike_length_samples=spike_length_samples,
            spatial_radius=reassignment_tpca_rank,
            tpca_n_wfs=reassignment_tpca_n_wfs,
            seed=seed,
        )

        # get local views of templates for reassignment
        reassignment_temps_up_loc = reassignment.reassignment_templates_local(
            templates_up, reassignment_proposed_pairs_up, channel_index
        )

    # create chunk feature objects
    denoised_tpca = None
    if (
        localize
        or (do_denoised_tpca and save_denoised_waveforms)
        or save_denoised_tpca_projs
    ):
        denoised_tpca = chunk_features.TPCA(
            tpca_rank, channel_index, "denoised"
        )
    featurizers = []
    no_save_featurizers = []
    if save_cleaned_waveforms:
        featurizers.append(
            chunk_features.Waveform(
                "cleaned",
                spike_length_samples=templates_up.shape[1],
                channel_index=channel_index,
            )
        )
    if save_denoised_waveforms:
        featurizers.append(
            chunk_features.Waveform(
                "denoised",
                spike_length_samples=templates_up.shape[1],
                channel_index=channel_index,
            )
        )
    if save_cleaned_tpca_projs:
        featurizers.append(
            chunk_features.TPCA(tpca_rank, channel_index, "cleaned")
        )
    if save_denoised_tpca_projs:
        featurizers.append(denoised_tpca)
    elif denoised_tpca is not None:
        no_save_featurizers.append(denoised_tpca)
    if localize:
        featurizers.append(
            chunk_features.Localization(
                geom, channel_index, loc_radius=loc_radius
            )
        )
        featurizers.append(chunk_features.MaxPTP())

    # which waveforms will we be computing?
    do_clean = do_reassignment or save_outlier_scores or len(featurizers)
    do_denoise = any(f.which_waveforms == "denoised" for f in featurizers)

    with h5py.File(out_h5, "a") as h5:
        load_or_fit_featurizers(
            featurizers + no_save_featurizers,
            h5,
            channel_index,
            templates_up,
            spike_train_up,
            output_directory,
            standardized_bin,
            scalings,
            geom,
            n_sec_train_feats,
            n_sec_chunk=n_sec_chunk,
            n_jobs=n_jobs,
            sampling_rate=sampling_rate,
            device=device,
            trough_offset=trough_offset,
            nn_denoise=nn_denoise,
            seed=seed,
        )

        if last_batch_end > 0:
            if save_outlier_scores:
                outlier_scores = h5["outlier_scores"]
            if do_reassignment:
                reassigned_labels_up = h5["reassigned_labels_up"]
        else:
            h5.create_dataset("templates_up", data=templates_up)
            h5.create_dataset(
                "templates_up_maxchans", data=templates_up_maxchans
            )
            h5.create_dataset("channel_index", data=channel_index)
            h5.create_dataset("templates_up_loc", data=templates_up_loc)
            h5.create_dataset("spike_train_up", data=spike_train_up)
            h5.create_dataset("scalings", data=scalings)
            h5.create_dataset("spike_index", data=spike_index_up)
            h5.create_dataset(
                "first_channels",
                data=channel_index[:, 0][spike_index_up[:, 1]],
            )
            h5.create_dataset("last_batch_end", data=0)

            if save_outlier_scores:
                outlier_scores = h5.create_dataset(
                    "outlier_scores", shape=(n_spikes,), dtype=np.float64
                )
            if do_reassignment:
                reassigned_labels_up = h5.create_dataset(
                    "reassigned_labels_up", shape=(n_spikes,), dtype=int
                )

            for f in featurizers:
                h5.create_dataset(
                    f.name,
                    shape=(n_spikes, *f.out_shape),
                    dtype=f.dtype,
                )

        feature_dsets = [h5[f.name] for f in featurizers]

        if n_jobs is not None and n_jobs > 1:
            print("Initializing threads", end="")
        pool, ctx = get_pool(n_jobs, cls=Pool)
        with pool(
            n_jobs,
            initializer=_extract_deconv_init,
            initargs=(
                geom,
                device,
                channel_index,
                batch_length,
                do_clean,
                do_denoise,
                save_outlier_scores,
                save_residual,
                trough_offset,
                spike_index_up,
                spike_train_up,
                standardized_bin,
                spike_length_samples,
                templates_up,
                up_maxchans,
                temp_dir,
                T_samples,
                templates_up_loc,
                scalings,
                n_chans,
                nn_denoise,
                denoised_tpca.tpca if denoised_tpca is not None else None,
                do_reassignment,
                reassignment_tpca,
                reassignment_proposed_pairs_up,
                reassignment_temps_up_loc,
                featurizers,
            ),
            context=ctx,
        ) as pool:
            if n_jobs is not None and n_jobs > 1:
                print(" Ok.", flush=True)
            for result in xqdm(
                pool.imap(_extract_deconv_worker, start_samples),
                desc="extract deconv",
                smoothing=0,
                total=len(start_samples),
                pbar=pbar,
            ):
                h5["last_batch_end"][()] = result.last_batch_end

                if save_residual:
                    np.load(result.resid_path).tofile(resid)
                    Path(result.resid_path).unlink()

                if save_outlier_scores and result.outlier_scores_path:
                    outlier_scores[result.inds] = np.load(
                        result.outlier_scores_path
                    )
                    Path(result.outlier_scores_path).unlink()

                if do_reassignment and result.reassignments_path:
                    reassigned_labels_up[result.inds] = np.load(
                        result.reassignments_path
                    )
                    Path(result.reassignments_path).unlink()

                for f, dset in zip(featurizers, feature_dsets):
                    fnpy = temp_dir / f"{result.batch_prefix}_{f.name}.npy"
                    if fnpy.exists():
                        dset[result.inds] = np.load(fnpy)
                        Path(fnpy).unlink()

    if save_residual:
        resid.close()
        return out_h5, residual_path

    return out_h5


JobResult = namedtuple(
    "JobResult",
    [
        "last_batch_end",
        "batch_prefix",
        "resid_path",
        "outlier_scores_path",
        "reassignments_path",
        "inds",
    ],
)


def _extract_deconv_worker(start_sample):
    # an easy name to extract the params set by _extract_deconv_init
    p = _extract_deconv_worker
    end_sample = min(p.T_samples, start_sample + p.batch_length)
    batch_str = f"{start_sample:012d}"
    which_spikes = np.flatnonzero(
        (p.spike_index_up[:, 0] >= start_sample)
        & (p.spike_index_up[:, 0] < end_sample)
    )
    spike_index = p.spike_index_up[which_spikes]
    spike_train = p.spike_train_up[which_spikes]
    scalings = p.scalings[which_spikes]

    # load raw recording with padding
    at_start = start_sample == 0
    at_end = end_sample == p.T_samples
    buffer_left = p.trough_offset * (not at_start)
    buffer_right = (p.spike_length_samples - p.trough_offset) * (not at_end)
    resid = subtract.read_data(
        p.standardized_bin,
        np.float32,
        start_sample - buffer_left,
        end_sample + buffer_right,
        p.n_chans,
    )

    # if no spikes, bail early now before padding the residual
    if not spike_train.size:
        if p.save_residual:
            np.save(
                p.temp_dir / f"resid_{batch_str}.npy",
                resid,
            )
        return JobResult(
            end_sample,
            batch_str,
            p.temp_dir / f"resid_{batch_str}.npy",
            None,
            None,
            which_spikes,
        )

    # pad left/right if necessary
    # note, for spikes which end up loading in this buffer,
    # the localizations will be junk.
    pad_left = p.trough_offset * at_start
    pad_right = (p.spike_length_samples - p.trough_offset) * at_end
    if pad_left > 0 or pad_right > 0:
        resid = np.pad(resid, [(pad_left, pad_right), (0, 0)])

    # subtract templates in-place
    rel_times = np.arange(
        -start_sample + buffer_left + pad_left - p.trough_offset,
        -start_sample
        + buffer_left
        + pad_left
        + p.spike_length_samples
        - p.trough_offset,
    )
    for i in range(len(spike_index)):
        resid[spike_index[i, 0] + rel_times] -= (
            scalings[i] * p.templates_up[spike_train[i, 1]]
        )

    if p.save_residual:
        np.save(
            p.temp_dir / f"resid_{batch_str}.npy",
            resid[
                buffer_left + pad_left : len(resid) - buffer_right - pad_right
            ],
        )

    # -- load collision-cleaned waveforms
    if p.do_clean:
        # initialize by reading from residual
        resid_waveforms = spikeio.read_waveforms_in_memory(
            resid,
            spike_index,
            spike_length_samples=p.spike_length_samples,
            channel_index=p.channel_index,
            buffer=-start_sample + buffer_left + pad_left,
        )

        # outlier score is easy if we don't do reassignment
        if not p.do_reassignment and p.save_outlier_scores:
            outlier_scores = np.nanmax(
                np.abs(
                    waveform_utils.apply_tpca(
                        resid_waveforms, p.reassignment_tpca
                    )
                ),
                axis=(1, 2),
            )
            np.save(
                p.temp_dir / f"outlier_scores_{batch_str}.npy", outlier_scores
            )

        # now add the templates
        cleaned_waveforms = resid_waveforms
        del resid_waveforms
        cleaned_waveforms += (
            scalings[:, None, None] * p.templates_up_loc[spike_train[:, 1]]
        )

        # compute and save features for cleaned wfs
        for f in p.featurizers:
            feat = f.transform(
                spike_index[:, 1],
                cleaned_wfs=cleaned_waveforms,
            )
            if feat is not None:
                np.save(
                    p.temp_dir / f"{batch_str}_{f.name}.npy",
                    feat,
                )

    # -- reassign these waveforms
    if p.do_reassignment:
        new_labels_up, outlier_scores = reassignment.reassign_waveforms(
            spike_train[:, 1],
            cleaned_waveforms,
            p.reassignment_pairs_up,
            p.reassignment_temps_up_loc,
            tpca=p.reassignment_tpca,
        )
        np.save(p.temp_dir / f"new_labels_up_{batch_str}.npy", new_labels_up)
        np.save(p.temp_dir / f"outlier_scores_{batch_str}.npy", outlier_scores)

    # -- denoise them
    if p.do_denoise:
        denoised_waveforms = subtract.full_denoising(
            cleaned_waveforms,
            spike_index[:, 1],
            p.channel_index,
            tpca=p.tpca,
            device=p.device,
            denoiser=p.denoiser,
        )
        del cleaned_waveforms

        # compute and save features for denoised wfs
        for f in p.featurizers:
            feat = f.transform(
                spike_index[:, 1],
                denoised_wfs=denoised_waveforms,
            )
            if feat is not None:
                np.save(
                    p.temp_dir / f"{batch_str}_{f.name}.npy",
                    feat,
                )

    return JobResult(
        end_sample,
        batch_str,
        p.temp_dir / f"resid_{batch_str}.npy",
        p.temp_dir / f"outlier_scores_{batch_str}.npy",
        p.temp_dir / f"new_labels_up_{batch_str}.npy",
        which_spikes,
    )


def _extract_deconv_init(
    geom,
    device,
    channel_index,
    batch_length,
    do_clean,
    do_denoise,
    save_outlier_scores,
    save_residual,
    trough_offset,
    spike_index_up,
    spike_train_up,
    standardized_bin,
    spike_length_samples,
    templates_up,
    template_maxchans,
    temp_dir,
    T_samples,
    templates_up_loc,
    scalings,
    n_chans,
    nn_denoise,
    tpca,
    do_reassignment,
    reassignment_tpca,
    reassignment_pairs_up,
    reassignment_temps_up_loc,
    featurizers,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = None
    if nn_denoise:
        denoiser = denoise.SingleChanDenoiser().load().to(device)

    p = _extract_deconv_worker
    p.geom = geom
    p.tpca = tpca
    p.do_reassignment = do_reassignment
    p.reassignment_tpca = reassignment_tpca
    p.device = device
    p.denoiser = denoiser
    p.channel_index = channel_index
    p.do_clean = do_clean
    p.do_denoise = do_denoise
    p.save_outlier_scores = save_outlier_scores
    p.save_residual = save_residual
    p.trough_offset = trough_offset
    p.spike_index_up = spike_index_up
    p.spike_train_up = spike_train_up
    p.standardized_bin = standardized_bin
    p.spike_length_samples = spike_length_samples
    p.templates_up = templates_up
    p.templates_up_loc = templates_up_loc
    p.scalings = scalings
    p.temp_dir = temp_dir
    p.T_samples = T_samples
    p.batch_length = batch_length
    p.n_chans = n_chans
    p.reassignment_pairs_up = reassignment_pairs_up
    p.reassignment_temps_up_loc = reassignment_temps_up_loc
    p.featurizers = featurizers

    print(".", end="", flush=True)


def load_or_fit_featurizers(
    featurizers,
    h5,
    channel_index,
    templates_up,
    spike_train_up,
    output_directory,
    standardized_bin,
    scalings,
    geom,
    n_sec_train_feats,
    n_sec_chunk=1,
    n_jobs=1,
    sampling_rate=30_000,
    device=None,
    trough_offset=42,
    nn_denoise=True,
    seed=0,
):
    if not any(f.needs_fit for f in featurizers):
        return

    for f in featurizers:
        f.from_h5(h5)

    if not any(f.needs_fit for f in featurizers):
        return

    # run a mini extract with waveform saving and no features
    # to fit the featurizers on

    # -- restrict the spike train to some randomly chosen seconds
    t_min = np.ceil(spike_train_up[:, 0].min() / sampling_rate)
    t_max = np.floor(spike_train_up[:, 0].max() / sampling_rate)
    valid_times = np.random.default_rng(seed).choice(
        np.arange(t_min, t_max),
        size=min(n_sec_train_feats, t_max - t_min),
        replace=False,
    )
    which_mini = (np.isin(spike_train_up[:, 0] // sampling_rate, valid_times),)
    spike_train_up_mini = spike_train_up[which_mini]
    scalings_mini = scalings[which_mini]

    # -- mini extraction
    (output_directory / "mini_extract_feats").mkdir(exist_ok=True)
    extract_h5 = extract_deconv(
        templates_up,
        spike_train_up_mini,
        output_directory / "mini_extract_feats",
        standardized_bin,
        scalings=scalings_mini,
        channel_index=channel_index,
        n_channels_extract=None,
        extract_radius_um=None,
        subtraction_h5=None,
        tpca_rank=None,
        save_residual=False,
        save_cleaned_waveforms=True,
        save_cleaned_tpca_projs=False,
        save_denoised_waveforms=True,
        do_denoised_tpca=False,
        save_denoised_tpca_projs=False,
        save_outlier_scores=False,
        do_reassignment=False,
        geom=geom,
        localize=False,
        n_sec_chunk=n_sec_chunk,
        n_jobs=n_jobs,
        sampling_rate=sampling_rate,
        device=device,
        trough_offset=trough_offset,
        pbar="Extract deconv: train featurizers",
        nn_denoise=nn_denoise,
        seed=seed,
    )

    # -- fit features
    with h5py.File(extract_h5, "r") as mini_h5:
        max_channels = mini_h5["spike_index"][:, 1]
        print(
            f"Training featurizers on {max_channels.size} waveforms from mini-extraction."
        )
        cleaned_wfs = mini_h5["cleaned_waveforms"][:]
        for f in featurizers:
            f.fit(max_channels=max_channels, cleaned_wfs=cleaned_wfs)
        del cleaned_wfs
        denoised_wfs = mini_h5["denoised_waveforms"][:]
        for f in featurizers:
            f.fit(max_channels=max_channels, denoised_wfs=denoised_wfs)
        del denoised_wfs

    # clean up after ourselves
    shutil.rmtree(output_directory / "mini_extract_feats")

    if any(f.needs_fit for f in featurizers):
        raise ValueError("Unable to fit all featurizers.")

    # save to output
    for f in featurizers:
        f.to_h5(h5)

    # done. at this point the caller can rely on fitted featurizers.


def xqdm(it, pbar=True, **kwargs):
    if pbar:
        if isinstance(pbar, str):
            kwargs["desc"] = pbar
        return tqdm(it, **kwargs)
    else:
        return it
