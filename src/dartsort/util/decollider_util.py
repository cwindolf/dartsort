"""
Note: a lot of the waveforms in this file are NCT rather than NTC, because this
is the expected input format for conv1ds.
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from ..transform.decollider import SingleChannelPredictor
from ..transform.single_channel_denoiser import SingleChannelDenoiser
from . import spikeio


def train_decollider(
    net,
    recordings,
    templates_train=None,
    templates_val=None,
    detection_times_train=None,
    detection_channels_train=None,
    detection_times_val=None,
    detection_channels_val=None,
    channel_index=None,
    channel_min_amplitude=0.0,
    channel_jitter_index=None,
    examples_per_epoch=10_000,
    noise_same_chans=False,
    noise2_alpha=1.0,
    trough_offset_samples=42,
    spike_length_samples=121,
    data_random_seed=0,
    noise_max_amplitude=np.inf,
    validation_oversamples=3,
    n_unsupervised_val_examples=2000,
    max_n_epochs=500,
    early_stop_decrease_epochs=5,
    batch_size=64,
    loss_class=torch.nn.MSELoss,
    opt_class=torch.optim.Adam,
    opt_kw=None,
    val_every=1,
    device=None,
    show_progress=True,
):
    """
    Arguments
    ---------
    net : torch.nn.Module
        Should expect input shape NCL, where C is the number of input
        channels (i.e. channel_index.shape[1]) and L is spike_length_samples.
        If C>1, will also receive a N1L array "mask" containing 1s on valid
        inputs and 0s elsewhere. The loss will be ignored on the zeros.
    recordings : List[BaseRecording]
    templates_{train,val} : List[np.ndarray], length == len(recordings)
    channel_index : Optional[np.ndarray]
        For training multi-chan nets
    recording_channel_indices : Optional[List[np.ndarray]]
        Per-recording channel indices (subsets of channel_index) to allow
        for holes in individual recordings' channel sets

    Returns
    -------
    net : torch.nn.Module
    train_losses : np.ndarray
    validation_dataframe : pd.DataFrame
    """
    rg = np.random.default_rng(data_random_seed)

    device = torch.device(device)

    # initial validation
    train_records = []
    val_records = []

    # allow training on recordings with different channels missing
    (
        n_channels_full,
        recording_channel_indices,
        channel_subsets,
        channel_index,
    ) = reconcile_channels(recordings, channel_index)

    # combine templates on different channels using NaN padding
    # the NaNs inform masking below
    # these are also padded with an extra channel of NaNs, to help
    # with indexing below
    templates_train_recording_origin = original_train_template_index = None
    if templates_train is not None:
        (
            templates_train,
            templates_train_recording_origin,
            original_train_template_index,
        ) = combine_templates(templates_train, channel_subsets)
        assert spike_length_samples == templates_train.shape[2]

    templates_val_recording_origin = original_val_template_index = None
    if templates_val is not None:
        (
            templates_val,
            templates_val_recording_origin,
            original_val_template_index,
        ) = combine_templates(templates_val, channel_subsets)
        assert spike_length_samples == templates_val.shape[2]
    if opt_kw is None:
        opt_kw = {}
    opt = opt_class(net.parameters(), **opt_kw)
    criterion = loss_class()
    examples_seen = 0
    val_losses = []
    xrange = trange if show_progress else range
    for epoch in xrange(max_n_epochs):
        epoch_dt = 0.0

        # get data
        tic = time.perf_counter()
        epoch_data = load_epoch(
            recordings,
            templates=templates_train,
            detection_times=detection_times_train,
            detection_channels=detection_channels_train,
            channel_index=channel_index,
            template_recording_origin=templates_train_recording_origin,
            recording_channel_indices=recording_channel_indices,
            channel_min_amplitude=channel_min_amplitude,
            channel_jitter_index=channel_jitter_index,
            examples_per_epoch=examples_per_epoch,
            noise_same_chans=noise_same_chans,
            noise2_alpha=noise2_alpha,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            data_random_seed=rg,
            noise_max_amplitude=noise_max_amplitude,
        )
        toc = time.perf_counter()
        epoch_data_load_wall_dt_s = toc - tic

        # train
        epoch_losses = []
        for i0 in range(
            0, len(epoch_data.noisy_waveforms) - batch_size, batch_size
        ):
            tic = time.perf_counter()

            i1 = i0 + batch_size
            noised_batch = epoch_data.noisier_waveforms[i0:i1].to(device)
            target_batch = epoch_data.noisy_waveforms[i0:i1].to(device)
            masks = None
            if epoch_data.channel_masks is not None:
                masks = epoch_data.channel_masks[i0:i1].to(device)

            opt.zero_grad()
            pred = net(noised_batch, channel_masks=masks)
            if masks is not None:
                loss = criterion(
                    pred * masks[:, :, None], target_batch * masks[:, :, None]
                )
            else:
                loss = criterion(pred, target_batch)
            loss.backward()
            opt.step()

            toc = time.perf_counter()
            batch_dt = toc - tic

            loss = float(loss.numpy(force=True))

            epoch_losses.append(loss)
            train_records.append(
                dict(
                    loss=loss,
                    batch_train_wall_dt_s=batch_dt,
                    epoch=epoch,
                    samples=examples_seen,
                )
            )

            # learning trackers
            examples_seen += noised_batch.shape[0]
            epoch_dt += batch_dt

        if epoch % val_every:
            continue

        # evaluate
        val_record = evaluate_decollider(
            net,
            recordings,
            templates=templates_val,
            detection_times=detection_times_val,
            detection_channels=detection_channels_val,
            recording_channel_indices=recording_channel_indices,
            template_recording_origin=templates_val_recording_origin,
            original_template_index=original_val_template_index,
            n_oversamples=validation_oversamples,
            n_unsupervised_val_examples=n_unsupervised_val_examples,
            channel_index=channel_index,
            channel_min_amplitude=channel_min_amplitude,
            channel_jitter_index=channel_jitter_index,
            noise_same_chans=noise_same_chans,
            noise2_alpha=noise2_alpha,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            data_random_seed=rg,
            noise_max_amplitude=noise_max_amplitude,
            device=device,
            summarize=True,
        )
        val_record["epoch"] = epoch
        val_record["epoch_train_wall_dt_s"] = epoch_dt
        val_record["epoch_data_load_wall_dt_s"] = epoch_data_load_wall_dt_s
        val_records.append(val_record)
        val_losses.append(val_record["val_loss"])
        if show_progress:
            summary = f"epoch {epoch}. "
            summary += f"mean train loss: {np.mean(epoch_losses):0.3f}, "
            summary += f"init train loss: {epoch_losses[0]:0.3f}, "
            summary += ", ".join(
                f"{k}: {v:0.3f}" for k, v in val_record.items() if k != "epoch"
            )
            tqdm.write(summary)

        # stop early
        if not early_stop_decrease_epochs or epoch < early_stop_decrease_epochs:
            continue

        # See: Early Stopping -- But When?
        best_epoch = np.argmin(val_losses)
        if epoch - best_epoch > early_stop_decrease_epochs:
            if show_progress:
                tqdm.write(f"Early stopping at {epoch=}, since {best_epoch=}.")
            break

    validation_dataframe = pd.DataFrame.from_records(val_records)
    training_dataframe = pd.DataFrame.from_records(train_records)
    return net, training_dataframe, validation_dataframe


# -- data helpers


@dataclass
class EpochData:
    noisy_waveforms: torch.Tensor
    channels: torch.LongTensor
    noisier_waveforms: torch.Tensor
    channel_masks: Optional[torch.Tensor] = None

    # for hybrid experiments
    gt_waveforms: Optional[torch.Tensor] = None
    template_indices: Optional[torch.LongTensor] = None


def load_epoch(
    recordings,
    templates=None,
    detection_times=None,
    detection_channels=None,
    template_recording_origin=None,
    channel_index=None,
    recording_channel_indices=None,
    channel_min_amplitude=0.0,
    channel_jitter_index=None,
    examples_per_epoch=10_000,
    n_oversamples=1,
    noise_same_chans=False,
    noise2_alpha=1.0,
    trough_offset_samples=42,
    spike_length_samples=121,
    data_random_seed=0,
    noise_max_amplitude=np.inf,
):
    rg = np.random.default_rng(data_random_seed)

    if templates is not None:
        (
            channels,
            gt_waveforms,
            noisy_waveforms,
            noise_chans,
            which_rec,
            which_templates,
        ) = get_noised_hybrid_waveforms(
            templates,
            noise_same_chans=noise_same_chans,
            recording_channel_indices=recording_channel_indices,
            trough_offset_samples=trough_offset_samples,
            template_recording_origin=template_recording_origin,
            noise_max_amplitude=noise_max_amplitude,
            channel_index=channel_index,
            channel_jitter_index=channel_jitter_index,
            n_oversamples=n_oversamples,
            channel_min_amplitude=channel_min_amplitude,
            examples_per_epoch=examples_per_epoch,
            data_random_seed=rg,
        )
    else:
        assert detection_times is not None
        assert detection_channels is not None
        which_templates = gt_waveforms = None

        noisy_waveforms, channels, which_rec = load_spikes(
            recordings,
            times=detection_times,
            channels=detection_channels,
            recording_channel_indices=recording_channel_indices,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            n=examples_per_epoch,
            rg=rg,
            to_torch=True,
        )

    # double noise
    noisier_waveforms = load_noise(
        recordings,
        channels=channels if noise_same_chans else None,
        which_rec=which_rec,
        recording_channel_indices=recording_channel_indices,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        n=noisy_waveforms.shape[0],
        max_abs_amp=noise_max_amplitude,
        dtype=recordings[0].dtype,
        rg=rg,
        to_torch=True,
        alpha=noise2_alpha,
    )
    noisier_waveforms += noisy_waveforms

    channel_masks = np.isfinite(noisier_waveforms[:, :, 0])
    channel_masks = torch.as_tensor(channel_masks, dtype=torch.bool)
    if gt_waveforms is not None:
        gt_waveforms[~channel_masks] = 0.0
    noisy_waveforms[~channel_masks] = 0.0
    noisier_waveforms[~channel_masks] = 0.0

    return EpochData(
        noisy_waveforms=noisy_waveforms,
        noisier_waveforms=noisier_waveforms,
        channels=channels,
        channel_masks=channel_masks,
        gt_waveforms=gt_waveforms,
        template_indices=which_templates,
    )


@torch.no_grad()
def evaluate_decollider(
    net,
    recordings,
    templates=None,
    detection_times=None,
    detection_channels=None,
    recording_channel_indices=None,
    template_recording_origin=None,
    original_template_index=None,
    n_oversamples=1,
    n_unsupervised_val_examples=2000,
    channel_index=None,
    channel_min_amplitude=0.0,
    channel_jitter_index=None,
    noise_same_chans=False,
    trough_offset_samples=42,
    spike_length_samples=121,
    data_random_seed=0,
    noise_max_amplitude=np.inf,
    noise2_alpha=1.0,
    device=None,
    summarize=True,
):
    tic = time.perf_counter()
    val_data = load_epoch(
        recordings,
        templates=templates,
        detection_times=detection_times,
        detection_channels=detection_channels,
        template_recording_origin=template_recording_origin,
        channel_index=channel_index,
        recording_channel_indices=recording_channel_indices,
        channel_min_amplitude=channel_min_amplitude,
        channel_jitter_index=channel_jitter_index,
        examples_per_epoch=None
        if templates is not None
        else n_unsupervised_val_examples,
        n_oversamples=n_oversamples,
        noise_same_chans=noise_same_chans,
        noise2_alpha=noise2_alpha,
        trough_offset_samples=trough_offset_samples,
        data_random_seed=data_random_seed,
        noise_max_amplitude=noise_max_amplitude,
        spike_length_samples=spike_length_samples,
    )
    toc = time.perf_counter()
    val_data_load_wall_dt_s = toc - tic

    # metrics timer
    # unsupervised task for learning: predict wf from noised_wf
    # preds_noised = net(val_data.noisier_waveforms)
    preds_noisy = batched_infer(
        net,
        val_data.noisier_waveforms,
        channel_masks=val_data.channel_masks,
        device=device,
    )
    if summarize:
        val_loss = F.mse_loss(preds_noisy, val_data.noisy_waveforms)

    if templates is None:
        assert summarize
        return dict(
            val_loss=float(val_loss),
            val_data_load_wall_dt_s=val_data_load_wall_dt_s,
            val_metrics_wall_dt_s=time.perf_counter() - tic,
        )

    # below here, summarize is True, and we are working with templates
    # so that gt_waveforms exists

    # supervised task: predict gt_wf (template) from wf (template + noise)
    # template_preds_naive = net(val_data.noisy_waveforms)
    template_preds_naive = batched_infer(
        net,
        val_data.noisy_waveforms,
        channel_masks=val_data.channel_masks,
        device=device,
    )
    naive_sup_max_err = (
        torch.abs(template_preds_naive - val_data.gt_waveforms)
        .max(dim=(1, 2))
        .values
    )

    # noisier2noise prediction of templates
    # template_preds_n2n = net.n2n_forward(
    #     val_data.noisier_waveforms,
    #     channel_masks=val_data.channel_masks,
    #     alpha=noise2_alpha,
    # )
    template_preds_n2n = batched_n2n_infer(
        net,
        val_data.noisier_waveforms,
        channel_masks=val_data.channel_masks,
        device=device,
        alpha=noise2_alpha,
    )
    n2n_sup_max_err = (
        torch.abs(template_preds_naive - val_data.gt_waveforms)
        .max(dim=(1, 2))
        .values
    )

    if summarize:
        naive_template_mean_max_err = naive_sup_max_err.mean()
        naive_template_mse = F.mse_loss(
            template_preds_naive, val_data.gt_waveforms
        )

        n2n_template_mean_max_err = n2n_sup_max_err.mean()
        n2n_template_mse = F.mse_loss(template_preds_n2n, val_data.gt_waveforms)

        return dict(
            naive_template_mean_max_err=naive_template_mean_max_err,
            naive_template_mse=naive_template_mse,
            n2n_template_mean_max_err=n2n_template_mean_max_err,
            n2n_template_mse=n2n_template_mse,
            val_loss=val_loss,
            val_data_load_wall_dt_s=val_data_load_wall_dt_s,
            val_metrics_wall_dt_s=time.perf_counter() - tic,
        )

    # more detailed per-example comparisons
    # break down some covariates
    gt_amplitude = templates.ptp(1)[val_data.which, val_data.channels]
    noise1_norm = torch.linalg.norm(
        val_data.noisy_waveforms - val_data.gt_waveforms, dim=(1, 2)
    )
    noise2_norm = torch.linalg.norm(
        val_data.noisier_waveforms - val_data.noisy_waveforms, dim=(1, 2)
    )

    # errors for naive template prediction
    naive_diff = template_preds_naive - val_data.gt_waveforms
    naive_template_mse = torch.square(naive_diff).mean(dim=(1, 2))
    naive_template_max_err = torch.abs(naive_diff).max(dim=(1, 2)).values

    # errors for noisier2noise template prediction
    n2n_diff = template_preds_n2n - val_data.gt_waveforms
    n2n_template_mse = torch.square(n2n_diff).mean(dim=(1, 2))
    n2n_template_max_err = torch.abs(n2n_diff).max(dim=(1, 2)).values

    return pd.DataFrame(
        dict(
            combined_template_index=val_data.template_indices,
            recording_index=template_recording_origin[
                val_data.template_indices
            ],
            original_template_index=original_template_index[
                val_data.template_indices
            ],
            gt_amplitude=gt_amplitude.numpy(force=True),
            noise1_norm=noise1_norm.numpy(force=True),
            noise2_norm=noise2_norm.numpy(force=True),
            naive_template_mse=naive_template_mse.numpy(force=True),
            naive_template_max_err=naive_template_max_err.numpy(force=True),
            n2n_template_mse=n2n_template_mse.numpy(force=True),
            n2n_template_max_err=n2n_template_max_err.numpy(force=True),
        )
    )


# -- hybrid data helper


def get_noised_hybrid_waveforms(
    templates,
    noise_same_chans=False,
    recording_channel_indices=None,
    trough_offset_samples=42,
    template_recording_origin=None,
    noise_max_amplitude=np.inf,
    channel_index=None,
    channel_jitter_index=None,
    n_oversamples=1,
    channel_min_amplitude=0.0,
    examples_per_epoch=10_000,
    data_random_seed=0,
):
    """
    templates are NCT here.

    If single_channel:
        Random single channels of templates_train landing inside their channel-indexed
        chans are returned, along with which channel each one was.
    Else:
        Random templates are selected
    """
    rg = np.random.default_rng(data_random_seed)

    # channel logic
    amplitude_vectors = templates.ptp(2)
    max_channels = amplitude_vectors.argmax(1)
    kept = np.arange(len(templates))
    if channel_min_amplitude > 0:
        kept = kept[amplitude_vectors[kept].max(1) > channel_min_amplitude]

    # this is where the random sampling happens!
    which = kept
    if examples_per_epoch is not None:
        which = rg.choice(kept, size=examples_per_epoch)
    if n_oversamples != 1:
        which = np.repeat(which, n_oversamples)

    # randomly jitter channel neighborhoods
    # use a large channel_jitter_index when training single channel nets
    channels = max_channels[which]
    if channel_jitter_index is not None:
        for i, c in enumerate(channels):
            choices = channel_jitter_index[c]
            choices = choices[choices < len(channel_jitter_index)]
            channels[i] = rg.choice(choices)

    # load noisy_waveforms
    waveform_channels = channel_index[channels][:, :, None]
    # multi channel, or i guess you could just have one channel
    # in your channel index if you're into that kind of thing
    gt_waveforms = templates[
        which[:, None, None],
        waveform_channels,
        np.arange(templates.shape[1])[None, None, :],
    ]

    # keep this fellow on CPU
    channel_masks = np.isfinite(gt_waveforms[:, :, 0])
    gt_waveforms[~channel_masks] = 0.0
    channel_masks = torch.from_numpy(channel_masks)

    # to torch
    channels = torch.from_numpy(channels)
    gt_waveforms = torch.from_numpy(gt_waveforms)

    # apply noise
    noise_chans = channels if noise_same_chans else None
    which_rec = template_recording_origin[which] if noise_same_chans else None
    noisy_waveforms = load_noise(
        channels=noise_chans,
        which_rec=which_rec,
        recording_channel_indices=recording_channel_indices,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=templates.shape[1],
        n=gt_waveforms.shape[0],
        max_abs_amp=noise_max_amplitude,
        dtype=gt_waveforms.dtype,
        rg=rg,
        to_torch=True,
    )
    noisy_waveforms += gt_waveforms

    return (
        channels,
        channel_masks,
        gt_waveforms,
        noisy_waveforms,
        which_rec,
        which,
    )


# -- noise helpers


def load_noise(
    recordings,
    channels=None,
    which_rec=None,
    recording_channel_indices=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    n=100,
    max_abs_amp=np.inf,
    dtype=None,
    alpha=1.0,
    rg=0,
    to_torch=True,
):
    """Get NCT noise arrays."""
    rg = np.random.default_rng(rg)

    if which_rec is None:
        which_rec = rg.integers(len(recordings), size=n)

    if dtype is None:
        dtype = recordings[0].dtype

    c = (
        recording_channel_indices[0].shape[1]
        if recording_channel_indices is not None
        else 1
    )

    noise = np.full((n, c, spike_length_samples), np.nan, dtype=dtype)
    for i, rec in enumerate(recordings):
        mask = np.flatnonzero(which_rec == i)
        rec_channels = None
        rec_ci = recording_channel_indices[i]
        if channels is not None:
            rec_channels = channels[mask]
        noise[mask] = load_noise_singlerec(
            rec,
            channels=rec_channels,
            channel_index=rec_ci,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            n=mask.size,
            max_abs_amp=max_abs_amp,
            dtype=dtype,
            rg=rg,
            alpha=alpha,
            to_torch=False,
        )

    if to_torch:
        noise = torch.from_numpy(noise)

    return noise


def load_noise_singlerec(
    recording,
    channels=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    channel_index=None,
    n=100,
    max_abs_amp=np.inf,
    alpha=1.0,
    dtype=None,
    rg=0,
    to_torch=True,
):
    rg = np.random.default_rng(rg)

    if dtype is None:
        dtype = recording.dtype

    c = channel_index.shape[1] if channel_index is not None else 1

    noise = np.full((n, c, spike_length_samples), np.nan, dtype=dtype)
    needs_load_ix = np.full((n,), True, dtype=bool)

    nc = recording.get_num_channels()
    nt = recording.get_num_samples()
    mint = trough_offset_samples
    maxt = nt - (spike_length_samples - trough_offset_samples)

    while np.any(needs_load_ix):
        n_load = needs_load_ix.sum()
        times = rg.integers(mint, maxt, size=n_load)
        order = np.argsort(times)

        if channels is None:
            channels = rg.integers(0, nc, size=n_load)

        wfs = spikeio.read_waveforms_channel_index(
            recording,
            times[order],
            channel_index,
            channels[order],
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )
        wfs = wfs.transpose(0, 2, 1)

        noise[needs_load_ix] = wfs[np.argsort(order)]
        needs_load_ix = np.nanmax(np.abs(noise)) > max_abs_amp

    if alpha != 1.0:
        noise *= alpha

    if to_torch:
        noise = torch.from_numpy(noise)

    return noise


# -- signal helpers


def load_spikes(
    recordings,
    times,
    channels,
    which_rec=None,
    recording_channel_indices=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    n=100,
    dtype=None,
    rg=0,
    to_torch=True,
):
    """Get NCT noise arrays."""
    rg = np.random.default_rng(rg)

    if which_rec is None:
        which_rec = rg.integers(len(recordings), size=n)

    if dtype is None:
        dtype = recordings[0].dtype

    c = (
        recording_channel_indices[0].shape[1]
        if recording_channel_indices is not None
        else 1
    )

    spikes = np.full((n, c, spike_length_samples), np.nan, dtype=dtype)
    channels_chosen = np.zeros(n, dtype=int)
    for i, rec in enumerate(recordings):
        mask = np.flatnonzero(which_rec == i)
        rec_ci = recording_channel_indices[i]

        (
            spikes[mask],
            channels_chosen[mask],
        ) = load_spikes_singlerec(
            rec,
            times=times[i],
            channels=channels[i],
            channel_index=rec_ci,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            n=mask.size,
            dtype=dtype,
            rg=rg,
            to_torch=False,
        )

    if to_torch:
        spikes = torch.from_numpy(spikes)

    return spikes, channels_chosen, which_rec


def load_spikes_singlerec(
    recording,
    times,
    channels,
    trough_offset_samples=42,
    spike_length_samples=121,
    channel_index=None,
    n=100,
    dtype=None,
    rg=0,
    to_torch=True,
):
    rg = np.random.default_rng(rg)

    if dtype is None:
        dtype = recording.dtype

    which = rg.choice(times.size, size=n, replace=False)
    times = times[which]
    channels = channels[which]
    order = np.argsort(times)

    wfs = spikeio.read_waveforms_channel_index(
        recording,
        times[order],
        channel_index,
        channels[order],
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
    )
    wfs = wfs.transpose(0, 2, 1)

    spikes = wfs[np.argsort(order)]

    if to_torch:
        spikes = torch.from_numpy(spikes)

    return spikes, channels


# -- multi-recording channel logic


def reconcile_channels(recordings, channel_index):
    """Validate that the multi-chan setup is workable, reconcile chans across recordings"""
    full_channel_set = recordings[0].channel_ids
    for rec in recordings:
        ids = [int(i.lstrip("AP")) for i in rec.channel_ids]
        assert np.array_equal(ids, np.sort(ids))
        full_channel_set = np.union1d(full_channel_set, rec.channel_ids)
    n_channels_full = full_channel_set.size

    if channel_index is not None:
        assert n_channels_full == channel_index.shape[0]
        assert channel_index.min() >= 0
        assert channel_index.max() <= channel_index.shape[0]
    else:
        channel_index = np.arange(n_channels_full)[:, None]

    channel_subsets = []
    recording_channel_indices = None
    recording_channel_indices = []
    for rec in recordings:
        subset = np.flatnonzero(np.isin(full_channel_set, rec.channel_ids))
        channel_subsets.append(subset)
        subset_ci = subset_recording_channel_index(channel_index, subset)
        recording_channel_indices.append(subset_ci)

    return (
        n_channels_full,
        recording_channel_indices,
        channel_subsets,
        channel_index,
    )


def subset_recording_channel_index(full_channel_index, channel_subset):
    """The output of this function is a channel index containing indices into the
    recording, but matching full_channel_index, just with holes.
    """
    subset_channel_index = np.full_like(full_channel_index, len(channel_subset))
    for recchan, fullchan in enumerate(channel_subset):
        full_ci = full_channel_index[fullchan]
        subset_channel_index[recchan] = np.searchsorted(channel_subset, full_ci)
    return subset_channel_index


def combine_templates(templates, channel_subsets):
    t = templates[0].shape[1]
    c = channel_subsets[0].size
    n = sum(map(len, templates))

    combined = np.full(
        (n, c + 1, t), fill_value=np.nan, dtype=templates[0].dtype
    )
    template_recording_origin = np.zeros(n, dtype=int)
    original_template_index = np.zeros(n, dtype=int)
    i = 0
    for r, (temps, subset) in enumerate(zip(templates, channel_subsets)):
        j = i + temps.shape[0]
        template_recording_origin[i:j] = r
        original_template_index[i:j] = np.arange(j - i)
        combined[i:j, subset] = temps

    return combined, template_recording_origin, original_template_index


# -- inference utils


def batched_infer(
    net,
    noisy_waveforms,
    channel_masks=None,
    batch_size=16,
    device=None,
    show_progress=False,
):
    is_tensor = torch.is_tensor(noisy_waveforms)
    if is_tensor:
        out = torch.empty_like(noisy_waveforms)
        out = out.pin_memory()
    else:
        out = np.empty_like(noisy_waveforms)

    xrange = trange if show_progress else range
    for batch_start in xrange(len(noisy_waveforms)):
        wfs = noisy_waveforms[batch_start : batch_start + batch_size]
        if not is_tensor:
            wfs = torch.from_numpy(wfs)
        wfs = wfs.to(device)

        cms = None
        if channel_masks is not None:
            cms = channel_masks[batch_start : batch_start + batch_size]
            if not is_tensor:
                cms = torch.from_numpy(cms).to(torch.bool)
            cms = cms.to(device)
        if cms is None and wfs.shape[1] > 1 and torch.isnan(wfs).any():
            cms = torch.isfinite(wfs[:, :, 0])

        wfs = net.predict(wfs, channel_masks=cms)

        if is_tensor:
            out[batch_start : batch_start + batch_size].copy_(
                wfs, non_blocking=True
            )
        else:
            out[batch_start : batch_start + batch_size] = wfs.numpy(force=True)

    return out


def batched_n2n_infer(
    net,
    noisier_waveforms,
    channel_masks=None,
    batch_size=16,
    device=None,
    alpha=1.0,
    show_progress=False
):
    is_tensor = torch.is_tensor(noisier_waveforms)
    if is_tensor:
        out = torch.empty_like(noisier_waveforms)
    else:
        out = np.empty_like(noisier_waveforms)

    xrange = trange if show_progress else range
    for batch_start in xrange(len(noisier_waveforms)):
        wfs = noisier_waveforms[batch_start : batch_start + batch_size]
        wfs = torch.as_tensor(wfs).to(device)

        cms = None
        if channel_masks is not None:
            cms = channel_masks[batch_start : batch_start + batch_size]
            if not is_tensor:
                cms = torch.from_numpy(cms).to(torch.bool)
        if cms is None and wfs.shape[1] > 1 and torch.isnan(wfs).any():
            cms = torch.isfinite(wfs[:, :, 0])

        wfs = wfs.to(device)
        wfs = net.n2n_predict(wfs, channel_masks=cms, alpha=alpha)

        if is_tensor:
            out[batch_start : batch_start + batch_size] = wfs.to(out.device)
        else:
            out[batch_start : batch_start + batch_size] = wfs.numpy(force=True)

    return out


# -- for testing


class SCDAsDecollider(SingleChannelDenoiser, SingleChannelPredictor):
    def forward(self, x):
        """N1T -> N1T"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x[:, None, :]
