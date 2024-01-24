"""
Note: a lot of the waveforms in this file are NCT rather than NTC, because this
is the expected input format for conv1ds.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

from . import spikeio


def hybrid_train(
    net,
    recordings,
    templates_train,
    templates_val,
    channel_index=None,
    channel_min_amplitude=0.0,
    channel_jitter_index=None,
    examples_per_epoch=10_000,
    noise_same_chans=False,
    noise2_alpha=1.0,
    trough_offset_samples=42,
    data_random_seed=0,
    noise_max_amplitude=np.inf,
    validation_oversamples=3,
    max_n_epochs=500,
    early_stop_decrease_epochs=10,
    batch_size=64,
    loss_class=torch.nn.MSELoss,
    device=None,
):
    """
    Arguments
    ---------
    net : torch.nn.Module
        Should expect input shape NCL, where C is the number of input
        channels (i.e. channel_index.shape[1]) and L is spike_length_samples.
        If C>1, will also receive a N1L array "mask" containing 1s on valid
        inputs and 0s elsewhere. The loss will be ignored on the zeros.

    Returns
    -------
    net : torch.nn.Module
    train_losses : np.ndarray
    validation_dataframe : pd.DataFrame
    """
    rg = np.random.default_rng(data_random_seed)
    logger = logging.getLogger("decollider:hybrid_train")

    # initial validation
    train_records = []
    val_records = []

    opt = torch.optim.Adam(net.parameters())
    criterion = loss_class()
    examples_seen = 0
    with logging_redirect_tqdm(loggers=(logger,), tqdm_class=tqdm):
        for epoch in trange(max_n_epochs):
            epoch_dt = 0.0

            # get data
            epoch_data = load_epoch_hybrid(
                recordings,
                templates_train,
                channel_index=channel_index,
                channel_min_amplitude=channel_min_amplitude,
                channel_jitter_index=channel_jitter_index,
                examples_per_epoch=examples_per_epoch,
                noise_same_chans=noise_same_chans,
                noise2_alpha=noise2_alpha,
                trough_offset_samples=trough_offset_samples,
                data_random_seed=rg,
                noise_max_amplitude=noise_max_amplitude,
                device=device,
            )

            # train
            for i0 in range(0, len(epoch_data.waveforms) - batch_size, batch_size):
                tic = time.perf_counter()

                i1 = i0 + batch_size
                noised_batch = epoch_data.noised_waveforms[i0:i1]
                target_batch = epoch_data.waveforms[i0:i1]
                masks = None
                if epoch_data.channel_masks is not None:
                    masks = torch.nonzero(
                        epoch_data.channel_masks[i0:i1], as_tuple=True
                    )

                opt.zero_grad()
                pred = net(noised_batch, channel_masks=masks)
                if masks is not None:
                    pred = pred[masks]
                    target_batch = target_batch[masks]
                loss = criterion(pred, target_batch)
                loss.backward()
                opt.step()

                toc = time.perf_counter()
                batch_dt = toc - tic

                train_records.append(
                    dict(
                        loss=loss.numpy(force=True),
                        wall_dt_s=batch_dt,
                        epoch=epoch,
                        samples=examples_seen,
                    )
                )

                # learning trackers
                examples_seen += noised_batch.shape[0]
                epoch_dt += batch_dt

            # evaluate
            val_record = evaluate_hybrid(
                net,
                recordings,
                templates_val,
                n_oversamples=validation_oversamples,
                channel_index=channel_index,
                channel_min_amplitude=channel_min_amplitude,
                channel_jitter_index=channel_jitter_index,
                noise_same_chans=noise_same_chans,
                noise2_alpha=noise2_alpha,
                trough_offset_samples=trough_offset_samples,
                data_random_seed=rg,
                noise_max_amplitude=noise_max_amplitude,
                device=device,
                summarize=True,
            )
            val_record["epoch"] = epoch
            val_record["epoch_wall_dt_s"] = epoch_dt
            val_records.append(val_record)
            logger.info(", ".join(f"{k}: {v:0.3f}" for k, v in val_record.items()))

            # stop early
            if epoch < early_stop_decrease_epochs:
                continue

            prev_epoch_best_val_mse = min(
                vr["unsup_mse"] for vr in val_records[-early_stop_decrease_epochs:]
            )
            if val_record["unsup_mse"] > prev_epoch_best_val_mse:
                logger.info(
                    f"Early stopping at {epoch=}, since {prev_epoch_best_val_mse=} and {val_record['unsup_mse']=}"
                )

    validation_dataframe = pd.DataFrame.from_records(val_records)
    training_dataframe = pd.DataFrame.from_records(train_records)
    return net, training_dataframe, validation_dataframe


# -- data helpers


@dataclass
class EpochData:
    waveforms: torch.Tensor
    channels: torch.LongTensor
    noised_waveforms: torch.Tensor
    channel_masks: Optional[torch.Tensor] = None

    # for hybrid experiments
    gt_waveforms: Optional[torch.Tensor] = None
    template_indices: Optional[torch.LongTensor] = None


def load_epoch_hybrid(
    recordings,
    templates,
    channel_index=None,
    channel_min_amplitude=0.0,
    channel_jitter_index=None,
    examples_per_epoch=10_000,
    n_oversamples=1,
    noise_same_chans=False,
    noise2_alpha=1.0,
    trough_offset_samples=None,
    data_random_seed=0,
    noise_max_amplitude=np.inf,
    device=None,
):
    """
    If single_channel:
        Random single channels of templates_train landing inside their channel-indexed
        chans are returned, along with which channel each one was.
    Else:
        Random templates are selected
    """
    rg = np.random.default_rng(data_random_seed)

    # channel logic
    amplitude_vectors = templates.ptp(1)
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

    # load waveforms
    if channel_index is None:
        waveform_channels = channels[:, None, None]
        # single channel case
        gt_waveforms = templates[
            which[:, None, None],
            np.arange(templates.shape[1])[None, :, None],
            waveform_channels,
        ]
        channel_masks = None
    else:
        waveform_channels = channel_index[channels][:, None, :]
        # multi channel, or i guess you could just have one channel
        # in your channel index if you're into that kind of thing
        gt_waveforms = templates[
            which[:, None, None],
            np.arange(templates.shape[1])[None, :, None],
            waveform_channels,
        ]
        channel_masks = waveform_channels < len(channel_index)
        # keep this fellow on CPU
        channel_masks = torch.from_numpy(channel_masks)

    # to torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = torch.from_numpy(channels).to(device)
    # NTC -> NCT
    gt_waveforms = gt_waveforms.transpose(0, 2, 1)
    gt_waveforms = torch.from_numpy(gt_waveforms).to(device)

    # apply noise
    noise_chans = channels if noise_same_chans else None
    waveforms = load_noise(
        channels=noise_chans,
        channel_index=channel_index,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=templates.shape[1],
        n=gt_waveforms.shape[0],
        max_abs_amp=noise_max_amplitude,
        dtype=gt_waveforms.dtype,
        rg=rg,
        to_torch=True,
    )
    waveforms += gt_waveforms

    # double noise
    noised_waveforms = load_noise(
        channels=noise_chans,
        channel_index=channel_index,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=templates.shape[1],
        n=gt_waveforms.shape[0],
        max_abs_amp=noise_max_amplitude,
        dtype=gt_waveforms.dtype,
        rg=rg,
        to_torch=True,
        alpha=noise2_alpha,
    )
    noised_waveforms += waveforms

    return EpochData(
        waveforms=waveforms,
        noised_waveforms=noised_waveforms,
        channels=channels,
        channel_masks=channel_masks,
        gt_waveforms=gt_waveforms,
        template_indices=which,
    )


@torch.no_grad
def evaluate_hybrid(
    net,
    recordings,
    templates,
    n_oversamples=1,
    channel_index=None,
    channel_min_amplitude=0.0,
    channel_jitter_index=None,
    noise_same_chans=False,
    trough_offset_samples=None,
    data_random_seed=0,
    noise_max_amplitude=np.inf,
    noise2_alpha=1.0,
    device=None,
    summarize=True,
):
    val_data = load_epoch_hybrid(
        recordings,
        templates,
        channel_index=channel_index,
        channel_min_amplitude=channel_min_amplitude,
        channel_jitter_index=channel_jitter_index,
        examples_per_epoch=None,
        n_oversamples=n_oversamples,
        noise_same_chans=noise_same_chans,
        noise2_alpha=noise2_alpha,
        trough_offset_samples=trough_offset_samples,
        data_random_seed=data_random_seed,
        noise_max_amplitude=noise_max_amplitude,
        device=device,
    )

    # supervised task: predict gt_wf (template) from wf (template + noise)
    template_preds_naive = net(val_data.waveforms)
    naive_sup_max_err = torch.abs(template_preds_naive - val_data.gt_waveforms).max(dim=(1, 2)).values

    # noisier2noise prediction of templates
    template_preds_n2n = net.n2n_predict(val_data.noised_waveforms, channel_masks=val_data.channel_masks, alpha=noise2_alpha)
    n2n_sup_max_err = torch.abs(template_preds_naive - val_data.gt_waveforms).max(dim=(1, 2)).values

    # unsupervised task for learning: predict wf from noised_wf
    preds_noised = net(val_data.noised_waveforms)

    if summarize:
        naive_template_mean_max_err = naive_sup_max_err.mean()
        naive_template_mse = F.mse_loss(template_preds_naive, val_data.gt_waveforms)

        n2n_template_mean_max_err = n2n_sup_max_err.mean()
        n2n_template_mse = F.mse_loss(template_preds_n2n, val_data.gt_waveforms)

        val_loss = F.mse_loss(preds_noised, val_data.waveforms)
        return dict(
            naive_template_mean_max_err=naive_template_mean_max_err,
            naive_template_mse=naive_template_mse,
            n2n_template_mean_max_err=n2n_template_mean_max_err,
            n2n_template_mse=n2n_template_mse,
            val_loss=val_loss,
        )

    # more detailed per-example comparisons
    # break down some covariates
    gt_amplitude = templates.ptp(1)[val_data.which, val_data.channels]
    noise1_norm = torch.linalg.norm(val_data.waveforms - val_data.gt_waveforms, dim=(1, 2))
    noise2_norm = torch.linalg.norm(val_data.noised_waveforms - val_data.waveforms, dim=(1, 2))

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
            template_indices=val_data.template_indices,
            gt_amplitude=gt_amplitude.numpy(force=True),
            noise1_norm=noise1_norm.numpy(force=True),
            noise2_norm=noise2_norm.numpy(force=True),
            naive_template_mse=naive_template_mse.numpy(force=True),
            naive_template_max_err=naive_template_max_err.numpy(force=True),
            n2n_template_mse=n2n_template_mse.numpy(force=True),
            n2n_template_max_err=n2n_template_max_err.numpy(force=True),
        )
    )


# -- noise helpers


def load_noise(
    recordings,
    channels=None,
    channel_index=None,
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
    which_rec = rg.integers(len(recordings), size=n)

    if dtype is None:
        dtype = recordings[0].dtype

    c = channel_index.shape[1] if channel_index is not None else 1

    noise = np.full((n, c, spike_length_samples), np.nan, dtype=dtype)
    for i, rec in enumerate(recordings):
        mask = np.flatnonzero(which_rec == i)
        noise[mask] = load_noise_singlerec(
            rec,
            channels=channels[mask] if channels is not None else None,
            channel_index=channel_index,
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.from_numpy(noise).to(device)

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

        if channel_index is None:
            wfs = spikeio.read_single_channel_waveforms(
                recording,
                times[order],
                channels[order],
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
            )
            wfs = wfs[:, None, :]
        else:
            wfs = spikeio.read_waveforms_channel_index(
                recording,
                times[order],
                channel_index,
                channels[order],
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
            )

        noise[needs_load_ix] = wfs[np.argsort(order)]
        needs_load_ix = np.nanmax(np.abs(noise)) > max_abs_amp

    if alpha != 1.0:
        noise *= alpha

    if to_torch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.from_numpy(noise).to(device)

    return noise
