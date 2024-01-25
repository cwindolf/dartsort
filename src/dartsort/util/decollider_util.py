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

from ..transform.Decollider import Decollider
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
    recording_channel_indices=None,
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
    logger = logging.getLogger("decollider:hybrid_train")

    # initial validation
    train_records = []
    val_records = []

    # allow training on recordings with different channels missing
    recording_channel_indices, channel_subsets = reconcile_channels(
        recordings, channel_index, recording_channel_indices
    )

    # combine templates on different channels using NaN padding
    # the NaNs inform masking below
    # these are also padded with an extra channel of NaNs, to help
    # with indexing below
    (
        templates_train,
        templates_train_recording_origin,
        original_train_template_index,
    ) = combine_templates(templates_train, channel_subsets)
    (
        templates_val,
        templates_val_recording_origin,
        original_val_template_index,
    ) = combine_templates(templates_val, channel_subsets)

    opt = torch.optim.Adam(net.parameters())
    criterion = loss_class()
    examples_seen = 0
    with logging_redirect_tqdm(loggers=(logger,), tqdm_class=tqdm):
        for epoch in trange(max_n_epochs):
            epoch_dt = 0.0

            # get data
            epoch_data = load_epoch(
                recordings,
                templates=templates_train,
                detection_times=detection_times_train,
                detection_channels=detection_channels_train,
                channel_index=channel_index,
                template_recording_origin=templates_train_recording_origin,
                recording_channel_indices=recording_channel_indices,
                channel_subsets=channel_subsets,
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
            val_record = evaluate_decollider(
                net,
                recordings,
                templates=templates_val,
                detection_times=detection_times_val,
                detection_channels=detection_channels_val,
                template_recording_origin=templates_val_recording_origin,
                original_template_index=original_val_template_index,
                n_oversamples=validation_oversamples,
                channel_index=channel_index,
                channel_subsets=channel_subsets,
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


def load_epoch(
    recordings,
    templates=None,
    detection_times=None,
    detection_channels=None,
    template_recording_origin=None,
    channel_index=None,
    recording_channel_indices=None,
    channel_subsets=None,
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
    rg = np.random.default_rng(data_random_seed)

    if templates is not None:
        (
            channels,
            channel_masks,
            gt_waveforms,
            waveforms,
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
            data_random_seed=data_random_seed,
            device=device,
        )
    else:
        assert detection_times is not None
        assert detection_channels is not None
        raise NotImplementedError("Unsupervised training.")

    # double noise
    noised_waveforms = load_noise(
        channels=noise_chans,
        which_rec=which_rec,
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
        template_indices=which_templates,
    )


@torch.no_grad()
def evaluate_decollider(
    net,
    recordings,
    templates=None,
    detection_times=None,
    detection_channels=None,
    template_recording_origin=None,
    original_template_index=None,
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
    val_data = load_epoch(
        recordings,
        templates=templates,
        detection_times=detection_times,
        detection_channels=detection_channels,
        template_recording_origin=template_recording_origin,
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

    # unsupervised task for learning: predict wf from noised_wf
    preds_noised = net(val_data.noised_waveforms)
    if summarize:
        val_loss = F.mse_loss(preds_noised, val_data.waveforms)

    if templates is None:
        assert summarize
        return dict(val_loss=val_loss)

    # below here, summarize is True, and we are working with templates
    # so that gt_waveforms exists

    # supervised task: predict gt_wf (template) from wf (template + noise)
    template_preds_naive = net(val_data.waveforms)
    naive_sup_max_err = (
        torch.abs(template_preds_naive - val_data.gt_waveforms).max(dim=(1, 2)).values
    )

    # noisier2noise prediction of templates
    template_preds_n2n = net.n2n_predict(
        val_data.noised_waveforms,
        channel_masks=val_data.channel_masks,
        alpha=noise2_alpha,
    )
    n2n_sup_max_err = (
        torch.abs(template_preds_naive - val_data.gt_waveforms).max(dim=(1, 2)).values
    )

    if summarize:
        naive_template_mean_max_err = naive_sup_max_err.mean()
        naive_template_mse = F.mse_loss(template_preds_naive, val_data.gt_waveforms)

        n2n_template_mean_max_err = n2n_sup_max_err.mean()
        n2n_template_mse = F.mse_loss(template_preds_n2n, val_data.gt_waveforms)

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
    noise1_norm = torch.linalg.norm(
        val_data.waveforms - val_data.gt_waveforms, dim=(1, 2)
    )
    noise2_norm = torch.linalg.norm(
        val_data.noised_waveforms - val_data.waveforms, dim=(1, 2)
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
            recording_index=template_recording_origin[val_data.template_indices],
            original_template_index=original_template_index[val_data.template_indices],
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
    device=None,
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

    # load waveforms
    if channel_index is None:
        waveform_channels = channels[:, None, None]
        # single channel case
        gt_waveforms = templates[
            which[:, None, None],
            waveform_channels,
            np.arange(templates.shape[2])[None, None, :],
        ]
        channel_masks = None
    else:
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
    channel_masks = torch.from_numpy(channel_masks)

    # to torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = torch.from_numpy(channels).to(device)
    gt_waveforms = torch.from_numpy(gt_waveforms).to(device)

    # apply noise
    noise_chans = channels if noise_same_chans else None
    which_rec = template_recording_origin[which] if noise_same_chans else None
    waveforms = load_noise(
        channels=noise_chans,
        which_rec=which_rec,
        channel_index=recording_channel_indices,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=templates.shape[1],
        n=gt_waveforms.shape[0],
        max_abs_amp=noise_max_amplitude,
        dtype=gt_waveforms.dtype,
        rg=rg,
        to_torch=True,
    )
    waveforms += gt_waveforms

    return (
        channels,
        channel_masks,
        gt_waveforms,
        waveforms,
        noise_chans,
        which_rec,
        which,
    )


# -- noise helpers


def load_noise(
    recordings,
    channels=None,
    which_rec=None,
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

    if which_rec is None:
        which_rec = rg.integers(len(recordings), size=n)

    if dtype is None:
        dtype = recordings[0].dtype

    c = channel_index[0].shape[1] if channel_index is not None else 1

    noise = np.full((n, c, spike_length_samples), np.nan, dtype=dtype)
    channel_masks = np.zeros((n, c, spike_length_samples), dtype=bool)
    for i, rec in enumerate(recordings):
        mask = np.flatnonzero(which_rec == i)
        noise[mask], channel_masks[mask] = load_noise_singlerec(
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

    # mask out nan channels
    channel_masks = np.isfinite(noise[:, :, 0])
    noise[~channel_masks] = 0.0

    if alpha != 1.0:
        noise *= alpha

    if to_torch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.from_numpy(noise).to(device), torch.from_numpy(channel_masks).to(
            device
        )

    return noise, channel_masks


def reconcile_channels(recordings, channel_index, recording_channel_indices):
    """Validate that the multi-chan setup is workable, reconcile chans across recordings"""
    if isinstance(recording_channel_indices, list):
        assert len(recording_channel_indices) == len(recordings)
    elif recording_channel_indices is None:
        if channel_index is not None:
            recording_channel_indices = [channel_index] * len(recordings)
    else:
        recording_channel_indices = [recording_channel_indices] * len(recordings)

    full_channel_set = recordings[0].channel_ids
    for rec, ci in zip(recordings, recording_channel_indices):
        assert np.array_equal(rec.channel_ids, np.sort(rec.channel_ids))
        assert rec.channel_ids.size == ci.shape[0]
        if channel_index is not None:
            assert ci.shape[1] == channel_index.shape[1]
        full_channel_set = np.union1d(full_channel_set, rec.channel_ids)
    n_channels_full = full_channel_set.size

    if channel_index is not None:
        assert n_channels_full == channel_index.shape[0]

    channel_subsets = []
    for rec in recordings:
        channel_subsets.append(
            np.flatnonzero(np.isin(full_channel_set, rec.channel_ids))
        )

    return n_channels_full, recording_channel_indices, channel_subsets


def combine_templates(templates, channel_subsets):
    t = templates[0].shape[1]
    c = channel_subsets[0].size
    n = sum(map(len, templates))

    combined = np.full((n, c + 1, t), fill_value=np.nan, dtype=templates[0].dtype)
    template_recording_origin = np.zeros(n, dtype=int)
    original_template_index = np.zeros(n, dtype=int)
    i = 0
    for r, (temps, subset) in enumerate(zip(templates, channel_subsets)):
        j = i + temps.shape[0]
        template_recording_origin[i:j] = r
        original_template_index[i:j] = np.arange(j - i)
        combined[i:j, subset] = temps

    return combined, template_recording_origin, original_template_index


# -- for testing


class SCDAsDecollider(SingleChannelDenoiser, Decollider):
    def forward(self, x):
        """N1T -> N1T"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x[:, None, :]
