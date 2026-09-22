import dataclasses
from math import isfinite
from typing import cast

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, StackDataset, TensorDataset

from ..util.internal_config import WaveformConfig
from ..util.logging_util import get_logger, progrange
from ..util.multiprocessing_util import handle_negative_jobs
from ..util.py_util import panic
from ..util.spiketorch import get_relative_index, reindex, spawn_torch_rg
from ..util.waveform_util import make_channel_index, regularize_channel_index
from ._multichan_denoiser_kit import (
    AOTIndicesWeightedRandomBatchSampler,
    AsyncSameChannelHDF5NoiseDataset,
    AsyncSameChannelRecordingNoiseDataset,
    BaseMultichannelDenoiser,
    NoneDataset,
    get_noise,
)
from .matched_filter_net import ScoreNet, signed_sqrt

logger = get_logger(__name__)


class Decollider(BaseMultichannelDenoiser):
    """Unsupervised spike waveform denoising."""

    default_name = "decollider"
    needs_residual = True

    def __init__(
        self,
        channel_index,
        geom,
        waveform_cfg,
        sampling_frequency=30_000.0,
        hidden_dims=(1024, 1024),
        norm_kind="none",
        name=None,
        name_prefix="",
        batch_size=512,
        batches_per_chunk=4,
        n_data_workers=1,
        learning_rate=2e-4,
        weight_decay=0.0,
        n_epochs=100,
        pad_depth_only=True,
        channelwise_dropout_p=0.0,
        with_conv_fullheight=False,
        svd_projection_rank: int | None = None,
        val_split_p=0.0,
        min_epochs=10,
        earlystop_eps=None,
        random_seed=0,
        res_type="none",
        lr_schedule="CosineAnnealingLR",
        lr_schedule_kwargs=None,
        inference_batch_size=1024,
        optimizer="Adam",
        optimizer_kwargs=None,
        nonlinearity="PReLU",
        scaling="max",
        signal_gates=True,
        step_callback=None,
        epoch_size=200 * 256,
        clip_value=None,
        clip_norm=None,
        warmup_epochs=0,
        warmup_lr=3e-4,
        # my args. todo: port over common ones.
        inference_z_samples=10,
        detach_amortizer=True,
        exz_estimator="n3n",
        inference_kind="amortized",
        eyz_res_type="none",
        e_exz_y_res_type="none",
        emz_res_type="none",
        l4_alpha=0,
        l1_alpha=0,
        output_l1_alpha=0.0,
        cycle_loss_alpha=1.0,
        cycle_null_alpha=0.0,
        cycle_kernel_alpha=1.0,
        separate_cycle_net=False,
        detach_cycle_loss=False,
        val_noise_random_seed=0,
        inf_net_hidden_dims=None,
        eyz_net_hidden_dims=None,
        queue_chunks=50,
        conv_fullheight_width_mult=1,
        conv_fullheight_depth=1,
        conv_fullheight_channel_mix=False,
        whiten_loss=False,
        whiten_loss_terms=(
            "eyz",
            "emz",
            "e_exz_y",
            "cycle",
            "cycle_null",
            "cycle_kernel",
        ),
        whiten_loss_temporal=True,
        whitener=None,
        score_radius_um: float | None = None,
        score_n_before=12,
        score_n_after=19,
        score_n_temporal=8,
        score_n_moments=4,
        score_spatial_mode="full",
        score_hidden_dims=(32, 32),
        score_half_square=False,
        score_energy_powers=(),
        score_loss_alpha=1.0,
        score_target_clamp=30.0,
        score_learning_rate=None,
        score_start_epoch=10,
    ):
        assert exz_estimator in ("n2n", "2n2", "n3n", "3n3")
        assert inference_kind in ("raw", "exz", "exz_fromz", "amortized", "exy_fake")

        super().__init__(
            geom=geom,
            channel_index=channel_index,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
            name=name,
            name_prefix=name_prefix,
            hidden_dims=hidden_dims,
            norm_kind=norm_kind,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            pad_depth_only=pad_depth_only,
            channelwise_dropout_p=channelwise_dropout_p,
            with_conv_fullheight=with_conv_fullheight,
            svd_projection_rank=svd_projection_rank,
            val_split_p=val_split_p,
            min_epochs=min_epochs,
            earlystop_eps=earlystop_eps,
            random_seed=random_seed,
            res_type=res_type,
            lr_schedule=lr_schedule,
            lr_schedule_kwargs=lr_schedule_kwargs,
            inference_batch_size=inference_batch_size,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            nonlinearity=nonlinearity,
            scaling=scaling,
            signal_gates=signal_gates,
            step_callback=step_callback,
            epoch_size=epoch_size,
            warmup_epochs=warmup_epochs,
            warmup_lr=warmup_lr,
            conv_fullheight_width_mult=conv_fullheight_width_mult,
            conv_fullheight_depth=conv_fullheight_depth,
            conv_fullheight_channel_mix=conv_fullheight_channel_mix,
        )
        self.queue_chunks = queue_chunks
        self.batches_per_chunk = batches_per_chunk
        self.n_data_workers = n_data_workers

        self.inference_z_samples = inference_z_samples
        self.detach_amortizer = detach_amortizer
        self.exz_estimator = exz_estimator
        self.inference_kind = inference_kind
        self.eyz_res_type = eyz_res_type
        self.e_exz_y_res_type = e_exz_y_res_type
        self.emz_res_type = emz_res_type
        self.val_noise_random_seed = val_noise_random_seed
        self.inf_net_hidden_dims = inf_net_hidden_dims
        self.eyz_net_hidden_dims = eyz_net_hidden_dims
        self.l1_alpha = l1_alpha
        self.l4_alpha = l4_alpha
        self.output_l1_alpha = output_l1_alpha
        self.cycle_loss_alpha = cycle_loss_alpha
        self.cycle_null_alpha = cycle_null_alpha
        self.cycle_kernel_alpha = cycle_kernel_alpha
        self.separate_cycle_net = separate_cycle_net
        self.detach_cycle_loss = detach_cycle_loss
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.whiten_loss = whiten_loss
        self.whiten_loss_terms = tuple(whiten_loss_terms)
        self.whiten_loss_temporal = whiten_loss_temporal
        self.score_radius_um = score_radius_um
        self.score_n_before = score_n_before
        self.score_n_after = score_n_after
        self.score_n_temporal = score_n_temporal
        self.score_n_moments = score_n_moments
        self.score_spatial_mode = score_spatial_mode
        self.score_hidden_dims = tuple(score_hidden_dims)
        self.score_half_square = score_half_square
        self.score_energy_powers = tuple(score_energy_powers)
        self.score_loss_alpha = score_loss_alpha
        self.score_target_clamp = score_target_clamp
        self.score_learning_rate = score_learning_rate
        self.score_start_epoch = score_start_epoch
        # weak reference, only used in training
        self._whitener_holder = [whitener]

        if self.score_radius_um:
            assert self.score_radius_um is not None
            score_channel_index = regularize_channel_index(
                geom=self.geom,
                channel_index=make_channel_index(self.b.geom, self.score_radius_um),
                depth_only=pad_depth_only,
            )
            score_channel_index = torch.from_numpy(score_channel_index)
            self.register_buffer("score_channel_index", score_channel_index)
            self.register_buffer(
                "score_relative_index",
                get_relative_index(self.b.model_channel_index, score_channel_index),
            )
            valid = score_channel_index < self.n_channels
            found = self.b.score_relative_index < self.b.model_channel_index.shape[1]
            if not torch.equal(valid, found):
                panic("score_channel_index not contained in model_channel_index")
        if self.svd_projection_rank:
            self.submodule_names = ["tpca"]

        if separate_cycle_net:
            assert cycle_loss_alpha > 0

    def initialize_spike_length_dependent_params(self):
        if hasattr(self, "inf_net"):
            logger.dartsortdebug("Already initialized.")
            return
        self.initialize_shapes()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_seed)
            if self.exz_estimator in ("n2n", "n3n"):
                self.eyz: torch.nn.Module = self.get_mlp(
                    res_type=self.eyz_res_type,
                    hidden_dims=self.eyz_net_hidden_dims,
                    message="eyz",
                )
            if self.exz_estimator in ("n3n", "2n2", "3n3"):
                self.emz: torch.nn.Module = self.get_mlp(
                    res_type=self.emz_res_type, output_layer="linear", message="emz"
                )
            if self.inference_kind == "amortized":
                self.inf_net: torch.nn.Module = self.get_mlp(
                    res_type=self.e_exz_y_res_type,
                    hidden_dims=self.inf_net_hidden_dims,
                    message="inf",
                )
            if self.separate_cycle_net:
                self.den_net: torch.nn.Module = self.get_mlp(
                    res_type=self.e_exz_y_res_type,
                    hidden_dims=self.inf_net_hidden_dims,
                    message="den",
                )
            else:
                self.den_net: torch.nn.Module = self.inf_net
        if self.svd_projection_rank:
            from .temporal_pca import BaseTemporalPCA

            self.tpca = BaseTemporalPCA(
                self.b.channel_index,
                geom=self.b.geom,
                waveform_cfg=cast(WaveformConfig, self.waveform_cfg),
                rank=self.svd_projection_rank,
            )
            self.tpca.spike_length_samples = self.spike_length_samples
            self.tpca.initialize_spike_length_dependent_params()
        else:
            self.tpca = None
        if self.score_radius_um:
            self.score_net: ScoreNet | None = ScoreNet(
                n_before=self.score_n_before,
                n_after=self.score_n_after,
                n_score_channels=self.b.score_channel_index.shape[1],
                n_temporal=self.score_n_temporal,
                n_moments=self.score_n_moments,
                spatial_mode=self.score_spatial_mode,
                hidden_dims=self.score_hidden_dims,
                half_square=self.score_half_square,
                energy_powers=self.score_energy_powers,
                random_seed=self.random_seed,
            )
        else:
            self.score_net = None
        self.initialize_whiteners()
        self.to(self.device)

    def _other_pre_load_state(self, state_dict, prefix):
        if not self.score_radius_um:
            return
        if getattr(self, "score_net", None) is None:
            self.initialize_spike_length_dependent_params()
        assert self.score_net is not None
        self.score_net.attach_persistent_buffers(state_dict, f"{prefix}score_net.")

    def needs_whiteners(self):
        return self.whiten_loss or bool(self.score_radius_um)

    def initialize_whiteners(self):
        whitener = self._whitener_holder[0]
        if not self.needs_whiteners() or whitener is None:
            return

        if not hasattr(self, "loss_local_whiteners"):
            lw = whitener.local_whiteners(self.b.model_channel_index.cpu())
            self.register_buffer(
                "loss_local_whiteners", lw.to(self.device), persistent=False
            )
            kernel = whitener.b.temporal_kernel
            if kernel is not None:
                kernel = kernel.clone().to(self.device)
            self.register_buffer_or_none(
                "gate_whitening_kernel", kernel, persistent=False
            )
            self.register_buffer_or_none(
                "loss_whitening_kernel",
                kernel if self.whiten_loss_temporal else None,
                persistent=False,
            )

        if self.score_radius_um and not hasattr(self, "score_local_whiteners"):
            sw = whitener.local_whiteners(self.b.score_channel_index.cpu())
            self.register_buffer(
                "score_local_whiteners", sw.to(self.device), persistent=False
            )

        if getattr(self, "score_net", None) is not None:
            assert self.score_net is not None
            self.score_net.set_whiteners(
                self.b.score_local_whiteners, self.b.gate_whitening_kernel
            )

    def fit(
        self,
        recording,
        waveforms,
        *,
        computation_cfg,
        channels,
        hdf5_filename=None,
        **spike_data,
    ):
        weights = spike_data.get("weights")
        if self.needs_whiteners() and self._whitener_holder[0] is None:
            self._whitener_holder[0] = self._estimate_loss_whitener(hdf5_filename)
        super().fit(
            recording, waveforms, computation_cfg=computation_cfg, channels=channels
        )
        self.initialize_whiteners()
        if self.tpca is not None and self.tpca.needs_fit():
            self.tpca.fit(
                recording=recording,
                waveforms=waveforms,
                computation_cfg=computation_cfg,
                channels=channels,
            )
        train_data, val_data = self._construct_datasets_from_waveforms(
            waveforms,
            channels,
            recording,
            weights,
            hdf5_filename=hdf5_filename,
            device=computation_cfg.actual_device(),
        )
        with torch.enable_grad():
            if self.tpca is not None:
                self.tpca.eval()
            res = self._fit(train_data, val_data)
        self._needs_fit = False
        if self.score_net is not None:
            self.score_net.eval()
            self.score_net.bake()
        return res

    def _estimate_loss_whitener(self, hdf5_filename):
        from ..util.data_util import DARTsortSorting
        from ..util.internal_config import WhiteningConfig
        from ..util.noise_util import Whitener

        if hdf5_filename is None or not _check_has_dataset(hdf5_filename, "residual"):
            panic(f"No residual in {hdf5_filename}")
        logger.dartsortdebug(f"Estimate loss whitener from {hdf5_filename}.")
        return Whitener.from_config(
            sorting=DARTsortSorting.from_peeling_hdf5(hdf5_filename),
            motion=None,
            whiten_cfg=WhiteningConfig(strategy="prewhiten_postapply"),
        )

    def _whiten(self, d, channels, kernel):
        w = self.b.loss_local_whiteners[channels].to(d)
        dw = w.bmm(d.nan_to_num().mT)
        if kernel is not None:
            n, c, t = dw.shape
            k = kernel[None, None].to(dw)
            dw = F.conv1d(dw.reshape(n * c, 1, t), k, padding="same").view(n, c, t)
        return dw

    def whiten_for_loss(self, d, channels):
        return self._whiten(d, channels, self.b.loss_whitening_kernel)

    def whiten_for_detection_score(self, d, channels):
        return self._whiten(d, channels, self.b.gate_whitening_kernel)

    def denoiser_detection_score(self, waveforms, denoised, channels):
        x = self.whiten_for_detection_score(waveforms, channels).reshape(
            len(waveforms), -1
        )
        xhat = self.whiten_for_detection_score(denoised, channels).reshape(
            len(denoised), -1
        )
        reduction = 2.0 * (x * xhat).sum(1) - xhat.square().sum(1)
        return signed_sqrt(reduction)

    def score_input(self, waveforms, channels):
        """Rework waveforms onto the score_net's input window"""
        assert self.score_net is not None
        assert self.trough_offset_samples is not None
        net = self.score_net
        start = self.trough_offset_samples - net.trough_offset
        assert start >= 0
        stop = start + net.receptive_field
        assert stop <= waveforms.shape[1]
        wfs = reindex(
            channels, waveforms[:, start:stop], self.b.score_relative_index, 0.0
        )
        mask = self.b.score_channel_index[channels] < self.n_channels
        return wfs.nan_to_num().mT, mask

    def score_forward(self, waveforms, channels):
        assert self.score_net is not None
        wfs, mask = self.score_input(waveforms, channels)
        return self.score_net(wfs, channels, mask).squeeze(1)

    def masked_mse(self, a, b, mask, term, channels):
        d = (a - b).mul(mask)
        if self.whiten_loss and channels is not None and term in self.whiten_loss_terms:
            d = self.whiten_for_loss(d, channels)
        return d.square().mean()

    def forward_unbatched(self, waveforms, channels):
        """Called only at inference time."""
        if self.tpca is not None:
            waveforms = self.tpca.force_embed(waveforms)
        waveforms, masks = self.to_nn_channels(waveforms, channels)
        net_input = waveforms, masks.unsqueeze(1)

        if self.inference_kind == "amortized":
            pred = self.den_net(net_input)
        elif self.inference_kind == "raw":
            if hasattr(self, "emz"):
                emz = self.emz(net_input)
                pred = waveforms - emz
            elif hasattr(self, "eyz"):
                pred = self.eyz(net_input)
            else:
                panic()
        elif self.inference_kind == "exz_fromz":
            pred = torch.zeros_like(waveforms)
            for _ in range(self.inference_z_samples):
                m = get_noise(
                    self.recording,
                    channels.numpy(force=True),
                    self.b.model_channel_index.numpy(force=True),
                    spike_length_samples=cast(int, self.spike_length_samples),
                    rg=None,
                )
                m = m.to(waveforms)
                z = waveforms + m
                net_input = z, masks.unsqueeze(1)
                if self.exz_estimator == "n2n":
                    eyz = self.eyz(net_input)
                    pred += 2 * eyz - z
                elif self.exz_estimator == "2n2":
                    emz = self.emz(net_input)
                    pred += z - 2 * emz
                elif self.exz_estimator == "n3n":
                    eyz = self.eyz(net_input)
                    emz = self.emz(net_input)
                    pred += eyz - emz
                elif self.exz_estimator == "3n3":
                    emz = self.emz(net_input)
                    pred += z - emz
                else:
                    panic(self.exz_estimator)
            pred /= self.inference_z_samples
        elif self.inference_kind == "exz":
            if self.exz_estimator == "n2n":
                eyz = self.eyz(net_input)
                pred = 2 * eyz - waveforms
            elif self.exz_estimator == "2n2":
                emz = self.emz(net_input)
                pred = waveforms - 2 * emz
            elif self.exz_estimator == "n3n":
                eyz = self.eyz(net_input)
                emz = self.emz(net_input)
                pred = eyz - emz
            elif self.exz_estimator == "3n3":
                emz = self.emz(net_input)
                pred = waveforms - emz
            else:
                panic(self.exz_estimator)
        else:
            panic(self.inference_kind)

        pred = self.to_orig_channels(pred, channels)

        if self.tpca is not None:
            pred = self.tpca.force_reconstruct(pred)

        return pred

    def train_forward(self, y, m, ell, mask, channels=None, with_score=True):
        z = y + m

        # predictions given z
        # TODO: variance given z and put it in the loss
        exz = eyz = emz = e_exz_y = cycle_output = cycle_null_output = None
        cycle_kernel_output = None
        net_input = z, mask.unsqueeze(1)
        if self.exz_estimator == "n2n":
            eyz = self.eyz(net_input)
            exz = 2 * eyz - z
        elif self.exz_estimator == "2n2":
            emz = self.emz(net_input)
            exz = z - 2 * emz
        elif self.exz_estimator == "n3n":
            eyz = self.eyz(net_input)
            emz = self.emz(net_input)
            exz = eyz - emz
        elif self.exz_estimator == "3n3":
            emz = self.emz(net_input)
            exz = y - emz
        else:
            panic(self.exz_estimator)

        # predictions given y, if relevant
        if self.inference_kind == "amortized":
            e_exz_y = self.inf_net((y, mask.unsqueeze(1)))

        if self.cycle_loss_alpha:
            assert e_exz_y is not None
            cycle_targ = e_exz_y.detach() if self.detach_cycle_loss else e_exz_y
            cycle_input = cycle_targ + ell
            cycle_output = self.den_net((cycle_input, mask.unsqueeze(1)))

        if self.cycle_null_alpha:
            assert ell is not None
            cycle_null_output = self.den_net((ell, mask.unsqueeze(1)))

        if self.cycle_kernel_alpha:
            assert e_exz_y is not None
            kernel_base = e_exz_y.detach() if self.detach_cycle_loss else e_exz_y
            cycle_kernel_output = self.den_net((y - kernel_base, mask.unsqueeze(1)))

        score_pred = score_target = None
        if with_score and self.score_net is not None and channels is not None:
            score_pred, score_target = self.score_train_forward(
                y, ell, mask, channels, e_exz_y, cycle_null_output
            )

        return dict(
            exz=exz,
            eyz=eyz,
            emz=emz,
            e_exz_y=e_exz_y,
            cycle_output=cycle_output,
            cycle_null_output=cycle_null_output,
            cycle_kernel_output=cycle_kernel_output,
            score_pred=score_pred,
            score_target=score_target,
        )

    def score_train_forward(self, y, ell, mask, channels, e_exz_y, cycle_null_output):
        assert e_exz_y is not None
        preds = [self.score_forward(y, channels)]
        targets = [self.denoiser_detection_score(y, e_exz_y.detach(), channels)]

        if ell is not None:
            null_output = cycle_null_output
            if null_output is None:
                null_output = self.den_net((ell, mask.unsqueeze(1)))
            preds.append(self.score_forward(ell, channels))
            targets.append(
                self.denoiser_detection_score(ell, null_output.detach(), channels)
            )

        return torch.cat(preds), torch.cat(targets).detach()

    def loss(
        self,
        mask,
        waveforms,
        m,
        net_outputs,
        l1_alpha=None,
        l4_alpha=None,
        output_l1_alpha=None,
        channels=None,
    ):
        loss_dict = {}
        mask = mask.unsqueeze(1)

        exz = net_outputs["exz"]
        eyz = net_outputs["eyz"]
        emz = net_outputs["emz"]
        e_exz_y = net_outputs["e_exz_y"]
        cycle_output = net_outputs["cycle_output"]

        if eyz is not None:
            eyz_mask = mask * eyz
            loss_dict["eyz"] = self.masked_mse(eyz, waveforms, mask, "eyz", channels)
            if l1_alpha:
                loss_dict["eyz_l1"] = (
                    l4_alpha * (eyz - waveforms).mul_(mask).abs_().mean()
                )
            if l4_alpha:
                loss_dict["eyz_l4"] = (
                    l4_alpha * ((eyz - waveforms).mul_(mask) ** 4).mean()
                )
            if output_l1_alpha:
                loss_dict["eyz_ol1"] = output_l1_alpha * eyz_mask.abs().mean()
        if emz is not None:
            loss_dict["emz"] = self.masked_mse(emz, m, mask, "emz", channels)
            if l1_alpha:
                loss_dict["emz_l1"] = l4_alpha * (emz - m).mul_(mask).abs_().mean()
            if l4_alpha:
                loss_dict["emz_l4"] = l4_alpha * ((emz - m).mul_(mask) ** 4).mean()
        if e_exz_y is not None:
            to_amortize = exz
            if self.detach_amortizer:
                # should amortize-ability affect the learning of eyz, emz?
                to_amortize = to_amortize.detach()
            am_mask = mask * to_amortize
            loss_dict["e_exz_y"] = self.masked_mse(
                e_exz_y, to_amortize, mask, "e_exz_y", channels
            )
            if l1_alpha:
                loss_dict["e_exz_y_l1"] = (
                    l4_alpha * (to_amortize - e_exz_y).mul_(mask).abs_().mean()
                )
            if l4_alpha:
                loss_dict["e_exz_y_l4"] = (
                    l4_alpha * ((to_amortize - e_exz_y).mul_(mask) ** 4).mean()
                )
            if output_l1_alpha:
                loss_dict["e_exz_y_ol1"] = output_l1_alpha * am_mask.abs().mean()
        cycle_kernel_output = net_outputs.get("cycle_kernel_output")
        if cycle_kernel_output is not None:
            loss_dict["cycle_kernel"] = self.cycle_kernel_alpha * self.masked_mse(
                cycle_kernel_output,
                torch.zeros_like(cycle_kernel_output),
                mask,
                "cycle_kernel",
                channels,
            )
        cycle_null_output = net_outputs.get("cycle_null_output")
        if cycle_null_output is not None:
            loss_dict["cycle_null"] = self.cycle_null_alpha * self.masked_mse(
                cycle_null_output,
                torch.zeros_like(cycle_null_output),
                mask,
                "cycle_null",
                channels,
            )
        if cycle_output is not None:
            coef = 1 if self.separate_cycle_net else self.cycle_loss_alpha
            cycle_targ = e_exz_y.detach() if self.detach_cycle_loss else e_exz_y
            loss_dict["cycle"] = coef * self.masked_mse(
                cycle_output, cycle_targ, mask, "cycle", channels
            )
            if l1_alpha:
                loss_dict["cycle_l1"] = (coef * l1_alpha) * (
                    (cycle_targ - cycle_output).mul_(mask).abs_().mean()
                )
        score_pred = net_outputs.get("score_pred")
        if score_pred is not None:
            clamp = self.score_target_clamp
            target = net_outputs["score_target"]
            if clamp:
                out = target.abs() > clamp
                score_pred = torch.where(
                    out, score_pred.clamp(-clamp, clamp), score_pred
                )
                target = target.clamp(-clamp, clamp)
            loss_dict["score"] = self.score_loss_alpha * F.mse_loss(score_pred, target)
        return loss_dict

    def clip_parameter_groups(self):
        return [g for g in self.parameter_groups() if g]

    def parameter_groups(self):
        """(denoiser parameters, score net parameters)."""
        if self.score_net is None:
            return list(self.parameters()), []
        score_params = set(map(id, self.score_net.parameters()))
        rest = [p for p in self.parameters() if id(p) not in score_params]
        return rest, list(self.score_net.parameters())

    def get_optimizer(self, params=None, lr=None):
        if params is None and self.score_net is not None:
            params = self.parameter_groups()[0]
        return super().get_optimizer(params=params, lr=lr)

    def get_score_optimizer(self):
        if self.score_net is None:
            return None
        lr = (
            self.learning_rate
            if self.score_learning_rate is None
            else (self.score_learning_rate)
        )
        return super().get_optimizer(params=self.parameter_groups()[1], lr=lr)

    def _fit(
        self,
        train_data: "DecolliderDataLoader",
        val_data: "DecolliderDataLoader | None",
    ):
        try:
            train_records = self._run_train_loop(train_data, val_data)
        finally:
            train_data.destroy()
            if val_data is not None:
                val_data.destroy()

        train_df = pd.DataFrame.from_records(train_records)
        return train_df

    def _construct_datasets_from_waveforms(
        self,
        waveforms,
        channels,
        recording,
        weights=None,
        hdf5_filename=None,
        dataset_name="residual",
        device="cpu",
    ):
        rg = np.random.default_rng(self.random_seed)
        device = torch.device(device)
        pin_memory = device.type == "cuda"

        _, n_data_workers = handle_negative_jobs(self.n_data_workers)

        val_size = 0
        train_indices = slice(None)
        val_indices = None
        if self.val_split_p:
            num_samples = len(waveforms)
            val_size = int(self.val_split_p * num_samples)
            train_size = num_samples - val_size
            train_indices = rg.choice(num_samples, size=train_size, replace=False)
            val_indices = np.setdiff1d(np.arange(num_samples), train_indices)

        can_load_h5 = _check_has_dataset(hdf5_filename, dataset_name)
        needs_cycle_noise = bool(self.cycle_loss_alpha or self.cycle_null_alpha)

        spike_length_samples = waveforms.shape[1]

        # training dataset
        train_waveforms = waveforms[train_indices]
        train_channels = channels[train_indices]
        train_dataset = TensorDataset(train_waveforms, train_channels)
        if can_load_h5:
            logger.dartsortdebug(
                f"Load Decollider train noise data from {hdf5_filename}."
            )
            train_noise_dataset = AsyncSameChannelHDF5NoiseDataset(
                hdf5_filename=hdf5_filename,
                channels=train_channels.numpy(force=True),
                channel_index=self.b.model_channel_index.numpy(force=True),
                spike_length_samples=spike_length_samples,
                rg=np.random.default_rng(rg.spawn(1)[0]),
                queue_chunks=self.queue_chunks,
                dataset_name=dataset_name,
                chunk_size=self.batches_per_chunk * self.batch_size,
                n_workers=n_data_workers,
                pin_memory=pin_memory,
            )
        else:
            logger.dartsortdebug("Load Decollider train noise data from recording.")
            train_noise_dataset = AsyncSameChannelRecordingNoiseDataset(
                recording,
                train_channels.numpy(force=True),
                self.b.model_channel_index.numpy(force=True),
                spike_length_samples=spike_length_samples,
                generator=spawn_torch_rg(rg),
                queue_chunks=self.queue_chunks,
                n_workers=n_data_workers,
                pin_memory=pin_memory,
            )
        if needs_cycle_noise and can_load_h5:
            logger.dartsortdebug(
                f"Load Decollider cycle noise data from {hdf5_filename}."
            )
            train_cycle_noise_dataset = AsyncSameChannelHDF5NoiseDataset(
                hdf5_filename=hdf5_filename,
                channels=train_channels.numpy(force=True),
                channel_index=self.b.model_channel_index.numpy(force=True),
                spike_length_samples=spike_length_samples,
                rg=np.random.default_rng(rg.spawn(1)[0]),
                queue_chunks=self.queue_chunks,
                dataset_name=dataset_name,
                chunk_size=self.batch_size,
                n_workers=n_data_workers,
                pin_memory=pin_memory,
            )
        elif needs_cycle_noise:
            logger.dartsortdebug("Load Decollider cycle noise data from recording.")
            train_cycle_noise_dataset = AsyncSameChannelRecordingNoiseDataset(
                recording,
                train_channels.numpy(force=True),
                self.b.model_channel_index.numpy(force=True),
                spike_length_samples=spike_length_samples,
                generator=spawn_torch_rg(rg),
                queue_chunks=self.queue_chunks,
                n_workers=n_data_workers,
                pin_memory=pin_memory,
            )
        else:
            train_cycle_noise_dataset = NoneDataset(len(train_channels))

        train_stack_dataset = StackDataset(
            train_dataset, train_noise_dataset, train_cycle_noise_dataset
        )
        train_weights = None if weights is None else weights[train_indices]
        train_sampler = AOTIndicesWeightedRandomBatchSampler(
            n_examples=len(train_channels),
            weights=train_weights,
            replacement=train_weights is not None,
            batch_size=self.batch_size,
            generator=spawn_torch_rg(rg),
            epoch_size=self.epoch_size,
        )
        train_loader = DataLoader(
            train_stack_dataset,
            sampler=train_sampler,
            num_workers=0,
            batch_size=None,
            pin_memory=pin_memory,
        )
        train_data = DecolliderDataLoader(
            loader=train_loader,
            sampler=train_sampler,
            noise_dataset=train_noise_dataset,
            cycle_noise_dataset=train_cycle_noise_dataset,
            spike_length_samples=spike_length_samples,
        )

        # initialize validation datasets only if val_split_p > 0
        if val_size > 0:
            val_waveforms = waveforms[val_indices]
            val_channels = channels[val_indices]
            val_noise = get_noise(
                recording,
                val_channels.numpy(force=True),
                self.b.model_channel_index.numpy(force=True),
                spike_length_samples=spike_length_samples,
                rg=rg,
            )
            if needs_cycle_noise:
                cycle_val_noise = get_noise(
                    recording,
                    val_channels.numpy(force=True),
                    self.b.model_channel_index.numpy(force=True),
                    spike_length_samples=spike_length_samples,
                    rg=rg,
                )
                cycle_val_noise = TensorDataset(cycle_val_noise)
            else:
                cycle_val_noise = NoneDataset(len(train_channels))
            val_dataset = TensorDataset(val_waveforms, val_channels)
            val_noise_dataset = TensorDataset(val_noise)

            # val set does not need shuffling
            val_loader = DataLoader(
                val_dataset,
                num_workers=n_data_workers,  # type: ignore  # ty: ignore[x]
                persistent_workers=bool(n_data_workers),
                batch_size=self.batch_size,
            )
            val_data = DecolliderDataLoader(
                loader=val_loader,
                sampler=None,
                noise_dataset=val_noise_dataset,
                cycle_noise_dataset=cycle_val_noise,
                spike_length_samples=spike_length_samples,
            )
        else:
            val_data = None

        return train_data, val_data

    def _run_train_loop(self, train_data, val_data):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        score_optimizer = self.get_score_optimizer()

        loss = 0.0
        last_val_loss = None
        train_records = []

        with progrange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                score_active = epoch >= self.score_start_epoch

                # deal with random indices...
                train_data.refresh()

                # Training phase
                self.train()
                train_losses = {}
                for waveform_b, channels_b, noise_b, cnoise_b in train_data:
                    waveform_b = waveform_b.to(device=self.device, non_blocking=True)
                    m = noise_b.to(
                        dtype=waveform_b.dtype, device=self.device, non_blocking=True
                    )
                    if cnoise_b is not None:
                        ell = cnoise_b.to(
                            dtype=waveform_b.dtype,
                            device=self.device,
                            non_blocking=True,
                        )
                    else:
                        ell = None

                    if self.tpca is not None:
                        with torch.no_grad():
                            waveform_b = self.tpca.force_embed(waveform_b)
                            m = self.tpca.force_embed(m)
                            if ell is not None:
                                ell = self.tpca.force_embed(ell)

                    channels_b = channels_b.to(device=self.device, non_blocking=True)
                    waveform_b = reindex(
                        channels_b,
                        waveform_b,
                        self.relative_index,
                        pad_value=0.0,
                    )

                    optimizer.zero_grad()
                    if score_optimizer is not None:
                        score_optimizer.zero_grad()

                    mask = self.get_masks(channels_b).to(
                        dtype=waveform_b.dtype, device=self.device, non_blocking=True
                    )
                    fres = self.train_forward(
                        waveform_b, m, ell, mask, channels_b, with_score=score_active
                    )
                    loss_dict = self.loss(
                        mask,
                        waveform_b,
                        m,
                        fres,
                        l1_alpha=self.l1_alpha,
                        l4_alpha=self.l4_alpha,
                        output_l1_alpha=self.output_l1_alpha,
                        channels=channels_b,
                    )
                    loss = sum(loss_dict.values())
                    loss.backward()

                    if self.clip_value is not None:
                        torch.nn.utils.clip_grad_value_(
                            self.parameters(), self.clip_value
                        )
                    if self.clip_norm is not None:
                        for group in self.clip_parameter_groups():
                            torch.nn.utils.clip_grad_norm_(group, self.clip_norm)
                    optimizer.step()
                    if score_optimizer is not None and score_active:
                        score_optimizer.step()

                    for k, v in loss_dict.items():
                        train_losses[k] = v + train_losses.get(k, 0.0)
                # // epoch loop
                train_data.cleanup()
                train_losses = {
                    k: v.item() / len(train_data) for k, v in train_losses.items()
                }
                if not all(isfinite(v) for v in train_losses.values()):
                    raise ValueError(f"Denoiser training diverged: {train_losses}.")
                train_records.append({**train_losses})

                # Validation phase (only if val_loader is not None)
                val_losses = {}
                val_loss = None
                if val_data is not None:
                    self.eval()
                    val_losses = {}
                    with torch.no_grad():
                        for waveform_b, channels_b, noise_b, ell_b in val_data:
                            waveform_b = waveform_b.to(self.device)
                            channels_b = channels_b.to(self.device)
                            noise_b = noise_b.to(self.device)
                            ell_b = None if ell_b is None else ell_b.to(self.device)
                            if self.tpca is not None:
                                with torch.no_grad():
                                    waveform_b = self.tpca.force_embed(waveform_b)
                                    noise_b = self.tpca.force_embed(noise_b)
                                    if ell_b is not None:
                                        ell_b = self.tpca.force_embed(ell_b)

                            waveform_b, mask = self.to_nn_channels(
                                waveform_b, channels_b
                            )
                            m = noise_b.to(waveform_b)
                            fres = self.train_forward(
                                waveform_b, m, ell_b, mask, channels_b
                            )
                            loss_dict = self.loss(
                                mask,
                                waveform_b,
                                m,
                                fres,
                                l1_alpha=self.l1_alpha,
                                l4_alpha=self.l4_alpha,
                                output_l1_alpha=self.output_l1_alpha,
                                channels=channels_b,
                            )
                            for k, v in loss_dict.items():
                                val_losses[k] = v + val_losses.get(k, 0.0)

                    val_losses = {
                        k: v.item() / len(val_data) for k, v in val_losses.items()
                    }
                    val_loss = sum(val_losses.values())
                    train_records[-1]["val_loss"] = val_loss

                    can_val = not (self.earlystop_eps is None or last_val_loss is None)
                    if can_val and val_loss - last_val_loss > self.earlystop_eps:
                        if epoch >= self.min_epochs:
                            logger.dartsortdebug(
                                f"Early stopping after {epoch} epochs."
                            )
                            break
                    last_val_loss = val_loss

                if self.step_callback is not None:
                    self.step_callback(self, epoch, val_loss)

                # Print loss summary
                loss_str = f"Train {loss:.4f} " + "|".join(
                    f"{k}: {v:.3f}" for k, v in train_losses.items()
                )
                if val_data is not None:
                    loss_str += f" Val {val_loss:.4f}" + "|".join(
                        f"{k}: {v:.3f}" for k, v in val_losses.items()
                    )
                pbar.set_description(f"Epochs [{loss_str}]")

                self.step_scheduler(scheduler, loss, val_loss)
        return train_records


@dataclasses.dataclass(kw_only=True, frozen=True)
class DecolliderDataLoader:
    loader: DataLoader
    sampler: AOTIndicesWeightedRandomBatchSampler | None
    noise_dataset: (
        AsyncSameChannelRecordingNoiseDataset
        | AsyncSameChannelHDF5NoiseDataset
        | TensorDataset
    )
    cycle_noise_dataset: (
        AsyncSameChannelRecordingNoiseDataset
        | AsyncSameChannelHDF5NoiseDataset
        | TensorDataset
        | NoneDataset
    )
    spike_length_samples: int

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        """Flatten..."""
        for batch in self.loader:
            batch_flat = []
            for item in batch:
                if isinstance(item, (list, tuple)):
                    batch_flat.extend(item)
                else:
                    batch_flat.append(item)
            yield batch_flat

    def refresh(self):
        if hasattr(self.sampler, "refresh"):
            self.sampler.refresh()
            if hasattr(self.noise_dataset, "refresh"):
                self.noise_dataset.refresh(self.sampler.indices)  # type: ignore
            if self.cycle_noise_dataset is not None and hasattr(
                self.cycle_noise_dataset, "refresh"
            ):
                self.cycle_noise_dataset.refresh(self.sampler.indices)  # type: ignore
        else:
            assert not hasattr(self.noise_dataset, "refresh")

    def cleanup(self):
        if hasattr(self.noise_dataset, "cleanup"):
            self.noise_dataset.cleanup()  # type: ignore
        if self.cycle_noise_dataset is not None and hasattr(
            self.cycle_noise_dataset, "cleanup"
        ):
            self.cycle_noise_dataset.cleanup()  # type: ignore

    def destroy(self):
        if hasattr(self.noise_dataset, "destroy"):
            self.noise_dataset.destroy()  # type: ignore
        if self.cycle_noise_dataset is not None and hasattr(
            self.cycle_noise_dataset, "destroy"
        ):
            self.cycle_noise_dataset.destroy()  # type: ignore


def _check_has_dataset(h5, dset):
    if h5 is None:
        return
    with h5py.File(h5, "r", locking=False) as h5h:
        return dset in h5h
