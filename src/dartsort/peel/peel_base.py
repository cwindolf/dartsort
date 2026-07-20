"""Home of the BasePeeler, where shared logic for other modules here lives."""

import gc
import tempfile
from concurrent.futures import CancelledError
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path
from sys import getrefcount
from threading import Lock, local
from typing import Any, Sequence, TypedDict
from warnings import catch_warnings, filterwarnings

import h5py
import numpy as np
import torch
from spikeinterface.core import BaseRecording
from spikeinterface.core.recording_tools import get_chunk_with_margin
from sympy import divisors

from ..transform import WaveformPipeline
from ..util.data_util import (
    SpikeDataset,
    divide_randomly,
    extract_random_snips,
    subsample_waveforms,
)
from ..util.internal_config import (
    FitSamplingConfig,
    WaveformConfig,
    default_peeling_fit_sampling_cfg,
    default_waveform_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger, progbar
from ..util.multiprocessing_util import handle_negative_jobs, pool_from_cfg
from ..util.py_util import delay_keyboard_interrupt
from ..util.torch_util import BModule, cleanup_and_log_gpu_usage

logger = get_logger(__name__)
_lock = Lock()


class BasePeeler(BModule):
    """Base class for peeling operations (subtraction, deconv, etc)

    Subtraction, template matching, and other things like just grabbing waveforms
    and featurizing a preexisting spike train (implemented with optional additional
    peeling of preexisting templates in grab.py) share a lot of logic: reading
    chunks of data in parallel, extracting spike waveform snippets, denoising,
    featurizing, .... This class implements the common logic.

    Instances of peeler subclasses are usually instantiated with their .from_config()
    alternate constructors and run using peel_util.run_peeler.

    Results of peeling operations are DARTsortSorting objects. Peelers save outputs in
    an HDF5 format with lots of datasets. Most of these datasets will have the same length
    (.shape[0]), which is the number of spikes. Those datasets are loaded up and dealt
    with via the DARTsortSorting.
    """

    peel_kind = ""

    def __init__(
        self,
        recording: BaseRecording,
        channel_index: np.ndarray | torch.Tensor,
        featurization_pipeline: WaveformPipeline | None = None,
        chunk_length_samples: int = 30_000,
        chunk_margin_samples: int = 0,
        fit_sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        fixed_property_keys: Sequence[str] = ("channels", "times_seconds"),
        dtype=torch.float,
    ):
        if recording.get_num_segments() > 1:
            raise ValueError("Peeling does not yet support multi-segment recordings.")
        super().__init__()
        self.recording = recording
        self.chunk_length_samples: int = chunk_length_samples
        self.chunk_margin_samples: int = chunk_margin_samples
        self.waveform_cfg = waveform_cfg
        self.trough_offset_samples: int = waveform_cfg.trough_offset_samples(
            recording.sampling_frequency
        )
        self.spike_length_samples: int = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        if fit_sampling_cfg.residual_snip_ms:
            self.resid_length_samples = waveform_cfg.ms_to_samples(
                fit_sampling_cfg.residual_snip_ms,
                sampling_frequency=recording.sampling_frequency,
            )
        else:
            self.resid_length_samples = self.spike_length_samples
        self.fit_sampling_cfg = fit_sampling_cfg
        self.random_seed = fit_sampling_cfg.seed
        self.fit_subsampling_random_state: np.random.Generator = np.random.default_rng(
            fit_sampling_cfg.seed
        )
        self.dtype: torch.dtype = dtype
        self.np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        if channel_index is not None:
            channel_index = torch.asarray(channel_index, copy=True).contiguous()
            self.register_buffer("channel_index", channel_index)
            assert recording.get_num_channels() == channel_index.shape[0]
        self.featurization_pipeline: WaveformPipeline | None = featurization_pipeline
        self.fixed_property_keys: Sequence[str] = fixed_property_keys

        # subclasses can append to this if they want to store more fixed
        # arrays in the output h5 file
        self.fixed_output_data: list[tuple[str, Any]] = [
            ("sampling_frequency", self.recording.get_sampling_frequency()),
            ("geom", self.recording.get_channel_locations()),
        ]
        if channel_index is not None:
            self.fixed_output_data.append(
                ("channel_index", self.b.channel_index.numpy(force=True).copy()),
            )

        self._rgs: local = local()

    # -- main functions for users to call
    # in practice users will interact with the functions `subtract(...)` in
    # subtract.py and similar functions in the other .py files here, but these
    # are the main API methods for this class

    def load_or_fit_and_save_models(
        self, save_folder, overwrite=False, computation_cfg=None
    ):
        """Load fitted models from save_folder if possible, or fit and save

        If the peeler has models that need to be trained, this function ensures
        that the models are fitted and that their fitted parameters are saved
        to `save_folder`
        """
        save_folder = Path(save_folder)
        if overwrite and save_folder.exists():
            for pt_file in save_folder.glob("*pipeline.pt"):
                pt_file.unlink()
        if self.needs_precompute() or self.needs_fit():
            self.load_models(save_folder)
        if self.needs_precompute() or self.needs_fit():
            computation_cfg = ensure_computation_config(computation_cfg)
            if self.needs_precompute():
                self.precompute_models(
                    save_folder, overwrite=overwrite, computation_cfg=computation_cfg
                )
            if self.needs_fit():
                save_folder.mkdir(exist_ok=True)
                self.fit_models(
                    save_folder, overwrite=overwrite, computation_cfg=computation_cfg
                )
            self.save_models(save_folder)

            cleanup_and_log_gpu_usage(computation_cfg, "peel: Usage after model fits:")
        assert not self.needs_precompute()
        assert not self.needs_fit()
        self.post_fit()

    def peel(
        self,
        output_hdf5_filename,
        chunk_starts_samples=None,
        chunk_length_samples=None,
        total_residual_snips=None,
        residual_snips_per_chunk=None,
        stop_after_n_waveforms: int | None = None,
        ensure_coverage: float | None = None,
        shuffle=False,
        overwrite=False,
        residual_filename=None,
        show_progress=True,
        skip_features=False,
        residual_to_h5=False,
        task_name=None,
        ignore_resuming=False,
        computation_cfg=None,
        known_spike_count: int | None = None,
    ):
        """Run the full (already fitted) peeling and featurization pipeline

        This gathers all results into an hdf5 file, resuming from where
        it left off in that file if overwrite=False.
        """
        if self.needs_fit():
            raise ValueError("Peeler needs to be fitted before peeling.")

        if output_hdf5_filename is None:
            # all of these features would require the h5 file.
            assert ignore_resuming
            assert not residual_to_h5
            assert residual_snips_per_chunk is None
            assert total_residual_snips is None

        if task_name is None:
            task_name = self.peel_kind

        # figure out which chunks to process, and exit early if already done
        chunk_starts_samples = self.get_chunk_starts(
            chunk_starts_samples=chunk_starts_samples
        )
        chunk_starts_samples = np.asarray(chunk_starts_samples, dtype=np.int64)
        if shuffle:
            chunk_starts_samples = self.sample_chunks(
                chunk_starts_samples=chunk_starts_samples,
                n_chunks=chunk_starts_samples.shape[0],
            )
        n_chunks_orig = len(chunk_starts_samples)

        # this is -1 if we haven't started yet
        if ignore_resuming:
            done = False
            last_chunk_index = -1
            next_chunk_index = 0
            resids_so_far = 0
        elif output_hdf5_filename is not None:
            done, last_chunk_index, resids_so_far = self.check_resuming(
                output_hdf5_filename,
                chunk_starts_samples=chunk_starts_samples,
                stop_after_n_waveforms=stop_after_n_waveforms,
                ensure_coverage=ensure_coverage,
                overwrite=overwrite,
            )
            next_chunk_index = last_chunk_index + 1
            logger.dartsortdebug(
                f"[{self.__class__.__name__}:{task_name}] Resuming at chunk "
                f"{next_chunk_index}/{n_chunks_orig}."
            )
        else:
            assert False
        if done:
            return
        chunks_to_do = chunk_starts_samples[next_chunk_index:]

        clen = chunk_length_samples or self.chunk_length_samples
        max_resid_per_chk = clen // self.spike_length_samples
        if (
            total_residual_snips is not None
            and ensure_coverage
            and stop_after_n_waveforms is not None
        ):
            assert 0 <= ensure_coverage <= 1
            assert residual_snips_per_chunk is None
            resids_remaining = total_residual_snips - resids_so_far
            chunks_remaining = len(chunks_to_do)
            chunks_done = n_chunks_orig - chunks_remaining
            chunks_cover = int(np.ceil(ensure_coverage * n_chunks_orig))
            chunks_cover_remaining = chunks_cover - chunks_done
            if chunks_cover_remaining == 0:
                assert resids_remaining == 0
                residual_snips_per_chunk = 0
            else:
                residual_snips_per_chunk = divide_randomly(
                    resids_remaining,
                    chunks_cover_remaining,
                    self.fit_subsampling_random_state,
                    len(chunks_to_do),
                    max_resid_per_chk,
                )
                logger.dartsortdebug(
                    f"Will draw {residual_snips_per_chunk[:chunks_cover_remaining].mean():.1f} "
                    f"residual snips on average per coverage chunk ({chunks_cover_remaining} "
                    "remaining to hit coverage)."
                )
        elif total_residual_snips is not None:
            assert residual_snips_per_chunk is None
            resids_remaining = total_residual_snips - resids_so_far
            residual_snips_per_chunk = divide_randomly(
                resids_remaining, len(chunks_to_do), self.fit_subsampling_random_state
            )

        if residual_snips_per_chunk is None:
            if residual_to_h5:
                residual_snips_per_chunk = repeat(1)
            else:
                residual_snips_per_chunk = repeat(None)
        elif isinstance(residual_snips_per_chunk, int):
            residual_to_h5 = bool(residual_snips_per_chunk)
            residual_snips_per_chunk = repeat(residual_snips_per_chunk)
        else:
            residual_to_h5 = True
            assert len(residual_snips_per_chunk) == len(chunks_to_do)

        jobs = list(zip(chunks_to_do, residual_snips_per_chunk))

        if not chunks_to_do.size:
            return output_hdf5_filename
        save_residual = residual_filename is not None
        compute_residual = save_residual or residual_to_h5
        if chunk_length_samples is None:
            chunk_length_samples = self.chunk_length_samples
        computation_cfg = ensure_computation_config(computation_cfg)

        to_cpu = output_hdf5_filename is not None

        # main peeling loop
        # wrap in try/finally to ensure file handles get closed if there
        # is some unforseen error
        try:
            n_jobs, Executor, context, rank_queue, is_local = pool_from_cfg(
                computation_cfg, with_rank_queue=True, check_local=True
            )
            with Executor(
                max_workers=n_jobs,
                mp_context=context,
                initializer=_peeler_process_init,
                initargs=(
                    self,
                    computation_cfg.actual_device(),
                    rank_queue,
                    compute_residual,
                    skip_features,
                    chunk_length_samples,
                    is_local,
                    to_cpu,
                ),
            ) as pool:
                if is_local:
                    self.to(computation_cfg.actual_device())

                # launch the jobs and wrap in a progress bar
                results = pool.map(_peeler_process_job, jobs)
                if show_progress:
                    s_chunk = chunk_length_samples / self.recording.sampling_frequency
                    dtag = computation_cfg.actual_device().type
                    results = progbar(
                        results,
                        total=n_chunks_orig,
                        initial=n_chunks_orig - len(chunks_to_do),
                        smoothing=0,
                        desc=f"{task_name}:{dtag} {s_chunk:.1f}s/it [spk/it=%%%]",
                        mininterval=1.0,
                    )
                else:
                    dtag = s_chunk = None

                # construct h5 after forking to avoid pickling it
                with self.initialize_files(
                    output_hdf5_filename,
                    chunk_starts_samples=chunk_starts_samples,
                    residual_filename=residual_filename,
                    overwrite=overwrite,
                    skip_features=skip_features,
                    residual_to_h5=residual_to_h5,
                    known_spike_count=known_spike_count,
                    ignore_resuming=ignore_resuming,
                    total_residual_snips=total_residual_snips,
                ) as (output_h5, h5_spike_datasets, residual_file, n_spikes):
                    batch_count = 0
                    try:
                        for result, chunk_start_samples in zip(results, chunks_to_do):
                            n_new_spikes = self.gather_chunk_result(
                                cur_n_spikes=n_spikes,
                                chunk_start_samples=chunk_start_samples,
                                chunk_result=result,
                                h5_spike_datasets=h5_spike_datasets,
                                output_h5=output_h5,
                                residual_file=residual_file,
                                ignore_resuming=ignore_resuming,
                                skip_features=skip_features,
                            )
                            batch_count += 1
                            n_spikes += n_new_spikes
                            if show_progress:
                                desc = f"{task_name}:{dtag}"
                                if not skip_features:
                                    assert s_chunk is not None
                                    desc += f" [spk/{s_chunk:g}s={n_spikes / batch_count:0.1f}]"
                                results.set_description(desc, refresh=False)

                            # early stopping criteria
                            # no early stopping if we're skipping features (used for residual)
                            can_stop = not skip_features
                            # only allow stop if we have enough waveforms
                            if stop_after_n_waveforms is not None:
                                enough_spikes = n_spikes >= stop_after_n_waveforms
                                can_stop = can_stop and enough_spikes
                            else:
                                can_stop = False
                            # prevent stop if we're asked to cover but didn't
                            if ensure_coverage is not None:
                                chunks_done = batch_count + last_chunk_index + 1
                                covered = chunks_done / n_chunks_orig >= ensure_coverage
                                can_stop = can_stop and covered
                            if can_stop:
                                pool.shutdown(cancel_futures=True)
                    except CancelledError:
                        if show_progress:
                            results.write(
                                f"Got {n_spikes} spikes, enough to stop early."
                            )
        finally:
            self.to("cpu")

            # we very much do not want to leak self into a global and have its
            # memory tied up while dartsort wants to go do other stuff
            global _peeler_process_context
            del _peeler_process_context.ctx
            _peeler_process_context.ctx = None

        return output_hdf5_filename

    # -- methods for subclasses to override

    def peel_chunk(
        self,
        traces: torch.Tensor,
        *,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ) -> "PeelingBatchResult":
        # subclasses should implement this method

        # they should return a dictionary with keys:
        #  - n_spikes
        #  - collisioncleaned_waveforms
        #  - times_samples (relative to start of traces)
        #  - channels
        #  - residual if requested
        #  - arbitrary subclass-specific stuff which should have keys in out_datasets()

        # data for spikes in the margin should not be returned
        # these spikes will have been processed by the neighboring chunk

        raise NotImplementedError

    def peeling_needs_fit(self) -> bool:
        return False

    def peeling_needs_precompute(self) -> bool:
        return False

    def post_fit(self):
        pass

    def precompute_peeling_data(
        self, save_folder, overwrite=False, computation_cfg=None
    ):
        # subclasses should override if they need to cache data for peeling
        # runs before fit_peeler_models()
        pass

    def fit_peeler_models(self, save_folder, tmp_dir=None, computation_cfg=None):
        # subclasses should override if they need to fit models for peeling
        assert not self.peeling_needs_fit()

    def precompute_peeler_models(
        self, save_folder, overwrite=False, computation_cfg=None
    ):
        if self.peeling_needs_precompute():
            self.precompute_peeling_data(
                save_folder=save_folder,
                overwrite=overwrite,
                computation_cfg=computation_cfg,
            )
        assert not self.peeling_needs_precompute()

    # subclasses can add to this list
    # for each dataset in this list, an output dataset in the
    # hdf5 (of .peel()) will be created with this dtype and with
    # shape (N_spikes, *shape_per_spike)
    # datasets will also be created corresponding to features
    # in featurization_pipeline
    def out_datasets(self):
        datasets = [
            SpikeDataset(name="times_samples", shape_per_spike=(), dtype=np.int64),
            SpikeDataset(name="times_seconds", shape_per_spike=(), dtype=np.float64),
            SpikeDataset(name="channels", shape_per_spike=(), dtype=np.int64),
        ]
        if self.featurization_pipeline is not None:
            datasets.extend(self.featurization_pipeline.spike_datasets())
        return datasets

    # -- utility methods which users likely won't touch

    def featurize_collisioncleaned_waveforms(
        self, collisioncleaned_waveforms, **fixed_properties
    ):
        if not self.featurization_pipeline:
            return {}

        waveforms, features = self.featurization_pipeline(
            collisioncleaned_waveforms, **fixed_properties
        )
        return features

    def process_chunk(
        self,
        chunk_start_samples: int,
        *,
        chunk_end_samples: int | None = None,
        return_residual: bool = False,
        skip_features: bool = False,
        n_resid_snips: int | None = None,
        to_cpu: bool = True,
        **peel_kw,
    ) -> "PeelingBatchResult":
        """Grab, peel, and featurize a chunk, returning a dict of numpy arrays

        Main function called in peeling workers
        """
        chunk, chunk_end_samples, left_margin, right_margin = self.get_chunk(
            chunk_start_samples, chunk_end_samples
        )
        return_waveforms = not skip_features and bool(self.featurization_pipeline)
        peel_result = self.peel_chunk(
            chunk,
            chunk_start_samples=chunk_start_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            return_waveforms=return_waveforms,
            return_residual=return_residual or bool(n_resid_snips),
            **peel_kw,
        )
        chunk_result = self.featurize_chunk_result(
            peel_result=peel_result,
            to_cpu=to_cpu,
            return_waveforms=return_waveforms,
            chunk_start_samples=chunk_start_samples,
            chunk_end_samples=chunk_end_samples,
            device=chunk.device,
            n_resid_snips=n_resid_snips,
        )
        return chunk_result

    def get_chunk(self, chunk_start_samples: int, chunk_end_samples: int | None = None):
        Ts = self.recording.get_num_samples()
        if chunk_end_samples is None:
            chunk_end_samples = chunk_start_samples + self.chunk_length_samples
            chunk_end_samples = min(Ts, chunk_end_samples)
        assert chunk_end_samples <= Ts
        chunk, left_margin, right_margin = get_chunk_with_margin(
            self.recording._recording_segments[0],
            start_frame=chunk_start_samples,
            end_frame=chunk_end_samples,
            channel_indices=None,
            margin=self.chunk_margin_samples,
        )
        assert isinstance(chunk, np.ndarray)
        device = self.b.channel_index.device
        if device.type == "cpu":
            chunk = torch.tensor(chunk, device=device, dtype=self.dtype)
        else:
            # we have to copy anyway to go to device, so working around the
            # torch warning here. this way, can avoid blocking.
            with catch_warnings():
                filterwarnings("ignore", message="The given NumPy array is not ")
                chunk = torch.from_numpy(chunk)
                chunk = chunk.to(device=device, dtype=self.dtype, non_blocking=True)
        return chunk, chunk_end_samples, left_margin, right_margin

    def featurize_chunk_result(
        self,
        *,
        peel_result,
        to_cpu: bool,
        return_waveforms: bool,
        chunk_start_samples: int,
        chunk_end_samples: int,
        device: torch.device,
        n_resid_snips: int | None,
    ):
        if peel_result["n_spikes"] > 0 and to_cpu:
            t_s = self.recording.sample_index_to_time(
                peel_result["times_samples"].numpy(force=True)
            )
            peel_result["times_seconds"] = torch.asarray(t_s, device=device)

        if peel_result["n_spikes"] > 0 and return_waveforms:
            chunk_start_s = self.recording.sample_index_to_time(chunk_start_samples)
            chunk_end_s = self.recording.sample_index_to_time(chunk_end_samples)
            chunk_center_s = (chunk_start_s + chunk_end_s) / 2
            fixed_properties = {k: peel_result[k] for k in self.fixed_property_keys}
            features = self.featurize_collisioncleaned_waveforms(
                peel_result["collisioncleaned_waveforms"],
                chunk_center_s=chunk_center_s,
                **fixed_properties,
            )
        else:
            features = {}

        # a user who wants these must featurize with a waveform node
        # then they'll end up in `features`
        if "collisioncleaned_waveforms" in peel_result:
            del peel_result["collisioncleaned_waveforms"]

        chunk_result = peel_result | features
        if to_cpu:
            chunk_result = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in chunk_result.items()
            }
            if "residual" in peel_result:
                if torch.is_tensor(peel_result["residual"]):
                    peel_result["residual"] = peel_result["residual"].cpu()

        # add times in seconds
        chunk_result["chunk_start_seconds"] = self.recording.sample_index_to_time(
            chunk_start_samples
        )

        if n_resid_snips:
            chunk_result["resid_snips"], resid_times_samples = extract_random_snips(
                self.rg,
                peel_result["residual"],
                n_resid_snips,
                self.resid_length_samples,
            )
            resid_times_samples = chunk_start_samples + resid_times_samples
            chunk_result["residual_times_seconds"] = (
                self.recording.sample_index_to_time(resid_times_samples)
            )

        return chunk_result

    def gather_chunk_result(
        self,
        *,
        cur_n_spikes: int,
        chunk_start_samples: int,
        chunk_result,
        h5_spike_datasets: dict[str, h5py.Dataset] | None,
        output_h5,
        residual_file,
        ignore_resuming: bool,
        skip_features: bool,
    ):
        # delay keyboard interrupts so we don't write half a batch
        # of data and leave files in an invalid state after ^C
        # not that something else couldn't happen...
        with delay_keyboard_interrupt:
            if not ignore_resuming:
                output_h5["last_chunk_index"][()] = (
                    output_h5["last_chunk_index"][()] + 1
                )
                output_h5["last_chunk_start"][()] = chunk_start_samples

            if residual_file is not None:
                chunk_result["residual"].tofile(residual_file)

            if "resid_snips" in chunk_result and output_h5["residual"].chunks is None:
                nr0 = output_h5["n_residuals"][()]
                nr1 = nr0 + len(chunk_result["resid_snips"])
                output_h5["residual"][nr0:nr1] = chunk_result["resid_snips"]
                output_h5["residual_times_seconds"][nr0:nr1] = chunk_result[
                    "residual_times_seconds"
                ]
                output_h5["n_residuals"][()] = nr1
            elif "resid_snips" in chunk_result:
                nr0 = output_h5["n_residuals"][()]
                assert nr0 == len(output_h5["residual"])
                nr1 = nr0 + len(chunk_result["resid_snips"])
                output_h5["residual"].resize(nr1, axis=0)
                output_h5["residual"][nr0:] = chunk_result["resid_snips"]
                output_h5["residual_times_seconds"].resize(nr1, axis=0)
                output_h5["residual_times_seconds"][nr0:] = chunk_result[
                    "residual_times_seconds"
                ]
                output_h5["n_residuals"][()] = nr1

            if skip_features:
                return 0
            assert h5_spike_datasets is not None

            n_new_spikes = chunk_result["n_spikes"]
            if not n_new_spikes:
                return 0

            for ds in self.out_datasets():
                h5ds = h5_spike_datasets[ds.name]
                i1 = cur_n_spikes + n_new_spikes
                if h5ds.chunks is not None:
                    h5ds.resize(i1, axis=0)
                h5ds[cur_n_spikes:i1] = chunk_result[ds.name]

        return n_new_spikes

    def needs_fit(self):
        if self.peeling_needs_fit():
            return True
        if self.featurization_pipeline is not None:
            return self.featurization_pipeline.needs_fit()
        return False

    def needs_precompute(self):
        if self.peeling_needs_precompute():
            return True
        if self.featurization_pipeline is not None:
            return self.featurization_pipeline.needs_precompute()
        return False

    def fit_models(
        self, save_folder, tmp_dir=None, overwrite=False, computation_cfg=None
    ):
        with torch.no_grad():
            if self.peeling_needs_fit():
                self.precompute_peeling_data(
                    save_folder=save_folder,
                    overwrite=overwrite,
                    computation_cfg=computation_cfg,
                )
                self.fit_peeler_models(
                    save_folder=save_folder,
                    tmp_dir=tmp_dir,
                    computation_cfg=computation_cfg,
                )
            self.fit_featurization_pipeline(
                tmp_dir=tmp_dir,
                computation_cfg=computation_cfg,
            )
        assert not self.needs_fit()

    def precompute_models(self, save_folder, overwrite=False, computation_cfg=None):
        if self.peeling_needs_precompute():
            self.precompute_peeler_models(
                save_folder=save_folder,
                overwrite=overwrite,
                computation_cfg=computation_cfg,
            )
        if self.featurization_pipeline is None:
            return
        self.featurization_pipeline.precompute()

    def fit_featurization_pipeline(self, tmp_dir=None, computation_cfg=None):
        if self.featurization_pipeline is None:
            return
        if not self.featurization_pipeline.needs_fit():
            return

        computation_cfg = ensure_computation_config(computation_cfg)
        device = computation_cfg.actual_device()

        # disable self's featurization pipeline, replacing it with a waveform
        # saving node. then, we'll use those waveforms to fit the original
        featurization_pipeline = self.featurization_pipeline
        self.featurization_pipeline = WaveformPipeline.from_class_names_and_kwargs(
            self.recording.get_channel_locations(),
            self.channel_index,
            [
                ("Voltage", {"name": "peeled_voltages_fit"}),
                ("Waveform", {"name": "peeled_waveforms_fit"}),
            ],
            waveform_cfg=self.waveform_cfg,
            sampling_frequency=self.recording.sampling_frequency,
        )

        if tmp_dir is None:
            tmp_dir = computation_cfg.tmpdir_parent

        with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
            temp_hdf5_filename = Path(temp_dir) / "peeler_fit.h5"
            waveforms = fixed_properties = None
            try:
                if featurization_pipeline.needs_residual():
                    n_resid_snips = self.fit_sampling_cfg.n_residual_snips
                else:
                    n_resid_snips = None
                more = featurization_pipeline.needs_more_features()
                self.run_subsampled_peeling(
                    temp_hdf5_filename,
                    computation_cfg=computation_cfg,
                    task_name="Load examples for feature fitting",
                    total_residual_snips=n_resid_snips,
                    more=more,
                )

                # park myself on cpu while models fit in case they need
                # to take up space
                self.to(device="cpu")

                # fit featurization pipeline and reassign
                # work in a try finally so we can delete the temp file
                # in case of an issue or a keyboard interrupt
                waveforms, fixed_properties = subsample_waveforms(
                    temp_hdf5_filename,
                    fit_sampling=self.fit_sampling_cfg.fit_sampling,
                    random_state=self.fit_subsampling_random_state,
                    n_waveforms_fit=self.fit_sampling_cfg.n_waveforms_fit,
                    fit_max_reweighting=self.fit_sampling_cfg.fit_max_reweighting,
                    voltages_dataset_name="peeled_voltages_fit",
                    waveforms_dataset_name="peeled_waveforms_fit",
                    fixed_property_keys=self.fixed_property_keys,
                    device=device,
                )
                if not len(waveforms):
                    raise ValueError("Found no spikes when trying to fit featurizers.")

                workers = computation_cfg.actual_n_jobs(small=True, cpu=True)
                _, workers = handle_negative_jobs(workers)
                featurization_pipeline.register_cpu_workers(workers)
                featurization_pipeline = featurization_pipeline.to(device)
                featurization_pipeline.fit(
                    recording=self.recording,
                    waveforms=waveforms,
                    computation_cfg=computation_cfg,
                    hdf5_filename=temp_hdf5_filename,
                    waveforms_dataset_name="peeled_waveforms_fit",
                    **fixed_properties,
                )
                if getrefcount(waveforms) > 2:
                    logger.warn(f"Fit waveforms had {getrefcount(waveforms)=}")
                    for obj in gc.get_referrers(waveforms):
                        logger.warn(f"{repr(obj)}: {str(obj)}")
                featurization_pipeline = featurization_pipeline.to("cpu")
                self.featurization_pipeline = featurization_pipeline
            finally:
                self.to(device="cpu")
                if temp_hdf5_filename.exists():
                    temp_hdf5_filename.unlink()
                del waveforms, fixed_properties
                gc.collect()

    def get_chunk_starts(
        self,
        chunk_starts_samples=None,
        chunk_length_samples=None,
        t_start=None,
        t_end=None,
        subsampled=False,
        n_chunks=None,
        ordered=False,
        skip_last=False,
    ):
        if chunk_starts_samples is not None:
            return chunk_starts_samples
        if chunk_length_samples is None:
            chunk_length_samples = self.chunk_length_samples
        assert isinstance(chunk_length_samples, int)

        T_samples = self.recording.get_num_samples()
        if t_end is None:
            t_end = T_samples
        if t_start is None:
            t_start = 0
        chunk_starts_samples = np.arange(t_start, t_end, chunk_length_samples)
        if skip_last:
            if t_end - chunk_starts_samples[-1] < chunk_length_samples:
                chunk_starts_samples = chunk_starts_samples[:-1]

        if not subsampled:
            return chunk_starts_samples

        if n_chunks is None:
            chk_per_s = self.recording.sampling_frequency / chunk_length_samples
            n_chunks = int(np.ceil(self.fit_sampling_cfg.n_seconds_fit * chk_per_s))
        n_chunks = min(len(chunk_starts_samples), n_chunks)
        assert isinstance(n_chunks, int)

        # make a random subset of chunks to use for fitting
        chunk_starts_samples = self.sample_chunks(
            chunk_starts_samples=chunk_starts_samples,
            n_chunks=n_chunks,
            ordered=ordered,
        )
        return chunk_starts_samples

    def sample_chunks(
        self, chunk_starts_samples: np.ndarray, n_chunks: int, ordered: bool = False
    ) -> np.ndarray:
        if self.fit_sampling_cfg.chunk_sampling == "random":
            rg = np.random.default_rng(self.random_seed)
            chunk_starts_samples = rg.choice(
                chunk_starts_samples, size=n_chunks, replace=False
            )
        elif self.fit_sampling_cfg.chunk_sampling == "kmeanspp":
            _ix = _kmeansppix(len(chunk_starts_samples), n_chunks, self.random_seed)
            chunk_starts_samples = chunk_starts_samples[_ix]
        else:
            assert False

        if ordered:
            chunk_starts_samples.sort()

        return chunk_starts_samples

    def run_subsampled_peeling(
        self,
        hdf5_filename: str | Path,
        chunk_length_samples=None,
        n_chunks: int | None = None,
        stop_after_n_waveforms: int | None = None,
        residual_to_h5=False,
        total_residual_snips: int | None = None,
        skip_features=False,
        ignore_resuming=False,
        more=False,
        skip_last=False,
        computation_cfg=None,
        task_name=None,
        overwrite=True,
        ordered=False,
        show_progress=True,
    ):
        if n_chunks is not None:
            chunk_starts = self.get_chunk_starts(
                chunk_length_samples=chunk_length_samples,
                subsampled=True,
                n_chunks=n_chunks,
                ordered=ordered,
                skip_last=skip_last,
            )
        else:
            assert not ordered
            chunk_starts = None

        if total_residual_snips:
            assert n_chunks is None
            # ensure minimal coverage to get residual snips
            clen = chunk_length_samples or self.chunk_length_samples
            snips_per_chunk = clen / self.spike_length_samples
            n_chunks_needed = total_residual_snips / snips_per_chunk
            n_chunks_needed /= self.fit_sampling_cfg.residual_sampling_target_density
            Tf = self.recording.get_num_frames()
            coverage = min(1.0, n_chunks_needed / (Tf / clen))
            assert coverage >= 0
            logger.dartsortdebug(
                f"Subsampled peeling will request coverage {coverage:.2f} "
                f"to help with grabbing {total_residual_snips} residual snips."
            )
        else:
            coverage = None

        if stop_after_n_waveforms is not None:
            pass
        elif more:
            stop_after_n_waveforms = self.fit_sampling_cfg.more_waveforms_fit
        else:
            stop_after_n_waveforms = self.fit_sampling_cfg.max_waveforms_fit

        return self.peel(
            output_hdf5_filename=hdf5_filename,
            chunk_starts_samples=chunk_starts,
            chunk_length_samples=chunk_length_samples,
            stop_after_n_waveforms=stop_after_n_waveforms,
            ensure_coverage=coverage,
            ignore_resuming=ignore_resuming,
            residual_to_h5=residual_to_h5,
            total_residual_snips=total_residual_snips,
            skip_features=skip_features,
            computation_cfg=computation_cfg,
            overwrite=overwrite,
            task_name=task_name,
            show_progress=show_progress,
            shuffle=chunk_starts is None,
        )

    def save_models(self, save_folder: str | Path):
        if self.featurization_pipeline is None:
            return

        save_folder = Path(save_folder)
        save_folder.mkdir(exist_ok=True)
        torch.save(
            self.featurization_pipeline.state_dict(),
            save_folder / "featurization_pipeline.pt",
        )

    def load_models(self, save_folder: str | Path):
        save_folder = Path(save_folder)
        if not save_folder.exists():
            return

        feats_pt = save_folder / "featurization_pipeline.pt"
        if feats_pt.exists():
            assert self.featurization_pipeline is not None
            state_dict = torch.load(feats_pt, weights_only=True)
            self.featurization_pipeline.load_state_dict(state_dict)

    def check_resuming(
        self,
        output_hdf5_filename: str | Path,
        chunk_starts_samples: np.ndarray,
        stop_after_n_waveforms: int | None = None,
        ensure_coverage: float | None = None,
        overwrite=False,
    ) -> tuple[bool, int, int]:
        output_hdf5_filename = Path(output_hdf5_filename)
        exists = output_hdf5_filename.exists()
        last_chunk_index = -1
        n_spikes = 0
        residual_snips_so_far = 0
        if exists and not overwrite:
            with h5py.File(output_hdf5_filename, "r", locking=False) as h5:
                last_chunk_start: int = h5["last_chunk_start"][()]
                h5_starts = h5["chunk_starts_samples"][:]
                if not np.array_equal(chunk_starts_samples, h5_starts):
                    raise ValueError(
                        "Trying to resume a peeling job, but the chunk starts "
                        "on disk differ. Maybe shuffle flag changed?"
                    )
                n_spikes = len(h5["times_samples"])
                if last_chunk_start >= 0:
                    is_last = np.flatnonzero(chunk_starts_samples == last_chunk_start)
                    assert is_last.shape == (1,)
                    last_chunk_index = is_last[0]
                assert last_chunk_index == h5["last_chunk_index"][()]

                residual_snips_so_far = 0
                if "residual" in h5:
                    residual_snips_so_far = len(h5["residual"])

        # am I done?
        done = all_done = last_chunk_index == len(chunk_starts_samples) - 1
        if stop_after_n_waveforms is not None:
            done = n_spikes >= stop_after_n_waveforms
        if ensure_coverage:
            coverage = (last_chunk_index + 1) / len(chunk_starts_samples)
            done = done and coverage >= ensure_coverage
        done = all_done or done
        return done, last_chunk_index, residual_snips_so_far

    @contextmanager
    def initialize_files(
        self,
        output_hdf5_filename: str | Path | None,
        chunk_starts_samples: np.ndarray,
        residual_filename=None,
        overwrite=False,
        chunk_size=1024,
        libver="latest",
        locking=False,
        residual_to_h5=False,
        skip_features=False,
        known_spike_count: int | None = None,
        total_residual_snips: int | None = None,
        ignore_resuming: bool = False,
    ):
        """Create, overwrite, or re-open output files"""
        if output_hdf5_filename is None:
            # Supports subclasses which specifically avoid doing any saving.
            assert residual_filename is None
            assert not residual_to_h5
            yield None, None, None, 0
            return

        output_hdf5_filename = Path(output_hdf5_filename)
        exists = output_hdf5_filename.exists()
        n_spikes = 0
        if exists and overwrite:
            # overwriting, destroy destroy
            output_hdf5_filename.unlink()
            output_h5 = h5py.File(
                output_hdf5_filename, "w", libver=libver, locking=locking
            )
            output_h5.create_dataset("last_chunk_start", data=-1, dtype=np.int64)
            output_h5.create_dataset("last_chunk_index", data=-1, dtype=np.int64)
            output_h5.create_dataset("chunk_starts_samples", data=chunk_starts_samples)
        elif exists and ignore_resuming:
            # ignore the previous chunk structure, don't touch it, ignore ignore.
            output_h5 = h5py.File(
                output_hdf5_filename, "r+", libver=libver, locking=locking
            )
            n_spikes = len(output_h5["times_samples"])
        elif exists:
            # exists and possibly resuming
            output_h5 = h5py.File(
                output_hdf5_filename, "r+", libver=libver, locking=locking
            )
            n_spikes = len(output_h5["times_samples"])
            # check chunks match, or else what are we even doin!
            _stored_chunks = output_h5["chunk_starts_samples"][:]
            assert np.array_equal(chunk_starts_samples, _stored_chunks)
        else:
            # create create
            output_h5 = h5py.File(
                output_hdf5_filename, "w", libver=libver, locking=locking
            )
            output_h5.create_dataset("last_chunk_start", data=-1, dtype=np.int64)
            output_h5.create_dataset("last_chunk_index", data=-1, dtype=np.int64)
            output_h5.create_dataset("chunk_starts_samples", data=chunk_starts_samples)

        if known_spike_count is not None:
            assert known_spike_count >= n_spikes

        # write some fixed arrays that are useful to have around
        for name, value in self.fixed_output_data:
            if name not in output_h5:
                output_h5.create_dataset(name, data=value)

        # don't used chunked storage for residuals if possible
        need_resid = residual_to_h5 and "residual" not in output_h5
        if need_resid and total_residual_snips is not None:
            assert "residual_times_seconds" not in output_h5
            n_chans = self.recording.get_num_channels()
            output_h5.create_dataset("n_residuals", data=np.zeros((), dtype=np.int64))
            output_h5.create_dataset(
                "residual",
                dtype=self.np_dtype,
                shape=(total_residual_snips, self.resid_length_samples, n_chans),
                fillvalue=np.nan,
            )
            output_h5.create_dataset(
                "residual_times_seconds",
                dtype=float,
                shape=(total_residual_snips,),
                fillvalue=np.nan,
            )
        elif need_resid:
            assert "residual_times_seconds" not in output_h5
            n_chans = self.recording.get_num_channels()
            output_h5.create_dataset("n_residuals", data=np.zeros((), dtype=np.int64))
            output_h5.create_dataset(
                "residual",
                dtype=self.np_dtype,
                shape=(0, self.resid_length_samples, n_chans),
                maxshape=(None, self.resid_length_samples, n_chans),
                chunks=(chunk_size, self.resid_length_samples, n_chans),
            )
            output_h5.create_dataset(
                "residual_times_seconds",
                dtype=float,
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
            )

        # create per-spike datasets
        # use chunks to support growing the dataset as we find spikes
        if skip_features:
            h5_spike_datasets = None
        else:
            h5_spike_datasets = {}
            for ds in self.out_datasets():
                if ds.name in output_h5:
                    dset = output_h5[ds.name]
                    assert isinstance(dset, h5py.Dataset)
                elif known_spike_count is None:
                    dset = output_h5.create_dataset(
                        ds.name,
                        dtype=ds.dtype,
                        shape=(n_spikes, *ds.shape_per_spike),
                        maxshape=(known_spike_count, *ds.shape_per_spike),
                        chunks=(chunk_size, *ds.shape_per_spike),
                    )
                else:
                    dset = output_h5.create_dataset(
                        ds.name,
                        dtype=ds.dtype,
                        shape=(known_spike_count, *ds.shape_per_spike),
                    )
                h5_spike_datasets[ds.name] = dset

        # residual file ignore/open/overwrite logic
        save_residual = residual_filename is not None
        residual_file = None
        if save_residual:
            assert not ignore_resuming
            assert residual_filename is not None
            last_chunk_start: int = output_h5["last_chunk_start"][()]
            last_chunk_index: int = output_h5["last_chunk_index"][()]
            if last_chunk_start >= 0:
                assert last_chunk_index >= 0
                residual_mode = "ab"
                assert Path(residual_filename).exists()
            else:
                residual_mode = "wb"
            residual_file = open(residual_filename, mode=residual_mode)

        try:
            yield output_h5, h5_spike_datasets, residual_file, n_spikes
        finally:
            output_h5.close()
            if save_residual:
                assert residual_file is not None
                residual_file.close()
            self.to("cpu")

    # -- thread-local rngs. they're not thread safe, and locals can't be pickled

    def __getstate__(self):
        state = super().__getstate__()
        del state["_rgs"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._rgs = local()

    @property
    def rg(self):
        if not hasattr(self._rgs, "rg"):
            with _lock:
                self._rgs.rg = self.fit_subsampling_random_state.spawn(1)[0]
        return self._rgs.rg

    @staticmethod
    def next_margin(length, factor=10):
        return factor * int(np.ceil(length / factor))

    def nearest_batch_length(self, target=512):
        factors = divisors(self.chunk_length_samples + 2 * self.chunk_margin_samples)
        factors = np.array(factors)
        return factors[np.abs(factors - target).argmin()]


# -- batch result type stub


try:
    # this isn't supported until 3.15, but it helps the type checker
    # so I'm keeping it in here. everything seems to be fine just setting
    # it to dict in practice in the except clause.

    class PeelingBatchResult(TypedDict, extra_items=torch.Tensor):
        n_spikes: int
except TypeError:
    PeelingBatchResult = dict  # type: ignore


peeling_empty_result = PeelingBatchResult(n_spikes=0)


# -- helper functions and objects for parallelism


class PeelerProcessContext:
    def __init__(
        self, peeler, compute_residual, skip_features, chunk_length_samples, to_cpu
    ):
        self.peeler = peeler
        self.compute_residual = compute_residual
        self.skip_features = skip_features
        self.chunk_length_samples = chunk_length_samples
        self.to_cpu = to_cpu
        self.Ts = peeler.recording.get_num_samples()


# this state will be set on each thread globally
_peeler_process_context = local()
_peeler_process_context.ctx = None


def _peeler_process_init(
    peeler,
    device,
    rank_queue,
    compute_residual,
    skip_features,
    chunk_length_samples,
    is_local,
    to_cpu,
):
    global _peeler_process_context

    # figure out what device to work on
    my_rank = rank_queue.get()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        if torch.cuda.device_count() > 1:
            index = my_rank % torch.cuda.device_count()
            device = torch.device("cuda", index=index)

    # move peeler to device and put everything in inference mode
    if not is_local:
        peeler.to(device)
    peeler.eval()

    # update process-local variables
    _peeler_process_context.ctx = PeelerProcessContext(
        peeler, compute_residual, skip_features, chunk_length_samples, to_cpu
    )


def _peeler_process_job(chunk_start_samples__n_resid_snips):
    chunk_start_samples, n_resid_snips = chunk_start_samples__n_resid_snips
    # by returning here, we are implicitly relying on pickle
    # TODO: replace with manual np.saves
    with torch.no_grad():
        chlen = _peeler_process_context.ctx.chunk_length_samples
        if chlen is not None:
            chunk_end_samples = chunk_start_samples + chlen
            chunk_end_samples = min(chunk_end_samples, _peeler_process_context.ctx.Ts)
        else:
            chunk_end_samples = None
        return _peeler_process_context.ctx.peeler.process_chunk(
            chunk_start_samples,
            n_resid_snips=n_resid_snips,
            chunk_end_samples=chunk_end_samples,
            return_residual=_peeler_process_context.ctx.compute_residual,
            skip_features=_peeler_process_context.ctx.skip_features,
            to_cpu=_peeler_process_context.ctx.to_cpu,
        )


def _kmeansppix(n: int, k: int | None = None, seed=0):
    rg = np.random.default_rng(seed)
    k = k if k is not None else n
    assert 0 < k <= n

    xx = np.arange(float(n))
    xx -= np.mean(xx)
    xx /= np.std(xx)

    inds = np.full((k,), n + 1, dtype=np.int64)
    inds[0] = rg.integers(n)

    dist = xx - xx[inds[0]]
    dist *= dist

    scratch = np.empty_like(dist)

    for j in range(1, k):
        p = np.divide(dist, dist.sum(), out=scratch)
        inds[j] = rg.choice(n, p=p)
        dj = np.subtract(xx, xx[inds[j]], out=scratch)
        dj *= dj
        np.minimum(dist, dj, out=dist)

    assert np.unique(inds).shape == (k,)
    return inds
