import tempfile
from concurrent.futures import CancelledError
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch
from dartsort.transform import WaveformPipeline
from dartsort.util import peel_util
from dartsort.util.data_util import SpikeDataset
from dartsort.util.multiprocessing_util import get_pool
from dartsort.util.py_util import delay_keyboard_interrupt
from spikeinterface.core.recording_tools import get_chunk_with_margin
from tqdm.auto import tqdm


class BasePeeler(torch.nn.Module):
    """Base class for peeling operations (subtraction, deconv, etc)

    Subtraction, deconv, and other things like just grabbing waveforms
    and featurizing a preexisting spike train (implemented with optional
    additional peeling of preexisting templates in grab.py) share a lot
    of logic for featurization and parallelization and data saving/loading.
    This class handles all of that stuff so that it can be shared.
    """

    peel_kind = ""

    def __init__(
        self,
        recording,
        channel_index,
        featurization_pipeline=None,
        chunk_length_samples=30_000,
        chunk_margin_samples=0,
        n_chunks_fit=40,
        max_waveforms_fit=50_000,
        n_waveforms_fit=20_000,
        fit_max_reweighting=20.0,
        fit_sampling="random",
        fit_subsampling_random_state=0,
        trough_offset_samples=42,
        spike_length_samples=121,
        dtype=torch.float,
    ):
        if recording.get_num_segments() > 1:
            raise ValueError("Peeling does not yet support multi-segment recordings.")
        super().__init__()
        self.recording = recording
        self.chunk_length_samples = chunk_length_samples
        self.chunk_margin_samples = chunk_margin_samples
        self.n_chunks_fit = n_chunks_fit
        self.max_waveforms_fit = max_waveforms_fit
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.fit_subsampling_random_state = np.random.default_rng(
            fit_subsampling_random_state
        )
        self.dtype = dtype
        self.np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        if channel_index is not None:
            self.register_buffer("channel_index", channel_index)
            assert recording.get_num_channels() == channel_index.shape[0]
        self.n_waveforms_fit = n_waveforms_fit
        self.fit_sampling = fit_sampling
        self.fit_max_reweighting = fit_max_reweighting
        if featurization_pipeline is not None:
            self.add_module("featurization_pipeline", featurization_pipeline)
        else:
            self.featurization_pipeline = None

        # subclasses can append to this if they want to store more fixed
        # arrays in the output h5 file
        self.fixed_output_data = [
            ("sampling_frequency", self.recording.get_sampling_frequency()),
            ("geom", self.recording.get_channel_locations()),
        ]
        if channel_index is not None:
            self.fixed_output_data.append(
                ("channel_index", self.channel_index.numpy(force=True).copy()),
            )

    # -- main functions for users to call
    # in practice users will interact with the functions `subtract(...)` in
    # subtract.py and similar functions in the other .py files here, but these
    # are the main API methods for this class

    def load_or_fit_and_save_models(
        self, save_folder, overwrite=False, n_jobs=0, device=None
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
            if self.needs_precompute():
                self.precompute_models()
            if self.needs_fit():
                save_folder.mkdir(exist_ok=True)
                self.fit_models(
                    save_folder, overwrite=overwrite, n_jobs=n_jobs, device=device
                )
            self.save_models(save_folder)
        assert not self.needs_precompute()
        assert not self.needs_fit()

    def peel(
        self,
        output_hdf5_filename,
        chunk_starts_samples=None,
        chunk_length_samples=None,
        stop_after_n_waveforms=None,
        n_jobs=0,
        overwrite=False,
        residual_filename=None,
        show_progress=True,
        skip_features=False,
        residual_to_h5=False,
        task_name=None,
        device=None,
        ignore_resuming=False,
    ):
        """Run the full (already fitted) peeling and featurization pipeline

        This gathers all results into an hdf5 file, resuming from where
        it left off in that file if overwrite=False.
        """
        if self.needs_fit():
            raise ValueError("Peeler needs to be fitted before peeling.")

        if task_name is None:
            task_name = self.peel_kind

        # this is -1 if we haven't started yet
        if ignore_resuming:
            last_chunk_start = 0
        else:
            last_chunk_start = self.check_resuming(
                output_hdf5_filename,
                overwrite=overwrite,
            )

        # figure out which chunks to process, and exit early if already done
        chunk_starts_samples = self.get_chunk_starts(
            chunk_starts_samples=chunk_starts_samples,
        )
        n_chunks_orig = len(chunk_starts_samples)
        chunks_to_do = [
            start for start in chunk_starts_samples if start > last_chunk_start
        ]
        if not chunks_to_do:
            return output_hdf5_filename
        save_residual = residual_filename is not None
        compute_residual = save_residual or residual_to_h5
        if chunk_length_samples is None:
            chunk_length_samples = self.chunk_length_samples

        # main peeling loop
        # wrap in try/finally to ensure file handles get closed if there
        # is some unforseen error
        try:
            n_jobs, Executor, context, rank_queue = get_pool(
                n_jobs, with_rank_queue=True
            )
            with Executor(
                max_workers=n_jobs,
                mp_context=context,
                initializer=_peeler_process_init,
                initargs=(
                    self,
                    device,
                    rank_queue,
                    compute_residual,
                    skip_features,
                    chunk_length_samples,
                ),
            ) as pool:
                # launch the jobs and wrap in a progress bar
                results = pool.map(_peeler_process_job, chunks_to_do)
                if show_progress:
                    n_sec_chunk = (
                        chunk_length_samples / self.recording.get_sampling_frequency()
                    )
                    results = tqdm(
                        results,
                        total=n_chunks_orig,
                        initial=n_chunks_orig - len(chunks_to_do),
                        smoothing=0.01,
                        desc=f"{task_name} {n_sec_chunk:.1f}s/it [spk/it=%%%]",
                    )

                # construct h5 after forking to avoid pickling it
                with self.initialize_files(
                    output_hdf5_filename,
                    residual_filename=residual_filename,
                    overwrite=overwrite,
                    skip_features=skip_features,
                    residual_to_h5=residual_to_h5,
                ) as (
                    output_h5,
                    h5_spike_datasets,
                    residual_file,
                    n_spikes,
                ):
                    batch_count = 0
                    try:
                        for result, chunk_start_samples in zip(results, chunks_to_do):
                            n_new_spikes = self.gather_chunk_result(
                                n_spikes,
                                chunk_start_samples,
                                result,
                                h5_spike_datasets,
                                output_h5,
                                residual_file,
                                residual_to_h5,
                                ignore_resuming,
                                skip_features,
                            )
                            batch_count += 1
                            n_spikes += n_new_spikes
                            if show_progress:
                                desc = f"{task_name}"
                                if not skip_features:
                                    desc += f" [spk/{n_sec_chunk:g}s={n_spikes / batch_count:0.1f}]"
                                results.set_description(desc)
                            if not skip_features and (
                                stop_after_n_waveforms
                                and n_spikes >= stop_after_n_waveforms
                            ):
                                pool.shutdown(cancel_futures=True)
                    except CancelledError:
                        if show_progress:
                            results.write(
                                f"Got {n_spikes} spikes, enough to stop early."
                            )
        finally:
            self.to("cpu")

        return output_hdf5_filename

    # -- methods for subclasses to override

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
    ):
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

    def peeling_needs_fit(self):
        return False

    def peeling_needs_precompute(self):
        return False

    def precompute_peeling_data(
        self, save_folder, overwrite=False, n_jobs=0, device=None
    ):
        # subclasses should override if they need to cache data for peeling
        # runs before fit_peeler_models()
        pass

    def fit_peeler_models(self, save_folder, tmp_dir=None, n_jobs=0, device=None):
        # subclasses should override if they need to fit models for peeling
        assert not self.peeling_needs_fit()

    def precompute_peeler_models(self, save_folder, n_jobs=0, device=None):
        # subclasses should override if they need to fit models for peeling
        assert not self.peeling_needs_precompute()

    # subclasses can add to this list
    # for each dataset in this list, an output dataset in the
    # hdf5 (of .peel()) will be created with this dtype and with
    # shape (N_spikes, *shape_per_spike)
    # datasets will also be created corresponding to features
    # in featurization_pipeline
    def out_datasets(self):
        datasets = [
            SpikeDataset(name="times_samples", shape_per_spike=(), dtype=int),
            SpikeDataset(name="times_seconds", shape_per_spike=(), dtype=float),
            SpikeDataset(name="channels", shape_per_spike=(), dtype=int),
        ]
        if self.featurization_pipeline is not None:
            for transformer in self.featurization_pipeline.transformers:
                if transformer.is_featurizer:
                    datasets.extend(transformer.spike_datasets)
        return datasets

    # -- utility methods which users likely won't touch

    def featurize_collisioncleaned_waveforms(
        self, collisioncleaned_waveforms, max_channels
    ):
        if self.featurization_pipeline is None:
            return {}

        waveforms, features = self.featurization_pipeline(
            collisioncleaned_waveforms, max_channels
        )
        return features

    def process_chunk(
        self,
        chunk_start_samples,
        chunk_end_samples=None,
        return_residual=False,
        skip_features=False,
    ):
        """Grab, peel, and featurize a chunk, returning a dict of numpy arrays

        Main function called in peeling workers
        """
        if chunk_end_samples is None:
            chunk_end_samples = chunk_start_samples + self.chunk_length_samples
        chunk_end_samples = min(self.recording.get_num_samples(), chunk_end_samples)
        chunk, left_margin, right_margin = get_chunk_with_margin(
            self.recording._recording_segments[0],
            start_frame=chunk_start_samples,
            end_frame=chunk_end_samples,
            channel_indices=None,
            margin=self.chunk_margin_samples,
        )
        chunk = torch.tensor(
            chunk,
            device=self.channel_index.device,
            dtype=self.dtype,
        )
        peel_result = self.peel_chunk(
            chunk,
            chunk_start_samples=chunk_start_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            return_residual=return_residual,
        )

        if peel_result["n_spikes"] > 0 and not skip_features:
            features = self.featurize_collisioncleaned_waveforms(
                peel_result["collisioncleaned_waveforms"],
                peel_result["channels"],
            )
        else:
            features = {}

        # a user who wants these must featurize with a waveform node
        # keeping them below adds overhead
        del peel_result["collisioncleaned_waveforms"]

        assert not any(k in features for k in peel_result)
        chunk_result = {**peel_result, **features}
        chunk_result = {
            k: v.numpy(force=True) if torch.is_tensor(v) else v
            for k, v in chunk_result.items()
        }

        # add times in seconds
        segment = self.recording._recording_segments[0]
        chunk_result["chunk_start_seconds"] = segment.sample_index_to_time(chunk_start_samples)
        if peel_result["n_spikes"]:
            chunk_result["times_seconds"] = segment.sample_index_to_time(
                chunk_result["times_samples"]
            )
        return chunk_result

    def gather_chunk_result(
        self,
        cur_n_spikes,
        chunk_start_samples,
        chunk_result,
        h5_spike_datasets,
        output_h5,
        residual_file,
        residual_to_h5,
        ignore_resuming,
        skip_features,
    ):
        # delay keyboard interrupts so we don't write half a batch
        # of data and leave files in an invalid state after ^C
        # not that something else couldn't happen...
        with delay_keyboard_interrupt:
            if not ignore_resuming:
                output_h5["last_chunk_start"][()] = chunk_start_samples

            if residual_file is not None:
                chunk_result["residual"].tofile(residual_file)

            if residual_to_h5 and "residual" in chunk_result:
                n_residuals = len(output_h5["residual"])
                output_h5["residual"].resize(n_residuals + 1, axis=0)
                output_h5["residual"][n_residuals:] = chunk_result["residual"]
                output_h5["residual_times_seconds"].resize(n_residuals + 1, axis=0)
                output_h5["residual_times_seconds"][n_residuals:] = chunk_result["chunk_start_seconds"]

            if skip_features:
                return 0

            n_new_spikes = chunk_result["n_spikes"]
            if not n_new_spikes:
                return 0

            for ds in self.out_datasets():
                h5ds = h5_spike_datasets[ds.name]
                h5ds.resize(cur_n_spikes + n_new_spikes, axis=0)
                h5ds[cur_n_spikes:] = chunk_result[ds.name]

        return n_new_spikes

    def needs_fit(self):
        it_does = self.peeling_needs_fit()
        if self.featurization_pipeline is not None:
            it_does = it_does or self.featurization_pipeline.needs_fit()
        return it_does

    def needs_precompute(self):
        it_does = self.peeling_needs_precompute()
        if self.featurization_pipeline is not None:
            it_does = it_does or self.featurization_pipeline.needs_precompute()
        return it_does

    def fit_models(
        self, save_folder, tmp_dir=None, overwrite=False, n_jobs=0, device=None
    ):
        with torch.no_grad():
            if self.peeling_needs_fit():
                self.precompute_peeling_data(
                    save_folder=save_folder,
                    overwrite=overwrite,
                    n_jobs=n_jobs,
                    device=device,
                )
                self.fit_peeler_models(
                    save_folder=save_folder,
                    tmp_dir=tmp_dir,
                    n_jobs=n_jobs,
                    device=device,
                )
            self.fit_featurization_pipeline(
                save_folder=save_folder, tmp_dir=tmp_dir, n_jobs=n_jobs, device=device
            )
        assert not self.needs_fit()

    def precompute_models(self):
        if self.peeling_needs_precompute():
            self.precompute_peeler_models()
        if self.featurization_pipeline is None:
            return
        self.featurization_pipeline.precompute()

    def fit_featurization_pipeline(
        self, save_folder, tmp_dir=None, n_jobs=0, device=None
    ):
        if self.featurization_pipeline is None:
            return

        if not self.featurization_pipeline.needs_fit():
            return

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

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
        )

        with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
            temp_hdf5_filename = Path(temp_dir) / "peeler_fit.h5"
            try:
                self.run_subsampled_peeling(
                    temp_hdf5_filename,
                    n_jobs=n_jobs,
                    device=device,
                    task_name="Load examples for feature fitting",
                )

                # fit featurization pipeline and reassign
                # work in a try finally so we can delete the temp file
                # in case of an issue or a keyboard interrupt
                channels, waveforms = peel_util.subsample_waveforms(
                    temp_hdf5_filename,
                    fit_sampling=self.fit_sampling,
                    random_state=self.fit_subsampling_random_state,
                    n_waveforms_fit=self.n_waveforms_fit,
                    fit_max_reweighting=self.fit_max_reweighting,
                    voltages_dataset_name="peeled_voltages_fit",
                    waveforms_dataset_name="peeled_waveforms_fit",
                )

                channels = torch.as_tensor(channels, device=device)
                waveforms = torch.as_tensor(waveforms, device=device)
                featurization_pipeline = featurization_pipeline.to(device)
                featurization_pipeline.fit(waveforms, max_channels=channels, recording=self.recording)
                featurization_pipeline = featurization_pipeline.to("cpu")
                self.featurization_pipeline = featurization_pipeline
            finally:
                self.to("cpu")
                if temp_hdf5_filename.exists():
                    temp_hdf5_filename.unlink()

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

        T_samples = self.recording.get_num_samples()
        if t_end is None:
            t_end = T_samples
        if t_start is None:
            t_start = 0
        chunk_starts_samples = range(t_start, t_end, chunk_length_samples)
        if skip_last:
            chunk_starts_samples = list(chunk_starts_samples)
            if t_end - chunk_starts_samples[-1] < chunk_length_samples:
                chunk_starts_samples = chunk_starts_samples[:-1]

        if not subsampled:
            return chunk_starts_samples

        if n_chunks is None:
            n_chunks = self.n_chunks_fit

        # make a random subset of chunks to use for fitting
        rg = np.random.default_rng(self.fit_subsampling_random_state)
        self.fit_subsampling_random_state = rg
        chunk_starts_samples = rg.choice(
            chunk_starts_samples,
            size=min(len(chunk_starts_samples), n_chunks),
            replace=False,
        )
        if ordered:
            chunk_starts_samples.sort()
        return chunk_starts_samples

    def run_subsampled_peeling(
        self,
        hdf5_filename,
        n_jobs=0,
        chunk_length_samples=None,
        residual_to_h5=False,
        skip_features=False,
        ignore_resuming=False,
        n_chunks=None,
        t_start=None,
        t_end=None,
        device=None,
        task_name=None,
        overwrite=True,
        ordered=False,
        skip_last=False,
    ):
        # run peeling on these chunks to the temp folder
        chunk_starts = self.get_chunk_starts(
            subsampled=True,
            chunk_length_samples=chunk_length_samples,
            t_start=t_start,
            t_end=t_end,
            n_chunks=n_chunks,
            ordered=ordered,
            skip_last=skip_last,
        )
        self.peel(
            hdf5_filename,
            chunk_starts_samples=chunk_starts,
            chunk_length_samples=chunk_length_samples,
            stop_after_n_waveforms=self.max_waveforms_fit,
            ignore_resuming=ignore_resuming,
            residual_to_h5=residual_to_h5,
            skip_features=skip_features,
            n_jobs=n_jobs,
            overwrite=overwrite,
            task_name=task_name,
            device=device,
        )

    def save_models(self, save_folder):
        torch.save(
            self.featurization_pipeline,
            Path(save_folder) / "featurization_pipeline.pt",
        )

    def load_models(self, save_folder):
        if not save_folder.exists():
            return

        feats_pt = Path(save_folder) / "featurization_pipeline.pt"
        if feats_pt.exists():
            self.featurization_pipeline = torch.load(feats_pt)

    def check_resuming(self, output_hdf5_filename, overwrite=False):
        output_hdf5_filename = Path(output_hdf5_filename)
        exists = output_hdf5_filename.exists()
        last_chunk_start = -1
        if exists and not overwrite:
            with h5py.File(output_hdf5_filename, "r") as h5:
                last_chunk_start = h5["last_chunk_start"][()]
        return last_chunk_start

    @contextmanager
    def initialize_files(
        self,
        output_hdf5_filename,
        residual_filename=None,
        overwrite=False,
        chunk_size=1024,
        # could also do:
        # libver=("earliest", "v110"),
        libver="latest",
        residual_to_h5=False,
        skip_features=False,
    ):
        """Create, overwrite, or re-open output files"""
        output_hdf5_filename = Path(output_hdf5_filename)
        exists = output_hdf5_filename.exists()
        n_spikes = 0
        if exists and overwrite:
            output_hdf5_filename.unlink()
            output_h5 = h5py.File(output_hdf5_filename, "w", libver=libver)
            output_h5.create_dataset("last_chunk_start", data=-1, dtype=np.int64)
        elif exists:
            # exists and not overwrite
            output_h5 = h5py.File(output_hdf5_filename, "r+", libver=libver)
            n_spikes = len(output_h5["times_samples"])
        else:
            # didn't exist, so overwrite does not matter
            output_h5 = h5py.File(output_hdf5_filename, "w", libver=libver)
            output_h5.create_dataset("last_chunk_start", data=-1, dtype=np.int64)
        last_chunk_start = output_h5["last_chunk_start"][()]

        # write some fixed arrays that are useful to have around
        for name, value in self.fixed_output_data:
            if name not in output_h5:
                output_h5.create_dataset(name, data=value)

        # create per-spike datasets
        # use chunks to support growing the dataset as we find spikes
        h5_spike_datasets = None
        if not skip_features:
            h5_spike_datasets = {}
            for ds in self.out_datasets():
                if ds.name in output_h5:
                    h5_spike_datasets[ds.name] = output_h5[ds.name]
                else:
                    h5_spike_datasets[ds.name] = output_h5.create_dataset(
                        ds.name,
                        dtype=ds.dtype,
                        shape=(n_spikes, *ds.shape_per_spike),
                        maxshape=(None, *ds.shape_per_spike),
                        chunks=(chunk_size, *ds.shape_per_spike),
                    )

        if residual_to_h5:
            if "residual" not in output_h5:
                n_chans = self.recording.get_num_channels()
                output_h5.create_dataset(
                    "residual",
                    dtype=self.np_dtype,
                    shape=(0, self.spike_length_samples, n_chans),
                    maxshape=(None, self.spike_length_samples, n_chans),
                    chunks=(chunk_size, self.spike_length_samples, n_chans),
                )
            if "residual_times_seconds" not in output_h5:
                output_h5.create_dataset(
                    "residual_times_seconds",
                    dtype=float,
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                )

        # residual file ignore/open/overwrite logic
        save_residual = residual_filename is not None
        residual_file = None
        if save_residual:
            residual_mode = "wb"
            if last_chunk_start >= 0:
                residual_mode = "ab"
                assert Path(residual_filename).exists()
            residual_file = open(residual_filename, mode=residual_mode)

        try:
            yield (
                output_h5,
                h5_spike_datasets,
                residual_file,
                n_spikes,
            )
        finally:
            self.to("cpu")
            output_h5.close()
            if save_residual:
                residual_file.close()


# -- helper functions and objects for parallelism


class PeelerProcessContext:
    def __init__(self, peeler, compute_residual, skip_features, chunk_length_samples):
        self.peeler = peeler
        self.compute_residual = compute_residual
        self.skip_features = skip_features
        self.chunk_length_samples = chunk_length_samples


# this state will be set on each process
# it means that BasePeeler.peel() itself is not thread-safe but that's ok
_peeler_process_context = None


def _peeler_process_init(
    peeler,
    device,
    rank_queue,
    compute_residual,
    skip_features,
    chunk_length_samples,
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
    peeler.to(device)
    peeler.eval()

    # update process-local variables
    _peeler_process_context = PeelerProcessContext(
        peeler, compute_residual, skip_features, chunk_length_samples
    )


def _peeler_process_job(chunk_start_samples):
    # by returning here, we are implicitly relying on pickle
    # TODO: replace with manual np.saves
    with torch.no_grad():
        chunk_end_samples = None
        chlen = _peeler_process_context.chunk_length_samples
        if chlen is not None:
            chunk_end_samples = chunk_start_samples + chlen
        return _peeler_process_context.peeler.process_chunk(
            chunk_start_samples,
            chunk_end_samples=chunk_end_samples,
            return_residual=_peeler_process_context.compute_residual,
        )
