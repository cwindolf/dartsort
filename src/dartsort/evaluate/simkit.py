from logging import getLogger
from pathlib import Path
import pickle
import warnings

from dredge import motion_util
import h5py
import numpy as np
import pandas as pd
from scipy.signal import sawtooth
from spikeinterface.core import read_binary_folder
from spikeinterface.core.recording_tools import get_chunk_with_margin
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)
import torch
from tqdm.auto import tqdm

from ..templates import TemplateData
from ..util.data_util import (
    DARTsortSorting,
    extract_random_snips,
    resolve_path,
    divide_randomly,
)
from ..util.multiprocessing_util import get_pool
from ..util.drift_util import registered_geometry
from ..util.spiketorch import ptp
from ..util.waveform_util import make_channel_index
from .simlib import simulate_sorting, add_features, default_sim_featurization_cfg
from .simlib import default_temporal_kernel_npy
from .noise_recording_tools import get_background_recording
from .sim_template_tools import get_template_simulator


logger = getLogger(__name__)


def generate_simulation(
    folder,
    noise_recording_folder,
    n_units,
    duration_seconds,
    # probe parameters
    probe_kwargs=None,
    # sorting parameters
    firing_kind="uniform",
    min_fr_hz=1.0,
    max_fr_hz=20.0,
    # noise args
    noise_kind="stationary_factorized_rbf",
    noise_spatial_kernel_bandwidth=15.0,
    noise_temporal_kernel: np.ndarray | str | Path = default_temporal_kernel_npy,
    noise_fft_t=121,
    white_noise_scale=1.0,
    # template args
    templates_kind="3exp",
    template_library=None,
    template_simulator_kwargs=None,
    # general parameters
    drift_type="triangle",
    drift_speed=0.0,
    drift_period=30.0,
    temporal_jitter=16,
    amplitude_jitter=0.05,
    amp_jitter_family="uniform",
    common_reference=True,
    sampling_frequency=30_000.0,
    refractory_ms=1.0,
    globally_refractory=False,
    extract_radius=100.0,
    recording_dtype="float16",
    features_dtype="float32",
    featurization_cfg=default_sim_featurization_cfg,
    # control
    max_drift_per_chunk=0.5,
    max_chunk_len_s=1.0,
    random_seed=0,
    overwrite=False,
    n_jobs=1,
    no_save=False,
    just_noise=False,
):
    if folder is not None and not (overwrite or just_noise or no_save):
        try:
            return load_simulation(folder)
        except Exception:
            pass

    noise_recording_folder = resolve_path(noise_recording_folder)
    duration_samples = int(duration_seconds * sampling_frequency)
    with warnings.catch_warnings(record=True) as ws:
        noise_recording = get_background_recording(
            noise_recording_folder,
            duration_samples=duration_samples,
            probe_kwargs=probe_kwargs,
            noise_kind=noise_kind,
            noise_spatial_kernel_bandwidth=noise_spatial_kernel_bandwidth,
            noise_temporal_kernel=noise_temporal_kernel,
            random_seed=random_seed,
            dtype=recording_dtype,
            noise_fft_t=noise_fft_t,
            white_noise_scale=white_noise_scale,
            sampling_frequency=sampling_frequency,
            n_jobs=n_jobs,
            overwrite=overwrite,
        )
        for w in ws:
            msg = str(w.message)
            if msg.startswith("The extractor is not serializable "):
                continue
            if msg.startswith("auto_cast_uint"):
                continue
            raise w
    assert noise_recording.dtype == np.dtype(recording_dtype)
    assert noise_recording.sampling_frequency == sampling_frequency
    assert noise_recording.get_num_frames() == duration_samples
    if just_noise:
        return

    folder = resolve_path(folder)

    template_simulator = get_template_simulator(
        n_units=n_units,
        templates_kind=templates_kind,
        template_library=template_library,
        geom=noise_recording.get_channel_locations(),
        sampling_frequency=sampling_frequency,
        common_reference=common_reference,
        random_seed=random_seed,
        temporal_jitter=temporal_jitter,
        **(template_simulator_kwargs or {}),
    )
    sim_recording = InjectSpikesPreprocessor(
        noise_recording,
        firing_kind=firing_kind,
        min_fr_hz=min_fr_hz,
        max_fr_hz=max_fr_hz,
        template_simulator=template_simulator,
        drift_type=drift_type,
        drift_speed=drift_speed,
        drift_period=drift_period,
        amplitude_jitter=amplitude_jitter,
        temporal_jitter=temporal_jitter,
        random_seed=random_seed,
        refractory_ms=refractory_ms,
        globally_refractory=globally_refractory,
        amp_jitter_family=amp_jitter_family,
        extract_radius=extract_radius,
        features_dtype=features_dtype,
    )
    if no_save:
        return sim_recording, template_simulator

    if drift_speed is None:
        chunk_len_s = max_chunk_len_s
    else:
        drift_per_chunk = drift_speed * max_chunk_len_s
        chunk_len_s = min(
            max_chunk_len_s, max_drift_per_chunk / max(drift_per_chunk, 1e-10)
        )

    sim_recording.save_simulation(
        folder,
        overwrite=overwrite,
        n_jobs=n_jobs,
        featurization_cfg=featurization_cfg,
        chunk_len_s=chunk_len_s,
    )
    return load_simulation(folder)


def load_simulation(folder):
    folder = resolve_path(folder, strict=True)
    recording_dir = folder / "recording"
    templates_npz = folder / "templates.npz"
    sorting_h5 = folder / "dartsort_sorting.h5"
    motion_est_pkl = folder / "motion_est.pkl"
    unit_info_csv = folder / "unit_information.csv"

    recording = read_binary_folder(recording_dir)
    templates = TemplateData.from_npz(templates_npz)
    sorting = DARTsortSorting.from_peeling_hdf5(sorting_h5)
    with open(motion_est_pkl, "rb") as jar:
        motion_est = pickle.load(jar)
    unit_info_df = pd.read_csv(unit_info_csv)

    return dict(
        recording=recording,
        templates=templates,
        sorting=sorting,
        motion_est=motion_est,
        unit_info_df=unit_info_df,
    )


class InjectSpikesPreprocessor(BasePreprocessor):
    def __init__(self, recording, **simulation_kwargs):
        super().__init__(recording)

        self._serializability["json"] = False
        self._serializability["pickle"] = False

        assert len(recording._recording_segments) == 1
        self.add_recording_segment(
            InjectSpikesPreprocessorSegment(
                recording._recording_segments[0],
                n_channels=self.get_num_channels(),
                **simulation_kwargs,
            )
        )
        self.segment: InjectSpikesPreprocessorSegment = self._recording_segments[0]

    def drift(self, t_samples):
        return self.segment.drift(t_samples)

    def templates(self, t_samples=None, up=False):
        return self.segment.templates(t_samples, up)

    def motion_estimate(self):
        if not self.segment.drift_speed:
            return None

        duration_s = np.ceil(self.get_duration())
        t = np.arange(duration_s)
        time_bin_centers = t + 0.5 * np.diff(t).mean()
        tbc_samples = time_bin_centers * self.sampling_frequency
        displacement = self.drift(tbc_samples)
        return motion_util.get_motion_estimate(
            displacement=displacement, time_bin_centers_s=time_bin_centers
        )

    def registered_geom(self):
        me = self.motion_estimate()
        geom = self.get_channel_locations()
        rgeom = registered_geometry(geom, motion_est=me)
        matches = np.square(geom[None] - rgeom[:, None]).sum(2).argmin(0)
        return rgeom, matches

    def template_data(self, hdf5_path=None):
        if not self.segment.drift_speed:
            return TemplateData(
                templates=self.templates()[1],
                unit_ids=np.arange(self.segment.n_units),
                spike_counts=np.ones(self.segment.n_units),
                registered_geom=self.get_channel_locations(),
                trough_offset_samples=self.segment.trough_offset_samples,
            )

        rgeom, matches = self.registered_geom()
        _, templates = self.templates()
        rtemplates = np.zeros(
            (*templates.shape[:-1], len(rgeom)), dtype=templates.dtype
        )
        rtemplates[:, :, matches] = templates
        return TemplateData(
            templates=rtemplates,
            unit_ids=np.arange(self.segment.n_units),
            spike_counts=np.ones(self.segment.n_units),
            registered_geom=rgeom,
            trough_offset_samples=self.segment.trough_offset_samples,
        )

    def gt_unit_information(self):
        ids = np.arange(self.segment.n_units)
        pos, templates = self.templates()
        x, y, z = pos.T
        counts = np.zeros(self.segment.n_units, dtype=np.int64)
        u, c = np.unique(self.segment.labels, return_counts=True)
        counts[u] = c
        df = dict(
            gt_unit_id=ids,
            x=x,
            y=y,
            z=z,
            ptp_amplitude=np.ptp(templates, axis=1).max(1),
            template_norm=np.linalg.norm(templates, axis=(1, 2)),
            gt_spike_count=counts,
            gt_fr_hz=counts / self.get_total_duration(),
        )
        return pd.DataFrame(df)

    def save_features_to_hdf5(
        self,
        hdf5_path,
        overwrite=False,
        n_jobs=1,
        show_progress=True,
        n_residual_snips=4096,
        chunk_len_s=0.5,
    ):
        if overwrite:
            if hdf5_path.exists():
                hdf5_path.unlink()
        else:
            assert not hdf5_path.exists()

        n_jobs, Executor, context = get_pool(n_jobs, cls="ThreadPoolExecutor")
        with Executor(max_workers=n_jobs, mp_context=context) as pool:
            nt = self.get_num_frames()
            bs = int(self.sampling_frequency * chunk_len_s)
            chunk_starts = range(0, nt, bs)
            residual_snips_per_chunk = divide_randomly(
                n_residual_snips, len(chunk_starts), self.segment.random_seed
            )
            jobs = (
                ((t, min(t + bs, nt)), dict(extract=True, n_residual_snips=nrs))
                for t, nrs in zip(chunk_starts, residual_snips_per_chunk)
            )
            with h5py.File(hdf5_path, "w", locking=False) as h5:
                n = self.segment.n_spikes

                # fixed arrays
                h5.create_dataset("sampling_frequency", data=self.sampling_frequency)
                h5.create_dataset("geom", data=self.get_channel_locations())
                h5.create_dataset(
                    "channel_index", data=self.segment.extract_channel_index
                )
                h5.create_dataset("times_samples", data=self.segment.times_samples)
                h5.create_dataset(
                    "times_seconds",
                    data=self.sample_index_to_time(self.segment.times_samples),
                )
                h5.create_dataset("labels", data=self.segment.labels)
                h5.create_dataset("scalings", data=self.segment.scalings)
                h5.create_dataset("jitter_ix", data=self.segment.jitter_ix)

                # arrays discovered in batches below
                f_dt = self.segment.features_dtype
                dataset_shapes = {
                    "localizations": ((3,), f_dt),
                    "displacements": ((), f_dt),
                    "ptp_amplitudes": ((), f_dt),
                    "channels": ((), np.int32),
                    "collisioncleaned_waveforms": (self.segment.inj_wf_shape, f_dt),
                }
                datasets = {
                    k: h5.create_dataset(k, dtype=dt, shape=(n, *sh))
                    for k, (sh, dt) in dataset_shapes.items()
                }

                # residual snippets
                residual = h5.create_dataset(
                    "residual",
                    shape=(n_residual_snips, *self.segment.wf_shape),
                    dtype=f_dt,
                )
                residual_times = h5.create_dataset(
                    "residual_times_seconds", shape=(n_residual_snips,), dtype=f_dt
                )

                results = pool.map(self.segment._get_traces_and_inject_spikes_job, jobs)
                if show_progress:
                    results = tqdm(
                        results,
                        total=len(chunk_starts),
                        desc="Extract GT features",
                        smoothing=0.02,
                    )

                i1_prev = 0
                n_injected = 0
                resid_ix = 0
                for res in results:
                    assert res is not None
                    _, s = res
                    del res, _
                    i0, i1 = s["i0"], s["i1"]
                    assert i0 == i1_prev
                    i1_prev = i1
                    n_injected += i1 - i0

                    for k, ds in datasets.items():
                        ds[i0:i1] = s[k]

                    nrs = s["n_residual_snips"]
                    if not nrs:
                        continue
                    residual[resid_ix : resid_ix + nrs] = s["residual"]
                    residual_times[resid_ix : resid_ix + nrs] = s["residual_times"]
                    resid_ix += nrs
                assert i1_prev == n
            assert n_injected == n

    def save_simulation(
        self,
        folder,
        overwrite=False,
        n_jobs=1,
        featurization_cfg=default_sim_featurization_cfg,
        chunk_len_s=0.5,
    ):
        folder = resolve_path(folder)
        folder.mkdir(exist_ok=True)
        recording_dir = folder / "recording"
        templates_npz = folder / "templates.npz"
        sorting_h5 = folder / "dartsort_sorting.h5"
        motion_est_pkl = folder / "motion_est.pkl"
        unit_info_csv = folder / "unit_information.csv"

        with warnings.catch_warnings(record=True) as ws:
            recording = self.save_to_folder(
                folder=recording_dir,
                overwrite=overwrite,
                n_jobs=n_jobs,
                pool_engine="thread",
                chunk_duration=chunk_len_s,
            )
            for w in ws:
                msg = str(w.message)
                if msg.startswith("The extractor is not serializable "):
                    continue
                if msg.startswith("auto_cast_uint"):
                    continue
                raise w
        n_residual_snips = (
            0 if featurization_cfg is None else featurization_cfg.n_residual_snips
        )
        self.save_features_to_hdf5(
            sorting_h5,
            n_jobs=n_jobs,
            overwrite=overwrite,
            n_residual_snips=n_residual_snips,
            chunk_len_s=chunk_len_s,
        )
        if featurization_cfg is not None and not featurization_cfg.skip:
            # this is only for the TPCA feature.
            torch.manual_seed(self.segment.random_seed)
            add_features(sorting_h5, recording, featurization_cfg)

        self.gt_unit_information().to_csv(unit_info_csv)
        with open(motion_est_pkl, "wb") as jar:
            pickle.dump(self.motion_estimate(), jar)
        self.template_data(sorting_h5).to_npz(templates_npz)


class InjectSpikesPreprocessorSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        n_channels,
        *,
        firing_kind,
        min_fr_hz,
        max_fr_hz,
        template_simulator,
        drift_type,
        drift_speed,
        drift_period,
        amplitude_jitter,
        amp_jitter_family,
        temporal_jitter,
        random_seed,
        refractory_ms,
        globally_refractory,
        extract_radius,
        features_dtype="float32",
    ):
        super().__init__(parent_recording_segment)
        assert self.sampling_frequency is not None
        assert drift_type in ("line", "triangle")

        self.drift_type = drift_type
        self.drift_speed = drift_speed
        self.drift_period = drift_period
        self.temporal_jitter = temporal_jitter
        self.features_dtype = features_dtype
        self.template_simulator = template_simulator
        self.refractory_samples = int(
            refractory_ms * (self.sampling_frequency / 1000.0)
        )

        # shapes
        self.n_units = self.template_simulator.n_units
        self.n_channels = n_channels
        self.chans_arange = np.arange(n_channels)
        self.spike_length_samples = self.template_simulator.spike_length_samples()
        self.trough_offset_samples = self.template_simulator.trough_offset_samples()
        self.snippet_time_ix = np.arange(
            -self.trough_offset_samples,
            self.spike_length_samples - self.trough_offset_samples,
        )
        self.margin = max(
            self.trough_offset_samples,
            self.spike_length_samples - self.trough_offset_samples,
        )
        self.extract_channel_index = make_channel_index(
            self.template_simulator.geom, extract_radius
        )
        self.wf_shape = (self.spike_length_samples, self.n_channels)
        self.inj_wf_shape = (
            self.spike_length_samples,
            self.extract_channel_index.shape[1],
        )
        self.template_shape = (self.n_units, *self.wf_shape)
        self.template_shape_up = (self.n_units, temporal_jitter, *self.wf_shape)

        # bake all random stuff
        rg = np.random.default_rng(random_seed)
        if firing_kind == "uniform":
            firing_rates = rg.uniform(min_fr_hz, max_fr_hz, size=self.n_units)
        else:
            assert False
        self.random_seed = random_seed
        self.sorting = simulate_sorting(
            self.n_units,
            self.get_num_samples(),
            firing_rates=firing_rates,
            rg=rg,
            nbefore=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            sampling_frequency=self.sampling_frequency,
            refractory_samples=self.refractory_samples,
            globally_refractory=globally_refractory,
        )
        sv: np.recarray = self.sorting.to_spike_vector()
        self.times_samples = sv["sample_index"]
        self.labels = sv["unit_index"]
        self.n_spikes = self.sorting.count_total_num_spikes()

        if amplitude_jitter and amp_jitter_family == "gamma":
            alpha = 1.0 / amplitude_jitter**2
            theta = amplitude_jitter**2
            self.scalings = rg.gamma(shape=alpha, scale=theta, size=self.n_spikes)
        elif amplitude_jitter and amp_jitter_family == "uniform":
            self.scalings = rg.uniform(
                1.0 - amplitude_jitter, 1.0 + amplitude_jitter, size=self.n_spikes
            )
        else:
            assert not amplitude_jitter
            self.scalings = np.ones(1, dtype=features_dtype)
            self.scalings = np.broadcast_to(self.scalings, (self.n_spikes,))

        if self.temporal_jitter > 1:
            self.jitter_ix = rg.integers(self.temporal_jitter, size=self.n_spikes)
        else:
            self.jitter_ix = np.zeros(1, dtype=np.int64)
            self.jitter_ix = np.broadcast_to(self.jitter_ix, (self.n_spikes,))

    def drift(self, t_samples):
        if not self.drift_speed:
            return np.zeros_like(t_samples)

        if self.drift_type == "line":
            t_center = self.get_num_samples() / 2
            dt = (t_samples - t_center) / self.sampling_frequency
            return dt * self.drift_speed

        if self.drift_type == "triangle":
            t_seconds = self.sample_index_to_time(t_samples)
            phase = t_seconds * (2 * np.pi / self.drift_period)
            wave = sawtooth(phase, width=0.5)
            # -1 to 1 and back to -1, so divide by 4 to have 2*ptp=drift_speed*drift_period.
            return wave * (self.drift_speed * self.drift_period / 4.0)

        assert False

    def templates(self, t_samples=None, up=False, padded=False, pad_value=np.nan):
        drift = 0 if t_samples is None else self.drift(t_samples)
        pos, templates = self.template_simulator.templates(
            drift=drift, up=up, padded=padded, pad_value=pad_value
        )
        templates = templates.astype(self.features_dtype)
        tunpad = templates[..., :-1] if padded else templates
        if up:
            assert tunpad.shape == self.template_shape_up
        else:
            assert tunpad.shape == self.template_shape

        return pos, templates

    def get_spikes(
        self,
        noise_with_margin,
        start_frame,
        end_frame,
        extract=False,
        n_residual_snips=0,
    ):
        i0 = np.searchsorted(self.times_samples, start_frame)
        i1 = np.searchsorted(self.times_samples, end_frame)
        t = self.times_samples[i0:i1]
        l = self.labels[i0:i1]
        s = self.scalings[i0:i1]
        u = self.jitter_ix[i0:i1]

        # temporal indices of snippets relative to chunk
        t_rel = t - (start_frame - self.margin)
        tix = t_rel[:, None] + self.snippet_time_ix
        tc = (start_frame + end_frame) / 2
        pos, temps = self.templates(t_samples=tc, up=True, padded=extract)
        temps = temps[l, u]
        temps *= s[:, None, None]
        temps_unpad = temps[..., :-1] if extract else temps

        spikes = dict(
            i0=i0,
            i1=i1,
            times_samples=t,
            labels=l,
            scalings=s,
            jitter_ix=u,
            tix=tix,
            waveforms=temps_unpad,
            n_residual_snips=n_residual_snips,
        )
        if not extract:
            return spikes

        spikes["localizations"] = pos[l]
        spikes["displacements"] = np.full(
            i1 - i0, self.drift(tc), dtype=self.features_dtype
        )
        ptp_vectors = ptp(temps_unpad, dim=1)
        c = ptp_vectors.argmax(axis=1)
        spikes["channels"] = c
        spikes["ptp_amplitudes"] = ptp_vectors.max(axis=1)
        if n_residual_snips:
            rsnips, rtimes = extract_random_snips(
                self.random_seed,
                noise_with_margin[self.margin : len(noise_with_margin) - self.margin],
                n=n_residual_snips,
                sniplen=self.spike_length_samples,
            )
            spikes["residual"] = rsnips
            rtimes += start_frame
            rtimes = self.sample_index_to_time(rtimes)
            spikes["residual_times"] = rtimes

        # extract the background noise which waveforms will be added into
        noise_padded = np.pad(
            noise_with_margin, [(0, 0), (0, 1)], constant_values=np.nan
        )
        echans = self.extract_channel_index[c]
        collisioncleaned_waveforms = noise_padded[tix[:, :, None], echans[:, None, :]]
        # the actual injected waveforms...
        injected_waveforms = np.take_along_axis(temps, echans[:, None, :], axis=2)
        collisioncleaned_waveforms += injected_waveforms
        spikes["collisioncleaned_waveforms"] = collisioncleaned_waveforms
        return spikes

    def get_traces_and_inject_spikes(
        self,
        start_frame,
        end_frame,
        channel_indices=None,
        extract=False,
        inject=False,
        n_residual_snips=0,
    ):
        traces, lm, rm = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            margin=self.margin,
            add_zeros=True,
            channel_indices=None,
        )
        assert lm == rm == self.margin
        assert traces.shape[1] == self.n_channels
        spikes = self.get_spikes(
            traces,
            start_frame,
            end_frame,
            extract=extract,
            n_residual_snips=n_residual_snips,
        )

        if not inject:
            return traces, spikes

        waveforms = spikes["waveforms"].astype(traces.dtype, copy=False)
        traces = traces.copy()
        np.add.at(
            traces,
            (spikes["tix"][:, :, None], self.chans_arange[None, None]),
            waveforms,
        )

        traces = traces[self.margin : len(traces) - self.margin]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces, spikes

    def _get_traces_and_inject_spikes_job(self, args_kwargs):
        args, kwargs = args_kwargs
        return self.get_traces_and_inject_spikes(*args, **kwargs)

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces, _ = self.get_traces_and_inject_spikes(
            start_frame, end_frame, channel_indices, inject=True
        )
        return traces
