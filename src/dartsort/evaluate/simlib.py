import gc
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import h5py
import numpy as np
import pandas as pd
import probeinterface
import torch
from scipy.spatial.distance import cdist
from spikeinterface.core import (
    BaseRecording,
    BaseSorting,
    NumpySorting,
    read_binary_folder,
)
from spikeinterface.core.recording_tools import get_chunk_with_margin
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
    BaseRecordingSegment,
)

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files  # type: ignore # ty: ignore[x]
    except ImportError as e:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.") from e

from ..templates import TemplateData
from ..transform import WaveformPipeline
from ..util.data_util import (
    DARTsortSorting,
    divide_randomly,
    ensure_path,
    extract_random_snips,
    subsample_waveforms,
    yield_chunks,
)
from ..util.internal_config import FeaturizationConfig, WaveformConfig
from ..util.logging_util import get_logger, progbar
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import get_pool
from ..util.spiketorch import ptp
from ..util.waveform_util import make_channel_index

if TYPE_CHECKING:
    from .sim_template_tools import TemplateSimulator

_can_buffer = sys.version_info >= (3, 14)
logger = get_logger(__name__)

default_temporal_kernel_npy = files("dartsort.pretrained")
default_temporal_kernel_npy = default_temporal_kernel_npy.joinpath(
    "default_temporal_kernel.npy"
)
default_temporal_kernel_npy = Path(str(default_temporal_kernel_npy))


# -- spike train sims


def refractory_poisson_spike_train(
    rate_hz,
    duration_samples,
    seed: int | np.random.Generator = 0,
    refractory_samples=40,
    trough_offset_samples=42,
    spike_length_samples=121,
    sampling_frequency=30000.0,
    overestimation=2.0,
    empty_ok=False,
):
    """Sample a refractory Poisson spike train

    Parameters
    ----------
    rate : float
        Spikes / second, well, except it'll be slower due to refractoriness.
    duration : float
    """
    rg = np.random.default_rng(seed)

    seconds_per_sample = 1.0 / sampling_frequency
    refractory_s = refractory_samples * seconds_per_sample
    duration_s = duration_samples * seconds_per_sample

    # overestimate the number of spikes needed
    overest_count = int(duration_s * rate_hz * overestimation)
    overest_count = max(50, overest_count)

    # generate interspike intervals
    intervals = rg.exponential(scale=1.0 / rate_hz, size=overest_count)
    intervals += refractory_s
    intervals_samples = np.floor(intervals * sampling_frequency).astype(int)

    # determine spike times and restrict to ones which we can actually
    # add into / read from a recording with this duration and trough offset
    spike_samples = np.cumsum(intervals_samples)
    max_spike_time = duration_samples - (spike_length_samples - trough_offset_samples)
    # check that we overestimated enough
    assert spike_samples.max() > max_spike_time, "Not enough overestimation"
    valid = spike_samples == spike_samples.clip(trough_offset_samples, max_spike_time)
    spike_samples = spike_samples[valid]
    assert empty_ok or spike_samples.size

    return spike_samples


def piecewise_refractory_poisson_spike_train(rates, bins, binsize_samples, **kwargs):
    """
    Returns a spike train with variable firing rate using refractory_poisson_spike_train().

    :param rates: list of firing rates in Hz
    :param bins: bin starting samples (same shape as rates)
    :param binsize_samples: number of samples per bin
    :param **kwargs: kwargs to feed to refractory_poisson_spike_train()
    """
    st = []
    for rate, bin in zip(rates, bins, strict=True):
        if rate < 0.05:
            continue
        binst = refractory_poisson_spike_train(
            rate,
            binsize_samples,
            overestimation=max(2.0, 50.0 / rate),
            empty_ok=True,
            **kwargs,
        )
        st.append(bin + binst)
    st = np.concatenate(st)
    return st


def simulate_sorting(
    num_units,
    n_samples,
    firing_rates=None,
    rg: int | np.random.Generator = 0,
    nbefore: int = 42,
    spike_length_samples: int = 128,
    sampling_frequency=30_000.0,
    globally_refractory=False,
    refractory_samples=40,
):
    rg = np.random.default_rng(rg)

    # Default firing rates drawn uniformly from 1-10Hz
    if firing_rates is not None and firing_rates.ndim == 1:
        assert firing_rates.shape[0] == num_units
    elif firing_rates is not None and firing_rates.ndim == 2:
        assert firing_rates.shape[1] == num_units
    elif firing_rates is None:
        firing_rates = rg.uniform(1.0, 10.0, num_units)
    else:
        raise ValueError(f"{firing_rates.shape=}")

    if firing_rates.ndim == 2:
        assert not globally_refractory
        assert int(sampling_frequency) == sampling_frequency
        assert firing_rates.shape[1] == num_units
        Tceil = np.ceil(n_samples / sampling_frequency)
        assert firing_rates.shape[0] == Tceil
        bins = np.arange(0, n_samples, sampling_frequency)
        assert firing_rates.shape[0] == bins.shape[0]

        spike_trains = [
            piecewise_refractory_poisson_spike_train(
                rates=firing_rates[:, i],
                bins=bins,
                binsize_samples=int(sampling_frequency),
                trough_offset_samples=nbefore,
                spike_length_samples=spike_length_samples,
                seed=rg,
                refractory_samples=refractory_samples,
                sampling_frequency=sampling_frequency,
            )
            for i in range(num_units)
        ]
        if num_units:
            spike_times = np.concatenate(spike_trains)
            spike_labels = np.repeat(
                np.arange(num_units),
                np.array([spike_trains[i].shape[0] for i in range(num_units)]),
            )
        else:
            spike_times = np.array([], dtype=np.int64)
            spike_labels = np.array([], dtype=np.int64)
    elif not globally_refractory:
        spike_trains = [
            refractory_poisson_spike_train(
                firing_rates[i],
                n_samples,
                trough_offset_samples=nbefore,
                spike_length_samples=spike_length_samples,
                seed=rg,
                refractory_samples=refractory_samples,
                sampling_frequency=sampling_frequency,
            )
            for i in range(num_units)
        ]
        if num_units:
            spike_times = np.concatenate(spike_trains)
            spike_labels = np.repeat(
                np.arange(num_units),
                np.array([spike_trains[i].shape[0] for i in range(num_units)]),
            )
        else:
            spike_times = np.array([], dtype=np.int64)
            spike_labels = np.array([], dtype=np.int64)
    else:
        global_rate = np.sum(firing_rates)
        spike_times = refractory_poisson_spike_train(
            global_rate,
            n_samples,
            trough_offset_samples=nbefore,
            spike_length_samples=spike_length_samples,
            seed=rg,
            refractory_samples=refractory_samples,
            sampling_frequency=sampling_frequency,
        )
        spike_times = spike_times[spike_times < n_samples - spike_length_samples]
        unit_proportions = firing_rates / global_rate
        spike_labels = rg.choice(num_units, p=unit_proportions, size=spike_times.size)

    # order = np.argsort(spike_train)
    # spike_train = spike_train[order]
    # spike_labels = spike_labels[order]

    sorting = NumpySorting.from_samples_and_labels(
        [spike_times], [spike_labels], sampling_frequency=sampling_frequency
    )

    return sorting


# -- spatial utils


def generate_geom(
    num_columns=4,
    num_contact_per_column=96,
    xpitch=-16,
    ypitch=40,
    x_start=59,
    y_start=20,
    y_shift_per_column: Literal["stagger", "flat"]
    | tuple[float]
    | list[float] = "stagger",
    stagger: float = 20.0,
    sort=True,
    sort_x_down=True,
):
    """Defaults match NP1 geometry as returned by ibl-neuropixel."""
    if y_shift_per_column == "stagger" and num_columns == 1:
        y_shift_per_column = [0.0]
    elif y_shift_per_column == "stagger":
        num_columns_even_half = 1 + (num_columns // 2)
        y_shift_per_column = [stagger, 0.0] * num_columns_even_half
        y_shift_per_column = y_shift_per_column[:num_columns]
    elif y_shift_per_column == "flat" or y_shift_per_column is None:
        y_shift_per_column = [0.0] * num_columns
    else:
        assert isinstance(y_shift_per_column, (tuple, list, np.ndarray))
    p = probeinterface.generate_multi_columns_probe(
        num_columns=num_columns,
        num_contact_per_column=num_contact_per_column,
        xpitch=xpitch,
        ypitch=ypitch,
        y_shift_per_column=np.asarray(y_shift_per_column),
    )
    geom = p.contact_positions
    assert geom is not None
    geom[:, 0] += x_start
    geom[:, 1] += y_start
    if sort_x_down:
        order = np.lexsort((geom * [-1, 1]).T)
    elif sort:
        order = np.lexsort(geom.T)
    else:
        order = slice(None)
    geom = geom[order]
    return geom


def rbf_kernel_sqrt(geom, bandwidth=15.0, dtype="float32"):
    x = geom / (np.sqrt(2.0) * bandwidth)
    k = cdist(x, x, metric="sqeuclidean")
    np.negative(k, out=k)
    np.exp(k, out=k)
    vals, vecs = np.linalg.eigh(k)
    spatial_std = np.sqrt(vals, dtype=dtype)
    spatial_vt = np.ascontiguousarray(vecs.T, dtype=dtype)
    return spatial_std, spatial_vt


# -- sorting h5 helpers


# collidedness special cased in sims
default_sim_featurization_cfg = FeaturizationConfig(
    do_enforce_decrease=False, additional_com_localization=True, save_collidedness=False
)


def add_features(h5_path, recording, featurization_cfg, computation_cfg):
    with h5py.File(h5_path, "r+", locking=False) as h5:
        geom = cast(h5py.Dataset, h5["geom"])[:]
        channel_index = cast(h5py.Dataset, h5["channel_index"])[:]
        waveform_dict, fixed_properties = subsample_waveforms(h5=h5)
        waveforms = waveform_dict["collisioncleaned_waveforms"]
        if not len(waveforms):
            return
        featurization_cfg = replace(featurization_cfg, do_localization=len(geom) > 1)
        gt_pipeline = WaveformPipeline.from_config(
            featurization_cfg=featurization_cfg,
            waveform_cfg=WaveformConfig(),
            geom=geom,
            channel_index=channel_index,
            sampling_frequency=recording.sampling_frequency,
        )
        gt_pipeline.fit(recording, waveforms, computation_cfg, **fixed_properties)
        models_dir = h5_path.parent / f"{h5_path.stem}_models"
        models_dir.mkdir(exist_ok=True)
        torch.save(gt_pipeline.state_dict(), models_dir / "featurization_pipeline.pt")

        wf_dset = cast(h5py.Dataset, h5["collisioncleaned_waveforms"])
        n = wf_dset.shape[0]
        f_dsets = {
            sd.name: h5.create_dataset(
                sd.name, shape=(n, *sd.shape_per_spike), dtype=sd.dtype
            )
            for sd in gt_pipeline.spike_datasets()
        }
        for sli, chunk in yield_chunks(
            h5["collisioncleaned_waveforms"], desc_prefix="Featurize"
        ):
            _, feats = gt_pipeline(
                chunk, channels=cast(h5py.Dataset, h5["channels"])[sli]
            )
            for k in f_dsets:  # noqa: PLC0206
                f_dsets[k][sli] = feats[k].numpy(force=True)


def simulate_twostate_switching(
    rg: np.random.Generator,
    n_bins: int,
    state_affinity: float,
    n_units: int,
    min_fr: float,
    down_max_fr: float,
    up_min_fr: float,
    max_fr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate population firing rates in bins

    There are two latent states (random) and three neural populations.
    One is up in state 0, down state 1, other is reverse, and pop 3 doesn't
    care.

    Algorithm:
     - Simulate states
     - Simulate state affinities (0, 1, 2=don't care)
     - Draw per-neuron firing rates in each state according to affinity
     - Do a one-hot matmul to gather the final firing rates
    """
    states = rg.binomial(n=1, p=0.5, size=n_bins).astype(np.int32)
    states_onehot = np.zeros((n_bins, 2))
    states_onehot[np.arange(n_bins), states] = 1.0

    assert 2 * state_affinity < 1
    affinity_p = np.array([state_affinity, state_affinity, 1 - 2 * state_affinity])
    affinities = rg.choice(3, size=n_units, p=affinity_p)

    frs_by_state = np.zeros((2, n_units))
    for aff in range(3):
        in_aff = np.flatnonzero(affinities == aff)
        n_aff = in_aff.size

        if aff == 2:
            frs_by_state[:, in_aff] = rg.uniform(size=n_aff, low=min_fr, high=max_fr)
        elif aff <= 1:
            frs_by_state[aff, in_aff] = rg.uniform(
                size=n_aff, low=up_min_fr, high=max_fr
            )
            frs_by_state[1 - aff, in_aff] = rg.uniform(
                size=n_aff, low=min_fr, high=down_max_fr
            )
        else:
            raise ValueError(aff)

    return states, states_onehot @ frs_by_state


# -- injecting simulated spikes into a recording


class InjectSpikesPreprocessor(BasePreprocessor):
    """Inject simulated spikes into a background recording

    Given a background (noise) recording, a spike train, a template simulator,
    and a MotionInfo describing the (rigid or nonrigid) drift, this adds
    drifting, temporally jittered and amplitude scaled templates into the
    recording.

    It can also extract the ground truth waveforms and features of the injected
    spikes, see save_features_to_hdf5() and save_simulation().

    See simkit.make_simulated_recording() for a helper which samples the spike
    train and the motion before constructing one of these.
    """

    def __init__(
        self,
        recording: BaseRecording,
        *,
        sorting: BaseSorting,
        motion: MotionInfo,
        template_simulator: "TemplateSimulator",
        amplitude_jitter: float = 0.0,
        amp_jitter_family: Literal["gamma", "uniform", "normal"] = "normal",
        temporal_jitter: int = 1,
        temporal_jitter_family: Literal["uniform", "by_unit"] = "uniform",
        extract_radius: float = 150.0,
        random_seed: np.random.Generator | int = 0,
        features_dtype="float32",
        compute_collision_waveforms=False,
        extra_data_to_save: dict[str, np.ndarray] | None = None,
    ):
        super().__init__(recording)
        assert len(recording._recording_segments) == 1
        assert sorting.get_num_segments() == 1

        self._serializability["json"] = False
        self._serializability["pickle"] = False

        self.parent_segment = recording._recording_segments[0]
        self.add_recording_segment(
            InjectSpikesPreprocessorSegment(self.parent_segment, self)
        )

        self.motion = motion
        self.template_simulator = template_simulator
        self.temporal_jitter = temporal_jitter
        self.temporal_jitter_family = temporal_jitter_family
        self.features_dtype = features_dtype
        self.random_seed = random_seed
        self.compute_collision_waveforms = compute_collision_waveforms
        self.fs = self.sampling_frequency
        self.extra_data_to_save = extra_data_to_save

        # shapes
        self.n_units = template_simulator.n_units
        self.n_channels = self.get_num_channels()
        self.chans_arange = np.arange(self.n_channels)
        self.spike_length_samples = template_simulator.spike_length_samples()
        self.trough_offset_samples = template_simulator.trough_offset_samples()
        self.snippet_time_ix = np.arange(
            -self.trough_offset_samples,
            self.spike_length_samples - self.trough_offset_samples,
        )
        self.spike_safe_pad = max(
            self.trough_offset_samples,
            self.spike_length_samples - self.trough_offset_samples,
        )
        self.margin = 2 * self.spike_safe_pad
        self.extract_channel_index = make_channel_index(
            template_simulator.geom, extract_radius
        )
        self.wf_shape = (self.spike_length_samples, self.n_channels)
        self.template_shape = (self.n_units, *self.wf_shape)
        self.template_shape_up = (self.n_units, temporal_jitter, *self.wf_shape)

        # registered unit depths, which is where the motion is evaluated
        self.template_depths = np.asarray(template_simulator.template_depths())
        assert self.template_depths.shape == (self.n_units,)

        # the spike train
        assert sorting.get_num_units() == self.n_units
        sv = cast(np.recarray, sorting.to_spike_vector())
        self.times_samples = sv["sample_index"]
        self.labels = sv["unit_index"]
        self.n_spikes = sorting.count_total_num_spikes()
        assert self.times_samples.shape == self.labels.shape == (self.n_spikes,)

        # bake all random stuff
        rg = np.random.default_rng(random_seed)
        if amplitude_jitter and amp_jitter_family == "gamma":
            alpha = 1.0 / amplitude_jitter**2
            theta = amplitude_jitter**2
            self.scalings = rg.gamma(shape=alpha, scale=theta, size=self.n_spikes)
        elif amplitude_jitter and amp_jitter_family == "uniform":
            self.scalings = rg.uniform(
                1.0 - amplitude_jitter, 1.0 + amplitude_jitter, size=self.n_spikes
            )
        elif amplitude_jitter and amp_jitter_family == "normal":
            self.scalings = rg.normal(
                loc=1.0, scale=amplitude_jitter, size=self.n_spikes
            )
            # make sure things are positive...
            np.maximum(self.scalings, 0.0, out=self.scalings)
        else:
            assert not amplitude_jitter
            self.scalings = np.ones(1, dtype=features_dtype)
            self.scalings = np.broadcast_to(self.scalings, (self.n_spikes,))

        if self.temporal_jitter > 1 and temporal_jitter_family == "uniform":
            self.jitter_ix = rg.integers(self.temporal_jitter, size=self.n_spikes)
        elif self.temporal_jitter > 1 and temporal_jitter_family == "by_unit":
            self.jitter_ix = self.labels % self.temporal_jitter
        else:
            self.jitter_ix = np.zeros(1, dtype=np.int64)
            self.jitter_ix = np.broadcast_to(self.jitter_ix, (self.n_spikes,))
        self.upsampling_offsets = template_simulator.offsets_up[
            self.labels, self.jitter_ix
        ]

    # -- simulation info

    def basic_sorting(self) -> DARTsortSorting:
        return DARTsortSorting(
            times_samples=self.times_samples + self.upsampling_offsets,
            labels=self.labels,
            channels=np.zeros_like(self.times_samples),
            sampling_frequency=self.fs,
            ephemeral_features=dict(
                time_shifts=self.upsampling_offsets,
                jitter_ix=self.jitter_ix,
                scalings=self.scalings,
            ),
        )

    def drift(self, t_samples):
        """Each unit's displacement (um) at the given sample index, shape (n_units,)."""
        if not self.motion.drifting:
            return np.zeros_like(t_samples)
        t_s = np.asarray(self.sample_index_to_time(np.asarray(t_samples)))
        assert t_s.ndim == 0
        disp = self.motion.disp_at_s(t_s.reshape(1), self.template_depths, grid=True)
        return disp[:, 0]

    def templates(
        self, t_samples=None, *, drift=None, up=False, padded=False, pad_value=np.nan
    ):
        if drift is None:
            drift = 0 if t_samples is None else self.drift(t_samples)
        pos, templates, offsets = self.template_simulator.templates(
            drift=drift, up=up, padded=padded, pad_value=pad_value
        )
        templates = templates.astype(self.features_dtype)
        tunpad = templates[..., :-1] if padded else templates
        if up:
            assert tunpad.shape == self.template_shape_up
        else:
            assert tunpad.shape == self.template_shape

        return pos, templates, offsets

    def registered_geom(self):
        geom = self.get_channel_locations()
        rgeom = self.motion.rgeom
        matches = np.square(geom[None] - rgeom[:, None]).sum(2).argmin(0)
        return rgeom, matches

    def template_data(self, hdf5_path=None):
        if not self.motion.drifting:
            return TemplateData(
                templates=self.templates()[1],
                unit_ids=np.arange(self.n_units),
                spike_counts=np.ones(self.n_units),
                registered_geom=self.get_channel_locations(),
                trough_offset_samples=self.trough_offset_samples,
                sampling_frequency=self.sampling_frequency,
            )

        rgeom, matches = self.registered_geom()
        _, templates, _ = self.templates()
        rtemplates = np.zeros(
            (*templates.shape[:-1], len(rgeom)), dtype=templates.dtype
        )
        rtemplates[:, :, matches] = templates
        return TemplateData(
            templates=rtemplates,
            unit_ids=np.arange(self.n_units),
            spike_counts=np.ones(self.n_units),
            registered_geom=rgeom,
            trough_offset_samples=self.trough_offset_samples,
            sampling_frequency=self.sampling_frequency,
        )

    def gt_unit_information(self):
        ids = np.arange(self.n_units)
        pos, templates, _ = self.templates()
        x, y, z = pos.T
        counts = np.zeros(self.n_units, dtype=np.int64)
        u, c = np.unique(self.labels, return_counts=True)
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

    # -- injection and ground truth feature extraction

    def get_spikes(
        self,
        noise_with_margin,
        start_frame,
        end_frame,
        *,
        in_chunk_only=True,
        extract=False,
        n_residual_snips=0,
        get_injected=True,
    ):
        if in_chunk_only:
            search_start = start_frame
            search_end = end_frame
        else:
            search_start = start_frame - self.spike_safe_pad
            search_end = end_frame + self.spike_safe_pad
        i0 = np.searchsorted(self.times_samples, search_start)
        i1 = np.searchsorted(self.times_samples, search_end)
        t = self.times_samples[i0:i1]
        ll = self.labels[i0:i1]
        s = self.scalings[i0:i1]
        u = self.jitter_ix[i0:i1]

        # temporal indices of snippets relative to chunk
        t_rel = t - (start_frame - self.margin)
        tix = t_rel[:, None] + self.snippet_time_ix
        tc = (start_frame + end_frame) / 2
        drift = self.drift(tc)
        pos, temps, offsets = self.templates(drift=drift, up=True, padded=extract)
        temps = temps[ll, u]
        temps *= s[:, None, None]
        temps_unpad = temps[..., :-1] if extract else temps
        offsets = offsets[ll, u]

        spikes = dict(
            i0=i0,
            i1=i1,
            times_samples=t + offsets,
            time_shifts=offsets.astype(np.int16),
            labels=ll,
            scalings=s,
            jitter_ix=u,
            tix=tix,
            waveforms=temps_unpad,
            n_residual_snips=n_residual_snips,
        )
        if not extract:
            return spikes

        spikes["localizations"] = pos[ll]
        drift = np.broadcast_to(drift, (self.n_units,))[ll]
        spikes["displacements"] = drift.astype(self.features_dtype)
        ptp_vectors = ptp(temps_unpad, dim=1)
        c = ptp_vectors.argmax(axis=1)
        spikes["channels"] = c
        spikes["ptp_amplitudes"] = ptp_vectors.max(axis=1)
        if n_residual_snips:
            rsnips, rtimes = extract_random_snips(
                rg=self.random_seed,
                chunk=noise_with_margin[
                    self.margin : len(noise_with_margin) - self.margin
                ],
                n=n_residual_snips,
                sniplen=self.spike_length_samples,
            )
            if torch.is_tensor(rsnips):
                rsnips = rsnips.numpy(force=True)
            spikes["residual"] = rsnips
            rtimes += start_frame
            rtimes = self.sample_index_to_time(rtimes)
            spikes["residual_times"] = rtimes
            spikes["n_residual_snips"] = rtimes.shape[0]

        # extract the background noise which waveforms will be added into
        noise_padded = np.pad(
            noise_with_margin, [(0, 0), (0, 1)], constant_values=np.nan
        )
        echans = self.extract_channel_index[c]
        if torch.is_tensor(echans):
            echans = echans.numpy(force=True)
        noise_waveforms = noise_padded[tix[:, :, None], echans[:, None, :]]
        spikes["collidedness"] = np.sqrt(
            np.nanmean(np.square(noise_waveforms).mean(1), 1)
        )
        # the actual injected waveforms...
        if get_injected:
            injected_waveforms = np.take_along_axis(temps, echans[:, None, :], axis=2)
            collisioncleaned_waveforms = noise_waveforms + injected_waveforms
            spikes["noise_waveforms"] = noise_waveforms
            spikes["injected_waveforms"] = injected_waveforms
            spikes["collisioncleaned_waveforms"] = collisioncleaned_waveforms
            spikes["echans"] = echans

        return spikes

    def get_traces_and_inject_spikes(
        self,
        start_frame,
        end_frame,
        channel_indices=None,
        *,
        extract=False,
        inject=False,
        n_residual_snips=0,
        get_injected=False,
    ):
        traces, lm, rm = get_chunk_with_margin(
            self.parent_segment,
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=None,
            margin=self.margin,
            add_zeros=True,
        )
        assert lm == rm == self.margin
        assert traces.shape[1] == self.n_channels
        assert extract != inject
        spikes = self.get_spikes(
            traces,
            start_frame,
            end_frame,
            in_chunk_only=not inject,
            extract=extract,
            n_residual_snips=n_residual_snips,
            get_injected=get_injected,
        )

        if not inject and not self.compute_collision_waveforms:
            del traces
            return None, spikes

        waveforms = cast(np.ndarray, spikes["waveforms"])
        tix = cast(np.ndarray, spikes["tix"])
        waveforms = waveforms.astype(traces.dtype, copy=False)
        traces = traces.copy()
        np.add.at(traces, (tix[:, :, None], self.chans_arange[None, None]), waveforms)

        if self.compute_collision_waveforms and not inject:
            echans = cast(np.ndarray, spikes["echans"])
            traces_pad = np.pad(traces, [(0, 0), (0, 1)], constant_values=np.nan)
            cwfs = traces_pad[tix[:, :, None], echans[:, None, :]]
            cwfs -= spikes["collisioncleaned_waveforms"]
            spikes["collision_waveforms"] = cwfs
            cwfs = np.nan_to_num(cwfs).reshape(len(cwfs), -1)
            # collidedness is what's relevant to the features, which are perfectly
            # decollided. gt_collidedness is worst case (no cc).
            spikes["gt_collidedness"] = np.linalg.norm(cwfs, axis=1)

        traces = traces[self.margin : len(traces) - self.margin]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces, spikes

    def _get_traces_and_inject_spikes_job(self, args_kwargs):
        args, kwargs = args_kwargs
        return self.get_traces_and_inject_spikes(*args, **kwargs)

    # -- saving

    def save_features_to_hdf5(
        self,
        hdf5_path,
        *,
        overwrite=False,
        n_jobs=1,
        show_progress=True,
        n_residual_snips=4096,
        save_injected_waveforms=False,
        save_noise_waveforms=False,
        save_collision_waveforms=False,
        save_collisioncleaned_waveforms=True,
        save_collidedness=False,
        chunk_len_s=0.5,
        buffersize=32,
    ):
        if overwrite:
            if hdf5_path.exists():
                hdf5_path.unlink()
        else:
            assert not hdf5_path.exists()

        get_injected = (
            save_injected_waveforms
            or save_collision_waveforms
            or save_collisioncleaned_waveforms
        )

        n_jobs, Executor, context = get_pool(n_jobs, cls="ThreadPoolExecutor")
        with h5py.File(hdf5_path, "w", locking=False) as h5:
            n = self.n_spikes

            # fixed arrays
            h5.create_dataset("sampling_frequency", data=self.sampling_frequency)
            h5.create_dataset("geom", data=self.get_channel_locations())
            h5.create_dataset("channel_index", data=self.extract_channel_index)
            times_samples = self.times_samples + self.upsampling_offsets
            h5.create_dataset("times_samples", data=times_samples)
            h5.create_dataset(
                "times_seconds",
                data=self.sample_index_to_time(times_samples),
            )
            h5.create_dataset("labels", data=self.labels)
            h5.create_dataset("scalings", data=self.scalings)
            h5.create_dataset("jitter_ix", data=self.jitter_ix)
            pos, temp, off = self.template_simulator.templates(up=True)
            h5.create_dataset("unit_positions", data=pos)
            h5.create_dataset("up_offsets", data=off)
            h5.create_dataset("templates_up", data=temp)
            for k, v in (self.extra_data_to_save or {}).items():
                h5.create_dataset(k, data=v)

            del times_samples, pos, temp, off

            gc.collect()
            gc.collect()

            # arrays discovered in batches below
            f_dt = self.features_dtype
            inj_wf_shape = (
                self.spike_length_samples,
                self.extract_channel_index.shape[1],
            )
            dataset_shapes = {
                "localizations": ((3,), f_dt),
                "displacements": ((), f_dt),
                "ptp_amplitudes": ((), f_dt),
                "channels": ((), np.int32),
                "time_shifts": ((), np.int16),
                "collidedness": ((), f_dt),
            }
            if save_collisioncleaned_waveforms:
                dataset_shapes["collisioncleaned_waveforms"] = (inj_wf_shape, f_dt)
            if save_injected_waveforms:
                dataset_shapes["injected_waveforms"] = (inj_wf_shape, f_dt)
            if save_noise_waveforms:
                dataset_shapes["noise_waveforms"] = (inj_wf_shape, f_dt)
            if save_collision_waveforms:
                dataset_shapes["collision_waveforms"] = (inj_wf_shape, f_dt)
            if save_collidedness:
                dataset_shapes["gt_collidedness"] = ((), f_dt)
            datasets = {
                k: h5.create_dataset(k, dtype=dt, shape=(n, *sh))
                for k, (sh, dt) in dataset_shapes.items()
            }

            nt = self.get_num_frames()
            n_residual_snips = min(n_residual_snips, nt // self.spike_length_samples)

            # residual snippets
            if n_residual_snips:
                nrs_dset = h5.create_dataset(
                    "n_residuals", data=np.zeros((), dtype=np.int64)
                )
                residual = h5.create_dataset(
                    "residual",
                    shape=(n_residual_snips, *self.wf_shape),
                    maxshape=(n_residual_snips, *self.wf_shape),
                    chunks=(min(16, n_residual_snips), *self.wf_shape),
                    dtype=f_dt,
                )
                residual_times = h5.create_dataset(
                    "residual_times_seconds",
                    shape=(n_residual_snips,),
                    maxshape=(n_residual_snips,),
                    chunks=(min(16, n_residual_snips),),
                    dtype=f_dt,
                )
            else:
                nrs_dset = residual = residual_times = None

            with Executor(max_workers=n_jobs, mp_context=context) as pool:
                bs = int(self.sampling_frequency * chunk_len_s)
                chunk_starts = range(0, nt, bs)
                residual_snips_per_chunk = divide_randomly(
                    n_residual_snips, len(chunk_starts), self.random_seed
                )
                jobs = (
                    (
                        (t, min(t + bs, nt)),
                        dict(
                            extract=True,
                            n_residual_snips=nrs,
                            get_injected=get_injected,
                        ),
                    )
                    for t, nrs in zip(
                        chunk_starts, residual_snips_per_chunk, strict=True
                    )
                )
                if _can_buffer:
                    results = pool.map(
                        self._get_traces_and_inject_spikes_job,
                        jobs,
                        buffersize=buffersize,
                    )
                else:
                    results = pool.map(self._get_traces_and_inject_spikes_job, jobs)
                if show_progress:
                    results = progbar(
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
                    i0 = cast(int, s["i0"])
                    i1 = cast(int, s["i1"])
                    assert i0 == i1_prev
                    i1_prev = i1
                    n_injected += i1 - i0

                    for k, ds in datasets.items():
                        ds[i0:i1] = s[k]

                    nrs = s["n_residual_snips"]
                    if not nrs:
                        continue
                    assert nrs == cast(np.ndarray, s["residual"]).shape[0]
                    assert nrs == cast(np.ndarray, s["residual_times"]).shape[0]
                    assert residual is not None
                    assert nrs_dset is not None
                    assert residual_times is not None
                    assert resid_ix is not None
                    residual[resid_ix : resid_ix + nrs] = s["residual"]
                    residual_times[resid_ix : resid_ix + nrs] = s["residual_times"]
                    nrs_dset[()] = resid_ix + nrs
                    resid_ix += nrs
                if residual is not None and resid_ix != n_residual_snips:
                    assert residual_times is not None
                    residual.resize((resid_ix, *residual.shape[1:]))
                    residual_times.resize((resid_ix, *residual_times.shape[1:]))
                if residual is not None:
                    assert residual_times is not None
                    assert nrs_dset is not None
                    assert residual.shape[0] == residual_times.shape[0], (
                        f"{residual.shape=} {residual_times.shape=}"
                    )
                    assert residual.shape[0] == resid_ix, (
                        f"{residual.shape=} {resid_ix=}"
                    )
                    assert nrs_dset[()] == resid_ix
                assert i1_prev == n
            assert n_injected == n

    def save_simulation(
        self,
        folder,
        *,
        overwrite=False,
        n_jobs=1,
        featurization_cfg=default_sim_featurization_cfg,
        n_residual_snips=4096,
        computation_cfg=None,
        save_injected_waveforms=False,
        save_noise_waveforms=False,
        save_collision_waveforms=False,
        save_collisioncleaned_waveforms=True,
        save_collidedness=False,
        extra_unit_information: pd.DataFrame | None = None,
        chunk_len_s=0.5,
        pool_engine="thread",
        mp_context="spawn",
    ):
        folder = ensure_path(folder)
        folder.mkdir(exist_ok=True)
        recording_dir = folder / "recording"
        templates_npz = folder / "templates.npz"
        sorting_h5 = folder / "dartsort_sorting.h5"
        unit_info_csv = folder / "unit_information.csv"

        # doing this up top to fail early
        unit_df = self.gt_unit_information()
        if extra_unit_information is not None:
            assert extra_unit_information.index.name is None
            assert len(unit_df) == len(extra_unit_information)
            unit_df = pd.concat([unit_df, extra_unit_information], axis=1)

        if recording_dir.exists():
            try:
                recording = read_binary_folder(recording_dir)
                logger.info("Loaded %s", recording_dir)
            except Exception:  # noqa: BLE001
                recording = None
        else:
            recording = None
        if recording is None:
            with warnings.catch_warnings(record=True) as ws:
                recording = self.save_to_folder(
                    folder=recording_dir,
                    overwrite=True,
                    n_jobs=n_jobs or 1,
                    pool_engine=pool_engine,
                    chunk_duration=chunk_len_s,
                    mp_context=mp_context,
                )
                for w in ws:
                    msg = str(w.message)
                    if msg.startswith("The extractor is not serializable "):
                        continue
                    if msg.startswith("auto_cast_uint"):
                        continue
                    raise w.category(w.message)
        n_residual_snips = 0 if featurization_cfg is None else n_residual_snips
        self.save_features_to_hdf5(
            sorting_h5,
            n_jobs=n_jobs,
            overwrite=overwrite,
            n_residual_snips=n_residual_snips,
            save_injected_waveforms=save_injected_waveforms,
            save_noise_waveforms=save_noise_waveforms,
            save_collision_waveforms=save_collision_waveforms,
            save_collisioncleaned_waveforms=save_collisioncleaned_waveforms,
            save_collidedness=save_collidedness,
            chunk_len_s=chunk_len_s,
        )
        if featurization_cfg is not None and not featurization_cfg.skip:
            # this is only for the TPCA feature.
            torch.manual_seed(self.random_seed)
            add_features(sorting_h5, recording, featurization_cfg, computation_cfg)

        unit_df.to_csv(unit_info_csv)
        self.motion.save(folder)
        self.template_data(sorting_h5).to_npz(templates_npz)


class InjectSpikesPreprocessorSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment: BaseRecordingSegment,
        recording: InjectSpikesPreprocessor,
    ):
        super().__init__(parent_recording_segment)
        self.recording = recording

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces, _ = self.recording.get_traces_and_inject_spikes(
            start_frame, end_frame, channel_indices, inject=True
        )
        return traces
