import warnings

import numpy as np
import probeinterface
from spikeinterface.core import NumpySorting, NumpyRecording
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline
import torch
from tqdm.auto import tqdm

from dartsort.templates.templates import TemplateData

from dredge import motion_util

from .noise_util import StationaryFactorizedNoise, WhiteNoise
from .data_util import DARTsortSorting
from .drift_util import registered_geometry


# -- core utils


def spawn_torch_rg(seed: int | np.random.Generator = 0):
    nprg = np.random.default_rng(seed)
    seeder = nprg.spawn(1)[0]
    seed = int.from_bytes(seeder.bytes(8))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# -- spike train sims


def refractory_poisson_spike_train(
    rate_hz,
    duration_samples,
    seed: int | np.random.Generator = 0,
    refractory_samples=40,
    trough_offset_samples=42,
    spike_length_samples=121,
    sampling_frequency=30000.0,
    overestimation=1.5,
):
    """Sample a refractory Poisson spike train

    Arguments
    ---------
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
    overest_count = max(10, overest_count)

    # generate interspike intervals
    intervals = rg.exponential(scale=1.0 / rate_hz, size=overest_count)
    intervals += refractory_s
    intervals_samples = np.floor(intervals * sampling_frequency).astype(int)

    # determine spike times and restrict to ones which we can actually
    # add into / read from a recording with this duration and trough offset
    spike_samples = np.cumsum(intervals_samples)
    max_spike_time = duration_samples - (spike_length_samples - trough_offset_samples)
    # check that we overestimated enough
    assert spike_samples.max() > max_spike_time
    valid = spike_samples == spike_samples.clip(trough_offset_samples, max_spike_time)
    spike_samples = spike_samples[valid]
    assert spike_samples.size

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
    for rate, bin in zip(rates, bins):
        if rate < 0.1:
            continue
        binst = refractory_poisson_spike_train(rate, binsize_samples, **kwargs)
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
    if firing_rates is not None:
        assert firing_rates.shape[0] == num_units
    else:
        firing_rates = rg.uniform(1.0, 10.0, num_units)

    if not globally_refractory:
        spike_trains = [
            refractory_poisson_spike_train(
                firing_rates[i],
                n_samples,
                trough_offset_samples=nbefore,
                spike_length_samples=spike_length_samples,
                seed=rg,
                refractory_samples=refractory_samples,
            )
            for i in range(num_units)
        ]
        spike_times = np.concatenate(spike_trains)
        spike_labels = np.repeat(
            np.arange(num_units),
            np.array([spike_trains[i].shape[0] for i in range(num_units)]),
        )
    else:
        global_rate = np.sum(firing_rates)
        spike_times = refractory_poisson_spike_train(
            global_rate,
            n_samples,
            trough_offset_samples=nbefore,
            spike_length_samples=spike_length_samples,
            seed=rg,
            refractory_samples=refractory_samples,
        )
        unit_proportions = firing_rates / global_rate
        spike_labels = rg.choice(num_units, p=unit_proportions, size=spike_times.size)

    # order = np.argsort(spike_train)
    # spike_train = spike_train[order]
    # spike_labels = spike_labels[order]

    sorting = NumpySorting.from_times_labels(
        [spike_times], [spike_labels], sampling_frequency=sampling_frequency
    )

    return sorting


# -- temporal utils


def upsample_singlechan(singlechan_templates, time_domain, temporal_jitter=1):
    if temporal_jitter == 1:
        return singlechan_templates[:, None]

    n, t = singlechan_templates.shape
    erp = CubicSpline(time_domain, singlechan_templates, axis=-1)
    dt_ms = np.diff(time_domain).mean()
    t_up = np.stack(
        [time_domain + dt_ms * (j / temporal_jitter) for j in range(temporal_jitter)],
        axis=1,
    )
    t_up = t_up.ravel()
    erp_y_up = erp(t_up)
    erp_y_up = erp_y_up.reshape(n, t, temporal_jitter)
    singlechan_templates_up = erp_y_up.transpose(0, 2, 1)
    return singlechan_templates_up


# -- spatial utils


def generate_geom(
    num_columns=4,
    num_contact_per_column=10,
    xpitch=20,
    ypitch=20,
    y_shift_per_column=(0, -10, 0, -10),
):
    p = probeinterface.generate_multi_columns_probe(
        num_columns=num_columns,
        num_contact_per_column=num_contact_per_column,
        xpitch=xpitch,
        ypitch=ypitch,
        y_shift_per_column=y_shift_per_column,
    )
    geom = p.contact_positions
    assert geom is not None
    geom = geom[np.lexsort(geom.T)]
    return geom


def rbf_kernel(geom, bandwidth=35.0):
    d = cdist(geom, geom, metric="sqeuclidean")
    return np.exp(-d / (2 * bandwidth))


def singlechan_to_probe(pos, alpha, waveforms, geom3, decay_model="squared"):
    dtype = waveforms.dtype
    if decay_model == "pointsource":
        amp = alpha / cdist(pos, geom3).astype(dtype)
    elif decay_model == "squared":
        amp = alpha * (cdist(pos, geom3).astype(dtype) ** -2)
    else:
        assert False
    n_dims_expand = waveforms.ndim - 1
    templates = waveforms[..., :, None] * amp[..., *([None] * n_dims_expand), :]
    return templates


# -- template sims


class PointSource3ExpSimulator:
    def __init__(
        self,
        geom,
        # timing params
        tip_before_min=0.1,
        tip_before_max=0.8,
        peak_after_min=0.4,
        peak_after_max=1.5,
        # width params
        trough_width_min=0.01,
        trough_width_max=0.1,
        tip_width_min=0.01,
        tip_width_max=0.15,
        peak_width_min=0.1,
        peak_width_max=0.3,
        # rel height params
        tip_rel_max=0.3,
        peak_rel_max=0.5,
        # pos/amplitude params
        pos_margin_um=40.0,
        orthdist_min_um=20.0,
        orthdist_max_um=30.0,
        alpha_mean=8000.0,
        alpha_var=400.0,
        # config
        ms_before=1.4,
        ms_after=2.6,
        fs=30_000.0,
        decay_model="squared",
        seed: int | np.random.Generator = 0,
        dtype=np.float32,
    ):
        self.rg = np.random.default_rng(seed)
        self.dtype = dtype

        self.geom = geom
        self.geom3 = geom
        if geom.shape[1] == 2:
            self.geom3 = np.pad(geom, [(0, 0), (0, 1)])
        self.geom3 = self.geom3.astype(self.dtype)
        self.ms_before = ms_before
        self.ms_after = ms_after
        self.fs = fs

        self.tip_rel_max = tip_rel_max
        self.peak_rel_max = peak_rel_max
        self.tip_before_min = tip_before_min
        self.tip_before_max = tip_before_max
        self.peak_after_min = peak_after_min
        self.peak_after_max = peak_after_max
        self.trough_width_min = trough_width_min
        self.trough_width_max = trough_width_max
        self.tip_width_min = tip_width_min
        self.tip_width_max = tip_width_max
        self.peak_width_min = peak_width_min
        self.peak_width_max = peak_width_max
        self.pos_margin_um = pos_margin_um
        self.orthdist_min_um = orthdist_min_um
        self.orthdist_max_um = orthdist_max_um
        self.alpha_mean = alpha_mean
        self.alpha_var = alpha_var
        theta = alpha_var / alpha_mean
        k = alpha_mean / theta
        self.alpha_shape = k
        self.alpha_scale = theta
        self.decay_model = decay_model

    def trough_offset_samples(self):
        return int(self.ms_before * (self.fs / 1000))

    def spike_length_samples(self):
        spike_len_ms = self.ms_before + self.ms_after
        length = int(spike_len_ms * (self.fs / 1000))
        length = 2 * (length // 2) + 1
        return length

    def time_domain_ms(self):
        t = np.arange(self.spike_length_samples(), dtype=self.dtype)
        t -= self.trough_offset_samples()
        t /= self.fs / 1000
        return t

    def expand_size(self, size=None):
        if size is not None:
            if isinstance(size, int):
                size = (size,)
            else:
                assert isinstance(size, (tuple, list))
            # time will broadcast on inner dimension
            size = (*size, 1)
        return size

    def simulate_singlechan(self, size=None):
        """Simulate a trough-normalized 3-exp action potential."""
        t = self.time_domain_ms()
        size = self.expand_size(size)

        tip = self.rg.uniform(-self.tip_before_max, -self.tip_before_min, size=size)
        peak = self.rg.uniform(self.peak_after_min, self.peak_after_max, size=size)
        tip_height = self.rg.uniform(high=self.tip_rel_max, size=size)
        peak_height = self.rg.uniform(high=self.peak_rel_max, size=size)
        trough_width = self.rg.uniform(
            self.trough_width_min, self.trough_width_max, size=size
        )
        tip_width = self.rg.uniform(self.tip_width_min, self.tip_width_max, size=size)
        peak_width = self.rg.uniform(
            self.peak_width_min, self.peak_width_max, size=size
        )

        trough = -np.exp(-np.square(t) / (2 * trough_width))
        tip = np.exp(-np.square(t - tip) / (2 * tip_width))
        peak = np.exp(-np.square(t - peak) / (2 * peak_width))

        waveforms = trough + tip_height * tip + peak_height * peak
        waveforms /= -waveforms[..., self.trough_offset_samples(), None]

        return t, waveforms.astype(self.dtype)

    def simulate_location(self, size=None):
        size = self.expand_size(size)
        x_low = self.geom[:, 0].min() - self.pos_margin_um
        x_high = self.geom[:, 0].max() + self.pos_margin_um
        y_low = self.geom[:, 1].min() - self.pos_margin_um
        y_high = self.geom[:, 1].max() + self.pos_margin_um

        x = self.rg.uniform(x_low, x_high, size=size)
        y = self.rg.uniform(y_low, y_high, size=size)
        orth = self.rg.uniform(self.orthdist_min_um, self.orthdist_max_um, size=size)

        alpha = self.rg.gamma(shape=self.alpha_shape, scale=self.alpha_scale, size=size)

        pos = np.c_[x, y, orth].astype(self.dtype)
        alpha = alpha.astype(self.dtype)
        return pos, alpha

    def simulate_template(self, size=None):
        pos, alpha = self.simulate_location(size=size)
        t, waveforms = self.simulate_singlechan(size=size)
        templates = singlechan_to_probe(
            pos, alpha, waveforms, self.geom3, decay_model=self.decay_model
        )
        if size is None:
            assert templates.shape[0] == 1
            templates = templates[0]
        return templates


# -- recording sim


class StaticSimulatedRecording:
    def __init__(
        self,
        template_simulator,
        noise,
        n_units,
        duration_samples,
        min_fr_hz=1.0,
        max_fr_hz=10.0,
        drift_speed=None,
        temporal_jitter=1,
        amplitude_jitter=0.0,
        refractory_samples=40,
        globally_refractory=False,
        seed: int | np.random.Generator = 0,
    ):
        self.n_units = n_units
        self.duration_samples = duration_samples
        self.template_simulator = template_simulator
        self.geom = template_simulator.geom
        self.noise = noise

        self.rg = np.random.default_rng(seed)
        self.torch_rg = spawn_torch_rg(self.rg)

        self.amplitude_jitter = amplitude_jitter
        self.temporal_jitter = temporal_jitter
        self.min_fr_hz = min_fr_hz
        self.max_fr_hz = max_fr_hz
        self.drift_speed = drift_speed

        # -- deterministic stuff
        pos, alpha = template_simulator.simulate_location(size=n_units)
        self.template_pos = pos
        self.template_alpha = alpha
        _, self.singlechan_templates = template_simulator.simulate_singlechan(
            size=n_units
        )
        # n, temporal_jitter, t
        self.singlechan_templates_up = upsample_singlechan(
            self.singlechan_templates,
            self.template_simulator.time_domain_ms(),
            temporal_jitter=temporal_jitter,
        )

        # -- random stuff
        # simulate spike trains
        self.firing_rates = self.rg.uniform(min_fr_hz, max_fr_hz, size=n_units)
        self.sorting = simulate_sorting(
            n_units,
            duration_samples,
            firing_rates=self.firing_rates,
            rg=self.rg,
            nbefore=self.template_simulator.trough_offset_samples(),
            spike_length_samples=self.template_simulator.spike_length_samples(),
            sampling_frequency=self.template_simulator.fs,
            refractory_samples=refractory_samples,
            globally_refractory=globally_refractory,
        )
        self.n_spikes = self.sorting.count_total_num_spikes()
        self.maxchans = np.full((self.n_spikes), -1)  # populated during simulate

        if amplitude_jitter:
            alpha = 1 / amplitude_jitter**2
            theta = amplitude_jitter**2
            self.scalings = self.rg.gamma(shape=alpha, scale=theta, size=self.n_spikes)
        else:
            self.scalings = np.ones(1)
            self.scalings = np.broadcast_to(self.scalings, (self.n_spikes,))

        if temporal_jitter > 1:
            self.jitter_ix = self.rg.integers(temporal_jitter, size=self.n_spikes)
        else:
            self.jitter_ix = np.zeros(1, dtype=int)
            self.jitter_ix = np.broadcast_to(self.jitter_ix, (self.n_spikes,))

    def drift(self, t_samples):
        if not self.drift_speed:
            return np.zeros_like(t_samples)
        t_center = self.duration_samples / 2
        dt = (t_samples - t_center) / self.template_simulator.fs
        return dt * self.drift_speed

    def motion_estimate(self):
        if not self.drift_speed:
            return None
        duration_s = np.ceil(self.duration_samples / self.template_simulator.fs)
        t = np.arange(duration_s)
        time_bin_centers = t + 0.5
        tbc_samples = time_bin_centers * self.template_simulator.fs
        displacement = self.drift(tbc_samples)
        return motion_util.get_motion_estimate(
            displacement=displacement, time_bin_centers_s=time_bin_centers
        )

    def registered_geom(self):
        me = self.motion_estimate()
        geom = self.template_simulator.geom
        rgeom = registered_geometry(geom, motion_est=me)
        matches = np.square(geom[None] - rgeom[:, None]).sum(2).argmin(0)
        return rgeom, matches

    def templates(self, t_samples=None, up=False):
        if t_samples is None:
            drift = 0
            pos = self.template_pos
        else:
            drift = self.drift(t_samples)
            pos = self.template_pos + [0, drift, 0]

        templates = singlechan_to_probe(
            pos,
            self.template_alpha,
            self.singlechan_templates_up if up else self.singlechan_templates,
            self.template_simulator.geom3,
            decay_model=self.template_simulator.decay_model,
        )
        if up:
            assert templates.shape == (
                self.n_units,
                self.temporal_jitter,
                self.template_simulator.spike_length_samples(),
                len(self.template_simulator.geom),
            )
        else:
            assert templates.shape == (
                self.n_units,
                self.template_simulator.spike_length_samples(),
                len(self.template_simulator.geom),
            )

        return templates

    def template_data(self):
        if not self.drift_speed:
            return TemplateData(
                templates=self.templates(),
                unit_ids=np.arange(self.n_units),
                spike_counts=np.ones(self.n_units),
                registered_geom=self.geom,
                trough_offset_samples=self.template_simulator.trough_offset_samples(),
                spike_length_samples=self.template_simulator.spike_length_samples(),
            )

        rgeom, matches = self.registered_geom()
        templates = self.templates()
        rtemplates = np.zeros(
            (*templates.shape[:-1], len(rgeom)), dtype=templates.dtype
        )
        rtemplates[:, :, matches] = templates
        pos = rgeom[np.ptp(rtemplates, axis=1).argmax(1), 1]
        return TemplateData(
            templates=rtemplates,
            unit_ids=np.arange(self.n_units),
            spike_counts=np.ones(self.n_units),
            registered_geom=rgeom,
            trough_offset_samples=self.template_simulator.trough_offset_samples(),
            spike_length_samples=self.template_simulator.spike_length_samples(),
            registered_template_depths_um=pos,
        )

    def to_dartsort_sorting(self, sorting=None):
        if sorting is None:
            sorting = self.sorting
        sv = sorting.to_spike_vector()
        times = sv["sample_index"]
        labels = sv["unit_index"]
        return DARTsortSorting(
            times_samples=times,
            labels=labels,
            channels=self.maxchans,
            extra_features=dict(times_seconds=times / self.template_simulator.fs),
        )

    def simulate(self):
        """
        Returns
        -------
        recording : NumpyRecording
        sorting : NumpySorting
        """
        print("Simulate noise...")
        x = self.noise.simulate(
            size=1,
            t=self.duration_samples,
            generator=self.torch_rg,
            chunk_t=int(self.template_simulator.fs),
        )
        assert x.shape == (1, self.duration_samples, len(self.geom))
        x = x[0].numpy(force=True).astype(self.singlechan_templates.dtype)

        t_rel_ix = np.arange(self.template_simulator.spike_length_samples())
        t_rel_ix -= self.template_simulator.trough_offset_samples()
        chan_ix = np.arange(len(self.geom))

        sv = self.sorting.to_spike_vector()
        times = sv["sample_index"]
        labels = sv["unit_index"]

        chunks = (times // self.template_simulator.fs).astype(np.int32)
        the_chunks = np.unique(chunks)

        for chunk in tqdm(the_chunks, "Adding templates..."):
            batch = np.flatnonzero(chunks == chunk)
            bt = times[batch]
            bl = labels[batch]
            bjitter = self.jitter_ix[batch]
            tix = bt[:, None, None] + t_rel_ix[None, :, None]

            temps = self.templates(
                t_samples=(chunk + 0.5) * self.template_simulator.fs, up=True
            )

            btemps = self.scalings[batch, None, None] * temps[bl, bjitter]
            self.maxchans[batch] = btemps.ptp(1).argmax(1)
            np.add.at(x, (tix, chan_ix[None, None]), btemps)

        recording = NumpyRecording(x, sampling_frequency=self.template_simulator.fs)
        recording.set_dummy_probe_from_locations(self.geom)

        return recording
