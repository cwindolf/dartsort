from logging import getLogger

import numpy as np
import probeinterface
from dredge import motion_util
import h5py
from scipy.spatial.distance import cdist
from spikeinterface.core import NumpyRecording, NumpySorting
from tqdm.auto import tqdm

from ..templates import TemplateData
from ..util.data_util import DARTsortSorting, extract_random_snips
from ..util.waveform_util import (
    make_channel_index,
    upsample_singlechan,
    upsample_multichan,
)
from ..util.spiketorch import spawn_torch_rg, ptp
from ..util.drift_util import registered_geometry
from ..util.interpolation_util import interp_precompute, kernel_interpolate

logger = getLogger(__name__)


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

    sorting = NumpySorting.from_samples_and_labels(
        [spike_times], [spike_labels], sampling_frequency=sampling_frequency
    )

    return sorting


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


def singlechan_to_probe(pos, alpha, waveforms, geom3, decay_model="32"):
    dtype = waveforms.dtype
    if decay_model == "pointsource":
        amp = alpha / cdist(pos, geom3).astype(dtype)
    elif decay_model == "squared":
        amp = alpha * (cdist(pos, geom3).astype(dtype) ** -2)
    elif decay_model == "32":
        amp = alpha * (cdist(pos, geom3).astype(dtype) ** -(3 / 2))
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
        n_units,
        cmr=False,
        temporal_upsampling=1,
        # timing params
        tip_before_min=0.1,
        tip_before_max=0.5,
        peak_after_min=0.2,
        peak_after_max=0.8,
        # width params
        trough_width_min=0.005,
        trough_width_max=0.025,
        tip_width_min=0.01,
        tip_width_max=0.075,
        peak_width_min=0.05,
        peak_width_max=0.2,
        # rel height params
        tip_rel_max=0.3,
        peak_rel_max=0.5,
        # pos/amplitude params
        pos_margin_um=25.0,
        orthdist_min_um=25.0,
        orthdist_max_um=50.0,
        alpha_family="uniform",
        alpha_min=5 * 25.0**2,
        alpha_max=40 * 25.0**2,
        alpha_mean=10.0 * np.square(25.0),
        alpha_var=5.0 * np.square(25.0),
        # config
        ms_before=1.4,
        ms_after=2.6,
        fs=30_000.0,
        depth_order=True,
        decay_model="squared",
        seed: int | np.random.Generator = 0,
        dtype=np.float32,
    ):
        self.rg = np.random.default_rng(seed)
        self.dtype = dtype
        self.n_units = n_units
        self.temporal_upsampling = temporal_upsampling
        self.cmr = cmr

        self.geom = geom
        self.geom3 = geom
        if geom.shape[1] == 2:
            self.geom3 = np.zeros((geom.shape[0], 3))
            self.geom3[:, [0, 2]] = geom
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
        self.alpha_family = alpha_family
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_mean = alpha_mean
        self.alpha_var = alpha_var
        theta = alpha_var / alpha_mean
        k = alpha_mean / theta
        self.alpha_shape = k
        self.alpha_scale = theta
        self.decay_model = decay_model
        self.depth_order = depth_order

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
            temporal_jitter=temporal_upsampling,
        )

    def templates(self, drift=0, up=False):
        pos = self.template_pos
        if drift:
            pos = pos + [0, 0, drift]

        templates = singlechan_to_probe(
            pos,
            self.template_alpha,
            self.singlechan_templates_up if up else self.singlechan_templates,
            self.template_simulator.geom3,
            decay_model=self.template_simulator.decay_model,
        )
        if self.cmr:
            templates -= np.median(templates, axis=-1, keepdims=True)
        return pos, templates

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
        z_low = self.geom[:, 1].min() - self.pos_margin_um
        z_high = self.geom[:, 1].max() + self.pos_margin_um

        x = self.rg.uniform(x_low, x_high, size=size)
        z = self.rg.uniform(z_low, z_high, size=size)
        if self.depth_order:
            z.sort()

        orth = self.rg.uniform(self.orthdist_min_um, self.orthdist_max_um, size=size)

        if self.alpha_family == "gamma":
            alpha = self.rg.gamma(
                shape=self.alpha_shape, scale=self.alpha_scale, size=size
            )
        elif self.alpha_family == "uniform":
            alpha = self.rg.uniform(self.alpha_min, self.alpha_max, size=size)
        else:
            assert False

        pos = np.c_[x, orth, z].astype(self.dtype)
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
        return pos, alpha, templates


class TemplateLibrarySimulator:
    def __init__(
        self,
        geom,
        templates_local,
        pos_local,
        temporal_jitter=1,
        cmr=False,
        interp_method="kriging",
        interp_kernel_name="thinplate",
        extrap_method="kernel",
        extrap_kernel_name="rbf",
        kriging_poly_degree=0,
        trough_offset_samples=42,
        fs=30_000.0,
    ):
        self.geom = geom
        self.temporal_jitter = temporal_jitter
        self.cmr = cmr
        self.fs = fs
        self._trough_offset_samples = trough_offset_samples

        self.interp_method = interp_method
        self.interp_kernel_name = interp_kernel_name
        self.extrap_method = extrap_method
        self.extrap_kernel_name = extrap_kernel_name
        self.kriging_poly_degree = kriging_poly_degree

        self.n_units = len(templates_local)
        self.templates_local = templates_local
        self.templates_local_up = upsample_multichan(
            templates_local, temporal_jitter=temporal_jitter
        )
        tpos = pos_local[
            np.arange(self.n_units),
            np.nan_to_num(np.abs(templates_local).max(1)).argmax(1),
        ]
        assert tpos.shape[1] == 2
        self.template_pos = np.c_[tpos[:, 0], 0 * tpos[:, 0], tpos[:, 1]]
        assert np.array_equal(
            np.isfinite(self.templates_local[:, 0]),
            np.isfinite(self.templates_local_up[:, 0, 0]),
        )
        self.pos_local = pos_local

        # precompute interpolation data
        self.precomputed_data = interp_precompute(
            source_pos=pos_local,
            method=interp_method,
            kernel_name=interp_kernel_name,
            kriging_poly_degree=kriging_poly_degree,
        )

    def trough_offset_samples(self):
        return self._trough_offset_samples

    def spike_length_samples(self):
        return self.templates_local.shape[1]

    @classmethod
    def from_template_library(
        cls,
        geom,
        n_units,
        templates,
        randomize_position=True,
        cmr=False,
        temporal_jitter=1,
        radius=250,
        trough_offset_samples=42,
        pos_margin_um=25.0,
        fs=30_000.0,
        seed=0,
    ):
        if templates.shape[0] > n_units:
            rg = np.random.default_rng(seed)
            choices = rg.choice(len(templates), size=n_units, replace=False)
            templates = templates[choices]

        assert np.isfinite(templates).all()
        channel_index = make_channel_index(geom, radius)
        main_channels = np.abs(templates).max(1).argmax(1)
        template_channels = channel_index[main_channels]

        geomp = np.pad(geom, [(0, 1), (0, 0)], constant_values=np.nan)
        templatesp = np.pad(templates, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan)
        pos_local = geomp[template_channels]
        templates_local = np.take_along_axis(
            templatesp, template_channels[:, None], axis=2
        )

        if randomize_position:
            z_low = geom[:, 1].min() - pos_margin_um
            z_high = geom[:, 1].max() + pos_margin_um
            z = rg.uniform(z_low, z_high, size=n_units)
            pos_local[..., 1] += z[:, None] - geom[main_channels, 1][:, None]

        return cls(
            geom=geom,
            templates_local=templates_local,
            pos_local=pos_local,
            temporal_jitter=temporal_jitter,
            cmr=cmr,
            trough_offset_samples=trough_offset_samples,
            fs=fs,
        )

    def templates(self, drift=None, up=False):
        source_pos = self.pos_local
        tpos = self.template_pos
        if drift is not None:
            source_pos = source_pos + [0, drift]
            tpos = tpos + [0, 0, drift]
        target_pos = np.broadcast_to(self.geom[None], (self.n_units, *self.geom.shape))

        if not up:
            templates = kernel_interpolate(
                self.templates_local,
                source_pos,
                target_pos,
                method=self.interp_method,
                kernel_name=self.interp_kernel_name,
                extrap_method=self.extrap_method,
                extrap_kernel_name=self.extrap_kernel_name,
                kriging_poly_degree=self.kriging_poly_degree,
                precomputed_data=self.precomputed_data,
            ).numpy(force=True)
        else:
            templates = [
                kernel_interpolate(
                    self.templates_local_up[:, u],
                    source_pos,
                    target_pos,
                    method=self.interp_method,
                    kernel_name=self.interp_kernel_name,
                    extrap_method=self.extrap_method,
                    extrap_kernel_name=self.extrap_kernel_name,
                    kriging_poly_degree=self.kriging_poly_degree,
                    precomputed_data=self.precomputed_data,
                ).numpy(force=True)
                for u in range(self.temporal_jitter)
            ]
            templates = np.stack(templates, axis=1)

        if self.cmr:
            templates -= np.median(templates, axis=-1, keepdims=True)

        return tpos, templates


# -- recording sim


class SimulatedRecording:
    def __init__(
        self,
        template_simulator,
        noise,
        duration_samples,
        min_fr_hz=1.0,
        max_fr_hz=10.0,
        drift_speed=None,
        amplitude_jitter=0.0,
        refractory_samples=40,
        globally_refractory=False,
        cmr=False,
        highpass_spatial_filter=False,
        amp_jitter_family="uniform",
        seed: int | np.random.Generator = 0,
        dtype="float32",
    ):
        self.n_units = template_simulator.n_units
        self.duration_samples = duration_samples
        self.template_simulator = template_simulator
        self.geom = template_simulator.geom
        self.noise = noise
        assert not (cmr and highpass_spatial_filter)
        self.cmr = cmr
        assert template_simulator.cmr == self.cmr
        self.highpass_spatial_filter = highpass_spatial_filter
        self.amp_jitter_family = amp_jitter_family
        self.dtype = dtype

        self.seed = seed
        self.rg = np.random.default_rng(seed)
        self.torch_rg = spawn_torch_rg(self.rg)

        self.amplitude_jitter = amplitude_jitter
        self.temporal_jitter = template_simulator.temporal_jitter
        self.min_fr_hz = min_fr_hz
        self.max_fr_hz = max_fr_hz
        self.drift_speed = drift_speed

        # -- random stuff
        # simulate spike trains
        self.firing_rates = self.rg.uniform(min_fr_hz, max_fr_hz, size=self.n_units)
        self.sorting = simulate_sorting(
            self.n_units,
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
            if amp_jitter_family == "gamma":
                alpha = 1 / amplitude_jitter**2
                theta = amplitude_jitter**2
                self.scalings = self.rg.gamma(
                    shape=alpha, scale=theta, size=self.n_spikes
                )
            elif amp_jitter_family == "uniform":
                self.scalings = self.rg.uniform(
                    1 - amplitude_jitter, 1 + amplitude_jitter, size=self.n_spikes
                )
            else:
                assert False
        else:
            self.scalings = np.ones(1)
            self.scalings = np.broadcast_to(self.scalings, (self.n_spikes,))

        if self.temporal_jitter > 1:
            self.jitter_ix = self.rg.integers(self.temporal_jitter, size=self.n_spikes)
        else:
            self.jitter_ix = np.zeros(1, dtype=np.int64)
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
        drift = 0 if t_samples is None else self.drift(t_samples)
        pos, templates = self.template_simulator.templates(drift=drift, up=up)
        templates = templates.astype(self.dtype)
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

        return pos, templates

    def template_data(self):
        if not self.drift_speed:
            return TemplateData(
                templates=self.templates()[1],
                unit_ids=np.arange(self.n_units),
                spike_counts=np.ones(self.n_units),
                registered_geom=self.geom,
                trough_offset_samples=self.template_simulator.trough_offset_samples(),
                spike_length_samples=self.template_simulator.spike_length_samples(),
            )

        rgeom, matches = self.registered_geom()
        _, templates = self.templates()
        rtemplates = np.zeros(
            (*templates.shape[:-1], len(rgeom)), dtype=templates.dtype
        )
        rtemplates[:, :, matches] = templates
        return TemplateData(
            templates=rtemplates,
            unit_ids=np.arange(self.n_units),
            spike_counts=np.ones(self.n_units),
            registered_geom=rgeom,
            trough_offset_samples=self.template_simulator.trough_offset_samples(),
            spike_length_samples=self.template_simulator.spike_length_samples(),
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

    def gt_unit_information(self):
        import pandas as pd

        ids = np.arange(self.n_units)
        x, y, z = self.template_simulator.template_pos.T
        counts = np.zeros(self.n_units, dtype=np.int64)
        labels = self.sorting.to_spike_vector()["unit_index"]
        u, c = np.unique(labels, return_counts=True)
        counts[u] = c

        return pd.DataFrame(
            dict(gt_unit_id=ids, x=x, y=y, z=z, gt_count=counts)
        )

    def simulate(self, gt_h5_path, extract_radius=100.0):
        """
        If gt_h5_path is not None, will save datasets:
         - times_samples
         - channels
         - labels
         - scalings
         - ptp_amplitudes
         - localizations
         - upsampling_ix
         - displacements
         - residual (residual snippets)
         - injected_waveforms
         - collisioncleaned_waveforms
        Will also save a channel_index and geom for good measure. These are
        full-probe.

        Returns
        -------
        recording : NumpyRecording
        gt_sorting: DARTSortSorting | None
        """
        logger.dartsortdebug("Simulate noise...")
        x = self.noise.simulate(
            size=1,
            t=self.duration_samples,
            generator=self.torch_rg,
            chunk_t=int(self.template_simulator.fs),
        )
        assert x.shape == (1, self.duration_samples, len(self.geom))
        x = x[0].numpy(force=True).astype(self.dtype)

        if self.cmr:
            # before adding temps, since already cmrd.
            x -= np.median(x, axis=1, keepdims=True)

        t_rel_ix = np.arange(self.template_simulator.spike_length_samples())
        t_rel_ix -= self.template_simulator.trough_offset_samples()
        chan_ix = np.arange(len(self.geom))

        sv = self.sorting.to_spike_vector()
        times = sv["sample_index"]
        labels = sv["unit_index"]

        with h5py.File(gt_h5_path, "w", libver="latest", locking=False) as h5:
            h5.create_dataset("times_samples", data=times)
            h5.create_dataset("times_seconds", data=times / self.template_simulator.fs)
            h5.create_dataset("labels", data=labels)
            h5.create_dataset("upsampling_ix", data=self.jitter_ix)
            h5.create_dataset("scalings", data=self.scalings)
            h5.create_dataset("geom", data=self.geom)
            ci = make_channel_index(self.geom, extract_radius)
            h5.create_dataset("channel_index", data=ci)
            h5.create_dataset("sampling_frequency", data=self.template_simulator.fs)

            # residual snips. use separate rg so this doesn't change sim data.
            rsnips, rtimes = extract_random_snips(
                np.random.default_rng(self.seed),
                x,
                n=4096,
                sniplen=self.template_simulator.spike_length_samples(),
            )
            h5.create_dataset("residual", data=rsnips)
            h5.create_dataset(
                "residual_times_seconds", data=rtimes / self.template_simulator.fs
            )

            # saved in loop below
            ds_channels = h5.create_dataset("channels", shape=len(times), dtype="int32")
            ds_displacements = h5.create_dataset(
                "displacements", shape=len(times), dtype="float32"
            )
            ds_localizations = h5.create_dataset(
                "localizations", shape=(len(times), 3), dtype="float32"
            )
            ds_ptp_amplitudes = h5.create_dataset(
                "ptp_amplitudes", shape=len(times), dtype="float32"
            )
            wf_shape = (
                len(times),
                self.template_simulator.spike_length_samples(),
                ci.shape[1],
            )
            ds_injected_waveforms = h5.create_dataset(
                "injected_waveforms", shape=wf_shape, dtype="float32"
            )
            ds_collisioncleaned_waveforms = h5.create_dataset(
                "collisioncleaned_waveforms", shape=wf_shape, dtype="float32"
            )

            chunks = (times // self.template_simulator.fs).astype(np.int32)
            the_chunks = np.unique(chunks)

            for chunk in tqdm(the_chunks, "Adding templates..."):
                batch = np.flatnonzero(chunks == chunk)
                assert np.all(np.diff(batch) == 1)
                batch = slice(batch[0], batch[-1] + 1)

                bt = times[batch]
                bl = labels[batch]
                bjitter = self.jitter_ix[batch]
                tix = bt[:, None, None] + t_rel_ix[None, :, None]

                t_samples = (chunk + 0.5) * self.template_simulator.fs
                pos, temps = self.templates(t_samples=t_samples, up=True)

                btemps = temps[bl, bjitter]
                btemps *= self.scalings[batch, None, None]
                amps = ptp(btemps)
                maxamps = amps.max(1)
                self.maxchans[batch] = amps.argmax(1)

                if h5 is not None:
                    # extract background before its too late
                    chans = ci[self.maxchans[batch]]
                    ccwf = x[tix, chan_ix[None, None]]
                    ccwf = np.pad(
                        ccwf, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
                    )
                    ccwf = np.take_along_axis(ccwf, chans[:, None, :], axis=2)

                np.add.at(x, (tix, chan_ix[None, None]), btemps)

                if h5 is not None:
                    ds_channels[batch] = self.maxchans[batch]
                    ds_displacements[batch] = self.drift(t_samples)
                    ds_localizations[batch] = pos[bl]
                    ds_ptp_amplitudes[batch] = maxamps

                    chans = ci[self.maxchans[batch]]
                    btemps = np.pad(
                        btemps, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
                    )
                    inj_wfs = np.take_along_axis(btemps, chans[:, None, :], axis=2)
                    ds_injected_waveforms[batch] = inj_wfs
                    ccwf += inj_wfs
                    ds_collisioncleaned_waveforms[batch] = ccwf

        recording = NumpyRecording(x, sampling_frequency=self.template_simulator.fs)
        recording.set_dummy_probe_from_locations(self.geom)

        if self.highpass_spatial_filter:
            from spikeinterface.preprocessing import highpass_spatial_filter

            recording = highpass_spatial_filter(recording)

        return recording
