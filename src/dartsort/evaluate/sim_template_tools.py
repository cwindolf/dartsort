import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import torch

from ..util.waveform_util import (
    upsample_singlechan,
    upsample_multichan,
    make_channel_index,
)
from ..util.interpolation_util import interp_precompute, kernel_interpolate
from ..templates.template_util import svd_compress_templates


def get_template_simulator(
    n_units,
    templates_kind="3exp",
    template_library=None,
    geom=None,
    sampling_frequency=30_000.0,
    random_seed=0,
    common_reference=False,
    temporal_jitter=1,
    **template_simulator_kwargs,
):
    if templates_kind == "3exp":
        return PointSource3ExpSimulator(
            geom=geom,
            n_units=n_units,
            sampling_frequency=sampling_frequency,
            seed=random_seed,
            common_reference=common_reference,
            temporal_jitter=temporal_jitter,
            **template_simulator_kwargs,
        )
    if templates_kind == "library":
        assert template_library is not None
        return TemplateLibrarySimulator.from_template_library(
            geom=geom,
            n_units=n_units,
            templates=template_library,
            seed=random_seed,
            common_reference=common_reference,
            temporal_jitter=temporal_jitter,
            **template_simulator_kwargs,
        )
    assert False


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
        common_reference=False,
        temporal_jitter=1,
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
        pos_margin_um_x=25.0,
        pos_margin_um_z=25.0,
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
        sampling_frequency=30_000.0,
        depth_order=True,
        decay_model="squared",
        seed: int | np.random.Generator = 0,
        dtype=np.float32,
    ):
        self.rg = np.random.default_rng(seed)
        self.dtype = dtype
        self.n_units = n_units
        self.temporal_jitter = temporal_jitter
        self.common_reference = common_reference

        self.geom = geom
        self.geom3 = geom
        if geom.shape[1] == 2:
            self.geom3 = np.zeros((geom.shape[0], 3))
            self.geom3[:, [0, 2]] = geom
        self.geom3 = self.geom3.astype(self.dtype)
        self.ms_before = ms_before
        self.ms_after = ms_after
        self.sampling_frequency = sampling_frequency

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
        self.pos_margin_um_x = pos_margin_um_x
        self.pos_margin_um_z = pos_margin_um_z
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

        pos, alpha = self.simulate_location(size=n_units)
        self.template_pos = pos
        self.template_alpha = alpha
        _, self.singlechan_templates = self.simulate_singlechan(size=n_units)
        # n, temporal_jitter, t
        self.singlechan_templates_up = upsample_singlechan(
            self.singlechan_templates,
            self.time_domain_ms(),
            temporal_jitter=temporal_jitter,
        )

    def templates(self, drift=0, up=False, padded=False, pad_value=np.nan):
        pos = self.template_pos
        if drift:
            pos = pos + [0, 0, drift]

        geom3 = self.geom3
        if padded:
            geom3 = np.pad(geom3, [(0, 1), (0, 0)])

        templates = singlechan_to_probe(
            pos,
            self.template_alpha,
            self.singlechan_templates_up if up else self.singlechan_templates,
            geom3,
            decay_model=self.decay_model,
        )
        if self.common_reference:
            tunpad = templates[..., :-1] if padded else templates
            tunpad -= np.median(tunpad, axis=-1, keepdims=True)
        if padded:
            templates[..., -1] = pad_value
        return pos, templates

    def trough_offset_samples(self):
        return int(self.ms_before * (self.sampling_frequency / 1000))

    def spike_length_samples(self):
        spike_len_ms = self.ms_before + self.ms_after
        length = int(spike_len_ms * (self.sampling_frequency / 1000))
        length = 2 * (length // 2) + 1
        return length

    def time_domain_ms(self):
        t = np.arange(self.spike_length_samples(), dtype=self.dtype)
        t -= self.trough_offset_samples()
        t /= self.sampling_frequency / 1000
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
        x_low = self.geom[:, 0].min() - self.pos_margin_um_x
        x_high = self.geom[:, 0].max() + self.pos_margin_um_x
        z_low = self.geom[:, 1].min() - self.pos_margin_um_z
        z_high = self.geom[:, 1].max() + self.pos_margin_um_z

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
        radius=250.0,
        temporal_jitter=1,
        common_reference=False,
        interp_method="kriging",
        interp_kernel_name="thinplate",
        extrap_method="kernel",
        extrap_kernel_name="rbf",
        rank=10,
        kriging_poly_degree=0,
        trough_offset_samples=42,
    ):
        self.geom = geom
        self.geom_kdt = KDTree(geom)
        self.temporal_jitter = temporal_jitter
        self.common_reference = common_reference
        self.radius = radius
        self.channel_index = make_channel_index(geom, radius)
        self._trough_offset_samples = trough_offset_samples

        self.interp_method = interp_method
        self.interp_kernel_name = interp_kernel_name
        self.extrap_method = extrap_method
        self.extrap_kernel_name = extrap_kernel_name
        self.kriging_poly_degree = kriging_poly_degree

        self.n_units = len(templates_local)
        self.templates_local = templates_local
        self.low_rank_templates = svd_compress_templates(
            templates_local, allow_na=True
        )
        self.temporal_up = upsample_multichan(
            self.low_rank_templates.temporal_components,
            temporal_jitter=temporal_jitter,
        )
        self.spatial_singular = (
            self.low_rank_templates.spatial_components
            * self.low_rank_templates.singular_values[..., None]
        )

        tpos = pos_local[
            np.arange(self.n_units),
            np.nan_to_num(np.abs(templates_local).max(1)).argmax(1),
        ]
        assert tpos.shape[1] == 2
        self.template_pos = np.c_[tpos[:, 0], 0 * tpos[:, 0], tpos[:, 1]]
        assert np.array_equal(
            np.isfinite(self.templates_local[:, 0]),
            np.isfinite(self.low_rank_templates.spatial_components[:, 0]),
        )
        self.pos_local = pos_local

        # precompute interpolation data
        if interp_method != "griddata":
            self.precomputed_data = interp_precompute(
                source_pos=pos_local,
                method=interp_method,
                kernel_name=interp_kernel_name,
                kriging_poly_degree=kriging_poly_degree,
            )
        else:
            self.precomputed_data = None
            assert interp_kernel_name in ("nearest", "linear", "cubic")

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
        common_reference=False,
        temporal_jitter=1,
        extract_radius=250.0,
        inject_radius=500.0,
        trough_offset_samples=42,
        pos_margin_um_z=25.0,
        seed=0,
        dtype="float32",
        **kwargs,
    ):
        rg = np.random.default_rng(seed)
        if templates.shape[0] > n_units:
            choices = rg.choice(len(templates), size=n_units, replace=False)
            templates = templates[choices]
        templates = templates.astype(dtype)

        assert np.isfinite(templates).all()
        channel_index = make_channel_index(geom, extract_radius)
        main_channels = np.abs(templates).max(1).argmax(1)
        template_channels = channel_index[main_channels]

        geomp = np.pad(geom.astype(dtype), [(0, 1), (0, 0)], constant_values=np.nan)
        templatesp = np.pad(templates, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan)
        pos_local = geomp[template_channels]
        templates_local = np.take_along_axis(
            templatesp, template_channels[:, None], axis=2
        )

        if randomize_position:
            z_low = geom[:, 1].min() - pos_margin_um_z
            z_high = geom[:, 1].max() + pos_margin_um_z
            z = rg.uniform(z_low, z_high, size=n_units)
            # z = rg.choice(np.unique(geom[:, 1]), size=n_units)
            pos_local[..., 1] += z[:, None] - geom[main_channels, 1][:, None]

        return cls(
            geom=geom,
            templates_local=templates_local,
            pos_local=pos_local,
            temporal_jitter=temporal_jitter,
            common_reference=common_reference,
            trough_offset_samples=trough_offset_samples,
            radius=inject_radius,
            **kwargs,
        )

    def interpolate_templates(self, source_pos, target_pos, unit_ids, up=False):
        # interpolate spatial components
        spatial_singular = self.spatial_singular[unit_ids]
        if self.interp_method == "kriging":
            assert self.precomputed_data is not None
            out = kernel_interpolate(
                spatial_singular,
                source_pos,
                target_pos,
                method=self.interp_method,
                kernel_name=self.interp_kernel_name,
                extrap_method=self.extrap_method,
                extrap_kernel_name=self.extrap_kernel_name,
                kriging_poly_degree=self.kriging_poly_degree,
                precomputed_data=self.precomputed_data[unit_ids],
            ).numpy(force=True)
        elif self.interp_method == "griddata":
            n, nct, _2 = target_pos.shape
            n_, f, nc_ = spatial_singular.shape
            assert _2 == 2
            assert n == n_
            assert nc_ == source_pos.shape[1]
            if torch.is_tensor(spatial_singular):
                spatial_singular = spatial_singular.numpy(force=True)
            if torch.is_tensor(source_pos):
                source_pos = source_pos.numpy(force=True)
            if torch.is_tensor(target_pos):
                target_pos = target_pos.numpy(force=True)
            out = np.full((n, f, nct), np.nan, dtype=spatial_singular.dtype)
            griddata_interp(
                spatial_singular, source_pos, target_pos, out, method=self.interp_kernel_name
            )
        else:
            assert False

        # temporal part...
        n, r, c = spatial_singular.shape
        if up:
            temporal = self.temporal_up[unit_ids].reshape(n, -1, r)
        else:
            temporal = self.low_rank_templates.temporal_components[unit_ids]

        return np.einsum("nrc,ntr->ntc", out, temporal)

    def templates(
        self, drift=None, up=False, padded=False, pad_value=np.nan, unit_ids=None
    ):
        if unit_ids is None:
            unit_ids = slice(None)

        source_pos = self.pos_local[unit_ids]
        true_template_pos = self.template_pos[unit_ids]
        if drift is not None:
            drift = self.templates_local.dtype.type(drift)
            zero = self.templates_local.dtype.type(0.0)
            source_pos = source_pos + [zero, drift]
            true_template_pos = true_template_pos + [zero, zero, drift]

        tgeom = self.geom.astype(self.templates_local.dtype)
        _, target_chans = self.geom_kdt.query(true_template_pos[:, [0, 2]])
        tgeom = np.pad(tgeom, [(0, 1), (0, 0)], constant_values=np.nan)
        target_chans = self.channel_index[target_chans]
        # target_chans = torch.arange(len(tgeom)).broadcast_to(len(source_pos), len(tgeom))
        target_pos = torch.asarray(tgeom[target_chans])
        # target_chans = target_chans.numpy()

        templates = self.interpolate_templates(source_pos, target_pos, unit_ids, up=up)

        nu = len(templates)
        nc_out = len(self.geom) + 1
        nt = self.templates_local.shape[1]
        up_factor = self.temporal_jitter if up else 1
        out = np.zeros((nu, up_factor * nt, nc_out), dtype=self.templates_local.dtype)
        np.put_along_axis(out, target_chans[:, None, :], templates, axis=2)
        # out[np.arange(nu)[:, None, None], np.arange(up_factor * nt)[None, :, None], target_chans[:, None, :]] = templates

        if up:
            out = out.reshape(nu, up_factor, nt, nc_out)

        if padded:
            out[..., -1] = pad_value

        if self.common_reference:
            tunpad = out[..., :-1]
            tunpad -= np.median(tunpad, axis=-1, keepdims=True)

        if not padded:
            out = out[..., :-1]

        return true_template_pos, out


def griddata_interp(templates, source_pos, target_pos, out, method):
    n = templates.shape[0]
    for j in range(n):
        valid_in = np.flatnonzero(np.isfinite(source_pos[j, :, 0]))
        valid_out = np.flatnonzero(np.isfinite(target_pos[j, :, 0]))
        out[j, :, valid_out] = griddata(
            points=source_pos[j, valid_in],
            values=templates[j, :, valid_in],
            xi=target_pos[j, valid_out],
            method=method,
            fill_value=0.0,
        )
    return out
