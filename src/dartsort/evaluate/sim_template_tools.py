from tempfile import tempdir
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_array

from ..templates.template_util import svd_compress_templates
from ..templates.templates import TemplateData
from ..templates.realignment import get_main_channels_and_alignments
from ..util.interpolation_util import (
    InterpolationParams,
    default_interpolation_params,
    interp_precompute,
    kernel_interpolate,
)
from ..util.waveform_util import (
    make_channel_index,
    upsample_multichan,
    upsample_singlechan,
)


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
) -> "TemplateSimulator":
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
    if templates_kind == "static":
        assert "template_data" in template_simulator_kwargs
        template_data = template_simulator_kwargs.pop("template_data")
        if geom is not None:
            assert np.array_equal(geom, template_data.registered_geom)
        assert not common_reference
        assert n_units == len(template_data.unit_ids)
        return StaticTemplateSimulator(
            template_data=template_data,
            temporal_jitter=temporal_jitter,
            **template_simulator_kwargs,
        )
    assert False


def singlechan_to_probe(pos, alpha, waveforms, geom3, decay_model="squared"):
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


class BaseTemplateSimulator:
    n_units: int
    geom: np.ndarray

    def trough_offset_samples(self) -> int:
        raise NotImplementedError

    def spike_length_samples(self) -> int:
        raise NotImplementedError

    def templates(
        self,
        drift: float = 0.0,
        up: bool = False,
        padded: bool = False,
        pad_value: float = float("nan"),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (pos, templates, trough_offsets) which are (K, 3), (K, t, C), (K,) with optional up dim after K."""
        raise NotImplementedError


class StaticTemplateSimulator(BaseTemplateSimulator):
    """Super simple version."""

    def __init__(self, template_data: TemplateData, temporal_jitter: int = 1):
        assert template_data.registered_geom is not None
        self.n_units = template_data.unit_ids.size
        self.geom = template_data.registered_geom
        self.template_data = template_data
        self.temporal_jitter = temporal_jitter
        self.templates_up = upsample_multichan(
            self.template_data.templates, temporal_jitter=temporal_jitter
        )
        _, mct, a0 = get_main_channels_and_alignments(template_data)
        self.offsets = a0 - template_data.trough_offset_samples
        assert self.templates_up.shape[:2] == (self.n_units, temporal_jitter)
        _, mct, a1 = get_main_channels_and_alignments(
            templates=self.templates_up.reshape(
                self.n_units * temporal_jitter, *self.templates_up.shape[2:]
            )
        )
        a1 = a1 - template_data.trough_offset_samples
        self.offsets_up = a1.reshape(self.n_units, temporal_jitter)

    def trough_offset_samples(self) -> int:
        return self.template_data.trough_offset_samples

    def spike_length_samples(self) -> int:
        return self.template_data.spike_length_samples

    def templates(
        self,
        drift: float = 0.0,
        up: bool = False,
        padded: bool = False,
        pad_value: float = float("nan"),
    ):
        assert not drift
        loc = self.template_data.template_locations()
        assert loc.shape[1] == 2
        loc = np.c_[loc[:, 0], np.zeros_like(loc[:, 0]), loc[:, 1]]
        temps = self.templates_up if up else self.template_data.templates
        if padded:
            zpads = [(0, 0)] * (temps.ndim - 1)
            temps = np.pad(temps, [*zpads, (0, 1)], constant_values=pad_value)
        off = self.offsets_up if up else self.offsets
        return loc, temps, off


class PointSource3ExpSimulator(BaseTemplateSimulator):
    def __init__(
        self,
        geom,
        n_units,
        common_reference=False,
        temporal_jitter=1,
        temporal_jitter_kind: Literal["exact", "cubic"] = "cubic",
        min_rms_distance=0.0,
        force_no_offset=False,
        snr_adjustment=1.0,
        # timing params
        tip_before_min=0.1,
        tip_before_max=0.5,
        peak_after_min=0.2,
        peak_after_max=0.8,
        # width params
        trough_width_min=0.005,
        trough_width_max=0.025,
        tip_width_min=0.01,
        tip_width_max=0.05,
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
        decay_model="squared",
        seed: int | np.random.Generator = 0,
        dtype=np.float32,
    ):
        self.rg = np.random.default_rng(seed)
        self.dtype = dtype
        self.n_units = n_units
        self.temporal_jitter = temporal_jitter
        self.common_reference = common_reference
        self.snr_adjustment = snr_adjustment
        self.decay_model = decay_model

        self.geom = geom
        self.geom3 = geom
        if geom.shape[1] == 2:
            self.geom3 = np.zeros((geom.shape[0], 3))
            self.geom3[:, [0, 2]] = geom
        self.geom3 = self.geom3.astype(self.dtype)
        self.ms_before = ms_before
        self.ms_after = ms_after
        self.sampling_frequency = sampling_frequency

        self.waveform_kw = dict(
            snr_adjustment=snr_adjustment,
            tip_before_min=tip_before_min,
            tip_before_max=tip_before_max,
            peak_after_min=peak_after_min,
            peak_after_max=peak_after_max,
            trough_width_min=trough_width_min,
            trough_width_max=trough_width_max,
            tip_width_min=tip_width_min,
            tip_width_max=tip_width_max,
            peak_width_min=peak_width_min,
            peak_width_max=peak_width_max,
            tip_rel_max=tip_rel_max,
            peak_rel_max=peak_rel_max,
        )
        alpha_shape = alpha_var / alpha_mean
        alpha_scale = alpha_mean / alpha_shape
        self.location_kw = dict(
            pos_margin_um_x=pos_margin_um_x,
            pos_margin_um_z=pos_margin_um_z,
            orthdist_min_um=orthdist_min_um,
            orthdist_max_um=orthdist_max_um,
            alpha_shape=alpha_shape,
            alpha_scale=alpha_scale,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_family=alpha_family,
        )

        t_up = self.time_domain_ms(up=True)
        pos, alpha, sct, sct_up = simulate_point_source_templates(
            n_units=n_units,
            rg=self.rg,
            temporal_jitter_kind=temporal_jitter_kind,
            temporal_jitter=temporal_jitter,
            trough_offset_samples=self.trough_offset_samples(),
            time_domain=self.time_domain_ms(),
            time_domain_up=t_up,
            geom=self.geom3,
            location_kw=self.location_kw,
            singlechan_kw=self.waveform_kw,
            decay_model=decay_model,
            min_rms_distance=min_rms_distance,
            force_no_offset=force_no_offset,
            dtype=dtype,
        )
        self.template_pos = pos
        self.template_alpha = alpha
        self.offsets = sct[:, :, 0].argmin(1) - self.trough_offset_samples()
        print(f"{self.offsets=}")
        self.singlechan_templates = sct[..., 0]
        self.singlechan_templates_up = sct_up[..., 0]

        up_half = temporal_jitter // 2
        off_up = (np.arange(temporal_jitter) > up_half).astype(np.int32)
        off_up = self.offsets[:, None] - off_up[None, :]
        self.offsets_up = off_up
        assert (np.abs(sct.argmin(1) - self.trough_offset_samples()) <= 2).all()
        np.testing.assert_allclose(sct, sct_up[:, 0], atol=1e-15)

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
        if self.common_reference and geom3.shape[0] > 1:
            tunpad = templates[..., :-1] if padded else templates
            tunpad -= np.median(tunpad, axis=-1, keepdims=True)
        if padded:
            templates[..., -1] = pad_value
        off = self.offsets_up if up else self.offsets
        return pos, templates, off

    def trough_offset_samples(self):
        return int(self.ms_before * (self.sampling_frequency / 1000))

    def spike_length_samples(self):
        spike_len_ms = self.ms_before + self.ms_after
        length = int(spike_len_ms * (self.sampling_frequency / 1000))
        length = 2 * (length // 2) + 1
        return length

    def time_domain_ms(self, up=False):
        t1 = self.spike_length_samples()
        if up:
            nt = t1 * self.temporal_jitter
        else:
            nt = t1
        t = np.linspace(0.0, t1, dtype=self.dtype, endpoint=False, num=nt)
        t -= self.trough_offset_samples()
        t /= self.sampling_frequency / 1000
        return t


class TemplateLibrarySimulator(BaseTemplateSimulator):
    def __init__(
        self,
        geom: np.ndarray,
        templates_local,
        pos_local,
        radius=250.0,
        temporal_jitter=1,
        common_reference=False,
        trough_offset_samples=42,
        interp_method: Literal["griddata", "dart"] = "dart",
        griddata_method: str = "cubic",
        interp_params: InterpolationParams = default_interpolation_params,
    ):
        self.geom = geom
        self.geom_kdt = KDTree(geom)
        self.temporal_jitter = temporal_jitter
        self.common_reference = common_reference
        self.radius = radius
        self.interp_params = interp_params.normalize()
        self.griddata_method = griddata_method
        self.channel_index = make_channel_index(
            geom=geom, radius=radius, to_torch=False
        )
        self._trough_offset_samples = trough_offset_samples

        self.n_units = len(templates_local)
        self.templates_local = templates_local
        self.low_rank_templates = svd_compress_templates(templates_local, allow_na=True)
        self.temporal_up = upsample_multichan(
            self.low_rank_templates.temporal_components, temporal_jitter=temporal_jitter
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
        self.interp_method = interp_method
        if interp_method != "griddata":
            self.precomputed_data = interp_precompute(
                source_pos=pos_local, params=self.interp_params
            )
        else:
            self.precomputed_data = None

        _, _, offsets_up = self.templates(up=True)
        offsets_up = offsets_up - self.trough_offset_samples()
        self.offsets_up = offsets_up.reshape(self.n_units, temporal_jitter)

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
        interp_params: InterpolationParams = default_interpolation_params,
        **kwargs,
    ):
        rg = np.random.default_rng(seed)
        if templates.shape[0] > n_units:
            choices = rg.choice(len(templates), size=n_units, replace=False)
            templates = templates[choices]
        templates = templates.astype(dtype)

        assert np.isfinite(templates).all()
        channel_index = make_channel_index(geom, extract_radius, to_torch=False)
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
            interp_params=interp_params,
            **kwargs,
        )

    def interpolate_templates(self, source_pos, target_pos, unit_ids, up=False):
        # interpolate spatial components
        spatial_singular = self.spatial_singular[unit_ids]
        if self.interp_method == "dart":
            assert self.precomputed_data is not None
            out = kernel_interpolate(
                spatial_singular,
                source_pos,
                target_pos,
                params=self.interp_params,
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
                spatial_singular,
                source_pos,
                target_pos,
                out,
                method=self.griddata_method,
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
        target_pos = torch.asarray(tgeom[target_chans])

        templates = self.interpolate_templates(source_pos, target_pos, unit_ids, up=up)

        nu = len(templates)
        nc_out = len(self.geom) + 1
        nt = self.templates_local.shape[1]
        up_factor = self.temporal_jitter if up else 1
        out = np.zeros((nu, up_factor * nt, nc_out), dtype=self.templates_local.dtype)
        np.put_along_axis(out, target_chans[:, None, :], templates, axis=2)

        if up:
            out = out.reshape(nu, up_factor, nt, nc_out)

        if padded:
            out[..., -1] = pad_value

        if self.common_reference:
            tunpad = out[..., :-1]
            tunpad -= np.median(tunpad, axis=-1, keepdims=True)

        if not padded:
            out = out[..., :-1]

        out_flat = out.reshape(nu * up_factor, nt, nc_out - (not padded)) if up else out
        _, _, offsets = get_main_channels_and_alignments(templates=out_flat)
        if up:
            offsets = offsets.reshape(nu, up_factor)

        return true_template_pos, out, offsets


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


TemplateSimulator = Union[
    StaticTemplateSimulator, PointSource3ExpSimulator, TemplateLibrarySimulator
]


# point source library fns


def simulate_source_locations(
    *,
    size: int,
    geom: np.ndarray,
    pos_margin_um_x: float,
    pos_margin_um_z: float,
    orthdist_min_um: float,
    orthdist_max_um: float,
    alpha_shape: float,
    alpha_scale: float,
    alpha_min: float,
    alpha_max: float,
    alpha_family: str,
    rg: np.random.Generator,
    depth_order: bool = True,
    dtype: np.typing.DTypeLike,
):
    x_low = geom[:, 0].min() - pos_margin_um_x
    x_high = geom[:, 0].max() + pos_margin_um_x
    z_low = geom[:, -1].min() - pos_margin_um_z
    z_high = geom[:, -1].max() + pos_margin_um_z

    x = rg.uniform(x_low, x_high, size=size)
    z = rg.uniform(z_low, z_high, size=size)
    if depth_order:
        z.sort()

    orth = rg.uniform(orthdist_min_um, orthdist_max_um, size=size)

    if alpha_family == "gamma":
        alpha = rg.gamma(shape=alpha_shape, scale=alpha_scale, size=size)
    elif alpha_family == "uniform":
        alpha = rg.uniform(alpha_min, alpha_max, size=size)
    else:
        assert False

    pos = np.c_[x, orth, z].astype(dtype)
    alpha = alpha.astype(dtype)[:, None]
    return pos, alpha


def simulate_singlechan(
    *,
    size: int,
    time_domain: np.ndarray,
    time_domain_up: np.ndarray,
    trough_offset_samples: int,
    snr_adjustment: float = 1.0,
    temporal_jitter: int = 1,
    # timing params
    tip_before_min=0.1,
    tip_before_max=0.5,
    peak_after_min=0.2,
    peak_after_max=0.8,
    # width params
    trough_width_min=0.005,
    trough_width_max=0.025,
    tip_width_min=0.01,
    tip_width_max=0.05,
    peak_width_min=0.05,
    peak_width_max=0.2,
    # rel height params
    tip_rel_max=0.3,
    peak_rel_max=0.5,
    rg: np.random.Generator,
    dtype: np.typing.DTypeLike,
    up=False,
):
    """Simulate a trough-normalized 3-exp action potential."""
    t = time_domain_up if up else time_domain

    size_pad = (size, 1)

    tip = rg.uniform(-tip_before_max, -tip_before_min, size=size_pad)
    peak = rg.uniform(peak_after_min, peak_after_max, size=size_pad)
    tip_height = rg.uniform(high=tip_rel_max, size=size_pad)
    peak_height = rg.uniform(high=peak_rel_max, size=size_pad)
    trough_width = rg.uniform(trough_width_min, trough_width_max, size=size_pad)
    tip_width = rg.uniform(tip_width_min, tip_width_max, size=size_pad)
    peak_width = rg.uniform(peak_width_min, peak_width_max, size=size_pad)

    trough = -np.exp(-np.square(t) / (2 * trough_width))
    tip = np.exp(-np.square(t - tip) / (2 * tip_width))
    peak = np.exp(-np.square(t - peak) / (2 * peak_width))

    waveforms = trough + tip_height * tip + peak_height * peak
    center = trough_offset_samples
    if up:
        center = center * temporal_jitter
    waveforms /= -waveforms[..., center, None]

    waveforms = waveforms * snr_adjustment

    return t, waveforms.astype(dtype)


def _simulate_point_source_templates(
    *,
    n_units: int,
    rg: np.random.Generator,
    temporal_jitter_kind: str,
    temporal_jitter: int,
    trough_offset_samples: int,
    time_domain: np.ndarray,
    time_domain_up: np.ndarray,
    geom: np.ndarray,
    location_kw: dict,
    singlechan_kw: dict,
    dtype: np.typing.DTypeLike,
):
    pos, alpha = simulate_source_locations(
        size=n_units,
        geom=geom,
        **location_kw,
        rg=rg,
        dtype=dtype,
    )
    if temporal_jitter_kind == "exact":
        t, sct_up = simulate_singlechan(
            size=n_units,
            time_domain=time_domain,
            time_domain_up=time_domain_up,
            trough_offset_samples=trough_offset_samples,
            temporal_jitter=temporal_jitter,
            **singlechan_kw,
            rg=rg,
            dtype=dtype,
            up=True,
        )
        sct_up = sct_up.reshape(n_units, -1, temporal_jitter, 1)
        sct_up = sct_up.transpose(0, 2, 1, 3)
        sct = sct_up[:, 0]
    else:
        t, sct = simulate_singlechan(
            size=n_units,
            time_domain=time_domain,
            time_domain_up=time_domain_up,
            trough_offset_samples=trough_offset_samples,
            **singlechan_kw,
            rg=rg,
            dtype=dtype,
            up=False,
        )
        sct_up = upsample_singlechan(sct, temporal_jitter=temporal_jitter)
        sct = sct[..., None]
        sct_up = sct_up[..., None]
    return pos, alpha, sct, sct_up


def _get_separated_group(
    *,
    n_grab: int,
    pos: np.ndarray,
    alpha: np.ndarray,
    singlechan_templates: np.ndarray,
    threshold: float,
    geom3: np.ndarray,
    decay_model: str,
    rg: np.random.Generator,
):
    n = len(pos)
    x = singlechan_to_probe(
        pos=pos,
        alpha=alpha,
        waveforms=singlechan_templates,
        geom3=geom3,
        decay_model=decay_model,
    )
    d = pdist(x.reshape(n, -1), metric="sqeuclidean")
    d = np.sqrt(d / np.prod(x.shape[1:]))
    ii, jj = np.triu_indices(n, k=1)
    mask = np.flatnonzero(d >= threshold)
    ii = ii[mask]
    jj = jj[mask]

    # largest connected component (connected=separated)
    v = np.ones(ii.shape)
    coo = coo_array(((v, (ii, jj))), shape=(n, n))
    coo = coo + coo.T
    _, labels = connected_components(coo, directed=False, connection="strong")
    u, c = np.unique(labels, return_counts=True)
    ibig = np.argmax(c)
    if c[ibig] < n_grab:
        return None

    # grab n random ones in there
    in_big = np.flatnonzero(labels == u[ibig])
    if in_big.shape[0] > n_grab:
        in_big = rg.choice(in_big, size=n_grab, replace=False)
        in_big.sort()

    return in_big


def simulate_point_source_templates(
    *,
    n_units: int,
    rg: np.random.Generator,
    temporal_jitter_kind: str,
    temporal_jitter: int,
    trough_offset_samples: int,
    time_domain: np.ndarray,
    time_domain_up: np.ndarray,
    geom: np.ndarray,
    location_kw: dict,
    singlechan_kw: dict,
    decay_model: str,
    min_rms_distance: float = 0.0,
    oversampling: int = 4,
    sampling_growth: int = 2,
    max_oversampling: int = 40,
    max_tries: int = 512,
    force_no_offset: bool = False,
    dtype: np.typing.DTypeLike,
):
    if not min_rms_distance:
        oversampling = max_oversampling = sampling_growth = 1
    puff = oversampling

    for _ in range(max_tries):
        sample_n = min(n_units * max_oversampling, n_units * puff)
        pos, alpha, singlechan_templates, singlechan_templates_up = (
            _simulate_point_source_templates(
                temporal_jitter_kind=temporal_jitter_kind,
                temporal_jitter=temporal_jitter,
                trough_offset_samples=trough_offset_samples,
                time_domain=time_domain,
                time_domain_up=time_domain_up,
                geom=geom,
                location_kw=location_kw,
                singlechan_kw=singlechan_kw,
                n_units=sample_n,
                rg=rg,
                dtype=dtype,
            )
        )

        # reject misaligned ones
        if force_no_offset:
            valid = singlechan_templates[:, :, 0].argmin(1) == trough_offset_samples
            valid = np.flatnonzero(valid)
            if valid.size < n_units:
                continue
            pos = pos[valid]
            alpha = alpha[valid]
            singlechan_templates = singlechan_templates[valid]
            singlechan_templates_up = singlechan_templates_up[valid]

        # reject too-close pairs
        if min_rms_distance:
            choices = _get_separated_group(
                n_grab=n_units,
                pos=pos,
                alpha=alpha,
                singlechan_templates=singlechan_templates,
                threshold=min_rms_distance,
                geom3=geom,
                decay_model=decay_model,
                rg=rg,
            )
            if choices is None:
                puff = min(max_oversampling, puff * sampling_growth)
                continue
            pos = pos[choices]
            alpha = alpha[choices]
            singlechan_templates = singlechan_templates[choices]
            singlechan_templates_up = singlechan_templates_up[choices]

        break
    else:
        raise ValueError(f"Hit max template tries for {min_rms_distance=}.")

    return pos, alpha, singlechan_templates, singlechan_templates_up
