from itertools import product
from pathlib import Path
import pytest
import tempfile

import numpy as np
import spikeinterface.core as sc
import torch

import dartsort
from dartsort.evaluate import simkit, config_grid
from dartsort.peel.matching_util import pairwise, pairwise_util
from dartsort.templates import (
    get_templates,
    template_util,
    templates,
    TemplateData,
)
from dartsort.util.data_util import DARTsortSorting
from dartsort.util.internal_config import TemplateConfig
from dredge.motion_util import IdentityMotionEstimate, get_motion_estimate
from test_util import no_overlap_recording_sorting


# simkit fixture based test of all algorithms with a global
# refractory sorting that has no noise
# can do one with tiny drift and do allclose to test motion pipeline
@pytest.fixture
def refractory_simulations(tmp_path_factory):
    sim_settings = config_grid(
        common_params=dict(
            probe_kwargs=dict(
                num_columns=2, num_contact_per_column=12, y_shift_per_column=None
            ),
            temporal_jitter=1,
            amplitude_jitter=0,
            common_reference=False,
            refractory_ms=4.5,
            noise_kind="zero",
            duration_seconds=1.9,
            n_units=16,
            min_fr_hz=20.0,
            max_fr_hz=31.0,
            globally_refractory=True,
        ),
        config_cls=None,
        drift={"y": dict(drift_speed=1e-4), "n": dict(drift_speed=0.0)},
    )
    assert len(sim_settings) == 2

    simulations = {}
    for sim_name, kw in sim_settings.items():
        p = tmp_path_factory.mktemp(f"simdata_{sim_name}")
        simulations[sim_name] = simkit.generate_simulation(p / "sim", p / "noise", **kw)

    return simulations


@pytest.mark.parametrize("denoising_method", ["none", "t"])
@pytest.mark.parametrize("drift", [False, 0, True])
@pytest.mark.parametrize(
    "realign_peaks", [False, "mainchan_trough_factor", "normsq_weighted_trough_factor"]
)
@pytest.mark.parametrize("reduction", ["mean", "median"])
@pytest.mark.parametrize("algorithm", ["by_unit", "by_chunk", "chunk_if_mean"])
def test_refractory_templates(
    refractory_simulations, drift, realign_peaks, reduction, algorithm, denoising_method
):
    sim_name = f"drift{'y' if drift else 'n'}"
    sim = refractory_simulations[sim_name]

    if denoising_method != "none" and reduction == "median":
        return
    if denoising_method != "none" and algorithm == "by_unit":
        return

    template_cfg = TemplateConfig(
        registered_templates=drift is not False,
        realign_peaks=bool(realign_peaks),
        realign_strategy=realign_peaks if realign_peaks else "mainchan_trough_factor",
        reduction=reduction,
        algorithm=algorithm,
        denoising_method=denoising_method,
        with_raw_std_dev=True,
        use_zero=denoising_method == "t",
    )
    td = TemplateData.from_config(
        recording=sim["recording"],
        sorting=sim["sorting"],
        motion_est=sim["motion_est"],
        template_cfg=template_cfg,
    )

    unit_ids, spike_counts = np.unique(sim["sorting"].labels, return_counts=True)
    np.testing.assert_array_equal(td.unit_ids, unit_ids)
    np.testing.assert_array_equal(td.spike_counts, spike_counts)
    assert td.spike_counts_by_channel is not None
    np.testing.assert_array_equal(
        np.nanmax(td.spike_counts_by_channel, 1), spike_counts
    )

    atol = 5e-2 if denoising_method == "none" else 0.5
    np.testing.assert_allclose(td.templates, sim["templates"].templates, atol=atol)
    assert td.raw_std_dev is not None
    np.testing.assert_allclose(td.raw_std_dev, 0.0, atol=atol)
    np.testing.assert_array_equal(td.unit_ids, sim["templates"].unit_ids)


@pytest.mark.parametrize("drift", [False, 0, True])
@pytest.mark.parametrize(
    "realign_peaks", [False, "mainchan_trough_factor", "normsq_weighted_trough_factor"]
)
@pytest.mark.parametrize("denoising_method", ["none", "exp_weighted", "t", "loot"])
def test_refractory_templates_algorithm_agreement(
    refractory_simulations, drift, realign_peaks, denoising_method
):
    sim_name = f"drift{'y' if drift else 'n'}"
    sim = refractory_simulations[sim_name]

    tsvd = None
    if denoising_method != "none":
        tsvd = get_templates.fit_tsvd(sim["recording"], sim["sorting"])

    tds = []
    for algorithm in ("by_chunk", "by_unit"):
        if algorithm == "by_unit" and denoising_method in ("t", "loot"):
            continue
        template_cfg = TemplateConfig(
            registered_templates=drift is not False,
            realign_peaks=bool(realign_peaks),
            realign_strategy=realign_peaks
            if realign_peaks
            else "mainchan_trough_factor",
            reduction="mean",
            algorithm=algorithm,
            denoising_method=denoising_method,
            with_raw_std_dev=True,
        )
        td = TemplateData.from_config(
            recording=sim["recording"],
            sorting=sim["sorting"],
            motion_est=sim["motion_est"],
            template_cfg=template_cfg,
            tsvd=tsvd,
        )
        tds.append(td)

    if len(tds) == 1:
        return

    td0, td1 = tds

    np.testing.assert_array_equal(td0.unit_ids, td1.unit_ids)
    np.testing.assert_array_equal(td0.spike_counts, td1.spike_counts)
    np.testing.assert_array_equal(
        td0.spike_counts_by_channel, td1.spike_counts_by_channel
    )

    np.testing.assert_allclose(td0.templates, td1.templates, atol=1e-5)
    np.testing.assert_allclose(td0.raw_std_dev, td1.raw_std_dev, atol=1e-2)


@pytest.mark.parametrize("denoising_method", ("none",))
@pytest.mark.parametrize("algorithm", ("by_unit", "by_chunk"))
def test_roundtrip(tmp_path, algorithm, denoising_method):
    rg = np.random.default_rng(0)
    temps = rg.normal(size=(11, 121, 384)).astype(np.float32)
    rec, st = no_overlap_recording_sorting(temps, pad=0)
    assert st.labels is not None
    np.testing.assert_array_equal(np.unique(st.labels), np.arange(len(temps)))
    if algorithm == "by_unit" and denoising_method in ("loot", "t"):
        return
    template_data = templates.TemplateData.from_config(
        recording=rec,
        sorting=st,
        template_cfg=dartsort.TemplateConfig(
            denoising_method=denoising_method,
            superres_bin_min_spikes=0,
            use_svd=False,
            realign_peaks=False,
            algorithm=algorithm,
        ),
        motion_est=IdentityMotionEstimate(),
        save_folder=tmp_path,
        overwrite=True,
    )
    np.testing.assert_array_equal(template_data.unit_ids, np.arange(len(temps)))
    if denoising_method == "none":
        np.testing.assert_array_equal(template_data.templates, temps)
    else:
        np.testing.assert_allclose(template_data.templates, temps, atol=0.5)


def test_static_templates(tmp_path):
    rec0 = np.zeros((11, 10))
    # geom = np.c_[np.zeros(10), np.arange(10).astype(float)]
    rec0[0, 1] = 1
    rec0[3, 5] = 2
    rec0 = sc.NumpyRecording(rec0, 1)
    # rec0.set_dummy_probe_from_locations(geom)

    sorting = DARTsortSorting(
        times_samples=np.array([0, 2]),
        labels=np.arange(2),
        channels=np.array([1, 5]),
        sampling_frequency=1,
    )

    with tempfile.TemporaryDirectory(dir=tmp_path, ignore_cleanup_errors=True) as tdir:
        rec1 = rec0.save_to_folder(str(Path(tdir) / "rec"))
        for rec in [rec0, rec1]:
            res = get_templates.get_templates(
                rec,
                sorting,
                trough_offset_samples=0,
                spike_length_samples=2,
                realign_peaks=False,
                low_rank_denoising=False,
            )
            temps = res["raw_templates"]
            assert isinstance(temps, np.ndarray)
            assert temps.shape == (2, 2, 10)

            assert temps[0, 0, 1] == 1
            temps[0, 0, 1] -= 1
            assert temps[1, 1, 5] == 2
            temps[1, 1, 5] -= 2
            assert np.all(temps == 0)


def test_drifting_templates(tmp_path):
    geom = np.c_[np.zeros(7), np.arange(7).astype(float)]
    rec0 = np.zeros((11, 7))
    rec0[0, 1] = 1
    rec0[2, 2] = 1
    rec0[6, 5] = 2
    rec0[8, 6] = 2
    rec0 = sc.NumpyRecording(rec0, 1)
    rec0.set_dummy_probe_from_locations(geom)

    with tempfile.TemporaryDirectory(dir=tmp_path, ignore_cleanup_errors=True) as tdir:
        rec1 = rec0.save_to_folder(str(Path(tdir) / "rec"), n_jobs=1)
        for rec in [rec0, rec1]:
            me = get_motion_estimate(
                0.5 * np.arange(11), time_bin_centers_s=np.arange(11).astype(float)
            )

            sorting = DARTsortSorting(
                times_samples=np.array([0, 2, 6, 8]),
                labels=np.array([0, 0, 1, 1]),
                channels=np.array([0, 0, 0, 0]),
                sampling_frequency=1,
            )
            t_s = [0, 2, 6, 8]

            res = template_util.get_registered_templates(
                rec,
                sorting,
                spike_times_s=t_s,
                spike_depths_um=np.zeros(4),
                geom=geom,
                motion_est=me,
                trough_offset_samples=0,
                spike_length_samples=2,
                realign_peaks=False,
                low_rank_denoising=False,
                show_progress=False,
            )
            reg_temps = res["templates"]
            registered_geom = res["registered_geom"]

            temps0 = template_util.templates_at_time(
                0,
                reg_temps,
                geom,
                registered_template_depths_um=[0, 0],
                registered_geom=registered_geom,
                motion_est=me,
            )
            assert isinstance(temps0, np.ndarray)
            assert temps0.shape == (2, 2, 7)
            assert temps0[0, 0, 1] == 1
            assert temps0[1, 0, 2] == 2

            temps6 = template_util.templates_at_time(
                6,
                reg_temps,
                geom,
                registered_template_depths_um=[0, 0],
                registered_geom=registered_geom,
                motion_est=me,
            )
            assert isinstance(temps6, np.ndarray)
            assert temps6.shape == (2, 2, 7)
            assert temps6[0, 0, 4] == 1
            assert temps6[1, 0, 5] == 2

            temps8 = template_util.templates_at_time(
                8,
                reg_temps,
                geom,
                registered_template_depths_um=[0, 0],
                registered_geom=registered_geom,
                motion_est=me,
            )
            assert isinstance(temps8, np.ndarray)
            assert temps8.shape == (2, 2, 7)
            assert temps8[0, 0, 5] == 1
            assert temps8[1, 0, 6] == 2


def test_main_object():
    probe = np.c_[np.zeros(7), np.arange(7).astype(float)]
    rec = np.zeros((11, 7))
    rec[0, 1] = 1
    rec[2, 2] = 1
    rec[6, 5] = 2
    rec[8, 6] = 2
    rec = sc.NumpyRecording(rec, 1)
    rec.set_dummy_probe_from_locations(probe)

    me = get_motion_estimate(
        0.5 * np.arange(11), time_bin_centers_s=np.arange(11).astype(float)
    )
    sorting = DARTsortSorting(
        times_samples=np.array([0, 2, 6, 8]),
        labels=np.array([0, 0, 1, 1]),
        channels=np.array([0, 0, 0, 0]),
        sampling_frequency=1,
        ephemeral_features=dict(
            point_source_localizations=np.zeros((4, 4)),
            times_seconds=np.array([0, 2, 6, 8]),
        ),
    )
    tdata = templates.TemplateData.from_config(
        rec,
        sorting,
        dartsort.TemplateConfig(
            denoising_method="none",
            realign_peaks=False,
            superres_templates=False,
            denoising_rank=2,
        ),
        motion_est=me,
        waveform_cfg=dartsort.WaveformConfig(ms_before=0, ms_after=2000),
    )


@pytest.mark.parametrize("unit_ids", [np.arange(5), np.array([0, 0, 1, 1, 2])])
def test_pconv(tmp_path, unit_ids):
    # want to make sure drift handling is as expected
    # design an experiment

    # 4 chans, no drift
    # 3 units (superres): 0 (0,1), 1 (2,3), 3 (4)
    # temps overlap like:
    # 0        chan=0   z=0
    #  12           1     1
    #   23          2     2
    #     4         3     3
    t = 2
    c = 4
    temps = np.zeros((5, t, c), dtype=np.float32)
    temps[0, 0, 0] = 2
    temps[1, 0, 1] = 2
    temps[2, 0, [1, 2]] = 2
    temps[3, 0, 2] = 2
    temps[4, 0, 3] = 2
    geom = np.c_[np.zeros(c), np.arange(c).astype(float)]
    overlaps = {(i, i): np.square(temps[i]).sum() for i in range(5)}
    overlaps[(1, 2)] = overlaps[(2, 1)] = (temps[1] * temps[2]).sum()
    overlaps[(2, 3)] = overlaps[(3, 2)] = (temps[3] * temps[2]).sum()

    print(f"--------- no drift")
    tdata = templates.TemplateData(
        templates=temps,
        unit_ids=unit_ids,
        spike_counts=np.ones(5),
        registered_geom=geom,
        trough_offset_samples=0,
    )
    svd_compressed = template_util.svd_compress_templates(temps, rank=1)
    ctempup = template_util.compressed_upsampled_templates(
        svd_compressed.temporal_components,
        ptps=np.ptp(temps, 1).max(1),
        max_upsample=1,
        kind="cubic",
    )

    for motion_est, chunk_centers in [(None, None), (IdentityMotionEstimate(), [1, 2])]:
        with tempfile.TemporaryDirectory(
            dir=tmp_path, ignore_cleanup_errors=True
        ) as tdir:
            pconvdb_path = pairwise_util.compressed_convolve_to_h5(
                Path(tdir) / "test.h5",
                geom=geom,
                template_data=tdata,
                low_rank_templates=svd_compressed,
                compressed_upsampled_temporal=ctempup,
                motion_est=motion_est,
                chunk_time_centers_s=chunk_centers,
            )
            assert pconvdb_path == Path(tdir) / "test.h5"
            pconvdb = pairwise.CompressedPairwiseConv.from_h5(pconvdb_path)

            assert (pconvdb.b.pconv[0] == 0.0).all()
            assert pconvdb.b.shifts_a.shape == (1,)
            assert pconvdb.b.shifts_b.shape == (1,)
            assert (pconvdb.b.shifts_a == 0).all()
            assert (pconvdb.b.shifts_b == 0).all()
            assert pconvdb.b.shifted_template_index_a.shape[1] == 1
            assert torch.equal(
                pconvdb.b.shifted_template_index_a, torch.arange(5)[:, None]
            )
            assert torch.equal(
                pconvdb.b.upsampled_shifted_template_index_b,
                torch.arange(5)[:, None, None],
            )

            ixa, pconvs, which_b = pconvdb.query(torch.arange(5), torch.arange(5))
            ixb = torch.arange(5)[which_b]
            pairs = set()
            for ii, jj, pc in zip(ixa, ixb, pconvs):
                ii = ii.item()
                jj = jj.item()
                pairs.add((ii, jj))
                assert isinstance(ii, int)
                assert isinstance(jj, int)
                assert (ii, jj) in overlaps
                assert np.isclose(pc.max(), overlaps[ii, jj])
            assert pairs == set(overlaps.keys())

    # drifting version
    # rigid drift from -1 to 0 to 1, note pitch=1
    # same templates but padded
    print(f"--------- rigid drift")
    tempspad = np.pad(temps, [(0, 0), (0, 0), (1, 1)])
    svd_compressed = template_util.svd_compress_templates(tempspad, rank=1)
    reg_geom = np.c_[np.zeros(c + 2), np.arange(c + 2).astype(float)]
    tdata = templates.TemplateData(
        templates=tempspad,
        unit_ids=unit_ids,
        spike_counts=np.ones(5),
        registered_geom=reg_geom,
        trough_offset_samples=0,
    )
    geom = np.c_[np.zeros(c), np.arange(1, c + 1).astype(float)]
    motion_est = get_motion_estimate(
        time_bin_centers_s=np.array([0.0, 1, 2]), displacement=[-1.0, 0, 1]
    )

    with tempfile.TemporaryDirectory(dir=tmp_path, ignore_cleanup_errors=True) as tdir:
        pconvdb_path = pairwise_util.compressed_convolve_to_h5(
            Path(tdir) / "test.h5",
            geom=geom,
            template_data=tdata,
            low_rank_templates=svd_compressed,
            compressed_upsampled_temporal=ctempup,
            motion_est=motion_est,
            chunk_time_centers_s=np.array([0.0, 1, 2]),
        )
        pconvdb = pairwise.CompressedPairwiseConv.from_h5(pconvdb_path)
        assert (pconvdb.b.pconv[0] == 0.0).all()

        for shifta, shiftb in product((0, -1, 1), (0, -1, 1)):
            ixa, pconvs, which_b = pconvdb.query(
                torch.arange(5),
                torch.arange(5),
                shifts_a=torch.zeros(5, dtype=torch.long) + shifta,
                shifts_b=torch.zeros(5, dtype=torch.long) + shiftb,
            )
            ixb = torch.arange(5)[which_b]
            if shifta != shiftb:
                # rigid, impossible
                assert not ixa.numel()
                assert not pconvs.numel()
                assert not which_b.numel()
                continue

            pairs = set()
            for ii, jj, pc in zip(ixa, ixb, pconvs):
                ii = ii.item()
                jj = jj.item()
                pairs.add((ii, jj))
                assert isinstance(ii, int)
                assert isinstance(jj, int)
                assert (ii, jj) in overlaps
                assert np.isclose(pc.max(), overlaps[ii, jj])
            if shifta == 0:
                assert pairs == set(overlaps.keys())
            elif shifta == 1:
                # 4 falls off the edge
                assert pairs == (set(overlaps.keys()) - {(4, 4)})
            elif shifta == -1:
                # 0 falls off the edge
                assert pairs == (set(overlaps.keys()) - {(0, 0)})
            else:
                assert False
