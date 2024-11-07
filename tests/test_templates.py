import tempfile
from pathlib import Path

import numpy as np
import spikeinterface.core as sc
from dartsort import config
from dartsort.templates import (get_templates, pairwise, pairwise_util,
                                template_util, templates)
from dartsort.util.data_util import DARTsortSorting
from dredge.motion_util import IdentityMotionEstimate, get_motion_estimate
from test_util import no_overlap_recording_sorting


def test_roundtrip(tmp_path):
    rg = np.random.default_rng(0)
    temps = rg.normal(size=(11, 121, 384)).astype(np.float32)
    template_data = templates.TemplateData.from_config(
        *no_overlap_recording_sorting(temps, pad=0),
        template_config=config.TemplateConfig(
            low_rank_denoising=False,
            superres_bin_min_spikes=0,
            realign_peaks=False,
        ),
        motion_est=IdentityMotionEstimate(),
        n_jobs=0,
        save_folder=tmp_path,
        overwrite=True,
    )
    assert np.array_equal(template_data.templates, temps)


def test_static_templates():
    rec0 = np.zeros((11, 10))
    # geom = np.c_[np.zeros(10), np.arange(10).astype(float)]
    rec0[0, 1] = 1
    rec0[3, 5] = 2
    rec0 = sc.NumpyRecording(rec0, 1)
    # rec0.set_dummy_probe_from_locations(geom)

    sorting = DARTsortSorting(
        times_samples=[0, 2], labels=[0, 1], channels=[1, 5], sampling_frequency=1
    )

    with tempfile.TemporaryDirectory() as tdir:
        rec1 = rec0.save_to_folder(Path(tdir) / "rec")
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
            assert temps.shape == (2, 2, 10)

            assert temps[0, 0, 1] == 1
            temps[0, 0, 1] -= 1
            assert temps[1, 1, 5] == 2
            temps[1, 1, 5] -= 2
            assert np.all(temps == 0)


def test_drifting_templates():
    geom = np.c_[np.zeros(7), np.arange(7).astype(float)]
    rec0 = np.zeros((11, 7))
    rec0[0, 1] = 1
    rec0[2, 2] = 1
    rec0[6, 5] = 2
    rec0[8, 6] = 2
    rec0 = sc.NumpyRecording(rec0, 1)
    rec0.set_dummy_probe_from_locations(geom)

    with tempfile.TemporaryDirectory() as tdir:
        rec1 = rec0.save_to_folder(Path(tdir) / "rec")
        for rec in [rec0, rec1]:
            me = get_motion_estimate(
                0.5 * np.arange(11), time_bin_centers_s=np.arange(11).astype(float)
            )

            sorting = DARTsortSorting(
                times_samples=[0, 2, 6, 8],
                labels=[0, 0, 1, 1],
                channels=[0, 0, 0, 0],
                sampling_frequency=1,
            )
            t_s = [0, 2, 6, 8]

            res = template_util.get_registered_templates(
                rec,
                sorting,
                spike_times_s=t_s,
                spike_depths_um=[0, 0, 0, 0],
                geom=geom,
                motion_est=me,
                registered_template_depths_um=[0, 0],
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
        times_samples=[0, 2, 6, 8],
        labels=[0, 0, 1, 1],
        channels=[0, 0, 0, 0],
        sampling_frequency=1,
        extra_features=dict(
            point_source_localizations=np.zeros((4, 4)), times_seconds=[0, 2, 6, 8]
        ),
    )
    tdata = templates.TemplateData.from_config(
        rec,
        sorting,
        config.TemplateConfig(
            realign_peaks=False,
            superres_templates=False,
            denoising_rank=2,
        ),
        motion_est=me,
        trough_offset_samples=0,
        spike_length_samples=2,
    )


def test_pconv():
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
        unit_ids=np.array([0, 0, 1, 1, 2]),
        spike_counts=np.ones(5),
        registered_geom=None,
        registered_template_depths_um=None,
    )
    svd_compressed = template_util.svd_compress_templates(temps, rank=1)
    ctempup = template_util.compressed_upsampled_templates(
        svd_compressed.temporal_components,
        ptps=np.ptp(temps, 1).max(1),
        max_upsample=1,
        kind="cubic",
    )

    with tempfile.TemporaryDirectory() as tdir:
        pconvdb_path = pairwise_util.compressed_convolve_to_h5(
            Path(tdir) / "test.h5",
            geom=geom,
            template_data=tdata,
            low_rank_templates=svd_compressed,
            compressed_upsampled_temporal=ctempup,
        )
        pconvdb = pairwise.CompressedPairwiseConv.from_h5(pconvdb_path)
        assert (pconvdb.pconv[0] == 0.0).all()

        for tixa in range(5):
            for tixb in range(5):
                ixa, ixb, pconv = pconvdb.query(tixa, tixb)
                if (tixa, tixb) not in overlaps:
                    assert not ixa.numel()
                    assert not ixb.numel()
                    assert not pconv.numel()
                    continue

                olap = overlaps[tixa, tixb]
                assert (ixa, ixb) == (tixa, tixb)
                assert np.isclose(pconv.max(), olap)

    # drifting version
    # rigid drift from -1 to 0 to 1, note pitch=1
    # same templates but padded
    print(f"--------- rigid drift")
    tempspad = np.pad(temps, [(0, 0), (0, 0), (1, 1)])
    svd_compressed = template_util.svd_compress_templates(tempspad, rank=1)
    reg_geom = np.c_[np.zeros(c + 2), np.arange(c + 2).astype(float)]
    tdata = templates.TemplateData(
        templates=tempspad,
        unit_ids=np.array([0, 0, 1, 1, 2]),
        spike_counts=np.ones(5),
        registered_geom=reg_geom,
        registered_template_depths_um=np.zeros(5),
    )
    geom = np.c_[np.zeros(c), np.arange(1, c + 1).astype(float)]
    motion_est = get_motion_estimate(time_bin_centers_s=np.array([0., 1, 2]), displacement=[-1., 0, 1])

    # visualize shifted temps
    # for tix in range(5):
    #     for shift in (-1, 0, 1):
    #         spatial_shifted = drift_util.get_waveforms_on_static_channels(
    #             spat[tix][None],
    #             reg_geom,
    #             n_pitches_shift=np.array([shift]),
    #             registered_geom=geom,
    #             fill_value=0.0,
    #         )
    #         print(f"{shift=}")
    #         print(f"{spatial_shifted=}")

    with tempfile.TemporaryDirectory() as tdir:
        pconvdb_path = pairwise_util.compressed_convolve_to_h5(
            Path(tdir) / "test.h5",
            geom=geom,
            template_data=tdata,
            low_rank_templates=svd_compressed,
            compressed_upsampled_temporal=ctempup,
            motion_est=motion_est,
            chunk_time_centers_s=[0, 1, 2],
        )
        pconvdb = pairwise.CompressedPairwiseConv.from_h5(pconvdb_path)
        assert (pconvdb.pconv[0] == 0.0).all()
        print(f"{pconvdb.pconv.shape=}")

        for tixa in range(5):
            for tixb in range(5):
                ixa, ixb, pconv = pconvdb.query(tixa, tixb, shifts_a=0, shifts_b=0)

                if (tixa, tixb) not in overlaps:
                    assert not ixa.numel()
                    assert not ixb.numel()
                    assert not pconv.numel()
                    continue

                olap = overlaps[tixa, tixb]
                assert (ixa, ixb) == (tixa, tixb)
                assert np.isclose(pconv.max(), olap)

        for tixb in range(5):
            for shiftb in (-1, 0, 1):
                ixa, ixb, pconv = pconvdb.query(0, tixb, shifts_a=-1, shifts_b=shiftb)
                assert not ixa.numel()
                assert not ixb.numel()
                assert not pconv.numel()

        for tixb in range(5):
            for shift in (-1, 0, 1):
                ixa, ixb, pconv = pconvdb.query(4, tixb, shifts_a=shift, shifts_b=shift)
                if tixb != 4 or shift == 1:
                    assert not ixa.numel()
                    assert not ixb.numel()
                    assert not pconv.numel()
                else:
                    assert np.isclose(pconv.max(), 4 if shift < 1 else 0)
                ixa, ixb, pconv = pconvdb.query(tixb, 4, shifts_a=shift, shifts_b=shift)
                if tixb != 4 or shift == 1:
                    assert not ixa.numel()
                    assert not ixb.numel()
                    assert not pconv.numel()
                else:
                    assert np.isclose(pconv.max(), 4)

        for shifta in (-1, 0, 1):
            for shiftb in (-1, 0, 1):
                for tixa in range(5):
                    for tixb in range(5):
                        ixa, ixb, pconv = pconvdb.query(tixa, tixb, shifts_a=shifta, shifts_b=shiftb)
                        if shifta != shiftb:
                            # this is because we are rigid here
                            assert not ixa.numel()
                            assert not ixb.numel()
                            assert not pconv.numel()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tdir:
        test_roundtrip(Path(tdir))
    # test_static_templates()
    # test_drifting_templates()
    # test_main_object()
    # test_pconv()
