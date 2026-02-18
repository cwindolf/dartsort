from dataclasses import dataclass
import string
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from spikeinterface.comparison import GroundTruthComparison
from tqdm.auto import tqdm

from ..clustering import merge
from .analysis import DARTsortAnalysis


@dataclass
class DARTsortGroundTruthComparison:
    gt_analysis: DARTsortAnalysis
    tested_analysis: DARTsortAnalysis

    delta_time: float = 0.8
    match_score: float = 0.0
    chance_score: float = 0.0
    well_detected_score: float = 0.8
    exhaustive_gt: bool = False
    n_jobs: int = -1
    match_mode: str = "hungarian"
    compute_labels: bool = True
    verbose: bool = False
    device: str | None = None

    compute_distances: bool = True
    compute_unsorted_recall: bool = True
    unsorted_match_radius: float = 50.0
    distance_kind: Literal["rms", "max", "deconv"] = "deconv"

    def __post_init__(self):
        self._check()
        self.comparison = GroundTruthComparison(
            gt_sorting=self.gt_analysis.sorting.to_numpy_sorting(),
            tested_sorting=self.tested_analysis.sorting.to_numpy_sorting(),
            gt_name=self.gt_analysis.name,
            tested_name=self.tested_analysis.name,
            delta_time=self.delta_time,
            match_score=self.match_score,
            chance_score=self.chance_score,
            well_detected_score=self.well_detected_score,
            exhaustive_gt=self.exhaustive_gt,
            match_mode=self.match_mode,
            verbose=self.verbose,
            compute_labels=self.compute_labels,
        )
        self.delta_frames = self.comparison.delta_frames
        self.has_templates = self.tested_analysis.template_data is not None
        self._agreement_scores = None

        if self.compute_distances and self.has_templates:
            if self.verbose:
                print("Calculating GT/tested template distances...")
            self._calculate_template_distances()

        self._unsorted_detection = self._greedy_confusion = None
        if self.compute_unsorted_recall:
            self._calculate_greedy_confusion_and_detection()

    def _check(self):
        gt_td = self.gt_analysis.template_data
        tested_td = self.tested_analysis.template_data
        if self.compute_distances and (gt_td is not None and tested_td is not None):
            gt_rg = gt_td.registered_geom
            tested_rg = tested_td.registered_geom
            if gt_rg is None:
                assert tested_rg is None
            assert gt_rg is not None
            assert tested_rg is not None
            if not np.array_equal(gt_rg, tested_rg):
                raise ValueError(
                    f"Template data had different registered geoms: "
                    f"{gt_rg.shape=} {tested_rg.shape=}"
                )

    @property
    def gt_name(self):
        return self.gt_analysis.name or "GT"

    @property
    def tested_name(self):
        return self.tested_analysis.name or "Tested"

    @property
    def unit_ids(self):
        return self.gt_analysis.sorting.unit_ids

    @property
    def n_gt_units(self):
        return len(self.gt_analysis.sorting.unit_ids)

    @property
    def agreement_scores(self):
        """Make sure that all GT and tested units are present."""
        if self._agreement_scores is not None:
            return self._agreement_scores
        a = self.comparison.agreement_scores.copy()
        gtids = np.arange(self.gt_analysis.sorting.unit_ids.max() + 1)
        tids = np.arange(self.tested_analysis.sorting.unit_ids.max() + 1)
        a = a.reindex(index=gtids, columns=tids, fill_value=0.0)
        self._agreement_scores = a
        return a

    def unit_amplitudes(self, unit_id):
        return self.gt_analysis.unit_amplitudes(unit_id)

    def get_match(self, gt_unit):
        assert self.comparison.hungarian_match_12 is not None
        return int(self.comparison.hungarian_match_12[gt_unit])

    def get_best_match(self, gt_unit):
        assert self.comparison.best_match_12 is not None
        return int(self.comparison.best_match_12[gt_unit])

    def unit_info_dataframe(self, force_distances=False, perf_only=False):
        amplitudes = self.gt_analysis.unit_amplitudes()
        firing_rates = self.gt_analysis.firing_rates()
        df = self.comparison.get_performance()
        assert isinstance(df, pd.DataFrame)
        df = df.astype(float)  # not sure what the problem was...
        df["gt_ptp_amplitude"] = amplitudes
        df["gt_firing_rate"] = firing_rates
        if perf_only:
            return df

        # coll, matched_coll, missed_coll = self.unit_collidedness()
        # assert coll.shape == matched_coll.shape == missed_coll.shape == df.index.shape
        # df["gt_collidedness"] = coll
        # df["gt_matched_collidedness"] = matched_coll
        # df["gt_missed_collidedness"] = missed_coll
        try:
            df["gt_dt_rms"] = self.unit_matched_misalignment_rms()
        except ValueError:
            pass
        if self.has_templates and (force_distances or self.compute_distances):
            dist = np.nan_to_num(self.template_distances, nan=np.inf).min(axis=1)
            df["min_temp_dist"] = dist
        rec = []
        for uid in df.index:
            assert self.unsorted_detection is not None
            rec.append(self.unsorted_detection[self.gt_analysis.in_unit(uid)].mean())
        df["unsorted_recall"] = rec
        return df

    @property
    def template_distances(self):
        self._calculate_template_distances()
        return self._template_distances

    @property
    def unsorted_detection(self):
        self._calculate_greedy_confusion_and_detection()
        return self._unsorted_detection

    @property
    def greedy_confusion(self):
        self._calculate_greedy_confusion_and_detection()
        return self._greedy_confusion

    @property
    def greedy_iou(self):
        self._calculate_greedy_confusion_and_detection()
        return self._greedy_iou

    def unit_collidedness(self):
        uids = self.gt_analysis.sorting.unit_ids
        c = np.full(len(uids), np.nan)
        matched_c = c.copy()
        missed_c = c.copy()

        collidedness = getattr(self.gt_analysis.sorting, "collidedness", None)
        if collidedness is None:
            return c, matched_c, missed_c

        for j, uid in enumerate(uids):
            inu, matchu, missu = self.matched_and_missed(uid)
            if inu.size:
                c[j] = collidedness[inu].mean()
            if matchu.size:
                matched_c[j] = collidedness[matchu].mean()
            if missu.size:
                missed_c[j] = collidedness[missu].mean()

        return c, matched_c, missed_c

    def unit_matched_misalignment_rms(self):
        uids = self.gt_analysis.sorting.unit_ids
        match_dt_rms = np.full(len(uids), np.nan)
        for j, uid in enumerate(uids):
            try:
                udt = self.matched_misalignment(uid)
                if udt is None:
                    continue
                if udt.size:
                    match_dt_rms[j] = np.sqrt(np.square(udt).mean())
            except ValueError as e:
                warnings.warn(
                    f"ValueError in misalignment. SI matching bug. {e=}"  # type: ignore
                )
        return match_dt_rms

    def matched_misalignment(self, gt_unit_id):
        spikes = self.get_spikes_by_category(gt_unit_id)
        gt_matched_t = self.gt_analysis.sorting.times_samples[
            spikes["matched_gt_indices"]
        ]
        test_matched_t = self.tested_analysis.sorting.times_samples[
            spikes["matched_tested_indices"]
        ]
        if test_matched_t.shape != gt_matched_t.shape:
            return None
        match_dt = test_matched_t - gt_matched_t
        return match_dt

    def nearby_gt_templates(self, gt_unit_id, n_neighbors=5):
        return self.gt_analysis.nearby_coarse_templates(
            gt_unit_id, n_neighbors=n_neighbors
        )

    def nearby_tested_templates(self, gt_unit_id, n_neighbors=5):
        gt_td = self.gt_analysis.coarse_template_data
        tested_td = self.tested_analysis.coarse_template_data
        assert gt_td is not None
        assert tested_td is not None

        gt_unit_ix = np.searchsorted(gt_td.unit_ids, gt_unit_id)
        assert gt_td.unit_ids[gt_unit_ix] == gt_unit_id

        unit_dists = self.template_distances[gt_unit_ix]
        distance_order = np.argsort(unit_dists)

        # assert distance_order[0] == unit_ix
        neighb_ixs = distance_order[:n_neighbors]
        neighb_ids = tested_td.unit_ids[neighb_ixs]
        neighb_dists = self.template_distances[gt_unit_ix, neighb_ixs[None, :]]
        neighb_coarse_templates = tested_td.templates[neighb_ixs]

        return neighb_ixs, neighb_ids, neighb_dists, neighb_coarse_templates

    def _calculate_template_distances(self):
        """Compute the merge distance matrix"""
        if hasattr(self, "_template_distances"):
            return

        (
            dists,
            shifts,
            snrs_a,
            snrs_b,
            a_mask,
            b_mask,
        ) = merge.cross_match_distance_matrix(
            self.gt_analysis.coarse_template_data,
            self.tested_analysis.coarse_template_data,
            sym_function=np.minimum,
            n_jobs=1,
            svd_compression_rank=10,
            device="cpu",
            min_spatial_cosine=0.7,
            distance_kind=self.distance_kind,
        )
        self._template_distances = dists

    def _calculate_greedy_confusion_and_detection(self):
        from ..evaluate.hybrid_util import greedy_match, greedy_match_counts

        if self._unsorted_detection is not None:
            return
        if self.verbose:
            print("Calculate unsorted detection...")
        frames_per_ms = self.gt_analysis.recording.sampling_frequency / 1000
        delta_frames = self.delta_time * frames_per_ms
        greedy_res = greedy_match_counts(
            self.gt_analysis.sorting,
            self.tested_analysis.sorting,
            radius_frames=int(delta_frames),
            show_progress=self.verbose,
        )
        c = greedy_res["counts"]
        self._greedy_confusion = c
        u = (c.sum(0, keepdims=True) + c.sum(1, keepdims=True)) - c
        self._greedy_iou = c / np.maximum(u, 1)

        self._tested_to_gt = greedy_res["test2gt_spike"]
        self._unsorted_detection = np.logical_not(greedy_res["gt_unmatched"])
        gtns = self.gt_analysis.sorting.n_spikes
        assert self._unsorted_detection.shape == (gtns,)

    def matched_and_missed(self, gt_unit):
        (gt_spike_labels,) = self.comparison.get_labels1(gt_unit)
        in_gt_unit = self.gt_analysis.in_unit(gt_unit)
        matched_gt_mask = gt_spike_labels == "TP"
        matched_gt_indices = in_gt_unit[matched_gt_mask]
        only_gt_indices = in_gt_unit[np.logical_not(matched_gt_mask)]
        return in_gt_unit, matched_gt_indices, only_gt_indices

    def get_spikes_by_category(self, gt_unit, tested_unit=None):
        if tested_unit is None:
            tested_unit = self.get_match(gt_unit)

        # convert to global index space
        in_gt_unit, matched_gt_indices, only_gt_indices = self.matched_and_missed(
            gt_unit
        )

        if tested_unit >= 0:
            (tested_spike_labels,) = self.comparison.get_labels2(tested_unit)
            in_tested_unit = self.tested_analysis.in_unit(tested_unit)
            matched_tested_mask = tested_spike_labels == "TP"
            matched_tested_indices = in_tested_unit[matched_tested_mask]
            only_tested_indices = in_tested_unit[np.logical_not(matched_tested_mask)]
            if matched_gt_indices.size != matched_tested_indices.size:
                # raise ValueError(
                warnings.warn(
                    f"Strange match sizes for {gt_unit=} {tested_unit=}: "
                    f"{matched_gt_indices.shape=} {matched_tested_indices.shape=} "
                    f"{matched_tested_mask.sum()=}"
                )
        else:
            matched_tested_indices = np.zeros(shape=(0,), dtype=np.int64)
            only_tested_indices = np.zeros(shape=(0,), dtype=np.int64)

        # add unsorted matches and misses
        if self.unsorted_detection is None:
            unsorted_tp_indices = unsorted_fn_indices = None
        else:
            unsorted_match_mask = self.unsorted_detection[in_gt_unit]
            unsorted_tp_indices = in_gt_unit[unsorted_match_mask]
            unsorted_fn_indices = in_gt_unit[np.logical_not(unsorted_match_mask)]

        return dict(
            tested_unit=tested_unit,
            matched_tested_indices=matched_tested_indices,
            matched_gt_indices=matched_gt_indices,
            only_gt_indices=only_gt_indices,
            only_tested_indices=only_tested_indices,
            unsorted_tp_indices=unsorted_tp_indices,
            unsorted_fn_indices=unsorted_fn_indices,
            fn_times_samples=self.gt_analysis.sorting.times_samples[only_gt_indices],
            fp_times_samples=self.tested_analysis.sorting.times_samples[
                only_tested_indices
            ],
        )

    def get_greedy_correspondence_in_si_match(self):
        tlabels = self.tested_analysis.sorting.labels
        gt_labels_for_tested = np.full_like(tlabels, -1)
        to_ms = 1000.0 / self.gt_analysis.recording.sampling_frequency
        for gtu in tqdm(self.gt_analysis.unit_ids, desc="Spike match"):
            tu = self.get_match(gtu)
            if tu < 0:
                continue
            in_gt = self.gt_analysis.in_unit(gtu)
            in_tu = self.tested_analysis.in_unit(tu)
            gt_times_ms = self.gt_analysis.sorting.times_samples[in_gt] * to_ms
            tested_times_ms = self.tested_analysis.sorting.times_samples[in_tu] * to_ms
            gt_kdt = KDTree(gt_times_ms[:, None])
            dd, ii = gt_kdt.query(
                tested_times_ms[:, None], distance_upper_bound=self.delta_time
            )
            ii = np.atleast_1d(ii)
            gt_labels_for_tested[in_tu[ii < gt_kdt.n]] = in_gt[ii[ii < gt_kdt.n]]

        return dict(
            gt_labels_for_tested=gt_labels_for_tested,
        )

    def get_raw_waveforms_by_category(
        self,
        gt_unit,
        tested_unit=None,
        max_samples_per_category=100,
        random_seed=0,
    ):
        rg = np.random.default_rng(random_seed)
        ind_groups = self.get_spikes_by_category(gt_unit, tested_unit=tested_unit)
        tested_unit = ind_groups["tested_unit"]

        # waveforms are read at GT unit max channel
        gt_max_chan = self.gt_analysis.unit_max_channel(gt_unit)
        waveform_kw = dict(
            max_count=max_samples_per_category, random_seed=rg, main_channel=gt_max_chan
        )

        # return vars dict. lots of stuff going in here.
        w: dict[str, int | np.ndarray | None] = dict(
            gt_unit=gt_unit, tested_unit=tested_unit, max_chan=gt_max_chan
        )

        # load TP waveforms
        # which, waveforms, max_chan, show_geom, show_channel_index
        tp_waves = self.gt_analysis.unit_raw_waveforms(
            which=ind_groups["matched_gt_indices"],  # type: ignore
            **waveform_kw,  # type: ignore
        )
        if tp_waves is None:
            w["which_tp"] = None
            w["tp"] = None
            w["geom"] = self.gt_analysis.registered_geom
            w["channel_index"] = self.gt_analysis.vis_channel_index
        else:
            w["which_tp"] = tp_waves.which
            w["tp"] = tp_waves.waveforms
            w["geom"] = tp_waves.geom
            w["channel_index"] = tp_waves.channel_index

        # load FN waveforms
        # which, waveforms, max_chan, show_geom, show_channel_index
        fn_waves = self.gt_analysis.unit_raw_waveforms(
            which=ind_groups["only_gt_indices"],  # type: ignore
            **waveform_kw,  # type: ignore
        )
        if fn_waves is None:
            w["which_fn"] = None
            w["fn"] = None
        else:
            w["which_fn"] = fn_waves.which
            w["fn"] = fn_waves.waveforms

        # load FP waveforms
        # which, waveforms, max_chan, show_geom, show_channel_index
        fp_waves = self.tested_analysis.unit_raw_waveforms(
            which=ind_groups["only_tested_indices"],  # type: ignore
            **waveform_kw,  # type: ignore
        )
        if fp_waves is None:
            w["which_fp"] = None
            w["fp"] = None
        else:
            w["which_fp"] = fp_waves.which
            w["fp"] = fp_waves.waveforms

        if self.unsorted_detection is None:
            w["unsorted_tp"] = w["unsorted_fn"] = None
        else:
            utp_waves = self.gt_analysis.unit_raw_waveforms(
                which=ind_groups["unsorted_tp_indices"],  # type: ignore
                **waveform_kw,  # type: ignore
            )
            if utp_waves is None:
                w["which_unsorted_tp"] = None
                w["unsorted_tp"] = None
            else:
                w["which_unsorted_tp"] = utp_waves.which
                w["unsorted_tp"] = utp_waves.waveforms
            ufn_waves = self.gt_analysis.unit_raw_waveforms(
                which=ind_groups["unsorted_fn_indices"],  # type: ignore
                **waveform_kw,  # type: ignore
            )
            if ufn_waves is None:
                w["which_unsorted_fn"] = None
                w["unsorted_fn"] = None
            else:
                w["which_unsorted_fn"] = ufn_waves.which
                w["unsorted_fn"] = ufn_waves.waveforms

        return w


class DARTsortGTVersus:
    default_ids = string.ascii_uppercase

    def __init__(
        self,
        gt_analysis: DARTsortAnalysis,
        *other_analyses: DARTsortAnalysis,
        sorter_var="sorter",
        comparison_kw=None,
        comparisons=None,
    ):
        comparison_kw = comparison_kw or {}
        self.sorter_var = sorter_var

        # some things we can only do for head to head comparisons
        # but sometimes it's a battle royale :P
        self.is_two = len(other_analyses) == 2
        self.n_vs = len(other_analyses)

        self.gt_name = gt_analysis.name or "GT"
        self.gt_templates = gt_analysis.template_data
        self.gt_sorting = gt_analysis.sorting
        self.gt_analysis = gt_analysis

        self.other_analyses = other_analyses
        self.other_names = [oa.name for oa in other_analyses]
        self.other_templates = [
            (oa.name or f"Test{c}") for oa, c in zip(other_analyses, self.default_ids)
        ]
        self.other_sortings = [oa.sorting for oa in other_analyses]

        if comparisons is None:
            comparisons = [None] * self.n_vs
        else:
            assert len(comparisons) == self.n_vs

        self.cmps = []
        for cmp, oa in zip(comparisons, other_analyses):
            if cmp is not None:
                assert cmp.tested_analysis.name == oa.name
            else:
                cmp = DARTsortGroundTruthComparison(
                    gt_analysis=gt_analysis, tested_analysis=oa, **comparison_kw
                )
            self.cmps.append(cmp)

        if self.is_two:
            self.a_name, self.b_name = self.other_names
            self.a_templates, self.b_templates = self.other_templates
            self.a_sorting, self.b_sorting = self.other_sortings
            self.a_cmp, self.b_cmp = self.cmps
        self._unit_vs_df = None
        self.n_gt_units = self.cmps[0].n_gt_units

    def unit_versus_dataframe(self) -> pd.DataFrame:
        """Combine the performance into one dataframe."""
        if self._unit_vs_df is not None:
            return self._unit_vs_df.copy(deep=True)
        dfs = [ocmp.unit_info_dataframe().reset_index() for ocmp in self.cmps]
        for df, sorter in zip(dfs, self.other_names):
            df[self.sorter_var] = sorter
        self._unit_vs_df = pd.concat(dfs, axis=0)
        return self._unit_vs_df.copy(deep=True)

    def tagname(self):
        vsstr = "vs_" + ",".join(
            oa.name or str(j) for j, oa in enumerate(self.other_analyses)
        )
        return f"{self.gt_name}_{vsstr}"
