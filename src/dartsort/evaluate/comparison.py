from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from spikeinterface.comparison import GroundTruthComparison
from scipy.spatial import KDTree

from ..cluster import merge
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
        if gt_td is not None and tested_td is not None:
            gt_rg = gt_td.registered_geom
            tested_rg = tested_td.registered_geom
            if gt_rg is None:
                assert tested_rg is None
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
        return self.gt_analysis.unit_ids

    @property
    def agreement_scores(self):
        """Make sure that all GT and tested units are present."""
        if self._agreement_scores is not None:
            return self._agreement_scores
        a = self.comparison.agreement_scores.copy()
        gtids = np.arange(self.gt_analysis.unit_ids.max() + 1)
        tids = np.arange(self.tested_analysis.unit_ids.max() + 1)
        a = a.reindex(index=gtids, columns=tids, fill_value=0.0)
        self._agreement_scores = a
        return a
            

    def unit_amplitudes(self, unit_ids=None):
        return self.gt_analysis.unit_amplitudes(unit_ids=unit_ids)

    def get_match(self, gt_unit):
        return int(self.comparison.hungarian_match_12[gt_unit])

    def get_best_match(self, gt_unit):
        return int(self.comparison.best_match_12[gt_unit])

    def unit_info_dataframe(self, force_distances=False):
        amplitudes = self.gt_analysis.unit_amplitudes()
        firing_rates = self.gt_analysis.firing_rates()
        df = self.comparison.get_performance()
        df = df.astype(float)  # not sure what the problem was...
        df["gt_ptp_amplitude"] = amplitudes
        df["gt_firing_rate"] = firing_rates
        if self.has_templates and (force_distances or self.compute_distances):
            dist = np.nan_to_num(self.template_distances, nan=np.inf).min(axis=1)
            df["min_temp_dist"] = dist
        rec = []
        for uid in df.index:
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

    def nearby_gt_templates(self, gt_unit_id, n_neighbors=5):
        return self.gt_analysis.nearby_coarse_templates(
            gt_unit_id, n_neighbors=n_neighbors
        )

    def nearby_tested_templates(self, gt_unit_id, n_neighbors=5):
        gt_td = self.gt_analysis.coarse_template_data
        tested_td = self.tested_analysis.coarse_template_data

        gt_unit_ix = np.searchsorted(gt_td.unit_ids, gt_unit_id)
        assert gt_td.unit_ids[gt_unit_ix] == gt_unit_id

        unit_dists = self.template_distances[gt_unit_ix]
        distance_order = np.argsort(unit_dists)

        # assert distance_order[0] == unit_ix
        neighb_ixs = distance_order[:n_neighbors]
        neighb_ids = tested_td.unit_ids[neighb_ixs]
        neighb_dists = self.template_distances[gt_unit_ix, neighb_ixs[None, :]]
        neighb_coarse_templates = tested_td.templates[neighb_ixs]

        return neighb_ids, neighb_dists, neighb_coarse_templates

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
            sym_function=np.maximum,
            n_jobs=1,
            svd_compression_rank=10,
            device="cpu",
            min_spatial_cosine=0.1,
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
            radius_frames=delta_frames,
            show_progress=self.verbose,
        )
        c = greedy_res['counts']
        self._greedy_confusion = c
        u = (c.sum(0, keepdims=True) + c.sum(1, keepdims=True)) - c
        self._greedy_iou = c / u

        self._unsorted_detection = np.logical_not(greedy_res['gt_unmatched'])
        gtns = self.gt_analysis.sorting.n_spikes
        assert self._unsorted_detection.shape == (gtns,)

    def get_spikes_by_category(self, gt_unit, tested_unit=None):
        if tested_unit is None:
            tested_unit = self.get_match(gt_unit)

        # convert to global index space
        (gt_spike_labels,) = self.comparison.get_labels1(gt_unit)
        in_gt_unit = self.gt_analysis.in_unit(gt_unit)
        matched_gt_mask = gt_spike_labels == "TP"
        matched_gt_indices = in_gt_unit[matched_gt_mask]
        only_gt_indices = in_gt_unit[np.logical_not(matched_gt_mask)]

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
                    f"{matched_gt_mask.sum()=} {matched_tested_mask.sum()=}"
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
            fn_times_samples=self.gt_analysis.times_samples(only_gt_indices),
            fp_times_samples=self.tested_analysis.times_samples(only_tested_indices),
        )

    def get_raw_waveforms_by_category(
        self,
        gt_unit,
        tested_unit=None,
        max_samples_per_category=100,
        random_seed=0,
        channel_show_radius_um=75,
        trough_offset_samples=42,
        spike_length_samples=121,
        channel_dist_p=np.inf,
    ):
        rg = np.random.default_rng(random_seed)
        ind_groups = self.get_spikes_by_category(gt_unit, tested_unit=tested_unit)
        tested_unit = ind_groups["tested_unit"]

        # waveforms are read at GT unit max channel
        gt_max_chan = self.gt_analysis.unit_max_channel(gt_unit)
        waveform_kw = dict(
            max_count=max_samples_per_category,
            random_seed=rg,
            channel_show_radius_um=channel_show_radius_um,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            channel_dist_p=channel_dist_p,
            max_chan=gt_max_chan,
        )

        # return vars dict. lots of stuff going in here.
        w = dict(gt_unit=gt_unit, tested_unit=tested_unit, max_chan=gt_max_chan)

        # load TP waveforms
        # which, waveforms, max_chan, show_geom, show_channel_index
        (
            w["which_tp"],
            w["tp"],
            _,
            w["geom"],
            w["channel_index"],
        ) = self.gt_analysis.unit_raw_waveforms(
            gt_unit,
            which=ind_groups["matched_gt_indices"],
            **waveform_kw,
        )

        # load FN waveforms
        # which, waveforms, max_chan, show_geom, show_channel_index
        w["which_fn"], w["fn"], *_ = self.gt_analysis.unit_raw_waveforms(
            gt_unit,
            which=ind_groups["only_gt_indices"],
            **waveform_kw,
        )

        # load FP waveforms
        # which, waveforms, max_chan, show_geom, show_channel_index
        w["which_fp"], w["fp"], *_ = self.tested_analysis.unit_raw_waveforms(
            tested_unit,
            which=ind_groups["only_tested_indices"],
            **waveform_kw,
        )

        if self.unsorted_detection is None:
            w["unsorted_tp"] = w["unsorted_fn"] = None
        else:
            w["which_unsorted_tp"], w["unsorted_tp"], *_ = self.gt_analysis.unit_raw_waveforms(
                gt_unit,
                which=ind_groups["unsorted_tp_indices"],
                **waveform_kw,
            )
            w["which_unsorted_fn"], w["unsorted_fn"], *_ = self.gt_analysis.unit_raw_waveforms(
                gt_unit,
                which=ind_groups["unsorted_fn_indices"],
                **waveform_kw,
            )

        return w
