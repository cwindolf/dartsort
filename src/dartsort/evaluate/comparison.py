from dataclasses import dataclass
from typing import Literal

import numpy as np
from spikeinterface.comparison import GroundTruthComparison
from scipy.spatial import KDTree

from ..cluster import merge
from .analysis import DARTsortAnalysis


@dataclass
class DARTsortGroundTruthComparison:
    gt_analysis: DARTsortAnalysis
    tested_analysis: DARTsortAnalysis

    delta_time: float = 0.8
    match_score: float = 0.1
    well_detected_score: float = 0.8
    exhaustive_gt: bool = False
    n_jobs: int = -1
    match_mode: str = "hungarian"
    compute_labels: bool = False
    verbose: bool = False
    device: str | None = None

    compute_distances: bool = False
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
            well_detected_score=self.well_detected_score,
            exhaustive_gt=self.exhaustive_gt,
            match_mode=self.match_mode,
            verbose=self.verbose,
            compute_labels=self.compute_labels,
        )
        self.has_templates = self.tested_analysis.template_data is not None

        if self.compute_distances and self.has_templates:
            if self.verbose:
                print("Calculating GT/tested template distances...")
            self._calculate_template_distances()

        self._unsorted_detection = None
        if self.compute_unsorted_recall:
            self._calculate_unsorted_detection()

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

    def get_match(self, gt_unit):
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
        self._calculate_unsorted_detection()
        return self._unsorted_detection

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

    def _calculate_unsorted_detection(self):
        if self._unsorted_detection is not None:
            return
        if self.verbose:
            print("Calculate unsorted detection...")

        frames_per_ms = self.gt_analysis.recording.sampling_frequency / 1000
        delta_frames = self.delta_time * frames_per_ms
        gt_chans = self.gt_analysis.sorting.channels
        gt_chanpos = self.gt_analysis.geom[gt_chans]
        gt_coord = np.c_[
            self.gt_analysis.sorting.times_samples / delta_frames,
            gt_chanpos / self.unsorted_match_radius,
        ]

        tested_chans = self.tested_analysis.sorting.channels
        tested_chanpos = self.tested_analysis.geom[tested_chans]
        tested_coord = np.c_[
            self.tested_analysis.sorting.times_samples / delta_frames,
            tested_chanpos / self.unsorted_match_radius,
        ]
        assert np.array_equal(self.gt_analysis.geom, self.tested_analysis.geom)

        tested_kdtree = KDTree(tested_coord)
        d, i = tested_kdtree.query(
            gt_coord, p=4, distance_upper_bound=1 + 1e-6, workers=4
        )
        has_match = i < tested_kdtree.n
        self._unsorted_detection = has_match

    def get_spikes_by_category(self, gt_unit, tested_unit=None):
        # TODO
        if tested_unit is None:
            tested_unit = self.get_match(gt_unit)

        (gt_spike_labels,) = self.comparison.get_labels1(gt_unit)
        (tested_spike_labels,) = self.comparison.get_labels2(tested_unit)

        # convert to global index space
        in_gt_unit = self.gt_analysis.in_unit(gt_unit)
        matched_gt_mask = gt_spike_labels == "TP"
        matched_gt_indices = in_gt_unit[matched_gt_mask]
        only_gt_indices = in_gt_unit[np.logical_not(matched_gt_mask)]

        in_tested_unit = self.tested_analysis.in_unit(tested_unit)
        matched_tested_mask = tested_spike_labels == "TP"
        matched_tested_indices = in_tested_unit[matched_tested_mask]
        only_tested_indices = in_tested_unit[np.logical_not(matched_tested_mask)]

        assert matched_gt_indices.size == matched_tested_indices.size

        return dict(
            tested_unit=tested_unit,
            matched_tested_indices=matched_tested_indices,
            matched_gt_indices=matched_gt_indices,
            only_gt_indices=only_gt_indices,
            only_tested_indices=only_tested_indices,
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

        return w
