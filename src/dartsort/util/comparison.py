from dataclasses import dataclass

import numpy as np
from spikeinterface.comparison import GroundTruthComparison

from ..cluster import merge
from .analysis import DARTsortAnalysis


@dataclass
class DARTsortGroundTruthComparison:
    gt_analysis: DARTsortAnalysis
    tested_analysis: DARTsortAnalysis
    gt_name: str
    tested_name: str

    delta_time: float = 0.4
    match_score: float = 0.1
    well_detected_score: float = 0.8
    exhaustive_gt: bool = False
    n_jobs: int = -1
    match_mode: str = "hungarian"

    def __post_init__(self):
        self.comparison = GroundTruthComparison(
            gt_sorting=self.gt_analysis.sorting.to_numpy_sorting(),
            tested_sorting=self.tested_analysis.sorting.to_numpy_sorting(),
            gt_name=self.gt_name,
            tested_name=self.tested_name,
            delta_time=self.delta_time,
            match_score=self.match_score,
            well_detected_score=self.well_detected_score,
            exhaustive_gt=self.exhaustive_gt,
            n_jobs=self.n_jobs,
            match_mode=self.match_mode,
        )

    def get_match(self, gt_unit):
        return self.comparison.best_match_12[gt_unit]

    def unit_info_dataframe(self):
        amplitudes = self.gt_analysis.unit_amplitudes()
        firing_rates = self.gt_analysis.firing_rates()
        df = self.comparison.get_performance()
        df = df.astype(float)  # not sure what the problem was...
        df['gt_ptp_amplitude'] = amplitudes
        df['gt_firing_rate'] = firing_rates
        df['gt_match_temp_dist'] = self.template_distances[:, self.comparison.best_match_12]
        return df

    @property
    def template_distances(self):
        self._calculate_template_distances()
        return self._template_distances

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
        neighb_dists = self.template_dist[gt_unit_ix, neighb_ixs[None, :]]
        neighb_coarse_templates = tested_td.templates[neighb_ixs]

        return neighb_ids, neighb_dists, neighb_coarse_templates

    def _calculate_template_distances(self):
        """Compute the merge distance matrix"""
        if hasattr(self, "template_distances"):
            return

        gt_td = self.gt_analysis.coarse_template_data
        tested_td = self.tested_analysis.coarse_template_data

        dists, shifts, snrs_a, snrs_b = merge.cross_match_distance_matrix(
            gt_td,
            tested_td,
            sym_function=np.maximum,
            n_jobs=self.n_jobs,
        )
        self.template_distances = dists

    def get_spikes_by_category(self, gt_unit, tested_unit=None):
        # TODO
        if tested_unit is None:
            tested_unit = self.get_match(gt_unit)

        gt_spike_labels = self.comparison.get_labels1(gt_unit)
        pred_spike_labels = self.comparison.get_labels2(tested_unit)

        return dict(
            matched_tested_indices=...,
            matched_gt_indices=...,
            only_gt_indices=...,
            only_tested_indices=...,
        )

    def get_waveforms_by_category(self, gt_unit, tested_unit=None):
        # TODO
        return ...
