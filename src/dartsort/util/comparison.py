from dataclasses import dataclass

from spikeinterface.comparison import GroundTruthComparison

from .analysis import DARTsortAnalysis


@dataclass
class DARTsortGroundTruthComparison:
    gt_analysis: DARTsortAnalysis
    predicted_analysis: DARTsortAnalysis
    gt_name: str
    predicted_name: str

    delta_time: float = 0.4
    match_score: float = 0.1
    well_detected_score: float = 0.8
    exhaustive_gt: bool = False
    n_jobs: int = -1
    match_mode: str = "hungarian"

    def __post_init__(self):
        self.comparison = GroundTruthComparison(
            gt_sorting=self.gt_analysis.sorting.to_numpy_sorting(),
            tested_sorting=self.predicted_analysis.sorting.to_numpy_sorting(),
            gt_name=self.gt_name,
            predicted_name=self.predicted_name,
            delta_time=self.delta_time,
            match_score=self.match_score,
            well_detected_score=self.well_detected_score,
            exhaustive_gt=self.exhaustive_gt,
            n_jobs=self.n_jobs,
            match_mode=self.match_mode,
        )

    def get_match(self, gt_unit):
        return self.comparison.best_match_12[gt_unit]

    def get_spikes_by_category(self, gt_unit, predicted_unit=None):
        if predicted_unit is None:
            predicted_unit = self.get_match(gt_unit)

        gt_spike_labels = self.comparison.get_labels1(gt_unit)
        pred_spike_labels = self.comparison.get_labels2(predicted_unit)

        return dict(
            matched_predicted_indices=...,
            matched_gt_indices=...,
            only_gt_indices=...,
            only_predicted_indices=...,
        )

    def get_performance(self, gt_unit):
        pass

    def get_waveforms_by_category(self, gt_unit, predicted_unit=None):
        return ...
