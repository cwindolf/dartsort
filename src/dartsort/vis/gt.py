from .layout import BasePlot
from .unit import UnitPlot, make_all_summaries


class GTUnitMixin:
    def draw(self, panel, comparison, gt_unit_id):
        return super().draw(panel, comparison.gt_analysis, gt_unit_id)


class PredictedUnitMixin:
    def draw(self, panel, comparison, predicted_unit_id):
        return super().draw(panel, comparison.predicted_analysis, predicted_unit_id)


class BestMatchUnitMixin:
    def draw(self, panel, comparison, gt_unit_id):
        predicted_unit_id = comparison.get_match(gt_unit_id)
        return super().draw(panel, comparison.predicted_analysis, predicted_unit_id)


class ComparisonPlot(BasePlot):
    kind = "comparison"
    width = 2
    height = 2

    def draw(self, panel, comparison):
        raise NotImplementedError
