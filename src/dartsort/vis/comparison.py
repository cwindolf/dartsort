class UnitComparisonPlot:
    kind: str
    width = 1
    height = 1

    def draw(self, axis, sorting_comparison, ground_truth_unit_id, predicted_unit_id=None):
        raise NotImplementedError


class RawWaveformComparisonPlot(UnitComparisonPlot):
    pass
