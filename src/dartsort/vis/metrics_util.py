import pandas as pd

from spikeinterface.comparison.comparisontools import _perf_keys


dartsort_extra_metrics = ["min_temp_dist", "unsorted_recall"]
# perf keys has stuff in it like accuracy, recall, ...
metrics = list(set(_perf_keys + dartsort_extra_metrics))


def melt_metrics(
    df, keep_metric_columns=None, metric_columns=metrics, var_name="metric"
):
    if keep_metric_columns is not None:
        keep_metric_columns = metric_columns
    value_vars = [c for c in keep_metric_columns if c in df.columns]
    id_vars = [c for c in df.columns if c not in metric_columns]
    return pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
    )