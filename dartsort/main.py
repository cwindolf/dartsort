from dartsort.config import FeaturizationConfig, SubtractionConfig
from dartsort.peel.subtract import SubtractionPeeler
from dartsort.transform import WaveformPipeline
from dartsort.util.data_util import DARTsortSorting
from dartsort.util.waveform_util import make_channel_index

default_featurization_config = FeaturizationConfig()
default_subtraction_config = SubtractionConfig()


def dartsort(
    recording,
    output_folder,
    *more_args_tbd,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    dartsort_config=...,
    n_jobs=0,
    overwrite=False,
    show_progress=True,
    device=None,
):
    # coming soon
    pass


def subtract(
    recording,
    output_hdf5_filename,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    chunk_starts_samples=None,
    n_jobs=0,
    overwrite=False,
    residual_filename=None,
    show_progress=True,
    device=None,
):
    subtraction_peeler = SubtractionPeeler.from_config(
        recording,
        subtraction_config=subtraction_config,
        featurization_config=featurization_config,
        device=device,
    )
    subtraction_peeler.peel(
        output_hdf5_filename,
        chunk_starts_samples=chunk_starts_samples,
        n_jobs=n_jobs,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
    )
    return DARTsortSorting.from_peeling_hdf5(output_hdf5_filename)
