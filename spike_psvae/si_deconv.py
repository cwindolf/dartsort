from tempfile import TemporaryDirectory

import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents.template_matching import (
    find_spikes_from_templates,
)


def si_deconv(
    standardized_bin,
    spike_train,
    # recording args
    num_channels=384,
    sampling_frequency=30_000,
    standardized_dtype=np.float32,
    # waveforms cache
    wf_cache_folder=None,
    job_kwargs={},
):
    # get a recording extractor for our destriped data
    recording = si.read_binary(
        standardized_bin,
        num_chan=num_channels,
        sampling_frequency=sampling_frequency,
        dtype=standardized_dtype,
        time_axis=0,
        is_filtered=True,
    )

    # make waveform extractor which will be used to compute templates
    sorting = si.NumpySorting.from_times_labels(
        times_list=spike_train[:, 0],
        labels_list=spike_train[:, 1],
        sampling_frequency=sampling_frequency,
    )

    with TemporaryDirectory(prefix="si_deconv") as tempdir:
        if wf_cache_folder is None:
            wf_cache_folder = tempdir

        we = si.extract_waveforms(
            recording,
            sorting,
            wf_cache_folder,
            load_if_exists=True,
            ms_before=2.5,
            ms_after=3.5,
            max_spikes_per_unit=500,
            n_jobs=20,
            chunk_size=sampling_frequency,
            progress_bar=True,
        )

        spikes = find_spikes_from_templates(
            recording,
            method="naive",
            method_kwargs=dict(waveform_extractor=we),
        )

    return spikes

