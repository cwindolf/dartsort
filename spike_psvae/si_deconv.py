from tempfile import TemporaryDirectory

import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents.template_matching import (
    find_spikes_from_templates,
)

from pathlib import Path


default_method_kwargs = {
    "naive": dict(
        random_chunk_kwargs=dict(return_scaled=False),
        threshold=4,
        n_shifts=10,
    ),
    "tridesclous": dict(
        threshold=4,
    ),
}


def si_deconv(
    standardized_bin,
    spike_train,
    geom,
    # recording args
    sampling_frequency=30_000,
    standardized_dtype=np.float32,
    # waveforms cache
    wf_cache_folder=None,
    job_kwargs=dict(
        n_jobs=16,
        chunk_size=30000,
        progress_bar=True
    ),
    method="naive",
    method_kwargs=None,
):
    # get a recording extractor for our destriped data
    recording = si.read_binary(
        standardized_bin,
        num_chan=len(geom),
        sampling_frequency=sampling_frequency,
        dtype=standardized_dtype,
        time_axis=0,
        is_filtered=True,
        gain_to_uV=1,
        offset_to_uV=0,
    )
    recording.set_channel_locations(geom)

    # make waveform extractor which will be used to compute templates
    sorting = si.NumpySorting.from_times_labels(
        times_list=spike_train[spike_train[:, 1] >= 0, 0],
        labels_list=spike_train[spike_train[:, 1] >= 0, 1],
        sampling_frequency=sampling_frequency,
    )

    with TemporaryDirectory(prefix="si_deconv") as tempdir:
        tempdir = Path(tempdir)
        
        # dumpable sorting
        si.NpzSortingExtractor.write_sorting(sorting, tempdir / "npz_sorting.npz")
        sorting = si.NpzSortingExtractor(tempdir / "npz_sorting.npz")
        
        if wf_cache_folder is None:
            wf_cache_folder = tempdir / "cache"
        wf_cache_folder.mkdir(exist_ok=True)
        
        we = si.extract_waveforms(
            recording,
            sorting,
            f"{wf_cache_folder}",
            overwrite=True,
            ms_before=2.5,
            ms_after=3.5,
            max_spikes_per_unit=500,
            n_jobs=20,
            chunk_size=sampling_frequency,
            progress_bar=True,
            return_scaled=False,
        )
        
        res = si.compute_quality_metrics(we, metric_names=['snr'], load_if_exists=True)
        
        if method_kwargs is None:
            method_kwargs = default_method_kwargs[method]
        print(method, method_kwargs)

        spikes = find_spikes_from_templates(
            recording,
            method=method,
            method_kwargs=dict(
                waveform_extractor=we,
                **method_kwargs,
            ),
            **job_kwargs,
        )

    return spikes

