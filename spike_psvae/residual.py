import os
import logging
import numpy as np
import parmap
from pathlib import Path
from spike_psvae.subtract import read_data


class RESIDUAL(object):
    def __init__(
        self,
        fname_templates_up,
        fname_spike_train_up,
        standardized_bin,
        output_directory,
        geom_path,
        n_sec_chunk=1,
        sampling_rate=30000,
        t_start=0,
        t_end=None,
        trough_offset=42,
    ):

        """Initialize by computing residuals
        provide: raw data block, templates, and deconv spike train;
        """

        # keep templates and spike train filname
        # will be loaded during each parallel process
        self.fname_templates_up = fname_templates_up
        self.fname_spike_train_up = fname_spike_train_up
        self.trough_offset = trough_offset

        n_chan = np.load(geom_path).shape[0]

        self.standardized_bin = standardized_bin
        standardized_bin = Path(standardized_bin)

        std_size = standardized_bin.stat().st_size
        assert not std_size % np.dtype(np.float32).itemsize
        std_size = std_size // np.dtype(np.float32).itemsize
        assert not std_size % n_chan
        T_samples = std_size // n_chan

        # time logic -- what region are we going to load
        T_sec = T_samples / sampling_rate
        assert t_start >= 0 and (t_end is None or t_end <= T_sec)
        print("Instantiating RESIDUAL on ", T_sec, "seconds long recording")

        self.start_sample = t_start
        if t_end is not None:
            self.end_sample = t_end
        else:
            self.end_sample = T_samples
        self.batch_len_samples = n_sec_chunk * sampling_rate
        self.n_batches = np.ceil(
            (self.end_sample - self.start_sample) / self.batch_len_samples
        ).astype(int)
        self.buffer = 300

        # save output name and dtype
        self.fname_out = os.path.join(output_directory, "residual.bin")

    def compute_residual(
        self, save_dir, multi_processing=False, n_processors=1
    ):

        batch_ids = []
        fnames_seg = []
        for batch_id in range(self.n_batches):
            batch_ids.append(batch_id)
            fnames_seg.append(
                os.path.join(
                    save_dir,
                    "residual_seg_{}.npy".format(str(batch_id).zfill(6)),
                )
            )

        if multi_processing:
            batches_in = np.array_split(batch_ids, n_processors)
            fnames_in = np.array_split(fnames_seg, n_processors)
            parmap.starmap(
                self.subtract_parallel,
                list(zip(batches_in, fnames_in)),
                processes=n_processors,
                pm_pbar=True,
            )

        else:
            for ctr in range(len(batch_ids)):
                self.subtract_parallel([batch_ids[ctr]], [fnames_seg[ctr]])

        self.fnames_seg = fnames_seg

    def subtract_parallel(self, batch_ids, fnames_out):
        """ """

        for batch_id, fname_out in zip(batch_ids, fnames_out):
            if os.path.exists(fname_out):
                continue

            templates = np.load(self.fname_templates_up)
            spike_train = np.load(self.fname_spike_train_up)

            # do not read spike train again here
            n_unit, n_time, n_chan = templates.shape
            time_idx = np.arange(0, n_time)

            # shift spike time so that it is aligned at
            # time 0
            spike_train[:, 0] -= self.trough_offset

            # get relevant spike times
            s_start = self.start_sample + batch_id * self.batch_len_samples
            s_end = min(self.end_sample, s_start + self.batch_len_samples)
            load_start = max(self.start_sample, s_start - self.buffer)
            load_end = min(self.end_sample, s_end + self.buffer)
            data = read_data(
                self.standardized_bin, np.float32, load_start, load_end, n_chan
            )

            # 0 padding if we were at the edge of the data
            pad_left = pad_right = 0
            if load_start == self.start_sample:
                pad_left = self.buffer
            if load_end == self.end_sample:
                pad_right = self.buffer - (self.end_sample - s_end)
            if pad_left != 0 or pad_right != 0:
                data = np.pad(
                    data, [(pad_left, pad_right), (0, 0)], mode="edge"
                )
            assert data.shape == (2 * self.buffer + s_end - s_start, n_chan)

            idx_in_chunk = np.where(
                np.logical_and(
                    spike_train[:, 0] >= load_start,
                    spike_train[:, 0] < load_end - self.buffer,
                )
            )[0]
            spikes_in_chunk = spike_train[idx_in_chunk]
            # offset
            spikes_in_chunk[:, 0] = (
                spikes_in_chunk[:, 0] - load_start + pad_left
            )

            for j in range(spikes_in_chunk.shape[0]):
                tt, ii = spikes_in_chunk[j]
                data[time_idx + tt, :] -= templates[ii]

            # remove buffer
            data = data[self.buffer : -self.buffer]

            # save
            np.save(fname_out, data)

    def save_residual(self):

        f = open(self.fname_out, "wb")
        for fname in self.fnames_seg:
            res = np.load(fname).astype(np.float32)
            f.write(res)
        f.close()

        # delete residual chunks after successful merging/save
        for fname in self.fnames_seg:
            os.remove(fname)


def run_residual(
    templates_up_path,
    spike_train_up_path,
    output_directory,
    standardized_bin,
    geom_path,
    n_sec_chunk=1,
    sampling_rate=30000,
    multi_processing=False,
    n_processors=1,
    trough_offset=42,
):

    # get residual object
    residual_object = RESIDUAL(
        templates_up_path,
        spike_train_up_path,
        standardized_bin,
        output_directory,
        geom_path,
        n_sec_chunk,
        sampling_rate,
        trough_offset=trough_offset,
    )

    if os.path.exists(residual_object.fname_out):
        return residual_object.fname_out

    # compute residual
    seg_dir = os.path.join(output_directory, "residual_tmp")
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    residual_object.compute_residual(seg_dir, multi_processing, n_processors)

    # concatenate all segments
    residual_object.save_residual()

    return residual_object.fname_out
