# %%
import numpy as np
from scipy import signal
import time
import os
import multiprocessing
from itertools import repeat
import copy
from tqdm.auto import tqdm, trange
import pickle
from spike_psvae.spikeio import read_data, read_waveforms
from pathlib import Path
from spike_psvae import snr_templates


# %%
class MatchPursuitObjectiveUpsample:
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(
        self,
        templates,
        deconv_dir,
        standardized_bin,
        lambd=0,
        allowed_scale=np.inf,
        save_residual=False,
        t_start=0,
        t_end=None,
        n_sec_chunk=1,
        sampling_rate=30000,
        max_iter=1000,
        upsample=8,
        threshold=30.0,
        conv_approx_rank=5,
        n_processors=1,
        multi_processing=False,
        vis_su=1.0,
        verbose=False,
        template_index_to_unit_id=None,
        refractory_period_frames=10,
    ):
        """Sets up the deconvolution object.

        Parameters:
        -----------
        data: numpy array of shape (T, C)
            Where T is number of time samples and C number of channels.
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        conv_approx_rank: int
            Rank of SVD decomposition for approximating convolution
            operations for templates.
        threshold: float
            amount of energy differential that is admissible by each
            spike. The lower this threshold, more spikes are recovered.
        vis_su: float
            threshold for visibility of template channel in terms
            of peak to peak standard unit.
        template_index_to_unit_id : int array of shape (K,)
            If multiple templates correspond to the same unit (e.g. supperres),
            specify that here so that we can correctly enforce the refractory
            period.
        """

        self.verbose = verbose

        self.standardized_bin = standardized_bin

        temps = templates.transpose(1, 2, 0)
        self.temps = temps.astype(np.float32)
        self.n_time, self.n_chan, self.n_unit = temps.shape

        # handle grouped templates, as in the superresolution case
        self.grouped_temps = False
        if template_index_to_unit_id is not None:
            self.grouped_temps = True
            assert template_index_to_unit_id.shape == (self.n_unit,)
            group_index = [
                np.flatnonzero(template_index_to_unit_id == u)
                for u in template_index_to_unit_id
            ]
            self.max_group_size = max(map(len, group_index))

            # like a channel index, sort of
            # this is a n_templates x group_size array that maps each
            # template index to the set of other template indices that
            # are part of its group. so that the array is not ragged,
            # we pad rows with -1s when their group is smaller than the
            # largest group.
            self.group_index = np.full(
                (self.n_unit, self.max_group_size), -1
            )
            for j, row in enumerate(group_index):
                self.group_index[j, : len(row)] = row

        # variance parameter for the amplitude scaling prior
        assert lambd is None or lambd >= 0
        self.lambd = lambd
        self.no_amplitude_scaling = lambd is None or lambd == 0
        self.scale_min = 1 / (1 + allowed_scale)
        self.scale_max = 1 + allowed_scale

        if verbose:
            print(
                "expected shape of templates loaded (n_times, n_chan, n_units):",
                temps.shape,
            )
        self.deconv_dir = deconv_dir
        self.max_iter = max_iter
        self.n_processors = n_processors
        self.multi_processing = multi_processing

        # figure out length of data
        if standardized_bin is not None:
            standardized_bin = Path(standardized_bin)
            std_size = standardized_bin.stat().st_size
            assert not std_size % np.dtype(np.float32).itemsize
            std_size = std_size // np.dtype(np.float32).itemsize
            assert not std_size % self.n_chan
            T_samples = std_size // self.n_chan

            # time logic -- what region are we going to load
            T_sec = T_samples / sampling_rate
            assert t_start >= 0 and (t_end is None or t_end <= T_sec)
            if verbose:
                print(
                    "Instantiating MatchPursuitObjectiveUpsample on ",
                    t_end - t_start,
                    "seconds long recording with threshold",
                    threshold,
                )

            self.start_sample = t_start * sampling_rate
            if t_end is not None:
                self.end_sample = t_end * sampling_rate
            else:
                self.end_sample = T_samples
            self.batch_len_samples = n_sec_chunk * sampling_rate
            self.n_batches = np.ceil(
                (self.end_sample - self.start_sample) / self.batch_len_samples
            ).astype(int)
        self.buffer = 300

        # Upsample and downsample time shifted versions
        # Dynamic Upsampling Setup; function for upsampling based on PTP
        # Cat: TODO find better ptp-> upsample function
        self.upsample_templates_mp(int(upsample))

        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.vis_su_threshold = vis_su
        self.visible_chans()
        self.template_overlaps()
        self.spatially_mask_templates()
        # Upsample the templates
        # Index of the original templates prior to
        # upsampling them.
        self.orig_n_unit = self.n_unit
        self.n_unit = self.orig_n_unit * self.up_factor

        # Computing SVD for each template.
        self.compress_templates()

        # Compute pairwise convolution of filters
        self.pairwise_filter_conv()

        # compute squared norm of templates
        self.norm = np.zeros(self.orig_n_unit, dtype=np.float32)
        for i in range(self.orig_n_unit):
            self.norm[i] = np.sum(
                np.square(self.temps[:, self.vis_chan[:, i], i])
            )

        # self.update_data(data)
        self.dec_spike_train = np.zeros((0, 2), dtype=np.int32)
        self.dec_scalings = np.zeros((0,), dtype=np.float32)

        # Energy reduction for assigned spikes.
        self.dist_metric = np.array([])

        # Single time preperation for high resolution matches
        # matching indeces of peaks to indices of upsampled templates
        factor = self.up_factor
        radius = factor // 2 + factor % 2
        self.up_window = np.arange(-radius, radius + 1)[:, None]
        self.up_window_len = len(self.up_window)

        # Indices of single time window the window around peak after upsampling
        self.zoom_index = radius * factor + np.arange(-radius, radius + 1)
        self.peak_to_template_idx = np.concatenate(
            (np.arange(radius, -1, -1), (factor - 1) - np.arange(radius))
        )
        self.peak_time_jitter = np.concatenate(
            ([0], np.array([0, 1]).repeat(radius))
        )

        # Refractory Period Setup.
        # DO NOT MAKE IT SMALLER THAN self.n_time - 1 !!!
        # (This is not actually the refractory condition we enforce.)
        self.refrac_radius = self.n_time - 1

        # Account for upsampling window so that np.inf does not fall into the
        # window around peak for valid spikes.
        # (This is the refractory condition we enforce.)
        self.adjusted_refrac_radius = refractory_period_frames

    def upsample_templates_mp(self, upsample):
        if upsample != 1:
            max_upsample = upsample
            # original function
            self.unit_up_factor = np.power(
                4, np.floor(np.log2(np.max(self.temps.ptp(axis=0), axis=0)))
            )
            self.up_factor = min(
                max_upsample, int(np.max(self.unit_up_factor))
            )
            self.unit_up_factor[
                self.unit_up_factor > max_upsample
            ] = max_upsample
            self.unit_up_factor = np.maximum(1, self.unit_up_factor)
            self.up_up_map = np.zeros(
                self.n_unit * self.up_factor, dtype=np.int32
            )
            for i in range(self.n_unit):
                u_idx = i * self.up_factor
                u_factor = self.unit_up_factor[i]
                skip = self.up_factor // u_factor
                self.up_up_map[
                    u_idx : u_idx + self.up_factor
                ] = u_idx + np.arange(0, self.up_factor, skip).repeat(skip)

        else:
            # Upsample and downsample time shifted versions
            self.up_factor = upsample
            self.unit_up_factor = np.ones(self.n_unit)
            self.up_up_map = range(self.n_unit * self.up_factor)

    def update_data(self):
        """Updates the data for the deconv to be run on with same templates."""
        self.data = self.data.astype(np.float32)
        self.data_len = self.data.shape[0]

        # Computing SVD for each template.
        self.obj_len = self.data_len + self.n_time - 1

        # Indicator for computation of the objective.
        self.obj_computed = False

        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])
        self.iter_spike_train = []

    def visible_chans(self):
        a = np.max(self.temps, axis=0) - np.min(self.temps, 0)
        self.vis_chan = a > self.vis_su_threshold

    def template_overlaps(self):
        """Find pairwise units that have overlap between."""
        vis = self.vis_chan.T
        self.unit_overlap = np.sum(
            np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2
        )
        self.unit_overlap = self.unit_overlap > 0
        self.unit_overlap = np.repeat(
            self.unit_overlap, self.up_factor, axis=0
        )

    def spatially_mask_templates(self):
        """Spatially mask templates so that non visible channels are zero."""
        idx = np.logical_xor(
            np.ones(self.temps.shape, dtype=bool), self.vis_chan
        )
        self.temps[idx] = 0.0

    def compress_templates(self):
        """Compresses the templates using SVD and upsample temporal compoents."""
        self.temporal, self.singular, self.spatial = np.linalg.svd(
            np.transpose(self.temps, (2, 0, 1))
        )

        # Keep only the strongest components
        self.temporal = self.temporal[:, :, : self.approx_rank]
        self.singular = self.singular[:, : self.approx_rank]
        self.spatial = self.spatial[:, : self.approx_rank, :]

        # Upsample the temporal components of the SVD in effect,
        # upsampling the reconstruction of the templates.
        if self.up_factor == 1:
            # No upsampling is needed.
            temporal_up = self.temporal
        else:
            temporal_up = signal.resample(
                self.temporal, self.n_time * self.up_factor, axis=1
            )
            idx = (
                np.arange(0, self.n_time * self.up_factor, self.up_factor)
                + np.arange(self.up_factor)[:, None]
            )
            temporal_up = np.reshape(
                temporal_up[:, idx, :], [-1, self.n_time, self.approx_rank]
            ).astype(np.float32)

        self.temporal = np.flip(self.temporal, axis=1)
        temporal_up = np.flip(temporal_up, axis=1)

        if self.multi_processing:
            np.savez(
                os.path.join(self.deconv_dir, "svd.npz"),
                temporal_up=temporal_up,
                temporal=self.temporal,
                singular=self.singular,
                spatial=self.spatial,
            )
        else:
            self.temporal_up = temporal_up

    def conv_filter(
        self,
        unit_array,
        temporal,
        temporal_up,
        singular,
        spatial,
    ):
        pairwise_conv_array = []
        for unit2 in unit_array:
            conv_res_len = self.n_time * 2 - 1
            n_overlap = np.sum(self.unit_overlap[unit2, :])
            pairwise_conv = np.zeros(
                [n_overlap, conv_res_len], dtype=np.float32
            )
            orig_unit = unit2 // self.up_factor
            masked_temp = np.flipud(
                np.matmul(
                    temporal_up[unit2] * singular[orig_unit][None, :],
                    spatial[orig_unit, :, :],
                )
            )

            for j, unit1 in enumerate(
                np.where(self.unit_overlap[unit2, :])[0]
            ):
                u, s, vh = temporal[unit1], singular[unit1], spatial[unit1]
                vis_chan_idx = self.vis_chan[:, unit1]
                mat_mul_res = np.matmul(
                    masked_temp[:, vis_chan_idx],
                    vh[: self.approx_rank, vis_chan_idx].T,
                )

                for i in range(self.approx_rank):
                    pairwise_conv[j, :] += np.convolve(
                        mat_mul_res[:, i], s[i] * u[:, i].flatten(), "full"
                    )

            pairwise_conv_array.append(pairwise_conv)

        return pairwise_conv_array

    def parallel_conv_filter(self, args):
        proc_index, unit_array = args

        # load template SVD from disk (pickle size limit)
        with np.load(os.path.join(self.deconv_dir, "svd.npz")) as data:
            temporal_up = data["temporal_up"]
            temporal = data["temporal"]
            singular = data["singular"]
            spatial = data["spatial"]

        pairwise_conv_array = self.conv_filter(
            unit_array,
            temporal,
            temporal_up,
            singular,
            spatial,
        )

        with open(
            Path(self.deconv_dir)
            / ("temp_temp_chunk_" + str(proc_index) + ".pkl"),
            "wb",
        ) as f:
            pickle.dump(pairwise_conv_array, f)

    def pairwise_filter_conv(self):
        # TODO: split based on size limit rather than n_processors
        units = np.array_split(np.unique(self.up_up_map), self.n_processors)
        if self.multi_processing:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(self.n_processors) as pool:
                for result in xqdm(
                    pool.imap_unordered(
                        self.parallel_conv_filter,
                        enumerate(units),
                    ),
                    pbar=self.verbose,
                    total=len(units),
                    desc="pairwise_filter_conv",
                ):
                    pass

                # load temp_temp saved files from disk due to memory overload otherwise
                temp_array = []
                for i in range(len(units)):
                    fname = os.path.join(
                        self.deconv_dir, "temp_temp_chunk_" + str(i) + ".pkl"
                    )
                    with open(fname, "rb") as f:
                        temp_array.extend(pickle.load(f))
                    os.remove(fname)
        else:
            units = np.unique(self.up_up_map)
            temp_array = []

            for k in xqdm(
                range(len(units)), desc="pairwise conv", pbar=self.verbose
            ):
                chunk = self.conv_filter(
                    [units[k]],
                    self.temporal,
                    self.temporal_up,
                    self.singular,
                    self.spatial,
                )
                temp_array.extend(chunk)

        # initialize empty list and fill only correct locations
        pairwise_conv = []
        for i in range(self.n_unit):
            pairwise_conv.append(None)

        ctr = 0
        for unit2 in np.unique(self.up_up_map):
            pairwise_conv[unit2] = temp_array[ctr]
            ctr += 1

        pairwise_conv = np.array(pairwise_conv, dtype=object)

        if self.multi_processing:
            # save to disk, don't keep in memory
            np.save(
                os.path.join(self.deconv_dir, "pairwise_conv.npy"),
                pairwise_conv,
            )
        else:
            self.pairwise_conv = pairwise_conv

    def get_sparse_upsampled_templates(self, save_npy=True):
        """Returns the fully upsampled sparse version of the original templates.
        returns:
        --------
        Tuple of numpy.ndarray. First element is of shape (t, C, M) is the set
        upsampled shifted templates that have been used in the dynamic
        upsampling approach. Second is an array of lenght K (number of original
        units) * maximum upsample factor. Which maps cluster ids that are result
        of deconvolution to 0,...,M-1 that corresponds to the sparse upsampled
        templates.
        """
        down_sample_idx = np.arange(
            0, self.n_time * self.up_factor, self.up_factor
        )
        down_sample_idx = (
            down_sample_idx + np.arange(0, self.up_factor)[:, None]
        )

        # Reordering the upsampling. This is done because we upsampled the time
        # reversed temporal components of the SVD reconstruction of the
        # templates. This means That the time-reveresed 10x upsampled indices
        # respectively correspond to [0, 9, 8, ..., 1] of the 10x upsampled of
        # the original templates.
        all_temps = []
        # TODO: unused variable, why?
        # reorder_idx = np.concatenate(
        #     (np.arange(0, 1), np.arange(self.up_factor - 1, 0, -1))
        # )

        # Sequentialize the number of up_up_map. For instance,
        # [0, 0, 0, 0, 4, 4, 4, 4, ...] turns to [0, 0, 0, 0, 1, 1, 1, 1, ...].
        deconv_id_sparse_temp_map = []
        tot_temps_so_far = 0

        for i in range(self.orig_n_unit):
            up_temps = signal.resample(
                self.temps[:, :, i], self.n_time * self.up_factor
            )[down_sample_idx, :]
            up_temps = up_temps.transpose([1, 2, 0])
            # up_temps = up_temps[:, :, reorder_idx]
            skip = self.up_factor // self.unit_up_factor[i]
            keep_upsample_idx = np.arange(0, self.up_factor, skip).astype(
                np.int32
            )
            deconv_id_sparse_temp_map.append(
                np.arange(self.unit_up_factor[i]).repeat(skip)
                + tot_temps_so_far
            )
            tot_temps_so_far += self.unit_up_factor[i]
            all_temps.append(up_temps[:, :, keep_upsample_idx])

        deconv_id_sparse_temp_map = np.concatenate(
            deconv_id_sparse_temp_map, axis=0
        )

        all_temps = np.concatenate(all_temps, axis=2)

        if save_npy:
            np.save(
                os.path.join(self.deconv_dir, "sparse_templates.npy"),
                all_temps,
            )
            np.save(
                os.path.join(self.deconv_dir, "deconv_id_sparse_temp_map.npy"),
                deconv_id_sparse_temp_map,
            )

        return all_temps, deconv_id_sparse_temp_map

    def get_upsampled_templates(self):
        """Returns the fully upsampled version of the original templates."""
        down_sample_idx = np.arange(
            0, self.n_time * self.up_factor, self.up_factor
        )
        down_sample_idx = (
            down_sample_idx + np.arange(0, self.up_factor)[:, None]
        )

        if self.multi_processing:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(self.n_processors) as pool:
                res = []
                for r in xqdm(
                    pool.imap(
                        self.upsample_templates_parallel,
                        zip(
                            self.temps.T,
                            repeat(self.n_time),
                            repeat(self.up_factor),
                            repeat(down_sample_idx),
                        ),
                    ),
                    total=len(self.temps.T),
                    desc="get_upsampled_templates",
                    pbar=self.verbose,
                ):
                    res.append(r)
        else:
            res = []
            for k in range(self.temps.T.shape[0]):
                res.append(
                    self.upsample_templates_parallel(
                        self.temps.T[k],
                        self.n_time,
                        self.up_factor,
                        down_sample_idx,
                    )
                )

        up_temps = np.array(res)
        up_temps = (
            up_temps.transpose([2, 3, 0, 1])
            .reshape([self.n_chan, -1, self.n_time])
            .transpose([2, 0, 1])
        )
        self.n_unit = self.n_unit * self.up_factor
        # Reordering the upsampling. This is done because we upsampled the time
        # reversed temporal components of the SVD reconstruction of the
        # templates. This means That the time-reveresed 10x upsampled indices
        # respectively correspond to [0, 9, 8, ..., 1] of the 10x upsampled of
        # the original templates.
        reorder_idx = np.tile(
            np.concatenate(
                (np.arange(0, 1), np.arange(self.up_factor - 1, 0, -1))
            ),
            self.orig_n_unit,
        )
        reorder_idx += np.arange(
            0, self.up_factor * self.orig_n_unit, self.up_factor
        ).repeat(self.up_factor)
        return up_temps[:, :, reorder_idx]

    def upsample_templates_parallel(
        template, n_time, up_factor, down_sample_idx
    ):
        return signal.resample(template.T, n_time * up_factor)[
            down_sample_idx, :
        ]

    def correct_shift_deconv_spike_train(self, dec_spike_train):
        """Get time shift corrected version of the deconvovled spike train.
        This corrected version only applies if you consider getting upsampled
        templates with get_upsampled_templates() method.
        """
        correct_spt = copy.copy(dec_spike_train)
        correct_spt[correct_spt[:, 1] % self.up_factor > 0, 0] += 1
        return correct_spt

    def compute_objective(self):
        """Computes the objective given current state of recording."""
        if self.obj_computed:
            return self.obj

        self.conv_result = np.zeros(
            [self.orig_n_unit, self.data_len + self.n_time - 1],
            dtype=np.float32,
        )
        for rank in range(self.approx_rank):
            matmul_result = np.matmul(
                self.spatial[:, rank] * self.singular[:, [rank]], self.data.T
            )
            filters = self.temporal[:, :, rank]
            for unit in range(self.orig_n_unit):
                self.conv_result[unit, :] += np.convolve(
                    matmul_result[unit, :], filters[unit], mode="full"
                )

        if self.no_amplitude_scaling:
            # the original objective with no amplitude scaling. note that
            # the objective below converges to this one as lambda -> 0
            self.obj = 2 * self.conv_result - self.norm[:, None]
        else:
            # the objective is (conv + 1/lambd)^2 / (norm + 1/lambd) - 1/lambd
            # we omit the final -1/lambd since it's ok to work up to a constant
            b = self.conv_result + 1 / self.lambd
            a = self.norm[:, None] + 1 / self.lambd

            # this is the objective with the optimal scaling *without hard clipping*
            # this order of operations is key to avoid overflows when squaring!
            # self.obj = b * (b / a) - 1 / self.lambd

            # but, in practice we do apply hard clipping. so we have to compute
            # the following more cumbersome formula:
            scalings = np.clip(b / a, self.scale_min, self.scale_max)
            self.obj = (
                2 * scalings * b - np.square(scalings) * a - 1 / self.lambd
            )

        # Set indicator to true so that it no longer is run for future
        # iterations in case subtractions are done implicitly.
        self.obj_computed = True

    def high_res_peak(self, times, unit_ids):
        """Finds best matching high resolution template.

        Given an original unit id and the infered spike times
        finds out which of the shifted upsampled templates of
        the unit best matches at that time to the residual.

        Parameters:
        -----------
        times: numpy.array of numpy.int
            spike times for the unit.
        unit_ids: numpy.array of numpy.int
            Respective to times, id of each spike corresponding
            to the original units.
        Returns:
        --------
            tuple in the form of (numpy.array, numpy.array, numpy.array)
            respectively the offset of shifted templates and a necessary time
            shift to correct the spike time, and the index of spike times that
            do not violate refractory period.
        """
        if self.up_factor == 1 or len(times) < 1:
            return 0, 0, range(len(times))

        idx = times + self.up_window
        peak_window = self.obj[unit_ids, idx]

        # Find times that the window around them do not include np.inf.
        # In other words do not violate refractory period.
        invalid_idx = np.logical_or(
            np.isinf(peak_window[0, :]), np.isinf(peak_window[-1, :])
        )

        # Turn off the invalid units for next iterations.
        turn_off_idx = (
            times[invalid_idx] + np.arange(-self.refrac_radius, 1)[:, None]
        )
        self.obj[unit_ids[invalid_idx], turn_off_idx] = -np.inf
        # added this line when including lambda, since
        # we recompute the objective differently now in subtraction
        if not self.no_amplitude_scaling:
            self.conv_result[unit_ids[invalid_idx], turn_off_idx] = -np.inf

        valid_idx = np.flatnonzero(np.logical_not(invalid_idx))
        peak_window = peak_window[:, valid_idx]
        if peak_window.shape[1] == 0:
            return np.array([]), np.array([]), valid_idx

        high_resolution_peaks = signal.resample(
            peak_window, self.up_window_len * self.up_factor, axis=0
        )
        shift_idx = np.argmax(
            high_resolution_peaks[self.zoom_index, :], axis=0
        )
        return (
            self.peak_to_template_idx[shift_idx],
            self.peak_time_jitter[shift_idx],
            valid_idx,
        )

    def find_peaks(self):
        """Finds peaks in subtraction differentials of spikes."""
        max_across_temp = np.max(self.obj, axis=0)
        spike_times = signal.argrelmax(
            max_across_temp[self.n_time - 1 : self.obj.shape[1] - self.n_time],
            order=self.refrac_radius,
        )[0] + (self.n_time - 1)
        spike_times = spike_times[
            max_across_temp[spike_times] > self.threshold
        ]
        dist_metric = max_across_temp[spike_times]

        # Upsample the objective and find the best upsampled template.
        spike_ids = np.argmax(self.obj[:, spike_times], axis=0)
        upsampled_template_idx, time_shift, valid_idx = self.high_res_peak(
            spike_times, spike_ids
        )

        # find amplitude scalings when lambda != 0
        # we run this before shifting the spike ids into the upsampled id space,
        # because it requires the conv result and we have only computed this with
        # the original, non-upsampled templates. probably it would be good to do
        # so with the upsampled templates, which would require computing the obj
        # and the pairwise_conv in the upsampled id space -- that could be expensive.
        # for now, we're leaving it like this.
        if self.no_amplitude_scaling:
            scalings = np.ones(len(spike_times), dtype=np.float32)
        else:
            scalings = (
                self.conv_result[spike_ids, spike_times] + 1 / self.lambd
            ) / (self.norm[spike_ids] + 1 / self.lambd)
            scalings = np.clip(scalings, self.scale_min, self.scale_max)

        # The spikes that had NaN in the window and could not be upsampled
        # should fall-back on default value.
        spike_ids *= self.up_factor
        if len(valid_idx):
            spike_ids[valid_idx] += upsampled_template_idx
            spike_times[valid_idx] += time_shift

        # Note that we shift the discovered spike times from convolution
        # space to actual raw voltage space by subtracting self.n_time + 1
        new_spike_train = np.c_[spike_times - (self.n_time - 1), spike_ids]

        return new_spike_train, scalings, dist_metric[valid_idx]

    def enforce_refractory(self, spike_train):
        """Enforces refractory period for units."""
        window = np.arange(
            -self.adjusted_refrac_radius, self.adjusted_refrac_radius + 1
        )
        # Re-adjust cluster id's so that they match with the original templates
        unit_idx = spike_train[:, 1] // self.up_factor
        spike_times = spike_train[:, 0]

        # correct for template grouping (for example, the superres case)
        if self.grouped_temps:
            # here, each index in unit_idx corresponds to a template which
            # is part of a set of templates that all correspond to the same
            # unit. so we want to deactivate all at once.
            # these arrays are both N_spikes x group size
            units_group_idx = self.group_index[unit_idx]
            spike_times = np.broadcast_to(
                spike_times[:, None], (spike_times.shape[0], self.max_group_size)
            )
            valid_spikes = units_group_idx > 0
            unit_idx = units_group_idx[valid_spikes]
            spike_times = spike_times[valid_spikes]

        # The offset self.n_time - 1 is necessary to revert the spike times
        # back to objective function indices which is the result of convoultion
        # operation.
        time_idx = (spike_times[:, None] + self.n_time - 1) + window

        # enforce refractory by setting objective to 0 in invalid regions
        self.obj[unit_idx[:, None], time_idx[:, 1:-1]] = -np.inf
        # added this line when including lambda, since
        # we recompute the objective differently now in subtraction
        if not self.no_amplitude_scaling:
            self.conv_result[unit_idx, time_idx[:, 1:-1]] = -np.inf

    def subtract_spike_train(self, spt, scalings):
        """Subtracts a spike train from the original spike_train."""
        present_units = np.unique(spt[:, 1])
        conv_res_len = self.n_time * 2 - 1
        for i in present_units:
            in_unit = np.flatnonzero(spt[:, 1] == i)
            unit_spt = spt[in_unit, :]
            spt_idx = np.arange(0, conv_res_len) + unit_spt[:, :1]

            # Grid idx of subset of channels and times
            # note: previously, even and odd times were done separately.
            # I think this was to handle overlaps: in place -= in numpy
            # won't subtract from the same index twice.
            # But np.subtract.at will! Switching to that.
            unit_idx = np.flatnonzero(self.unit_overlap[i])
            idx = np.ix_(unit_idx, spt_idx.ravel())
            pconv = self.pairwise_conv[self.up_up_map[i]]
            if self.no_amplitude_scaling:
                np.subtract.at(
                    self.obj,
                    idx,
                    np.tile(
                        2 * pconv,
                        len(unit_spt),
                    ),
                )
            else:
                # this particular broadcasting makes things line up with the ravel()
                # used in the definition of `idx` above
                to_subtract = (
                    pconv[:, None, :] * scalings[in_unit][None, :, None]
                )
                to_subtract = to_subtract.reshape(*pconv.shape[:-1], -1)
                ninf_ix = np.where(self.conv_result[idx] == -np.inf)
                np.subtract.at(
                    self.conv_result,
                    idx,
                    to_subtract,
                )

                # now we update the objective just at the changed
                # indices -- no need to do the whole thing.
                b = self.conv_result[idx] + 1 / self.lambd
                a = self.norm[unit_idx, None] + 1 / self.lambd
                bba = b * (b / a) - 1 / self.lambd
                bba[ninf_ix] = -np.inf
                self.obj[idx] = bba

        self.enforce_refractory(spt)

    def _run_batch(self, args):
        # helper for multiprocessing
        return self.run_batch(*args)

    def load_saved_state(self):
        # helper -- initializer for threads
        if self.multi_processing:
            MatchPursuitObjectiveUpsample.pairwise_conv = np.load(
                os.path.join(self.deconv_dir, "pairwise_conv.npy"),
                allow_pickle=True,
            )

    def run_array(self, data):
        self.data = data
        self.data_len = self.data.shape[0]

        # update data
        self.update_data()

        # compute objective function
        self.compute_objective()
        if int(self.verbose) > 1:
            start_time = time.time()
            print(f"Objective took: {time.time() - start_time:.2f}")

        ctr = 0
        tot_max = np.inf
        while tot_max > self.threshold and ctr < self.max_iter:
            spt, scalings, dist_met = self.find_peaks()

            if len(spt) == 0:
                break

            self.dec_spike_train = np.concatenate((self.dec_spike_train, spt))
            self.dec_scalings = np.concatenate((self.dec_scalings, scalings))

            self.subtract_spike_train(spt, scalings)

            self.dist_metric = np.concatenate((self.dist_metric, dist_met))

            if int(self.verbose) > 1:
                print(
                    f"Iteration {ctr} found {spt.shape[0]} spikes "
                    f"with {np.sum(dist_met):.2f} energy reduction."
                )

            ctr += 1

        # order spike times
        idx = np.argsort(self.dec_spike_train[:, 0])
        self.dec_spike_train = self.dec_spike_train[idx]
        self.dec_scalings = self.dec_scalings[idx]

        return ctr

    def run_batch(self, batch_id, fname_out):
        start_time = time.time()

        # ********* run deconv ************
        # read raw data for segment using idx_list vals
        # load raw data with buffer
        s_start = self.start_sample + batch_id * self.batch_len_samples
        s_end = min(self.end_sample, s_start + self.batch_len_samples)
        load_start = max(self.start_sample, s_start - self.buffer)
        load_end = min(self.end_sample, s_end + self.buffer)
        data = read_data(
            self.standardized_bin,
            np.float32,
            load_start,
            load_end,
            self.n_chan,
        )

        # 0 padding if we were at the edge of the data
        pad_left = pad_right = 0
        if load_start == self.start_sample:
            pad_left = self.buffer
        if load_end == self.end_sample:
            pad_right = self.buffer - (self.end_sample - s_end)
        if pad_left != 0 or pad_right != 0:
            data = np.pad(data, [(pad_left, pad_right), (0, 0)], mode="edge")
        assert data.shape == (
            2 * self.buffer + s_end - s_start,
            self.n_chan,
        )

        ctr = self.run_array(data)

        if int(self.verbose) > 1:
            print(
                f"deconv seg {batch_id}, # iter: {ctr}, "
                f"tot_spikes: {self.dec_spike_train.shape[0]}, "
                f"tot_time: {time.time() - start_time:.2f}"
            )

        # ******** ADJUST SPIKE TIMES TO REMOVE BUFFER AND OFSETS *******
        # find spikes inside data block, i.e. outside buffers
        if pad_right > 0:  # end of the recording (TODO: need to check)
            subtract_t = self.buffer + self.n_time
        else:
            subtract_t = self.buffer

        idx = np.where(
            np.logical_and(
                self.dec_spike_train[:, 0] >= self.buffer,
                self.dec_spike_train[:, 0] < self.data.shape[0] - subtract_t,
            )
        )[0]
        self.dec_spike_train = self.dec_spike_train[idx]
        self.dec_scalings = self.dec_scalings[idx]

        # offset spikes to start of index
        batch_offset = s_start - self.buffer
        self.dec_spike_train[:, 0] += batch_offset

        np.savez(
            fname_out,
            spike_train=self.dec_spike_train,
            scalings=self.dec_scalings,
            dist_metric=self.dist_metric,
        )

    def run(self, batch_ids, fnames_out):
        # loop over each assigned segment
        self.load_saved_state()
        for batch_id, fname_out in xqdm(
            zip(batch_ids, fnames_out), total=len(batch_ids), pbar=self.verbose
        ):
            self.run_batch(batch_id, fname_out)


# %%
def deconvolution(
    spike_index,
    cluster_labels,
    output_directory,
    standardized_bin,
    residual_bin,
    template_spike_train,
    geom_path,
    unit_maxchans=None,
    threshold=50,
    max_upsample=8,
    vis_su_threshold=1.0,
    approx_rank=5,
    multi_processing=False,
    cleaned_temps=True,
    n_processors=1,
    sampling_rate=30000,
    deconv_max_iters=1000,
    t_start=0,
    t_end=None,
    n_sec_chunk=1,
    verbose=False,
    trough_offset=42,
    reducer=np.median,
    overwrite=True,
    lambd=0,
    allowed_scale=np.inf,
):
    r"""Deconvolution.
    YASS deconvolution (cpu version) refactored: https://github.com/paninski-lab/yass/blob/yass-registration/src/yass/deconvolve/run_original.py

    Arguments
    ---------
    cluster_labels: cluster labels
    output_directory: save directory
    standardized_recording_path: standardized raw data path
    threshold: threshold for deconvolution
    lambd : float
        Variance for amplitude scaling prior
    allowed_scale : float
        Recovered scales will be clipped to the interval `[1 / (1 + allowed_scale), 1 + allowed_scale]`.
        Set to np.inf to allow any positive scaling.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    geom = np.load(geom_path)

    # get templates
    template_path = os.path.join(output_directory, "templates.npy")
    if overwrite or not os.path.exists(template_path):
        print("Computing templates")
        if cleaned_temps:
            if unit_maxchans is None:
                raise ValueError(
                    "We need unit maxchans to get the cleaned templates"
                )
            templates, extra = snr_templates.get_templates(
                template_spike_train,
                geom,
                standardized_bin,
                unit_maxchans,
                trough_offset=trough_offset,
                reducer=reducer,
            )
        else:
            templates = get_templates(
                standardized_bin,
                spike_index,
                cluster_labels,
                geom,
                trough_offset=trough_offset,
                reducer=reducer,
                pbar=True,
            )  # .astype(np.float32)
        np.save(template_path, templates)
    else:
        print(f"Loading templates from {template_path}")
        templates = np.load(template_path)

    fname_spike_train = os.path.join(output_directory, "spike_train.npy")
    fname_scalings = os.path.join(output_directory, "scalings.npy")
    fname_templates_up = os.path.join(output_directory, "templates_up.npy")
    fname_spike_train_up = os.path.join(output_directory, "spike_train_up.npy")
    fname_map = os.path.join(output_directory, "deconv_id_sparse_temp_map.npy")

    #         if (os.path.exists(template_path) and
    #             os.path.exists(fname_spike_train) and
    #             os.path.exists(fname_templates_up) and
    #             os.path.exists(fname_spike_train_up)):
    #             return (fname_templates_up,fname_spike_train_up,template_path,fname_spike_train)

    deconv_dir = os.path.join(output_directory, "deconv_tmp")
    if not os.path.exists(deconv_dir):
        os.makedirs(deconv_dir)
    # match and pursuit object
    mp_object = MatchPursuitObjectiveUpsample(
        templates=templates,
        deconv_dir=deconv_dir,
        standardized_bin=standardized_bin,
        t_start=t_start,
        t_end=t_end,
        n_sec_chunk=n_sec_chunk,
        sampling_rate=sampling_rate,
        max_iter=deconv_max_iters,
        upsample=max_upsample,
        threshold=threshold,
        conv_approx_rank=approx_rank,
        n_processors=n_processors,
        multi_processing=multi_processing,
        verbose=verbose,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )

    (
        templates_up,
        deconv_id_sparse_temp_map,
    ) = mp_object.get_sparse_upsampled_templates()
    np.save(fname_templates_up, templates_up.transpose(2, 0, 1))
    np.save(fname_map, deconv_id_sparse_temp_map)

    fnames_out = []
    batch_ids = []
    for batch_id in range(mp_object.n_batches):
        fname_temp = os.path.join(
            deconv_dir, f"seg_{str(batch_id).zfill(6)}_deconv.npz"
        )
        fnames_out.append(fname_temp)
        batch_ids.append(batch_id)

    if len(batch_ids) > 0:
        if multi_processing:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                n_processors,
                initializer=mp_object.load_saved_state,
            ) as pool:
                for res in tqdm(
                    pool.imap_unordered(
                        mp_object._run_batch,
                        zip(batch_ids, fnames_out),
                    ),
                    total=len(batch_ids),
                    desc="run template matching",
                ):
                    pass
        else:
            mp_object.run(batch_ids, fnames_out)

    deconv_st = []
    deconv_scalings = []
    print("gathering deconvolution results")
    for batch_id in range(mp_object.n_batches):
        fname_out = os.path.join(
            deconv_dir, "seg_{}_deconv.npz".format(str(batch_id).zfill(6))
        )
        with np.load(fname_out) as d:
            deconv_st.append(d["spike_train"])
            deconv_scalings.append(d["scalings"])
    deconv_st = np.concatenate(deconv_st, axis=0)
    deconv_scalings = np.concatenate(deconv_scalings, axis=0)

    print(f"Number of Spikes deconvolved: {deconv_st.shape[0]}")

    # get spike train and save
    spike_train = np.copy(deconv_st)
    # map back to original id
    spike_train[:, 1] = np.int32(spike_train[:, 1] / max_upsample)
    spike_train[:, 0] += trough_offset
    # save
    np.save(fname_spike_train, spike_train)
    print(fname_spike_train, spike_train.shape)
    np.save(fname_scalings, deconv_scalings)
    print(fname_scalings, deconv_scalings.shape)

    # get upsampled spike train
    spike_train_up = np.copy(deconv_st)
    spike_train_up[:, 0] += trough_offset
    np.save(
        os.path.join(output_directory, "spike_train_up_orig.npy"),
        spike_train_up,
    )
    spike_train_up[:, 1] = deconv_id_sparse_temp_map[spike_train_up[:, 1]]
    np.save(fname_spike_train_up, spike_train_up)
    print(fname_spike_train_up, spike_train_up.shape)
    np.savez(
        os.path.join(output_directory, "up_up_maps.npz"),
        up_up_map=mp_object.up_up_map,
        unit_up_factor=mp_object.unit_up_factor,
    )

    return (
        fname_templates_up,
        fname_spike_train_up,
        template_path,
        fname_spike_train,
        fname_scalings,
    )


# %%
def get_templates(
    standardized_bin,
    spike_index,
    labels,
    geom,
    n_times=121,
    n_samples=250,
    trough_offset=42,
    reducer=np.median,
    pbar=False,
):

    n_chans = geom.shape[0]

    unique_labels = np.unique(labels)
    n_templates = unique_labels.shape[0]
    if -1 in unique_labels:
        n_templates -= 1

    templates = np.empty((n_templates, n_times, n_chans))
    units = (
        trange(n_templates, desc="Templates") if pbar else range(n_templates)
    )
    for unit in units:
        spike_times_unit = spike_index[labels == unit, 0]
        which = slice(None)
        if spike_times_unit.shape[0] > n_samples:
            which = np.random.choice(
                spike_times_unit.shape[0], n_samples, replace=False
            )

        wfs_unit, skipped_idx = read_waveforms(
            spike_times_unit[which],
            standardized_bin,
            geom.shape[0],
            spike_length_samples=n_times,
            trough_offset=trough_offset,
        )
        templates[unit] = reducer(wfs_unit, axis=0)

    return templates


# %%
def xqdm(it, pbar=True, **kwargs):
    if pbar:
        return tqdm(it, **kwargs)
    else:
        return it
