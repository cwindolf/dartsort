import os
import numpy as np
import parmap
from scipy.signal import butter, filtfilt
from spike_psvae import spikeio

def filter_standardize_rec(output_directory, filename_raw, dtype_raw,
    rec_len_sec, n_channels = 384,
    dtype_output = np.float32,
    apply_filter = True,
    low_frequency =300, high_factor = 0.1, order = 3, sampling_frequency= 30000, 
    buffer = None,
    n_sec_chunk=1, multi_processing = True, n_processors = 6, overwrite = True):
    """Preprocess pipeline: filtering, standarization and whitening filter
    This step (optionally) performs filtering on the data, standarizes it
    and computes a whitening filter. Filtering and standardized data are
    processed in chunks and written to disk.
    Parameters
    ----------
    output_directory: str
        where results will be saved
    Returns
    -------
    standardized_path: str
        Path to standardized data binary file
    standardized_params: str
        Path to standardized data parameters
    The files will be saved in output_directory
    
    n_sec_chunk: temporal length of chunks of data preprocessed in parallel
    """

    # make output parameters
    standardized_path = os.path.join(output_directory, "standardized.bin")
    standardized_params = dict(
        dtype=dtype_output,
        n_channels=n_channels)


    # Check if data already saved to disk and skip:
    if os.path.exists(standardized_path) and not overwrite:
        return standardized_path, standardized_params['dtype']

    # estimate std from a small chunk
    chunk_5sec = 5*sampling_frequency
    rec_len = rec_len_sec*sampling_frequency
    if rec_len < chunk_5sec:
        chunk_5sec = rec_len
    small_batch = reader.read_data(
        data_start=CONFIG.rec_len//2 - chunk_5sec//2,
        data_end=CONFIG.rec_len//2 + chunk_5sec//2)

    small_batch = spikeio.read_data(
        filename_raw, 
        dtype = dtype_raw,
        s_start=rec_len//2 - chunk_5sec//2,
        s_end=rec_len//2 + chunk_5sec//2, 
        n_channels = n_channels)

    fname_mean_sd = os.path.join(
        output_directory, 'mean_and_standard_dev_value.npz')
    if not os.path.exists(fname_mean_sd) and not overwrite:
        get_std(small_batch, sampling_frequency,
                fname_mean_sd, apply_filter,
                low_frequency, high_factor, order)

    # turn it off
    small_batch = None

    # Make directory to hold filtered batch files:
    filtered_location = os.path.join(output_directory, "filtered_files")
    if not os.path.exists(filtered_location):
        os.makedirs(filtered_location)

    n_batches = rec_len_sec//n_sec_chunk

    # define a size of buffer if not defined
    if buffer is None:
        buffer = int(max(sampling_frequency/100, 200))


    # read config params
    if multi_processing:
        parmap.map(
            filter_standardize_batch,
            [i for i in range(n_batches)],
            fname_mean_sd,
            filename_raw, 
            apply_filter,
            dtype_raw, 
            dtype_output,
            filtered_location,
            n_channels,
            buffer,
            low_frequency,
            high_factor,
            order,
            sampling_rate,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(n_batches):
            filter_standardize_batch(
                batch_id, fname_mean_sd,
                filename_raw, 
                apply_filter,
                dtype_raw, 
                dtype_output,
                filtered_location,
                n_channels, 
                buffer,
                low_frequency,
                high_factor,
                order,
                sampling_rate,
                )

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(filtered_location, output_directory)

    return standardized_path, standardized_params['dtype']



def _butterworth(ts, low_frequency, high_factor, order, sampling_frequency):
    """Butterworth filter
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """

    low = float(low_frequency) / sampling_frequency * 2
    high = float(high_factor) * 2
    b, a = butter(order, low, btype='high', analog=False)

    if ts.ndim == 1:
        return filtfilt(b, a, ts)
    else:
        T, C = ts.shape
        output = np.zeros((T, C), 'float32')
        for c in range(C):
            output[:, c] = filtfilt(b, a, ts[:, c])

        return output


def _mean_standard_deviation(rec, centered=False):
    """Determine standard deviation of noise in each channel
    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
    centered : bool
        if not standardized, center it
    Returns
    -------
    sd : vector [number of channels]
        standard deviation in each channel
    """

    # find standard deviation using robust method
    if not centered:
        centers = np.mean(rec, axis=0)
        rec = rec - centers[None]
    else:
        centers = np.zeros(rec.shape[1], 'float32')

    return np.median(np.abs(rec), 0)/0.6745, centers


def _standardize(rec, sd=None, centers=None):
    """Determine standard deviation of noise in each channel
    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
        recording
    sd : vector [number of chnanels,]
        standard deviation
    centered : bool
        if not standardized, center it
    Returns
    -------
    matrix [length of recording, number of channels]
        standardized recording
    """

    # find standard deviation using robust method
    if (sd is None) or (centers is None):
        sd, centers = _mean_standard_deviation(rec, centered=False)

    # standardize all channels with SD> 0.1 (Voltage?) units
    # Cat: TODO: ensure that this is actually correct for all types of channels
    idx1 = np.where(sd>=0.1)[0]
    rec[:,idx1] = np.divide(rec[:,idx1] - centers[idx1][None], sd[idx1])
    
    # zero out bad channels
    idx2 = np.where(sd<0.1)[0]
    rec[:,idx2]=0.
    
    return rec
    #return np.divide(rec, sd)



def filter_standardize_batch(batch_id, bin_file, fname_mean_sd,
                             apply_filter, dtype_input, out_dtype, output_directory,
                             n_channels, buffer,
                             low_frequency=None, high_factor=None,
                             order=None, sampling_frequency=None):
    """Butterworth filter for a one dimensional time series
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """
    logger = logging.getLogger(__name__)

    
    # filter
    if apply_filter:
        # read a batch
        # Add buffer into s_start and s_end
        s_start = batch_id*sampling_frequency-buffer
        s_end = (batch_id+1)*sampling_frequency+buffer
        ts = spikeio.read_data(bin_file, dtype_input, s_start, s_end, n_channels)
        ts = _butterworth(ts, low_frequency, high_factor,
                              order, sampling_frequency)
        ts = ts[buffer:-buffer]
    else:
        # read a batch
        s_start = batch_id*sampling_frequency-buffer
        s_end = (batch_id+1)*sampling_frequency+buffer
        ts = spikeio.read_data(bin_file, dtype_input, s_start, s_end, n_channels)
    # standardize
    temp = np.load(fname_mean_sd)
    sd = temp['sd']
    centers = temp['centers']
    ts = _standardize(ts, sd, centers)
    
    # save
    fname = os.path.join(
        output_directory,
        "standardized_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, ts.astype(out_dtype))

    #fname = os.path.join(
    #    output_directory,
    #    "standardized_{}.bin".format(
    #        str(batch_id).zfill(6)))
    #f = open(fname, 'wb')
    #f.write(ts.astype(out_dtype))



def get_std(ts,
            sampling_frequency,
            fname,
            apply_filter=False, 
            low_frequency=None,
            high_factor=None,
            order=None):
    """Butterworth filter for a one dimensional time series
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """

    # filter
    if apply_filter:
        ts = _butterworth(ts, low_frequency, high_factor,
                          order, sampling_frequency)

    # standardize
    sd, centers = _mean_standard_deviation(ts)
    
    # save
    np.savez(fname,
             centers=centers,
             sd=sd)


def merge_filtered_files(filtered_location, output_directory):

    logger = logging.getLogger(__name__)

    filenames = os.listdir(filtered_location)
    filenames_sorted = sorted(filenames)

    f_out = os.path.join(output_directory, "standardized.bin")
    logger.info('...saving standardized file: %s', f_out)

    f = open(f_out, 'wb')
    for fname in filenames_sorted:
        res = np.load(os.path.join(filtered_location, fname))
        res.tofile(f)
        os.remove(os.path.join(filtered_location, fname))