# %%
import os
from tqdm.auto import tqdm, trange
import numpy as np
import multiprocessing
from .multiprocessing_utils import get_pool, MockQueue
from scipy.signal import butter, filtfilt
from spike_psvae import spikeio
import numpy.fft as fft
import math

#ADC shift correction
def phaseShiftSig(sig, fs, nSamples):
    # % function sig1 = phaseShiftSig(sig, fs, nSamples)
    # %
    # % Shift the fourier phase components of a vector to perform a sub-sample
    # % shifting of the data, return the data in the time domain.

    n = len(sig)
    f = np.arange(-n/2,n/2)*fs/n
    # % take fft
    y = fft.fftshift(fft.fft(sig))/n

    # % shift the phase of each sample in a frequency-dependent manner so the
    # % absolute time shift is constant across frequencies 
    y1 = y*np.exp(-2*math.pi*1j*f*nSamples/fs) #ELEMENT WISE MULTIPLICATION

    # % ifft back to time domain
    sig1 = np.real(n*(fft.ifft(fft.ifftshift(y1))))
    return sig1

def npSampShifts():
# % return the true sampling time (relative to the average sample time) of
# % each channel of a neuropixels 1.0 or 2.0 probe. 

    nCh = 384
    sampShifts = np.zeros(nCh)
#     if ver:
#         case 1
    nADC = 32 #similar to NP 1

    nChPerADC = nCh//nADC
    startChan = np.array([np.arange(0, nCh, nChPerADC*2), np.arange(1, nCh, nChPerADC*2)]).flatten()
    
    for n in range(nADC):
        sampShifts[np.arange(startChan[n], startChan[n]+2*nChPerADC-1, 2)] = np.arange(nChPerADC)

# % here just centering the shift amount (arbitrary) and returning in units
# % of samples
    sampShifts = (sampShifts-nChPerADC/2)/nChPerADC
    return sampShifts

def shiftWF(thisWF):
# % thisWF is a matrix of raw neuropixels data nChannels x nSamples
# % ver is either 1 or 2, for a 1.0 or 2.0 Neuropixels probe
# % newWF has the same size as thisWF but appropriately shifted


# % determine how much each channel should be shifted by for this probe
    sampShifts = npSampShifts()

    assert(len(sampShifts)==len(thisWF))

#     % go through each channel and shift that channel alone
    fs = 30000
    newWF = np.zeros(thisWF.shape)
    for ch in range(len(sampShifts)):
        newWF[ch,:] = phaseShiftSig(thisWF[ch,:], fs, sampShifts[ch])
    
    return newWF

def filter_standardize_rec_mp(output_directory, filename_raw, dtype_raw,
    rec_len_sec, n_channels = 384,
    dtype_output = np.float32,
    apply_filter = True,
    low_frequency =300, high_factor = 0.1, order = 3, sampling_frequency= 30000, 
    channels_to_remove=None,
    buffer = None, t_start=0, t_end=None,
    n_sec_chunk=1, multi_processing = True, n_jobs = 1, overwrite = True,
    adcshift_correction=False,median_subtraction=False):
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

    small_batch = spikeio.read_data(
        filename_raw, 
        dtype = dtype_raw,
        s_start=rec_len//2 - chunk_5sec//2,
        s_end=rec_len//2 + chunk_5sec//2, 
        n_channels = n_channels)

    fname_mean_sd = os.path.join(
        output_directory, 'mean_and_standard_dev_value.npz')
    if overwrite or not os.path.exists(fname_mean_sd):
        get_std(small_batch, sampling_frequency,
                fname_mean_sd, apply_filter,
                low_frequency, high_factor, order)

    # turn it off
    small_batch = None

    # Make directory to hold filtered batch files:
    filtered_location = os.path.join(output_directory, "filtered_files")
    if not os.path.exists(filtered_location):
        os.makedirs(filtered_location)
    
    if t_end is not None:
        n_batches = (t_end-t_start)//n_sec_chunk
        all_batches = np.arange(t_start, t_end, n_sec_chunk)
    else:
        n_batches = rec_len_sec//n_sec_chunk
        all_batches = np.arange(n_batches)
    
#     my_fnames = [
#         Path(filtered_location) / f"standardized_{bid:06d}.npy" for bid in all_batches
#     ]

    # define a size of buffer if not defined
    if buffer is None:
        buffer = int(max(sampling_frequency/100, 200))

    
    n_jobs = n_jobs or 1
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() - 1
    
    if n_jobs > 1:
        ctx = multiprocessing.get_context("spawn")


    if n_jobs <= 1:
        filter_standardize_for_loop(all_batches,
            filename_raw,
            fname_mean_sd,
            apply_filter,
            dtype_raw, 
            dtype_output,
            filtered_location,
            n_channels, 
            buffer,
            rec_len, 
            low_frequency,
            high_factor,
            order,
            sampling_frequency,
            channels_to_remove,
            adcshift_correction,
            median_subtraction,
        )
    else:
        mp_object = multi_proc_object(
            filename_raw,
            fname_mean_sd,
            apply_filter,
            dtype_raw, 
            dtype_output,
            filtered_location,
            n_channels, 
            buffer,
            rec_len, 
            low_frequency,
            high_factor,
            order,
            sampling_frequency,
            channels_to_remove,
            adcshift_correction,
            median_subtraction)
        
        with ctx.Pool(
            n_jobs,
        ) as pool:
            for res in tqdm(
                pool.imap_unordered(
                    mp_object.filter_standardize_batch_mp,
                    all_batches,
                ),
                total=len(all_batches),
                desc="Preprocessing",
            ):
                pass    
            
    merge_filtered_files(filtered_location, output_directory)

    return standardized_path, standardized_params['dtype']


class multi_proc_object:
    
    def __init__(
        self,
        filename_raw,
        fname_mean_sd,
        apply_filter,
        dtype_raw, 
        dtype_output,
        filtered_location,
        n_channels, 
        buffer,
        rec_len, 
        low_frequency,
        high_factor,
        order,
        sampling_frequency,
        channels_to_remove,
        adcshift_correction,
        median_subtraction):
        
        self.filename_raw = filename_raw
        self.fname_mean_sd = fname_mean_sd
        self.apply_filter = apply_filter
        self.dtype_raw = dtype_raw 
        self.dtype_output = dtype_output
        self.filtered_location = filtered_location
        self.n_channels = n_channels 
        self.buffer = buffer
        self.rec_len = rec_len 
        self.low_frequency = low_frequency
        self.high_factor = high_factor
        self.order = order
        self.sampling_frequency = sampling_frequency
        self.channels_to_remove = channels_to_remove
        self.adcshift_correction = adcshift_correction
        self.median_subtraction = median_subtraction

    def filter_standardize_batch_mp(self, batch_id):
        filter_standardize_batch(
            batch_id, self.filename_raw, 
            self.fname_mean_sd,
            self.apply_filter,
            self.dtype_raw, 
            self.dtype_output,
            self.filtered_location,
            self.n_channels, 
            self.buffer,
            self.rec_len, 
            self.low_frequency,
            self.high_factor,
            self.order,
            self.sampling_frequency,
            self.channels_to_remove,
            self.adcshift_correction,
            self.median_subtraction,
            )        
        

        
        
def filter_standardize_for_loop(all_batches,
            filename_raw,
            fname_mean_sd,
            apply_filter,
            dtype_raw, 
            dtype_output,
            filtered_location,
            n_channels, 
            buffer,
            rec_len, 
            low_frequency,
            high_factor,
            order,
            sampling_frequency,
            channels_to_remove,
            adcshift_correction,
            median_subtraction):
    
    for batch_id in all_batches:
        filter_standardize_batch(
            batch_id,
            filename_raw,
            fname_mean_sd,
            apply_filter,
            dtype_raw, 
            dtype_output,
            filtered_location,
            n_channels, 
            buffer,
            rec_len, 
            low_frequency,
            high_factor,
            order,
            sampling_frequency,
            channels_to_remove,
            adcshift_correction,
            median_subtraction,
            )


# %%
def filter_standardize_rec(output_directory, filename_raw, dtype_raw,
    rec_len_sec, n_channels = 384,
    dtype_output = np.float32,
    apply_filter = True,
    low_frequency =300, high_factor = 0.1, order = 3, sampling_frequency= 30000, 
    channels_to_remove=None,
    buffer = None, t_start=0, t_end=None,
    n_sec_chunk=1, multi_processing = True, n_processors = 6, overwrite = True,
    adcshift_correction=False,median_subtraction=False):
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

    small_batch = spikeio.read_data(
        filename_raw, 
        dtype = dtype_raw,
        s_start=rec_len//2 - chunk_5sec//2,
        s_end=rec_len//2 + chunk_5sec//2, 
        n_channels = n_channels)

    fname_mean_sd = os.path.join(
        output_directory, 'mean_and_standard_dev_value.npz')
    if overwrite or not os.path.exists(fname_mean_sd):
        get_std(small_batch, sampling_frequency,
                fname_mean_sd, apply_filter,
                low_frequency, high_factor, order)

    # turn it off
    small_batch = None

    # Make directory to hold filtered batch files:
    filtered_location = os.path.join(output_directory, "filtered_files")
    if not os.path.exists(filtered_location):
        os.makedirs(filtered_location)
    
    if t_end is not None:
        n_batches = (t_end-t_start)//n_sec_chunk
        all_batches = np.arange(t_start, t_end, n_sec_chunk)
    else:
        n_batches = rec_len_sec//n_sec_chunk
        all_batches = np.arange(n_batches)

    # define a size of buffer if not defined
    if buffer is None:
        buffer = int(max(sampling_frequency/100, 200))

# Multiprocessing doesn't work - switch to spikeinterface
    # read config params
#     if multi_processing:
#         parmap.map(
#             filter_standardize_batch,
#             [i for i in all_batches],
#             filename_raw, 
#             fname_mean_sd,
#             apply_filter,
#             dtype_raw, 
#             dtype_output,
#             filtered_location,
#             n_channels,
#             buffer,
#             rec_len,
#             low_frequency,
#             high_factor,
#             order,
#             sampling_frequency, 
#             channels_to_remove,
#             adcshift_correction,
#             median_subtraction,
#             processes=n_processors,
#             pm_pbar=True)
#     else:
    for batch_id in all_batches:
        filter_standardize_batch(
            batch_id, filename_raw, 
            fname_mean_sd,
            apply_filter,
            dtype_raw, 
            dtype_output,
            filtered_location,
            n_channels, 
            buffer,
            rec_len, 
            low_frequency,
            high_factor,
            order,
            sampling_frequency,
            channels_to_remove,
            adcshift_correction,
            median_subtraction,
            )

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(filtered_location, output_directory)

    return standardized_path, standardized_params['dtype']


# %%

# %%
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


# %%
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


# %%
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


# %%

# %%
def filter_standardize_batch(batch_id, bin_file, fname_mean_sd,
                             apply_filter, dtype_input, out_dtype, output_directory,
                             n_channels, buffer, rec_len,
                             low_frequency=None, high_factor=None,
                             order=None, sampling_frequency=None, channels_to_remove=None,
                             adcshift_correction=False,median_subtraction=False):
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
        buffer_before = True
        buffer_after = True
        # read a batch
        # Add buffer into s_start and s_end
        s_start = batch_id*sampling_frequency-buffer
        s_end = (batch_id+1)*sampling_frequency+buffer
        if s_start<0:
            s_start=0
            buffer_before = False
        if s_end>rec_len:
            s_end = rec_len
            buffer_after = False
        ts = spikeio.read_data(bin_file, dtype_input, s_start, s_end, n_channels)
        ts = _butterworth(ts, low_frequency, high_factor,
                              order, sampling_frequency)
        if buffer_before:
            ts = ts[buffer:]
        if buffer_after:
            ts = ts[:-buffer]
    else:
        # read a batch
        s_start = batch_id*sampling_frequency
        s_end = (batch_id+1)*sampling_frequency
        if s_start<0:
            s_start=0
        if s_end>rec_len:
            s_end = rec_len
        ts = spikeio.read_data(bin_file, dtype_input, s_start, s_end, n_channels)
    # standardize
    temp = np.load(fname_mean_sd)
    sd = temp['sd']
    centers = temp['centers']
    ts = _standardize(ts, sd, centers)
    if channels_to_remove is not None:
        
        ts = np.delete(ts, channels_to_remove, axis=1)
    
    if adcshift_correction:
        ts = shiftWF(ts.T).T
    if median_subtraction:
        ts = ts - np.median(ts, axis = 1)[:, None]
    
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


# %%

# %%
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


# %%
def merge_filtered_files(filtered_location, output_directory):

    filenames = os.listdir(filtered_location)
    filenames_sorted = sorted(filenames)

    f_out = os.path.join(output_directory, "standardized.bin")

    f = open(f_out, 'wb')
    for fname in tqdm(filenames_sorted):
        res = np.load(os.path.join(filtered_location, fname))
        res.tofile(f)
        os.remove(os.path.join(filtered_location, fname))
