import h5py
import random
import numpy as np
import scipy as sp


def amp_jitter(max_chan_templates):
    
    ''' jitter window around ptp 
    
        max_chan_templates: max ptp templates of each cluster
        
        Returns: max channel templates with ptps jittered by gaussian noise
    '''
    templates = np.copy(max_chan_templates)
    n_templates, n_times = templates.shape

    # get trough times and find corresponding peak
    chan_trough_times = np.array([np.argmin(template) for template in templates])
    trough_masked_temps = np.array([np.pad(templates[i, chan_trough_times[i]+1:-3], (chan_trough_times[i], 0), 'constant', constant_values=(-1000)) for i in range(n_templates)])
    chan_peak_times = np.array([np.argmax(temp) for temp in trough_masked_temps])

    scale_facs = np.random.uniform(0.9, 1.1, n_templates)
    scale_masks = []
    for i in range(len(templates)):
        scale_fac, trough_time, peak_time = scale_facs[i], chan_trough_times[i], chan_peak_times[i]
        # make scaled mask for each channel in the current template
        scale_masks.append(np.pad(scale_fac * np.ones(peak_time-trough_time+7), (trough_time-3, n_times-peak_time-4), 'constant', constant_values=(1)))
    scale_masks = np.array(scale_masks)

    amp_jittered_templates = np.multiply(templates, scale_masks)
    
    return amp_jittered_templates

def amp_jitter_trough(max_chan_templates):
    
    ''' jitter window around trough of wf
    
        max_chan_templates: max ptp templates of each cluster
        
        Returns: max channel templates with troughs jittered by gaussian noise
    '''
    templates = np.copy(max_chan_templates)
    n_templates, n_times = templates.shape

    chan_trough_times = np.array([np.argmin(template) for template in templates])

    scale_facs = np.random.uniform(0.9, 1.1, n_templates)
    scale_masks = []
    for i in range(len(templates)):
        scale_fac, trough_time = scale_facs[i], chan_trough_times[i]
        # make scaled mask for each channel in the current template
        scale_arr = np.pad(scale_fac * np.ones(9), (trough_time-4, n_times-trough_time-5), 'constant', constant_values=(1))
        scale_masks.append(scale_arr)
    scale_masks = np.array(scale_masks)

    amp_jittered_templates = np.multiply(templates, scale_masks)
    
    return amp_jittered_templates

def amp_whole_jitter(max_chan_templates):
    
    ''' jitter amplitude of whole template 
    
        max_chan_templates: max ptp templates of each cluster
        
        Returns: max channel templates with amplitudes jittered by gaussian noise
    '''
    templates = np.copy(max_chan_templates)
    n_templates, n_times = templates.shape

    chan_trough_times = np.array([np.argmin(template) for template in templates])

    scale_facs = np.random.normal(1, 0.05, n_templates)
    scale_masks = np.ones(templates.shape)
    scale_masks = np.array([scale_facs[i]*scale_masks[i] for i in range(n_templates)])

    amp_jittered_templates = np.multiply(templates, scale_masks)
    
    return amp_jittered_templates


def jitter(max_chan_templates, up_factor=8):
    templates = np.copy(max_chan_templates)

    n_templates, n_times = templates.shape
    print(templates.shape)

    # upsample best fit template
    up_temp = sp.signal.resample(
        x=templates,
        num=n_times*up_factor,
        axis=1)
    up_temp = up_temp.T

    idx = (np.arange(0, n_times)[:,None]*up_factor + np.arange(up_factor))
    up_shifted_temps = up_temp[idx].transpose(2,0,1)
    up_shifted_temps = np.concatenate(
        (up_shifted_temps,
            np.roll(up_shifted_temps, shift=1, axis=1)),
        axis=2)
    templates = up_shifted_temps.transpose(0,2,1).reshape(-1, n_times)

    ref = np.mean(templates, 0)
    shifts = align_get_shifts_with_ref(
        templates, ref, upsample_factor=1)
    templates = shift_chans(templates, shifts)
    
    add_shifts = (2* np.random.binomial(1, 0.5, n_templates)-1) * np.random.uniform(0, 2, n_templates)
    templates = shift_chans(templates, add_shifts)
    
    return templates

def align_get_shifts_with_ref(wf, ref=None, upsample_factor=5, nshifts=7):

    ''' Align all waveforms on a single channel
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    # Cat: TODO: Peter's fix to clip/lengthen loaded waveforms to match reference templates    
    n_data, n_time = wf.shape

    if ref is None:
        ref = np.mean(wf, axis=0)

    #n_time_rf = len(ref)
    #if n_time > n_time_rf:
    #    left_cut = (n_time - n_time_rf)//2
    #    right_cut = n_time - n_time_rf - left_cut
    #    wf = wf[:, left_cut:-right_cut]
    #elif n_time < n_time_rf:
    #    left_buffer = np.zeros((n_data, (n_time_rf - n_time)//2))
    #    right_buffer = np.zeros((n_data,n_time_rf - n_time - left_buffer))
    #    wf = np.concatenate((left_buffer, wf, right_buffer), axis=1)
      
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1

    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = nshifts//2
    wf_end = -nshifts//2
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last channel which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-(nshifts//2), (nshifts//2)+1)):
        ref_shifted[:,i] = ref_upsampled[s + wf_start: s + wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts/np.float32(upsample_factor)

def upsample_resample(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = sp.signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces

def shift_chans(wf, best_shifts):
    # use template feat_channel shifts to interpolate shift of all spikes on all other chans
    wfs_final= np.zeros(wf.shape, 'float32')
    for k, shift_ in enumerate(best_shifts):
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        curr_wf_pos = np.pad(wf[k], (0, int_shift), 'constant') 
        curr_wf_neg = np.pad(wf[k], (int_shift, 0), 'constant')
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(curr_wf_pos,ceil,axis=0)[:-int_shift] if shift_ > 0 else np.roll(curr_wf_neg,ceil,axis=0)[int_shift:]
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            if shift_ > 0:
                temp = (np.roll(curr_wf_pos,ceil,axis=0)*(shift_-floor))[:-ceil] + (np.roll(curr_wf_pos,floor, axis=0)*(ceil-shift_))[:-ceil]
            else:
                temp = (np.roll(curr_wf_neg,ceil,axis=0)*(shift_-floor))[-floor:] + (np.roll(curr_wf_neg,floor, axis=0)*(ceil-shift_))[-floor:]
        wfs_final[k] = temp
    
    return wfs_final


def collide_templates(templates):
    
    ''' collide max channel templates with randomly scaled wf of another template
    
        templates: templates of each cluster on every channel (n_clusters, n_times, n_channels)
        
        Returns: collided max channel templates and randomly selected, scaled, and offset wfs that are used to make collision
    '''
    max_chan_templates = np.copy(templates)
    n_templates, n_times = max_chan_templates.shape
    
    merge_temp_inds = np.random.choice(np.repeat(np.arange(n_templates), 10), n_templates, replace=True)
    for i in range(len(merge_temp_inds)):
        while merge_temp_inds[i] == i:
            merge_temp_inds[i] = np.random.choice(np.arange(n_templates), 1)
    merge_temps = max_chan_templates[merge_temp_inds]
    merge_temps_peak_inds = np.array([np.argmax(template, axis=0) for template in merge_temps])
    
    temp_scale_fac = np.random.uniform(0.2, 1, n_templates)
    temp_offsets = (2* np.random.binomial(1, 0.5, n_templates)-1) * np.random.randint(5, 60, n_templates)
    
    offset_merge_temps = shift_chans(merge_temps, temp_offsets)
    scaled_offset_temps = np.array([np.multiply(scale, temp) for (scale, temp) in zip(temp_scale_fac, offset_merge_temps)])
    collided_temps = np.add(max_chan_templates, scaled_offset_temps)

    return collided_temps, scaled_offset_temps

def noisify(max_chan_templates):
    
    ''' add random gaussian noise to each timestep
    
        max_chan_templates: max ptp templates of each cluster
        
        Returns: max channel templates with noise added
    '''
    templates = np.copy(max_chan_templates)
    n_templates, n_times = templates.shape
    
    noise = np.random.normal(0, 0.1, templates.shape)
    templates = np.add(templates, noise)
    return templates

def noise_wfs(recordings, temporal_size, window_size, sample_size=1000,
              threshold=3.0, max_trials_per_sample=100,
              allow_smaller_sample_size=False):
    """Compute noise temporal and spatial covariance
    Parameters
    ----------
    recordings: numpy.ndarray
        Recordings
    temporal_size:
        Waveform size
    sample_size: int
        Number of noise snippets of temporal_size to search
    threshold: float
        Observations below this number are considered noise
    Returns
    -------
    spatial_SIG: numpy.ndarray
    temporal_SIG: numpy.ndarray
    """
    # logger = logging.getLogger(__name__)

    # kill signal above threshold in recordings
    print('Get Noise Floor')
    rec, is_noise_idx = kill_signal(recordings, threshold, window_size)

#     # compute spatial covariance, output: (n_channels, n_channels)
#     print('Compute Spatial Covariance')
#     spatial_cov = np.divide(np.matmul(np.transpose(rec, (0, 2, 1)), rec),
#                             np.matmul(np.transpose(is_noise_idx, (0, 2, 1)), is_noise_idx))
    
#     spatial_cov[np.isnan(spatial_cov)] = 0
#     spatial_cov[np.isinf(spatial_cov)] = 0
#     # print(np.isnan(spatial_cov))

#     # compute spatial sig
#     w_spatial, v_spatial = np.linalg.eig(spatial_cov)
#     diag_mat_spatial = np.array([np.diag(1/np.sqrt(w)) for w in w_spatial])
#     diag_mat_whitener = np.array([np.diag(np.sqrt(w)) for w in w_spatial])
#     spatial_SIG = np.matmul(np.matmul(v_spatial,
#                                       diag_mat_spatial),
#                             np.transpose(v_spatial, (0, 2, 1)))

#     # apply spatial whitening to recordings
#     print('Compute Temporal Covariance')
#     spatial_whitener = np.matmul(np.matmul(v_spatial, diag_mat_whitener),
#                                  np.transpose(v_spatial, (0, 2, 1)))
    # rec = np.matmul(rec, spatial_whitener)

    # search single noise channel snippets
    noise_wf = search_noise_snippets(
        recordings, is_noise_idx, sample_size,
        temporal_size,
        channel_choices=None,
        max_trials_per_sample=max_trials_per_sample,
        allow_smaller_sample_size=allow_smaller_sample_size)

    return noise_wf


def kill_signal(recordings, threshold, window_size):
    """
    Thresholds recordings, values above 'threshold' are considered signal
    (set to 0), a window of size 'window_size' is drawn around the signal
    points and those observations are also killed
    Returns
    -------
    recordings: numpy.ndarray
        The modified recordings with values above the threshold set to 0
    is_noise_idx: numpy.ndarray
        A boolean array with the same shap as 'recordings' indicating if the
        observation is noise (1) or was killed (0).
    """
    new_recordings = np.copy(recordings)

    n_templates, T, C = new_recordings.shape
    R = int((window_size-1)/2)

    # this will hold a flag 1 (noise), 0 (signal) for every obseration in the
    # recordings
    is_noise_idx = np.zeros(new_recordings.shape)

    # go through every neighboring channel on every cluster
    for t in range(n_templates):
        for c in range(C):

            # get observations where observation is above threshold
            idx_temp = np.where(np.abs(new_recordings[t, :, c]) > threshold)[0]

            # shift every index found
            for j in range(-R, R+1):

                # shift
                idx_temp2 = idx_temp + j
                

                # remove indexes outside range [0, T]
                idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
                                                     idx_temp2 < T)]

                # set surviving indexes to nan
                new_recordings[t, idx_temp2, c] = np.nan

            # noise indexes are the ones that are not nan
            # FIXME: compare to np.nan instead
            is_noise_idx_temp = (new_recordings[t, :, c] == recordings[t, :, c])
            if is_noise_idx_temp.shape[0] != 121:
                print(is_noise_idx_temp.shape)
                
            if np.isnan(np.nanstd(new_recordings[t, :, c])) or np.nanstd(new_recordings[t, :, c]) == 0:
                # print(new_recordings[t, :, c])
                std_dev = 10**-5
            else: 
                std_dev = np.nanstd(new_recordings[t, :, c])

            # standarize data, ignoring nans
            new_recordings[t, :, c] = recordings[t, :, c]/std_dev

            # set non noise indexes to 0 in the recordings
            new_recordings[t, ~is_noise_idx_temp, c] = 10**-7

            # save noise indexes
            is_noise_idx[t, is_noise_idx_temp, c] = 1

    return new_recordings, is_noise_idx


def search_noise_snippets(recordings, is_noise_idx, sample_size,
                          temporal_size, channel_choices=None,
                          max_trials_per_sample=1000,
                          allow_smaller_sample_size=True):
    """
    Randomly search noise snippets of 'temporal_size'
    Parameters
    ----------
    channel_choices: list
        List of sets of channels to select at random on each trial
    max_trials_per_sample: int, optional
        Maximum random trials per sample
    allow_smaller_sample_size: bool, optional
        If 'max_trials_per_sample' is reached and this is True, the noise
        snippets found up to that time are returned
    Raises
    ------
    ValueError
        if after 'max_trials_per_sample' trials, no noise snippet has been
        found this exception is raised
    Notes
    -----
    Channels selected at random using the random module from the standard
    library (not using np.random)
    """

    n_templates, T, C = recordings.shape

    if channel_choices is None:
        noise_wf = np.zeros((sample_size, temporal_size))
    else:
        lenghts = set([len(ch) for ch in channel_choices])

        if len(lenghts) > 1:
            raise ValueError('All elements in channel_choices must have '
                             'the same length, got {}'.format(lenghts))

        n_channels = len(channel_choices[0])
        noise_wf = np.zeros((sample_size, temporal_size, n_channels))

    count = 0

    # logger.debug('Starting to search noise snippets...')

    trial = 0

    # repeat until you get sample_size noise snippets
    while count < sample_size:

        # random number for the start of the noise snippet
        t_start = 0

        if channel_choices is None:
            # random channel
            temp = random.randint(0, n_templates-1)
            ch = random.randint(0, C - 1)
        else:
            temp = random.randint(0, n_templates-1)
            ch = random.choice(channel_choices)

        t_slice = slice(t_start, t_start+temporal_size)

        # get a snippet from the recordings and the noise flags for the same
        # location
        snippet = recordings[temp, t_slice, ch]
        snipped_idx_noise = is_noise_idx[temp, t_slice, ch]

        # check if all observations in snippet are noise
        if snipped_idx_noise.all():
            # add the snippet and increase count
            noise_wf[count] = snippet
            count += 1
            trial = 0


        trial += 1

        if trial == max_trials_per_sample:
            if allow_smaller_sample_size:
                return noise_wf[:count]
            else:
                raise ValueError("Couldn't find snippet {} of size {} after "
                                 "{} iterations (only {} found)"
                                 .format(count + 1, temporal_size,
                                         max_trials_per_sample,
                                         count))

    return noise_wf

def smart_noisify(templates, max_chan_templates, noise_thresh=1):
    
    ''' add realistic noise computed from all templates to the max channel templates
    
        templates: templates of each cluster on every channel (n_clusters, n_times, n_channels)
        max_chan_templates: max ptp templates of each cluster
        noise_thresh: observations below this number are considered noise
        
        Returns: max channel templates with realistic noise added
    '''
    noise_wf = noise_wfs(templates, 121, 20, threshold=noise_thresh, max_trials_per_sample=1000000)
    
    noise_selections = np.random.randint(0, len(noise_wf), len(max_chan_templates))
    
    noise_2_add = noise_wf[noise_selections]
    noisy_templates = np.add(max_chan_templates, noise_2_add)
    
    return noisy_templates

