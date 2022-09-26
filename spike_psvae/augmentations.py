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
    
    noise = np.random.normal(0, 1, templates.shape)
    templates = np.add(templates, noise)
    return templates

def smart_noisify2(max_chan_templates, temporal_cov):
    # temporal_cov = np.load('temporal_cov_example.npy')
    n_templates, n_times = max_chan_templates.shape
    
    noise_2_add = np.random.multivariate_normal(np.zeros(n_times), temporal_cov, n_templates)
    noisy_templates = np.add(max_chan_templates, noise_2_add)
    
    return noisy_templates
