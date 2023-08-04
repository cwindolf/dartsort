import numpy as np
import multiprocessing
import h5py
from scipy.stats import linregress
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.stats import t


def get_spike_velocity(wf, sp_idx, local_geom):
    
    nan_ch = np.unique(np.where(np.isnan(wf))[1])
    
    wf = np.delete(wf, nan_ch, axis = 1)
    sp_idx = np.delete(sp_idx, nan_ch)
    local_geom = np.delete(local_geom, nan_ch, axis = 0)
    
    colomn_idx = local_geom[:,0] == 0
    
    # sp_idx[colomn_idx]
    
    # above_idx = (local_geom[:,1] >= 0) & sp_idx
    # below_idx = (local_geom[:,1] <= 0) & sp_idx

    max_idx = np.nanargmax(np.abs(wf))
    
    if np.sign(wf.flatten()[max_idx]) == 1:
        times = np.nanargmax(wf, axis = 0)
    else:
        times = np.nanargmin(wf, axis = 0)
        
    if (np.sum(sp_idx[colomn_idx]) > 1)& (len(np.unique(local_geom[sp_idx & colomn_idx,1]))>1):
        slope, intercept, r_value, p_value, std_err = linregress(local_geom[sp_idx&colomn_idx,1], times[sp_idx & colomn_idx])
        tinv = lambda p_value, df: abs(t.ppf(p_value/2, df))
        ts = tinv(0.05, len(times[sp_idx&colomn_idx])-2)
        ci_error = ts*std_err
        velocity = slope
        ci = ci_error
    else:
        velocity = np.nan
        ci = np.nan



#     if (np.sum(above_idx) > 1) & (len(np.unique(local_geom[above_idx,1]))>1):
#         slope_above, intercept, r_value, p_value, std_err = linregress(local_geom[above_idx,1], times[above_idx])
#         tinv = lambda p_value, df: abs(t.ppf(p_value/2, df))
#         ts = tinv(0.05, len(times[below_idx])-2)
#         ci_error = ts*std_err
#         velocity_above = slope_above 
#         ci_above = ci_error
#     else:
#         velocity_above = np.nan
#         ci_above = np.nan

    # if (np.sum(below_idx) > 1)& (len(np.unique(local_geom[below_idx,1]))>1):
    #     slope_below, intercept, r_value, p_value, std_err = linregress(local_geom[below_idx,1], times[below_idx])
    #     tinv = lambda p_value, df: abs(t.ppf(p_value/2, df))
    #     ts = tinv(0.05, len(times[below_idx])-2)
    #     ci_error = ts*std_err
    #     velocity_below = slope_below 
    #     ci_below = ci_error
    # else:
    #     velocity_below = np.nan
    #     ci_below = np.nan

    return velocity, ci



def get_spikes_velocity(wfs,
                        geom,
                        maxchans, 
                        spread_idx,
                        channel_index,
                        n_workers=None,
                        radius = None,
                        n_channels = None
                       ):
    N, T, C = wfs.shape
    maxchans = maxchans.astype(int)

    local_geoms = np.pad(geom, [(0, 1), (0, 0)])[channel_index[maxchans]]
    local_geoms[:, :, 1] -= geom[maxchans, 1][:, None]
    local_geoms[:, :, 0] -= geom[maxchans, 0][:, None]
    
    if n_channels is not None or radius is not None:
        subset = channel_index_subset(
            geom, channel_index, n_channels=n_channels, radius=radius
        )
    else:
        subset = [slice(None)] * len(geom)
        
    xqdm = tqdm

    # -- run the linear regression
    # vel_above = np.empty(N)
    # vel_below = np.empty(N)
    # ci_above = np.empty(N)
    # ci_below = np.empty(N)
    
    vel = np.empty(N)
    ci_err = np.empty(N)
    
    with Parallel(n_workers) as pool:
        for n, (v, ci) in enumerate(
        # for n, (v1, v2, ci1, ci2) in enumerate(
            pool(
                delayed(get_spike_velocity)(
                    wf[:,subset[mc]],
                    sp_idx[subset[mc]],
                    local_geom[subset[mc]],
                )
                for wf, sp_idx, mc, local_geom in xqdm(
                    zip(wfs, spread_idx, maxchans, local_geoms), total=N, desc="lsq"
                )
            )
        ):
            vel[n] = v
            ci_err[n] = ci
            # vel_above[n] = v1
            # vel_below[n] = v2
            # ci_above[n] = ci1
            # ci_below[n] = ci2
            
    return vel, ci #vel_above, vel_below, ci_above, ci_below
