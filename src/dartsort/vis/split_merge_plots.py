import numpy as np
from sklearn.decomposition import PCA
from dartsort.util import spikeio
from dartsort.util.drift_util import registered_geometry, get_spike_pitch_shifts, get_waveforms_on_static_channels
from dartsort.templates.templates import TemplateData
from dartsort.cluster.merge import calculate_merge_distances
from dartsort.util.data_util import chunk_time_ranges, subchunks_time_ranges

from dartsort.vis.unit import correlogram, bar
from tqdm.auto import tqdm

import colorcet as cc 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import h5py

ccolors = np.array(cc.glasbey[:31])
# %%
def get_ccolor(k):
    if isinstance(k, int):
        if k == -1:
            return "#808080"
        else:
            return ccolors[k % len(ccolors)]
    else:
        col_array = ccolors[k % len(ccolors)]
        col_array[k==-1] = "#808080"
        return col_array

def check_overmerges_post_deconv_with_smoothed_templates(
    recording,
    template_config,
    matching_config,
    sorting_post_split,
    chunk_time_ranges_s,
    output_directory,
    times_seconds_list,
    localization_results_list,
    a_list,
    depth_reg_list,
    collisioncleaned_tpca_features_all_list, 
    channels_list,
    tpca_list,
    channel_index,
    geom,
    me,
    tpca_templates_list, #can be temp_data_smoothed, here one per subchunk
    triaged_idx=None,
    units_all=None,
    n_neigh = 3,
    zlim = (-100, 382),
    bin_ms=0.1, 
    max_ms=5,
    max_lag=50,
    time_smoothed=False,
):

    """
    here, everything should be a list containing infor about each chunk!! 
    """

    tpca_templates_list[np.isnan(tpca_templates_list)]=0

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 4* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    units_all = np.unique(sorting_post_split.labels)
    units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)

    if triaged_idx is None: 
        triaged_idx=np.zeros(sorting_post_split.labels.shape, type='int')
    color_array = np.take(["blue", "red"], triaged_idx)

    for unit in tqdm(units_all): 

        idx_unit = np.flatnonzero(sorting_post_split.labels==unit)
        
        # if previous_labels is not None:
        #     units_prior = np.unique(previous_labels[sorting_post_split.labels == unit])
        # else:
        #     units_prior = unit
        
        fig = plt.figure(figsize=(40, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :(n_cols-2)//2]) 
        ax_z_time = fig.add_subplot(gs[0, (n_cols-2)//2:-2]) 
        ax_fr_time = fig.add_subplot(gs[0, -2:]) 

        idx_unit = np.flatnonzero(sorting_post_split.labels==unit)
        # if unit_labels.max()>len(colors_split):

        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_unit], sorting_post_split.denoised_ptp_amplitudes[idx_unit], c=color_array[idx_unit], s=1, alpha = 0.25)
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_unit], sorting_post_split.point_source_localizations[idx_unit, 2], c=color_array[idx_unit], s=1, alpha = 0.25)
            
        med_depth_reg = np.median(sorting_post_split.point_source_localizations[idx_unit, 2])
        std_depth_reg = np.median(np.abs(sorting_post_split.point_source_localizations[idx_unit, 2] - med_depth_reg))/0.675
        ax_z_time.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))
        
        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_fr_time.set_ylabel("FR (N/s)")
        ax_fr_time.set_xlabel("Time (s)")
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")
        # ax_z_time.set_ylim(zlim)

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()
        ax_fr_time.xaxis.set_label_position('top') 
        ax_fr_time.xaxis.tick_top()
        ax_fr_time.yaxis.tick_right()

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s[:-1]):

            sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_range, template_config.subchunk_size_s,
                                                  divider_samples=matching_config.chunk_length_samples)
            n_sub_chunks = len(sub_chunk_time_range_s)

            idx_chunk_time = np.logical_and(
                sorting_post_split.times_seconds>=chunk_range[0], sorting_post_split.times_seconds<chunk_range[1]
            )
            times_seconds = times_seconds_list[j]
            n_spike_chunks = times_seconds.size
            channels = channels_list[j]
            tpca = tpca_list[j] # This should change
            depth_reg = depth_reg_list[j]
            localization_results = localization_results_list[j]
            a = a_list[j]
            collisioncleaned_tpca_features_all = collisioncleaned_tpca_features_all_list[j]

            temp_unit_tpca = tpca_templates_list[j*n_sub_chunks:j*n_sub_chunks+n_sub_chunks].mean(0)[unit]
            subtract_tpca_temp = True
            
            ax_ptp_time.axvline(chunk_range[0], c='black')
            ax_ptp_time.axvline(chunk_range[1], c='black')
            ax_z_time.axvline(chunk_range[0], c='black')
            ax_z_time.axvline(chunk_range[1], c='black')

            ax_pcs = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+1]) 
            ax_ISI = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+2]) 
            ax_ACG = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+2]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4)+1:2*int(j//4)+3, 4*int(j - 4*(j//4))+3]) 

            for k, subchunk_range in enumerate(sub_chunk_time_range_s):
                idxtemp = int(j*n_sub_chunks+k)
                ax_ptp_time.plot([subchunk_range[0], subchunk_range[1]], [tpca_templates_list[idxtemp][unit].ptp(0).max(), tpca_templates_list[idxtemp][unit].ptp(0).max()],
                        c = "orange", label = "template ptp", 
                )
            # ax_z_time.plot([chunk_range[0], chunk_range[1]],
            #               [template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit], template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit]], c = "orange", label = "template depth")

            idx_chunk = np.arange(cmp, cmp+n_spike_chunks)
            # np.flatnonzero(sorting_post_split.labels[cmp:cmp+n_spike_chunks]==unit)

            all_indices_unit = np.flatnonzero(sorting_post_split.labels[idx_chunk]==unit)

            dt_ms = np.diff(times_seconds[all_indices_unit][triaged_idx[idx_chunk][all_indices_unit]==0]) * 1000
            lags, acg = correlogram(times_seconds[all_indices_unit][triaged_idx[idx_chunk][all_indices_unit]==0]*30_000, max_lag=max_lag)
                
            fr_unit = len(all_indices_unit)/(chunk_range[1]-chunk_range[0])
            ax_fr_time.plot([chunk_range[0], chunk_range[1]], [fr_unit, fr_unit], c = 'k')
            cmp+=n_spike_chunks

            
            n_nontriaged = triaged_idx[idx_chunk][all_indices_unit]==0
            n_triaged = triaged_idx[idx_chunk][all_indices_unit]==1
            if n_nontriaged.sum()>150:
                subsample_1 = np.random.choice(n_nontriaged.sum(), 150, replace=False)
                subsample_1 = all_indices_unit[n_nontriaged][subsample_1]
            else:
                subsample_1 = all_indices_unit[triaged_idx[idx_chunk][all_indices_unit]==0]
            if n_triaged.sum()>150:
                subsample_2 = np.random.choice(n_triaged.sum(), 150, replace=False)
                subsample_2 = all_indices_unit[n_triaged][subsample_2]
            else:
                subsample_2 = all_indices_unit[triaged_idx[idx_chunk][all_indices_unit]==1]
            # idx_subsample = np.random.choice(len(all_indices_unit), 150, replace=False)
            all_indices_unit = np.concatenate((subsample_1, subsample_2))
            all_indices_unit.sort()
        

            # TODO: stay around median
            if len(all_indices_unit):
                ax_loc.scatter(localization_results[all_indices_unit, 0], depth_reg[all_indices_unit], s=1, c = color_array[idx_chunk][all_indices_unit])
                med_depth_reg = np.median(depth_reg[all_indices_unit])
                std_depth_reg = np.median(np.abs(depth_reg[all_indices_unit] - med_depth_reg))/0.675
                std_depth_reg = max(std_depth_reg, 1)
                med_x = np.median(localization_results[all_indices_unit, 0])
                std_x = np.median(np.abs(localization_results[all_indices_unit, 0] - med_x))/0.675
                std_x = max(std_x, 1)
                ax_loc.set_xlim((med_x-5*std_x, med_x+5*std_x))
                ax_loc.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))
                ax_loc.set_title("Localization", fontsize=7)
                ax_loc.set_xlabel("x (um)", fontsize=7)
                ax_loc.set_ylabel("reg z (um)", fontsize=7)
            
            bin_edges = np.arange(
                0,
                max_ms + bin_ms,
                bin_ms,
            )
            ax_ISI.hist(dt_ms, bin_edges, color="k")
            ax_ISI.set_title("isi (ms)", fontsize=7)
            ax_ISI.set_ylabel(f"count (out of {dt_ms.size} isis)", fontsize = 7, labelpad=0)

            bar(ax_ACG, lags, acg, fill=True, color="k")
            ax_ACG.set_title("lag (samples)", fontsize=7)
            ax_ACG.set_ylabel("acg", fontsize=7, labelpad=0)

            # ax_ISI.yaxis.set_label_position("right")
            ax_ISI.yaxis.tick_right()
            # ax_ACG.yaxis.set_label_position("right")
            ax_ACG.yaxis.tick_right()
            
            ax_heatmap.imshow(temp_unit_tpca.ptp(0).reshape(-1, 8))
        
            n_pitches_shift = get_spike_pitch_shifts(localization_results[all_indices_unit, 2], geom, times_s = times_seconds[all_indices_unit], motion_est=me)

            temp_unit_tpca_bis = temp_unit_tpca.copy()
            temp_unit_tpca_bis[np.isnan(temp_unit_tpca_bis)]=0
            
            max_chan_registered_geom = temp_unit_tpca_bis.ptp(0).argmax()
            
            max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
            max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
            chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)

            temp_chunk = temp_unit_tpca[15:75][:, chans_to_plot_registered_geom]
            
            med_ptp = 1.5*temp_chunk.ptp(0).max()
            # med_ptp = 1.5*np.median(a[idx_chunk])
            
            # do all this before chunk / PCs...
            waveforms_target_chan = get_waveforms_on_static_channels(
                collisioncleaned_tpca_features_all[all_indices_unit],
                geom,
                main_channels=channels[all_indices_unit],
                channel_index=channel_index,
                target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                n_pitches_shift=n_pitches_shift,
                registered_geom=registered_geom,
            )
            no_nans = np.flatnonzero(np.isfinite(waveforms_target_chan[:, 0, :]).all(axis=1))

            if len(no_nans)>2:
                pcs = PCA(2).fit_transform(waveforms_target_chan.reshape(waveforms_target_chan.shape[0], -1)[no_nans])
                # idx_chunk = idx_chunk[no_nans]

                ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = color_array[idx_chunk][all_indices_unit][no_nans])
                ax_pcs.set_title("PCs", fontsize=7)
            
                # subsampled wfs to plot
                waveforms_target_chan = tpca.inverse_transform(waveforms_target_chan.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
                waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
            
                for k in range(5):
                    for i in range(waveforms_target_chan.shape[0]):
                        ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = color_array[idx_chunk][all_indices_unit][i], alpha = 0.05)
                        if not subtract_tpca_temp: 
                            ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = color_array[idx_chunk][all_indices_unit][i])
                        else:
                            ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_unit_tpca[15:75, chans_to_plot_registered_geom[k*2:k*2+2]].T.flatten() + k*med_ptp, alpha = 0.05, c = color_array[idx_chunk][all_indices_unit][i])
                ax_wfs_temp_subtracted.set_ylabel(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7, labelpad=0)
                ax_wfs.set_ylabel("Wfs", fontsize=7, labelpad = 0)

                ax_wfs_temp_subtracted.set_xticks([])
                ax_wfs_temp_subtracted.set_yticks([])
                ax_wfs.set_xticks([])
                ax_wfs.set_yticks([])
                    
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        plt.savefig(output_directory / f"unit_{unit}_overmerge")
        plt.close()
        

def plot_reassignment_units(
    labels_soft_assignment,
    hard_assigned,
    recording,
    template_config,
    matching_config,
    sorting_post_split,
    chunk_time_ranges_s,
    output_directory,
    times_seconds_list,
    localization_results_list,
    a_list,
    depth_reg_list,
    collisioncleaned_tpca_features_all_list, 
    channels_list,
    tpca_list,
    channel_index,
    geom,
    me,
    tpca_templates_list, #can be temp_data_smoothed, here one per subchunk
    units_all=None,
    n_neigh = 3,
    zlim = (-100, 382),
    bin_ms=0.1, 
    max_ms=5,
    max_lag=50,
    time_smoothed=False,
):

    """
    here, everything should be a list containing infor about each chunk!! 
    """

    colors_reassignment = ["blue", "green", "palegreen", "goldenrod", "red", "violet"]

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 4* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    units_all = np.unique(sorting_post_split.labels)
    units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)
    
    for unit in tqdm(units_all[units_all>37]): 

        idx_hard_assign = np.flatnonzero(np.logical_and(hard_assigned, 
            labels_soft_assignment[:, 0]==unit))
        idx_first_assign = np.flatnonzero(np.logical_and(~hard_assigned, 
            labels_soft_assignment[:, 0]==unit))
        idx_second_assign = np.flatnonzero(np.logical_and(~hard_assigned, 
            labels_soft_assignment[:, 1]==unit))
        idx_third_assign = np.flatnonzero(np.logical_and(~hard_assigned, 
            labels_soft_assignment[:, 2]==unit))
        idx_triaged = np.flatnonzero(np.logical_and(
            np.all(labels_soft_assignment == -1, axis=1), sorting_post_split.labels==unit))
        idx_toosmall = np.flatnonzero(
            labels_soft_assignment[:, 3] == unit)
        
        # if previous_labels is not None:
        #     units_prior = np.unique(previous_labels[sorting_post_split.labels == unit])
        # else:
        #     units_prior = unit
        
        fig = plt.figure(figsize=(40, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :(n_cols-2)//2]) 
        ax_z_time = fig.add_subplot(gs[0, (n_cols-2)//2:-2]) 
        ax_fr_time = fig.add_subplot(gs[0, -2:]) 

        idx_unit = np.flatnonzero(sorting_post_split.labels==unit)
        # if unit_labels.max()>len(colors_split):

        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_hard_assign], sorting_post_split.denoised_ptp_amplitudes[idx_hard_assign], c='blue', s=1, alpha = 0.25, label="hardassigned")
        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_first_assign], sorting_post_split.denoised_ptp_amplitudes[idx_first_assign], c='green', s=1, alpha = 0.25)
        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_second_assign], sorting_post_split.denoised_ptp_amplitudes[idx_second_assign], c='palegreen', s=1, alpha = 0.25)
        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_third_assign], sorting_post_split.denoised_ptp_amplitudes[idx_third_assign], c='goldenrod', s=1, alpha = 0.25)
        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_triaged], sorting_post_split.denoised_ptp_amplitudes[idx_triaged], c='red', s=1, alpha = 0.25, label="triaged")
        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_toosmall], sorting_post_split.denoised_ptp_amplitudes[idx_toosmall], c='violet', s=1, alpha = 0.25, label="template too small")
        ax_ptp_time.legend(fontsize=7, loc='upper right')

        ax_z_time.scatter(sorting_post_split.times_seconds[idx_hard_assign], sorting_post_split.point_source_localizations[idx_hard_assign, 2], c='blue', s=1, alpha = 0.25)
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_first_assign], sorting_post_split.point_source_localizations[idx_first_assign, 2], c='green', s=1, alpha = 0.25, label="soft assigned 1")
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_second_assign], sorting_post_split.point_source_localizations[idx_second_assign, 2], c='palegreen', s=1, alpha = 0.25, label="soft assigned 2")
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_third_assign], sorting_post_split.point_source_localizations[idx_third_assign, 2], c='goldenrod', s=1, alpha = 0.25, label="soft assigned 3")
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_triaged], sorting_post_split.point_source_localizations[idx_triaged, 2], c='red', s=1, alpha = 0.25)
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_toosmall], sorting_post_split.point_source_localizations[idx_toosmall, 2], c='violet', s=1, alpha = 0.25)
        ax_z_time.legend(fontsize=7, loc='upper right')
        
        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_fr_time.set_ylabel("FR (N/s)")
        ax_fr_time.set_xlabel("Time (s)")
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")
        ax_z_time.set_ylim(zlim)

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()
        ax_fr_time.xaxis.set_label_position('top') 
        ax_fr_time.xaxis.tick_top()
        ax_fr_time.yaxis.tick_right()

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s):

            sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_range, template_config.subchunk_size_s,
                                                  divider_samples=matching_config.chunk_length_samples)
            n_sub_chunks = len(sub_chunk_time_range_s)

            idx_chunk_time = np.logical_and(
                sorting_post_split.times_seconds>=chunk_range[0], sorting_post_split.times_seconds<chunk_range[1]
            )
            times_seconds = times_seconds_list[j]
            n_spike_chunks = times_seconds.size
            channels = channels_list[j]
            tpca = tpca_list[j]
            depth_reg = depth_reg_list[j]
            localization_results = localization_results_list[j]
            a = a_list[j]
            collisioncleaned_tpca_features_all = collisioncleaned_tpca_features_all_list[j]

            temp_unit_tpca = tpca_templates_list[j*n_sub_chunks:j*n_sub_chunks+n_sub_chunks].mean(0)[unit]
            subtract_tpca_temp = True
            
            ax_ptp_time.axvline(chunk_range[0], c='red')
            ax_ptp_time.axvline(chunk_range[1], c='red')
            ax_z_time.axvline(chunk_range[0], c='red')
            ax_z_time.axvline(chunk_range[1], c='red')

            ax_pcs = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+1]) 
            ax_ISI = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+2]) 
            ax_ACG = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+2]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4)+1:2*int(j//4)+3, 4*int(j - 4*(j//4))+3]) 

            for k, subchunk_range in enumerate(sub_chunk_time_range_s):
                idxtemp = int(j*n_sub_chunks+k)
                ax_ptp_time.plot([subchunk_range[0], subchunk_range[1]], [tpca_templates_list[idxtemp][unit].ptp(0).max(), tpca_templates_list[idxtemp][unit].ptp(0).max()],
                        c = "orange", label = "template ptp", 
                )
            # ax_z_time.plot([chunk_range[0], chunk_range[1]],
            #               [template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit], template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit]], c = "orange", label = "template depth")

            idx_chunk = np.arange(cmp, cmp+n_spike_chunks)
            # np.flatnonzero(sorting_post_split.labels[cmp:cmp+n_spike_chunks]==unit)

            idx_hard_assign = np.flatnonzero(np.logical_and(hard_assigned[idx_chunk], 
                labels_soft_assignment[idx_chunk, 0]==unit))
            idx_first_assign = np.flatnonzero(np.logical_and(~hard_assigned[idx_chunk], 
                labels_soft_assignment[idx_chunk, 0]==unit))
            idx_second_assign = np.flatnonzero(np.logical_and(~hard_assigned[idx_chunk], 
                labels_soft_assignment[idx_chunk, 1]==unit))
            idx_third_assign = np.flatnonzero(np.logical_and(~hard_assigned[idx_chunk], 
                labels_soft_assignment[idx_chunk, 2]==unit))
            idx_triaged = np.flatnonzero(np.logical_and(
                np.all(labels_soft_assignment[idx_chunk] == -1, axis=1), sorting_post_split.labels[idx_chunk]==unit))
            idx_toosmall = np.flatnonzero(
                labels_soft_assignment[idx_chunk, 3] == unit)

            all_indices_unit = np.concatenate([
                idx_hard_assign, idx_first_assign, idx_second_assign, idx_third_assign, idx_triaged, idx_toosmall
            ])
            all_indices_unit.sort()
            
            fr_unit = len(all_indices_unit)/(chunk_range[1]-chunk_range[0])
            ax_fr_time.plot([chunk_range[0], chunk_range[1]], [fr_unit, fr_unit], c = 'k')
            cmp+=n_spike_chunks
            
            if len(idx_hard_assign)>150:
                idx_subsample = np.random.choice(len(idx_hard_assign), 150, replace=False)
                idx_hard_assign = idx_hard_assign[idx_subsample]
            if len(idx_first_assign)>150:
                idx_subsample = np.random.choice(len(idx_first_assign), 150, replace=False)
                idx_first_assign = idx_first_assign[idx_subsample]
            if len(idx_second_assign)>150:
                idx_subsample = np.random.choice(len(idx_second_assign), 150, replace=False)
                idx_second_assign = idx_second_assign[idx_subsample]
            if len(idx_third_assign)>150:
                idx_subsample = np.random.choice(len(idx_third_assign), 150, replace=False)
                idx_third_assign = idx_third_assign[idx_subsample]
            if len(idx_triaged)>150:
                idx_subsample = np.random.choice(len(idx_triaged), 150, replace=False)
                idx_triaged = idx_triaged[idx_subsample]
            if len(idx_toosmall)>150:
                idx_subsample = np.random.choice(len(idx_toosmall), 150, replace=False)
                idx_toosmall = idx_toosmall[idx_subsample]

            subsampled_indices_unit = np.concatenate([
                idx_hard_assign, idx_first_assign, idx_second_assign, idx_third_assign, idx_triaged, idx_toosmall
            ])
            color_array = np.concatenate([
                np.full(len(idx_hard_assign), "blue"), np.full(len(idx_first_assign), "green"), np.full(len(idx_second_assign), "palegreen"),
                np.full(len(idx_third_assign), "goldenrod"), np.full(len(idx_triaged), "red"), np.full(len(idx_toosmall), "violet"),
            ])
            sorted_indices = subsampled_indices_unit.argsort()
            color_array = color_array[sorted_indices]
            subsampled_indices_unit = subsampled_indices_unit[sorted_indices]

            ax_loc.scatter(localization_results[subsampled_indices_unit, 0], depth_reg[subsampled_indices_unit], s=1, c = color_array)
            ax_loc.set_title("Localization", fontsize=7)
            ax_loc.set_xlabel("x (um)", fontsize=7)
            ax_loc.set_ylabel("reg z (um)", fontsize=7)

            dt_ms = np.diff(times_seconds[all_indices_unit]) * 1000
            lags, acg = correlogram(times_seconds[all_indices_unit]*30_000, max_lag=max_lag)
            
            bin_edges = np.arange(
                0,
                max_ms + bin_ms,
                bin_ms,
            )
            ax_ISI.hist(dt_ms, bin_edges, color="k")
            ax_ISI.set_title("isi (ms)", fontsize=7)
            ax_ISI.set_ylabel(f"count (out of {dt_ms.size} isis)", fontsize = 7, labelpad=0)

            bar(ax_ACG, lags, acg, fill=True, color="k")
            ax_ACG.set_title("lag (samples)", fontsize=7)
            ax_ACG.set_ylabel("acg", fontsize=7, labelpad=0)

            # ax_ISI.yaxis.set_label_position("right")
            ax_ISI.yaxis.tick_right()
            # ax_ACG.yaxis.set_label_position("right")
            ax_ACG.yaxis.tick_right()
            
            ax_heatmap.imshow(temp_unit_tpca.ptp(0).reshape(-1, 8))
        
            n_pitches_shift = get_spike_pitch_shifts(localization_results[subsampled_indices_unit, 2], geom, times_s = times_seconds[subsampled_indices_unit], motion_est=me)

            max_chan_registered_geom = temp_unit_tpca.ptp(0).argmax()
            
                # print(f"MAX CHAN {max_chan_registered_geom}")

            max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
            max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
            chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)

            temp_chunk = temp_unit_tpca[15:75][:, chans_to_plot_registered_geom]
            
            med_ptp = 1.5*temp_chunk.ptp(0).max()
            # med_ptp = 1.5*np.median(a[idx_chunk])
            
            # do all this before chunk / PCs...
            ccwfs_targetchans = get_waveforms_on_static_channels(
                collisioncleaned_tpca_features_all[subsampled_indices_unit],
                geom,
                main_channels=channels[subsampled_indices_unit],
                channel_index=channel_index,
                target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                n_pitches_shift=n_pitches_shift,
                registered_geom=registered_geom,
            )
            no_nans = np.flatnonzero(np.isfinite(ccwfs_targetchans[:, 0, :]).all(axis=1))

            if len(no_nans)>2:
                pcs = PCA(2).fit_transform(ccwfs_targetchans.reshape(ccwfs_targetchans.shape[0], -1)[no_nans])
                # idx_chunk = idx_chunk[no_nans]

                ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = color_array[no_nans])
                ax_pcs.set_title("PCs", fontsize=7)
            
                # subsampled wfs to plot
                waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
                waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
            
                for k in range(5):
                    for i in range(waveforms_target_chan.shape[0]):
                        ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = color_array[i], alpha = 0.05)
                        if not subtract_tpca_temp: 
                            ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = color_array[i])
                        else:
                            ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_unit_tpca[15:75, chans_to_plot_registered_geom[k*2:k*2+2]].T.flatten() + k*med_ptp, alpha = 0.05, c = color_array[i])
                ax_wfs_temp_subtracted.set_ylabel(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7, labelpad=0)
                ax_wfs.set_ylabel("Wfs", fontsize=7, labelpad = 0)

                ax_wfs_temp_subtracted.set_xticks([])
                ax_wfs_temp_subtracted.set_yticks([])
                ax_wfs.set_xticks([])
                ax_wfs.set_yticks([])
                    
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        plt.savefig(output_directory / f"unit_{unit}_overmerge")
        plt.close()
        


def check_oversplits(
    sorting,
    chunk_time_ranges_s,
    data_dir_temp_data,
    output_directory,
    times_seconds,
    localization_results,
    depth_reg,
    collisioncleaned_tpca_features_all,
    geom,
    me,
    channels,
    channel_index,
    tpca,
    name_end_chunk_tempdata = "post_merge_GC",
    template_npz_filename = "template_data.npz",
    n_neigh = 3,
):

    print("computing distances")
    units_all = []
    dists_all_list = []
    for j, chunk_range in enumerate(chunk_time_ranges_s):
        model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
        model_dir_chunk = data_dir_temp_data / model_subdir_chunk
        template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

        units, dists, shifts, template_snrs = calculate_merge_distances(
            template_data_chunk,
            superres_linkage=np.max,
            sym_function=np.maximum,
            min_channel_amplitude=0.0,
            # amplitude_scaling_variance=0,
            # TODO check this param makes sense - 0.2 should avoid that weird spikes get matched together. interpreted as channel overlap weighted by ptp? 
            min_spatial_cosine=0.5, #
            n_jobs=0,
            show_progress=False,
        )
        units_all.append(units)
        dists_all_list.append(dists)

    units_all_ids = np.unique(np.hstack(units_all))
    dists_all = []
    for j, chunk_range in enumerate(chunk_time_ranges_s):
        dist_full = np.full((units_all_ids.max()+1, units_all_ids.max()+1), np.nan)
        n_units = len(units_all[j])
        dist_full[np.repeat(units_all[j], n_units), np.repeat(units_all[j], n_units).reshape((n_units, n_units)).T.flatten()] = dists_all_list[j].flatten()
        dists_all.append(dist_full)

    # dists_all = np.array(dists_all)
    dists_min_across_chunks = np.nanmax(dists_all, axis=0)

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 3* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    units_all = np.unique(sorting.labels)
    units_all = units_all[units_all>-1]
    
    for unit in tqdm(units_all): 
        neighbors_ordered = dists_min_across_chunks[unit].argsort()[:n_neigh+1]
        
        fig = plt.figure(figsize=(25, 20))
        
        gs = fig.add_gridspec(n_rows, n_cols)
    
        for j, chunk_range in enumerate(chunk_time_ranges_s):
            
            ax_pcs = fig.add_subplot(gs[2*int(j//4), 3*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4), 3*int(j - 4*(j//4))+1]) 
            ax_temp_over_time = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))]) 
            ax_temp_distance = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))+1]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4):2*int(j//4)+2, 3*int(j - 4*(j//4))+2]) 

            dist_chunk = dists_all[j]
            dist_chunk = dist_chunk[neighbors_ordered][:, neighbors_ordered]
            ax_temp_distance.imshow(dist_chunk)
            ax_temp_distance.set_xticks(np.arange(len(neighbors_ordered)), labels=neighbors_ordered)
            ax_temp_distance.set_yticks(np.arange(len(neighbors_ordered)), labels=neighbors_ordered)
            for n1 in range(n_neigh+1):
                for n2 in range(n_neigh+1):
                    if not np.isnan(dist_chunk[n1, n2]) and dist_chunk[n1, n2]<10: #avoid infinity cases
                        text = ax_temp_distance.text(n2, n1, int(dist_chunk[n1, n2]*100)/100, fontsize=7,
                                       ha="center", va="center", color="w")
                    else:
                        text = ax_temp_distance.text(n2, n1, "NAN", fontsize=7,
                                       ha="center", va="center", color="w")
        
            model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
            model_dir_chunk = data_dir_temp_data / model_subdir_chunk
            template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

            if np.any(template_data_chunk.unit_ids == unit):
                ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, 8))
        
                max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
                max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 10)
                max_chan_registered_geom = max(max_chan_registered_geom, 10)
                chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)
            
                temp_chunk = template_data_chunk.templates[np.isin(template_data_chunk.unit_ids , neighbors_ordered)][:, 15:75][:, :, chans_to_plot_registered_geom]
                med_ptp = temp_chunk.ptp(1).max()
        
                for k in range(5):
                    for n, neigh in enumerate(neighbors_ordered[np.isin(neighbors_ordered, template_data_chunk.unit_ids)]):
                        ax_temp_over_time.plot(np.arange(120), temp_chunk[n, :, k*2:k*2+2].T.flatten() + k*med_ptp, c = get_ccolor(n))
                ax_temp_over_time.set_title(f"Templates (max chan {max_chan_registered_geom})", fontsize=7)
            
                idx_neigh_chunks = np.flatnonzero(np.logical_and(
                    np.isin(sorting.labels, neighbors_ordered),
                    np.logical_and(times_seconds>=chunk_range[0],
                                   times_seconds<chunk_range[1]
                    )
                ))

            for n, neigh in enumerate(neighbors_ordered):
                idx_unit = sorting.labels[idx_neigh_chunks]==neigh 
                ax_loc.scatter(localization_results[idx_neigh_chunks, 0][idx_unit], depth_reg[idx_neigh_chunks][idx_unit], s=1, c = get_ccolor(n), label = f"Unit {neigh}")
            ax_loc.set_title("Localization", fontsize=7)
            ax_loc.set_xlabel("x (um)", fontsize=7)
            ax_loc.set_ylabel("reg z (um)", fontsize=7)
            ax_loc.legend(loc = "upper left", fontsize = 5)

            
            n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_neigh_chunks, 2], geom, times_s = times_seconds[idx_neigh_chunks], motion_est=me)
            ccwfs_targetchans = get_waveforms_on_static_channels(
                collisioncleaned_tpca_features_all[idx_neigh_chunks],
                geom,
                main_channels=channels[idx_neigh_chunks],
                channel_index=channel_index,
                target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                n_pitches_shift=n_pitches_shift,
                registered_geom=registered_geom,
            )
            no_nans = np.flatnonzero(np.isfinite(ccwfs_targetchans[:, 0, :]).all(axis=1))
            
            if len(no_nans)>2:
                pcs = PCA(2).fit_transform(ccwfs_targetchans.reshape(ccwfs_targetchans.shape[0], -1)[no_nans])
                idx_neigh_chunks = idx_neigh_chunks[no_nans]
            
                for n, neigh in enumerate(neighbors_ordered):
                    idx_unit = sorting.labels[idx_neigh_chunks]==neigh 
                    ax_pcs.scatter(pcs[idx_unit, 0], pcs[idx_unit, 1], s=1, c = get_ccolor(n))
                ax_pcs.set_title("PCs", fontsize=7)
            
                # # subsampled wfs to plot
                # all_wfs_to_plot = []
                # for n, neigh in enumerate(neighbors_ordered):
                #     if (sorting.labels[idx_neigh_chunks]==neigh).sum()<150:
                #         all_wfs_to_plot.append(np.flatnonzero(sorting.labels[idx_neigh_chunks]==neigh))
                #     else:
                #         all_wfs_to_plot.append(np.random.choice(np.flatnonzero(sorting.labels[idx_neigh_chunks]==neigh), 150, replace=False))
                # all_wfs_to_plot = np.hstack(all_wfs_to_plot)
                # all_wfs_to_plot.sort()

                # waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans[all_wfs_to_plot].transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
                # waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
            
                # for k in range(5):
                #     for n, neigh in enumerate(neighbors_ordered):
                #         idx_neigh = np.flatnonzero(sorting.labels[idx_neigh_chunks][all_wfs_to_plot]==neigh)
                #         for i in idx_neigh:
                #             ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = get_ccolor(n), alpha = 0.05)
                # ax_wfs.set_title("Wfs", fontsize=7)
        plt.suptitle(f"Unit {unit}", y=0.9)
        
        # PCs on max chan 
        # WFS on same chans 
        plt.savefig(output_directory / f"unit_{unit}_oversplit")
        plt.close()


# def check_oversplits_post_deconv(
#     sorting_list,
#     chunk_time_ranges_s,
#     data_dir_temp_data,
#     output_directory,
#     times_seconds_list,
#     localization_results_list,
#     depth_reg_list,
#     collisioncleaned_tpca_features_all_list,
#     geom,
#     me,
#     channels_list,
#     channel_index_list,
#     tpca_list,
#     name_end_chunk_tempdata = "post_merge_GC",
#     template_npz_filename = "template_data.npz",
#     n_neigh = 3,
# ):

#     print("computing distances")
#     units_all = []
#     dists_all_list = []
#     for j, chunk_range in enumerate(chunk_time_ranges_s):
#         model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
#         model_dir_chunk = data_dir_temp_data / model_subdir_chunk
#         template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

#         units, dists, shifts, template_snrs = calculate_merge_distances(
#             template_data_chunk,
#             superres_linkage=np.max,
#             sym_function=np.maximum,
#             min_channel_amplitude=0.0,
#             # amplitude_scaling_variance=0,
#             # TODO check this param makes sense - 0.2 should avoid that weird spikes get matched together. interpreted as channel overlap weighted by ptp? 
#             min_spatial_cosine=0.5, #
#             n_jobs=0,
#             show_progress=False,
#         )
#         units_all.append(units)
#         dists_all_list.append(dists)

#     units_all_ids = np.unique(np.hstack(units_all))
#     dists_all = []
#     for j, chunk_range in enumerate(chunk_time_ranges_s):
#         dist_full = np.full((units_all_ids.max()+1, units_all_ids.max()+1), np.nan)
#         n_units = len(units_all[j])
#         dist_full[np.repeat(units_all[j], n_units), np.repeat(units_all[j], n_units).reshape((n_units, n_units)).T.flatten()] = dists_all_list[j].flatten()
#         dists_all.append(dist_full)

#     # dists_all = np.array(dists_all)
#     dists_min_across_chunks = np.nanmax(dists_all, axis=0)

#     registered_geom = registered_geometry(geom, me)
#     n_chans_reg_geom = len(registered_geom)

#     n_chunks = len(chunk_time_ranges_s)
#     n_cols = 3* (n_chunks // 4 + 1)
#     if n_chunks % 4 == 0:
#         n_rows = int(2*(n_chunks // 4))
#     else:
#         n_rows = int(2*(n_chunks // 4 + 1))

#     units_all=[]
#     for sorting in sorting_list:
#         units_all.append(np.unique(sorting.labels))
#     units_all = np.unique(np.hstack(units_all))
#     units_all = units_all[units_all>-1]
    
#     for unit in tqdm(units_all): 
#         neighbors_ordered = dists_min_across_chunks[unit].argsort()[:n_neigh+1]
        
#         fig = plt.figure(figsize=(25, 20))
        
#         gs = fig.add_gridspec(n_rows, n_cols)
    
#         for j, chunk_range in enumerate(chunk_time_ranges_s):

#             sorting = sorting_list[j]
#             times_seconds = times_seconds_list[j]
#             channels = channels_list[j]
#             channel_index = channel_index_list[j]
#             tpca = tpca_list[j]
#             depth_reg = depth_reg_list[j]
#             localization_results = localization_results_list[j]
#             collisioncleaned_tpca_features_all = collisioncleaned_tpca_features_all_list[j]
            
#             ax_pcs = fig.add_subplot(gs[2*int(j//4), 3*int(j - 4*(j//4))]) 
#             ax_loc = fig.add_subplot(gs[2*int(j//4), 3*int(j - 4*(j//4))+1]) 
#             ax_temp_over_time = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))]) 
#             ax_temp_distance = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))+1]) 
#             ax_heatmap = fig.add_subplot(gs[2*int(j//4):2*int(j//4)+2, 3*int(j - 4*(j//4))+2]) 

#             dist_chunk = dists_all[j]
#             dist_chunk = dist_chunk[neighbors_ordered][:, neighbors_ordered]
#             ax_temp_distance.imshow(dist_chunk)
#             ax_temp_distance.set_xticks(np.arange(len(neighbors_ordered)), labels=neighbors_ordered)
#             ax_temp_distance.set_yticks(np.arange(len(neighbors_ordered)), labels=neighbors_ordered)
#             for n1 in range(n_neigh+1):
#                 for n2 in range(n_neigh+1):
#                     if not np.isnan(dist_chunk[n1, n2]) and dist_chunk[n1, n2]<10: #avoid infinity cases
#                         text = ax_temp_distance.text(n2, n1, int(dist_chunk[n1, n2]*100)/100, fontsize=7,
#                                        ha="center", va="center", color="w")
#                     else:
#                         text = ax_temp_distance.text(n2, n1, "NAN", fontsize=7,
#                                        ha="center", va="center", color="w")
        
#             model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
#             model_dir_chunk = data_dir_temp_data / model_subdir_chunk
#             template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

#             if np.any(template_data_chunk.unit_ids == unit):
#                 ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, 8))
        
#                 max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
#                 max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 10)
#                 max_chan_registered_geom = max(max_chan_registered_geom, 10)
#                 chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)
            
#                 temp_chunk = template_data_chunk.templates[np.isin(template_data_chunk.unit_ids , neighbors_ordered)][:, 15:75][:, :, chans_to_plot_registered_geom]
#                 med_ptp = temp_chunk.ptp(1).max()
        
#                 for k in range(5):
#                     for n, neigh in enumerate(neighbors_ordered[np.isin(neighbors_ordered, template_data_chunk.unit_ids)]):
#                         ax_temp_over_time.plot(np.arange(120), temp_chunk[n, :, k*2:k*2+2].T.flatten() + k*med_ptp, c = get_ccolor(n))
#                 ax_temp_over_time.set_title(f"Templates (max chan {max_chan_registered_geom})", fontsize=7)
            
#                 idx_neigh_chunks = np.flatnonzero(np.logical_and(
#                     np.isin(sorting.labels, neighbors_ordered),
#                     np.logical_and(times_seconds>=chunk_range[0],
#                                    times_seconds<chunk_range[1]
#                     )
#                 ))

#             for n, neigh in enumerate(neighbors_ordered):
#                 idx_unit = sorting.labels[idx_neigh_chunks]==neigh 
#                 ax_loc.scatter(localization_results[idx_neigh_chunks, 0][idx_unit], depth_reg[idx_neigh_chunks][idx_unit], s=1, c = get_ccolor(n), label = f"Unit {neigh}")
#             ax_loc.set_title("Localization", fontsize=7)
#             ax_loc.set_xlabel("x (um)", fontsize=7)
#             ax_loc.set_ylabel("reg z (um)", fontsize=7)
#             ax_loc.legend(loc = "upper left", fontsize = 5)

            
#             n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_neigh_chunks, 2], geom, times_s = times_seconds[idx_neigh_chunks], motion_est=me)
#             ccwfs_targetchans = get_waveforms_on_static_channels(
#                 collisioncleaned_tpca_features_all[idx_neigh_chunks],
#                 geom,
#                 main_channels=channels[idx_neigh_chunks],
#                 channel_index=channel_index,
#                 target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
#                 n_pitches_shift=n_pitches_shift,
#                 registered_geom=registered_geom,
#             )
#             no_nans = np.flatnonzero(np.isfinite(ccwfs_targetchans[:, 0, :]).all(axis=1))
            
#             if len(no_nans)>2:
#                 pcs = PCA(2).fit_transform(ccwfs_targetchans.reshape(ccwfs_targetchans.shape[0], -1)[no_nans])
#                 idx_neigh_chunks = idx_neigh_chunks[no_nans]
            
#                 for n, neigh in enumerate(neighbors_ordered):
#                     idx_unit = sorting.labels[idx_neigh_chunks]==neigh 
#                     ax_pcs.scatter(pcs[idx_unit, 0], pcs[idx_unit, 1], s=1, c = get_ccolor(n))
#                 ax_pcs.set_title("PCs", fontsize=7)
            
#                 # # subsampled wfs to plot
#                 # all_wfs_to_plot = []
#                 # for n, neigh in enumerate(neighbors_ordered):
#                 #     if (sorting.labels[idx_neigh_chunks]==neigh).sum()<150:
#                 #         all_wfs_to_plot.append(np.flatnonzero(sorting.labels[idx_neigh_chunks]==neigh))
#                 #     else:
#                 #         all_wfs_to_plot.append(np.random.choice(np.flatnonzero(sorting.labels[idx_neigh_chunks]==neigh), 150, replace=False))
#                 # all_wfs_to_plot = np.hstack(all_wfs_to_plot)
#                 # all_wfs_to_plot.sort()

#                 # waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans[all_wfs_to_plot].transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
#                 # waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
            
#                 # for k in range(5):
#                 #     for n, neigh in enumerate(neighbors_ordered):
#                 #         idx_neigh = np.flatnonzero(sorting.labels[idx_neigh_chunks][all_wfs_to_plot]==neigh)
#                 #         for i in idx_neigh:
#                 #             ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = get_ccolor(n), alpha = 0.05)
#                 # ax_wfs.set_title("Wfs", fontsize=7)
#         plt.suptitle(f"Unit {unit}", y=0.9)
        
#         # PCs on max chan 
#         # WFS on same chans 
#         plt.savefig(output_directory / f"unit_{unit}_oversplit")
#         plt.close()

def check_split_step_post_deconv(
    sorting_post_split,
    sorting_list_pre_deconv,
    chunk_time_ranges_s,
    data_dir_temp_data,
    output_directory,
    times_seconds_list,
    localization_results_list,
    a_list,
    depth_reg_list,
    collisioncleaned_tpca_features_all_list, 
    geom,
    me,
    channels_list,
    channel_index_list,
    tpca_list, # to get from h5 every time / tpca list? 
    units_all=None,
    name_end_chunk_tempdata = "post_merge_GC",
    template_npz_filename = "template_data.npz",
    n_neigh = 3,
    zlim = (-100, 382),
):

    """
    here, everything should be a list containing infor about each chunk!! 
    """

    colors_split = ["blue", "red", "green", "yellow", "cyan", "pink", "orange", 
                    "magenta", "black", "lime", "olive", "goldenrod", "grey"]

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 3* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    if units_all is None:
        units_all=[]
        for sorting in sorting_list_pre_deconv:
            units_all.append(np.unique(sorting.labels))
        units_all = np.unique(np.hstack(units_all))
        units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)
    
    for unit in tqdm(units_all): 
        
        fig = plt.figure(figsize=(25, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :n_cols//2]) 
        ax_z_time = fig.add_subplot(gs[0, n_cols//2:]) 

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s):
            n_spike_chunks = sorting_list_pre_deconv[j].labels.size
            idx_unit = np.flatnonzero(sorting_list_pre_deconv[j].labels==unit)
            unit_labels = sorting_post_split.labels[cmp:cmp+n_spike_chunks][idx_unit]
            # _, unit_labels[unit_labels>-1] = np.unique(unit_labels[unit_labels>-1], return_inverse=True)
            cmp+=n_spike_chunks
            # if unit_labels.max()>len(colors_split):
            color_array = ccolors[unit_labels.astype('int')%31]
            color_array[unit_labels == -1] = "grey"
            # else:
            #     color_array = np.take(colors_split, unit_labels)
            ax_ptp_time.scatter(times_seconds_list[j][idx_unit], a_list[j][idx_unit], s=1, c = color_array)
            ax_z_time.scatter(times_seconds_list[j][idx_unit], depth_reg_list[j][idx_unit], s=1, c = color_array)
        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")
        ax_z_time.set_ylim(zlim)

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s):

            n_spike_chunks = sorting_list_pre_deconv[j].labels.size
            sorting = sorting_list_pre_deconv[j]
            times_seconds = times_seconds_list[j]
            channels = channels_list[j]
            channel_index = channel_index_list[j]
            tpca = tpca_list[j]
            depth_reg = depth_reg_list[j]
            localization_results = localization_results_list[j]
            a = a_list[j]
            collisioncleaned_tpca_features_all = collisioncleaned_tpca_features_all_list[j]
            
            ax_ptp_time.axvline(chunk_range[0], c='red')
            ax_ptp_time.axvline(chunk_range[1], c='red')
            ax_z_time.axvline(chunk_range[0], c='red')
            ax_z_time.axvline(chunk_range[1], c='red')

            ax_pcs = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//4)+2, 3*int(j - 4*(j//4))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//4)+2, 3*int(j - 4*(j//4))+1]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4)+1:2*int(j//4)+3, 3*int(j - 4*(j//4))+2]) 
        
            model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
            model_dir_chunk = data_dir_temp_data / model_subdir_chunk
            template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

            # idx_chunk = np.flatnonzero(np.logical_and(
            #     sorting.labels==unit,
            #     np.logical_and(times_seconds>=chunk_range[0],
            #                    times_seconds<chunk_range[1]
            #     )
            # ))

            idx_chunk = np.flatnonzero(sorting.labels==unit)


            unit_label_split = sorting_post_split.labels[cmp:cmp+n_spike_chunks][sorting.labels==unit]
            # _, unit_label_split[unit_label_split>-1] = np.unique(unit_label_split[unit_label_split>-1], return_inverse=True)
            cmp+=n_spike_chunks
            
            if len(idx_chunk)>150:
                idx_subsample = np.random.choice(len(idx_chunk), 150, replace=False)
                idx_chunk = idx_chunk[idx_subsample]
                unit_label_split = unit_label_split[idx_subsample]
            
            ax_loc.scatter(localization_results[idx_chunk, 0], depth_reg[idx_chunk], s=1, c = "blue")
            ax_loc.set_title("Localization", fontsize=7)
            ax_loc.set_xlabel("x (um)", fontsize=7)
            ax_loc.set_ylabel("reg z (um)", fontsize=7)

            if np.any(template_data_chunk.unit_ids == unit):
                ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, 8))
            
                n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_chunk, 2], geom, times_s = times_seconds[idx_chunk], motion_est=me)
    
                max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
                max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
                max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
                chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)
    
                temp_chunk = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0][15:75][:, chans_to_plot_registered_geom]
                med_ptp = 1.5*temp_chunk.ptp(0).max()
                
                # do all this before chunk / PCs...
                ccwfs_targetchans = get_waveforms_on_static_channels(
                    collisioncleaned_tpca_features_all[idx_chunk],
                    geom,
                    main_channels=channels[idx_chunk],
                    channel_index=channel_index,
                    target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                    n_pitches_shift=n_pitches_shift,
                    registered_geom=registered_geom,
                )
                no_nans = np.flatnonzero(np.isfinite(ccwfs_targetchans[:, 0, :]).all(axis=1))

                # if unit_label_split.max()>len(colors_split):
                #     color_array_no_nan = ccolors[unit_label_split.astype('int')]
                # else:
                #     color_array_no_nan = np.take(colors_split, unit_label_split)
                color_array_no_nan = ccolors[unit_label_split.astype('int')%31]
                color_array_no_nan[unit_label_split == -1] = "grey"


                if len(no_nans)>2:
                    pcs = PCA(2).fit_transform(ccwfs_targetchans.reshape(ccwfs_targetchans.shape[0], -1)[no_nans])
                    # idx_chunk = idx_chunk[no_nans]

                    ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = color_array_no_nan[no_nans])
                    ax_pcs.set_title("PCs", fontsize=7)
                
                    # subsampled wfs to plot
                    waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
                    waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
                
                    for k in range(5):
                        for i in range(waveforms_target_chan.shape[0]):
                            ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = color_array_no_nan[i], alpha = 0.05)
                            ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = color_array_no_nan[i])
                    ax_wfs_temp_subtracted.set_title(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7)
                    ax_wfs.set_title("Wfs", fontsize=7)
    
                    ax_wfs_temp_subtracted.set_xticks([])
                    ax_wfs_temp_subtracted.set_yticks([])
                    ax_wfs.set_xticks([])
                    ax_wfs.set_yticks([])
                    
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        plt.savefig(output_directory / f"unit_{unit}_overmerge")
        plt.close()

def check_overmerges_NP1(
    recording,
    sorting,
    chunk_time_ranges_s,
    data_dir_temp_data,
    output_directory,
    depth_reg_all,
    subh5, 
    geom,
    me,
    channel_index,
    tpca,
    slice_s,
    units_all = None,
    overlap_templates=True,
    tpca_templates_list=None,
    template_npz_filename = "template_data.npz",
    tpca_features_dataset_name="collisioncleaned_tpca_features",
    n_neigh = 3,
    zlim = (-100, 382),
    bin_ms=0.1, 
    max_ms=5,
    max_lag=50,
    time_smoothed=False,
    raw=False,
):

    """
    here, everything should be a list containing info about each chunk!! 
    """

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 4* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    if units_all is None:
        units_all = np.unique(sorting.labels)
        units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)
    
    for unit in tqdm(units_all): 
        
        fig = plt.figure(figsize=(40, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :(n_cols-2)//2]) 
        ax_z_time = fig.add_subplot(gs[0, (n_cols-2)//2:-2]) 
        ax_fr_time = fig.add_subplot(gs[0, -2:]) 

        idx_unit = np.flatnonzero(sorting.labels==unit)
        # if unit_labels.max()>len(colors_split):

        ax_ptp_time.scatter(sorting.times_seconds[idx_unit], sorting.denoised_ptp_amplitudes[idx_unit], s=1, c = "blue")
        ax_z_time.scatter(sorting.times_seconds[idx_unit], sorting.point_source_localizations[idx_unit, 2], s=1, c = "blue")

        med_depth_reg = np.median(sorting.point_source_localizations[idx_unit, 2])
        std_depth_reg = np.median(np.abs(sorting.point_source_localizations[idx_unit, 2] - med_depth_reg))/0.675
        ax_z_time.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))

        ax_z_time.set_xlim(slice_s)
        ax_ptp_time.set_xlim(slice_s)

        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_fr_time.set_ylabel("FR (N/s)")
        ax_fr_time.set_xlabel("Time (s)")
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")
        # ax_z_time.set_ylim(zlim)

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()
        ax_fr_time.xaxis.set_label_position('top') 
        ax_fr_time.xaxis.tick_top()
        ax_fr_time.yaxis.tick_right()

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s):

            idx_chunk = np.flatnonzero(
                np.logical_and(sorting.times_seconds>=chunk_range[0], sorting.times_seconds<chunk_range[1])
            )
            idx_unit_chunk= idx_chunk[sorting.labels[idx_chunk]==unit]

            ax_ptp_time.axvline(chunk_range[0], c='red')
            ax_ptp_time.axvline(chunk_range[1], c='red')
            ax_z_time.axvline(chunk_range[0], c='red')
            ax_z_time.axvline(chunk_range[1], c='red')

            ax_pcs = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+1]) 
            ax_ISI = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+2]) 
            ax_ACG = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+2]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4)+1:2*int(j//4)+3, 4*int(j - 4*(j//4))+3]) 
            
            if len(idx_unit_chunk):
                times_seconds = sorting.times_seconds[idx_unit_chunk]
                times_samples = sorting.times_samples[idx_unit_chunk]
                n_spike_chunks = times_seconds.size
                channels = sorting.channels[idx_unit_chunk]
                depth_reg = depth_reg_all[idx_unit_chunk]
                localization_results = sorting.point_source_localizations[idx_unit_chunk]
                a = sorting.denoised_ptp_amplitudes[idx_unit_chunk]
        
                subtract_tpca_temp = False
                if tpca_templates_list is not None:
                    if type(tpca_templates_list[j])==TemplateData:
                        if np.any(tpca_templates_list[j].unit_ids==unit):
                            temp_unit_tpca = tpca_templates_list[j].templates[tpca_templates_list[j].unit_ids==unit][0].transpose(1, 0)
                            temp_unit_tpca = tpca.inverse_transform(temp_unit_tpca).transpose(1, 0)
                            subtract_tpca_temp = True
                    else:
                        temp_unit_tpca = tpca_templates_list[j][unit]
                        subtract_tpca_temp = True
                            
                template_data_chunk = TemplateData.from_npz(data_dir_temp_data / f"chunk_{j}_{template_npz_filename}")
                template_data_chunk.templates[np.isnan(template_data_chunk.templates)] = 0
    
                ax_ptp_time.plot([chunk_range[0], chunk_range[1]], [template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).max(), template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).max()],
                    c = "orange", label = "template ptp", 
                )
                ax_z_time.plot([chunk_range[0], chunk_range[1]],
                              [template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit], template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit]], c = "orange", label = "template depth")
    
                fr_unit = len(idx_unit_chunk)/(chunk_range[1]-chunk_range[0])
                ax_fr_time.plot([chunk_range[0], chunk_range[1]], [fr_unit, fr_unit], c = 'k')
                cmp+=n_spike_chunks
    
                ax_loc.scatter(localization_results[:, 0], depth_reg, s=1, c = "blue")
                med_depth_reg = np.median(depth_reg)
                std_depth_reg = np.median(np.abs(depth_reg - med_depth_reg))/0.675
                std_depth_reg = max(std_depth_reg, 1)
                
                med_x = np.median(localization_results[:, 0])
                std_x = np.median(np.abs(localization_results[:, 0] - med_x))/0.675
                std_x = max(std_x, 1)
                ax_loc.set_xlim((med_x-5*std_x, med_x+5*std_x))
                ax_loc.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))
                ax_loc.set_title("Localization", fontsize=7)
                ax_loc.set_xlabel("x (um)", fontsize=7)
                ax_loc.set_ylabel("reg z (um)", fontsize=7)
    
                dt_ms = np.diff(times_seconds) * 1000
                lags, acg = correlogram(times_seconds*30_000, max_lag=max_lag)
                        
                bin_edges = np.arange(
                    0,
                    max_ms + bin_ms,
                    bin_ms,
                )
                ax_ISI.hist(dt_ms, bin_edges, color="k")
                ax_ISI.set_title("isi (ms)", fontsize=7)
                ax_ISI.set_ylabel(f"count (out of {dt_ms.size} isis)", fontsize = 7, labelpad=0)
    
                bar(ax_ACG, lags, acg, fill=True, color="k")
                ax_ACG.set_title("lag (samples)", fontsize=7)
                ax_ACG.set_ylabel("acg", fontsize=7, labelpad=0)
    
                # ax_ISI.yaxis.set_label_position("right")
                ax_ISI.yaxis.tick_right()
                # ax_ACG.yaxis.set_label_position("right")
                ax_ACG.yaxis.tick_right()
    
                if len(idx_unit_chunk)>150:
                    idx_subsample = np.random.choice(len(idx_unit_chunk), 150, replace=False)
                    idx_subsample.sort()
                    # idx_unit_chunk = idx_unit_chunk[idx_subsample]
                else:
                    idx_subsample = np.arange(len(idx_unit_chunk))
    
                if np.any(template_data_chunk.unit_ids == unit):
                    ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, 4), aspect="auto")
                
                    n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_subsample, 2], geom, times_s = times_seconds[idx_subsample], motion_est=me)


                    chans_idx, count_chans_idx = np.unique(channel_index[channels[idx_subsample]].flatten(), return_counts=True)
                    good_chans = chans_idx[count_chans_idx>25]
                    good_chans = good_chans[good_chans<384]
                    if len(good_chans):
                        # print(good_chans)
                        # chans_to_plot_registered_geom = good_chans[template_data_chunk.templates[template_data_chunk.unit_ids == unit][0, :, good_chans].ptp(0).argsort()[::-1][:10]]
                        # print("channels to plot on")
                        # print(chans_to_plot_registered_geom)
    
                        max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
                        # chans_idx, chans_count = np.unique(channels[idx_subsample]+8*n_pitches_shift, return_counts=True)
                        # max_chan_registered_geom = chans_idx[chans_count.argmax()]
                        max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 5)
                        max_chan_registered_geom = max(max_chan_registered_geom, 5)
                        # print(f"CHAN PITCH SHIFT WFS {chans_idx, chans_count}")

                        # print(np.unique(channels[idx_subsample], return_counts=True))
                        # print(np.unique(channels[idx_subsample]+8*n_pitches_shift, return_counts=True))
        
                        # max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
                        # max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
                        chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 5, max_chan_registered_geom+5, 1)
        
                        temp_chunk = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0][15:75][:, chans_to_plot_registered_geom]
                        
                        med_ptp = 1.5*temp_chunk.ptp(0).max()
    
                        if raw:
                            # print("Channels")
                            # print(channels[idx_subsample])
                            # print("n_pitches shifts")
                            # print(n_pitches_shift)
                            # print("sub channels")
                            # print(channels[idx_subsample] + 8*n_pitches_shift)
                            collisioncleaned_tpca_features = spikeio.read_waveforms_channel_index(
                                    recording,
                                    times_samples[idx_subsample],
                                    channel_index,
                                    channels[idx_subsample],
                            )
                        else:
                            with h5py.File(subh5, "r+") as h5:
                                collisioncleaned_tpca_features = h5[tpca_features_dataset_name][idx_unit_chunk[idx_subsample]]

                        # do all this before chunk / PCs...
                        waveforms_target_chan = get_waveforms_on_static_channels(
                            collisioncleaned_tpca_features,
                            geom,
                            main_channels=channels[idx_subsample],
                            channel_index=channel_index,
                            target_channels=chans_to_plot_registered_geom, #np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                            n_pitches_shift=n_pitches_shift,
                            registered_geom=registered_geom,
                        )                        

                        # print("channels no nan")
                        # print(np.where(~np.isnan(waveforms_target_chan[0, 0])))

                        chans_no_nans = np.isfinite(waveforms_target_chan[:, 0, :]).all(axis=0)
                        no_nans = np.flatnonzero(np.isfinite(waveforms_target_chan[:, 0, chans_no_nans]).all(axis=1))
                        if len(no_nans)>1 and chans_no_nans.sum()>1:
                            pcs = PCA(2).fit_transform(waveforms_target_chan[:, :, chans_no_nans].reshape(waveforms_target_chan.shape[0], -1)[no_nans])
        
                            ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = "blue")
                            ax_pcs.set_title("PCs", fontsize=7)
                        
                            # subsampled wfs to plot
                            if not raw:
                                waveforms_target_chan = tpca.inverse_transform(waveforms_target_chan[:, :, chans_no_nans].transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, len(chans_no_nans), 121).transpose(0, 2, 1)
                            waveforms_target_chan = waveforms_target_chan[:, 15:75, :]

                            # mean_waveforms = np.nanmean(waveforms_target_chan, axis = 0)
    
                            for k in range(5):
                                for i in range(waveforms_target_chan.shape[0]):
                                    ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, k*2:k*2+2].T.flatten() + k*med_ptp, c = "blue", alpha = 0.05)                                        
                                    if not subtract_tpca_temp: 
                                        ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, k*2:k*2+2].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = "blue")
                                    else:
                                        ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, k*2:k*2+2].T.flatten() - temp_unit_tpca[15:75, chans_to_plot_registered_geom].T.flatten() + k*med_ptp, alpha = 0.05, c = "blue")
                            if overlap_templates:
                                for k in range(5):
                                    if tpca_templates_list is not None:
                                        ax_wfs.plot(np.arange(120), temp_unit_tpca[15:75, chans_to_plot_registered_geom].T.flatten(), c = "orange", alpha = 1)                                        
                                    else:
                                        ax_wfs.plot(np.arange(120), temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, c = "orange", alpha = 1)                                        
                                        # ax_wfs.plot(np.arange(120), mean_waveforms[:, k*2:k*2+2].T.flatten() + k*med_ptp, c = "red", alpha = 1)                                        
                            ax_wfs.set_ylabel("Wfs", fontsize=7, labelpad = 0)
                            ax_wfs_temp_subtracted.set_ylabel(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7, labelpad=0)
                            ax_wfs_temp_subtracted.set_xticks([])
                            ax_wfs_temp_subtracted.set_yticks([])
                            ax_wfs.set_xticks([])
                            ax_wfs.set_yticks([])
                        
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        if raw:
            plt.savefig(output_directory / f"rawwfs_post_deconv_{unit}_overmerge")
        else:
            plt.savefig(output_directory / f"post_deconv_{unit}_overmerge")
        plt.close()
        



def check_overmerges(
    recording,
    sorting,
    chunk_time_ranges_s,
    data_dir_temp_data,
    output_directory,
    depth_reg_all,
    subh5, 
    geom,
    me,
    channel_index,
    tpca,
    slice_s,
    units_all = None,
    overlap_templates=True,
    tpca_templates_list=None,
    template_npz_filename = "template_data.npz",
    tpca_features_dataset_name="collisioncleaned_tpca_features",
    n_neigh = 3,
    zlim = (-100, 382),
    bin_ms=0.1, 
    max_ms=5,
    max_lag=50,
    time_smoothed=False,
    raw=False,
    n_col=4,
    n_col_templates=8,
    sorting_split=None,
):

    """
    here, everything should be a list containing info about each chunk!! 
    """

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = n_col * (n_chunks // n_col + 1)
    if n_chunks % n_col == 0:
        n_rows = int(2*(n_chunks // n_col))
    else:
        n_rows = int(2*(n_chunks // n_col + 1))

    if units_all is None:
        units_all = np.unique(sorting.labels)
        units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)
    
    for unit in tqdm(units_all): 
        
        fig = plt.figure(figsize=(40, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :(n_cols-2)//2]) 
        ax_z_time = fig.add_subplot(gs[0, (n_cols-2)//2:-2]) 
        ax_fr_time = fig.add_subplot(gs[0, -2:]) 

        idx_unit = np.flatnonzero(sorting.labels==unit)
        # if unit_labels.max()>len(colors_split):

        if sorting_split is not None:
            color_array = get_ccolor(sorting_split.labels[idx_unit])
        else:
            color_array='blue'
            
        ax_ptp_time.scatter(sorting.times_seconds[idx_unit], sorting.denoised_ptp_amplitudes[idx_unit], s=1, c = color_array)
        ax_z_time.scatter(sorting.times_seconds[idx_unit], sorting.point_source_localizations[idx_unit, 2], s=1, c = color_array)

        med_depth_reg = np.median(sorting.point_source_localizations[idx_unit, 2])
        std_depth_reg = np.median(np.abs(sorting.point_source_localizations[idx_unit, 2] - med_depth_reg))/0.675
        ax_z_time.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))

        ax_z_time.set_xlim(slice_s)
        ax_ptp_time.set_xlim(slice_s)

        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_fr_time.set_ylabel("FR (N/s)")
        ax_fr_time.set_xlabel("Time (s)")
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")
        # ax_z_time.set_ylim(zlim)

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()
        ax_fr_time.xaxis.set_label_position('top') 
        ax_fr_time.xaxis.tick_top()
        ax_fr_time.yaxis.tick_right()

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s):

            idx_chunk = np.flatnonzero(
                np.logical_and(sorting.times_seconds>=chunk_range[0], sorting.times_seconds<chunk_range[1])
            )
            idx_unit_chunk= idx_chunk[sorting.labels[idx_chunk]==unit]

            ax_ptp_time.axvline(chunk_range[0], c='red')
            ax_ptp_time.axvline(chunk_range[1], c='red')
            ax_z_time.axvline(chunk_range[0], c='red')
            ax_z_time.axvline(chunk_range[1], c='red')

            ax_pcs = fig.add_subplot(gs[2*int(j//n_col)+1, 4*int(j - n_col*(j//n_col))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//n_col)+1, 4*int(j - n_col*(j//n_col))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//n_col)+2, 4*int(j - n_col*(j//n_col))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//n_col)+2, 4*int(j - n_col*(j//n_col))+1]) 
            ax_ISI = fig.add_subplot(gs[2*int(j//n_col)+1, 4*int(j - n_col*(j//n_col))+2]) 
            ax_ACG = fig.add_subplot(gs[2*int(j//n_col)+2, 4*int(j - n_col*(j//n_col))+2]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//n_col)+1:2*int(j//n_col)+3, 4*int(j - n_col*(j//n_col))+3]) 
            
            if len(idx_unit_chunk):
                
                if sorting_split is not None:
                    color_array = get_ccolor(sorting_split.labels[idx_unit_chunk])
                else:
                    color_array='blue'

                times_seconds = sorting.times_seconds[idx_unit_chunk]
                times_samples = sorting.times_samples[idx_unit_chunk]
                n_spike_chunks = times_seconds.size
                channels = sorting.channels[idx_unit_chunk]
                depth_reg = depth_reg_all[idx_unit_chunk]
                localization_results = sorting.point_source_localizations[idx_unit_chunk]
                a = sorting.denoised_ptp_amplitudes[idx_unit_chunk]
        
                subtract_tpca_temp = False
                if tpca_templates_list is not None:
                    if type(tpca_templates_list[j])==TemplateData:
                        if np.any(tpca_templates_list[j].unit_ids==unit):
                            temp_unit_tpca = tpca_templates_list[j].templates[tpca_templates_list[j].unit_ids==unit][0].transpose(1, 0)
                            temp_unit_tpca = tpca.inverse_transform(temp_unit_tpca).transpose(1, 0)
                            subtract_tpca_temp = True
                    else:
                        temp_unit_tpca = tpca_templates_list[j][unit]
                        subtract_tpca_temp = True
                            
                # template_data_chunk = TemplateData.from_npz(data_dir_temp_data / f"chunk_{j}_{template_npz_filename}")
                template_data_chunk = TemplateData.from_npz(data_dir_temp_data / f"{template_npz_filename}")
                template_data_chunk.templates[np.isnan(template_data_chunk.templates)] = 0
    
                ax_ptp_time.plot([chunk_range[0], chunk_range[1]], [template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).max(), template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).max()],
                    c = "orange", label = "template ptp", 
                )
                ax_z_time.plot([chunk_range[0], chunk_range[1]],
                              [template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit], template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit]], c = "orange", label = "template depth")
    
                fr_unit = len(idx_unit_chunk)/(chunk_range[1]-chunk_range[0])
                ax_fr_time.plot([chunk_range[0], chunk_range[1]], [fr_unit, fr_unit], c = 'k')
                cmp+=n_spike_chunks

                ax_loc.scatter(localization_results[:, 0], depth_reg, s=1, c = color_array)
                med_depth_reg = np.median(depth_reg)
                std_depth_reg = np.median(np.abs(depth_reg - med_depth_reg))/0.675
                std_depth_reg = max(std_depth_reg, 1)
                
                med_x = np.median(localization_results[:, 0])
                std_x = np.median(np.abs(localization_results[:, 0] - med_x))/0.675
                std_x = max(std_x, 1)
                ax_loc.set_xlim((med_x-5*std_x, med_x+5*std_x))
                ax_loc.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))
                ax_loc.set_title("Localization", fontsize=7)
                ax_loc.set_xlabel("x (um)", fontsize=7)
                ax_loc.set_ylabel("reg z (um)", fontsize=7)
    
                dt_ms = np.diff(times_seconds) * 1000
                lags, acg = correlogram(times_seconds*30_000, max_lag=max_lag)
                        
                bin_edges = np.arange(
                    0,
                    max_ms + bin_ms,
                    bin_ms,
                )
                ax_ISI.hist(dt_ms, bin_edges, color="k")
                ax_ISI.set_title("isi (ms)", fontsize=7)
                ax_ISI.set_ylabel(f"count (out of {dt_ms.size} isis)", fontsize = 7, labelpad=0)
    
                bar(ax_ACG, lags, acg, fill=True, color="k")
                ax_ACG.set_title("lag (samples)", fontsize=7)
                ax_ACG.set_ylabel("acg", fontsize=7, labelpad=0)
    
                # ax_ISI.yaxis.set_label_position("right")
                ax_ISI.yaxis.tick_right()
                # ax_ACG.yaxis.set_label_position("right")
                ax_ACG.yaxis.tick_right()
    
                if len(idx_unit_chunk)>150:
                    idx_subsample = np.random.choice(len(idx_unit_chunk), 150, replace=False)
                    idx_subsample.sort()
                    # idx_unit_chunk = idx_unit_chunk[idx_subsample]
                else:
                    idx_subsample = np.arange(len(idx_unit_chunk))
    
                if np.any(template_data_chunk.unit_ids == unit):
                    ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, n_col_templates))
                
                    n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_subsample, 2], geom, times_s = times_seconds[idx_subsample], motion_est=me)


                    chans_idx, count_chans_idx = np.unique(channel_index[channels[idx_subsample]].flatten(), return_counts=True)
                    good_chans = chans_idx[count_chans_idx>0]
                    good_chans = good_chans[good_chans<384]
                    
                    if len(good_chans):
                        # print(good_chans)
                        # chans_to_plot_registered_geom = good_chans[template_data_chunk.templates[template_data_chunk.unit_ids == unit][0, :, good_chans].ptp(0).argsort()[::-1][:10]]
                        # print("channels to plot on")
                        # print(chans_to_plot_registered_geom)
    
                        max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
                        # chans_idx, chans_count = np.unique(channels[idx_subsample]+8*n_pitches_shift, return_counts=True)
                        # max_chan_registered_geom = chans_idx[chans_count.argmax()]
                        max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 10)
                        max_chan_registered_geom = max(max_chan_registered_geom, 10)

                        # print(f"CHAN PITCH SHIFT WFS {chans_idx, chans_count}")
                        # print(np.unique(channels[idx_subsample], return_counts=True))
                        # print(np.unique(channels[idx_subsample]+8*n_pitches_shift, return_counts=True))
        
                        # max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
                        # max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
                        chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)        
                        temp_chunk = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0][15:75][:, chans_to_plot_registered_geom]
                        
                        med_ptp = 1.5*temp_chunk.ptp(0).max()
    
                        if raw:
                            # print("Channels")
                            # print(channels[idx_subsample])
                            # print("n_pitches shifts")
                            # print(n_pitches_shift)
                            # print("sub channels")
                            # print(channels[idx_subsample] + 8*n_pitches_shift)
                            collisioncleaned_tpca_features = spikeio.read_waveforms_channel_index(
                                    recording,
                                    times_samples[idx_subsample],
                                    channel_index,
                                    channels[idx_subsample],
                            )
                        else:
                            with h5py.File(subh5, "r+") as h5:
                                collisioncleaned_tpca_features = h5[tpca_features_dataset_name][idx_unit_chunk[idx_subsample]]

                        
                        # do all this before chunk / PCs...
                        waveforms_target_chan = get_waveforms_on_static_channels(
                            collisioncleaned_tpca_features,
                            geom,
                            main_channels=channels[idx_subsample],
                            channel_index=channel_index,
                            target_channels=chans_to_plot_registered_geom, #np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                            n_pitches_shift=n_pitches_shift,
                            registered_geom=registered_geom,
                        )

                        # print("channels no nan")
                        # print(np.where(~np.isnan(waveforms_target_chan[0, 0])))
                        
                        chans_no_nans = np.isfinite(waveforms_target_chan[:, 0, :]).all(axis=0)
                        no_nans = np.flatnonzero(np.isfinite(waveforms_target_chan[:, 0, chans_no_nans]).all(axis=1))

                        if len(no_nans)>1 and chans_no_nans.sum()>1:
                            pcs = PCA(2).fit_transform(waveforms_target_chan[:, :, chans_no_nans].reshape(waveforms_target_chan.shape[0], -1)[no_nans])
                            if sorting_split is not None:
                                color_array = get_ccolor(sorting_split.labels[idx_unit_chunk][idx_subsample][no_nans])
                            else:
                                color_array="blue"
    
                            ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = color_array)
                            ax_pcs.set_title("PCs", fontsize=7)
                        
                            # subsampled wfs to plot
                            if not raw:
                                waveforms_target_chan = tpca.inverse_transform(waveforms_target_chan[:, :, chans_no_nans].transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, len(chans_no_nans), 121).transpose(0, 2, 1)
                            waveforms_target_chan = waveforms_target_chan[:, 15:75, :]

                            mean_waveforms = np.nanmean(waveforms_target_chan, axis = 0)
                            
                            if sorting_split is not None:
                                color_array = get_ccolor(sorting_split.labels[idx_unit_chunk][idx_subsample])
                            else:
                                color_array=np.full(len(idx_subsample), "blue")

                            ax_wfs.axvline(42-15, color = "grey")
                            ax_wfs.axvline(42-15+60, color = "grey")
                            for k in range(5):
                                for i in range(waveforms_target_chan.shape[0]):
                                    ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, k*2:k*2+2].T.flatten() + k*med_ptp, c = color_array[i], alpha = 0.05)                                        
                                    if not subtract_tpca_temp: 
                                        ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, k*2:k*2+2].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = color_array[i])
                                    else:
                                        ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, k*2:k*2+2].T.flatten() - temp_unit_tpca[15:75, chans_to_plot_registered_geom].T.flatten() + k*med_ptp, alpha = 0.05, c = color_array[i])
                            if overlap_templates:
                                for k in range(5):
                                    if tpca_templates_list is not None:
                                        ax_wfs.plot(np.arange(120), temp_unit_tpca[15:75, chans_to_plot_registered_geom].T.flatten(), c = "orange", alpha = 1)                                        
                                    else:
                                        ax_wfs.plot(np.arange(120), temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, c = "orange", alpha = 1)                                        
                                        # ax_wfs.plot(np.arange(120), mean_waveforms[:, k*2:k*2+2].T.flatten() + k*med_ptp, c = "red", alpha = 1)                                        
                            ax_wfs.set_ylabel("Wfs", fontsize=7, labelpad = 0)
                            ax_wfs_temp_subtracted.set_ylabel(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7, labelpad=0)
                            ax_wfs_temp_subtracted.set_xticks([])
                            ax_wfs_temp_subtracted.set_yticks([])
                            ax_wfs.set_xticks([])
                            ax_wfs.set_yticks([])
                        
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        if raw:
            plt.savefig(output_directory / f"rawwfs_post_deconv_{unit}_overmerge")
        else:
            plt.savefig(output_directory / f"post_deconv_{unit}_overmerge")
        plt.close()
        

def check_overmerges_post_deconv(
    sorting_post_split,
    chunk_time_ranges_s,
    data_dir_temp_data,
    output_directory,
    times_seconds_list,
    localization_results_list,
    a_list,
    depth_reg_list,
    collisioncleaned_tpca_features_all_list, 
    geom,
    me,
    channels_list,
    channel_index_list,
    tpca_list, # to get from h5 every time / tpca list? 
    tpca_templates_list=None,
    units_all=None,
    name_end_chunk_tempdata = "post_merge_GC",
    template_npz_filename = "template_data.npz",
    n_neigh = 3,
    zlim = (-100, 382),
    bin_ms=0.1, 
    max_ms=5,
    max_lag=50,
    time_smoothed=False,
):

    """
    here, everything should be a list containing infor about each chunk!! 
    """

    colors_split = ["blue", "red", "green", "yellow", "cyan", "pink", "orange", 
                    "magenta", "black", "lime", "olive", "goldenrod", "grey"]

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 4* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    units_all = np.unique(sorting_post_split.labels)
    units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)
    
    for unit in tqdm(units_all): 

        # if previous_labels is not None:
        #     units_prior = np.unique(previous_labels[sorting_post_split.labels == unit])
        # else:
        #     units_prior = unit
        
        fig = plt.figure(figsize=(40, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :(n_cols-2)//2]) 
        ax_z_time = fig.add_subplot(gs[0, (n_cols-2)//2:-2]) 
        ax_fr_time = fig.add_subplot(gs[0, -2:]) 

        idx_unit = np.flatnonzero(sorting_post_split.labels==unit)
        # if unit_labels.max()>len(colors_split):

        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_unit], sorting_post_split.denoised_ptp_amplitudes[idx_unit], s=1, c = "blue")
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_unit], sorting_post_split.point_source_localizations[idx_unit, 2], s=1, c = "blue")


        med_depth_reg = np.median(sorting_post_split.point_source_localizations[idx_unit, 2])
        std_depth_reg = np.median(np.abs(sorting_post_split.point_source_localizations[idx_unit, 2] - med_depth_reg))/0.675
        ax_z_time.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))

        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_fr_time.set_ylabel("FR (N/s)")
        ax_fr_time.set_xlabel("Time (s)")
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")
        # ax_z_time.set_ylim(zlim)

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()
        ax_fr_time.xaxis.set_label_position('top') 
        ax_fr_time.xaxis.tick_top()
        ax_fr_time.yaxis.tick_right()

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s):

            times_seconds = times_seconds_list[j]
            n_spike_chunks = times_seconds.size
            channels = channels_list[j]
            channel_index = channel_index_list[j]
            tpca = tpca_list[j]
            depth_reg = depth_reg_list[j]
            localization_results = localization_results_list[j]
            a = a_list[j]
            collisioncleaned_tpca_features_all = collisioncleaned_tpca_features_all_list[j]

            subtract_tpca_temp = False
            if tpca_templates_list is not None:
                if type(tpca_templates_list[j])==TemplateData:
                    if np.any(tpca_templates_list[j].unit_ids==unit):
                        temp_unit_tpca = tpca_templates_list[j].templates[tpca_templates_list[j].unit_ids==unit][0].transpose(1, 0)
                        temp_unit_tpca = tpca.inverse_transform(temp_unit_tpca).transpose(1, 0)
                        subtract_tpca_temp = True
                else:
                    temp_unit_tpca = tpca_templates_list[j][unit]
                    subtract_tpca_temp = True
            
            ax_ptp_time.axvline(chunk_range[0], c='red')
            ax_ptp_time.axvline(chunk_range[1], c='red')
            ax_z_time.axvline(chunk_range[0], c='red')
            ax_z_time.axvline(chunk_range[1], c='red')

            ax_pcs = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+1]) 
            ax_ISI = fig.add_subplot(gs[2*int(j//4)+1, 4*int(j - 4*(j//4))+2]) 
            ax_ACG = fig.add_subplot(gs[2*int(j//4)+2, 4*int(j - 4*(j//4))+2]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4)+1:2*int(j//4)+3, 4*int(j - 4*(j//4))+3]) 
        
            model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
            model_dir_chunk = data_dir_temp_data / model_subdir_chunk
            template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

            ax_ptp_time.plot([chunk_range[0], chunk_range[1]], [template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).max(), template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).max()],
                c = "orange", label = "template ptp", 
            )
            ax_z_time.plot([chunk_range[0], chunk_range[1]],
                          [template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit], template_data_chunk.registered_template_depths_um[template_data_chunk.unit_ids == unit]], c = "orange", label = "template depth")

            idx_chunk = np.flatnonzero(sorting_post_split.labels[cmp:cmp+n_spike_chunks]==unit)
            fr_unit = len(idx_chunk)/(chunk_range[1]-chunk_range[0])
            ax_fr_time.plot([chunk_range[0], chunk_range[1]], [fr_unit, fr_unit], c = 'k')
            cmp+=n_spike_chunks
            
            if len(idx_chunk)>150:
                idx_subsample = np.random.choice(len(idx_chunk), 150, replace=False)
                idx_chunk = idx_chunk[idx_subsample]
            
            ax_loc.scatter(localization_results[idx_chunk, 0], depth_reg[idx_chunk], s=1, c = "blue")
            med_depth_reg = np.median(depth_reg[idx_chunk])
            std_depth_reg = np.median(np.abs(depth_reg[idx_chunk] - med_depth_reg))/0.675
            std_depth_reg = max(std_depth_reg, 1)
            if len(idx_chunk):
                med_x = np.median(localization_results[idx_chunk, 0])
                std_x = np.median(np.abs(localization_results[idx_chunk, 0] - med_x))/0.675
                std_x = max(std_x, 1)
                ax_loc.set_xlim((med_x-5*std_x, med_x+5*std_x))
                ax_loc.set_ylim((med_depth_reg-5*std_depth_reg, med_depth_reg+5*std_depth_reg))
            ax_loc.set_title("Localization", fontsize=7)
            ax_loc.set_xlabel("x (um)", fontsize=7)
            ax_loc.set_ylabel("reg z (um)", fontsize=7)

            dt_ms = np.diff(times_seconds[idx_chunk]) * 1000
            lags, acg = correlogram(times_seconds[idx_chunk]*30_000, max_lag=max_lag)
            
            bin_edges = np.arange(
                0,
                max_ms + bin_ms,
                bin_ms,
            )
            ax_ISI.hist(dt_ms, bin_edges, color="k")
            ax_ISI.set_title("isi (ms)", fontsize=7)
            ax_ISI.set_ylabel(f"count (out of {dt_ms.size} isis)", fontsize = 7, labelpad=0)

            bar(ax_ACG, lags, acg, fill=True, color="k")
            ax_ACG.set_title("lag (samples)", fontsize=7)
            ax_ACG.set_ylabel("acg", fontsize=7, labelpad=0)

            # ax_ISI.yaxis.set_label_position("right")
            ax_ISI.yaxis.tick_right()
            # ax_ACG.yaxis.set_label_position("right")
            ax_ACG.yaxis.tick_right()
            
            if np.any(template_data_chunk.unit_ids == unit):
                ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, 8), aspect="auto")
            
                n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_chunk, 2], geom, times_s = times_seconds[idx_chunk], motion_est=me)

                max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
                    # print(f"MAX CHAN {max_chan_registered_geom}")

                max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
                max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
                chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)

                temp_chunk = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0][15:75][:, chans_to_plot_registered_geom]
                
                med_ptp = 1.5*temp_chunk.ptp(0).max()
                # med_ptp = 1.5*np.median(a[idx_chunk])
                
                # do all this before chunk / PCs...
                
                
                ccwfs_targetchans = get_waveforms_on_static_channels(
                    collisioncleaned_tpca_features_all[idx_chunk],
                    geom,
                    main_channels=channels[idx_chunk],
                    channel_index=channel_index,
                    target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                    n_pitches_shift=n_pitches_shift,
                    registered_geom=registered_geom,
                )
                no_nans = np.flatnonzero(np.isfinite(ccwfs_targetchans[:, 0, :]).all(axis=1))

                if len(no_nans)>2:
                    pcs = PCA(2).fit_transform(ccwfs_targetchans.reshape(ccwfs_targetchans.shape[0], -1)[no_nans])
                    # idx_chunk = idx_chunk[no_nans]

                    ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = "blue")
                    ax_pcs.set_title("PCs", fontsize=7)
                
                    # subsampled wfs to plot
                    waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
                    waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
                
                    for k in range(5):
                        for i in range(waveforms_target_chan.shape[0]):
                            ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = "blue", alpha = 0.05)
                            if not subtract_tpca_temp: 
                                ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = "blue")
                            else:
                                ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_unit_tpca[15:75, chans_to_plot_registered_geom[k*2:k*2+2]].T.flatten() + k*med_ptp, alpha = 0.05, c = "blue")
                    ax_wfs_temp_subtracted.set_ylabel(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7, labelpad=0)
                    ax_wfs.set_ylabel("Wfs", fontsize=7, labelpad = 0)
    
                    ax_wfs_temp_subtracted.set_xticks([])
                    ax_wfs_temp_subtracted.set_yticks([])
                    ax_wfs.set_xticks([])
                    ax_wfs.set_yticks([])
                    
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        plt.savefig(output_directory / f"unit_{unit}_overmerge")
        plt.close()
        

def check_overmerges_post_split_deconv(
    sorting_post_split,
    chunk_time_ranges_s,
    data_dir_temp_data,
    output_directory,
    times_seconds_list,
    localization_results_list,
    a_list,
    depth_reg_list,
    collisioncleaned_tpca_features_all_list, 
    geom,
    me,
    channels_list,
    channel_index_list,
    tpca_list, # to get from h5 every time / tpca list? 
    previous_labels=None,
    units_all=None,
    name_end_chunk_tempdata = "post_merge_GC",
    template_npz_filename = "template_data.npz",
    n_neigh = 3,
    zlim = (-100, 382),
):

    """
    here, everything should be a list containing infor about each chunk!! 
    """

    colors_split = ["blue", "red", "green", "yellow", "cyan", "pink", "orange", 
                    "magenta", "black", "lime", "olive", "goldenrod", "grey"]

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 3* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    units_all = np.unique(sorting_post_split.labels)
    units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)
    
    for unit in tqdm(units_all): 

        if previous_labels is not None:
            units_prior = np.unique(previous_labels[sorting_post_split.labels == unit])
        else:
            units_prior = unit
        
        fig = plt.figure(figsize=(25, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :n_cols//2]) 
        ax_z_time = fig.add_subplot(gs[0, n_cols//2:]) 

        idx_unit = np.flatnonzero(sorting_post_split.labels==unit)
        # if unit_labels.max()>len(colors_split):

        ax_ptp_time.scatter(sorting_post_split.times_seconds[idx_unit], sorting_post_split.denoised_ptp_amplitudes[idx_unit], s=1, c = "blue")
        ax_z_time.scatter(sorting_post_split.times_seconds[idx_unit], sorting_post_split.point_source_localizations[idx_unit, 2], s=1, c = "blue")
        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")
        ax_z_time.set_ylim(zlim)

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()

        cmp = 0
        for j, chunk_range in enumerate(chunk_time_ranges_s):

            times_seconds = times_seconds_list[j]
            n_spike_chunks = times_seconds.size
            channels = channels_list[j]
            channel_index = channel_index_list[j]
            tpca = tpca_list[j]
            depth_reg = depth_reg_list[j]
            localization_results = localization_results_list[j]
            a = a_list[j]
            collisioncleaned_tpca_features_all = collisioncleaned_tpca_features_all_list[j]
            
            ax_ptp_time.axvline(chunk_range[0], c='red')
            ax_ptp_time.axvline(chunk_range[1], c='red')
            ax_z_time.axvline(chunk_range[0], c='red')
            ax_z_time.axvline(chunk_range[1], c='red')

            ax_pcs = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//4)+2, 3*int(j - 4*(j//4))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//4)+2, 3*int(j - 4*(j//4))+1]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4)+1:2*int(j//4)+3, 3*int(j - 4*(j//4))+2]) 
        
            model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
            model_dir_chunk = data_dir_temp_data / model_subdir_chunk
            template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

            idx_chunk = np.flatnonzero(sorting_post_split.labels[cmp:cmp+n_spike_chunks]==unit)
            cmp+=n_spike_chunks
            
            if len(idx_chunk)>150:
                idx_subsample = np.random.choice(len(idx_chunk), 150, replace=False)
                idx_chunk = idx_chunk[idx_subsample]
            
            ax_loc.scatter(localization_results[idx_chunk, 0], depth_reg[idx_chunk], s=1, c = "blue")
            ax_loc.set_title("Localization", fontsize=7)
            ax_loc.set_xlabel("x (um)", fontsize=7)
            ax_loc.set_ylabel("reg z (um)", fontsize=7)

            if np.any(template_data_chunk.unit_ids == unit):
                # ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, 8))
            
                n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_chunk, 2], geom, times_s = times_seconds[idx_chunk], motion_est=me)

                if np.all(units_prior == unit):
                    max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
                else:
                    temp_chunk = template_data_chunk.templates[np.isin(template_data_chunk.unit_ids, units_prior)].mean(0)
                    max_chan_registered_geom = temp_chunk.ptp(0).argmax()
                    # print(f"MAX CHAN {max_chan_registered_geom}")

                max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
                max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
                chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)

                # temp_chunk = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0][15:75][:, chans_to_plot_registered_geom]
                # med_ptp = 1.5*temp_chunk.ptp(0).max()
                med_ptp = 1.5*np.median(a[idx_chunk])
                
                # do all this before chunk / PCs...
                ccwfs_targetchans = get_waveforms_on_static_channels(
                    collisioncleaned_tpca_features_all[idx_chunk],
                    geom,
                    main_channels=channels[idx_chunk],
                    channel_index=channel_index,
                    target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                    n_pitches_shift=n_pitches_shift,
                    registered_geom=registered_geom,
                )
                no_nans = np.flatnonzero(np.isfinite(ccwfs_targetchans[:, 0, :]).all(axis=1))

                if len(no_nans)>2:
                    pcs = PCA(2).fit_transform(ccwfs_targetchans.reshape(ccwfs_targetchans.shape[0], -1)[no_nans])
                    # idx_chunk = idx_chunk[no_nans]

                    ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = "blue")
                    ax_pcs.set_title("PCs", fontsize=7)
                
                    # subsampled wfs to plot
                    waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
                    waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
                
                    for k in range(5):
                        for i in range(waveforms_target_chan.shape[0]):
                            ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = "blue", alpha = 0.05)
                            # ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = "blue")
                    ax_wfs_temp_subtracted.set_title(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7)
                    ax_wfs.set_title("Wfs", fontsize=7)
    
                    ax_wfs_temp_subtracted.set_xticks([])
                    ax_wfs_temp_subtracted.set_yticks([])
                    ax_wfs.set_xticks([])
                    ax_wfs.set_yticks([])
                    
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        plt.savefig(output_directory / f"unit_{unit}_overmerge")
        plt.close()
        
        
def check_overmerges_multi_chan(
    sorting,
    chunk_time_ranges_s,
    data_dir_temp_data,
    output_directory,
    times_seconds,
    localization_results,
    a,
    depth_reg,
    collisioncleaned_tpca_features_all,
    geom,
    me,
    channels,
    channel_index,
    tpca,
    name_end_chunk_tempdata = "post_merge_GC",
    template_npz_filename = "template_data.npz",
    n_neigh = 3,
):

    registered_geom = registered_geometry(geom, me)
    n_chans_reg_geom = len(registered_geom)

    n_chunks = len(chunk_time_ranges_s)
    n_cols = 3* (n_chunks // 4 + 1)
    if n_chunks % 4 == 0:
        n_rows = int(2*(n_chunks // 4))
    else:
        n_rows = int(2*(n_chunks // 4 + 1))

    units_all = np.unique(sorting.labels)
    units_all = units_all[units_all>-1]

    height_ratio = [1]
    for k in range(n_rows):
        height_ratio.append(2)
    
    for unit in tqdm(units_all): 
        
        fig = plt.figure(figsize=(25, 20))
        
        gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=height_ratio)

        ax_ptp_time = fig.add_subplot(gs[0, :n_cols//2]) 
        ax_z_time = fig.add_subplot(gs[0, n_cols//2:]) 

        idx_unit = np.flatnonzero(sorting.labels==unit)
        ax_ptp_time.scatter(times_seconds[idx_unit], a[idx_unit], s=1, c = 'blue')
        ax_ptp_time.set_ylabel("PTP (s.u.)")
        ax_ptp_time.set_xlabel("Time (s)")
        ax_z_time.scatter(times_seconds[idx_unit], depth_reg[idx_unit], s=1, c = 'blue')
        ax_z_time.yaxis.set_label_position("right")
        ax_z_time.yaxis.tick_right()
        ax_z_time.set_ylabel("Reg z (um)")
        ax_z_time.set_xlabel("Time (s)")

        ax_z_time.xaxis.set_label_position('top') 
        ax_z_time.xaxis.tick_top()
        ax_ptp_time.xaxis.set_label_position('top') 
        ax_ptp_time.xaxis.tick_top()
    
        for j, chunk_range in enumerate(chunk_time_ranges_s):

            ax_ptp_time.axvline(chunk_range[0], c='red')
            ax_ptp_time.axvline(chunk_range[1], c='red')
            ax_z_time.axvline(chunk_range[0], c='red')
            ax_z_time.axvline(chunk_range[1], c='red')

            ax_pcs = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))]) 
            ax_loc = fig.add_subplot(gs[2*int(j//4)+1, 3*int(j - 4*(j//4))+1]) 
            ax_wfs = fig.add_subplot(gs[2*int(j//4)+2, 3*int(j - 4*(j//4))]) 
            ax_wfs_temp_subtracted = fig.add_subplot(gs[2*int(j//4)+2, 3*int(j - 4*(j//4))+1]) 
            ax_heatmap = fig.add_subplot(gs[2*int(j//4)+1:2*int(j//4)+3, 3*int(j - 4*(j//4))+2]) 
        
            model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
            model_dir_chunk = data_dir_temp_data / model_subdir_chunk
            template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)

            idx_chunk = np.flatnonzero(np.logical_and(
                sorting.labels==unit,
                np.logical_and(times_seconds>=chunk_range[0],
                               times_seconds<chunk_range[1]
                )
            ))
            if len(idx_chunk)>150:
                idx_chunk = np.random.choice(idx_chunk, 150, replace=False)
            
            ax_loc.scatter(localization_results[idx_chunk, 0], depth_reg[idx_chunk], s=1, c = "blue")
            ax_loc.set_title("Localization", fontsize=7)
            ax_loc.set_xlabel("x (um)", fontsize=7)
            ax_loc.set_ylabel("reg z (um)", fontsize=7)

            if np.any(template_data_chunk.unit_ids == unit):
                ax_heatmap.imshow(template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).reshape(-1, 8))
            
                n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_chunk, 2], geom, times_s = times_seconds[idx_chunk], motion_est=me)
    
                max_chan_registered_geom = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0].ptp(0).argmax()
                max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 6*8)
                max_chan_registered_geom = max(max_chan_registered_geom, 6*8)
                chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)
    
                temp_chunk = template_data_chunk.templates[template_data_chunk.unit_ids == unit][0][15:75][:, chans_to_plot_registered_geom]
                med_ptp = 1.5*temp_chunk.ptp(0).max()
                
                # do all this before chunk / PCs...
                ccwfs_targetchans = get_waveforms_on_static_channels(
                    collisioncleaned_tpca_features_all[idx_chunk],
                    geom,
                    main_channels=channels[idx_chunk],
                    channel_index=channel_index,
                    target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10),
                    n_pitches_shift=n_pitches_shift,
                    registered_geom=registered_geom,
                )
                no_nans = np.flatnonzero(np.isfinite(ccwfs_targetchans[:, 0, :]).all(axis=1))
                
                if len(no_nans)>2:
                    pcs = PCA(2).fit_transform(ccwfs_targetchans.reshape(ccwfs_targetchans.shape[0], -1)[no_nans])
                    idx_chunk = idx_chunk[no_nans]
                
                    ax_pcs.scatter(pcs[:, 0], pcs[:, 1], s=1, c = "blue")
                    ax_pcs.set_title("PCs", fontsize=7)
                
                    # subsampled wfs to plot
                    waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 20, 121).transpose(0, 2, 1)
                    waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
                
                    for k in range(5):
                        for i in range(waveforms_target_chan.shape[0]):
                            ax_wfs.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() + k*med_ptp, c = "blue", alpha = 0.05)
                            ax_wfs_temp_subtracted.plot(np.arange(120), waveforms_target_chan[i][:, np.arange(k*4,k*4+4, 2)].T.flatten() - temp_chunk[:, k*2:k*2+2].T.flatten() + k*med_ptp, alpha = 0.05, c = "blue")
                    ax_wfs_temp_subtracted.set_title(f"Template-subtracted wfs (max chan {max_chan_registered_geom})", fontsize=7)
                    ax_wfs.set_title("Wfs", fontsize=7)
    
                    ax_wfs_temp_subtracted.set_xticks([])
                    ax_wfs_temp_subtracted.set_yticks([])
                    ax_wfs.set_xticks([])
                    ax_wfs.set_yticks([])
                    
        plt.suptitle(f"Unit {unit}", y=0.925)
        
        # PCs on max chan 
        # WFS on same chans 
        plt.savefig(output_directory / f"unit_{unit}_overmerge")
        plt.close()



def check_overmerges_previous(
    sorting,
    chunk_time_ranges_s,
    localization_results,
    depth_reg,
    a,
    times_seconds,
    waveforms_maxchan,
    output_directory,
):

    units_all = np.unique(sorting.labels)
    units_all = units_all[units_all>-1]
    
    for unit in tqdm(units_all): 
    
        idx_unit = np.flatnonzero(np.logical_and(
            sorting.labels==unit, 
            np.logical_and(
                np.logical_and(
                    localization_results[:, 2]>-150,
                    localization_results[:, 2]<150+282),
                np.logical_and(
                    localization_results[:, 0]>-150,
                    localization_results[:, 0]<150+42)
            )))
        
        fig = plt.figure(figsize=(15, 20))
        
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 3])
        
        ax1 = fig.add_subplot(gs[0, 0]) 
        ax2 = fig.add_subplot(gs[1, 0]) 
        ax3 = fig.add_subplot(gs[2, 0]) 
        ax4 = fig.add_subplot(gs[3, 0]) 
        ax5 = fig.add_subplot(gs[4, 0]) 
        ax6 = fig.add_subplot(gs[:, 1]) 
        
        ax1.scatter(times_seconds[idx_unit], a[idx_unit], s=1, c='blue')
        ax1.set_ylabel("PTP (s.u.)")
        ax1.set_xticks([])
        
        ax2.scatter(times_seconds[idx_unit], depth_reg[idx_unit], s=1, c='blue')
        ax2.set_ylabel("reg depth (um)")
        ax2.set_xticks([])
        
        ax3.scatter(times_seconds[idx_unit], localization_results[idx_unit, 2], s=1, c='blue')
        ax3.set_ylabel("depth (um)")
        ax3.set_xticks([])
        
        # ax4.scatter(times_seconds_all[idx_unit], localization_results[idx_unit, 0], s=1, c='blue')
        # ax4.set_ylabel("x (um)")
        # ax4.set_xticks([])
        
        ax4.scatter(times_seconds[idx_unit], localization_results[idx_unit, 0], s=1, c='blue')
        ax4.set_ylabel("x (um)")
        # ax5.set_ylabel("conv scores")
        ax4.set_xlabel("time (seconds)")
    
        pcs = PCA(2).fit_transform(waveforms_maxchan[idx_unit])
        ax5.scatter(pcs[:, 0], pcs[:, 1], s=1, c='blue')
        ax5.set_ylabel("PC1")
        ax5.set_xlabel("PC2")
    
        med_ptp = 0.75*np.max(a[idx_unit])
        for j, chunk_time in enumerate(chunk_time_ranges_s):
            sub_chunks = np.linspace(chunk_time[0], chunk_time[1], 6)
            for k in range(5):
                # ax1.axvline(sub_chunks[k], c='grey')
                # ax1.axvline(sub_chunks[k+1], c='grey')
                idx_sub_chunk = np.flatnonzero(np.logical_and(times_seconds[idx_unit]>=sub_chunks[k], times_seconds[idx_unit]<sub_chunks[k+1]))
                if len(idx_sub_chunk):
                    if len(idx_sub_chunk)>150:
                        idx_sub_chunk = np.random.choice(idx_sub_chunk, 150, replace=False)
                        idx_sub_chunk.sort()
                    idx_sub_chunk = idx_unit[idx_sub_chunk]
                    for i in idx_sub_chunk:
                        ax6.plot(np.arange(k*60, (k+1)*60), waveforms_maxchan[i, 15:75] + j*med_ptp, c='grey', alpha = 0.1)
            ax1.axvline(chunk_time[0], c='red')
            ax1.axvline(chunk_time[1], c='red')
    
        ax6.set_title("max chan over time (bottom to top)")
        ax6.set_xticks([])
        ax6.set_yticks([])
        
        plt.suptitle(f"Unit {unit}", y=0.9)
    
        plt.savefig(output_directory / f"unit_{unit}_overmerge")
        plt.close()


def check_overmerges_per_chunk(
    sorting,
    geom,
    me,
    chunk_time_ranges_s,
    localization_results,
    depth_reg,
    a,
    times_seconds,
    collisioncleaned_tpca_features_all,
    channels,
    channel_index,
    tpca,
    data_dir_temp_data,
    output_directory,
    name_end_chunk_tempdata = "post_merge_GC",
    template_npz_filename = "template_data.npz",

):

    registered_geom = registered_geometry(geom, me)

    units_all = np.unique(sorting.labels)
    units_all = units_all[units_all>-1]
    
    for unit in tqdm(units_all): 
    
        idx_unit = np.flatnonzero(np.logical_and(
            sorting.labels==unit, 
            np.logical_and(
                np.logical_and(
                    localization_results[:, 2]>-150,
                    localization_results[:, 2]<150+282),
                np.logical_and(
                    localization_results[:, 0]>-150,
                    localization_results[:, 0]<150+42)
            )))
            
        med_ptp = 0.75*np.max(a[idx_unit])
        for j, chunk_time in enumerate(chunk_time_ranges_s):

            model_subdir_chunk = f"chunk_{j}_{name_end_chunk_tempdata}"
            model_dir_chunk = data_dir_temp_data / model_subdir_chunk
            template_data_chunk = TemplateData.from_npz(model_dir_chunk / template_npz_filename)
            n_chans_reg_geom = template_data_chunk.templates.shape[2]
    
            max_chan_registered_geom = template_data_chunk.templates[unit].ptp(0).argmax()
            max_chan_registered_geom = min(max_chan_registered_geom, n_chans_reg_geom - 10)
            max_chan_registered_geom = max(max_chan_registered_geom, 10)
            chans_to_plot_registered_geom = np.arange(max_chan_registered_geom - 10, max_chan_registered_geom+10, 2)
            
            plt.figure(figsize=(10, 20))
            
            sub_chunks = np.linspace(chunk_time[0], chunk_time[1], 6)
            for k in range(5):
                idx_sub_chunk = np.flatnonzero(np.logical_and(times_seconds[idx_unit]>=sub_chunks[k], times_seconds[idx_unit]<sub_chunks[k+1]))
                if len(idx_sub_chunk):
                    n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_unit, 2][idx_sub_chunk], geom, times_s = times_seconds[idx_unit][idx_sub_chunk], motion_est=me)

                    ccwfs_targetchans = get_waveforms_on_static_channels(
                        collisioncleaned_tpca_features_all[idx_unit][idx_sub_chunk],
                        geom,
                        main_channels=channels[idx_unit][idx_sub_chunk],
                        channel_index=channel_index,
                        target_channels=np.arange(max_chan_registered_geom-10, max_chan_registered_geom+10, 2),
                        n_pitches_shift=n_pitches_shift,
                        registered_geom=registered_geom,
                    )
                    
                    waveforms_target_chan = tpca.inverse_transform(ccwfs_targetchans.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, 10, 121).transpose(0, 2, 1)
                    waveforms_target_chan = waveforms_target_chan[:, 15:75, :]
    
                    if len(waveforms_target_chan)>150:
                        waveforms_target_chan = waveforms_target_chan[np.random.choice(len(waveforms_target_chan), 150, replace=False)]
                    for i in range(len(waveforms_target_chan)):
                        for c in range(waveforms_target_chan.shape[2]):
                            plt.plot(np.arange(k*60, (k+1)*60), waveforms_target_chan[i, :, c] + c*med_ptp, c='grey', alpha = 0.1)

            plt.title("10 chans over time within sub chunks (left to right)")                    
            plt.savefig(output_directory / f"unit_{unit}_overmerge_chunk_{j}")
            plt.close()







