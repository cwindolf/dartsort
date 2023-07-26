# %%
import time
from itertools import zip_longest
from pathlib import Path

import numpy as np
# import numpy.linalg as la
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from torch import nn
from tqdm.auto import trange

try:
    from .denoise_temporal_decrease import (
        _enforce_temporal_decrease_right, _enforce_temporal_decrease_left
    )
except ImportError:
    pass


pretrained_path = (
    Path(__file__).parent.parent / "pretrained/single_chan_denoiser.pt"
)


class SingleChanDenoiser(nn.Module):
    """Cleaned up a little. Why is conv3 here and commented out in forward?"""

    def __init__(
        # self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121
        self, n_filters=[16, 8], filter_sizes=[5, 11], spike_size=121
    ):
        super(SingleChanDenoiser, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, n_filters[0], filter_sizes[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(n_filters[0], n_filters[1], filter_sizes[1]), nn.ReLU())
        if len(n_filters) > 2:
            self.conv3 = nn.Sequential(nn.Conv1d(n_filters[1], n_filters[2], filter_sizes[2]), nn.ReLU())
        n_input_feat = n_filters[1] * (spike_size - filter_sizes[0] - filter_sizes[1] + 2)
        self.out = nn.Linear(n_input_feat, spike_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self, fname_model=pretrained_path):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self

def wfs_corr(wfs_raw, wfs_denoise):
    return torch.sum(wfs_denoise*wfs_raw, 1)/torch.sqrt(torch.sum(wfs_raw*wfs_raw,1) * torch.sum(wfs_denoise*wfs_denoise,1))

def denoise_with_phase_shift(chan_wfs, phase_shift, dn, spk_signs, offset=42, small_threshold = 2, corr_th = 0.8):
    '''
    input an NxT matrix of spike wavforms, and return 1) the denoised waveforma according to the phaseshift 2) the phaseshift of the denoised waveform, 3) index that shows whether the denoised waveform is identified as hallucination
    '''
    N, T = chan_wfs.shape

    chan_wfs = torch.roll(chan_wfs[i, :], - int(phase_shift[i]))
    
    wfs_denoised = dn(chan_wfs)
    
    
    which = slice(offset-10, offset+10)
    
    d_s_corr = wfs_corr(chan_wfs[:, which], wfs_denoised[:, which])#torch.sum(wfs_denoised[which]*chan_wfs[which], 1)/torch.sqrt(torch.sum(chan_wfs[which]*chan_wfs[which],1) * torch.sum(wfs_denoised[which]*wfs_denoised[which],1)) ## didn't use which at the beginning! check whether this changes the results
    
    halu_idx = (ptp(wfs_denoised, 1)<small_threshold) & (d_s_corr<corr_th)
    
    wfs_denoised = torch.roll(wfs_denoised, int(phase_shift), 1)
    
    #CHECK THE CORRELATION BETWEEN THE DENOISED WAVEFORM AND THE RAW WAVEFORM, HALLUCINATION WILL HAVE A SMALL VALUE
    phase_shifted = torch.argmax(torch.swapaxes(wfs_denoised, 0, 1) * spk_signs, 0) - offset
    phase_shifted[halu_idx] = 0
    
    
    return wfs_denoised, phase_shifted.long(), halu_idx.long()



def phase_shift_and_hallucination_idx_preshift(waveforms_roll_denoise, waveforms_roll, spk_signs, offset=42, small_threshold = 2, corr_th = 0.8):
    phase_shifted = torch.argmax(waveforms_roll_denoise.permute(2, 3, 1, 0) * spk_signs, 2) - offset 
    N, T, C, S = waveforms_roll_denoise.shape
    waveforms_roll_denoise = waveforms_roll_denoise.permute(0, 2, 3, 1)  #N x C x S xT
    waveforms_roll = waveforms_roll.permute(0, 2, 3, 1)
    
    waveforms_roll_denoise = torch.reshape(waveforms_roll_denoise, (-1, T))
    waveforms_roll = torch.reshape(waveforms_roll, (-1, T))
    
    which = slice(offset-10, offset+10)
    
    d_s_corr = wfs_corr(waveforms_roll_denoise[:, which], waveforms_roll[:, which])#torch.sum(wfs_denoised[which]*chan_wfs[which], 1)/torch.sqrt(torch.sum(chan_wfs[which]*chan_wfs[which],1) * torch.sum(wfs_denoised[which]*wfs_denoised[which],1)) ## didn't use which at the beginning! check whether this changes the results
    
    halu_idx = (ptp(waveforms_roll_denoise, 1)<small_threshold) & (d_s_corr<corr_th)
    halu_idx = halu_idx.long()
    halu_idx = torch.reshape(halu_idx, (N, C, S))

    #CHECK THE CORRELATION BETWEEN THE DENOISED WAVEFORM AND THE RAW WAVEFORM, HALLUCINATION WILL HAVE A SMALL VALUE
    # phase_shifted = torch.argmax(torch.reshape(waveforms_roll_denoise, (N, C, S, T)).permute(1, 2, 3, 0) * spk_signs, 2) - offset  #C x S x N
    phase_shifted = phase_shifted.permute(2, 0, 1)

    # phase_shifted = torch.where(halu_idx>0, phase_shifted, 0)

    phase_shifted = phase_shifted.long()
    
    return phase_shifted, halu_idx



def make_ci_graph(channel_index, geom, device, CH_N=384):

    channel_index = torch.tensor(channel_index, device = device)
    geom = torch.tensor(geom, device = device)
    CH_N = torch.tensor(CH_N, device = device)
    
    N, L = channel_index.shape
    x_pitch = torch.diff(torch.unique(geom[:, 0]))[0]
    y_pitch = torch.diff(torch.unique(geom[:, 1]))[0]

    ci_graph_all = torch.full((CH_N, L, 7), L, device = device)
    maxCH_neighbor = torch.full((CH_N, 8), L, device = device)
    
    for i in range(N):
        ci = channel_index[i]
        non_nan_idx = torch.where(ci < CH_N)[0]
        ci = ci[non_nan_idx]
        l = len(ci)
        ci_geom = geom[ci]
        for ch in range(l):
            group = torch.where(((torch.abs(ci_geom[:, 0] - ci_geom[ch, 0]) == x_pitch) & (torch.abs(ci_geom[:, 1] - ci_geom[ch, 1]) == y_pitch)) |
                                ((torch.abs(ci_geom[:, 0] - ci_geom[ch, 0]) == 0) & (torch.abs(ci_geom[:, 1] - ci_geom[ch, 1]) == 2 * y_pitch)) |
                                ((torch.abs(ci_geom[:, 0] - ci_geom[ch, 0]) == 2 * x_pitch) & (torch.abs(ci_geom[:, 1] - ci_geom[ch, 1]) == 0)))[0]
            ci_graph_all[i, ch, :len(group)] = group

        maxCH_idx = torch.where(ci == i)[0]
        maxCH_n = ci_graph_all[i, maxCH_idx[0], : ]
        maxCH_neighbor[i, 0:(len(maxCH_n) + 1)] =  torch.cat((maxCH_n, maxCH_idx)).sort().values

    return ci_graph_all, maxCH_neighbor


def make_ci_graph_all_maxCH(ci_graph_all, maxCH_neighbor, device):
    N, C, L = ci_graph_all.shape
    ci_graph_all_maxCH_uniq = torch.zeros((N, C, L, L+1), device = device)
    ci_graph_all_maxCH_uniq = ci_graph_all.clone().detach().unsqueeze(3).repeat(1, 1, 1, L+1)  #clone?
    
    for i in range(N):
        mcneighbors = maxCH_neighbor[i, :]
        mcneighbors_valid = mcneighbors[mcneighbors<C]
        
        all_CH = torch.arange(C, device = device)
        
        for mcs_idx in mcneighbors_valid:
            idx = torch.where(mcneighbors==mcs_idx)[0]
            for ch in all_CH[all_CH>mcs_idx]:
                group = ci_graph_all[i, ch, :].clone().detach()
                if (len(torch.nonzero(group < mcs_idx))!=0):
                    new_neighbors = torch.cat((group[group > mcs_idx], torch.tensor([mcs_idx], device = device)))
                    # print(F.pad(new_neighbors, (0, C - new_neighbors.shape[0]), "constant", L).shape)
                    ci_graph_all_maxCH_uniq[i, ch, :, idx[0]] = F.pad(new_neighbors, (0, L - new_neighbors.shape[0]), "constant", C)
            for ch in all_CH[all_CH<mcs_idx]:
                group = ci_graph_all[i, ch, :].clone().detach()
                if (len(torch.nonzero(group[group<C] > mcs_idx))!=0):
                    new_neighbors = torch.cat((group[group < mcs_idx], torch.tensor([mcs_idx], device = device)))
                    ci_graph_all_maxCH_uniq[i, ch, :, idx[0]] = F.pad(new_neighbors, (0, L - new_neighbors.shape[0]), "constant", C)

    return ci_graph_all_maxCH_uniq  #384xCxLx(L+1)




def ptp(t, axis):
    # ptp for torch
    t = torch.nan_to_num(t, nan=0.0)
    return t.max(axis).values - t.min(axis).values




def multichan_phase_shift_denoise_preshift(waveforms, ci_graph_all_maxCH_uniq, maxCH_neighbor, Denoiser, maxchans, device, CH_N=384, offset=42):
    N, T, C = waveforms.shape
    waveforms = F.pad(waveforms, (0, 1), 'constant', 0) #waveforms.shape = NxTx(C+1)
    C = C + 1
    
    ci_graph_all_maxCH_uniq = F.pad(ci_graph_all_maxCH_uniq, (0, 0, 0, 0, 0, 1), 'constant', C - 1)
    
    phase_shift_array = torch.tensor([0, 15, 12, 9, 6, 3, -3, -6, -9, -12, -15], device = device)
    # phase_shift_array = torch.tensor([0, 15, 10, 5, -5, 10, 15], device = device)
    S = phase_shift_array.shape[0]

    pick_idx = torch.zeros((N, C, phase_shift_array.shape[0]), device = device)
    pick_idx = F.pad(pick_idx, (0, 1), 'constant', S) # one-hot array to pick the correct phase-shift, default: pick all 0 waveforms 

    CH_checked = torch.zeros((N, C), device = device)
    
    CH_checked = F.pad(CH_checked, (0, 1), 'constant', 1)
    
    waveforms_roll_all = torch.cat((waveforms,
                                    torch.roll(waveforms, -15, 1),
                                    torch.roll(waveforms, -12, 1), 
                                    torch.roll(waveforms, -9, 1), 
                                    torch.roll(waveforms, -6, 1), 
                                    torch.roll(waveforms, -3, 1), 
                                    torch.roll(waveforms, 3, 1), 
                                    torch.roll(waveforms, 6, 1), 
                                    torch.roll(waveforms, 9, 1), 
                                    torch.roll(waveforms, 12, 1), 
                                    torch.roll(waveforms, 15, 1)),
                                    2)

    # phase_shift_array = F.pad(phase_shift_array, (0, 1), 'constant', 0)
    waveforms_roll_all = waveforms_roll_all.permute(0, 2, 1)
    
    waveforms_roll_denoise = Denoiser(waveforms_roll_all.reshape(-1, T)).reshape([N, S, C, T])
    
    waveforms_roll_denoise = waveforms_roll_denoise.permute(0, 3, 2, 1)#waveforms_roll_denoise.permute(0, 3, 1, 2) # NxTxCx11
    waveforms_roll_all = waveforms_roll_all.permute(0, 2, 1).reshape(N, T, -1, C).permute(0, 1, 3, 2)                             
                                   
    for i in range(S):
        waveforms_roll_denoise[:,:,:,i] = torch.roll(waveforms_roll_denoise[:,:,:,i], int(phase_shift_array[i]), 1)

    waveforms_roll_denoise = F.pad(waveforms_roll_denoise, (0, 1), 'constant', 0) # NxTxCx12
    waveforms_ptp_denoise = ptp(waveforms_roll_denoise, 1) # NxCx12
    
    col_idx = maxCH_neighbor[maxchans,:] #shape: N x 8
    row_idx = torch.arange(N, device = device)[:, None].repeat(1, 8).reshape(-1)

    maxCH_denoised_ptp = waveforms_ptp_denoise[row_idx, torch.flatten(col_idx), 0].reshape(N, 8)
    real_maxCH_info = torch.max(maxCH_denoised_ptp, dim = 1)
    
    
    real_maxCH_idx = real_maxCH_info[1]
    real_maxCH = col_idx[range(N), real_maxCH_idx].long()
    
    spk_signs = torch.sign(waveforms_roll_denoise[range(N), offset, real_maxCH, 0])
                                   
    
    phase_shift, halluci_idx = phase_shift_and_hallucination_idx_preshift(waveforms_roll_denoise[:, :, :, :S], waveforms_roll_all, spk_signs)
    #compute phase-shift and hallucination index for all waveforms denoised at all phase-shifts                               
    
    del(waveforms_roll_all)
    del(waveforms)
    
    phase_shift = F.pad(phase_shift, (0, 1), 'constant', 0) # pad for the additional values
    halluci_idx = F.pad(halluci_idx, (0, 1), 'constant', 0)  
    
    # ci_graph_idx = ci_graph_all_maxCH_uniq[maxchans, :, :, real_maxCH_idx] # picks the right ci_graph to use, size: N x C x L
    
       
                            
    pick_idx[range(N), real_maxCH, 0] = 1
    pick_idx[range(N), real_maxCH, S] = 0
                                   
    previous_pick_idx = pick_idx.clone().detach()
    
    thresholds = torch.max(0.3*real_maxCH_info[0], torch.tensor(3))
    
    CH_checked[range(N), real_maxCH] = 1
     
                                   
    
    Q = [torch.cat((torch.arange(N, device = device).unsqueeze(1), real_maxCH.unsqueeze(1)), dim = 1)]
    
                               
    while True:
        if all(len(v)==0 for v in Q):
            return torch.einsum("itjk,ijk->itj", waveforms_roll_denoise, pick_idx)[:, :, range(40)]#, waveforms_roll_denoise[:,:,range(40),0]               
        u = Q.pop()

        
        Q_neighbors = ci_graph_all_maxCH_uniq[maxchans[u[:,0]], u[:,1], :, real_maxCH_idx[u[:,0]]] # picks the right ci_graph to use, size: N x C x L ci_graph_idx[u[:,0], u[:,1], :] #size: ? x L
                        
        
        b = Q_neighbors.shape[1]

        unfold_idx = u[:,0][:, None].repeat(1, b).reshape(-1)
                     
        Q_neighbors = Q_neighbors.reshape(-1)  # ?*L

        neighbor_check = CH_checked[unfold_idx, Q_neighbors]
        
        unchecked = (neighbor_check == 0)
        unchecked_idx = torch.squeeze(torch.nonzero(unchecked), 1)
                                   
        unfold_idx_unchecked = unfold_idx[unchecked_idx]
        Q_neighbors_unchecked = Q_neighbors[unchecked_idx]        # unfold the indeces to be a long array               
        
            
        CH_checked[unfold_idx_unchecked, Q_neighbors_unchecked] = 1
                                   
        Q_neighbors_neighbors = ci_graph_all_maxCH_uniq[maxchans[unfold_idx_unchecked], Q_neighbors_unchecked, :, real_maxCH_idx[unfold_idx_unchecked]] #ci_graph_idx[unfold_idx_unchecked, Q_neighbors_unchecked, :]

        unfold_neighbor_idx = unfold_idx_unchecked[:, None].repeat(1, b).reshape(-1)

        #√√√√√√√
        
        ptp_neighbors_neighbors = torch.einsum("ijk,ijk->ij", waveforms_ptp_denoise, pick_idx)[unfold_neighbor_idx, Q_neighbors_neighbors.reshape(-1)]

        ptp_neighbors_neighbors = torch.reshape(ptp_neighbors_neighbors, (-1, b)) #size: ?*L x L
        
        max_ptp_info  = torch.max(ptp_neighbors_neighbors, 1) #size: ?*L
        
        neighbor_max_ptps = max_ptp_info[0]
        max_ptp_idx = max_ptp_info[1]                   
        threshold_accept_idx = (neighbor_max_ptps > thresholds[unfold_idx_unchecked])
        
        #check!!!
        phase_shift_all = torch.einsum("ijk,ijk->ij", phase_shift.float(), pick_idx) # N x C
        # print()
        # phase_shift_all = torch.einsum("k,ijk->ij", phase_shift_array.float(), pick_idx) # N x C
        
        neighbor_phaseshift = phase_shift_all[unfold_idx_unchecked, Q_neighbors_neighbors[range(len(max_ptp_idx)),max_ptp_idx]]
        
        phase_shift_previous = torch.einsum("ijk,ijk->ij", phase_shift.float(), previous_pick_idx) # N x C
        # print(neighbor_phaseshift)
        
        phase_accept_idx = (torch.min(torch.abs(phase_shift_previous[unfold_idx_unchecked,:] - neighbor_phaseshift[:, None]), 1)[0]<=10)
     
        pick_phaseshift_info = torch.min(torch.abs(neighbor_phaseshift[:, None] - phase_shift_array[None,:]), 1)
        pick_phaseshift_idx = pick_phaseshift_info[1]
        
        previous_pick_idx = pick_idx.clone().detach()

        accept_idx = torch.squeeze(torch.nonzero(phase_accept_idx & threshold_accept_idx), 1)
        unaccept_idx = torch.squeeze(torch.nonzero((phase_accept_idx & threshold_accept_idx) == 0), 1)
        pick_idx[unfold_idx_unchecked, Q_neighbors_unchecked, S] = 0
        pick_idx[unfold_idx_unchecked[accept_idx], Q_neighbors_unchecked[accept_idx], pick_phaseshift_idx[accept_idx]] = 1
        pick_idx[unfold_idx_unchecked[unaccept_idx], Q_neighbors_unchecked[unaccept_idx], 0] = 1
        

        halluci_idx_neighbors = torch.einsum("ijk,ijk->ij", halluci_idx.float(), pick_idx)[unfold_idx, Q_neighbors].reshape(-1, b)
        halluci_keep_spike_idx = (torch.sum(halluci_idx_neighbors, 1)<3)
        
        halluci_keep_spike_idx = halluci_keep_spike_idx[:, None].repeat(1, b).reshape(-1)

        seek_idx = torch.squeeze(torch.nonzero(unchecked & halluci_keep_spike_idx), 1)

        Q.insert(0, torch.cat((unfold_idx[seek_idx][:, None], Q_neighbors[seek_idx][:, None]), 1))

            
            


    
    
def multichan_phase_shift_denoise(waveforms, ci_graph_on_probe, maxCH_neighbor, Denoiser, maxchans, CH_N=384, offset=42):
    t = time.time()
    N, T, C = waveforms.shape
    
    if waveforms.get_device() >= 0:
        device = "cuda"
    else:
        device = "cpu"
    
    DenoisedWF = torch.zeros(waveforms.shape, device = device)
        

    col_idx = maxCH_neighbor[maxchans, :]
    row_idx = torch.arange(N)[None,:].repeat(8, 1)
    maxCH_neighbor_wfs = waveforms[torch.reshape(row_idx.T,(-1,)), : , torch.flatten(col_idx)]
    wfs_denoised_mc_neighbors = Denoiser(maxCH_neighbor_wfs).reshape([N, 8, T])
    # wfs_denoised_mc_neighbors = torch.nan_to_num(wfs_denoised_mc_neighbors, nan=0.0)
    max_neighbor_ptps = ptp(wfs_denoised_mc_neighbors, 2)
    real_maxCH_info = torch.max(max_neighbor_ptps, dim = 1)
    real_maxCH_idx = real_maxCH_info[1]
    
    # print(real_maxCH_idx.get_device())
    # real_maxCH = col_idx[range(N), real_maxCH_idx.to('cpu')].long()
    real_maxCH = col_idx[range(N), real_maxCH_idx].long()
    
    wfs_denoised = wfs_denoised_mc_neighbors[range(N), real_maxCH_idx, :]
    
    
    
    
    DenoisedWF[range(N), : ,real_maxCH] = wfs_denoised # denoise all maxCH in one batch
    
    
    thresholds = torch.max(0.3*real_maxCH_info[0], torch.tensor(3)) # threshold to identify trustable neighboring channel
    
    mcs_phase_shift = torch.argmax(torch.abs(torch.squeeze(wfs_denoised)), 1) - offset
    spk_signs = torch.sign(wfs_denoised[range(N), np.ones(N)*offset])
    
    CH_checked = torch.zeros((N, C), device = device)
    CH_phase_shift = torch.zeros((N, C), dtype = torch.int64, device = device)
    parent = torch.full((N, C), float('nan'), device = device)


    parent_peak_phase = torch.zeros((N, C), device = device)

    CH_phase_shift[range(N), real_maxCH] = mcs_phase_shift

    wfs_ptp = torch.zeros((N, C), device = device)
    halluci_idx = torch.zeros((N, C), device = device).long()    

    # wfs_ptp[range(N), real_maxCH.to(device)] = max_neighbor_ptps[range(N), real_maxCH_idx]
    wfs_ptp[range(N), real_maxCH] = max_neighbor_ptps[range(N), real_maxCH_idx]
    
    real_maxCH = real_maxCH.detach().numpy()
    
    CH_checked[np.arange(N), real_maxCH] = 1
    
    Q = dict()
    for i in range(N):
        Q[i] = []
        Q[i].append(real_maxCH[i])

    ci_graph_all = dict()
    for i in range(N):
        ci_graph =  dict()
        l = len(ci_graph_on_probe[maxchans[i]])
        mcs_idx = real_maxCH[i]
        for ch in range(l):
            group = ci_graph_on_probe[maxchans[i]][ch].clone().detach()

            if (len(torch.nonzero(group > mcs_idx))!=0) & (len(torch.nonzero(group < mcs_idx))!=0) & (ch!= mcs_idx):
                if ch>mcs_idx:
                    ci_graph[ch] = torch.cat([group[group > mcs_idx], torch.tensor([mcs_idx], device = device)])
                else:
                    ci_graph[ch] = torch.cat([group[group < mcs_idx], torch.tensor([mcs_idx], device = device)])
            else:
                ci_graph[ch] = group

        ci_graph_all[i] = ci_graph


    while True:
        if len(sum(Q.values(),[])) == 0:
            return DenoisedWF
        
        Q_neighbors = dict()
        for i in range(N):
            q = Q[i]
            if len(q)>0:
                ci_graph = ci_graph_all[i]

                u = q.pop()
                v = ci_graph[u]

                
                Q_neighbors[i] = list(v[CH_checked[i, v] == 0])
            else:
                Q_neighbors[i] = []
                
  
        #####
        for nodes in zip_longest(*(Q_neighbors[i] for i in range(N))):
            keep_N_idx = torch.tensor([j for j in range(len(nodes)) if nodes[j] is not None], device = device)
            for j in keep_N_idx:
                
                k = nodes[j].item()
                Q[j.item()].insert(0, k)
                    
                neighbors = ci_graph_all[j.item()][k]
                checked_neighbors = neighbors[CH_checked[j, neighbors] == 1]
                

                phase_shift_ref = torch.argmax(wfs_ptp[j,checked_neighbors])
                
                rest_phase_shift = torch.cat((parent_peak_phase[j,:checked_neighbors[phase_shift_ref]], parent_peak_phase[j,checked_neighbors[phase_shift_ref]+1:])).to(device)
                
                
                if (wfs_ptp[j, checked_neighbors[phase_shift_ref]] > thresholds[j]) & (torch.min(torch.abs(rest_phase_shift - CH_phase_shift[j, checked_neighbors[phase_shift_ref]]))<=5):
                    parent_peak_phase[j, k] = CH_phase_shift[j, checked_neighbors[phase_shift_ref]]
                else:
                    parent_peak_phase[j, k] = 0

                parent[j, k] = checked_neighbors[phase_shift_ref]
                CH_checked[j, k] = 1

            CH_idx = torch.tensor([nodes[j] for j in keep_N_idx], device = device)
            
            wfs = waveforms[keep_N_idx, : , CH_idx]

            spk_denoised_wfs, CH_phase_shift[keep_N_idx, CH_idx], halluci_idx[keep_N_idx, CH_idx] = denoise_with_phase_shift(wfs, parent_peak_phase[keep_N_idx, CH_idx], Denoiser, spk_signs[keep_N_idx])
                    
            
            DenoisedWF[keep_N_idx, : , CH_idx] = spk_denoised_wfs      

            wfs_ptp[keep_N_idx, CH_idx] = ptp(spk_denoised_wfs, 1)

        #####
        for i in range(N):
            v = Q_neighbors[i]
            if torch.sum(halluci_idx[i, v])>=3:
                q_partial = v#.tolist()
                while len(q_partial)>0:
                    x = q_partial.pop().item()
                    y = ci_graph_all[i][x]
                    for z in y:
                        if CH_checked[i, z] == 0:
                            CH_checked[i, z] = 1
                            q_partial.insert(0,z)
                            halluci_idx[i, z] = 1


# def denoise_phaseshift_two_split(waveforms):
    

def temporal_align(waveforms, maxchans=None, offset=42):
    N, T, C = waveforms.shape
    if maxchans is None:
        maxchans = waveforms.ptp(1).argmax(1)
    offsets = waveforms[np.arange(N), :, maxchans].argmin(1)
    rolls = offset - offsets
    out = np.empty_like(waveforms)
    pads = [(0, 0), (0, 0)]
    for i, roll in enumerate(rolls):
        if roll > 0:
            pads[0] = (roll, 0)
            start, end = 0, T
        elif roll < 0:
            pads[0] = (0, -roll)
            start, end = -roll, T - roll
        else:
            out[i] = waveforms[i]
            continue

        pwf = np.pad(waveforms[i], pads, mode="linear_ramp")
        out[i] = pwf[start:end, :]

    return out, rolls


def invert_temporal_align(aligned, rolls):
    T = aligned.shape[1]
    out = np.empty_like(aligned)
    pads = [(0, 0), (0, 0)]
    for i, roll in enumerate(-rolls):
        if roll > 0:
            pads[0] = (roll, 0)
            start, end = 0, T
        elif roll < 0:
            pads[0] = (0, -roll)
            start, end = -roll, T - roll
        else:
            out[i] = aligned[i]
            continue

        pwf = np.pad(aligned[i], pads, mode="linear_ramp")
        out[i] = pwf[start:end, :]

    return out


def enforce_decrease(waveform, max_chan=None, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    ptp = wf.ptp(0)
    if max_chan is None:
        max_chan = ptp.argmax()

    max_chan_even = max_chan - max_chan % 2
    max_chan_odd = max_chan_even + 1

    len_reg = (max_chan_even - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_even - 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_even - 4 - 2 * i])
            regularizer[len_reg - i - 1] = (
                ptp[max_chan_even - 4 - 2 * i] / max_ptp
            )
        wf[:, np.arange(0, max_chan_even - 2, 2)] /= regularizer

    len_reg = (n_chan - 1 - max_chan_even - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_even + 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_even + 4 + 2 * i])
            regularizer[i] = ptp[max_chan_even + 4 + 2 * i] / max_ptp
        wf[:, np.arange(max_chan_even + 4, n_chan, 2)] /= regularizer

    len_reg = (max_chan_odd - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_odd - 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_odd - 4 - 2 * i])
            regularizer[len_reg - i - 1] = (
                ptp[max_chan_odd - 4 - 2 * i] / max_ptp
            )
        wf[:, np.arange(1, max_chan_odd - 2, 2)] /= regularizer

    len_reg = (n_chan - 1 - max_chan_odd - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_odd + 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_odd + 4 + 2 * i])
            regularizer[i] = ptp[max_chan_odd + 4 + 2 * i] / max_ptp
        wf[:, np.arange(max_chan_odd + 4, n_chan, 2)] /= regularizer

    return wf


def enforce_decrease_np1(waveform, max_chan=None, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    ptp = wf.ptp(0)
    if max_chan is None:
        max_chan = ptp[16:28].argmax() + 16

    max_chan_a = max_chan - max_chan % 4
    for i in range(4, max_chan_a, 4):
        if wf[:, max_chan_a - i - 4].ptp() > wf[:, max_chan_a - i].ptp():
            wf[:, max_chan_a - i - 4] = (
                wf[:, max_chan_a - i - 4]
                * wf[:, max_chan_a - i].ptp()
                / wf[:, max_chan_a - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_a - 4, 4):
        if wf[:, max_chan_a + i + 4].ptp() > wf[:, max_chan_a + i].ptp():
            wf[:, max_chan_a + i + 4] = (
                wf[:, max_chan_a + i + 4]
                * wf[:, max_chan_a + i].ptp()
                / wf[:, max_chan_a + i + 4].ptp()
            )

    max_chan_b = max_chan - max_chan % 4 + 1
    for i in range(4, max_chan_b, 4):
        if wf[:, max_chan_b - i - 4].ptp() > wf[:, max_chan_b - i].ptp():
            wf[:, max_chan_b - i - 4] = (
                wf[:, max_chan_b - i - 4]
                * wf[:, max_chan_b - i].ptp()
                / wf[:, max_chan_b - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_b - 3, 4):
        if wf[:, max_chan_b + i + 4].ptp() > wf[:, max_chan_b + i].ptp():
            wf[:, max_chan_b + i + 4] = (
                wf[:, max_chan_b + i + 4]
                * wf[:, max_chan_b + i].ptp()
                / wf[:, max_chan_b + i + 4].ptp()
            )

    max_chan_c = max_chan - max_chan % 4 + 2
    for i in range(4, max_chan_c, 4):
        if wf[:, max_chan_c - i - 4].ptp() > wf[:, max_chan_c - i].ptp():
            wf[:, max_chan_c - i - 4] = (
                wf[:, max_chan_c - i - 4]
                * wf[:, max_chan_c - i].ptp()
                / wf[:, max_chan_c - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_c - 2, 4):
        if wf[:, max_chan_c + i + 4].ptp() > wf[:, max_chan_c + i].ptp():
            wf[:, max_chan_c + i + 4] = (
                wf[:, max_chan_c + i + 4]
                * wf[:, max_chan_c + i].ptp()
                / wf[:, max_chan_c + i + 4].ptp()
            )

    max_chan_d = max_chan - max_chan % 4 + 3
    for i in range(4, max_chan_d, 4):
        if wf[:, max_chan_d - i - 4].ptp() > wf[:, max_chan_d - i].ptp():
            wf[:, max_chan_d - i - 4] = (
                wf[:, max_chan_d - i - 4]
                * wf[:, max_chan_d - i].ptp()
                / wf[:, max_chan_d - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_d - 3, 4):
        if wf[:, max_chan_d + i + 4].ptp() > wf[:, max_chan_d + i].ptp():
            wf[:, max_chan_d + i + 4] = (
                wf[:, max_chan_d + i + 4]
                * wf[:, max_chan_d + i].ptp()
                / wf[:, max_chan_d + i + 4].ptp()
            )

    return wf


def make_shell(channel, geom, n_jumps=1):
    """See make_shells"""
    pt = geom[channel]
    dists = cdist([pt], geom).ravel()
    radius = np.unique(dists)[1 : n_jumps + 1][-1]
    return np.setdiff1d(np.flatnonzero(dists <= radius + 1e-8), [channel])


def make_shells(geom, n_jumps=1):
    """Get the neighbors of a channel within a radius

    That radius is found by figuring out the distance to the closest channel,
    then the channel which is the next closest (but farther than the closest),
    etc... for n_jumps.

    So, if n_jumps is 1, it will return the indices of channels which are
    as close as the closest channel. If n_jumps is 2, it will include those
    and also the indices of the next-closest channels. And so on...

    Returns
    -------
    shell_neighbors : list
        List of length geom.shape[0] (aka, the number of channels)
        The ith entry in the list is an array with the indices of the neighbors
        of the ith channel.
        i is not included in these arrays (a channel is not in its own shell).
    """
    return [make_shell(c, geom, n_jumps=n_jumps) for c in range(geom.shape[0])]


def make_radial_order_parents(
    geom, channel_index, n_jumps_per_growth=1, n_jumps_parent=3
):
    """Pre-computes a helper data structure for enforce_decrease_shells"""
    n_channels = len(channel_index)

    # which channels should we consider as possible parents for each channel?
    shells = make_shells(geom, n_jumps=n_jumps_parent)

    radial_parents = []
    for channel, neighbors in enumerate(channel_index):
        channel_parents = []

        # the closest shell will do nothing
        already_seen = [channel]
        shell0 = make_shell(channel, geom, n_jumps=n_jumps_per_growth)
        already_seen += sorted(c for c in shell0 if c not in already_seen)

        # so we start at the second jump
        jumps = 2
        while len(already_seen) < (neighbors < n_channels).sum():
            # grow our search -- what are the next-closest channels?
            new_shell = make_shell(
                channel, geom, n_jumps=jumps * n_jumps_per_growth
            )
            new_shell = list(
                sorted(
                    c
                    for c in new_shell
                    if (c not in already_seen) and (c in neighbors)
                )
            )

            # for each new channel, find the intersection of the channels
            # from previous shells and that channel's shell in `shells`
            for new_chan in new_shell:
                parents = np.intersect1d(shells[new_chan], already_seen)
                parents_rel = np.flatnonzero(np.isin(neighbors, parents))
                if not len(parents_rel):
                    # this can happen for some strange geometries
                    # in that case, let's just bail.
                    continue
                channel_parents.append(
                    (np.flatnonzero(neighbors == new_chan).item(), parents_rel)
                )

            # add this shell to what we have seen
            already_seen += new_shell
            jumps += 1

        radial_parents.append(channel_parents)

    return radial_parents


def enforce_decrease_shells(
    waveforms, maxchans, radial_parents, in_place=False
):
    """Radial enforce decrease"""
    N, T, C = waveforms.shape
    assert maxchans.shape == (N,)

    # compute original ptps and allocate storage for decreasing ones
    is_torch = False
    if torch.is_tensor(waveforms):
        orig_ptps = (waveforms.max(dim=1).values - waveforms.min(dim=1).values).cpu().numpy()
        is_torch = True
    else:
        orig_ptps = waveforms.ptp(axis=1)
    decreasing_ptps = orig_ptps.copy()

    # loop to enforce ptp decrease
    for i in range(N):
        decr_ptp = decreasing_ptps[i]
        for c, parents_rel in radial_parents[maxchans[i]]:
            if decr_ptp[c] > decr_ptp[parents_rel].max():
                decr_ptp[c] *= decr_ptp[parents_rel].max() / decr_ptp[c]

    # apply decreasing ptps to the original waveforms
    rescale = (decreasing_ptps / orig_ptps)[:, None, :]
    if is_torch:
        rescale = torch.as_tensor(rescale, device=waveforms.device)
    if in_place:
        waveforms *= rescale
    else:
        waveforms = waveforms * rescale

    return waveforms


def enforce_temporal_decrease(
    waveforms,
    left=20,
    right=100,
    trough_offset=42,
    in_place=False,
):
    """Enforce monotonicity of abs values at the edges

    Finds the peaks to the left and right of the trough, and
    makes sure we decrease to either side of those.
    """
    N, T, C = waveforms.shape
    waveforms = waveforms.transpose(0, 2, 1).reshape(N * C, T)

    if not in_place:
        waveforms = waveforms.copy()

    if left > 0:
        # not good to do this in a data-driven way
        # because of collisions
        # left_peaks = waveforms[:, :trough_offset].argmax(1)
        left_peaks = np.full(N * C, left)
        _enforce_temporal_decrease_left(
            waveforms, left_peaks
        )

    if right is not None and (right < T):
        # right_peaks = waveforms[:, trough_offset:].argmax(1)
        right_peaks = np.full(N * C, right)
        _enforce_temporal_decrease_right(
            waveforms, right_peaks
        )

    waveforms = waveforms.reshape(N, C, T).transpose(0, 2, 1)

    return waveforms


@torch.no_grad()
def cleaned_waveforms(
    waveforms,
    spike_index,
    firstchans,
    residual,
    s_start=0,
    tpca_rank=7,
    pbar=True,
):
    N, T, C = waveforms.shape
    denoiser = SingleChanDenoiser().load()
    cleaned = np.empty((N, C, T), dtype=waveforms.dtype)
    ixs = (
        trange(len(spike_index), desc="Cleaning and denoising")
        if pbar
        else range(len(spike_index))
    )
    for ix in ixs:
        t, mc = spike_index[ix]
        fc = firstchans[ix]
        t = t - s_start

        if t + 79 > residual.shape[0]:
            raise ValueError("Spike time outside range")

        cleaned[ix] = denoiser(
            torch.as_tensor(
                (residual[t - 42 : t + 79, fc : fc + C] + waveforms[ix]).T,
                dtype=torch.float,
            )
        ).numpy()

    tpca = PCA(tpca_rank)
    cleaned = cleaned.reshape(N * C, T)
    cleaned = tpca.inverse_transform(tpca.fit_transform(cleaned))
    cleaned = cleaned.reshape(N, C, T).transpose(0, 2, 1)

    for i in range(N):
        enforce_decrease(cleaned[i], in_place=True)

    return cleaned


# %%
def denoise_wf_nn_tmp_single_channel(wf, denoiser, device):
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0] > 0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch).data
        denoised_wf = denoised_wf.reshape(n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros(
            (wf.shape[0], wf.shape[1] * wf.shape[2]), "float32"
        )

    return denoised_wf
