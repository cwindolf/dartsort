import numpy as np
import scipy.linalg as la
import torch
from tqdm.auto import trange

from .localization import localize_waveforms
from .point_source_centering import ptp_fit, shift
from .waveform_utils import get_local_waveforms, temporal_align
from .denoise import SingleChanDenoiser

from . import vis_utils


def svd_recon(x, rank=1):
    u, s, vh = la.svd(x, full_matrices=False)
    return u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank]


@torch.inference_mode()
def simdata(
    output_h5,
    templates,
    geom,
    noise_segment,
    # generating params
    alpha_shape=3,
    alpha_scale=65,
    y_shape=4,
    y_scale=5,
    xlims=(-50, 82),
    z_rel_scale=15,
    x_noise=3,
    z_noise=5,
    y_noise=1.5,
    # clustering params
    n_clusters=20,
    spikes_per_cluster=100,
    # waveform params
    template_channel_radius=8,
    output_channel_radius=8,
    geomkind="standard",
    # other
    seed=0,
):
    assert geomkind == "standard"
    assert template_channel_radius == output_channel_radius

    rg = np.random.default_rng(seed)
    Nt, T, C = templates.shape
    noise_T = noise_segment.shape[0]
    rad_dif = template_channel_radius - output_channel_radius

    # -- mean localization features for each cluster
    mean_x = rg.uniform(*xlims, size=n_clusters)
    mean_y = rg.gamma(y_shape, y_scale, size=n_clusters)
    mean_z_rel = rg.normal(scale=z_rel_scale, size=n_clusters)
    mean_alpha = rg.gamma(alpha_shape, alpha_scale, size=n_clusters)
    cluster_maxchans = rg.integers(
        template_channel_radius,
        C - template_channel_radius - 1,
        size=n_clusters,
    )

    # -- add noise to generate localizations -- clusters x n_per_cluster
    size = (spikes_per_cluster, n_clusters)
    xs = mean_x + rg.normal(scale=x_noise, size=size)
    ys = np.abs(mean_y + rg.normal(scale=y_noise, size=size))
    z_rels = mean_z_rel + rg.normal(scale=z_noise, size=size)
    # scale mixture of gamma
    alphas = rg.gamma(alpha_shape, scale=mean_alpha / alpha_shape, size=size)

    # -- localize and cull templates
    nonzero = np.flatnonzero(templates.ptp(1).ptp(1) > 0)
    templates = templates[nonzero]
    loc_templates, tmaxchans = get_local_waveforms(
        templates,
        template_channel_radius,
        geom,
        geomkind=geomkind,
    )
    loc_ptp = loc_templates.ptp(1)
    txs, tys, tz_rels, tz_abss, talphas = localize_waveforms(
        loc_templates,
        geom,
        maxchans=tmaxchans,
        channel_radius=template_channel_radius,
        geomkind=geomkind,
    )
    _, loc_pred_ptp = ptp_fit(
        templates,
        geom,
        tmaxchans,
        *(txs, tys, tz_rels, talphas),
        channel_radius=template_channel_radius,
        geomkind=geomkind
    )

    # choose good templates by point source error, SVD error, y, and max ptp
    ps_err = np.square(loc_ptp - loc_pred_ptp).mean(axis=1)
    svd_recons = np.array([svd_recon(locwf) for locwf in loc_templates])
    svd_err = np.square(loc_templates - svd_recons).mean(axis=(1, 2))
    cull = ps_err < np.median(ps_err)
    cull &= svd_err < np.median(svd_err)
    cull &= tys > 1
    cull &= loc_ptp.max(1) > np.percentile(loc_ptp.max(1), 25)
    # remove templates near the edge of the probe
    cull &= np.isin(
        loc_ptp.argmax(1),
        (template_channel_radius, template_channel_radius + 1),
    )

    # choose our templates
    choice = rg.choice(np.flatnonzero(cull), size=n_clusters, replace=False)
    print("units:", nonzero[choice])
    choice_loc_templates = loc_templates[choice]
    tmaxchans = tmaxchans[choice]
    txs = txs[choice]
    tz_rels = tz_rels[choice]

    # temporal alignment (not subpixel)
    choice_loc_templates = temporal_align(choice_loc_templates)
    loc_chans = choice_loc_templates.shape[2]

    # -- load denoiser
    denoiser = SingleChanDenoiser()
    denoiser.to(torch.device("cpu"))
    denoiser.load()
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # else:
    #     device = torch.device(device)
    # denoiser.to(device)

    # -- create cluster data
    cluster_ids = np.empty(n_clusters * spikes_per_cluster, dtype=int)
    noised_waveforms = np.empty(
        (n_clusters * spikes_per_cluster, T, loc_chans),
        dtype=np.float32,
    )
    denoised_waveforms = np.empty(
        (n_clusters * spikes_per_cluster, T, 2 * (output_channel_radius + 1)),
        dtype=np.float32,
    )
    maxchans = np.empty(n_clusters * spikes_per_cluster, dtype=int)
    for c in trange(n_clusters, desc="Clusters"):
        twf = choice_loc_templates[c]
        # tmc = tmaxchans[c]
        tmc = cluster_maxchans[c]
        tx = txs[c]
        tzr = tz_rels[c]
        loc_tmc = twf.ptp(0).argmax()

        for j in range(spikes_per_cluster):
            # target localizations
            x = xs[j, c]
            y = ys[j, c]
            z_rel = z_rels[j, c]
            alpha = alphas[j, c]

            # shifted template
            shifted_twf, _ = shift(
                twf,
                tmc,
                geom,
                dx=x - tx,
                dz=z_rel - tzr,
                y1=y,
                alpha1=alpha,
                channel_radius=template_channel_radius,
                geomkind=geomkind,
            )

            # noise segment
            nt0 = rg.integers(0, noise_T - T)
            nc0 = rg.integers(0, C - loc_chans)
            noise = noise_segment[nt0 : nt0 + T, nc0 : nc0 + loc_chans]

            # denoise and extract smaller neighborhood around denoised maxchan
            noised_wf = torch.as_tensor(shifted_twf + noise, dtype=torch.float)
            denoised_wf = denoiser(noised_wf.T).T.cpu().numpy()
            # dmc = rad_dif + denoised_wf.ptp(0)[rad_dif + 1:-rad_dif - 1].argmax()
            dmc = loc_tmc
            out_mc = tmc + dmc - loc_tmc
            dmc -= dmc % 2  # TODO: for PCA, maybe want to keep it to one side?
            denoised_wf = denoised_wf[
                :,
                dmc - output_channel_radius : dmc + output_channel_radius + 2,
            ]

            # save
            noised_waveforms[c * spikes_per_cluster + j] = noised_wf
            denoised_waveforms[c * spikes_per_cluster + j] = denoised_wf
            cluster_ids[c * spikes_per_cluster + j] = c
            maxchans[c * spikes_per_cluster + j] = out_mc

    return (
        choice_loc_templates,
        noised_waveforms,
        denoised_waveforms,
        cluster_ids,
        maxchans,
    )
