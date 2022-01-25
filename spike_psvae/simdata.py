import numpy as np
import scipy.linalg as la
import torch
from tqdm.auto import trange

from .localization import localize_waveforms
from .point_source_centering import ptp_fit, shift
from .waveform_utils import get_local_waveforms, temporal_align
from .denoise import SingleChanDenoiser


def svd_recon(x, rank=1):
    u, s, vh = la.svd(x, full_matrices=False)
    return u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank]


def cull_templates(
    templates, geom, n, rg=None, seed=0, channel_radius=20, geomkind="standard"
):
    if rg is None:
        rg = np.random.default_rng(seed)

    nonzero = np.flatnonzero(templates.ptp(1).ptp(1) > 0)
    templates = templates[nonzero]
    loc_templates, tmaxchans, tfirstchans = get_local_waveforms(
        templates,
        channel_radius,
        geom,
        geomkind=geomkind,
        compute_firstchans=True,
    )
    loc_ptp = loc_templates.ptp(1)
    txs, tys, tz_rels, tz_abss, talphas = localize_waveforms(
        loc_templates,
        geom,
        maxchans=tmaxchans,
        channel_radius=channel_radius,
        geomkind=geomkind,
        # jac=True,
    )
    _, loc_pred_ptp = ptp_fit(
        templates,
        geom,
        tmaxchans,
        *(txs, tys, tz_rels, talphas),
        channel_radius=channel_radius,
        geomkind=geomkind
    )

    # choose good templates by point source error, SVD error, y, and max ptp
    point_source_err = np.square(loc_ptp - loc_pred_ptp).mean(axis=1)
    svd_recons = np.array(
        [svd_recon(locwf, rank=2) for locwf in loc_templates]
    )
    svd_err = np.square(loc_templates - svd_recons).mean(axis=(1, 2))
    cull = point_source_err < np.percentile(point_source_err, 60)
    cull &= svd_err < np.percentile(svd_err, 60)
    cull &= tys > 0.2
    cull &= loc_ptp.max(1) > np.percentile(loc_ptp.max(1), 30)
    # remove templates at the edge of the probe
    cull &= np.isin(
        loc_ptp.argmax(1),
        (channel_radius, channel_radius + 1),
    )

    # choose our templates
    choice = rg.choice(np.flatnonzero(cull), size=n, replace=False)
    units = nonzero[choice]
    print("template units:", units)
    choice_loc_templates = loc_templates[choice]
    tmaxchans = tmaxchans[choice]
    tfirstchans = tfirstchans[choice]
    txs = txs[choice]
    tys = tys[choice]
    tz_rels = tz_rels[choice]
    talphas = talphas[choice]

    return (
        units,
        choice_loc_templates,
        tmaxchans,
        tfirstchans,
        txs,
        tys,
        tz_rels,
        talphas,
        point_source_err[choice],
    )


@torch.inference_mode()
def simdata(
    output_h5,
    templates,
    geom,
    noise_segment,
    # generating params
    centers="simulate",
    alpha_shape=3,
    alpha_scale=65,
    y_shape=4,
    y_scale=5,
    xlims=(-50, 82),
    z_rel_scale=15,
    x_noise=3,
    z_noise=5,
    y_noise=1.5,
    maxchan_pad=5,
    # clustering params
    n_clusters=20,
    spikes_per_cluster=100,
    # waveform params
    channel_range=(0, 100),
    template_channel_radius=20,
    output_channel_radius=8,
    geomkind="standard",
    # other
    seed=0,
):
    assert geomkind == "standard"

    rg = np.random.default_rng(seed)
    Nt, T, C = templates.shape
    noise_T = noise_segment.shape[0]
    full_C = channel_range[1] - channel_range[0]
    # rad_dif = template_channel_radius - output_channel_radius
    # dz = geom[2, 1] - geom[0, 1]

    # -- localize and cull templates
    (
        units,
        choice_loc_templates,
        tmaxchans,
        tfirstchans,
        txs,
        tys,
        tz_rels,
        talphas,
        pserrs,
    ) = cull_templates(
        templates,
        geom,
        n_clusters,
        channel_radius=template_channel_radius,
        geomkind=geomkind,
        # rg=rg,
    )

    # temporal alignment (not subpixel)
    choice_loc_templates = temporal_align(choice_loc_templates)
    loc_chans = choice_loc_templates.shape[2]

    # -- mean localization features for each cluster
    if centers == "original":
        mean_x = txs
        mean_y = tys
        mean_z_rel = tz_rels
        mean_alpha = talphas
    elif centers == "simulate":
        mean_x = rg.uniform(*xlims, size=n_clusters)
        mean_y = rg.gamma(y_shape, y_scale, size=n_clusters)
        mean_z_rel = rg.normal(scale=z_rel_scale, size=n_clusters)
        mean_alpha = rg.gamma(alpha_shape, alpha_scale, size=n_clusters)
    else:
        raise ValueError

    # pick random z offsets
    cluster_chan_offsets = rg.integers(
        0,
        full_C - 2 * (template_channel_radius + 1),
        size=n_clusters,
    )
    cluster_chan_offsets -= cluster_chan_offsets % 2

    # -- add noise to generate localizations -- clusters x n_per_cluster
    size = (spikes_per_cluster, n_clusters)
    xs = mean_x + rg.normal(scale=x_noise, size=size)
    ys = np.abs(mean_y + rg.normal(scale=y_noise, size=size))
    z_rels = mean_z_rel + rg.normal(scale=z_noise, size=size)
    # scale mixture of gamma
    alphas = rg.gamma(alpha_shape, scale=mean_alpha / alpha_shape, size=size)

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
    waveform = np.zeros((T, full_C), dtype=np.float32)
    cluster_ids = np.empty(n_clusters * spikes_per_cluster, dtype=int)
    full_shifted_templates = np.empty(
        (n_clusters * spikes_per_cluster, T, full_C),
        dtype=np.float32,
    )
    loc_shifted_templates = np.empty(
        (n_clusters * spikes_per_cluster, T, 2 * (output_channel_radius + 1)),
        dtype=np.float32,
    )
    full_noised_waveforms = np.empty(
        (n_clusters * spikes_per_cluster, T, full_C),
        dtype=np.float32,
    )
    full_denoised_waveforms = np.empty(
        (n_clusters * spikes_per_cluster, T, full_C),
        dtype=np.float32,
    )
    denoised_waveforms = np.empty(
        (n_clusters * spikes_per_cluster, T, 2 * (output_channel_radius + 1)),
        dtype=np.float32,
    )
    maxchans = np.empty(n_clusters * spikes_per_cluster, dtype=int)
    z_abss = np.empty(n_clusters * spikes_per_cluster)
    out_z_rels = np.empty(n_clusters * spikes_per_cluster)
    for c in trange(n_clusters, desc="Clusters"):
        twf = choice_loc_templates[c]
        tx = txs[c]
        tzr = tz_rels[c]
        tmc = tmaxchans[c]
        loc_tmc = twf.ptp(0).argmax()
        startchan = cluster_chan_offsets[c]
        mc0 = startchan + loc_tmc
        # print(tmc, loc_tmc)

        for j in range(spikes_per_cluster):
            # print(j, "-" * 20)
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
            # print("tzr", tzr, "z_rel", z_rel)
            # print("tx", tx, "x", x)
            shifted_mc = shifted_twf.ptp(0).argmax()

            # write to waveform
            waveform[:] = 0.0
            waveform[:, startchan : startchan + loc_chans] = shifted_twf

            # noise segment
            nt0 = rg.integers(0, noise_T - T)
            nc0 = rg.integers(0, C - full_C)
            noise = noise_segment[nt0 : nt0 + T, nc0 : nc0 + full_C]

            # denoise and extract smaller neighborhood around denoised maxchan
            noised_wf = torch.as_tensor(waveform + noise, dtype=torch.float)
            full_denoised_wf = denoiser(noised_wf.T).T.cpu().numpy()

            # re-center at new max channel
            # dmc = rad_dif + denoised_wf.ptp(0)[rad_dif + 1:-rad_dif - 1].argmax()  # noqa
            # keep template max channel
            # dmc = loc_tmc
            # shifted template max channel
            dmc = shifted_mc
            out_mc = startchan + dmc
            out_start = out_mc - out_mc % 2 - output_channel_radius
            out_end = out_mc - out_mc % 2 + output_channel_radius + 2
            # dmc -= dmc % 2
            # mcdif = dmc - loc_tmc + loc_tmc % 1
            # print(startchan + loc_tmc, out_mc, )

            # TODO: for PCA, maybe want to keep it to one side?
            # dmc -= dmc % 2
            # print(loc_shifted_templates.shape, out_end - out_start)

            # save
            ix = c * spikes_per_cluster + j
            full_shifted_templates[ix] = waveform
            loc_shifted_templates[ix] = waveform[:, out_start:out_end]
            full_noised_waveforms[ix] = noised_wf
            full_denoised_waveforms[ix] = full_denoised_wf  # noqa
            denoised_waveforms[ix] = full_denoised_wf[:, out_start:out_end]
            cluster_ids[ix] = c
            maxchans[ix] = out_mc
            # print("mc0,mc1", mc0, out_mc)
            z_abss[ix] = geom[mc0, 1] + z_rel
            out_z_rels[ix] = z_abss[ix] - geom[out_mc, 1]
            # print("z_abs", z_abss[ix], "z_rel - tzr", z_rel - tzr)
            # print("out_z_rel", out_z_rels[ix], "z mc0,mc1", geom[mc0, 1], geom[out_mc, 1])

    return (
        choice_loc_templates,
        full_shifted_templates,
        loc_shifted_templates,
        full_noised_waveforms,
        full_denoised_waveforms,
        denoised_waveforms,
        cluster_ids,
        maxchans,
        np.c_[
            xs.ravel(order="F"),
            ys.ravel(order="F"),
            out_z_rels.ravel(order="F"),
            z_abss.ravel(order="F"),
            alphas.ravel(order="F"),
        ],
        pserrs,
    )
