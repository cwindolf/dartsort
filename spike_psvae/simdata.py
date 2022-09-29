import numpy as np
import scipy.linalg as la
import torch
from tqdm.auto import trange, tqdm

from .localization import localize_waveforms
from .point_source_centering import ptp_fit, shift
# from .waveform_utils import get_local_waveforms, temporal_align
from .denoise import SingleChanDenoiser

from spike_psvae import localize_index


def svd_recon(x, rank=1):
    u, s, vh = la.svd(x, full_matrices=False)
    return u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank]


def cull_templates(
    templates,
    geom,
    n,
    y_min=0.2,
    rg=None,
    seed=0,
    channel_radius=20,
    geomkind="standard",
    pserr_pctile=50,
    svderr_pctile=50,
    ptp_pctile=0,
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
    cull = point_source_err < np.percentile(point_source_err, pserr_pctile)
    cull &= svd_err < np.percentile(svd_err, svderr_pctile)
    if y_min > 0:
        cull &= tys > y_min
    elif y_min < 0:
        cull &= tys < -y_min
    if ptp_pctile > 0:
        cull &= loc_ptp.max(1) > np.percentile(loc_ptp.max(1), ptp_pctile)
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


@torch.no_grad()
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


@torch.no_grad()
def hybrid_recording(
    input_bin,
    templates,
    geom,
    loc_channel_index,
    write_channel_index,
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
    maxchan_pad=5,
    mean_spike_rate=3,
    # clustering params
    n_clusters=40,
    # other
    do_noise=True,
    seed=0,
    rate_spread=10,
    trough_offset=42,
):
    rg = np.random.default_rng(seed)
    Nt, T, n_channels = templates.shape
    assert geom.shape == (n_channels, 2)
    write_n_channels = write_channel_index.shape[1]
    print(f"Making hybrid data based on {input_bin}. Templates shape: {Nt, T, n_channels}.")

    # choose clusters
    template_ptps = templates.ptp(1)
    template_maxchans = template_ptps.argmax(1)
    print(f"{template_maxchans.shape}")
    choices = np.flatnonzero(
        (write_n_channels // 2 < template_maxchans)
        & (template_maxchans < (n_channels - write_n_channels // 2))
    )
    choices = rg.choice(choices, replace=False, size=n_clusters)
    choices.sort()
    print(f"Choosing templates: {choices}")
    templates = templates[choices]
    template_ptps = template_ptps[choices]
    template_maxchans = template_maxchans[choices]

    # localize templates
    template_ptps_loc = template_ptps[
        np.arange(n_clusters)[:, None],
        loc_channel_index[template_maxchans],
    ]
    tx, ty, tz_rel, tz_abs, talpha = localize_index.localize_ptps_index(
        template_ptps_loc,
        geom,
        template_maxchans,
        loc_channel_index,
    )

    # pick random z offsets
    new_maxchans = rg.integers(
        (write_n_channels // 2 + 1),
        n_channels - (write_n_channels // 2 + 1),
        size=n_clusters,
    )
    new_maxchans -= new_maxchans % 4
    new_maxchans += template_maxchans % 4
    
    # load raw data
    raw = np.fromfile(input_bin, dtype=np.float32)
    raw = raw.reshape(-1, 384)
    t_total = raw.shape[0]
    t_total_s = t_total / 30000

    # pick point process params
    try:
        spike_rates = rg.uniform(low=mean_spike_rate[0], high=mean_spike_rate[1], size=n_clusters)
    except TypeError:
        spike_rates = rg.gamma(rate_spread, scale=mean_spike_rate / rate_spread, size=n_clusters)
    n_spikes = rg.poisson(lam=spike_rates * t_total_s, size=n_clusters)

    # -- generate spike train
    spike_trains = []
    spike_indices = []
    waveforms = []
    for i, unit in enumerate(tqdm(choices)):
        while True:
            spike_train = rg.choice(
                t_total - T, size=n_spikes[i], replace=False
            )
            # be refractory
            if (np.abs(np.diff(spike_train)) > 40).all():
                break
        spike_trains.append(np.c_[spike_train, [i] * n_spikes[i]])
        spike_indices.append(
            np.c_[spike_train, [new_maxchans[i]] * n_spikes[i]]
        )
        write_template0 = templates[i]
        write_template0 = write_template0[:, write_channel_index[template_maxchans[i]]]
        if do_noise:
            waveforms.append(
                [
                    shift(
                        write_template0,
                        write_channel_index[template_maxchans[i], 0],
                        template_maxchans[i],
                        geom,
                        dx=rg.normal(scale=x_noise),
                        dz=rg.normal(z_noise),
                        y1=np.abs(ty[i] + rg.normal(scale=y_noise)),
                        alpha1=rg.gamma(
                            alpha_shape, scale=talpha[i] / (alpha_shape - 1)
                        ),
                        loc0=(
                            tx[i],
                            ty[i],
                            tz_rel[i],
                            tz_abs[i],
                            talpha[i],
                        ),
                    )[0]
                    for _ in range(n_spikes[i])
                ]
            )
        else:
            waveforms.append([write_template0] * n_spikes[i])
    spike_train = np.concatenate(spike_trains, axis=0)
    spike_index = np.concatenate(spike_indices, axis=0)
    waveforms = np.concatenate(waveforms, axis=0)
    print(spike_train.shape, spike_index.shape, waveforms.shape)

    # now we np.add.at and return
    time_range = np.arange(-trough_offset, T - trough_offset)
    time_ix = spike_train[:, 0, None] + time_range[None, :]
    chan_ix = write_channel_index[spike_index[:, 1]]
    np.add.at(
        raw,
        (time_ix[:, :, None], chan_ix[:, None, :]),
        waveforms,
    )
    
    # templates at offset chans
    write_templates = np.zeros_like(templates)
    for i, wt in enumerate(templates):
        write_templates[i][:, write_channel_index[new_maxchans[i]]] = wt[:, write_channel_index[template_maxchans[i]]]

    return raw, spike_train, spike_index, waveforms, choices, write_templates, new_maxchans
