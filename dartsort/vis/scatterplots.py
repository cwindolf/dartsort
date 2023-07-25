import h5py
import numpy as np
import matplotlib.pyplot as plt


def scatter_time_vs_depth(
    times_s=None,
    depths_um=None,
    amplitudes=None,
    hdf5_filename=None,
    geom=None,
    depth_margin_um=100,
    ax=None,
    max_spikes_plot=500_000,
    vmax=15,
    random_seed=0,
    s=1,
    linewidth=0,
    **scatter_kw,
):
    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            times_s = h5["times"][:] / h5["sampling_frequency"][()]
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]
        
    assert times_s.shape == depths_um.shape == amplitudes.shape
    assert times_s.ndim == 1

    if ax is None:
        ax = plt.gca()
    
    # subset spikes according to margin and sampling
    n_spikes = len(times_s)
    to_show = np.arange(n_spikes)
    if geom is not None:
        to_show = np.flatnonzero(
            (depths_um > geom[:, 1].min() - depth_margin_um)
            & (depths_um < geom[:, 1].max() + depth_margin_um)
        )
    if len(to_show) > max_spikes_plot:
        rg = np.random.default_rng(random_seed)
        to_show = rg.choice(to_show, size=max_spikes_plot, replace=False)
    
    # order by amplitude so that high amplitude units show up
    to_show = to_show[np.argsort(amplitudes[to_show])]
    
    return ax.scatter(
        times_s[to_show],
        depths_um[to_show],
        c=np.clip(amplitudes[to_show], 0, vmax),
        s=s,
        linewidth=linewidth,
        **scatter_kw,
    )
        