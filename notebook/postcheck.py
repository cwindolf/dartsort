# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import cmdstanpy
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import ujson
from IPython.display import display
import colorcet

# %%
from spike_psvae import simdata, vis_utils, waveform_utils, point_source_centering, localization, laplace

# %%
plt.rc("figure", dpi=200)

# %%
rg = lambda k=0: np.random.default_rng(k)

# %%
ctx_h5 = h5py.File("../data/ks_np2_nzy_cortex.h5", "r")
geom_np2 = ctx_h5["geom"][:]


# %%
def stanc(name, code, workdir=".stan"):
    Path(workdir).mkdir(exist_ok=True)
    path = Path(workdir) / f"{name}.stan"
    with open(path, "w") as f:
        f.write(code)
    model = cmdstanpy.CmdStanModel(stan_file=path, stanc_options={'warn-pedantic': True})
    return model

def tojson(path, **kwargs):
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        # print(k, v)
        out[k] = v
    with open(path, "w") as f:
        ujson.dump(out, f)
    return path


# %%
big_y = False
threshy = 0.5

# %%
if big_y:
    postfigdir = Path("../figs/post_bigy")
else:
    postfigdir = Path("../figs/post_smally")

postfigdir.mkdir(exist_ok=True)
postfigdir

# %%
(
    choice_units,
    choice_loc_templates,
    tmaxchans,
    tfirstchans,
    txs,
    tys,
    tz_rels,
    talphas,
    pserrs
) = simdata.cull_templates(ctx_h5["templates"][:], geom_np2, 30, y_min=threshy if big_y else -threshy, rg=rg(), channel_radius=20)
C = choice_loc_templates.shape[-1]

# %%
vis_utils.labeledmosaic(
    [choice_loc_templates[:10], choice_loc_templates[10:20], choice_loc_templates[20:]]
)
gtlt = ">" if big_y else "<"
plt.title(f"the templates in question (all have y{gtlt}{threshy})", x=-10)

# %%
lgeoms = [geom_np2[f : f + C] - np.array([[0, geom_np2[m, 1]]])  for f, m in zip(tfirstchans, tmaxchans)]

# %%
code = """
data {
    int<lower=0> C;
    vector[C] ptp;
    vector[C] gx;
    vector[C] gz;
}
parameters {
    real<lower=-100, upper=132> x;
    real<lower=0, upper=250> y;
    real<lower=-100, upper=100> z;
    real<lower=0> alpha;
}
transformed parameters {
    vector[C] pred_ptp = alpha ./ sqrt(square(gx - x) + square(gz - z) + square(y));
}
model {
    ptp - pred_ptp ~ normal(0, 1);
    // target += -sum(square(ptp - pred_ptp));
    // target += 0.1 * log(y);
}
"""

# %%
code = """
data {
    int<lower=0> C;
    vector[C] ptp;
    vector[C] gx;
    vector[C] gz;
}
parameters {
    real<lower=-100, upper=132> x;
    real<lower=0, upper=250> y;
    real<lower=-100, upper=100> z;
    real<lower=0> alpha;
}
transformed parameters {
    vector[C] pred_ptp = alpha ./ sqrt(square(gx - x) + square(gz - z) + square(y));
}
model {
    //alpha ~ gamma(3, 1./50.);
    ptp - pred_ptp ~ normal(0, 0.1);
}
"""

# %%
model = stanc("lsq", code)

# %%
xs = []; ys = []; z_rels = []; alphas = []
mxs = []; mys = []; mzrs = []; mas = []
units = []
for u in range(len(choice_units)):
    res = model.sample(
        tojson(
            f".stan/data{u}.json",
            C=C,
            ptp=choice_loc_templates[u].ptp(0),
            gx=lgeoms[u][:, 0],
            gz=lgeoms[u][:, 1],
        ),
    )
    print("unit", choice_units[u])
    display(res.summary().loc[["lp__", "x", "y", "z", "alpha"]])
    xs.append(res.stan_variable("x"))
    mxs.append(xs[-1].mean())
    ys.append(res.stan_variable("y"))    
    mys.append(ys[-1].mean())
    z_rels.append(res.stan_variable("z"))   
    mzrs.append(z_rels[-1].mean())
    alphas.append(res.stan_variable("alpha"))   
    mas.append(alphas[-1].mean())
    units.append(np.full(xs[-1].shape, u))
    



# %%
xs = np.hstack(xs)
ys = np.hstack(ys)
z_rels = np.hstack(z_rels)
alphas = np.hstack(alphas)
units = np.hstack(units)
z_abss = geom_np2[tmaxchans[units], 1] + z_rels

# %%
tz_abss = tz_rels + geom_np2[tmaxchans, 1]

# %%
fig, (aa, ab, ac) = plt.subplots(1, 3, figsize=(6, 4), sharey=True)

vis_utils.cluster_scatter(xs, z_abss, units, ax=aa, alpha=0.01)
vis_utils.cluster_scatter(ys, z_abss, units, ax=ab, alpha=0.01)
vis_utils.cluster_scatter(alphas, z_abss, units, ax=ac, alpha=0.01)

cc = np.array(colorcet.glasbey_hv)
aa.scatter(txs, tz_abss, color="w", s=5)
ab.scatter(tys, tz_abss, color="w", s=5)
ac.scatter(talphas, tz_abss, color="w", s=5)
aa.scatter(txs, tz_abss, color="k", s=3)
ab.scatter(tys, tz_abss, color="k", s=3)
ac.scatter(talphas, tz_abss, color="k", s=3)
aa.scatter(txs, tz_abss, color=cc[np.unique(units)], s=1)
ab.scatter(tys, tz_abss, color=cc[np.unique(units)], s=1)
ac.scatter(talphas, tz_abss, color=cc[np.unique(units)], s=1)

ab.scatter(mys, tz_abss, color="w", marker="+", s=8)
ab.scatter(mys, tz_abss, color="w", marker="+", s=8)
ab.scatter(mys, tz_abss, marker="+", s=2, color=cc[np.unique(units)])
ab.scatter(mys, tz_abss, marker="+", s=2, color=cc[np.unique(units)])


aa.set_ylabel("z")
aa.set_xlabel("x")
ab.set_xlabel("y")
ac.set_xlabel("alpha")

fig.suptitle("Stan sampled localizations by template ID")

plt.show()

# %%
vis_utils.traceplot(choice_loc_templates[0])

# %% tags=[] jupyter={"outputs_hidden": true}
# mos = """\
# xab
# xcd
# """
mos = "xabc"
sckw = dict(s=2, color="k", alpha=0.1)
for u in range(len(choice_loc_templates)):
    fig, axes = plt.subplot_mosaic(mos, figsize=(8, 3))
    axes["x"].imshow(choice_loc_templates[u][20:-20])
    axes["a"].scatter(ys[units == u], xs[units == u], **sckw)
    axes["b"].scatter(ys[units == u], z_rels[units == u], **sckw) 
    axes["c"].scatter(ys[units == u], alphas[units == u], **sckw)  
    
    for k in "abc":
        axes[k].set_aspect("equal")
    
    s=6
    axes["a"].scatter([tys[u]], [txs[u]], c="crimson", s=s)
    axes["b"].scatter([tys[u]], [tz_rels[u]], c="crimson", s=s)
    ha = axes["c"].scatter([tys[u]], [talphas[u]], c="crimson", s=s, label="LS")
    
    axes["a"].scatter([mys[u]], [mxs[u]], c="limegreen", s=s)
    axes["b"].scatter([mys[u]], [mzrs[u]], c="limegreen", s=s)
    hb = axes["c"].scatter([mys[u]], [mas[u]], c="limegreen", s=s, label="Post. mean")
    
#     axes["c"].legend(loc="upper center", bbox_to_anchor=(1, 1))
    fig.legend([ha, hb], ["LS", "Stan"])
    
    axes["a"].set_ylabel("x")
    axes["b"].set_ylabel("z")
    axes["c"].set_ylabel("alpha")
    axes["b"].set_xlabel("y")
    
    locstr = tuple(float(f"{q:0.1f}") for q in (txs[u],tys[u],tz_rels[u],talphas[u]))
    fig.suptitle(f"KS unit {choice_units[u]} posterior. LS loc (x,y,z,a)={locstr}")
    fig.tight_layout(pad=0.5)
    
    fig.savefig(postfigdir / f"u{choice_units[u]}.png")

# %%
import matplotlib

# %%
choice_loc_templates.shape

# %%
plt.rc("figure", dpi=300)

# %%
# mos = """\
# xab
# xcd
# """
# mos = "xabc"
# sckw = dict(s=2, color="k", alpha=0.1)
fig, axes = plt.subplots(4, 3, figsize=(6, 4), gridspec_kw=dict(height_ratios=[2.5, 3, 3, 3], hspace=0.15, wspace=0.3))
axes = axes.T

for i, (u, c) in enumerate(zip([29, 1, 26], ["y", "orange", "crimson"])):
    axes[i, 0].imshow(choice_loc_templates[u, 20:82])
    axes[i, 0].set_yticks([])
    axes[i, 0].set_xticks([])
    pos = axes[i, 0].get_position()
    pos.y0 = pos.y0 - 0.025
    axes[i, 0].set_position(pos)
    
    axes[i,1+ 0].scatter(ys[units == u], xs[units == u], s=0.1, alpha=0.2, color="k")
    axes[i,1+ 1].scatter(ys[units == u], z_rels[units == u], s=0.1, alpha=0.2, color="k") 
    hc = axes[i,1+ 2].scatter(ys[units == u], alphas[units == u], s=0.1, alpha=0.2, color="k")  
    
    # for k in "abc":
    #     axes[k].set_aspect("equal")
    print(u)
    
    axes[i,1+ 0].scatter([tys[u]], [txs[u]], c="blue", s=s)
    axes[i,1+ 1].scatter([tys[u]], [tz_rels[u]], c="blue", s=s)
    ha = axes[i,1+ 2].scatter([tys[u]], [talphas[u]], c="blue", s=s, label="LS")
    
    axes[i,1+ 0].scatter([mys[u]], [mxs[u]], c="limegreen", s=s)
    axes[i,1+ 1].scatter([mys[u]], [mzrs[u]], c="limegreen", s=s)
    hb = axes[i,1+ 2].scatter([mys[u]], [mas[u]], c="limegreen", s=s, label="Post. mean")
    
#     axes["c"].legend(loc="upper center", bbox_to_anchor=(1, 1))
hc = matplotlib.lines.Line2D([0], [0], marker='o', color='k', markersize=1, linestyle="None")
plt.figlegend(
    handles=[hc, ha, hb],
    labels=["Stan samples", "Mode", "Post. mean"],
    loc=[0.1, 0.0],
    frameon=False,
    fancybox=False,
    borderpad=0,
    borderaxespad=0,
    ncol=3,
    
)

for ax in axes[:, 1:].flat:
    ax.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5]))

for ax in axes[:, :-1].flat:
    ax.set_xticks([])
    
axes[0, 1+0].set_ylabel("$x$")
axes[0, 1+1].set_ylabel("$z$")
axes[0, 1+2].set_ylabel("$\\alpha$")
for ax in axes[:, -1]:
    ax.set_xlabel("$y$")

for ax, u in zip(axes[:,0], [29, 1, 26]):
    ax.set_title(f"template {choice_units[u]}")

    # locstr = tuple(float(f"{q:0.1f}") for q in (txs[u],tys[u],tz_rels[u],talphas[u]))
    # fig.suptitle(f"KS unit {choice_units[u]} posterior. LS loc (x,y,z,a)={locstr}")
# fig.tight_layout()
    
    # fig.savefig(postfigdir / f"u{choice_units[u]}.png")

# %%
from spike_psvae import posterior

# %%
# mos = """\
# xab
# xcd
# """
# mos = "xabc"
# sckw = dict(s=2, color="k", alpha=0.1)
fig, axes = plt.subplots(4, 3, figsize=(6, 4), gridspec_kw=dict(height_ratios=[2.5, 3, 3, 3], hspace=0.15, wspace=0.3))
axes = axes.T

for i, (u, c) in enumerate(zip([29, 1, 26], ["y", "orange", "crimson"])):
    axes[i, 0].imshow(choice_loc_templates[u, 20:82])
    axes[i, 0].set_yticks([])
    axes[i, 0].set_xticks([])
    pos = axes[i, 0].get_position()
    pos.y0 = pos.y0 - 0.025
    axes[i, 0].set_position(pos)
    
    ptp = choice_loc_templates[u].ptp(0)
    summary, x, y, z, alpha = posterior.sample(ptp, lgeoms[u], logbarrier=True)
    
    axes[i,1+ 0].scatter(y, x, s=0.1, alpha=0.2, color="k")
    axes[i,1+ 1].scatter(y, z, s=0.1, alpha=0.2, color="k") 
    hc = axes[i,1+ 2].scatter(y, alpha, s=0.1, alpha=0.2, color="k")  
    
    # for k in "abc":
    #     axes[k].set_aspect("equal")
    print(u)
    
#     axes[i,1+ 0].scatter([tys[u]], [txs[u]], c="blue", s=s)
#     axes[i,1+ 1].scatter([tys[u]], [tz_rels[u]], c="blue", s=s)
#     ha = axes[i,1+ 2].scatter([tys[u]], [talphas[u]], c="blue", s=s, label="LS")
    
#     axes[i,1+ 0].scatter([mys[u]], [mxs[u]], c="limegreen", s=s)
#     axes[i,1+ 1].scatter([mys[u]], [mzrs[u]], c="limegreen", s=s)
#     hb = axes[i,1+ 2].scatter([mys[u]], [mas[u]], c="limegreen", s=s, label="Post. mean")
    
#     axes["c"].legend(loc="upper center", bbox_to_anchor=(1, 1))
# hc = matplotlib.lines.Line2D([0], [0], marker='o', color='k', markersize=1, linestyle="None")
# plt.figlegend(
#     handles=[hc, ha, hb],
#     labels=["Stan samples", "Mode", "Post. mean"],
#     loc=[0.1, 0.025],
#     frameon=False,
#     fancybox=False,
#     borderpad=0,
#     borderaxespad=0,
#     ncol=3,
    
# )

for ax in axes[:, 1:].flat:
    ax.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5]))

for ax in axes[:, :-1].flat:
    ax.set_xticks([])
    
axes[0, 1+0].set_ylabel("x")
axes[0, 1+1].set_ylabel("z")
axes[0, 1+2].set_ylabel("alpha")

for ax, u in zip(axes[:,0], [29, 1, 26]):
    ax.set_title(f"template {choice_units[u]}")

    # locstr = tuple(float(f"{q:0.1f}") for q in (txs[u],tys[u],tz_rels[u],talphas[u]))
    # fig.suptitle(f"KS unit {choice_units[u]} posterior. LS loc (x,y,z,a)={locstr}")
# fig.tight_layout()
    
    # fig.savefig(postfigdir / f"u{choice_units[u]}.png")

# %%

# %%

# %% [markdown]
# # big_y

# %%
# mos = """\
# xab
# xcd
# """
mos = "xabc"
sckw = dict(s=2, color="k", alpha=0.1)
for u in range(len(choice_loc_templates)):
    ptp = choice_loc_templates[u].ptp(0)
    lap_xs, lap_ys = laplace.laplace_approx_samples_polar(txs[u], tys[u], tz_rels[u], talphas[u], ptp, lgeoms[u])
    olap_xs, olap_ys = laplace.laplace_approx_samples(txs[u], tys[u], tz_rels[u], talphas[u], ptp, lgeoms[u])
    
    fig, (a0, aa, ab) = plt.subplots(1, 3)
    
    a0.imshow(choice_loc_templates[u][20:-20])
    
    xx, yy = np.meshgrid(
        np.linspace(xs[units == u].min(), xs[units == u].max(), num=100),
        np.linspace(ys[units == u].min(), ys[units == u].max(), num=100),
        indexing="ij",
    )
    post = np.zeros_like(xx)
    for i in range(100):
        for j in range(100):
            post[i, j] = laplace.point_source_post_lpdf(xx[i, j], yy[i, j], tz_rels[u], talphas[u], ptp, lgeoms[u], ptp_sigma=0.1)
    aa.contourf(xx, yy, post, levels=30)
    
    
    aa.scatter(xs[units == u], ys[units == u], s=2, alpha=0.1)
    aa.scatter([txs[u]], [tys[u]], c="k", s=5)
    aa.scatter([mxs[u]], [mys[u]], c="r", s=5)
    aa.scatter([lap_xs.mean()], [lap_ys.mean()], c="g", label="polar Laplace", s=5)
    aa.scatter([olap_xs.mean()], [olap_ys.mean()], c="g", marker="x", label="orig. Laplace", s=5)
    
    
    aa.set_title("stan samples", size=8)
    ab.set_title("polar laplace samples", size=8)
    aa.set_ylabel("y")
    aa.set_xlabel("x")
    ab.set_xlabel("x")
    
    ab.scatter(lap_xs, lap_ys, s=1, alpha=0.1, label="samples")
    ab.scatter([txs[u]], [tys[u]], c="k", label="LS", s=5)
    ab.scatter([mxs[u]], [mys[u]], c="r", label="Stan", s=5)
    ab.scatter([lap_xs.mean()], [lap_ys.mean()], c="g", label="polar Laplace", s=5)
    ab.scatter([olap_xs.mean()], [olap_ys.mean()], c="g", marker="x", label="orig. Laplace", s=5)
    
    plt.suptitle(f"unit {choice_units[u]}")
    
    ab.legend()
    plt.show()


# %%
_, bins, _ = plt.hist(ys, bins=128, density=True);
plt.ylabel("density")
plt.xlabel("posterior y samples (from all units, 20 templates with y<0.5)")
plt.show()
plt.hist(tys, bins=bins, label="LS");
plt.hist(mys, bins=bins, label="Stan");
plt.title("LS vs. mean AP")
plt.ylabel("frequency")
plt.xlabel("y")
plt.legend()
plt.show()

# %%
lsq_ptps = np.array(
    [
        ta / np.sqrt(np.square(tx - lg[:, 0]) + np.square(tzr - lg[:, 1]) + np.square(ty))
        for lg, tx, tzr, ty, ta in zip(lgeoms, txs, tz_rels, tys, talphas)
    ]
)

# %%
map_ptps = np.array(
    [
        ta / np.sqrt(np.square(tx - lg[:, 0]) + np.square(tzr - lg[:, 1]) + ty ** 2)
        for lg, tx, tzr, ty, ta in zip(lgeoms, mxs, mzrs, mys, mas)
    ]
)

# %%
np.abs(lsq_ptps - map_ptps).min()

# %%
vis_utils.vis_ptps(
    [choice_loc_templates.ptp(1), lsq_ptps, map_ptps],
    ["ptp", "scipy pred", "stan map pred"],
    ["k", "purple", "green"],
    subplots_kwargs=dict(figsize=(4,4), sharex=True, sharey=True)
);

# %%
fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, figsize=(6, 6))

s=4
aa.scatter(txs, mxs, s=s)
ab.scatter(tys, mys, s=s)
ac.scatter(tz_rels, mzrs, s=s)
ad.scatter(talphas, mas, s=s)


for ax in (aa, ab, ac, ad):
    # ax.plot(ax.get_xlim(), ax.get_ylim(), c="w", lw=3)
    xa, xb = xlim = np.array(ax.get_xlim())
    ya, yb = ylim = np.array(ax.get_ylim())
    ax.plot((xa - 100, xb + 100), (xa - 100, xb + 100), c="k", lw=1)
    ax.set_xlim([min(xa, ya), max(xb, yb)])
    ax.set_ylim([min(xa, ya), max(xb, yb)])  

aa.set_title("x")
ab.set_title("y")
ac.set_title("z")
ad.set_title("alpha")

fig.suptitle("Stan means (y axes) vs LS est (x axes)")

plt.show()

# %%
np.square(tz_rels - mzrs).mean()

# %%

# %% tags=[]
# JB experiment
target_y = 1

for u in range(len(choice_units)):
    # shift template to nearest electrode
    ptp = choice_loc_templates[u].ptp(0)
    x, y, z_rel, z_abs, alpha = localization.localize_ptp(ptp, tmaxchans[u], geom_np2, geomkind="standard")
    print(y)
    dx = -x if x < 16 else 32 - x
    shifted_wf, target_ptp = point_source_centering.shift(choice_loc_templates[u], tmaxchans[u], geom_np2, dx=dx + 0.01, dz=-z_rel + 0.01, y1=target_y, channel_radius=C // 2 - 1, geomkind="standard")
    shifted_ptp = shifted_wf.ptp(0)
    # max chan should not change
    assert shifted_ptp.argmax() == ptp.argmax()
    # get LS for shifted
    lsx, lsy, lszr, lsza, lsa = localization.localize_ptp(shifted_ptp, tmaxchans[u], geom_np2, geomkind="standard")
    
    # sample posterior for shifted
    res = model.sample(
        tojson(
            f".stan/data{u}shifted.json",
            C=C,
            ptp=shifted_ptp,
            gx=lgeoms[u][:, 0],
            gz=lgeoms[u][:, 1],
        )
    )
    display(res.summary().loc[["lp__", "x", "y", "z", "alpha"]])
    
    fig, (aa, ab, ac) = plt.subplots(1, 3)
    
    aa.scatter(res.stan_variable("z"), res.stan_variable("x"), s=2, alpha=0.1, c="k")
    o = ab.scatter(res.stan_variable("z"), res.stan_variable("y"), s=2, alpha=0.1, c="k")
    ac.scatter(res.stan_variable("z"), res.stan_variable("alpha"), s=2, alpha=0.1, c="k")
    
    aa.scatter([0], [x + dx])
    t = ab.scatter([0], [target_y])
    ac.scatter([0], [alpha])
    
    bs = "big" if big_y else "small"
    
    plt.figlegend(
        handles=[o, t],
        labels=["samples", "shift target"],
        loc="lower center",
        frameon=False,
        fancybox=False,
        borderpad=0,
        borderaxespad=0,
        ncol=2,
    )
    
    for ax in (aa, ab, ac):
        ax.set_ylabel("z")
    aa.set_xlabel("x")
    ab.set_xlabel("y")
    ac.set_xlabel("alpha")

    fig.suptitle(f"{bs} y template (unit {choice_units[u]}) -- shifted to y={target_y:0.2f}")
    plt.tight_layout(pad=0.5)
    plt.show()



# %%
# JB experiment
target_y = 1

for u in range(len(choice_units)):
    # shift template to nearest electrode
    ptp = choice_loc_templates[u].ptp(0)
    x, y, z_rel, z_abs, alpha = localization.localize_ptp(ptp, tmaxchans[u], geom_np2, geomkind="standard")
    print(y)
    dx = -x if x < 16 else 32 - x
    shifted_wf, target_ptp = point_source_centering.shift(choice_loc_templates[u], tmaxchans[u], geom_np2, dx=dx + 0.01, dz=-z_rel + 0.01, y1=target_y, channel_radius=8, geomkind="standard")
    shifted_ptp = shifted_wf.ptp(0)
    # max chan should not change
    assert shifted_ptp.argmax() == ptp.argmax()
    # get LS for shifted
    lsx, lsy, lszr, lsza, lsa = localization.localize_ptp(shifted_ptp, tmaxchans[u], geom_np2, geomkind="standard")
    
    # sample posterior for shifted
    res = model.sample(
        tojson(
            f".stan/data{u}shifted.json",
            C=C,
            ptp=ptp,
            gx=lgeoms[u][:, 0],
            gz=lgeoms[u][:, 1],
        )
    )
    display(res.summary().loc[["lp__", "x", "y", "z", "alpha"]])
    
    fig, (aa, ab, ac) = plt.subplots(1, 3)
    
    aa.scatter(res.stan_variable("z"), res.stan_variable("x"), s=2, alpha=0.1, c="k")
    o = ab.scatter(res.stan_variable("z"), res.stan_variable("y"), s=2, alpha=0.1, c="k")
    ac.scatter(res.stan_variable("z"), res.stan_variable("alpha"), s=2, alpha=0.1, c="k")
    
    aa.scatter([z_rel], [x], c="r")
    t = ab.scatter([z_rel], [y], c="r")
    ac.scatter([z_rel], [alpha], c="r")
    
    bs = "big" if big_y else "small"
    
    plt.figlegend(
        handles=[o, t],
        labels=["samples", "LS"],
        loc="lower center",
        frameon=False,
        fancybox=False,
        borderpad=0,
        borderaxespad=0,
        ncol=2,
    )
    
    for ax in (aa, ab, ac):
        ax.set_ylabel("z")
    aa.set_xlabel("x")
    ab.set_xlabel("y")
    ac.set_xlabel("alpha")

    fig.suptitle(f"{bs} y template (unit {choice_units[u]}) -- not shifted")
    plt.tight_layout(pad=0.5)
    plt.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
