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
from spike_psvae import simdata, vis_utils, waveform_utils, point_source_centering

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
) = simdata.cull_templates(ctx_h5["templates"][:], geom_np2, 20, rg(), channel_radius=20)
C = choice_loc_templates.shape[-1]

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
        )
    )
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

vis_utils.cluster_scatter(xs, z_abss, units, ax=aa, alpha=0.5, do_ellipse=False)
vis_utils.cluster_scatter(ys, z_abss, units, ax=ab, alpha=0.5, do_ellipse=False)
vis_utils.cluster_scatter(alphas, z_abss, units, ax=ac, alpha=0.5, do_ellipse=False)

aa.set_ylabel("z")
aa.set_xlabel("x")
ab.set_xlabel("y")
ac.set_xlabel("alpha")

fig.suptitle("Stan sampled localizations by template ID")

plt.show()

# %%
units.shape

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

aa.set_ylabel("z")
aa.set_xlabel("x")
ab.set_xlabel("y")
ac.set_xlabel("alpha")

fig.suptitle("Stan sampled localizations by template ID")

plt.show()

# %%
lsq_ptps = np.array(
    [
        ta / np.sqrt(np.square(tx - lg[:, 0]) + np.square(tzr - lg[:, 1]) + np.square(ty))
        for lg, tx, tzr, ty, ta in zip(lgeoms, txs, tz_rels, tys, talphas)
    ]
)

# %%
lsq_ptps.shape

# %%
map_ptps = np.array(
    [
        ta / np.sqrt(np.square(tx - lg[:, 0]) + np.square(tzr - lg[:, 1]) + ty ** 2)
        for lg, tx, tzr, ty, ta in zip(lgeoms, mxs, mzrs, mys, mas)
    ]
)

# %%
vis_utils.vis_ptps(
    [ptps, lsq_ptps, map_ptps],
    ["ptp", "scipy pred", "stan map pred"],
    ["k", "purple", "green"],
);

# %%

# %%

# %%

# %%
