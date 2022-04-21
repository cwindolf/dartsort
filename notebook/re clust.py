# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
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
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from tqdm.auto import tqdm

# %%
from spike_psvae import subtract, vis_utils, ibme, triage, cluster, cluster_viz

# %%
# from spike_psvae.cluster_viz import plot_array_scatter, plot_self_agreement, plot_single_unit_summary
from spike_psvae.cluster_viz_index import array_scatter, single_unit_summary, plot_agreement_venn

# %%
plt.rc("figure", dpi=200)

# %%
snip_dir = Path("/mnt/3TB/charlie/re_snips/")

# %%
list(snip_dir.glob("*.ap.bin"))

# %%
# %ll -h {{snip_dir}}

# %%
res_dir = Path("/mnt/3TB/charlie/re_snip_res/")
res_dir.mkdir(exist_ok=True)

# %%
fig_dir = Path("/mnt/3TB/charlie/re_snip_fig/")
fig_dir.mkdir(exist_ok=True)

# %%
for f in snip_dir.glob("*.ap.bin"):
    print(f)
    out_dir = res_dir / (f.stem.split("_snip")[0] + "_nn4_radial")
    subtract.subtraction(
        f,
        out_dir,
        n_sec_pca=20,
        thresholds=[10, 8, 6, 5, 4],
        nn_detect=True,
        neighborhood_kind='box',
        enforce_decrease_kind='radial',
        n_jobs=2,
    )

# %% tags=[]
for f in sorted(snip_dir.glob("*.ap.bin")):
    print(f)
    out_dir = res_dir / (f.stem.split("_snip")[0] + "_fc")
    subtract.subtraction(
        f,
        out_dir,
        n_sec_pca=20,
        thresholds=[10, 8, 6, 5, 4],
        # nn_detect=True,
        # neighborhood_kind='box',
        # enforce_decrease_kind='radial',
        n_jobs=2,
    )

# %%
for f in sorted(snip_dir.glob("*.ap.bin")):
    print(f)
    out_dir = res_dir / (f.stem.split("_snip")[0] + "_dnd")
    subtract.subtraction(
        f,
        out_dir,
        n_sec_pca=20,
        thresholds=[12, 10, 8, 6, 5, 4],
        # nn_detect=True,
        denoise_detect=True,
        neighborhood_kind='box',
        enforce_decrease_kind='radial',
        n_jobs=2,
    )

# %%
for f in sorted(snip_dir.glob("*.ap.bin")):
    print(f)
    out_dir = res_dir / (f.stem.split("_snip")[0] + "_dnd_noradial")
    subtract.subtraction(
        f,
        out_dir,
        n_sec_pca=20,
        thresholds=[12, 10, 8, 6, 5, 4],
        # nn_detect=True,
        denoise_detect=True,
        # neighborhood_kind='box',
        # enforce_decrease_kind='radial',
        n_jobs=2,
    )

# %% tags=[]
for ds in sorted(res_dir.glob("*")):

    h5 = next(ds.glob("*.h5"))
    stem = ds.stem.split(".")[0]
    print(stem, h5)
    
    with h5py.File(h5, "r+") as d:
        
        if "z_reg" in d:
            continue
            
        x, y, z, alpha = d["localizations"][:, :4].T
        maxptp = d["maxptps"][:]
        geom = d["geom"][:]
        samples = d["spike_index"][:, 0]
        
        z_reg, dispmap = ibme.register_nonrigid(
            maxptp,
            z,
            samples / 30000,
            robust_sigma=1,
            rigid_disp=200,
            disp=100,
            denoise_sigma=0.1,
            n_windows=[5, 10, 20],
            widthmul=0.5,
        )
        z_reg -= (z_reg - z).mean()
        dispmap -= dispmap.mean()

        d.create_dataset("z_reg", data=z_reg)
        d.create_dataset("dispmap", data=dispmap)   


# %% tags=[]
for ds in sorted(res_dir.glob("*")):

    h5 = next(ds.glob("*.h5"))
    stem = ds.stem.split(".")[0]
    print(stem, h5)
    
    with h5py.File(h5) as d:
        x, y, z, alpha = d["localizations"][:, :4].T
        maxptp = d["maxptps"][:]
        geom = d["geom"][:]
        z_reg = d["z_reg"][:]
        samples = d["spike_index"][:, 0]
        
    r0, *_ = ibme.fast_raster(maxptp, z, samples / 30000)
    r1, *_ = ibme.fast_raster(maxptp, z, samples / 30000)
    
    fig, (aa, ab) = plt.subplots(2, 1, figsize=(4, 3), sharex=True)
    aa.imshow(r0, aspect=0.5/np.divide(*r0.shape), vmin=0, vmax=np.percentile(r0, 98), interpolation="nearest")
    ab.imshow(r1, aspect=0.5/np.divide(*r1.shape), vmin=0, vmax=np.percentile(r1, 98), interpolation="nearest")

# %% tags=[]
for ds in sorted(res_dir.glob("*")):

    h5 = next(ds.glob("*.h5"))
    stem = ds.stem.split(".")[0]
    print(stem, h5)
    
    with h5py.File(h5) as d:
        x, y, z, alpha = d["localizations"][:, :4].T
        maxptp = d["maxptps"][:]
        geom = d["geom"][:]
        z_reg = d["z_reg"][:]
    
    (fig_dir / stem).mkdir(exist_ok=True)
    
    fig = vis_utils.plotlocs(x, y, z_reg, alpha, maxptp, geom, ylim=[-1, 6])
    plt.suptitle(f"{stem} {len(x)} spikes")
    plt.savefig(fig_dir / stem / "a_scatter_full.png")
    plt.close(fig)
    
    fig = vis_utils.plotlocs(x, y, z_reg, alpha, maxptp, geom, which=z_reg > 2800, ylim=[-1, 6])
    plt.savefig(fig_dir / stem / "a_scatter_1.png")
    plt.close(fig)
    
    fig = vis_utils.plotlocs(x, y, z_reg, alpha, maxptp, geom, which=(z_reg < 2800) & (z_reg > 1900), ylim=[-1, 6])
    plt.savefig(fig_dir / stem / "a_scatter_2.png")
    plt.close(fig)
    
    fig = vis_utils.plotlocs(x, y, z_reg, alpha, maxptp, geom, which=z_reg < 1900, ylim=[-1, 6])
    plt.savefig(fig_dir / stem / "a_scatter_3.png")
    plt.close(fig)

# %%
clusterers = {}
keepers = {}

for ds in sorted(res_dir.glob("*")):

    h5 = next(ds.glob("*.h5"))
    stem = ds.stem.split(".")[0]
    print(stem, h5)
    
    with h5py.File(h5) as d:
        x, y, z, alpha = d["localizations"][:, :4].T
        maxptps = d["maxptps"][:]
        geom = d["geom"][:]
        z_reg = d["z_reg"][:]
        print(len(x), "spikes")
    
    # triage
    if "_fc" in stem:
        # do triaging
        tx, ty, tz, talpha, tmaxptps, _, ptp_keep, idx_keep  = triage.run_weighted_triage(x, y, z_reg, alpha, maxptps, threshold=75)
        idx_keep = ptp_keep[idx_keep]
    elif "_nn" in stem:
        # no triaging
        tx, ty, tz, talpha, tmaxptps = x, y, z, alpha, maxptps
        idx_keep = slice(None)
    elif "_dnd" in stem:
        # do triaging
        tx, ty, tz, talpha, tmaxptps, _, ptp_keep, idx_keep  = triage.run_weighted_triage(x, y, z_reg, alpha, maxptps, threshold=75)
        idx_keep = ptp_keep[idx_keep]
    else:
        assert False
        
    # cluster
    clusterer = cluster.cluster_hdbscan(np.c_[tx, tz, np.log(tmaxptps) * 30])
    labels = np.full(x.shape, -1)
    labels[idx_keep] = clusterer.labels_
    
    clusterers[stem] = clusterer
    keepers[stem] = idx_keep
    
    with h5py.File(h5, "r+") as d:
        if "labels" in d:
            del d["labels"]
        d.create_dataset("labels", data=labels)

# %%
for ds in sorted(res_dir.glob("*")):

    h5 = next(ds.glob("*.h5"))
    stem = ds.stem.split(".")[0]
    print(stem, h5)
    
    with h5py.File(h5) as d:
        x, y, z, alpha = d["localizations"][:, :4].T
        maxptps = d["maxptps"][:]
        geom = d["geom"][:]
        z_reg = d["z_reg"][:]
        labels = d["labels"][:]
        
    plt.figure()
    plt.hist(maxptps, bins=128)
    plt.show()
    plt.close("all")
    
    fig = array_scatter(labels, geom, x, z_reg, maxptps)
    plt.suptitle(f"{stem} {len(x)} spikes")
    plt.savefig(fig_dir / stem / "full_cluster.png", dpi=200)
    plt.show()
    plt.close(fig)

# %%
# 

# %%
plt.close("all")

# %%

# %%
from concurrent.futures import ProcessPoolExecutor

# %% tags=[]
# # %%prun -D prof.prof

for ds in sorted(res_dir.glob("*")):
    print("-" * 80)
    h5 = next(ds.glob("*.h5"))
    res_bin = next(ds.glob("residual*.bin"))
    
    stem = ds.stem.split(".")[0]
    print(stem, h5)
    
    if "_nn" in stem:
        clstem = stem.split("_nn")[0]
    if "_fc" in stem:
        clstem = stem.split("_fc")[0]
    if "_dnd" in stem:
        clstem = stem.split("_dnd")[0]
    
    raw_bin = next(snip_dir.glob(f"{clstem}*.bin"))
    print(res_bin, raw_bin)
    
    
    # (fig_dir / stem / "singleunit").mkdir(exist_ok=True)
    # for figpath in (fig_dir / stem / "singleunit").glob("*"):
    #     figpath.unlink()
    
    # d = 
    
    # tx = x[idx_keep]
    # ty = y[idx_keep]
    # tz = z_reg[idx_keep]
    # tmaxptp = maxptps[idx_keep]
    # tspind = spike_index[idx_keep]
    # tsub = subwf[idx_keep]
    # tdn2 = cwf[idx_keep]
    
    if stem not in ("CSH_ZAD_026_dnd", "CSH_ZAD_026_fc"):
        continue
    
    clusterer = clusterers[stem]
    idx_keep = keepers[stem]
    
    def job(unit):
        if (fig_dir / stem / "singleunit" / f"unit_{unit:03d}.png").exists():
            return
        x, y, z, a = job.xyza
        ci = job.ci
        maxptps = job.maxptps
        geom = job.geom
        z_reg = job.z_reg
        labels = job.labels
        spike_index = job.spike_index
        subwf = job.subwf
        cwf = job.cwf
        # try:
        fig = single_unit_summary(
            unit,
            clusterer,
            labels,
            geom,
            idx_keep,
            x,
            z,
            maxptps,
            ci,
            spike_index,
            cwf,
            subwf,
            raw_bin,
            res_bin,
        )
        fig.savefig(fig_dir / stem / "singleunit" / f"unit_{unit:03d}.png", transparent=False, pad_inches=0)
        plt.close(fig)
        # except ValueError:
            # pass
            
    def pinit(h5):
        d = h5py.File(h5, "r")
        job.xyza = d["localizations"][:, :4].T
        job.ci = d["channel_index"][:]
        job.maxptps = d["maxptps"][:]
        job.geom = d["geom"][:]
        job.z_reg = d["z_reg"][:]
        job.labels = d["labels"][:]
        job.spike_index = d["spike_index"][:]
        job.subwf = d["subtracted_waveforms"]
        job.cwf = d["cleaned_waveforms"]
    
    i = 0
    with ProcessPoolExecutor(
        12,
        initializer=pinit,
        initargs=(h5,)
    ) as p:
        units = np.setdiff1d(np.unique(clusterer.labels_), [-1])
        for res in tqdm(p.map(job, units), total=len(units)):
            pass


# %%

# %%

# %%
raw_bin

# %%
snip_dir

# %%
from joblib import Parallel, delayed

# %%
pairs = [
    ("CSH_ZAD_026", "CSH_ZAD_026_dnd", "CSH_ZAD_026_fc"),
    ("CSH_ZAD_026", "CSH_ZAD_026_fc", "CSH_ZAD_026_dnd"),
]

with h5py.File(next((res_dir / "CSH_ZAD_026_fc").glob("*.h5"))) as f:
    ci = f["channel_index"][:]

for pair in pairs:
    ds, stem1, stem2 = pair
    print(stem1, stem2)
    
    h51 = next((res_dir / stem1).glob("*.h5"))
    h52 = next((res_dir / stem2).glob("*.h5"))
    
    out_dir = fig_dir / stem1 / f"venn_vs_{stem2}"
    out_dir.mkdir(exist_ok=True)
               
    raw_bin = next(snip_dir.glob(f"{ds}*.ap.bin"))
    print(raw_bin)
    
    with h5py.File(h51) as d:
        # x, y, z, alpha = d["localizations"][:, :4].T
        maxptps1 = d["maxptps"][:]
        geom = d["geom"][:]
        labels1 = d["labels"][:]
        spike_index1 = d["spike_index"][:]
               
    with h5py.File(h52) as d:
        # x, y, z, alpha = d["localizations"][:, :4].T
        maxptps2 = d["maxptps"][:]
        geom = d["geom"][:]
        labels2 = d["labels"][:]
        spike_index2 = d["spike_index"][:]
    
    
    def job(unit):
        fig = plot_agreement_venn(
            unit,
            geom,
            ci,
            spike_index1,
            spike_index2,
            labels1,
            labels2,
            stem1,
            stem2,
            raw_bin,
            maxptps1,
            maxptps2,
        )
        if fig is not None:
            fig.savefig(out_dir / f"unit_{unit:03d}.png", transparent=False, pad_inches=0)
            plt.close(fig)

    with Parallel(
        12,
    ) as p:
        units = np.setdiff1d(np.unique(labels1), [-1])
        for res in p(delayed(job)(u) for u in tqdm(units)):
            pass

# %%
# %matplotlib inline

# %%
import time

# %%
np.arange(35).reshape(5, 7)[m].reshape(5, 3)

# %%
m = np.c_[np.zeros((5, 4), dtype=bool), np.ones((5, 3), dtype=bool)]
m

# %%
import numpy as np
from scipy import sparse


# %%
def test_spsolve(N):
    rg = np.random.default_rng()
    T = 121
    C = 20
    K = 3
    S = rg.normal(size=(T*C, N))
    W = rg.normal(size=(T*C, N))
    B = rg.normal(size=(T*C, K))
    L = rg.normal(size=(N, 4))
    
    # construct problem
    y = (S * W).T.ravel()
    dvS = sparse.dia_matrix((S.ravel(), 0), shape=(N*T*C,N*T*C))
    X = dvS @ sparse.kron(sparse.eye(N), B)

    A = sparse.kron(L.T, sparse.eye(K))
    XTX = X.T @ X
    nyTX = -y.T @ X
    
    zeros = sparse.dok_matrix((4 * K, 4 * K))
    coefts = sparse.bmat(
        [
            [XTX, A.T],
            [A, zeros],
        ],
        format="csc",
    )
    
    targ = sparse.bmat(
        [
            [nyTX[:, None]],
            [sparse.dok_matrix((4*K,1))],
        ],
        format="csc",
    )
    
    tic = time.time()
    res_spsolve = sparse.linalg.spsolve(coefts, targ.toarray()[:, 0])
    return time.time() - tic


# %%
def test_lsmr(N):
    rg = np.random.default_rng()
    T = 121
    C = 20
    K = 3
    S = rg.normal(size=(T*C, N))
    W = rg.normal(size=(T*C, N))
    B = rg.normal(size=(T*C, K))
    L = rg.normal(size=(N, 4))
    
    # construct problem
    y = (S * W).T.ravel()
    dvS = sparse.dia_matrix((S.ravel(), 0), shape=(N*T*C,N*T*C))
    X = dvS @ sparse.kron(sparse.eye(N), B)

    A = sparse.kron(L.T, sparse.eye(K))
    XTX = X.T @ X
    nyTX = -y.T @ X
    
    zeros = sparse.dok_matrix((4 * K, 4 * K))
    coefts = sparse.bmat(
        [
            [XTX, A.T],
            [A, zeros],
        ],
        format="csc",
    )
    
    targ = sparse.bmat(
        [
            [nyTX[:, None]],
            [sparse.dok_matrix((4*K,1))],
        ],
        format="csc",
    )
    
    tic = time.time()
    res_lsmr = sparse.linalg.lsmr(coefts, targ.toarray()[:, 0], atol=1e-6 / (N * T * C),  btol=1e-6 / (N * T * C))
    return time.time() - tic


# %%
def test_cg(N):
    rg = np.random.default_rng()
    T = 121
    C = 20
    K = 3
    S = rg.normal(size=(T*C, N))
    W = rg.normal(size=(T*C, N))
    B = rg.normal(size=(T*C, K))
    L = rg.normal(size=(N, 4))
    
    # construct problem
    y = (S * W).T.ravel()
    dvS = sparse.dia_matrix((S.ravel(), 0), shape=(N*T*C,N*T*C))
    X = dvS @ sparse.kron(sparse.eye(N), B)

    A = sparse.kron(L.T, sparse.eye(K))
    XTX = X.T @ X
    nyTX = -y.T @ X
    
    zeros = sparse.dok_matrix((4 * K, 4 * K))
    coefts = sparse.bmat(
        [
            [XTX, A.T],
            [A, zeros],
        ],
        format="csc",
    )
    
    targ = sparse.bmat(
        [
            [nyTX[:, None]],
            [sparse.dok_matrix((4*K,1))],
        ],
        format="csc",
    )
    
    tic = time.time()
    res_lsmr = sparse.linalg.cg(coefts, targ.toarray()[:, 0], tol=1e-6 / (N * T * C))
    return time.time() - tic


# %%
def test_cgs(N):
    rg = np.random.default_rng()
    T = 121
    C = 20
    K = 3
    S = rg.normal(size=(T*C, N))
    W = rg.normal(size=(T*C, N))
    B = rg.normal(size=(T*C, K))
    L = rg.normal(size=(N, 4))
    
    # construct problem
    y = (S * W).T.ravel()
    dvS = sparse.dia_matrix((S.ravel(), 0), shape=(N*T*C,N*T*C))
    X = dvS @ sparse.kron(sparse.eye(N), B)

    A = sparse.kron(L.T, sparse.eye(K))
    XTX = X.T @ X
    nyTX = -y.T @ X
    
    zeros = sparse.dok_matrix((4 * K, 4 * K))
    coefts = sparse.bmat(
        [
            [XTX, A.T],
            [A, zeros],
        ],
        format="csc",
    )
    
    targ = sparse.bmat(
        [
            [nyTX[:, None]],
            [sparse.dok_matrix((4*K,1))],
        ],
        format="csc",
    )
    
    tic = time.time()
    D = sparse.dia_matrix((1/sparse.linalg.norm(XTX, axis=1), 0), coefts.shape)
    res_lsmr = sparse.linalg.cgs(coefts, targ.toarray()[:, 0], tol=1e-6 / (N * T * C), M=D)
    return time.time() - tic


# %%

# %%

# %%

# %%

# %%
from tqdm import tqdm

# %%
spsolves = [test_spsolve(N) for N in tqdm([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])]

# %%
cgs = [test_cg(N) for N in tqdm([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])]

# %%
cgss = [test_cgs(N) for N in tqdm([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])]

# %%
lsmrs = [test_lsmr(N) for N in tqdm([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])]#, 20000, 50000])]

# %%
# lsmrdenses = [test_lsmr_dense(N) for N in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000]]

# %%
# %matplotlib inline

# %%
plt.figure()
plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], spsolves, label="spsolve")
plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], cgs, label="cg")
plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], cgss, label="cgs")
plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], lsmrs, label="lsmr")
# plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000], lsmrdenses, label="lsmr (dense)")
# plt.loglog()
plt.legend()
plt.ylabel("wall time in solver")
plt.xlabel("N")
plt.show()

# %%
# %%timeit

F = cp.Variable((K, N))
obj = cp.Minimize(cp.sum_squares(cp.multiply(S, (W - B @ F))))
constraints = [F @ L == 0]
prob = cp.Problem(obj, constraints)

res = prob.solve(verbose=True)

# %%
from scipy import sparse
import time

# %%
tic = time.time()

# %%
y = (S * W).T.ravel()
dvS = sparse.dia_matrix((S.ravel(), 0), shape=(N*T*C,N*T*C))
X = dvS @ sparse.kron(sparse.eye(N), B)

# %%
X.shape

# %%
A = sparse.kron(L.T, sparse.eye(K))

# %%
XTX = X.T @ X

# %%
XTX.shape

# %%
XTX.nnz

# %%
zeros = sparse.dok_matrix((4 * K, 4 * K))
coefts = sparse.bmat(
    [
        [XTX, A.T],
        [A, zeros],
    ],
    format="csc",
)

# %%
nyTX = -y.T @ X
nyTX.shape

# %%
targ = sparse.bmat(
    [
        [nyTX[:, None]],
        [sparse.dok_matrix((4*K,1))],
    ],
    format="csc",
)

# %%
print("took", time.time() - tic, "seconds")

# %%
coefts.shape, targ.shape

# %%

# %%
tic = time.time()

# %%
res_spsolve = sparse.linalg.spsolve(coefts, targ.toarray()[:, 0])

# %%
print("took", time.time() - tic, "seconds")

# %%
tic = time.time()

# %%
res_spsolve = sparse.linalg.spsolve(coefts, targ.toarray()[:, 0])

# %%
print("took", time.time() - tic, "seconds")

# %%
F__ = res_spsolve[:N * K].reshape(N, K).T

# %%
tic = time.time()

# %%

# %%
print("took", time.time() - tic, "seconds")

# %%
res_lsmr

# %%
F_ = res_lsmr[0][:N * K].reshape(N, K).T

# %%
F.value.shape

# %%
np.isclose(F.value, F_).all()

# %%
np.isclose(F__, F_, atol=5e-6).all()

# %%
import matplotlib.pyplot as plt

# %%
plt.hist((F.value - F_).ravel(), bins=128);

# %%
plt.hist((F_ - F__).ravel(), bins=128);

# %%
plt.hist(F.value.ravel(), bins=128);

# %%
plt.hist(F_.ravel(), bins=128);

# %%

# %%

# %%

# %%
import spikeinterface 
from spikeinterface.toolkit import compute_correlograms
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.widgets import plot_agreement_matrix

# %%

# %%
import colorcet as cc

# %%
# plt.cm.viridis?

# %%
