import numpy as np
import cmdstanpy
from pathlib import Path
import ujson
from tempfile import NamedTemporaryFile


model_code = """
data {{
    int<lower=0> C;
    vector[C] ptp;
    vector[C] gx;
    vector[C] gz;
}}
parameters {{
    real<lower=-100, upper=132> x;
    real<lower=0, upper=250> y;
    real<lower=-100, upper=100> z;
    real<lower=0> alpha;
}}
transformed parameters {{
    vector[C] pred_ptp = alpha ./ sqrt(
        square(gx - x) + square(gz - z) + square(y)
    );
}}
model {{
    // alpha ~ gamma(3, 1./50.); // alpha prior for posterity
    {logbarrier}target += -log1p(y / max(ptp)) / 50.0;
    ptp - pred_ptp ~ normal(0, {sigma});
}}
"""


def stanc(name, code, workdir=".stan"):
    Path(workdir).mkdir(exist_ok=True)
    path = Path(workdir) / f"{name}.stan"
    # try to avoid stan recompilation if possible
    overwrite = True
    if path.exists():
        with open(path, "r") as f:
            if f.read() == code:
                overwrite = False
    if overwrite:
        with open(path, "w") as f:
            f.write(code)
    model = cmdstanpy.CmdStanModel(
        stan_file=path, stanc_options={"warn-pedantic": True}
    )
    return model


def tojson(file, **kwargs):
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        # print(k, v)
        out[k] = v
    ujson.dump(out, file)


def sample(ptp, local_geom, sigma=0.1, logbarrier=False):
    assert local_geom.shape == (*ptp.shape, 2)

    model_name = f"lsq_sigma{sigma:0.2f}.stan"
    lb = "" if logbarrier else "// "
    model = stanc(model_name, model_code.format(sigma=sigma, logbarrier=lb))
    C = ptp.shape[0]

    with NamedTemporaryFile(mode="w", prefix="post", suffix=".json") as f:
        with open(f.name, "w") as tmp:
            tojson(tmp, C=C, ptp=ptp, gx=local_geom[:, 0], gz=local_geom[:, 1])
        res = model.sample(f.name, show_progress=False)
        summary = res.summary().loc[["lp__", "x", "y", "z", "alpha"]]
        x = res.stan_variable("x")
        y = res.stan_variable("y")
        z = res.stan_variable("z")
        alpha = res.stan_variable("alpha")

    return summary, x, y, z, alpha
