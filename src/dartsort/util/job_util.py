from .internal_config import ComputationConfig, default_computation_cfg

comp_cfg_holder = {}


def set_global_computation_config(
    comp_cfg: ComputationConfig | None = None,
    *,
    n_jobs: int | None = None,
    **comp_cfg_kwargs,
):
    global comp_cfg_holder
    if comp_cfg is None and n_jobs is not None:
        comp_cfg = ComputationConfig.from_n_jobs(n_jobs=n_jobs)
    elif comp_cfg is None:
        comp_cfg = ComputationConfig(**comp_cfg_kwargs)
    else:
        assert comp_cfg is not None
    comp_cfg_holder["comp_cfg"] = comp_cfg


def get_global_computation_config() -> ComputationConfig:
    global comp_cfg_holder
    return comp_cfg_holder.get("comp_cfg", default_computation_cfg)


def ensure_computation_config(cfg: ComputationConfig | None) -> ComputationConfig:
    if cfg is None:
        cfg = get_global_computation_config()
    return cfg
