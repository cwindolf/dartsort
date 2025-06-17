from .internal_config import ComputationConfig, default_computation_cfg

comp_cfg_holder = {}


def set_global_computation_config(comp_cfg: ComputationConfig):
    global comp_cfg_holder
    comp_cfg_holder["comp_cfg"] = comp_cfg


def get_global_computation_config() -> ComputationConfig:
    global comp_cfg_holder
    return comp_cfg_holder.get("comp_cfg", default_computation_cfg)
