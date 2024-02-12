from ..config import ComputationConfig, default_computation_config

comp_cfg_holder = {}


def set_global_computation_config(comp_cfg):
    global comp_cfg_holder
    comp_cfg_holder["comp_cfg"] = comp_cfg


def get_global_computation_config():
    global comp_cfg_holder
    return comp_cfg_holder.get("comp_cfg", default_computation_config)
