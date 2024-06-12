from dataclasses import dataclass
from typing import List

from MEMIT.util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # NEW PARAMS
    loss_break: float = 1e-3

    # Defaults
    batch_size: int = 1
    wd_power_law: tuple = None  # Scale weight decay by number of edits
