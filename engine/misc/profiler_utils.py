"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

try:
    from calflops import calculate_flops
except ModuleNotFoundError:
    calculate_flops = None

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()
    params = sum(p.numel() for p in model_for_info.parameters())

    # calflops monkey-patches torch functionals during profiling and is not
    # compatible with the current torch/torchvision stack in this workspace.
    # That leaked patching into the real training graph, so we disable FLOPs
    # profiling entirely to keep training stable.
    if calculate_flops is None:
        del model_for_info
        return params, {"Model Params:%s   FLOPs/MACs: unavailable (install calflops to enable)" % params}

    del model_for_info
    return params, {"Model Params:%s   FLOPs/MACs: disabled (calflops compatibility issue)" % params}
