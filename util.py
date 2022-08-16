import torch
import os
import numpy as np
from google.protobuf import text_format

def read_from_pbtxt(config, path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            text_format.Parse(f.read(), config)
            f.close()

def write_to_pbtxt(config, path):
    with open(path, 'w') as f:
        f.write(text_format.MessageToString(config, use_short_repeated_primitives=True))
        f.close()

def travel_layers(mod, name_path=None):
    layers = []
    if name_path is None:
        name_path = mod.__class__.__qualname__

    if len(list(mod.children())) == 0:
        if isinstance(mod, torch.nn.Dropout):
            return []

        _name_path = f"{name_path}.{mod.__class__.__qualname__}"
        setattr(mod, "__qualname__", _name_path)

        return [mod]
    else:
        for i, (name, child) in enumerate(mod.named_children()):
            _name_path = f"{name_path}.{child.__class__.__qualname__}{i}"

            layers += travel_layers(child, _name_path)

        return layers

def get_module_size(module, ignore_cuda=False):
    size = 0
    for key, parm in module._parameters.items():
        if parm is not None:
            if ignore_cuda is True and parm.is_cuda:
                continue
            size += np.prod(np.array(parm.size())) * 4
    for key, buf in module._buffers.items():
        if buf is not None:
            if ignore_cuda is True and buf.is_cuda:
                continue
            size += np.prod(np.array(buf.size())) * 4

    for child in module.children():
        size += get_module_size(child, ignore_cuda)

    return size
