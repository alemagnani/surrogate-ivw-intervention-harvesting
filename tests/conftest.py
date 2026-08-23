"""Provide a minimal torch stub so tests that don't use AllPairsEstimator
can be collected without a full PyTorch installation."""

import sys
import types


def _make_torch_stub():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")

    class _Tensor:
        pass

    class _Module:
        def __init__(self):
            pass

        def parameters(self):
            return iter([])

    class _Embedding(_Module):
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return _Tensor()

    class _Sigmoid(_Module):
        def __call__(self, *a, **kw):
            return _Tensor()

    class _Sequential(_Module):
        def __init__(self, *layers):
            self._layers = layers

        def __call__(self, *a, **kw):
            return _Tensor()

    class _Dataset:
        pass

    class _DataLoader:
        def __init__(self, dataset, **kw):
            pass

        def __iter__(self):
            return iter([])

    class _Adam:
        def __init__(self, params, **kw):
            pass

        def zero_grad(self):
            pass

        def step(self):
            pass

    class _optim:
        Adam = _Adam

    torch.tensor = lambda *a, **kw: _Tensor()
    torch.arange = lambda *a, **kw: _Tensor()
    torch.log = lambda x: x
    torch.Tensor = _Tensor

    nn.Module = _Module
    nn.Embedding = _Embedding
    nn.Sigmoid = _Sigmoid
    nn.Sequential = _Sequential

    data.Dataset = _Dataset
    data.DataLoader = _DataLoader

    torch.nn = nn
    torch.utils = utils
    torch.utils.data = data
    torch.optim = _optim

    return torch, nn, utils, data


if "torch" not in sys.modules:
    _torch, _nn, _utils, _data = _make_torch_stub()
    sys.modules["torch"] = _torch
    sys.modules["torch.nn"] = _nn
    sys.modules["torch.utils"] = _utils
    sys.modules["torch.utils.data"] = _data
