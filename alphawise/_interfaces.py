"""Abstract base classes for the vendored audio encoder and audio transforms.

Vendored verbatim from the AudioCLIP training harness (``ignite_trainer._interfaces``).
Upstream, importing these two classes pulled in the whole harness together with
pytorch-ignite and visdom; nothing else from that package is used here, so they
are inlined to keep this repository self-contained. See README.md.
"""

import abc
from typing import Callable, Optional, Tuple, Union

import torch

TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorOrTwo = Union[torch.Tensor, TensorPair]


class AbstractNet(abc.ABC, torch.nn.Module):

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> TensorOrTwo:
        pass

    @abc.abstractmethod
    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def loss_fn_name(self) -> str:
        pass


class AbstractTransform(abc.ABC, Callable[[torch.Tensor], torch.Tensor]):

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'
