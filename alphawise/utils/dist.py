"""Torch-distributed helpers, so DDP details exist once instead of in every trainer.

Every function here is safe to call in a single process with no ``torchrun`` around it.
That is the point: the dev tree called ``torch.distributed`` entry points unguarded, which
made a plain ``python train.py`` crash on a machine that could otherwise have run it.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed():
    """True only once a process group has been initialised, not merely when several GPUs
    are visible."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """This process's rank in the process group, 0 when there is no process group."""
    # The dev tree called dist.get_rank() and dist.barrier() with no is_initialized()
    # guard (par_train.py:159,163; delta_der_par_train.py:487), so any run outside
    # torchrun died at the first checkpoint with "process group is not initialized".
    return dist.get_rank() if is_distributed() else 0


def is_master():
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def local_rank():
    """This process's device index on its own node, from ``LOCAL_RANK``, 0 when unset.

    Node-local, so this and not :func:`get_rank` is the index to hand to ``cuda.set_device``
    and to DDP's ``device_ids``.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def setup(backend="nccl", timeout_s=7200):
    """Select this rank's device and, under torchrun, join the process group.

    Returns the device string. With no ``RANK`` in the environment there is no rendezvous,
    so importing and calling this on a single CPU-only machine is a no-op rather than a
    hang. The default timeout is two hours because an end-of-phase evaluation runs on rank
    0 only while the other ranks sit at a barrier, which overruns the ten-minute default.
    """
    rank = local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))
    return device


def wrap(model, find_unused_parameters=True, broadcast_buffers=False, static_graph=False):
    """Wrap for DDP, or return the model unchanged when there is nothing to synchronise.

    ``broadcast_buffers=False`` drops the per-forward buffer-sync collective, which was
    stalling rank 0 after a cold DataLoader start. ``static_graph`` is needed when a
    parameter is read both inside and outside the wrapped forward (DER touches the logit
    scales directly), otherwise DDP's methods can fire twice for it.
    """
    if not is_distributed() or torch.cuda.device_count() <= 1:
        return model
    wrapped = DistributedDataParallel(
        model,
        device_ids=[local_rank()],
        output_device=local_rank(),
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )
    if static_graph:
        wrapped._set_static_graph()
    return wrapped


def unwrap(model):
    """The bare module inside a DDP wrapper, or the model itself if it is not wrapped.

    Checkpoint reads and writes go through this so that saved keys never carry the
    ``module.`` prefix DDP would otherwise add.
    """
    return model.module if hasattr(model, "module") else model
