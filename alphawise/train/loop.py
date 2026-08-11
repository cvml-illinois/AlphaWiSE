"""The one per-phase training loop.

Seven trainers in the dev tree shared roughly half their code. They differed in exactly
three places: the extra loss terms they add, the modules they own besides the backbone,
and what they do at the phase boundary. All three are now
:class:`~alphawise.train.methods.base.ContinualMethod` callbacks; everything else, DDP setup,
parameter groups, optimizer and schedule, the epoch loop, checkpointing and the optional
end-of-phase evaluation, lives here and is written once.

Phase 0 is the frozen bootstrap that every method starts from and is never trained, so
:func:`run_phase` covers phases 1..7. ``phase=None`` means joint training over all classes
at once, with no phase split and no replay.

Data and evaluation are injected, not imported: the caller passes a built ``train_loader``
and an optional ``evaluate`` callable. That keeps the loop testable without a dataset and
keeps the dataset layer free to change independently.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.optim.lr_scheduler import ExponentialLR

from ..utils import dist
from ..utils.checkpoint import load_model_state, resume_trainer_state, save_phase_checkpoint
from ..utils.seed import seed_everything
from .methods.base import Batch, ContinualMethod, PhaseContext, StepOutput

BOOTSTRAP_PHASE = 0


@dataclass
class LoopConfig:
    """Everything the loop needs that a method does not decide.

    Defaults reproduce the project-standard setup (SGD 5e-5 with an exponential decay of
    0.96 over 30 epochs). The methods that deviate, EWC and DER on the optimizer, C-CLIP on
    freezing, say so in their own configuration rather than in their own loop.
    """

    epochs: int = 30
    #: "full" trains everything, "partial" trains only the audio path, "text" freezes the
    #: text encoder, "method" means the method manages freezing itself (C-CLIP's LoRA).
    freeze: str = "full"
    optimizer: str = "sgd"
    lr: float = 5e-5
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    betas: Tuple[float, float] = (0.9, 0.999)
    #: When set, the logit-scale temperatures get their own group at this learning rate
    #: and no weight decay.
    scale_group_lr: Optional[float] = None
    scheduler_gamma: float = 0.96
    find_unused_parameters: bool = True
    broadcast_buffers: bool = False
    static_graph: bool = False
    ddp_backend: str = "nccl"
    ddp_timeout_s: int = 7200
    eval_in_loop: bool = True
    cudnn_benchmark: bool = True
    checkpoint_filename: str = "model_weights_epoch_{epoch}.pth"
    resume_strict: bool = True
    #: None keeps the historical unseeded behaviour; see alphawise.utils.seed.
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, cfg, *, seed=None):
        """Flatten a resolved config document into loop settings.

        The config groups these by concern (optimizer, scheduler, freeze, ddp) because
        that is how a reader thinks about them, while the loop wants them flat. This is
        the single place the two shapes meet, so no call site hand-picks keys.
        """
        freeze = "full"
        if cfg["train.freeze.partial_training"]:
            freeze = "partial"
        elif cfg["train.freeze.text_freeze"]:
            freeze = "text"
        scale_group = cfg["train.optimizer.scale_group"]
        return cls(
            epochs=cfg["train.epochs"],
            freeze=freeze,
            optimizer=cfg["train.optimizer.name"],
            lr=cfg["train.optimizer.lr"],
            momentum=cfg["train.optimizer.momentum"],
            weight_decay=cfg["train.optimizer.weight_decay"],
            nesterov=cfg["train.optimizer.nesterov"],
            betas=tuple(cfg["train.optimizer.betas"]),
            scale_group_lr=scale_group["lr"] if scale_group["enabled"] else None,
            scheduler_gamma=cfg["train.scheduler.gamma"],
            find_unused_parameters=cfg["train.ddp.find_unused_parameters"],
            broadcast_buffers=cfg["train.ddp.broadcast_buffers"],
            static_graph=cfg["train.ddp.static_graph"],
            ddp_backend=cfg["train.ddp.backend"],
            ddp_timeout_s=cfg["train.ddp.timeout_s"],
            seed=seed,
        )


def apply_freeze(model, config):
    """Set ``requires_grad`` according to the freeze policy."""
    if config.freeze == "method":
        return
    if config.freeze not in ("full", "partial", "text"):
        raise ValueError(f"unknown freeze policy {config.freeze!r}")

    for parameter in model.parameters():
        parameter.requires_grad = config.freeze != "partial"
    if config.freeze == "text":
        for parameter in model.transformer.parameters():
            parameter.requires_grad = False
    elif config.freeze == "partial":
        for parameter in model.audio.parameters():
            parameter.requires_grad = True
        # The image-text temperature stays frozen; only the two audio ones adapt.
        model.logit_scale_ai.requires_grad = True
        model.logit_scale_at.requires_grad = True


def _temperature_group(model, config):
    """Split the trainable parameters into a body group and a logit-scale group."""
    scales, body = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (scales if "logit_scale" in name else body).append(parameter)
    return [
        {"params": body, "lr": config.lr},
        {"params": scales, "lr": config.scale_group_lr, "weight_decay": 0.0},
    ]


def _partial_group(model, config):
    """Audio path in one group, filter bank and temperatures in an undecayed second one."""
    undecayed = list(model.audio.fbsp.parameters()) + [model.logit_scale_ai, model.logit_scale_at]
    undecayed_ids = {id(p) for p in undecayed}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in undecayed_ids]
    return [{"params": body}, {"params": undecayed, "weight_decay": 0.0}]


def build_param_groups(model, config):
    """The optimizer's parameter groups for this freeze policy.

    A partial-training phase always uses its own two-group split, so ``scale_group_lr`` has
    no effect there; the temperatures already sit in the undecayed group.
    """
    if config.freeze == "partial":
        return _partial_group(model, config)
    if config.scale_group_lr is not None:
        return _temperature_group(model, config)
    return [{"params": [p for p in model.parameters() if p.requires_grad]}]


def build_optimizer(param_groups, config):
    """The optimizer the config names. SGD and AdamW are the only two the paper uses.

    The values here are group defaults: a parameter group that carries its own ``lr`` or
    ``weight_decay`` keeps that value, and the AdamW branch passes no ``lr`` at all.
    """
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def _flatten_eval(evaluation):
    """Evaluation output to flat metric keys, matching the historical logging layout."""
    metrics = {}
    for phase, per_metric in evaluation.get("per_phase", {}).items():
        for key, value in per_metric.items():
            metrics[f"eval/phase_{phase}/{key}"] = value
    for key, value in evaluation.get("avg", {}).items():
        metrics[f"eval/avg/{key}"] = value
    return metrics


def print_epoch(epoch, phase, metrics):
    """Default logger. The prefix is load-bearing: the log parsers match on it."""
    parts = [f"Epoch Num: {epoch}", f"Epoch Loss: {metrics['epoch_loss']:.6f}"]
    parts += [
        f"{key}: {value:.6f}"
        for key, value in metrics.items()
        if key != "epoch_loss" and not key.startswith("eval/")
    ]
    parts += [
        f"{key.rsplit('/', 1)[-1]}: {value:.4f}"
        for key, value in metrics.items()
        if key.startswith("eval/avg/")
    ]
    print(" | ".join(parts), flush=True)


def _train_batch(ctx, model, optimizer, method, raw_batch):
    """One optimization step. Returns the per-term scalars for logging."""
    optimizer.zero_grad()
    batch = Batch.collate(raw_batch, ctx.device)
    (features, logits), base_loss = model(**batch.as_kwargs())
    if base_loss.ndim > 0:
        base_loss = base_loss.mean()
    out = StepOutput(features=features, logits=logits, loss=base_loss)

    terms = method.loss_terms(ctx, batch, out)
    total = base_loss if method.uses_base_loss else torch.zeros((), device=base_loss.device)
    for value in terms.values():
        total = total + value

    total.backward()
    # step()'s positional argument is a loss-reevaluation closure. The dev tree's LwF
    # trainer passed None into it explicitly; that is the default, so it was harmless,
    # but it reads as though a closure had been intended.
    optimizer.step()

    scalars = {"epoch_loss": float(total), "base_loss": float(base_loss)}
    scalars.update({name: float(value) for name, value in terms.items()})
    return scalars


def run_phase(model, train_loader, checkpoint_dir, config=None, method=None, sampler=None,
              phase=None, class_order="forward", resume_ckpt=None, evaluate=None, log=None,
              device=None):
    """Train one continual-learning phase and write its checkpoint.

    Args:
        model: an unwrapped backbone on any device; the loop moves and wraps it.
        train_loader: this phase's DataLoader.
        checkpoint_dir: existing directory for this phase's checkpoint.
        config: :class:`LoopConfig`.
        method: the method. Defaults to plain sequential fine-tuning.
        sampler: the loader's DistributedSampler, so each epoch can be reshuffled.
        phase: 1..7, or None for joint training. Phase 0 is the bootstrap and is rejected.
        resume_ckpt: the previous phase's checkpoint, or the phase-0 bootstrap for phase 1.
        evaluate: ``(model, device, phase, class_order) -> dict | None``, called at the last
            epoch only. Evaluation is heavy enough that a per-epoch call overruns the DDP
            watchdog while the non-master ranks wait at the barrier.
        log: ``(epoch, phase, metrics) -> None``, called on the master rank only.

    Returns the last epoch's metrics plus the path of the checkpoint that was written.
    """
    if phase == BOOTSTRAP_PHASE:
        raise ValueError("phase 0 is the shared frozen bootstrap and is never trained")

    config = config or LoopConfig()
    method = method or ContinualMethod()
    log = log or print_epoch
    seed_everything(config.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    device = device or dist.setup(backend=config.ddp_backend, timeout_s=config.ddp_timeout_s)

    if dist.is_master():
        # The banner states only what the loop can observe. Its predecessor announced
        # "exemplar training" unconditionally, including for joint training, which uses
        # no replay at all; the dataset layer had the same message inverted outright,
        # announcing exemplar replay from the branch that omits it.
        scope = "all classes" if phase is None else f"phase {phase}"
        print(f"=== {method.name} | {class_order} | {scope} | {len(train_loader.dataset)} "
              f"training samples | bootstrap = {resume_ckpt} ===", flush=True)

    model = model.to(device)
    blob = None
    if resume_ckpt and os.path.isfile(resume_ckpt):
        blob = load_model_state(model, resume_ckpt, strict=config.resume_strict,
                                map_location=device)

    apply_freeze(model, config)
    ctx = PhaseContext(config=config, model=model, train_loader=train_loader, device=device,
                       checkpoint_dir=checkpoint_dir, phase=phase, class_order=class_order,
                       resume_blob=blob)
    method.before_phase(ctx)

    owned = {name: module.to(device) for name, module in method.modules().items()}
    param_groups = build_param_groups(model, config)
    param_groups += [
        {"params": list(module.parameters()), "lr": config.lr} for module in owned.values()
    ]
    optimizer = build_optimizer(param_groups, config)
    scheduler = ExponentialLR(optimizer, gamma=config.scheduler_gamma)

    # A phase-to-phase bootstrap deliberately restarts the optimizer; only an interrupted
    # run of the same phase (joint training, which has no phases) resumes its state.
    start_epoch = 0
    if blob is not None and phase is None and "optimizer_state_dict" in blob:
        start_epoch = resume_trainer_state(optimizer, scheduler, blob)

    model = dist.wrap(model, find_unused_parameters=config.find_unused_parameters,
                      broadcast_buffers=config.broadcast_buffers,
                      static_graph=config.static_graph)
    for name, module in owned.items():
        setattr(method, name, dist.wrap(module))

    metrics = {}
    for epoch in range(start_epoch, config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()

        totals = defaultdict(float)
        batches = 0
        for raw_batch in train_loader:
            for name, value in _train_batch(ctx, model, optimizer, method, raw_batch).items():
                totals[name] += value
            batches += 1
        scheduler.step()

        metrics = {name: value / max(1, batches) for name, value in totals.items()}
        is_last = epoch + 1 == config.epochs
        if is_last and evaluate is not None and config.eval_in_loop and dist.is_master():
            evaluation = evaluate(model, device, phase, class_order)
            if evaluation is not None:
                metrics.update(_flatten_eval(evaluation))
        if dist.is_master():
            log(epoch + 1, phase, metrics)

    method.before_save(ctx)
    checkpoint_path = os.path.join(
        checkpoint_dir, config.checkpoint_filename.format(epoch=config.epochs)
    )
    if dist.is_master():
        extras = dict(method.checkpoint_extras(ctx))
        extras.update({
            f"{name}_state_dict": dist.unwrap(getattr(method, name)).state_dict()
            for name in owned
        })
        save_phase_checkpoint(model, optimizer, scheduler, checkpoint_path,
                              epoch=config.epochs, avg_loss=metrics.get("epoch_loss"),
                              extras=extras)
    dist.barrier()
    method.after_phase(ctx)

    return {"metrics": metrics, "checkpoint": checkpoint_path}
