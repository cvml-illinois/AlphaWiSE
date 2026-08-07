"""Train one continual-learning trajectory over a phase sequence.

Phase 0 is the shared frozen bootstrap and is never trained. Training starts at phase 1
from that bootstrap, and every later phase resumes from the checkpoint the previous one
wrote, which is what makes the sequence continual rather than a set of independent runs.

``--method`` names the method and loads ``configs/method/<key>.yaml`` as the last
config overlay, so one flag selects both the implementation and its published
hyperparameters.

    python -m alphawise.train --config configs/artifacts/smoke.yaml --method ewc \\
        --class-order forward --phases 1,2,3,4,5,6,7
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..config import Config, load, overrides_from_cli
from ..data.audioset import CLASS_ORDERS, ContinualAudioSet, collate_batch
from ..data.roots import DataRoots
from ..models.factory import build_audioclip
from ..utils import dist
from ..utils.paths import resolve_bootstrap
from ..utils.seed import seed_everything, worker_init_fn
from .methods import build_method
from .loop import LoopConfig, print_epoch, run_phase


def phase_loader(cfg: Config, phase: int, class_order: str,
                 seed: Optional[int]) -> Tuple[ContinualAudioSet, DataLoader, object]:
    """This phase's training data, plus the sampler the loop reshuffles each epoch."""
    dataset = ContinualAudioSet(
        DataRoots.from_config(cfg, "train"),
        phase=phase,
        class_order=class_order,
        memory_size=cfg["continual.memory.size"],
        classes_per_phase=cfg["continual.classes_per_phase"],
        balancing=cfg["continual.memory.balancing"],
        wav_id_mode=cfg.get("data.wav_id_mode"),
    )
    sampler = (torch.utils.data.distributed.DistributedSampler(dataset)
               if dist.is_distributed() else None)
    loader = DataLoader(
        dataset,
        batch_size=cfg["train.batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=cfg["train.dataloader.num_workers"],
        pin_memory=True,
        drop_last=True,
        # The factory must be called: it returns the per-worker callable, and None for
        # seed=None, which keeps the historical unseeded worker behaviour.
        worker_init_fn=worker_init_fn(seed),
    )
    return dataset, loader, sampler


def main(argv=None) -> int:
    """Run the phase sequence, chaining each phase's checkpoint into the next."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", default=[],
                        help="config overlay, repeatable")
    parser.add_argument("--method", required=True,
                        help="method key; loads configs/method/<key>.yaml as the last overlay")
    parser.add_argument("--class-order", required=True, choices=CLASS_ORDERS,
                        help="class ordering, which also selects the phase-0 bootstrap")
    parser.add_argument("--phases", default="1,2,3,4,5,6,7")
    parser.add_argument("--checkpoint-root", help="overrides the config's checkpoint root")
    parser.add_argument("--seed", type=int, help="makes the run deterministic; omitted "
                                                 "leaves the historical unseeded behaviour")
    parser.add_argument("--from-backbone", action="store_true",
                        help="start from the pretrained backbone when no phase-0 bootstrap "
                             "exists; without it a missing bootstrap is an error, because "
                             "silently skipping it changes the protocol")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load(*args.config, f"method/{args.method}.yaml",
               cli_overrides=overrides_from_cli(args.set))
    if args.seed is not None:
        seed_everything(args.seed)

    config = LoopConfig.from_config(cfg, seed=args.seed)
    method_name = cfg["method.name"]
    root = Path(args.checkpoint_root or cfg["paths.checkpoint_root"]) / \
        f"{method_name}_{args.class_order}"

    # setup() is what joins the process group under torchrun; guarding it behind
    # is_distributed() could never fire, so every rank used to train the full data.
    device = torch.device(dist.setup(config.ddp_backend, config.ddp_timeout_s))

    resume = resolve_bootstrap(cfg, args.class_order)
    if resume is None:
        message = (f"no phase-0 bootstrap declared or found under "
                   f"{cfg['paths.bootstrap_root']}; phase {args.phases.split(',')[0]} "
                   f"would start from the pretrained backbone, which is not the "
                   f"published protocol")
        if not args.from_backbone:
            raise SystemExit(message + ". Pass --from-backbone to do it anyway.")
        if dist.is_master():
            print(message + " (--from-backbone given, proceeding)", flush=True)

    for phase in [int(p) for p in args.phases.split(",")]:
        checkpoint_dir = root / f"phase_{phase}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dataset, loader, sampler = phase_loader(cfg, phase, args.class_order, args.seed)
        if dist.is_master():
            print(f"\nphase {phase}: {len(dataset)} samples, resuming from {resume}",
                  flush=True)

        result = run_phase(
            build_audioclip(), loader, str(checkpoint_dir),
            config=config,
            method=build_method(method_name, **cfg.get(f"method.{method_name}", {})),
            sampler=sampler, phase=phase, class_order=args.class_order,
            resume_ckpt=str(resume) if resume else None, log=print_epoch, device=device,
        )
        # Chaining the checkpoint forward is what makes this continual: the next phase
        # never sees this phase's data again except through the exemplar memory.
        resume = result["checkpoint"]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
