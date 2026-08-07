"""Regenerate or verify any artifact of the paper.

An artifact is a table or a figure, declared in ``configs/artifacts/`` as an ordered list
of steps. This runs those steps, or with ``--verify`` additionally checks that every step
produced the files it declares. The repository ships no reference numbers or weights, so
verification is structural: it answers whether the pipeline's stages connect end to end
on your inputs, and the produced tables are then compared against the published ones by
reading the paper.

The resolved commands are always printed, so this wrapper is auditable and can be bypassed
entirely by running them directly. It never special cases a number.

    python -m alphawise.reproduce --list-artifacts
    python -m alphawise.reproduce --artifact main_comparison --verify
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from alphawise.config import ConfigError, config_dir, load

REPO_ROOT = Path(__file__).resolve().parents[1]

EXIT_OK, EXIT_MISMATCH, EXIT_MISSING = 0, 1, 2


def artifact_names() -> list[str]:
    """Every declared artifact, one per ``configs/artifacts/*.yaml``, in name order."""
    return sorted(p.stem for p in (config_dir() / "artifacts").glob("*.yaml"))


def describe(name: str) -> tuple[str, str]:
    """An artifact's paper reference and its provenance, ``"unstated"`` when it declares none."""
    cfg = load(f"artifacts/{name}.yaml")
    return cfg["artifact.paper_ref"], cfg.get("artifact.provenance", "unstated")


def resolved_steps(cfg) -> list[tuple[str, str]]:
    """Flatten the step list, expanding a ``foreach`` step into one command per item.

    ``foreach: methods`` names a list elsewhere in the artifact, and every ``{item}`` in
    the command is replaced by that element. An item may be a mapping, in which case
    ``{item.key}`` addresses its fields.
    """
    steps = []
    for step in cfg["artifact.steps"]:
        template = " ".join(step["command"].split())
        source = step.get("foreach")
        if source is None:
            steps.append((step["name"], template, step))
            continue
        for item in cfg[f"artifact.{source}"]:
            fields = item if isinstance(item, dict) else {}
            command = template
            for key, value in fields.items():
                command = command.replace(f"{{item.{key}}}", str(value))
            label = fields.get("id", item if not fields else fields)
            steps.append((f"{step['name']}[{label}]", command.replace("{item}", str(label)), step))
    return steps


#: Placeholders a command legitimately passes through, because the program it invokes
#: expands them itself once per phase.
_DEFERRED = {"{phase}"}


def unresolved(command: str) -> list[str]:
    """Placeholders and missing modules, so a broken step fails loudly before it runs."""
    problems = [token for token in re.findall(r"\{[^}]+\}|\$\{[^}]+\}", command)
                if token not in _DEFERRED]
    module = re.search(r"python -m ([\w.]+)", command)
    if module:
        # A target is either a module file or a package with a __main__, so check both.
        stem = REPO_ROOT / module.group(1).replace(".", "/")
        if not (stem.with_suffix(".py").exists() or (stem / "__main__.py").exists()):
            problems.append(f"no module {module.group(1)}")
    return problems


def missing_inputs(step: dict) -> list[str]:
    """The declared inputs of one step that do not exist on this machine.

    An artifact names what it needs (assets, data roots, checkpoint directories) so a
    reproducer is told everything that is absent before any compute is spent, instead of
    discovering one missing path per failed run.
    """
    return [entry for entry in step.get("inputs", [])
            if not Path(entry.replace("{phase}", "1")).expanduser().exists()]


def run_steps(steps, *, dry_run: bool) -> int:
    """Run an artifact's resolved steps in order, stopping at the first one that fails.

    Every command is printed before it is considered, in ``--dry-run`` too, so the wrapper
    is auditable and anything it does can be rerun by hand without it. A step whose command
    still holds placeholders, or whose declared inputs are absent, is refused rather than
    handed to the shell, which is why the return value can be ``EXIT_MISSING`` as well as
    a step's own exit code.
    """
    for name, command, step in steps:
        print(f"\n$ {command}")
        problems = unresolved(command)
        if problems:
            print(f"step {name!r} is not runnable: {', '.join(problems)}", file=sys.stderr)
            return EXIT_MISSING
        if dry_run:
            continue
        absent = missing_inputs(step)
        if absent:
            print(f"step {name!r} MISSING INPUTS:", file=sys.stderr)
            for entry in absent:
                print(f"  {entry}", file=sys.stderr)
            return EXIT_MISSING
        # Run every step under the interpreter that launched this one. A bare "python"
        # would resolve against PATH, so a step could silently execute in a different
        # environment than the one the caller chose.
        result = subprocess.run(
            command.replace("python ", f"{sys.executable} ", 1), shell=True, cwd=REPO_ROOT)
        if result.returncode:
            print(f"step {name!r} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
    return EXIT_OK


def verify_outputs(cfg) -> int:
    """Check that every step produced the files it declares, as a process exit code.

    Nothing else ships to compare against, so this is what verification means here: the
    stages connect and each one delivered its output. Whether the numbers match the paper
    is read off the produced tables against the published ones.
    """
    missing = []
    for step in cfg["artifact.steps"]:
        for output in step.get("outputs", []):
            # A foreach step declares one output pattern for all its items.
            pattern = output.lstrip("./").replace("{item}", "*")
            if not list(REPO_ROOT.glob(pattern)):
                missing.append(f"{step['name']}: {output}")
    for entry in missing:
        print(f"  missing output  {entry}", file=sys.stderr)
    if missing:
        print(f"\nVERIFY FAILED  {len(missing)} declared outputs were not produced", file=sys.stderr)
        return EXIT_MISMATCH
    produced = sum(len(step.get("outputs", [])) for step in cfg["artifact.steps"])
    print(f"\nVERIFY OK  every stage produced its declared output ({produced} checked)")
    return EXIT_OK


def main(argv=None) -> int:
    """Regenerate, or regenerate and verify, one artifact or every artifact in turn.

    The exit code is the worst seen over the artifacts visited: ``EXIT_OK``, a failing
    step's own code, ``EXIT_MISMATCH`` when a declared output was not produced, and
    ``EXIT_MISSING`` when a config or a module is absent. Verification only runs after
    the steps succeeded and never under ``--dry-run``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", help="one of --list-artifacts")
    parser.add_argument("--all", action="store_true", help="every artifact in turn")
    parser.add_argument("--list-artifacts", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="after the steps, check every declared output was produced")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = parser.parse_args(argv)

    if args.list_artifacts:
        for name in artifact_names():
            paper_ref, provenance = describe(name)
            print(f"  {name:<10} {paper_ref}  [provenance: {provenance}]")
        return EXIT_OK

    names = artifact_names() if args.all else ([args.artifact] if args.artifact else [])
    if not names:
        parser.error("pass --artifact NAME, --all, or --list-artifacts")

    worst = EXIT_OK
    for name in names:
        try:
            cfg = load(f"artifacts/{name}.yaml")
        except ConfigError as error:
            print(error, file=sys.stderr)
            return EXIT_MISSING
        print(f"\n=== artifact {name}  ({cfg['artifact.paper_ref']}) ===")
        print(f"provenance {cfg.get('artifact.provenance', 'unstated')}")
        code = run_steps(resolved_steps(cfg), dry_run=args.dry_run)
        if code == EXIT_OK and args.verify and not args.dry_run:
            code = verify_outputs(cfg)
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
