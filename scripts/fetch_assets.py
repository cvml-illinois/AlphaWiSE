"""Fetch or verify the pretrained weights and the tokenizer vocabulary.

The four files below are too large to commit, so the repository ships only their SHA-256
digests, embedded here. Verification is the default action, because a truncated or
mismatched backbone surfaces otherwise as an inscrutable failure deep inside model
construction.

    python scripts/fetch_assets.py              # verify what is on disk, fetch what is missing

Nothing is downloaded unless ALPHAWISE_ASSET_BASE_URL is set, in which case a missing
asset is fetched from <base url>/<name> and then verified like any other. No download
location is hardcoded, so this works for a private mirror as well as a public one.
Downloads land under a temporary name and are renamed into place only when complete, so an
interrupted transfer never leaves a half-written file that later runs mistake for a bad
asset. Assets that are symlinks verify like regular files; a dangling symlink counts as
missing and a download replaces the link itself.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphawise.utils.paths import assets_dir  # noqa: E402

BASE_URL_VAR = "ALPHAWISE_ASSET_BASE_URL"

#: The assets the package loads, with the digest each one must have. Files present in the
#: assets directory but absent here are left alone: they belong to some other experiment,
#: not to this release. See assets/README.md for where each file comes from.
EXPECTED = {
    "ViT_CLIP.pt": "9e468423360d2afa152d16f2176c8cc928e0014ffd65c695f2ec66a17b7e8fbb",
    "CLIP.pt": "124e6d59d54c0837c456c953b3147adf438a21570d5c5b01ad83f6ad3e78d15e",
    "ESRNXFBSP.pt": "acfcc6b0a0f07025cc660ed6dcf9b5a05fbd7c8ebf65edaa0a81d88a92ff6834",
    "bpe_simple_vocab_16e6.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
}


class DownloadError(RuntimeError):
    """A download that was attempted and did not complete."""


def digest(path: Path) -> str:
    """SHA-256 of a file, read in blocks so a multi-gigabyte checkpoint never lands in memory."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def download(name: str, target: Path) -> bool:
    """Fetch one asset from the configured base URL; False means no base URL was set.

    False says nothing was attempted. A failed or interrupted transfer raises
    :class:`DownloadError` naming the asset and the URL, and leaves no file at ``target``:
    the payload goes to a temporary sibling first and is renamed into place atomically
    only once the transfer finished.
    """
    base_url = os.environ.get(BASE_URL_VAR)
    if not base_url:
        return False
    url = f"{base_url.rstrip('/')}/{name}"
    print(f"fetching {name} from {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f"{name}.partial"
    try:
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, target)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"could not download {name} from {url} ({error}). Check {BASE_URL_VAR}, or "
            f"place the file at {target} by hand."
        ) from error
    return True


def verify(directory: Path) -> int:
    """Check every declared asset against its embedded digest; 0 only when all four match.

    Missing, mismatched and failed-to-download all count as failures. Every asset is
    reported before the return, so one run names every problem rather than only the
    first, and an asset that had to be downloaded is checked exactly like one that was
    already there.
    """
    failures = 0
    for name, expected in EXPECTED.items():
        asset = directory / name
        if not asset.exists():
            try:
                if not download(name, asset):
                    print(f"MISSING  {name}: not in {directory}, and {BASE_URL_VAR} is not set")
                    failures += 1
                    continue
            except DownloadError as error:
                print(f"FAILED   {error}")
                failures += 1
                continue
        if digest(asset) != expected:
            print(f"MISMATCH {name}: contents differ from the digest in {Path(__file__).name}")
            failures += 1
        else:
            print(f"ok       {name}")
    return 1 if failures else 0


def main() -> int:
    """Verify the assets and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets-dir", type=Path, default=None,
                        help="defaults to ALPHAWISE_ASSETS, then <repo>/assets")
    args = parser.parse_args()
    return verify(args.assets_dir or assets_dir())


if __name__ == "__main__":
    raise SystemExit(main())
