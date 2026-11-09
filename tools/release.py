#!/usr/bin/env python3
"""Bump the version, commit it, tag it, and optionally push the two.

    python3 tools/release.py 1.2.1
    python3 tools/release.py 1.2.1 --push

This automates the mechanical, typo-prone part of RELEASING.md's procedure. The
version lives in `src/rederive/__init__.py`, and `build.yml` rejects a tag that doesn't
match it exactly; both come from the single argument here, so they can't disagree.

The tag is annotated, since `git describe` and the release page want its tagger, date
and message.

The rest of the procedure (running the build on master first, attaching artifacts) is
manual. `--push` starts the release workflow, so it is opt-in.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "src" / "rederive" / "__init__.py"

VERSION_RE = re.compile(r'^__version__ = "([^"]*)"$', re.MULTILINE)

# Allowed versions, deliberately no broader than existing releases: a version the
# packaging can't express is better refused here than after a tag has been pushed.
ALLOWED_RE = re.compile(r"^\d+\.\d+(\.\d+)?([ab]|rc)?\d*$")


def git(*args: str, capture: bool = True) -> str:
    """Run git in the repository, failing the script if it fails."""
    result = subprocess.run(
        ("git", "-C", str(ROOT)) + args,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        die(f"git {' '.join(args)} failed: {message}" if message else f"git {' '.join(args)} failed")
    return result.stdout.strip() if capture else ""


def die(message: str) -> None:
    print(f"release: {message}", file=sys.stderr)
    sys.exit(1)


def check_master() -> None:
    """Refuse to release from anywhere but master."""
    # symbolic-ref rather than `branch --show-current`, which is silent rather
    # than failing on a detached HEAD.
    head = subprocess.run(
        ("git", "-C", str(ROOT), "symbolic-ref", "--quiet", "--short", "HEAD"),
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head:
        die("HEAD is detached; check out master to release")
    if head != "master":
        die(f"on branch {head}; check out master to release")


def check_clean() -> None:
    """Refuse to release from a tree with uncommitted changes to tracked files."""
    if git("status", "--porcelain", "--untracked-files=no"):
        die("working tree has changes; commit or stash them first")


def bump(version: str) -> None:
    """Write the new version into the one file that holds it."""
    text = INIT.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if match is None:
        die(f"no __version__ assignment found in {INIT}")
    if match.group(1) == version:
        die(f"version is already {version}")
    INIT.write_text(
        text[: match.start()] + f'__version__ = "{version}"' + text[match.end() :],
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the new version, without the leading v")
    parser.add_argument(
        "--push",
        action="store_true",
        help="push master and the tag to origin, which starts the release workflow",
    )
    parser.add_argument(
        "--remote", default="origin", help="the remote to push to (default: origin)"
    )
    args = parser.parse_args()

    version = args.version.removeprefix("v")
    if not ALLOWED_RE.match(version):
        die(f"{args.version!r} does not look like a version")
    tag = f"v{version}"

    check_master()
    check_clean()
    if git("tag", "--list", tag):
        die(f"tag {tag} already exists")

    bump(version)
    # Commit only the version file, in case the tree changed since the check above.
    git("commit", "--only", str(INIT), "-m", f"Release {version}")
    git("tag", "-a", tag, "-m", f"Rederive {version}")
    # Flush so this appears before the output of the pushes below.
    print(f"committed and tagged {tag}", flush=True)

    if not args.push:
        print(f"not pushed; run: git push {args.remote} master {tag}")
        return

    # Push the branch first, so the tagged commit is on master before the release
    # is built.
    git("push", args.remote, "master", capture=False)
    git("push", args.remote, tag, capture=False)
    print(f"pushed master and {tag} to {args.remote}")


if __name__ == "__main__":
    main()
