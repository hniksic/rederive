#!/usr/bin/env python3
"""Bump the version, commit it, tag it, and optionally push the two.

    python3 tools/release.py 1.2.1
    python3 tools/release.py 1.2.1 --push

This is the mechanical half of RELEASING.md's procedure - the half that is the
same every time and that a typo in makes a release nobody can undo. The version
is written in one place, `src/rederive/__init__.py`, and the tag has to name it
exactly, `build.yml` refusing one that does not; both come from the single
argument here so they cannot disagree.

The tag is annotated, for the tagger, date and message of its own that `git
describe` and the release page both want.

Everything else in the procedure - running the build by hand on master first,
opening a plot from the binary, attaching the artifacts - is still by hand, and
`--push` is what starts the workflow that does the rest, so it is opt-in.
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

# What a tag may be called, which is what the version may be. Deliberately no
# wider than the releases that exist: a version the packaging cannot express is
# better refused here than discovered by a workflow that has already been given
# a tag.
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
    """Refuse a dirty tree, which the version commit would otherwise sweep up.

    Untracked files are fine; the version commit only picks up tracked changes.
    """
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
    # The commit is scoped to the version file, so a tree that went dirty
    # between the check above and here cannot ride along in the release commit.
    git("commit", "--only", str(INIT), "-m", f"Release {version}")
    git("tag", "-a", tag, "-m", f"Rederive {version}")
    # Flushed, so that what this says stays in step with what the pushes below
    # write straight to the terminal.
    print(f"committed and tagged {tag}", flush=True)

    if not args.push:
        print(f"not pushed; run: git push {args.remote} master {tag}")
        return

    # Branch before tag: a tag whose commit is not yet on master is a release
    # built from a commit the branch does not have.
    git("push", args.remote, "master", capture=False)
    git("push", args.remote, tag, capture=False)
    print(f"pushed master and {tag} to {args.remote}")


if __name__ == "__main__":
    main()
