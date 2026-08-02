# Releasing

A release is a set of files attached to a GitHub release, one pair per platform, built
by `packaging/rederive.spec`. Nobody who downloads one needs Python, uv, or anything
else; that is the whole point of shipping binaries, and it is what makes the process
below worth its trouble.

## What a release ships

Two forms of the same program, because the two things people want are different.

The **single file** is for trying Rederive. It is downloaded, made executable and run,
and it leaves nothing behind. It pays for that at every launch: one file has nowhere to
keep an unpacked interpreter, so it writes about 32 MiB to a temporary directory each
time it starts and removes it on the way out. That costs roughly a tenth of a second.

The **archive** is for keeping Rederive. It unpacks to a launcher with an `_internal`
directory beside it, which is the same thing the single file would unpack, except that
it stays unpacked. Nothing is written at run time and start-up is that tenth of a second
shorter. A symbolic link into `~/.local/bin` works: the launcher resolves its own real
path, so it finds `_internal` through the link, and so does the engine worker it spawns.

Asset names carry no version number, so that
`https://github.com/hniksic/rederive/releases/latest/download/rederive-linux-x86_64`
keeps working across releases and the README never has to be edited for a version bump:

| Platform | Single file | Archive |
| --- | --- | --- |
| Linux x86_64 | `rederive-linux-x86_64` | `rederive-linux-x86_64.tar.gz` |
| macOS arm64 | `rederive-macos-arm64` | `rederive-macos-arm64.tar.gz` |
| Windows x86_64 | `rederive-windows-x86_64.exe` | `rederive-windows-x86_64.zip` |

Three platforms, one architecture each, which is what covers nearly everyone: Apple has
sold nothing but arm64 for years, and the Intel macOS build machines are the ones that
cost money. Adding an architecture means adding a build machine and nothing else.

## Building

**There is no cross-compilation.** Each file has to be built on the system it is for,
because a frozen program contains that system's interpreter and that system's shared
libraries. Three machines, or three CI runners, or a very patient afternoon.

On each, from a clean checkout of the tag:

```
uv sync --frozen --group packaging
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm \
    --distpath dist/tree -- --tree
```

`--frozen` matters. Building from a resolution that is not `uv.lock` ships a set of
dependencies nobody tested, and the whole test suite ran against the locked one.

Then check both, because a bundle can build perfectly and still be missing a file it
only reads when the user asks for help:

```
uv run python packaging/smoke.py dist/rederive
uv run python packaging/smoke.py dist/tree/rederive/rederive
```

On Windows only the first of the smoke script's checks runs - driving a terminal needs
a pty, and Windows has none - so the Windows build is covered for unpacking and for
nothing above it. It says so when it runs. Until that gap is closed by other means,
start the Windows build by hand once per release and press a few keys.

Finally, package the archive. On Linux and macOS:

```
tar -C dist/tree -czf dist/rederive-<platform>.tar.gz rederive
```

and on Windows, zip `dist\tree\rederive` so that the archive holds a single `rederive`
directory. Generate `SHA256SUMS` over everything and attach that too.

## Signing, and what its absence costs

Nothing is signed. A Developer ID certificate for macOS is an annual fee, a Windows
certificate is a larger one, and neither has been paid.

The consequence lands on whoever downloads a release, so the README's instructions have
to stay in step with this section. A browser marks a downloaded file as having come from
the internet - a quarantine attribute on macOS, a mark-of-the-web stream on Windows -
and the system then refuses to run an unsigned program that carries the mark. Neither
`curl` nor `curl.exe` sets it, which is why every instruction in the README downloads
that way, and why the fallbacks for people who used a browser anyway are spelled out
there.

If a certificate is ever bought, macOS wants more than a signature: a bare executable
cannot have a notarization ticket stapled to it, so a signed release would have to ship
a `.dmg` or `.pkg`, or accept that Gatekeeper checks Apple's servers on first launch.

## Things worth not rediscovering

The **glibc floor is 2.17**, and it comes from the interpreter uv installs rather than
from the build machine, so the Linux build does not need an old distribution or a
manylinux container to be portable. Beyond glibc the binary wants only `libdl`, `libz`,
`libpthread` and the loader.

The spec bundles **all of sympy**, not the two thirds the import analysis finds. The
third it leaves out is nearly all parts this program never reaches, but a handful of
live algorithms are in there, reached by imports written inside the functions that use
them. The corpus does not reach any of them; this costs eight megabytes and buys not
having to find out in the field.

Frozen builds compute about **15% slower** than the same code run from an installed
package. It is not the unpacking - a directory build is exactly as slow - and it is not
the optimization level. It has not been explained, only measured.

## Doing it by hand, for now

There is no release workflow in CI yet. Until there is: bump the version in
`pyproject.toml`, commit, tag `v<version>`, build and check on each platform, and create
the GitHub release with all six files and `SHA256SUMS`. When CI does arrive it should do
exactly the above, and the smoke script is what tells it whether the build is worth
uploading.
