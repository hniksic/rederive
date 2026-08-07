# Releasing

`packaging/rederive.spec` builds the binaries and explains itself; this is the
procedure around it.

## Artifacts

Names carry no version, so that INSTALL.md's
`releases/latest/download/<name>` links keep working across releases.

| Platform | Single file | Archive | Installer |
| --- | --- | --- | --- |
| Linux x86_64 | `rederive-linux-x86_64` | `rederive-linux-x86_64.tar.gz` | - |
| macOS arm64 | `rederive-macos-arm64` | `rederive-macos-arm64.tar.gz` | - |
| Windows x86_64 | `rederive-windows-x86_64.exe` | `rederive-windows-x86_64.zip` | `rederive-setup.exe` |

Plus `install.sh`, which the Unix instructions pipe into `sh`, and
`SHA256SUMS` over everything.

## Procedure

Bump `__version__` in `src/rederive/__init__.py`, which is where the version is
written and where the build reads it from, commit, then tag and push:

```
git tag -a v<version> -m "Rederive <version>"
git push origin v<version>
```

The tag is annotated rather than lightweight, so that it carries a tagger, a date and
a message of its own - `git describe` considers no other kind, and the message shows
on the release page beside the generated notes.
`.github/workflows/build.yml` refuses a tag that does not match that version, then
builds on all three platforms, checks every build, compiles and test-installs the
Windows installer, and publishes the release.

The same tag sets `.github/workflows/web.yml` going, which builds the browser demo,
opens it in Chromium and Firefox, and publishes it to GitHub Pages. The two workflows
run beside each other and neither waits on the other: the page carries a wheel built
from the tagged source rather than anything the release attaches. That is what keeps
README's [live demo](https://hniksic.github.io/rederive/) link on the latest release
without the link itself ever changing.

To build by hand instead - there is no cross-compilation, so this is once per
platform, from a clean checkout of the tag. On Linux, install `libxcb-cursor0` first:
a bundle carries the libraries the build machine had, and without it Qt's xcb plugin
ships incomplete and the binary opens no window on X11.

The `full` extra is named because a release bundles the whole program. The plot
toolkit and the two libraries behind the clipboard and the memory gauge are opt-in
for an install from source, and a binary that quietly left them out would be a
release nobody asked for.

```
uv sync --frozen --extra full --group packaging
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm \
    --distpath dist/tree -- --tree
uv run python packaging/smoke.py dist/rederive
uv run python packaging/smoke.py dist/tree/rederive/rederive
```

Archive `dist/tree/rederive` so the archive holds one `rederive` directory - `tar -C
dist/tree -czf rederive-<platform>.tar.gz rederive`, or a zip on Windows. On Windows
the installer is then built from the same tree, with the version read out of the
package rather than typed - it is written in one place and this is one of the two
that ask for it, the tag being the other:

```
$version = uv run python -c "import rederive; print(rederive.__version__)"
iscc /DAppVersion=$version packaging\rederive.iss
```

Attach everything to the GitHub release.

## Notes

- `--frozen` matters: the test suite ran against `uv.lock`, so ship what it tested.
- `rederive --version` reports what a bundle carries rather than what the machine has
  installed, which is the only way to tell the two apart from outside. Python, sympy,
  Qt and pyqtgraph are all on it, and the workflow holds all four against
  `.python-version` and `uv.lock`; building by hand, run it and read it. It is also
  the first thing to ask for in a bug report - which Qt a bundle carries is where a
  plot that misbehaves is answered from.
- On Windows the smoke script only checks that the bundle unpacks and says what it
  carries, there being no pty to drive the rest through. What covers the rest there is
  the workflow's installer check, which installs, runs and uninstalls.
- No check anywhere draws a plot from a build: the runners have no display, and the
  suite draws offscreen but against the source tree. What `--version` says is that the
  toolkit imports. Before a release, open a 2D and a 3D plot from the binary by hand -
  the 3D one especially, since PyOpenGL finds its backend by a name no build can see.
- Nothing is signed, so macOS and Windows refuse a browser's download and SmartScreen
  warns about the installer. INSTALL.md's instructions work around that and have to
  stay in step with it. There is more of it to object to since Qt arrived - a bundle
  is several hundred libraries rather than one binary - so any signing done later is
  that much larger a job.
- The Linux binary needs glibc 2.34, which is what PySide6's wheels are built for -
  Ubuntu 22.04, Debian 12 and RHEL 9 and later. The interpreter comes from uv and
  would run on far older, so this is Qt's floor and not the build machine's.
- The macOS binary needs macOS 13, for the same reason. PySide6's wheels are
  `universal2`, so the arm64 bundle carries x86_64 code it never runs unless it is
  thinned.
- `--managed-python` and `.python-version` are what keep all three platforms on one
  interpreter. Without them uv takes whatever the machine offers, and a runner with
  its own Python installed will quietly build against that instead.
