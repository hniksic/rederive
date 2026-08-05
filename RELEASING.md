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

To build by hand instead - there is no cross-compilation, so this is once per
platform, from a clean checkout of the tag:

```
uv sync --frozen --group packaging
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm \
    --distpath dist/tree -- --tree
uv run python packaging/smoke.py dist/rederive
uv run python packaging/smoke.py dist/tree/rederive/rederive
```

Archive `dist/tree/rederive` so the archive holds one `rederive` directory - `tar -C
dist/tree -czf rederive-<platform>.tar.gz rederive`, or a zip on Windows. On Windows,
`iscc /DAppVersion=<version> packaging\rederive.iss` then builds the installer from
the same tree. Attach everything to the GitHub release.

## Notes

- `--frozen` matters: the test suite ran against `uv.lock`, so ship what it tested.
- `rederive --version` reports what a bundle carries rather than what the machine has
  installed, which is the only way to tell the two apart from outside. The workflow
  holds it against `.python-version` and `uv.lock`; building by hand, run it and read
  it. It is also the first thing to ask for in a bug report.
- On Windows the smoke script only checks that the bundle unpacks and says what it
  carries, there being no pty to drive the rest through. What covers the rest there is
  the workflow's installer check, which installs, runs and uninstalls.
- Nothing is signed, so macOS and Windows refuse a browser's download and SmartScreen
  warns about the installer. INSTALL.md's instructions work around that and have to
  stay in step with it.
- The Linux binary needs glibc 2.17, which comes from uv's interpreter rather than
  from the build machine - no old distribution or container needed.
- `--managed-python` and `.python-version` are what keep all three platforms on one
  interpreter. Without them uv takes whatever the machine offers, and a runner with
  its own Python installed will quietly build against that instead.
