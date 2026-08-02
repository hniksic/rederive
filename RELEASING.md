# Releasing

`packaging/rederive.spec` builds the binaries and explains itself; this is the
procedure around it.

## Artifacts

Names carry no version, so that the README's
`releases/latest/download/<name>` links keep working across releases.

| Platform | Single file | Archive |
| --- | --- | --- |
| Linux x86_64 | `rederive-linux-x86_64` | `rederive-linux-x86_64.tar.gz` |
| macOS arm64 | `rederive-macos-arm64` | `rederive-macos-arm64.tar.gz` |
| Windows x86_64 | `rederive-windows-x86_64.exe` | `rederive-windows-x86_64.zip` |

Plus `SHA256SUMS` over all six.

## Procedure

Bump the version in `pyproject.toml`, commit, tag `v<version>`. Then on each of the
three platforms - there is no cross-compilation - from a clean checkout of the tag:

```
uv sync --frozen --group packaging
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm
uv run --group packaging pyinstaller packaging/rederive.spec --noconfirm \
    --distpath dist/tree -- --tree
uv run python packaging/smoke.py dist/rederive
uv run python packaging/smoke.py dist/tree/rederive/rederive
```

Archive `dist/tree/rederive` so the archive holds one `rederive` directory - `tar -C
dist/tree -czf rederive-<platform>.tar.gz rederive`, or a zip on Windows - and attach
everything to the GitHub release.

## Notes

- `--frozen` matters: the test suite ran against `uv.lock`, so ship what it tested.
- On Windows the smoke script only checks that the bundle unpacks, there being no pty
  to drive the rest through. Start that build by hand once per release.
- Nothing is signed, so macOS and Windows refuse a browser's download. The README's
  `curl` instructions work around that and have to stay in step with it.
- The Linux binary needs glibc 2.17, which comes from uv's interpreter rather than
  from the build machine - no old distribution or container needed.
- There is no CI yet.
