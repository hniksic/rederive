# Installing Rederive

Rederive ships as a single self-contained binary for Linux, macOS and Windows, needing no
Python and no dependencies. You can run that file as it is, install it properly, or skip
the binaries and run Rederive from source.

## Trying it out

Download one file and run it. It needs no Python and installs nothing, and deleting the
file is the whole of uninstalling.

**Linux:**

```
curl -LO https://github.com/hniksic/rederive/releases/latest/download/rederive-linux-x86_64
chmod +x rederive-linux-x86_64
./rederive-linux-x86_64
```

**macOS (Apple Silicon):**

```
curl -LO https://github.com/hniksic/rederive/releases/latest/download/rederive-macos-arm64
chmod +x rederive-macos-arm64
./rederive-macos-arm64
```

**Windows:**

```
curl.exe -LO https://github.com/hniksic/rederive/releases/latest/download/rederive-windows-x86_64.exe
.\rederive-windows-x86_64.exe
```

The download goes through `curl` on purpose. Rederive's binaries are not code-signed -
signing certificates cost money that a hobby project has not spent - and both macOS and
Windows refuse to run unsigned programs that a *browser* downloaded. The refusal is
about the mark the browser attaches, not about the file: `curl` attaches no mark, so a
binary fetched this way simply runs. (`curl.exe` ships with Windows 10 and later.)

If you download through a browser anyway, you can undo the mark:

- macOS: `xattr -d com.apple.quarantine rederive-macos-arm64`, or open System Settings →
  Privacy & Security and press **Open Anyway** after the first refusal.
- Windows: on the SmartScreen warning, choose **More info** → **Run anyway**.

On Windows, run Rederive in [Windows
Terminal](https://aka.ms/terminal) rather than the old console window, which has neither
the colours nor the mouse support Rederive expects.

## Installing it

The single file above unpacks itself into a temporary directory every time it starts,
which costs about a tenth of a second. Installing avoids that, and puts `rederive` on
your `PATH` so it starts by name from anywhere.

**Windows:** download and run
[`rederive-setup.exe`](https://github.com/hniksic/rederive/releases/latest/download/rederive-setup.exe).
It is an ordinary installer - next, next, finish - and it adds Rederive to your `PATH`
and to Add/Remove Programs, where uninstalling it undoes both. SmartScreen warns once
before it starts, the installer being unsigned; choose **More info** → **Run anyway**.

**Linux and macOS:**

```
curl -LsSf https://github.com/hniksic/rederive/releases/latest/download/install.sh | sh
```

That unpacks Rederive into `~/.local/share/rederive`, links it into `~/.local/bin`, and
says so if that directory is not on your `PATH`. Options go through `sh` - `sh -s --
--prefix /opt --bin /usr/local/bin` - and `sh -s -- --uninstall` removes it again.

## Running from source

1. Download Rederive with `git clone https://github.com/hniksic/rederive`, or [grab the
   ZIP](https://github.com/hniksic/rederive/archive/refs/heads/master.zip) and unpack it.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. In the `rederive` directory, run `uv run rederive`.
