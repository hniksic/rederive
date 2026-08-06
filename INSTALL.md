# Installing Rederive

Rederive releases ship as a single self-contained binary for Linux, macOS and Windows,
needing no Python and no dependencies. You can run that file as it is, install it
properly, or skip the binaries and run Rederive from source.

The binaries want a system from the last few years: Ubuntu 22.04, Debian 12 or RHEL 9
and newer on Linux, macOS 13 and newer, Windows 10 and newer. They are a bit over a
hundred megabytes to download. From source, Rederive asks for Python 3.11 and nothing
else.

## Trying it out

Download one file and run it. It needs no Python and installs nothing, and you uninstall
by just deleting the file.

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
Windows refuse to run unsigned programs that a *browser* downloaded. Windows 10 and
later ship `curl.exe`.

If you download through a browser anyway, you can undo the mark:

- macOS: `xattr -d com.apple.quarantine rederive-macos-arm64`, or open System Settings →
  Privacy & Security and press **Open Anyway** after the first refusal.
- Windows: on the SmartScreen warning, choose **More info** → **Run anyway**.

On Windows, run Rederive in [Windows Terminal](https://aka.ms/terminal) rather than the
old console window, which has neither the colors nor the mouse support Rederive expects.

## Installing it

The single file above unpacks itself into a temporary directory every time it starts -
a quarter of a gigabyte of it, most of it Qt, which costs a few seconds of every launch.
Installing avoids that, and is worth doing for anything past a first look.

**Windows:** download and run
[`rederive-setup.exe`](https://github.com/hniksic/rederive/releases/latest/download/rederive-setup.exe).
It is an ordinary installer and it adds Rederive to your `PATH` and to Add/Remove
Programs, where uninstalling it undoes both. SmartScreen might warn once before it starts,
due to the installer being unsigned; choose **More info** and **Run anyway**.

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

That gives you the whole program - `uv` installs the extras below along with it.

## Installing from PyPI, and what the extras are

Rederive is one package with two optional parts, because two of the things it does
need a desktop underneath them and the rest of it does not:

```
pip install rederive[full]
```

is the whole program. `pip install rederive` on its own is the algebra: authoring,
simplifying, solving, the typeset display and the worksheet files, with nothing that
asks anything of the machine it runs on. Two extras add the rest, and either can be
asked for by itself - `rederive[plot]`, `rederive[desktop]`, or `rederive[full]` for
both:

- **`plot`** is Qt, pyqtgraph and PyOpenGL - a few hundred megabytes, and what a plot
  window is drawn with. Without it the Plot command refuses in a sentence and says
  what to install; nothing else changes.
- **`desktop`** is psutil and pyperclip - what reads the memory figure on the status
  line and what hands a copied expression to the desktop's own clipboard. Without it
  the memory field is empty and Ctrl-C copies through the terminal alone, which is
  the road it takes over ssh anyway.

The released binaries above carry everything, so none of this applies to them.
