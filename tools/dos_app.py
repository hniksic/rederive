#!/usr/bin/env python3
"""Drive a DOS text-mode program under DOSBox and read its screen back as text.

A research instrument, not part of the application. Nothing in src/ or tests/
uses it. Needs `dosbox` on PATH (Debian/Ubuntu: `apt install dosbox`) and a copy
of the program to drive. Neither is bundled.

    python3 tools/dos_app.py --home DIR --run APP --repl
    python3 tools/dos_app.py --home DIR --run APP --keys a 'x+y{enter}'
    python3 tools/dos_app.py --home DIR --run APP --keys '{right}' --highlights

    with DosApp("/path/to/install", run="APP") as app:
        app.press("a")
        app.press("x (x + 1){enter}")
        print(app.highlights())      # [(15, 8, 'x + 1', '0x70')]
        app.press("p")               # program switches to a graphics mode
        app.capture("plot.png")      # ... and it can still be looked at

--home defaults to $DOSAPP_HOME. --run is the executable's base name, started
once the directory is mounted as C:. --require FILE refuses to start without it.

How: dosbox runs as a child with SDL_VIDEODRIVER=dummy, so no window appears.
The emulated VGA text buffer is read out of that process with
process_vm_readv(); keys go in by writing the guest's BIOS keyboard ring buffer
with process_vm_writev(). No X server, no OCR. The screen's address is found at
startup rather than hardcoded: a generated .COM stamps a random marker into
video memory and the marker is searched for.

highlights() is the useful one - a menu highlight or an inverse-video selection
is an *attribute*, invisible in the text alone.

Graphics: a second generated .COM enters mode 12h and stamps markers into each
of the four EGA/VGA bit planes, which locates DOSBox's emulated video RAM the
same way. That buffer is allocated once for the life of the emulator, so the
address stays good across the guest's own mode changes; capture() decodes the
planar bitmap for the current mode and writes a PNG. Planar 16-colour modes
(0Dh, 0Eh, 10h, 12h) and the chained 256-colour mode 13h are handled.

Limits:
  * read_screen() and friends need a text mode (0-3, 7); capture() needs one of
    the graphics modes above. is_text_mode() says which you are in. Starting in
    a graphics mode raises, because the settle logic reads characters.
  * Settling means the screen stopped changing, not that the program finished
    thinking. For anything slow use wait_for_text().
  * Keys arrive via the BIOS ring buffer, so they reach anything reading
    through INT 16h/INT 21h. A program polling port 60h directly will not.
  * dosbox must be a child of this process: kernel.yama.ptrace_scope=1 lets
    only an ancestor use process_vm_readv/writev.
"""

import ctypes
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib

COLS, ROWS = 80, 25
SCREEN_BYTES = COLS * ROWS * 2

# BIOS Data Area, guest physical addresses
BDA_KBD_HEAD = 0x41A
BDA_KBD_TAIL = 0x41C
BDA_KBD_START = 0x480
BDA_KBD_END = 0x482
BDA_CURSOR = 0x450
BDA_VIDMODE = 0x449


class DosError(RuntimeError):
    pass


# -- another process's memory ------------------------------------------------

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


class _IOVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


_libc.process_vm_readv.restype = ctypes.c_ssize_t
_libc.process_vm_writev.restype = ctypes.c_ssize_t


class ProcMem:
    def __init__(self, pid):
        self.pid = pid

    def read(self, addr, size):
        buf = (ctypes.c_char * size)()
        local = _IOVec(ctypes.cast(buf, ctypes.c_void_p), size)
        remote = _IOVec(ctypes.c_void_p(addr), size)
        n = _libc.process_vm_readv(self.pid, ctypes.byref(local), 1,
                                   ctypes.byref(remote), 1, 0)
        if n < 0:
            raise OSError(ctypes.get_errno(),
                          "process_vm_readv failed at 0x%x" % addr)
        return bytes(buf[:n])

    def try_read(self, addr, size):
        try:
            return self.read(addr, size)
        except OSError:
            return None

    def write(self, addr, data):
        buf = ctypes.create_string_buffer(data, len(data))
        local = _IOVec(ctypes.cast(buf, ctypes.c_void_p), len(data))
        remote = _IOVec(ctypes.c_void_p(addr), len(data))
        n = _libc.process_vm_writev(self.pid, ctypes.byref(local), 1,
                                    ctypes.byref(remote), 1, 0)
        if n != len(data):
            raise OSError(ctypes.get_errno(),
                          "process_vm_writev failed at 0x%x" % addr)

    def regions(self, max_size=256 << 20):
        out = []
        for line in open("/proc/%d/maps" % self.pid):
            f = line.split()
            if f[1][0] != "r":
                continue
            a, b = (int(x, 16) for x in f[0].split("-"))
            if b - a > max_size:
                continue
            out.append((a, b))
        return out

    def scan(self, needle, stride=1, phase=0):
        for a, b in self.regions():
            off = a
            while off < b:
                n = min(1 << 22, b - off)
                data = self.try_read(off, n)
                if data:
                    s = data[phase::stride] if stride > 1 else data
                    i = s.find(needle)
                    while i >= 0:
                        yield off + i * stride + phase
                        i = s.find(needle, i + 1)
                off += n


# -- keys --------------------------------------------------------------------

# scancodes for the unshifted US layout, indexed by the character produced
_SCAN_BY_CHAR = {}
for _row, _codes in [
    ("1234567890-=", range(0x02, 0x0E)),
    ("qwertyuiop[]", range(0x10, 0x1C)),
    ("asdfghjkl;'`", list(range(0x1E, 0x29)) + [0x29]),
    ("\\zxcvbnm,./", [0x2B] + list(range(0x2C, 0x36))),
]:
    for _c, _s in zip(_row, _codes):
        _SCAN_BY_CHAR[_c] = _s
_SCAN_BY_CHAR[" "] = 0x39

_SHIFT_PAIRS = dict(zip('!@#$%^&*()_+{}:"~|<>?', '1234567890-=[];\'`\\,./'))

NAMED_KEYS = {
    "enter": (0x1C, 0x0D), "return": (0x1C, 0x0D), "cr": (0x1C, 0x0D),
    "esc": (0x01, 0x1B), "escape": (0x01, 0x1B),
    "tab": (0x0F, 0x09), "backtab": (0x0F, 0x00), "shift-tab": (0x0F, 0x00),
    "bs": (0x0E, 0x08), "backspace": (0x0E, 0x08),
    "space": (0x39, 0x20),
    "up": (0x48, 0x00), "down": (0x50, 0x00),
    "left": (0x4B, 0x00), "right": (0x4D, 0x00),
    "home": (0x47, 0x00), "end": (0x4F, 0x00),
    "pgup": (0x49, 0x00), "pgdn": (0x51, 0x00),
    "ins": (0x52, 0x00), "insert": (0x52, 0x00),
    "del": (0x53, 0x00), "delete": (0x53, 0x00),
}
for _i in range(1, 11):
    NAMED_KEYS["f%d" % _i] = (0x3A + _i, 0x00)
NAMED_KEYS["f11"] = (0x85, 0x00)
NAMED_KEYS["f12"] = (0x86, 0x00)
for _i in range(1, 11):
    NAMED_KEYS["alt-f%d" % _i] = (0x67 + _i, 0x00)
for _c in "abcdefghijklmnopqrstuvwxyz":
    NAMED_KEYS["ctrl-" + _c] = (_SCAN_BY_CHAR.get(_c, 0), ord(_c) - 96)
for _c in "abcdefghijklmnopqrstuvwxyz":
    NAMED_KEYS["alt-" + _c] = (_SCAN_BY_CHAR.get(_c, 0), 0x00)

_KEY_RE = re.compile(r"\{([^{}]+)\}")


def parse_keys(spec):
    """'DIR{enter}' -> [(scan, ascii), ...]. '{{' is a literal '{'."""
    out = []
    i = 0
    while i < len(spec):
        if spec.startswith("{{", i):
            out.append((0x1A, ord("{")))
            i += 2
            continue
        m = _KEY_RE.match(spec, i)
        if m:
            name = m.group(1).strip().lower()
            if name not in NAMED_KEYS:
                raise KeyError("unknown key name %r" % name)
            out.append(NAMED_KEYS[name])
            i = m.end()
            continue
        ch = spec[i]
        i += 1
        if ch == "\n":
            out.append(NAMED_KEYS["enter"])
        elif ch == "\t":
            out.append(NAMED_KEYS["tab"])
        else:
            shifted = _SHIFT_PAIRS.get(ch, "")
            scan = _SCAN_BY_CHAR.get(ch.lower()) or _SCAN_BY_CHAR.get(shifted, 0)
            out.append((scan, ord(ch) & 0xFF))
    return out


VGA_COLORS = (
    "black", "blue", "green", "cyan", "red", "magenta", "brown", "lightgray",
    "darkgray", "lightblue", "lightgreen", "lightcyan", "lightred",
    "lightmagenta", "yellow", "white",
)


def color_name(attr):
    """'yellow-on-black' for a VGA attribute byte."""
    return f"{VGA_COLORS[attr & 15]}-on-{VGA_COLORS[(attr >> 4) & 7]}"


# -- graphics modes ----------------------------------------------------------

# The 16 colours a BIOS mode set leaves in the DAC, as RGB.
EGA_PALETTE = [
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
]

# mode -> (width, height, planes); planes 0 means the chained 256-colour mode
GRAPHICS_MODES = {
    0x0D: (320, 200, 4),
    0x0E: (640, 200, 4),
    0x0F: (640, 350, 2),
    0x10: (640, 350, 4),
    0x11: (640, 480, 1),
    0x12: (640, 480, 4),
    0x13: (320, 200, 0),
}

# byte -> the 8 pixels it contributes to plane p, as 8 bytes worth 0 or 1<<p
_BIT_TABLES = [
    [bytes(((b >> (7 - i)) & 1) << p for i in range(8)) for b in range(256)]
    for p in range(4)
]


def _or_bytes(a, b):
    return bytes(x | y for x, y in zip(a, b))


def write_png(path, width, height, pixels, palette):
    """An 8-bit indexed PNG. `pixels` is one byte per pixel, row-major."""
    def chunk(tag, data):
        c = tag + data
        return (struct.pack(">I", len(data)) + c +
                struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))

    raw = b"".join(b"\0" + pixels[y * width:(y + 1) * width]
                   for y in range(height))
    plte = b"".join(bytes(c) for c in palette)
    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)) +
           chunk(b"PLTE", plte) +
           chunk(b"IDAT", zlib.compress(raw, 9)) +
           chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    return path


def _gcalib_com(marks):
    """A .COM that enters mode 12h, stamps one marker into each bit plane at
    video offset 0, waits for a key, and restores text mode."""
    code = bytes([0xB8, 0x12, 0x00,        # mov ax,0012h   (640x480x16)
                  0xCD, 0x10,              # int 10h
                  0xB8, 0x00, 0xA0,        # mov ax,0A000h
                  0x8E, 0xC0,              # mov es,ax
                  0xFC])                   # cld
    per_plane = 18
    data = 0x100 + len(code) + 4 * per_plane + 11
    for p in range(4):
        src = data + 32 * p
        code += bytes([0xBA, 0xC4, 0x03,               # mov dx,3C4h
                       0xB8, 0x02, 1 << p,             # mov ax,(plane<<8)|02
                       0xEF,                           # out dx,ax (map mask)
                       0xBE, src & 0xFF, src >> 8,     # mov si,marker
                       0xBF, 0x00, 0x00,               # mov di,0
                       0xB9, 0x20, 0x00,               # mov cx,32
                       0xF3, 0xA4])                    # rep movsb
    code += bytes([0xB4, 0x00, 0xCD, 0x16,   # wait for a key
                   0xB8, 0x03, 0x00, 0xCD, 0x10,
                   0xCD, 0x20])
    assert len(code) + 0x100 == data, (len(code), data)
    return code + b"".join(marks)


def resolve_home(home=None, require=None):
    home = home or os.environ.get("DOSAPP_HOME")
    if not home:
        raise DosError("No install directory. Pass --home, or set $DOSAPP_HOME.")
    home = os.path.abspath(os.path.expanduser(home))
    if not os.path.isdir(home):
        raise DosError(f"Not a directory: {home}")
    if require and require.upper() not in {n.upper() for n in os.listdir(home)}:
        raise DosError(f"No {require} in {home}")
    return home


class DosApp:
    def __init__(self, home=None, run=None, require=None, boot_wait=6.0,
                 dosbox="dosbox", conf=None, verbose=False, graphics=True):
        if not run:
            raise DosError("No program named. Pass run= (or --run).")
        self.home = resolve_home(home, require)
        self.run = run
        self.boot_wait = boot_wait
        self.dosbox = dosbox
        self.conf = conf
        self.verbose = verbose
        self.graphics = graphics
        self.proc = None
        self.mem = None
        self.membase = None       # host address of guest physical 0
        self.screen_addr = None   # host address of the text buffer
        self.vram_addr = None     # host address of emulated video RAM
        self.vram_gap = 0         # bytes between planes, 0 = 4-byte interleave
        self.follow_page = True
        self._log = None
        self._caldir = None
        self._marker = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        # A .COM we generate stamps a random marker at B800:0000; finding it in
        # the dosbox process gives the host address of the top-left cell.
        marker = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
                         for _ in range(16))
        self._caldir = tempfile.mkdtemp(prefix="dosapp-cal-")
        with open(os.path.join(self._caldir, "CALIB.COM"), "wb") as f:
            f.write(bytes([0xEB, 0x10]) + marker.encode("ascii") +
                    bytes([0xB8, 0x00, 0xB8,        # mov ax,0B800h
                           0x8E, 0xC0,              # mov es,ax
                           0x31, 0xFF,              # xor di,di
                           0xBE, 0x02, 0x01,        # mov si,102h
                           0xB9, 0x10, 0x00,        # mov cx,16
                           0xAC,                    # lodsb
                           0xB4, 0x07,              # mov ah,7
                           0xAB,                    # stosw
                           0xE2, 0xFA,              # loop
                           0xCD, 0x20]))            # int 20h
        gmarks = [bytes(random.choice(b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
                        for _ in range(32)) for _ in range(4)]
        with open(os.path.join(self._caldir, "GCALIB.COM"), "wb") as f:
            f.write(_gcalib_com(gmarks))
        # PAUSE holds the autoexec until the marker has been found, so nothing
        # can scroll row 0 away first.
        cmds = ['mount d "%s"' % self._caldir, 'mount c "%s"' % self.home,
                "cls", "d:\\calib.com", "pause"]
        if self.graphics:
            cmds.append("d:\\gcalib.com")
        cmds += ["c:", "cls", self.run]
        argv = [self.dosbox]
        if self.conf:
            argv += ["-conf", self.conf]
        for c in cmds:
            argv += ["-c", c]
        env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                   SDL_VIDEO_X11_XRANDR="0")
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        self._log = open(os.devnull, "w")
        self.proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                     stdout=self._log, stderr=subprocess.STDOUT)
        self.mem = ProcMem(self.proc.pid)
        self._marker = marker
        self._locate(marker)
        self.send_keys("{enter}")          # release the PAUSE
        if self.graphics:
            self._locate_vram(gmarks)
            self.send_keys("{enter}")      # release GCALIB's key wait
        self.wait_settle(quiet=0.10, timeout=5.0)
        return self

    def _locate(self, marker, timeout=25.0):
        deadline = time.time() + timeout
        needle = marker.encode()
        while time.time() < deadline:
            time.sleep(0.3)
            if self.proc.poll() is not None:
                raise DosError("dosbox exited with %s" % self.proc.returncode)
            for a in self.mem.scan(needle, stride=2, phase=0):
                cells = self.mem.try_read(a, 32)
                if not cells or set(cells[1::2]) != {0x07}:
                    continue                # not a run of text cells
                self.screen_addr = a
                self._find_membase()
                if self.verbose:
                    print("screen @ 0x%x  membase @ 0x%x"
                          % (self.screen_addr, self.membase), file=sys.stderr)
                return
        raise DosError("timed out locating the DOSBox text buffer")

    def _locate_vram(self, marks, timeout=25.0):
        # DOSBox keeps emulated video RAM in one buffer for the life of the
        # emulator, so an address found now stays good across mode changes.
        deadline = time.time() + timeout
        while self.video_mode() != 0x12:
            if time.time() > deadline or self.proc.poll() is not None:
                raise DosError("mode 12h never came up; is machine= a VGA?")
            time.sleep(0.1)
        time.sleep(0.2)
        woven = bytes(marks[p][i] for i in range(32) for p in range(4))
        for a in self.mem.scan(woven):
            self.vram_addr, self.vram_gap = a, 0
            break
        else:
            # planes held apart rather than interleaved: find each one
            found = [next(self.mem.scan(m), None) for m in marks]
            if None in found:
                raise DosError("could not locate dosbox video RAM")
            gaps = {found[p + 1] - found[p] for p in range(3)}
            if len(gaps) != 1:
                raise DosError("video RAM planes are not evenly spaced")
            self.vram_addr, self.vram_gap = found[0], gaps.pop()
        if self.verbose:
            print("vram @ 0x%x gap %d" % (self.vram_addr, self.vram_gap),
                  file=sys.stderr)

    def recalibrate(self):
        self._locate(self._marker)

    def _find_membase(self):
        # Anchored on the BIOS date at 0xFFFF5, checked against the reset
        # vector, the 640K word and the keyboard ring bounds.
        for a in self.mem.scan(b"01/01/92"):
            mb = a - 0xFFFF5
            head = self.mem.try_read(mb + 0xFFFF0, 16)
            bda = self.mem.try_read(mb + 0x400, 0x90)
            if not head or not bda or len(bda) < 0x90:
                continue
            if head[0] != 0xEA or 0xFC not in head[13:16]:
                continue
            if int.from_bytes(bda[0x13:0x15], "little") != 640:
                continue
            if int.from_bytes(bda[0x80:0x82], "little") != 0x1E:
                continue
            self.membase = mb
            return
        raise DosError("could not locate guest RAM inside dosbox")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self._log:
            self._log.close()
            self._log = None
        if self._caldir:
            shutil.rmtree(self._caldir, ignore_errors=True)
            self._caldir = None

    def __enter__(self):
        self.start()
        # A splash screen is painted a beat after DOS hands over control, so
        # settle detection alone would return the DOS prompt.
        time.sleep(self.boot_wait)
        self.wait_settle(quiet=0.3, timeout=20)
        if not self.is_text_mode():
            mode = self.video_mode()
            self.stop()
            raise DosError(
                "video mode %d is a graphics mode, so there is no character "
                "buffer to read. If the program's own config selects it, copy "
                "the install, set a text mode in the copy, and drive that."
                % mode
            )
        return self

    def __exit__(self, *exc):
        self.stop()

    # -- screen ------------------------------------------------------------

    def geometry(self):
        """(cols, rows) from the BDA; handles 80x43/80x50."""
        cols = int.from_bytes(self.mem.read(self.membase + 0x44A, 2), "little")
        rows = self.mem.read(self.membase + 0x484, 1)[0] + 1
        if not 40 <= cols <= 132:
            cols = COLS
        if not 12 <= rows <= 60:
            rows = ROWS
        return cols, rows

    def video_mode(self):
        return self.mem.read(self.membase + BDA_VIDMODE, 1)[0]

    def is_text_mode(self):
        return self.video_mode() in (0, 1, 2, 3, 7)

    def page_offset(self):
        if not self.follow_page:
            return 0
        off = int.from_bytes(self.mem.read(self.membase + 0x44E, 2), "little")
        return off if off % 2 == 0 and off + SCREEN_BYTES <= 0x8000 else 0

    def raw_screen(self):
        cols, rows = self.geometry()
        return self.mem.read(self.screen_addr + self.page_offset(), cols * rows * 2)

    def read_screen(self, strip=True):
        """One string per row."""
        cols, rows = self.geometry()
        raw = self.raw_screen()
        lines = []
        for r in range(rows):
            row = raw[r * cols * 2:(r + 1) * cols * 2][0::2]
            text = row.decode("cp437").replace("\x00", " ")
            lines.append(text.rstrip() if strip else text)
        return lines

    def read_attrs(self):
        """One list of attribute bytes per row (bg<<4 | fg, bit7 = blink)."""
        cols, rows = self.geometry()
        raw = self.raw_screen()
        return [list(raw[r * cols * 2:(r + 1) * cols * 2][1::2]) for r in range(rows)]

    def read_screen_ansi(self):
        """Rows with SGR escapes, for eyeballing colour in a terminal."""
        cols, rows = self.geometry()
        raw = self.raw_screen()
        fg = [30, 34, 32, 36, 31, 35, 33, 37]
        out = []
        for r in range(rows):
            row = raw[r * cols * 2:(r + 1) * cols * 2]
            parts, last = [], None
            for c in range(cols):
                ch, at = row[c * 2], row[c * 2 + 1]
                if at != last:
                    parts.append("\033[0;%d;%dm" % (fg[at & 7] + (60 if at & 8 else 0),
                                                    fg[(at >> 4) & 7] + 10))
                    last = at
                parts.append(chr(ch).encode("cp437", "replace").decode("cp437")
                             if ch else " ")
            parts.append("\033[0m")
            out.append("".join(parts))
        return out

    # -- graphics ----------------------------------------------------------

    def is_graphics_mode(self):
        return self.video_mode() in GRAPHICS_MODES

    def _plane(self, index, count, size):
        if self.vram_gap:
            return self.mem.read(self.vram_addr + self.vram_gap * index, size)
        return self.mem.read(self.vram_addr, size * 4)[index::4]

    def pixels(self, mode=None):
        """(width, height, one index byte per pixel) for the current mode."""
        if self.vram_addr is None:
            raise DosError("no video RAM address; construct with graphics=True")
        mode = self.video_mode() if mode is None else mode
        if mode not in GRAPHICS_MODES:
            raise DosError("video mode %#x is not a graphics mode I decode"
                           % mode)
        width, height, planes = GRAPHICS_MODES[mode]
        if planes == 0:                                   # mode 13h, chain-4
            raw = self.mem.read(self.vram_addr, width * height * 4)
            return width, height, b"".join(raw[i:i + 4]
                                           for i in range(0, len(raw), 16))
        size = width // 8 * height
        out = None
        for p in range(planes):
            table = _BIT_TABLES[p]
            layer = b"".join(map(table.__getitem__, self._plane(p, planes, size)))
            out = layer if out is None else _or_bytes(out, layer)
        return width, height, out

    def capture(self, path, mode=None, palette=None):
        """Write the graphics screen to `path` as an indexed PNG."""
        width, height, data = self.pixels(mode)
        return write_png(path, width, height, data, palette or EGA_PALETTE)

    def cursor(self):
        """(row, col) of the hardware cursor on page 0."""
        d = self.mem.read(self.membase + BDA_CURSOR, 2)
        return d[1], d[0]

    def highlights(self, rows=None):
        """(row, col, text, attr) for each run whose attribute differs from the
        rest of its row - that is, what is selected right now."""
        lines, attrs, out = self.read_screen(strip=False), self.read_attrs(), []
        for row in (range(len(lines)) if rows is None else rows):
            if row >= len(attrs):
                continue
            common = max(set(attrs[row]), key=list(attrs[row]).count)
            run, start = "", None
            for col, char in enumerate(lines[row]):
                if attrs[row][col] != common:
                    if start is None:
                        start = col
                    run += char
                elif run:
                    out.append((row, start, run, hex(attrs[row][start])))
                    run, start = "", None
            if run:
                out.append((row, start, run, hex(attrs[row][start])))
        return out

    def palette(self, rows=None):
        """{colour name: sample text} for every colour on screen."""
        lines, attrs, seen = self.read_screen(strip=False), self.read_attrs(), {}
        for row in (range(len(lines)) if rows is None else rows):
            for col, char in enumerate(lines[row]):
                if char != " ":
                    seen.setdefault(color_name(attrs[row][col]), []).append(char)
        return {name: "".join(chars)[:60] for name, chars in seen.items()}

    def show(self, title="", rows=None, numbers=True):
        if title:
            print(f"--- {title} ---")
        if not self.is_text_mode():
            print("[video mode %#x: a graphics mode, capture() it]"
                  % self.video_mode())
            return self
        lines = self.read_screen(strip=False)
        for i in (range(len(lines)) if rows is None else rows):
            print((f"{i:2}|" if numbers else "") + lines[i].rstrip())
        return self

    # -- keyboard ----------------------------------------------------------

    def _rw(self, phys):
        return int.from_bytes(self.mem.read(self.membase + phys, 2), "little")

    def _ww(self, phys, val):
        self.mem.write(self.membase + phys, val.to_bytes(2, "little"))

    def _push_key(self, scan, ascii_, timeout=5.0):
        deadline = time.time() + timeout
        while True:
            start, end = self._rw(BDA_KBD_START), self._rw(BDA_KBD_END)
            tail, head = self._rw(BDA_KBD_TAIL), self._rw(BDA_KBD_HEAD)
            nxt = start if tail + 2 >= end else tail + 2
            if nxt != head:
                break                                   # room in the ring
            if time.time() > deadline:
                raise DosError("guest keyboard buffer stayed full")
            time.sleep(0.01)
        # write the entry, then publish it by advancing the tail
        self.mem.write(self.membase + 0x400 + tail,
                       bytes([ascii_ & 0xFF, scan & 0xFF]))
        self._ww(BDA_KBD_TAIL, nxt)

    def send_keys(self, spec, delay=0.012, drain=True, drain_timeout=3.0):
        """Send literal text plus {enter}, {f3}, {down}, {ctrl-c}..."""
        for scan, ascii_ in parse_keys(spec):
            self._push_key(scan, ascii_)
            if delay:
                time.sleep(delay)
        if not drain:
            return True
        deadline = time.time() + drain_timeout
        while time.time() < deadline:
            if self._rw(BDA_KBD_HEAD) == self._rw(BDA_KBD_TAIL):
                return True
            time.sleep(0.005)
        return False

    # -- waiting -----------------------------------------------------------

    def _frame(self):
        """Whatever the screen currently *is*, as bytes: characters in a text
        mode, video RAM in a graphics mode."""
        if self.vram_addr is not None and self.is_graphics_mode():
            width, height, planes = GRAPHICS_MODES[self.video_mode()]
            size = width * height // 8 * max(planes, 1)
            return self.mem.read(self.vram_addr, size * (4 if planes else 1))
        return self.raw_screen()

    def wait_settle(self, quiet=0.12, timeout=5.0, poll=0.02):
        """Block until the screen has been unchanged for `quiet` seconds."""
        t0 = last_change = time.time()
        last = self._frame()
        while True:
            time.sleep(poll)
            cur, now = self._frame(), time.time()
            if cur != last:
                last, last_change = cur, now
            elif now - last_change >= quiet:
                return now - t0
            if now - t0 > timeout:
                return now - t0

    def wait_change(self, before=None, timeout=5.0, poll=0.002):
        t0 = time.time()
        if before is None:
            before = self._frame()
        while time.time() - t0 < timeout:
            if self._frame() != before:
                return time.time() - t0
            time.sleep(poll)
        return None

    def wait_for_text(self, substr, timeout=10.0, poll=0.03):
        """The row `substr` appears on, or None."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            for i, line in enumerate(self.read_screen()):
                if substr in line:
                    return i
            time.sleep(poll)
        return None

    def press(self, spec, quiet=0.2):
        """send_keys + wait_settle."""
        self.send_keys(spec)
        self.wait_settle(quiet=quiet, timeout=6)
        return self


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--home", metavar="DIR",
                        help="install directory (else $DOSAPP_HOME)")
    parser.add_argument("--run", required=True, metavar="NAME",
                        help="base name of the executable to start")
    parser.add_argument("--require", metavar="FILE",
                        help="refuse to start unless FILE is in the directory")
    parser.add_argument("--boot-wait", type=float, default=6.0, metavar="SECONDS",
                        help="how long to let the program paint before reading")
    parser.add_argument("--keys", nargs="*", default=[], metavar="SPEC",
                        help="key specs to send in order, e.g. a 'x+y{enter}'")
    parser.add_argument("--repl", action="store_true",
                        help="prompt for key specs, dump the screen after each")
    parser.add_argument("--highlights", action="store_true",
                        help="also report which cells are in inverse video")
    parser.add_argument("--palette", action="store_true",
                        help="also report every colour on screen")
    parser.add_argument("--no-graphics", action="store_true",
                        help="skip video RAM calibration (a little faster)")
    parser.add_argument("--capture", metavar="FILE.png",
                        help="write the graphics screen as a PNG and exit")
    args = parser.parse_args(argv[1:])

    try:
        app = DosApp(args.home, run=args.run, require=args.require,
                     boot_wait=args.boot_wait, graphics=not args.no_graphics)
        with app:
            for spec in args.keys:
                app.press(spec)
            if args.repl:
                shots = 0
                app.show()
                while True:
                    try:
                        line = input("keys> ")
                    except EOFError:
                        break
                    if line.strip() in (".quit", ".q"):
                        break
                    if line.startswith(".png"):
                        name = (line.split(None, 1) + ["shot%d.png" % shots])[1]
                        print("wrote", app.capture(name.strip()))
                        shots += 1
                        continue
                    app.press(line)
                    app.show()
                    if args.highlights and app.is_text_mode():
                        print("highlights:", app.highlights())
                return 0
            if args.capture:
                print("wrote", app.capture(args.capture))
                return 0
            app.show()
            if args.highlights:
                print("highlights:", app.highlights())
            if args.palette:
                for name, sample in sorted(app.palette().items()):
                    print(f"{name:28} {sample!r}")
    except DosError as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
