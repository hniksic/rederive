"""Entry point: `rederive` / `python -m rederive`."""

from __future__ import annotations

from rederive.ui.app import RederiveApp


def main() -> None:
    RederiveApp().run()


if __name__ == "__main__":
    main()
