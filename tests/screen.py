"""Reading a running app's screen back the way the original's is read: as text,
plus which runs of it are in inverse video."""

from rich.text import Text
from textual.widgets import Static


def text_of(widget):
    """The Rich text a widget currently shows."""
    content = getattr(widget, "content", None)
    return content if isinstance(content, Text) else widget.render()


def styled(widget, style):
    """The runs of `widget`'s rendered text carrying `style`."""
    text = text_of(widget)
    return [
        text.plain[span.start : span.end] for span in text.spans if span.style == style
    ]


async def laid_out(pilot, app, identifier):
    """Wait until the widget `identifier` names has been given a size.

    Showing a widget or setting how tall it stands writes a style; the size that
    follows from it is the compositor's, and arrives some turns of the message
    loop later, on the screen's own idle. Until it does the widget measures
    nothing, so a test that reads `size` or `region` after a keystroke waits here
    for the layout rather than for the widget's queue to drain.
    """
    for _ in range(20):
        if app.query_one(identifier).size.height:
            return
        await pilot.pause()
    raise AssertionError(f"{identifier} was never laid out")


def band_id(app):
    """Which of the two bands is showing: a menu, or an Options dialog."""
    return "#fields" if app.editor is not None else "#menu"


def band(app):
    """The command band as lines of text, stripped of trailing blanks."""
    text = text_of(app.query_one(band_id(app)))
    return [line.rstrip() for line in text.plain.splitlines()]


def highlighted(app):
    """What the band shows in inverse video, or None if nothing is."""
    options = styled(app.query_one(band_id(app)), app.palette.styles["option-highlight"])
    return options[0] if options else None


def pointed(app):
    """The band word the pointer is marking, or None if it is marking none."""
    words = styled(app.query_one(band_id(app)), app.palette.styles["option-pointed"])
    return words[0] if words else None


def content(app, number=None):
    """The Static holding a window's expressions; the active window's by default."""
    pane = app.work_area if number is None else app.panes[app.windows.numbered(number)]
    return pane.query_one(".work-content", Static)


def highlighted_rows(app, number=None):
    """The rows of the selection rectangle, trailing blanks stripped."""
    style = app.palette.styles["selection"]
    return [row.rstrip() for row in styled(content(app, number), style)]


def highlighted_expression(app, number=None):
    """The selected subexpression as it is drawn, one line per row."""
    return "\n".join(highlighted_rows(app, number))


def work_area(app, number=None):
    """A window's expressions as lines of text, stripped of trailing blanks."""
    text = text_of(content(app, number))
    return [line.rstrip() for line in text.plain.splitlines()]


def frame(app):
    """The window borders as lines of text, the rule below them included."""
    lines = text_of(app.query_one("#frame")).plain.splitlines()
    return lines + [text_of(app.query_one("#rule")).plain]


def window_type(app):
    """The status line's right field: the product name and the window's type."""
    return text_of(app.query_one("#status")).plain.rstrip().split("  ")[-1].strip()


def message(app):
    return text_of(app.query_one("#message")).plain.strip()


def annotation(app):
    """The status line's left field: where the selected entry came from."""
    return text_of(app.query_one("#status")).plain.strip().split("  ")[0]


def flags(app):
    """The mode words the status line shows beside the memory field."""
    said = text_of(app.query_one("#status")).plain
    return [word for word in ("Ovr", "Lin") if f" {word}" in said]


def prompt(app):
    """The prompt band: what it asks for, and what is on the line."""
    label = text_of(app.query_one("#prompt-label")).plain
    return label.strip(), app.query_one("#prompt-input").value


def completions(app):
    """The names the file prompt's list is showing, or None if it is closed."""
    listing = app.query_one("#completions")
    if not listing.display:
        return None
    return [row.strip() for row in text_of(listing).plain.splitlines()]


def chosen(app):
    """The name that list is pointing at, or None while it points at none."""
    rows = styled(app.query_one("#completions"), app.palette.styles["selection"])
    return rows[0].strip() if rows else None


def listing_title(app):
    """What the border of that list says: the directory, and how much shows."""
    return app.query_one("#completions").border_title.strip()


def entries(app):
    return [entry.text for entry in app.session.entries]
