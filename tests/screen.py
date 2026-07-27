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


def highlighted_rows(app):
    """The rows of the selection rectangle, trailing blanks stripped."""
    style = app.palette.styles["selection"]
    return [row.rstrip() for row in styled(app.query_one("#work-content", Static), style)]


def highlighted_expression(app):
    """The selected subexpression as it is drawn, one line per row."""
    return "\n".join(highlighted_rows(app))


def work_area(app):
    """The work area as lines of text, stripped of trailing blanks."""
    text = text_of(app.query_one("#work-content", Static))
    return [line.rstrip() for line in text.plain.splitlines()]


def message(app):
    return text_of(app.query_one("#message")).plain.strip()


def annotation(app):
    """The status line's left field: where the selected entry came from."""
    return text_of(app.query_one("#status")).plain.strip().split("  ")[0]


def flags(app):
    """The mode words the status line shows beside the memory field."""
    said = text_of(app.query_one("#status")).plain
    return [word for word in ("Ins", "Lin") if f" {word}" in said]


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
