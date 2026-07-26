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


def highlighted_expression(app):
    style = app.palette.styles["selection"]
    return "".join(styled(app.query_one("#work-content", Static), style))


def message(app):
    return text_of(app.query_one("#message")).plain.strip()


def entries(app):
    return [entry.text for entry in app.session.entries]
