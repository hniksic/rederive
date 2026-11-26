"""The help: the topic files and their index, and the dialog that shows them."""

from collections import Counter

import pytest
from screen import band, highlighted, message, prompt, work_area
from textual.events import MouseScrollDown
from textual.widgets import Input
from textual.widgets.markdown import Markdown

from rederive import library
from rederive.model import help as helps
from rederive.model.settings import DIALOGS, Settings
from rederive.syntax import DeriveSyntaxError, ParseState, names, parse_expression
from rederive.ui import menu as menus
from rederive.ui.app import MODE_HELP, RederiveApp
from rederive.ui.help import COPIED, MARKER, HelpPage, HelpScreen, HelpTable, TopicList
from rederive.ui.theme import EGA

DOCUMENT = helps.document()

#: Every built-in that has a home row: the constants, functions and word operators,
#: without the old names that are aliases of current ones.
BUILT_IN = (
    names.BUILTIN_FUNCTIONS
    | names.PROGRAM_FUNCTIONS
    | names.CONSTANTS
    | names.KEYWORD_OPERATORS
    | names.WORD_OPERATORS
    | names.DISPLACEABLE_OPERATORS
) - set(helps.ALIASES)


@pytest.fixture
def app():
    return RederiveApp()


# -- the document ----------------------------------------------------------


def test_the_topics_are_read_in_order_each_with_an_intro_page():
    assert [topic.stem for topic in DOCUMENT.topics] == list(helps.ORDER)
    assert DOCUMENT.pages[0].address == helps.START
    for topic in DOCUMENT.topics:
        assert topic.intro.address == f"help:{topic.stem}"
        assert len(topic.pages) > 1
        # The intro page leads to each of the topic's pages.
        assert helps.links(topic.intro.source) == [
            page.address for page in topic.pages[1:]
        ]


def test_a_page_address_is_its_topic_and_slug():
    assert helps.slug("Numbers, input and display") == "numbers-input-and-display"
    page = DOCUMENT.page("help:functions/vectors-and-matrices")
    assert page is not None
    assert page.crumbs == ("Functions", "Vectors and matrices")
    assert DOCUMENT.page(helps.LINE_EDITING) is not None


def test_a_topic_file_is_divided_at_its_sub_topic_headings_only():
    topic = helps.read_topic(
        "stem",
        "# Title\n\nIntro.\n\n## One\n\nText.\n\n```\n## not a heading\n```\n\n"
        "## Two, too\n\nMore.\n",
    )
    assert [page.address for page in topic.pages] == [
        "help:stem",
        "help:stem/one",
        "help:stem/two-too",
    ]
    assert "## not a heading" in topic.pages[1].source
    assert topic.intro.source.startswith("Intro.")


def test_pages_are_read_in_order_across_topics():
    last = DOCUMENT.topics[0].pages[-1]
    assert DOCUMENT.step(last.address, 1) is DOCUMENT.topics[1].intro
    assert DOCUMENT.step(DOCUMENT.topics[1].intro.address, -1) is last
    assert DOCUMENT.step(DOCUMENT.pages[0].address, -1) is None
    assert DOCUMENT.step(DOCUMENT.pages[-1].address, 1) is None


def test_every_link_lands_somewhere():
    for page in DOCUMENT.pages:
        for href in helps.links(page.source):
            assert DOCUMENT.resolve(href) is not None, (page.address, href)


def test_a_name_link_lands_on_the_home_row_however_it_is_spelled():
    place = DOCUMENT.resolve("name:PRIME?")
    assert place is not None and place.row is not None
    assert DOCUMENT.resolve("name:PRIME%3F") == place
    assert DOCUMENT.resolve("name:prime?") == place
    # An old name lands where the current one does.
    assert DOCUMENT.resolve("name:DIMENSION") == DOCUMENT.resolve("name:DIM")
    assert DOCUMENT.resolve("name:NO_SUCH_NAME") is None
    assert DOCUMENT.resolve("help:no/such-page") is None


def _homes() -> Counter[str]:
    """How many home rows each name has."""
    homes: Counter[str] = Counter()
    for row in DOCUMENT.rows:
        if row.header not in (
            helps.COMMAND_HEADER,
            helps.KEY_HEADER,
            helps.AREA_HEADER,
        ):
            homes.update(helps.row_names(row))
    return homes


def test_every_built_in_name_has_exactly_one_home_row():
    homes = _homes()
    assert sorted(name for name in BUILT_IN if homes[name] == 0) == []
    assert sorted(name for name in BUILT_IN if homes[name] > 1) == []


def test_every_alias_lands_on_the_row_of_the_name_that_replaced_it():
    for alias, current in helps.ALIASES.items():
        assert DOCUMENT.resolve(f"name:{alias}") == DOCUMENT.names[current.casefold()]


def _menu_paths() -> dict[str, list[str]]:
    """Each command-menu path, with the paths of the items its menu holds."""
    paths: dict[str, list[str]] = {}

    def walk(menu: menus.Menu, prefix: str) -> None:
        for word in menu.words:
            path = f"{prefix} {word}".strip()
            target = menus.TARGETS.get(menu, {}).get(word)
            paths[path] = []
            if prefix:
                paths[prefix].append(path)
            if isinstance(target, menus.Menu):
                walk(target, path)

    walk(menus.ALGEBRA, "")
    return paths


def test_every_command_is_covered_by_one_command_row():
    paths = _menu_paths()
    rows = Counter(
        row.text for row in DOCUMENT.rows if row.header == helps.COMMAND_HEADER
    )
    assert [path for path, count in rows.items() if count > 1] == []
    assert sorted(set(rows) - set(paths)) == []

    def covered(path: str) -> bool:
        words = path.split()
        if any(" ".join(words[:end]) in rows for end in range(1, len(words) + 1)):
            return True
        children = paths[path]
        return bool(children) and all(covered(child) for child in children)

    assert [path for path in paths if not covered(path)] == []


def test_the_utility_library_table_indexes_library_names_but_homes_none():
    page = DOCUMENT.library_page
    assert page is not None
    catalan = DOCUMENT.areas["catalan"]
    assert catalan.address == page.address and catalan.row is not None
    assert "catalan" not in DOCUMENT.names
    # Every name the table lists is one the library defines.
    defined = {name.casefold() for name in library.NAMES}
    assert sorted(set(DOCUMENT.areas) - defined) == []


def test_every_library_name_is_found_by_search():
    found = {entry.label for entry in DOCUMENT.entries if entry.kind == "library"}
    homed = {name for name in library.NAMES if name.casefold() in DOCUMENT.names}
    assert found | homed == set(library.NAMES)


def test_a_library_name_the_table_omits_lands_on_its_group_s_row():
    # BERNOULLI_POLY is not listed, but CATALAN and BERNOULLI, from the same group, are.
    assert library.GROUPS["BERNOULLI_POLY"] == library.GROUPS["CATALAN"]
    assert "bernoulli_poly" not in DOCUMENT.areas
    [entry] = [entry for entry in DOCUMENT.search("BERNOULLI_POLY")]
    assert entry.place == DOCUMENT.areas["catalan"]


def test_search_finds_a_name_first_by_its_exact_spelling():
    results = DOCUMENT.search("dif")
    assert results[0].label == "DIF"
    assert results[0].place == DOCUMENT.resolve("name:DIF")


def test_search_finds_an_old_name_and_says_what_replaced_it():
    [result] = [entry for entry in DOCUMENT.search("dimension")]
    assert (result.kind, result.note) == ("alias", "DIM")
    assert result.place == DOCUMENT.resolve("name:DIM")


def test_search_finds_commands_and_pages():
    labels = [(entry.kind, entry.label) for entry in DOCUMENT.search("integrate")]
    assert ("command", "Calculus Integrate") in labels
    assert ("page", "Calculus") in [
        (entry.kind, entry.label) for entry in DOCUMENT.search("calculus")
    ]
    assert DOCUMENT.search("  ") == []


def test_the_settings_page_has_a_table_per_screen_as_the_settings_stand():
    page = DOCUMENT.page("help:settings/current-settings")
    assert page is not None and helps.SETTINGS in page.source
    settings = Settings()
    text = DOCUMENT.text(page, settings)
    assert helps.SETTINGS not in text
    headers = [line for line in text.splitlines() if line.endswith("| Name | Value |")]
    assert len(headers) == len(DIALOGS)
    assert "| Options Precision | Name | Value |" in headers
    assert "| Mode | `Precision` | Exact |" in text
    # A setting no assignment reaches has no name.
    assert "| Active |  | No |" in text
    settings.apply({"Precision": "Approximate"})
    assert "| Mode | `Precision` | Approximate |" in DOCUMENT.text(page, settings)


def test_the_items_of_a_page_are_its_examples_programs_and_links_in_order():
    source = (
        "See [A](help:a).\n\n"
        "```examples\nF(1)      a remark\n          that goes on\nG(2)\n```\n\n"
        "```\nnot(copied)  either\n```\n\n"
        "```program\nP(x) :=\n  x + 1\n\nQ := 1\n```\n\n"
        "| Name | Meaning |\n|---|---|\n| `B` | see [B](name:B) |\n"
    )
    assert helps.items(source) == [
        helps.Item("link", "help:a"),
        helps.Item("example", "F(1)"),
        helps.Item("example", "G(2)"),
        helps.Item("program", "P(x) :=\n  x + 1"),
        helps.Item("program", "Q := 1"),
        helps.Item("link", "name:B"),
    ]


def test_an_example_takes_up_the_lines_of_its_remark():
    code = "F(1)      a remark\n          that goes on\nG(x, y)  b\nH(1)\n"
    assert helps.snippets("examples", code) == [
        helps.Snippet("F(1)", (0, 2)),
        helps.Snippet("G(x, y)", (2, 3)),
        helps.Snippet("H(1)", (3, 4)),
    ]
    assert helps.snippets("", code) == []


def test_every_copyable_item_in_the_help_parses():
    failed = []
    for page in DOCUMENT.pages:
        for item in helps.items(page.source):
            if item.kind == "link":
                continue
            try:
                parse_expression(item.text, ParseState())
            except DeriveSyntaxError as error:
                failed.append((page.address, item.text, str(error)))
    assert failed == []
    # The files do mark blocks of both kinds.
    kinds = {item.kind for page in DOCUMENT.pages for item in helps.items(page.source)}
    assert kinds == {"example", "program", "link"}


# -- the dialog ------------------------------------------------------------


async def idle(pilot, turns=10):
    """Let the page wanted load and be laid out.

    A page is parsed off the loop, which under load can take longer than any fixed
    number of turns, so this waits for the document to be free, then for the layout.
    """
    for _ in range(500):
        screen = pilot.app.screen
        if not (
            isinstance(screen, HelpScreen)
            and (screen.fetching or screen.markdown.lock.is_locked)
        ):
            turns -= 1
            if turns <= 0:
                return
        await pilot.pause()


def dialog(app) -> HelpScreen:
    assert isinstance(app.screen, HelpScreen)
    return app.screen


def screen_lines(app):
    strips = app.screen._compositor.render_strips()
    return ["".join(segment.text for segment in strip).rstrip() for strip in strips]


def shown(app):
    return "\n".join(screen_lines(app))


async def opened(pilot, app, place=None):
    if place is not None:
        app.help_place = helps.Place(place)
    await pilot.press("h")
    await idle(pilot)
    return dialog(app)


async def test_help_opens_over_the_worksheet_and_closes_back_to_the_menu(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x+1", "enter")
        help = await opened(pilot, app)
        assert app.mode == MODE_HELP
        assert help.address == helps.START
        assert "Basics" in shown(app) and "Pages in this topic" in shown(app)
        await pilot.press("escape")
        await idle(pilot)
        assert not isinstance(app.screen, HelpScreen)
        assert work_area(app) == ["#1:  x + 1"]
        assert band(app)[0].startswith(" COMMAND:")
        assert highlighted(app) == "Author"
        assert message(app) == "Enter option"


async def test_help_opens_again_where_it_was_left(app):
    async with app.run_test() as pilot:
        help = await opened(pilot, app)
        await pilot.press("down", "down")
        await idle(pilot)
        address = help.address
        assert address != helps.START
        await pilot.press("escape")
        await idle(pilot)
        assert (await opened(pilot, app)).address == address


async def test_up_and_down_walk_closed_topics_without_opening_them(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        topics = DOCUMENT.topics
        assert len(help.tree.options) == len(topics)
        # Up on the first line stays there: the highlight doesn't wrap around.
        await pilot.press("up")
        await idle(pilot)
        assert help.address == topics[0].intro.address
        await pilot.press("down")
        await idle(pilot)
        assert help.address == topics[1].intro.address
        assert len(help.tree.options) == len(topics)
        await pilot.press(*["down"] * (len(topics) + 3))
        await idle(pilot)
        assert help.address == topics[-1].intro.address
        assert len(help.tree.options) == len(topics)


async def test_right_and_enter_open_a_topic_and_left_closes_it(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        topics = DOCUMENT.topics
        for key in ("right", "enter"):
            # The first press opens the topic, the second moves on to the page.
            await pilot.press(key)
            await idle(pilot)
            assert isinstance(app.focused, TopicList)
            assert len(help.tree.options) == len(topics) + len(topics[0].pages) - 1
            await pilot.press(key)
            assert app.focused is help.page
            await pilot.press("left")
            assert isinstance(app.focused, TopicList)
            await pilot.press("left")
            await idle(pilot)
            assert len(help.tree.options) == len(topics)
        # Left on one of an open topic's pages closes it and shows the topic's page.
        await pilot.press("right", "down", "down")
        await idle(pilot)
        assert help.address == topics[0].pages[2].address
        await pilot.press("left")
        await idle(pilot)
        assert help.address == topics[0].intro.address
        assert len(help.tree.options) == len(topics)


async def test_walking_on_from_an_open_topic_opens_the_next(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        first, second = DOCUMENT.topics[:2]
        await pilot.press("right")
        await pilot.press(*["down"] * len(first.pages))
        await idle(pilot)
        assert help.address == second.intro.address
        options = [str(option.prompt) for option in help.tree.options]
        assert any(second.pages[1].title in option for option in options)
        assert not any(first.pages[1].title in option for option in options)
        await pilot.press("up")
        await idle(pilot)
        assert help.address == first.intro.address
        options = [str(option.prompt) for option in help.tree.options]
        assert any(first.pages[1].title in option for option in options)


async def test_q_closes_help_but_is_typed_in_the_search_box(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        await pilot.press("slash", "q")
        assert help.search.value == "q"
        await pilot.press("escape", "q")
        await idle(pilot)
        assert not isinstance(app.screen, HelpScreen)


async def test_the_tree_shows_each_page_it_reaches_in_an_open_topic(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        topic = DOCUMENT.topics[0]
        await pilot.press("right", "down")
        await idle(pilot)
        assert help.address == topic.pages[1].address
        assert topic.pages[1].title in shown(app)
        options = [str(option.prompt) for option in help.tree.options]
        assert options[0].startswith("\N{BLACK DOWN-POINTING SMALL TRIANGLE} ")
        assert any(topic.pages[2].title in option for option in options)
        # Another topic's pages are not listed.
        other = DOCUMENT.topics[1]
        assert not any(other.pages[1].title in option for option in options)


async def test_enter_moves_into_the_page_and_left_back_to_the_tree(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        await pilot.press("enter", "enter")
        assert app.focused is help.page
        await pilot.press("left")
        assert isinstance(app.focused, TopicList)
        await pilot.press("tab")
        assert app.focused is help.page
        await pilot.press("shift+tab")
        assert isinstance(app.focused, TopicList)


async def test_paging_past_the_end_reads_on_into_the_next_page(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app, "help:basics/first-steps")
        await pilot.press("enter")
        for _ in range(10):
            if help.address != "help:basics/first-steps":
                break
            await pilot.press("pagedown")
            await idle(pilot)
        assert help.address == "help:basics/menus-and-the-highlight"
        assert help.page.scroll_y == 0
        # Paging up from the top goes back, to the end of the previous page.
        await pilot.press("pageup")
        await idle(pilot)
        assert help.address == "help:basics/first-steps"
        assert help.page.scroll_y == help.page.max_scroll_y


async def test_backspace_goes_back_through_the_pages_visited(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        await pilot.press("right", "down")
        await idle(pilot)
        await pilot.press("down")
        await idle(pilot)
        visited = [page.address for page in DOCUMENT.pages[:3]]
        assert help.address == visited[2]
        await pilot.press("backspace")
        await idle(pilot)
        assert help.address == visited[1]
        await pilot.press("backspace")
        await idle(pilot)
        assert help.address == visited[0]
        await pilot.press("backspace")
        await idle(pilot)
        assert help.address == visited[0]


async def test_a_link_leads_to_its_page_or_its_row(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        help.markdown.post_message(
            Markdown.LinkClicked(help.markdown, "help:functions/calculus")
        )
        await idle(pilot)
        assert help.address == "help:functions/calculus"
        help.markdown.post_message(Markdown.LinkClicked(help.markdown, "name:VECTOR"))
        await idle(pilot)
        assert help.address == DOCUMENT.names["vector"].address
        assert help.marked and help.marked[0].startswith("VECTOR(")
        await pilot.press("backspace")
        await idle(pilot)
        assert help.address == "help:functions/calculus"


async def test_a_click_on_a_link_follows_it(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        page = DOCUMENT.topics[0].pages[1]
        # The intro page's first link, on the line after the heading of the list.
        lines = screen_lines(app)
        row = next(at for at, line in enumerate(lines) if f"• {page.title}" in line)
        column = lines[row].index(page.title)
        await pilot.click(offset=(column, row))
        await idle(pilot)
        assert help.address == page.address


async def test_a_click_on_the_tree_opens_a_topic_and_then_reads_its_page(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        lines = screen_lines(app)
        title = DOCUMENT.topics[2].title
        row = next(at for at, line in enumerate(lines) if f"  {title}" in line)
        await pilot.click(offset=(lines[row].index(title), row))
        await idle(pilot)
        # A click on a closed topic shows its page and opens it in the tree; a click
        # on what is already open moves on to the page.
        assert help.address == DOCUMENT.topics[2].intro.address
        assert help.expanded and isinstance(app.focused, TopicList)
        await pilot.click(offset=(lines[row].index(title), row))
        await idle(pilot)
        assert app.focused is help.page


async def test_the_wheel_scrolls_the_page(app):
    async with app.run_test(size=(100, 24)) as pilot:
        help = await opened(pilot, app, "help:functions/vectors-and-matrices")
        at = help.page.region.offset + (2, 2)
        for _ in range(3):
            app.screen._forward_event(
                MouseScrollDown(
                    None, x=at.x, y=at.y, delta_x=0, delta_y=0, button=0,
                    shift=False, meta=False, ctrl=False, screen_x=at.x, screen_y=at.y,
                )
            )
        await idle(pilot)
        assert help.page.scroll_y > 0


async def test_search_lists_results_in_place_of_the_tree(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        await pilot.press("slash")
        assert isinstance(app.focused, Input)
        await pilot.press(*"dimens")
        await idle(pilot)
        assert [entry.label for entry in help.results] == ["DIMENSION"]
        assert "DIMENSION  DIM" in shown(app)
        # Esc clears the search and brings the tree back.
        await pilot.press("escape")
        await idle(pilot)
        assert help.results is None and help.search.value == ""
        assert isinstance(app.focused, TopicList)
        assert DOCUMENT.topics[1].title in shown(app)


async def test_enter_on_a_result_goes_to_its_row_and_highlights_it(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        await pilot.press("slash", *"dimension", "enter")
        await idle(pilot)
        assert help.address == DOCUMENT.names["dim"].address
        assert help.marked[0].startswith("DIM(v)")
        assert app.focused is help.page
        assert help.results is None
        # The row is scrolled into view.
        assert "DIM(v)" in shown(app)


async def test_up_and_down_in_the_search_box_choose_a_result(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        await pilot.press("slash", *"calculus")
        await idle(pilot)
        assert len(help.results) > 1
        await pilot.press("down", "enter")
        await idle(pilot)
        assert help.address == DOCUMENT.search("calculus")[1].place.address


async def test_a_command_result_lands_on_its_row(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        await pilot.press("slash", *"calculus integrate", "enter")
        await idle(pilot)
        assert help.address == DOCUMENT.commands["Calculus Integrate"].address
        assert help.marked[0].startswith("Calculus Integrate")


async def test_f1_opens_line_editing_and_gives_the_line_back(app):
    async with app.run_test() as pilot:
        await pilot.press("a", *"x^2+1")
        assert prompt(app) == ("AUTHOR expression:", "x^2+1")
        await pilot.press("f1")
        await idle(pilot)
        help = dialog(app)
        assert help.address == helps.LINE_EDITING
        # Reading elsewhere and closing returns to the line, with the typed text intact.
        await pilot.press("down", "escape")
        await idle(pilot)
        assert app.helping is None
        assert prompt(app) == ("AUTHOR expression:", "x^2+1")
        assert message(app) == "Enter expression (Alt-Enter: new line, F1: help)"
        await pilot.press("enter")
        assert [entry.text for entry in app.session.entries] == ["x^2 + 1"]


async def test_f1_at_the_command_menu_is_still_the_next_window(app):
    async with app.run_test() as pilot:
        await pilot.press("w", "s", "h", "enter")
        assert app.windows.number == 1
        await pilot.press("f1")
        assert app.windows.number == 2
        assert app.helping is None


async def test_the_app_s_keys_do_nothing_under_help(app):
    async with app.run_test() as pilot:
        await opened(pilot, app)
        # Letters, Tab and F1 would run commands or move the menu.
        await pilot.press("q", "a", "f1")
        await idle(pilot)
        assert isinstance(app.screen, HelpScreen)
        assert app.mode == MODE_HELP


async def test_the_opening_notice_is_still_there_when_help_closes(app):
    async with app.run_test() as pilot:
        await opened(pilot, app)
        await pilot.press("escape")
        await idle(pilot)
        assert "H for help" in "\n".join(work_area(app))


async def test_a_demo_started_from_outside_closes_help(app):
    async with app.run_test() as pilot:
        await opened(pilot, app)
        app.demonstrate("NO_SUCH_DEMO", text="")
        await idle(pilot)
        assert not isinstance(app.screen, HelpScreen)
        assert app.helping is None


async def test_a_narrow_screen_shows_the_tree_and_the_page_in_turn(app):
    async with app.run_test(size=(64, 24)) as pilot:
        help = await opened(pilot, app)
        reader = help.query_one("#help-reader")
        side = help.query_one("#help-side")
        assert side.display and not reader.display
        await pilot.press("enter", "enter")
        await idle(pilot)
        assert reader.display and not side.display
        assert app.focused is help.page
        # Esc on the page returns to the tree, and on the tree closes help.
        await pilot.press("escape")
        await idle(pilot)
        assert side.display and not reader.display
        await pilot.press("escape")
        await idle(pilot)
        assert not isinstance(app.screen, HelpScreen)


async def test_a_wide_screen_shows_the_tree_and_the_page_together(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        assert help.query_one("#help-side").display
        assert help.query_one("#help-reader").display
        await pilot.press("enter")
        await pilot.press("escape")
        await idle(pilot)
        assert not isinstance(app.screen, HelpScreen)


def _colors(app) -> set[str]:
    """Every color drawn on the screen."""
    found: set[str] = set()
    for strip in app.screen._compositor.render_strips():
        for segment in strip:
            style = segment.style
            if style is None:
                continue
            for color in (style.color, style.bgcolor):
                if color is not None and color.triplet is not None:
                    found.add(color.triplet.hex)
    return found


@pytest.mark.parametrize(
    "scheme",
    [
        {},
        {
            "WorkColor": "Black",
            "WorkBackgroundColor": "Gray",
            "OptionColor": "Blue",
            "FrameColor": "Magenta",
            "MenuBackgroundColor": "Cyan",
        },
    ],
)
async def test_every_color_in_help_is_one_of_the_slots(app, scheme):
    # Narrow, so the dialog covers the screen and the dimmed worksheet doesn't show.
    async with app.run_test(size=(70, 30)) as pilot:
        app.settings.apply(scheme)
        await idle(pilot)
        await opened(pilot, app, "help:entering/operators-and-constants")
        await pilot.press("enter", "slash", "d")
        await idle(pilot)
        drawn = _colors(app)
        assert drawn <= set(EGA), drawn - set(EGA)
        if scheme:
            assert EGA[1] in drawn  # the option color, blue


async def test_a_table_fits_the_pane_and_wraps_its_last_column(app):
    async with app.run_test(size=(80, 30)) as pilot:
        help = await opened(pilot, app, "help:entering/operators-and-constants")
        tables = list(help.markdown.query(HelpTable).results(HelpTable))
        assert tables
        for table in tables:
            lines = str(table.render()).splitlines()
            assert max(len(line) for line in lines) <= table.size.width
            assert table.size.height == len(lines)
        assert help.query_one(HelpPage).max_scroll_x == 0


async def test_holding_down_in_the_tree_walks_it_without_waiting_for_pages(app):
    """Keys arrive faster than pages load: no page is shown for a table gone, and the
    highlight ends where the keys put it, on the page shown."""
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        basics = DOCUMENT.topics[0].pages
        await pilot.press("right")
        # Without a pause between them, as autorepeat sends them.
        await pilot.press(*["down"] * 3)
        await idle(pilot)
        assert help.address == _highlighted(help) == basics[3].address
        await pilot.press(*["down"] * 30)
        await idle(pilot)
        assert help.address == _highlighted(help)
        # Each topic opens as the highlight reaches it, however fast the keys come:
        # thirty lines on from the third page of Basics, as the pages are read.
        order = [page.address for page in DOCUMENT.pages]
        assert help.address == order[3 + 30]


def _highlighted(help: HelpScreen) -> str | None:
    tree = help.tree
    assert tree.highlighted is not None
    return tree.get_option_at_index(tree.highlighted).id


async def test_a_page_is_whole_when_it_first_appears(app):
    """A table has its lines and its height in the layout that first shows it."""
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app)
        help.go(helps.Place("help:functions/elementary"))
        while help.fetching:
            await pilot.pause()
        # One turn for the layout, and nothing drawn after it but the scrollbar.
        await pilot.pause()
        heights = [
            table.size.height
            for table in help.markdown.query(HelpTable).results(HelpTable)
        ]
        assert heights and all(height > 2 for height in heights)
        first = _without_bar(shown(app))
        await idle(pilot)
        assert _without_bar(shown(app)) == first


def _without_bar(text: str) -> str:
    """`text` without the blocks a scrollbar's thumb is drawn with."""
    return "".join(" " if "\u2581" <= each <= "\u2588" else each for each in text)


# -- items -----------------------------------------------------------------


LOOPS = "help:programming/conditions-and-loops"
CALCULUS = "help:functions/calculus"


def keys(app) -> str:
    """The footer's line of key hints."""
    return str(app.screen.query_one("#help-keys").render())


async def reading(pilot, app, place):
    """Open help at `place` from the menu, with the page given the keys."""
    help = await opened(pilot, app, place)
    await pilot.press("enter")
    await idle(pilot)
    return help


async def from_the_line(pilot, app, place):
    """Open help with F1 from the Author line, then go to `place` and read it."""
    await pilot.press("f1")
    await idle(pilot)
    help = dialog(app)
    help.go(helps.Place(place))
    await idle(pilot)
    await pilot.press("tab")
    await idle(pilot)
    assert app.focused is help.page
    return help


@pytest.mark.parametrize(
    "address",
    [LOOPS, CALCULUS, helps.START, "help:functions/algebra-and-solving"],
)
async def test_the_dialog_finds_the_items_of_the_page(app, address):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app, address)
        page = DOCUMENT.page(address)
        found = [helps.Item(spot.kind, spot.text) for spot in help.spots]
        assert found == helps.items(page.source)


async def test_tab_steps_through_the_items_and_shift_tab_back_to_the_tree(app):
    async with app.run_test(size=(100, 24)) as pilot:
        help = await reading(pilot, app, LOOPS)
        items = helps.items(DOCUMENT.page(LOOPS).source)
        assert help.picked is None
        for item in items:
            await pilot.press("tab")
            assert (help.picked.kind, help.picked.text) == (item.kind, item.text)
        # The last item is scrolled into view.
        assert help.page.scroll_y > 0
        await idle(pilot)
        assert "COLLATZ(n, k := 0) :=" in shown(app)
        await pilot.press("shift+tab")
        assert help.picked.text == items[-2].text
        for _ in items[:-2]:
            await pilot.press("shift+tab")
        assert help.picked.text == items[0].text
        await pilot.press("shift+tab")
        assert isinstance(app.focused, TopicList)
        assert help.picked is None
        # Tab past the last item leaves for the tree too.
        await pilot.press("tab", *["tab"] * len(items), "tab")
        assert isinstance(app.focused, TopicList)


async def test_left_returns_to_the_tree_and_lets_go_of_the_item(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await reading(pilot, app, LOOPS)
        await pilot.press("tab")
        assert help.picked is not None
        await pilot.press("left")
        assert isinstance(app.focused, TopicList)
        assert help.picked is None


async def test_tab_on_a_page_without_items_returns_to_the_tree(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await reading(pilot, app, "help:functions/finance")
        assert help.spots == []
        assert "Tab pane" in keys(app)
        await pilot.press("tab")
        assert isinstance(app.focused, TopicList)


async def test_scrolling_the_item_out_of_view_lets_go_of_it(app):
    async with app.run_test(size=(100, 24)) as pilot:
        help = await reading(pilot, app, LOOPS)
        await pilot.press("tab", "tab")
        assert help.picked.kind == "program"
        await pilot.press(*["down"] * 12)
        await idle(pilot)
        assert help.picked is None
        # Tab then picks the first item in view.
        await pilot.press("tab")
        first = help._place(help.picked)[0]
        assert first >= help.page.scroll_y


async def test_enter_copies_an_example_for_ctrl_v_on_the_author_line(app):
    async with app.run_test(size=(100, 32)) as pilot:
        await reading(pilot, app, CALCULUS)
        assert "Enter" not in keys(app)
        await pilot.press("tab", "tab")
        assert "Enter copy" in keys(app)
        await pilot.press("enter")
        assert app.clipboard == "INT(#e^(-x^2), x, -inf, inf)"
        assert keys(app).strip() == COPIED
        await pilot.press("escape")
        await idle(pilot)
        await pilot.press("a", "ctrl+v")
        assert prompt(app)[1] == "INT(#e^(-x^2), x, -inf, inf)"


async def test_enter_follows_a_link(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await reading(pilot, app, LOOPS)
        await pilot.press("tab")
        assert help.picked.text == "name:IF"
        assert "Enter follow" in keys(app)
        await pilot.press("enter")
        await idle(pilot)
        assert help.address == DOCUMENT.names["if"].address
        assert help.picked is None
        assert app.clipboard == ""


async def test_a_link_in_a_table_is_picked_and_followed(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await reading(pilot, app, "help:functions/algebra-and-solving")
        await pilot.press(*["tab"] * 5)
        assert help.picked.text == "help:functions/vectors-and-matrices"
        assert isinstance(help.picked.block, HelpTable)
        await pilot.press("enter")
        await idle(pilot)
        assert help.address == "help:functions/vectors-and-matrices"


async def test_the_pointer_marks_an_example_and_a_click_copies_it(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await opened(pilot, app, CALCULUS)
        lines = screen_lines(app)
        row = next(at for at, line in enumerate(lines) if "LIM(SIN(x)/x" in line)
        column = lines[row].index("LIM(SIN(x)/x")
        assert lines[row].rstrip(" \u2502").endswith(MARKER)
        await pilot.hover(offset=(column + 3, row))
        await idle(pilot)
        assert screen_lines(app)[row].rstrip(" \u2502").endswith(MARKER)
        await pilot.click(offset=(column + 3, row))
        await idle(pilot)
        assert app.clipboard == "LIM(SIN(x)/x, x, 0)"
        assert app.focused is help.page
        assert help.picked.text == "LIM(SIN(x)/x, x, 0)"
        assert keys(app).strip() == COPIED


async def test_f3_inserts_an_example_into_the_line_at_the_cursor(app):
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press("a", *"()+1", "home", "right")
        help = await from_the_line(pilot, app, CALCULUS)
        assert "F3" not in keys(app)
        await pilot.press("tab")
        assert "F3 insert" in keys(app)
        await pilot.press("f3")
        await idle(pilot)
        assert not isinstance(app.screen, HelpScreen)
        assert prompt(app) == ("AUTHOR expression:", "(DIF(SIN(x)^2, x))+1")
        assert app.help_place.address == help.address == CALCULUS
        await pilot.press("enter")
        assert [entry.text for entry in app.session.entries] == [
            "DIF(SIN(x)^2, x) + 1"
        ]


async def test_f3_inserts_a_program_with_its_rows(app):
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press("a")
        await from_the_line(pilot, app, LOOPS)
        texts = [item.text for item in helps.items(DOCUMENT.page(LOOPS).source)]
        program = next(text for text in texts if text.startswith("ROOTS(a"))
        assert "\n" in program
        await pilot.press(*["tab"] * (texts.index(program) + 1))
        await pilot.press("f3")
        await idle(pilot)
        assert prompt(app)[1] == program


async def test_f3_does_nothing_when_help_was_opened_from_the_menu(app):
    async with app.run_test(size=(100, 32)) as pilot:
        help = await reading(pilot, app, CALCULUS)
        await pilot.press("tab")
        assert help.picked is not None
        assert "F3" not in keys(app)
        await pilot.press("f3")
        await idle(pilot)
        assert app.screen is help
        assert help.picked is not None


async def test_f3_does_nothing_on_a_link(app):
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press("a")
        help = await from_the_line(pilot, app, LOOPS)
        await pilot.press("tab")
        assert help.picked.kind == "link"
        assert "F3" not in keys(app)
        await pilot.press("f3")
        await idle(pilot)
        assert app.screen is help


async def test_items_work_in_the_narrow_layout(app):
    async with app.run_test(size=(64, 24)) as pilot:
        help = await reading(pilot, app, LOOPS)
        await pilot.press("tab", "tab")
        assert help.picked.kind == "program"
        assert "Enter copy" in keys(app)
        await pilot.press("enter")
        assert app.clipboard.startswith("FIB(n)")
        # Shift-Tab from the first item shows the tree again.
        await pilot.press("shift+tab", "shift+tab")
        await idle(pilot)
        assert help.query_one("#help-side").display
        assert not help.query_one("#help-reader").display


async def test_every_color_of_a_picked_item_is_one_of_the_slots(app):
    async with app.run_test(size=(70, 30)) as pilot:
        help = await reading(pilot, app, "help:functions/algebra-and-solving")
        # An example under the pointer, and a link picked from the keyboard.
        lines = screen_lines(app)
        row = next(at for at, line in enumerate(lines) if "EXPAND((x" in line)
        await pilot.hover(offset=(lines[row].index("EXPAND((x") + 1, row))
        await pilot.press(*["tab"] * 5)
        await idle(pilot)
        assert help.picked.kind == "link"
        drawn = _colors(app)
        assert drawn <= set(EGA), drawn - set(EGA)
