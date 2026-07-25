"""Shapes and spans produced by the placeholder parser."""

from rederive.placeholder_parser import parse


def spans(text, node):
    """The node's kind and the source text its span covers."""
    return node.kind, text[node.start : node.end]


def test_implicit_multiplication_excludes_parentheses():
    text = "x (x + 1)"
    tree = parse(text)
    assert spans(text, tree) == ("product", "x (x + 1)")
    assert [spans(text, child) for child in tree.children] == [
        ("atom", "x"),
        ("sum", "x + 1"),
    ]


def test_sums_are_flattened():
    text = "a + b + c"
    tree = parse(text)
    assert tree.kind == "sum"
    assert [spans(text, child) for child in tree.children] == [
        ("atom", "a"),
        ("atom", "b"),
        ("atom", "c"),
    ]


def test_juxtaposition_is_an_n_ary_product():
    text = "x x x"
    tree = parse(text)
    assert tree.kind == "product"
    assert len(tree.children) == 3


def test_precedence_and_associativity():
    text = "1 + 2 * 3 ^ 4 ^ 5"
    tree = parse(text)
    assert tree.kind == "sum"
    product = tree.children[1]
    assert spans(text, product) == ("product", "2 * 3 ^ 4 ^ 5")
    power = product.children[1]
    assert spans(text, power) == ("power", "3 ^ 4 ^ 5")
    assert spans(text, power.children[1]) == ("power", "4 ^ 5")


def test_subtraction_negates_the_following_term():
    text = "a - b"
    tree = parse(text)
    assert tree.kind == "sum"
    assert [spans(text, child) for child in tree.children] == [
        ("atom", "a"),
        ("neg", "- b"),
    ]


def test_unparsable_input_degrades_to_one_atom():
    text = "x + )"
    tree = parse(text)
    assert spans(text, tree) == ("atom", text)
    assert tree.is_atom
