from __future__ import annotations

# Normalization unit tests (spec §3, §12).
from nmt.data.normalize import near_dup_key, normalize_text


def test_apostrophe_variants_unified_to_ascii() -> None:
    """’ ‘ ʼ all unify to a plain ASCII apostrophe."""
    assert normalize_text("l’école") == "l'école"
    assert normalize_text("‘quoted‘") == "'quoted'"
    assert normalize_text("Moucheboʼuf") == "Mouchebo'uf"


def test_quote_variants_unified_to_ascii_double_quote() -> None:
    """« » “ ” all unify to a plain ASCII double quote."""
    assert normalize_text("“Bonjour”") == '"Bonjour"'
    assert normalize_text("«Bonjour»") == '"Bonjour"'


def test_guillemet_nbsp_spacing_is_kept_as_a_plain_space() -> None:
    """NFKC maps the NBSP/narrow-NBSP that French guillemet spacing commonly uses to
    a plain space *before* whitespace collapsing runs, so that space survives as an
    ordinary internal space -- only the whole string's leading/trailing whitespace is
    stripped, internal runs are collapsed, never deleted (documented in normalize.py)."""
    nbsp = " "
    narrow_nbsp = " "
    assert normalize_text(f"«{nbsp}Bonjour{nbsp}»") == '" Bonjour "'
    assert normalize_text(f"«{narrow_nbsp}Bonjour{narrow_nbsp}»") == '" Bonjour "'


def test_whitespace_runs_collapse_to_single_space_and_strip() -> None:
    assert normalize_text("  a   b\t\tc\n\nd  ") == "a b c d"
    assert normalize_text("\n\nsurrounded\n\n") == "surrounded"


def test_casing_is_preserved() -> None:
    assert normalize_text("Le Grand MEAULNES") == "Le Grand MEAULNES"


def test_nfkc_ligature_fi_becomes_two_chars() -> None:
    """U+FB01 (ﬁ ligature) decomposes to plain 'fi' under NFKC."""
    assert normalize_text("ﬁn") == "fin"


def test_normalize_text_is_idempotent() -> None:
    samples = [
        "« Bonjour », l’école  a   deux \t\tportes.",
        "  ﬁn de l’histoire  ",
        "",
        "already normal",
    ]
    for s in samples:
        once = normalize_text(s)
        twice = normalize_text(once)
        assert once == twice


def test_normalize_text_empty_string() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \t\n  ") == ""


def test_near_dup_key_lowercases_and_strips_non_alnum() -> None:
    assert near_dup_key("Bonjour, le monde !") == "bonjourlemonde"
    assert near_dup_key("BONJOUR LE MONDE") == "bonjourlemonde"
    assert near_dup_key("Bonjour,  le   monde!") == "bonjourlemonde"


def test_near_dup_key_empty_for_punctuation_only_string() -> None:
    assert near_dup_key("!!! ... ???") == ""
    assert near_dup_key("") == ""


def test_near_dup_key_distinguishes_different_alnum_content() -> None:
    assert near_dup_key("Bonjour") != near_dup_key("Bonsoir")
