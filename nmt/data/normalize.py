from __future__ import annotations

# Shared text normalization used identically at train and inference time.
#
# Provides `normalize_text`: NFKC normalization, apostrophe/quote unification
# (`' ' ʼ` -> `'`, `« » " "` -> `"`), whitespace collapse and strip, casing preserved.
# Used by prepare.py, tokenize.py, translate.py and evaluate.py -- never re-implemented
# elsewhere (see PLAN.md decisions). Spec §3.
#
# Also provides `near_dup_key`, the lowercase-alphanumeric-only key used for
# near-duplicate / leakage matching throughout the data pipeline (PLAN.md decisions).
import re
import unicodedata

# French curly/alternate apostrophes and CJK-style/French guillemet quotes, unified to
# plain ASCII ' and " respectively so downstream tokenization/matching sees one form.
_APOSTROPHES = "’‘ʼ"  # ' ' ʼ
_QUOTES = "«»“”"  # « » " "
_APOSTROPHE_TABLE = str.maketrans({c: "'" for c in _APOSTROPHES})
_QUOTE_TABLE = str.maketrans({c: '"' for c in _QUOTES})
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Normalize `s` identically at train and inference time (spec §3).

    Steps, in order: NFKC normalization; unify apostrophe variants to `'` and quote
    variants to `"`; collapse any run of whitespace to a single space; strip leading/
    trailing whitespace. Casing is preserved (never lowercased here).

    Note on guillemet spacing: French text commonly uses a non-breaking space (U+00A0)
    or narrow no-break space (U+202F) between a guillemet and the quoted text, e.g.
    "« Bonjour »". NFKC maps both of those to a plain space (U+0020)
    *before* whitespace collapsing runs, so that space ends up as an ordinary internal
    space character -- it is not stripped, only whitespace at the very start/end of the
    whole string is removed, and internal runs are collapsed (never deleted). This is
    intentional: the space is kept, e.g. '"« Bonjour »"' normalizes to
    '" Bonjour "', not '"Bonjour"'.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_APOSTROPHE_TABLE).translate(_QUOTE_TABLE)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def near_dup_key(s: str) -> str:
    """Return the near-duplicate key for `s`: lowercase, alphanumeric characters only.

    Used for near-duplicate leakage matching (PLAN.md decisions). An empty key (e.g.
    from a string with no alphanumeric characters) never matches anything -- callers
    must treat an empty key as "no key", not as a wildcard.
    """
    return "".join(c for c in s.lower() if c.isalnum())
