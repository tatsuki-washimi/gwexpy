"""Tests for the small standard-library PO reader used by docs checks."""

from __future__ import annotations

from scripts.po_utils import read_po


def test_read_po_collects_context_multiline_and_plural_strings(tmp_path) -> None:
    catalog = tmp_path / "messages.po"
    catalog.write_text(
        "#: guide.md:12\n"
        'msgctxt "navigation"\n'
        'msgid "A label"\n'
        'msgstr "ラベル"\n'
        "\n"
        'msgid "A long sentence"\n'
        'msgstr "長い "\n'
        '"文章です"\n'
        "\n"
        'msgid "item"\n'
        'msgid_plural "items"\n'
        'msgstr[0] "項目"\n'
        "\n"
        '#~ msgid "obsolete"\n'
        '#~ msgstr "旧文"\n',
        encoding="utf-8",
    )

    entries = read_po(catalog)

    assert entries == [
        {
            "context": "navigation",
            "msgid": "A label",
            "msgid_plural": None,
            "msgstr": "ラベル",
            "plural": [],
            "flags": set(),
            "line": 4,
        },
        {
            "context": None,
            "msgid": "A long sentence",
            "msgid_plural": None,
            "msgstr": "長い 文章です",
            "plural": [],
            "flags": set(),
            "line": 7,
        },
        {
            "context": None,
            "msgid": "item",
            "msgid_plural": "items",
            "msgstr": "",
            "plural": ["項目"],
            "flags": set(),
            "line": 12,
        },
    ]
