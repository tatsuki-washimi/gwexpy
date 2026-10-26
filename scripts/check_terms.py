#!/usr/bin/env python3
# scripts/check_terms.py
import csv
import re
import sys
from pathlib import Path

try:
    from scripts.po_utils import read_po
except (
    ModuleNotFoundError
):  # Support direct execution as `python scripts/check_terms.py`.
    from po_utils import read_po

# Constants for directory and file paths
# DOCS_DIR: root for Japanese gettext catalogs used by the published site
DOCS_DIR = Path("docs_redesign/locales/ja/LC_MESSAGES")
# TERMS_CSV: CSV file defining canonical terms and their forbidden variants
TERMS_CSV = Path("scripts/terms.csv")


def _mask_text(text):
    """Replace non-newline characters with spaces to preserve line offsets."""
    return re.sub(r"[^\n]", " ", text)


def _preserve_label(full_text, label):
    """Keep the visible label while masking the remainder of the matched markup."""
    return label + _mask_text(full_text[len(label) :])


def _looks_like_url(text):
    """Return True when text resembles a URL or domain/path label."""
    return bool(re.fullmatch(r"(https?://\S+|(?:[\w-]+\.)+[\w-]+/\S*)", text.strip()))


def load_terms():
    """Load canonical terms and variants from CSV."""
    terms = {}
    if not TERMS_CSV.exists():
        print(f"Warning: {TERMS_CSV} not found.")
        return {}
    with TERMS_CSV.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows or comment rows
            if not row or row[0].startswith("#"):
                continue
            canonical = row[0].strip()
            variants = []
            if len(row) > 1 and row[1].strip():
                # Split variants by '|' and strip whitespace
                variants = [v.strip() for v in row[1].split("|") if v.strip()]
            terms[canonical] = variants
    return terms


def scan_po_catalog(path, terms):
    """Find forbidden terminology variants in translated Japanese PO strings."""
    hits = []
    for entry in read_po(path):
        if "fuzzy" in entry["flags"]:
            continue
        translated = [entry["msgstr"], *entry["plural"]]
        for text in translated:
            clean_text = strip_code(text)
            for canonical, variants in terms.items():
                for variant in variants:
                    for match in re.finditer(variant, clean_text, flags=re.IGNORECASE):
                        matched_text = match.group(0)
                        if matched_text == canonical:
                            continue
                        hits.append(
                            {
                                "path": path,
                                "canonical": canonical,
                                "variant_rule": variant,
                                "matched": matched_text,
                                "line": entry["line"],
                            }
                        )
    return hits


def strip_code(text):
    """Remove code blocks and inline code to avoid false positives."""
    # Remove fenced code blocks (```...```)
    text = re.sub(r"```.*?```", lambda m: _mask_text(m.group(0)), text, flags=re.DOTALL)

    # Preserve visible labels in reStructuredText roles while masking link targets.
    text = re.sub(
        r":[a-zA-Z0-9_+-]+:`(?P<label>[^`<>]+?)\s*<(?P<target>[^`>]+)>`",
        lambda m: _preserve_label(m.group(0), m.group("label")),
        text,
    )

    # Preserve visible labels in generic reStructuredText links unless the label is itself a URL.
    text = re.sub(
        r"`(?P<label>[^`<>]+?)\s*<(?P<target>[^`>]+)>`_?",
        lambda m: (
            _mask_text(m.group(0))
            if _looks_like_url(m.group("label"))
            else _preserve_label(m.group(0), m.group("label"))
        ),
        text,
    )

    # Preserve Markdown link labels while masking link targets.
    text = re.sub(
        r"\[(?P<label>[^\]]+)\]\((?P<target>[^)]+)\)",
        lambda m: _preserve_label(m.group(0), m.group("label")),
        text,
    )

    # Remove bare URLs and URL-like labels that should not participate in terminology checks.
    text = re.sub(
        r"https?://\S+|(?:[\w-]+\.)+[\w-]+/\S*",
        lambda m: _mask_text(m.group(0)),
        text,
    )

    # Remove remaining inline code (`...`)
    text = re.sub(r"`[^`]+`", lambda m: _mask_text(m.group(0)), text)

    # Remove hidden toctree entries and MyST-style anchor labels.
    text = re.sub(
        r"^\s+[A-Za-z0-9_./-]+(?:\s*<[^>]+>)?\s*$",
        lambda m: _mask_text(m.group(0)),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^\([A-Za-z0-9_.-]+\)=\s*$",
        lambda m: _mask_text(m.group(0)),
        text,
        flags=re.MULTILINE,
    )

    # Remove rST directives and comments that might contain code-like text
    text = re.sub(
        r"^\.\..*$", lambda m: _mask_text(m.group(0)), text, flags=re.MULTILINE
    )
    return text


def main():
    """Scan documentation files for potential term mismatches."""
    terms = load_terms()
    if not terms:
        print("No terms to check.")
        return 0

    # API reference and glossary catalogs intentionally contain names and term
    # definitions that are exempt from editorial terminology rules.
    files = sorted(DOCS_DIR.rglob("*.po"))
    hits = []
    for p in files:
        relative = p.relative_to(DOCS_DIR).as_posix()
        if relative.startswith("reference/api/") or "glossary" in relative:
            continue
        try:
            hits.extend(scan_po_catalog(p, terms))
        except Exception as e:
            print(f"Error reading {p}: {e}")
            return 1

    if hits:
        print("Found potential term mismatches (Forbidden variants used):")
        print("-" * 60)
        for hit in hits:
            print(f"{hit['path']}:{hit['line']}: matched '{hit['matched']}'")
            print(
                f"  -> Recommended canonical term: '{hit['canonical']}' (Rule: '{hit['variant_rule']}')"
            )
        print("-" * 60)
        print(f"Total mismatches: {len(hits)}")
        sys.exit(1)
    else:
        print("Success: No suspicious variants found in Japanese translations.")
        return 0


if __name__ == "__main__":
    main()
