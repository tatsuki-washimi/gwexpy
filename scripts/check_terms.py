#!/usr/bin/env python3
# scripts/check_terms.py
import csv
import re
from pathlib import Path
import sys

# Constants for directory and file paths
# DOCS_DIR: root for Japanese documentation
DOCS_DIR = Path('docs/web/ja')
# TERMS_CSV: CSV file defining canonical terms and their forbidden variants
TERMS_CSV = Path('scripts/terms.csv')


def _mask_text(text):
    """Replace non-newline characters with spaces to preserve line offsets."""
    return re.sub(r"[^\n]", " ", text)


def _preserve_label(full_text, label):
    """Keep the visible label while masking the remainder of the matched markup."""
    return label + _mask_text(full_text[len(label):])


def _looks_like_url(text):
    """Return True when text resembles a URL or domain/path label."""
    return bool(re.fullmatch(r"(https?://\S+|(?:[\w-]+\.)+[\w-]+/\S*)", text.strip()))

def load_terms():
    """Load canonical terms and variants from CSV."""
    terms = {}
    if not TERMS_CSV.exists():
        print(f"Warning: {TERMS_CSV} not found.")
        return {}
    with TERMS_CSV.open(encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows or comment rows
            if not row or row[0].startswith('#'):
                continue
            canonical = row[0].strip()
            variants = []
            if len(row) > 1 and row[1].strip():
                # Split variants by '|' and strip whitespace
                variants = [v.strip() for v in row[1].split('|') if v.strip()]
            terms[canonical] = variants
    return terms

def strip_code(text):
    """Remove code blocks and inline code to avoid false positives."""
    # Remove fenced code blocks (```...```)
    text = re.sub(r'```.*?```', lambda m: _mask_text(m.group(0)), text, flags=re.DOTALL)

    # Preserve visible labels in reStructuredText roles while masking link targets.
    text = re.sub(
        r':[a-zA-Z0-9_+-]+:`(?P<label>[^`<>]+?)\s*<(?P<target>[^`>]+)>`',
        lambda m: _preserve_label(m.group(0), m.group("label")),
        text,
    )

    # Preserve visible labels in generic reStructuredText links unless the label is itself a URL.
    text = re.sub(
        r'`(?P<label>[^`<>]+?)\s*<(?P<target>[^`>]+)>`_?',
        lambda m: _mask_text(m.group(0))
        if _looks_like_url(m.group("label"))
        else _preserve_label(m.group(0), m.group("label")),
        text,
    )

    # Preserve Markdown link labels while masking link targets.
    text = re.sub(
        r'\[(?P<label>[^\]]+)\]\((?P<target>[^)]+)\)',
        lambda m: _preserve_label(m.group(0), m.group("label")),
        text,
    )

    # Remove bare URLs and URL-like labels that should not participate in terminology checks.
    text = re.sub(
        r'https?://\S+|(?:[\w-]+\.)+[\w-]+/\S*',
        lambda m: _mask_text(m.group(0)),
        text,
    )

    # Remove remaining inline code (`...`)
    text = re.sub(r'`[^`]+`', lambda m: _mask_text(m.group(0)), text)

    # Remove hidden toctree entries and MyST-style anchor labels.
    text = re.sub(
        r'^\s+[A-Za-z0-9_./-]+(?:\s*<[^>]+>)?\s*$',
        lambda m: _mask_text(m.group(0)),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^\([A-Za-z0-9_.-]+\)=\s*$',
        lambda m: _mask_text(m.group(0)),
        text,
        flags=re.MULTILINE,
    )

    # Remove rST directives and comments that might contain code-like text
    text = re.sub(r'^\.\..*$', lambda m: _mask_text(m.group(0)), text, flags=re.MULTILINE)
    return text

def main():
    """Scan documentation files for potential term mismatches."""
    terms = load_terms()
    if not terms:
        print("No terms to check.")
        return 0

    # Collect markdown and rST files
    files = list(DOCS_DIR.rglob('*.md')) + list(DOCS_DIR.rglob('*.rst'))
    hits = []
    for p in files:
        # Skip auto-generated API reference and Glossary (which contains variants as definitions)
        if 'reference/api' in str(p) or 'glossary' in str(p):
            continue
        try:
            txt = p.read_text(encoding='utf-8')
            txt_no_code = strip_code(txt)
            for canonical, variants in terms.items():
                for v in variants:
                    # Use case-insensitive word matching for the variant
                    # Note: We use the variant regex directly from the CSV
                    for m in re.finditer(v, txt_no_code, flags=re.IGNORECASE):
                        # Ensure we are not matching the canonical term itself if it overlaps
                        matched_text = m.group(0)
                        if matched_text == canonical:
                            continue
                        # Record the hit details
                        hits.append({
                            'path': p,
                            'canonical': canonical,
                            'variant_rule': v,
                            'matched': matched_text,
                            'line': txt.count('\n', 0, m.start()) + 1
                        })
        except Exception as e:
            print(f"Error reading {p}: {e}")

    if hits:
        print("Found potential term mismatches (Forbidden variants used):")
        print("-" * 60)
        for hit in hits:
            print(f"{hit['path']}:{hit['line']}: matched '{hit['matched']}'")
            print(f"  -> Recommended canonical term: '{hit['canonical']}' (Rule: '{hit['variant_rule']}')")
        print("-" * 60)
        print(f"Total mismatches: {len(hits)}")
        sys.exit(1)
    else:
        print("Success: No suspicious variants found in Japanese documentation.")
        return 0

if __name__ == '__main__':
    main()
