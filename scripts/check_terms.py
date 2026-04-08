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
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code (`...`)
    text = re.sub(r'`[^`]+`', '', text)
    # Remove rST directives and comments that might contain code-like text
    text = re.sub(r'^\.\..*$', '', text, flags=re.MULTILINE)
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
