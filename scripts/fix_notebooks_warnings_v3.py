#!/usr/bin/env python3
"""
Notebook warnings fix script v3 (Aggressive Reset).

Cleans up all corrupted or duplicate warnings blocks and re-wraps cells correctly.
"""
import nbformat
from pathlib import Path
import re
import argparse
import shutil

def clean_and_rewrap(src):
    """Clean the source of warnings garbage and wrap the whole cell in a single with block."""
    lines = src.splitlines()
    cleaned = []
    # Identify lines to remove
    for ln in lines:
        # Remove literal with blocks and filterwarnings
        if "warnings.catch_warnings" in ln:
            continue
        if "warnings.filterwarnings" in ln:
            continue
        if "warnings.simplefilter" in ln:
            continue
        # Also remove redundant warnings imports if they are in the middle
        if re.match(r'^\s*import\s+warnings\s*$', ln):
            continue
        cleaned.append(ln)

    body = "\n".join(cleaned).strip()
    if not body:
        return src

    # Re-wrap
    indented = "\n".join(("    " + l) if l.strip() else "" for l in body.splitlines())
    wrapped = (
        "import warnings\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('ignore')\n\n"
        + indented + "\n"
    )
    return wrapped

def process_nb(path, dry_run=True):
    """Fix all code cells in the notebook that use warnings.catch_warnings."""
    nb = nbformat.read(path, as_version=4)
    changed = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        # if cell has "warnings" and either "catch_warnings" or corrupted state
        if ("warnings.catch_warnings" in cell.source or
            "warnings.filterwarnings" in cell.source):
            print(f"[{'DRY RUN' if dry_run else 'FIXING'}] Cell {i} in {path}")
            if not dry_run:
                # backup once per file
                orig_path = str(path) + ".orig"
                if not Path(orig_path).exists():
                    shutil.copy(path, orig_path)
                cell.source = clean_and_rewrap(cell.source)
                changed = True

    if changed and not dry_run:
        nbformat.write(nb, path)
    return changed

def main():
    """Discover notebooks and apply fixes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/web")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    nb_paths = list(Path(args.root).rglob("*.ipynb"))
    changed_any = []
    for p in nb_paths:
        if ".ipynb_checkpoints" in str(p):
            continue
        try:
            if process_nb(p, dry_run=not args.fix):
                changed_any.append(str(p))
        except Exception as e:
            print(f"Error {p}: {e}")

    if args.fix:
        print("\n[DONE] Modified:", changed_any)
    else:
        print("\n[DRY RUN] Would modify:", changed_any)

if __name__ == "__main__":
    main()
