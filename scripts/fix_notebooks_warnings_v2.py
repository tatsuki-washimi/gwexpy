#!/usr/bin/env python3
"""
Notebook warnings fix script v2.

Detects and fixes corrupted or missing indentation in with warnings.catch_warnings() blocks.
"""
import nbformat
from pathlib import Path
import re
import argparse
import shutil

def needs_fix(src):
    """Detect if a cell source has a broken warnings block."""
    if "warnings.catch_warnings" not in src:
        return False
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if "warnings.catch_warnings" in ln:
            # find next non-empty line
            for j in range(i+1, len(lines)):
                if lines[j].strip()=="" or lines[j].strip().startswith("#"):
                    continue
                # if next non-empty line is not indented -> broken
                if not re.match(r'^\s+', lines[j]):
                    return True
                break
    # also check multiple with blocks in same cell (conservative)
    if src.count("warnings.catch_warnings") > 1:
        return True
    return False

def wrap_cell(src):
    """Provide a single canonical warnings wrapper for a cell's code."""
    # Remove existing 'with warnings.catch_warnings()' occurrences and create one canonical wrapper.
    # Also ensure import warnings exists.
    lines = src.splitlines()
    filtered = []
    for ln in lines:
        if "warnings.catch_warnings" in ln:
            # skip the with line
            continue
        # avoid removing warnings.simplefilter lines — keep them inside wrapper
        filtered.append(ln)
    body = "\n".join(filtered).rstrip()
    # Ensure a warnings import exists at top of the cell
    if not re.search(r'^\s*import\s+warnings', body, flags=re.M):
        pre = "import warnings\n"
    else:
        pre = ""
    indented = "\n".join(("    " + l) if l.strip()!="" else "" for l in body.splitlines())
    wrapped = (pre + "with warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n\n"
               + indented + ("\n" if not indented.endswith("\n") else ""))
    return wrapped

def process_nb(path, dry_run=True):
    """Read a notebook and fix necessary cells."""
    nb = nbformat.read(path, as_version=4)
    changed = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        if needs_fix(cell.source):
            print(f"[{'DRY RUN' if dry_run else 'FIXING'}] Cell {i} in {path}")
            if not dry_run:
                # backup if not already backed up
                orig_path = str(path) + ".orig"
                if not Path(orig_path).exists():
                    shutil.copy(path, orig_path)
                cell.source = wrap_cell(cell.source)
                changed = True
    if changed and not dry_run:
        nbformat.write(nb, path)
    return changed

def discover(root="docs/web", dry_run=True):
    """Search for notebooks in a directory and process them."""
    changed_files = []
    for p in Path(root).rglob("*.ipynb"):
        if ".ipynb_checkpoints" in str(p):
            continue
        try:
            if process_nb(p, dry_run=dry_run):
                changed_files.append(str(p))
        except Exception as e:
            print(f"Error processing {p}: {e}")
    return changed_files

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/web")
    ap.add_argument("--dry-run", action="store_true", help="Do not apply changes")
    ap.add_argument("--fix", action="store_true", help="Apply changes (opposite of --dry-run)")
    args = ap.parse_args()

    # User requested dry-run review first unless --fix is explicitly passed.
    is_dry_run = not args.fix or args.dry_run

    changed = discover(args.root, dry_run=is_dry_run)
    if is_dry_run:
        print("\n[DRY RUN] Notebooks that would be changed:", changed)
    else:
        print("\n[DONE] Changed notebooks:", changed)
