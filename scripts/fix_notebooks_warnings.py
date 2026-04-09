#!/usr/bin/env python3
"""Notebook warnings fix script.

Cleanup corrupted warnings.catch_warnings() blocks in notebooks.
"""

import argparse
from pathlib import Path

import nbformat


def get_indent(line):
    """Return indentation level of a line."""
    return len(line) - len(line.lstrip())

def cell_needs_fix(src):
    """Check if a cell needs fixing based on indentation after warnings block."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "with warnings.catch_warnings()" in line:
            current_indent = get_indent(line)
            # find next non-empty
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    continue
                next_indent = get_indent(lines[j])
                # if next non-empty line is not more indented, treat as broken
                if next_indent <= current_indent:
                    return True
                break
    return False

def fix_cell(src):
    """Wrap cell content in a single warnings.catch_warnings() block."""
    lines = src.splitlines()
    # Remove any bare occurrences of the with line to avoid duplication
    filtered = []
    for _idx, line in enumerate(lines):
        if "with warnings.catch_warnings()" in line:
            continue
        # Also remove redundant filter lines if they were added multiply
        if "warnings.filterwarnings" in line and "tight_layout" in line:
            continue
        filtered.append(line)

    body = "\n".join(filtered).strip()
    if not body:
        return src

    indented = "\n".join("    " + ln if ln.strip() != "" else ""
                         for ln in body.splitlines())
    wrapped = ("import warnings\n"
               "with warnings.catch_warnings():\n"
               "    warnings.simplefilter('ignore')\n\n"
               + indented + "\n")
    return wrapped

def process_notebook(nbpath: Path, dry_run=True):
    """Process a single notebook file."""
    try:
        nb = nbformat.read(nbpath, as_version=4)
    except Exception as e:
        print(f"Error reading {nbpath}: {e}")
        return False

    changed = False
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        src = cell.source
        if "warnings.catch_warnings" in src and cell_needs_fix(src):
            print(f"[{'DRY RUN' if dry_run else 'FIXING'}] Cell in {nbpath}")
            if not dry_run:
                cell.source = fix_cell(src)
            changed = True

    if changed and not dry_run:
        nbformat.write(nb, nbpath)
        print(f"Wrote modified notebook to {nbpath}")

    return changed

def discover_and_process(root="docs/web", dry_run=True):
    """Discover notebooks and process them."""
    nb_paths = list(Path(root).rglob("*.ipynb"))
    changed_any = []
    for p in nb_paths:
        if ".ipynb_checkpoints" in str(p):
            continue
        try:
            if process_notebook(p, dry_run=dry_run):
                changed_any.append(str(p))
        except Exception as e:
            print(f"Error processing {p}: {e}")
    return changed_any

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Files to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not write")
    parser.add_argument("--fix", action="store_true", help="Apply fixes")
    args = parser.parse_args()

    is_dry_run = not args.fix

    if args.files:
        for t in args.files:
            process_notebook(Path(t), dry_run=is_dry_run)
    else:
        changed_notebooks = discover_and_process(dry_run=is_dry_run)
        if is_dry_run:
            print("\n[DRY RUN] Notebooks that would be changed:", changed_notebooks)
        else:
            print("\n[DONE] Changed notebooks:", changed_notebooks)
