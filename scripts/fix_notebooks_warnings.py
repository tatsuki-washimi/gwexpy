#!/usr/bin/env python3
import nbformat
from nbformat import v4
from pathlib import Path
import textwrap
import sys

def get_indent(line):
    return len(line) - len(line.lstrip())

def cell_needs_fix(src):
    # 判定ルール：
    # - セルに "with warnings.catch_warnings()" を含むが、
    #   次の非空行がインデントを深めていない（壊れている） -> 修正対象
    lines = src.splitlines()
    for i,l in enumerate(lines):
        if "with warnings.catch_warnings()" in l:
            current_indent = get_indent(l)
            # find next non-empty
            for j in range(i+1, len(lines)):
                if lines[j].strip() == "":
                    continue
                next_indent = get_indent(lines[j])
                # if next non-empty line is not more indented than the 'with' line, treat as broken
                if next_indent <= current_indent:
                    return True
                break
    return False

def fix_cell(src):
    # Wrap entire cell in a single correct context manager.
    # If cell already contains "with warnings.catch_warnings()", remove existing occurrences to avoid nested duplicates.
    lines = src.splitlines()
    # Remove any bare occurrences of the with line to avoid duplication
    filtered = []
    for idx,l in enumerate(lines):
        if "with warnings.catch_warnings()" in l:
            continue
        # Also remove redundant filter lines if they were added multiply
        if "warnings.filterwarnings" in l and "tight_layout" in l:
             # we will add a single simplefilter('ignore') or keep one
             continue
        filtered.append(l)
    
    body = "\n".join(filtered).strip()
    if not body:
        return src # should not happen if cell_needs_fix was True
        
    indented = "\n".join("    "+ln if ln.strip() != "" else "" for ln in body.splitlines())
    wrapped = ("import warnings\n"
               "with warnings.catch_warnings():\n"
               "    warnings.simplefilter('ignore')\n\n"
               + indented + "\n")
    return wrapped

def process_notebook(nbpath: Path, dry_run=True):
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Files to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    parser.add_argument("--fix", action="store_true", help="Apply fixes (opposite of dry-run)")
    args = parser.parse_args()

    # Default to dry-run unless --fix is specified
    is_dry_run = not args.fix

    if args.files:
        for t in args.files:
            process_notebook(Path(t), dry_run=is_dry_run)
    else:
        changed = discover_and_process(dry_run=is_dry_run)
        if is_dry_run:
            print("\n[DRY RUN] Notebooks that would be changed:", changed)
        else:
            print("\n[DONE] Changed notebooks:", changed)
