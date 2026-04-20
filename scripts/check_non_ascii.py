#!/usr/bin/env python3
# scripts/check_non_ascii.py
import argparse
import os
import re
import sys

# Pattern for CJK characters (fullwidth space + hiragana/katakana/kanji)
# \u3000: ideographic space
# \u3040-\u30ff: hiragana/katakana
# \u4e00-\u9fff: common CJK unified ideographs
PAT = re.compile(r'[\u3000\u3040-\u30ff\u4e00-\u9fff]')

def check_file(path, exclude_patterns=None):
    """Check a single file for CJK characters."""
    if exclude_patterns and any(p in path for p in exclude_patterns):
        return True

    bad_lines = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh, 1):
                if PAT.search(line):
                    bad_lines.append(f"{path}:{i}: {line.strip()}")
    except Exception as e:
        print(f"Could not read {path}: {e}", file=sys.stderr)
        return False

    if bad_lines:
        for item in bad_lines:
            print(item)
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Check for Non-ASCII (CJK) characters.")
    parser.add_argument("--root", default="gwexpy", help="Directory to scan if no files provided.")
    parser.add_argument("--include-docs", action="store_true", help="Also scan 'docs/' directory.")
    parser.add_argument("files", nargs="*", help="Specific files to check.")
    args = parser.parse_args()

    exclude_dirs = [
        os.path.join('docs', 'web', 'ja'),
        os.path.join('docs', '_build')
    ]

    success = True
    if args.files:
        # Check specific files (usually from pre-commit)
        for f in args.files:
            if not f.endswith(('.py', '.md')):
                continue
            if not check_file(f, exclude_patterns=exclude_dirs):
                success = False
    else:
        # Scan directories
        scan_dirs = [args.root]
        if args.include_docs:
            scan_dirs.append("docs")

        for root_dir in scan_dirs:
            if not os.path.exists(root_dir):
                continue
            for dp, _, fns in os.walk(root_dir):
                if any(dp.startswith(ex) for ex in exclude_dirs):
                    continue
                for fn in fns:
                    if not fn.endswith(('.py', '.md')):
                        continue
                    path = os.path.join(dp, fn)
                    if not check_file(path):
                        success = False

    if not success:
        sys.exit(1)
    else:
        print("Success: No CJK characters found (excluding JA docs).")
        sys.exit(0)

if __name__ == "__main__":
    main()
