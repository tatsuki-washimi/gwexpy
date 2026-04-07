#!/usr/bin/env python3
"""Shared script to check for Non-ASCII (CJK) characters in the repository.

Used by:
1. .github/workflows/docs-lint.yml
2. .agent/skills/verify_hardening/SKILL.md
"""

import os
import re
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check for Non-ASCII (CJK) characters in files.")
    parser.add_argument("--root", default="gwexpy", help="Directory to scan (default: gwexpy)")
    parser.add_argument("--include-docs", action="store_true", help="Also scan top-level 'docs/' directory")
    args = parser.parse_args()

    # Scan library and optionally documentation
    scan_dirs = [args.root]
    if args.include_docs:
        scan_dirs.append("docs")

    # Exclude Japanese translations and build artifacts if scanning 'docs/'
    exclude_dirs = [
        os.path.join('docs', 'web', 'ja'),
        os.path.join('docs', '_build')
    ]

    # Match CJK characters (Hiragana, Katakana, Kanji)
    # Range: \u3040-\u30ff (Hiragana/Katakana), \u4e00-\u9fff (Common CJK Unified Ideographs)
    pattern = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff]')
    bad = []
    
    for root_dir in scan_dirs:
        if not os.path.exists(root_dir):
            if root_dir == args.root:
                print(f"Warning: Root directory '{root_dir}' not found.")
            continue

        for dp, _, fns in os.walk(root_dir):
            # Skip excluded directories
            if any(dp.startswith(ex) for ex in exclude_dirs):
                continue

            for fn in fns:
                # Check .py and .md files
                if not fn.endswith(('.py', '.md')):
                    continue

                path = os.path.join(dp, fn)
                try:
                    # Ignore encoding errors to focus on successful UTF-8 matches
                    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                        for i, l in enumerate(fh, 1):
                            if pattern.search(l):
                                bad.append(f"{path}:{i}: {l.strip()}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    
    if bad:
        print(f"Found {len(bad)} Non-ASCII (CJK) characters in repository:")
        # Print first 50 findings to avoid overwhelming output
        for item in bad[:50]:
            print(f"  {item}")
        
        if len(bad) > 50:
            print(f"  ... and {len(bad) - 50} more findings.")
        
        sys.exit(1)

    print(f"Success: No CJK characters found in {', '.join(scan_dirs)} (excluding JA docs).")
    sys.exit(0)

if __name__ == "__main__":
    main()
