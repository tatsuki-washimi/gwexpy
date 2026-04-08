#!/usr/bin/env python3
# scripts/check_docs_sync.py
import re
from pathlib import Path
import sys

# Define roots for Japanese (JA) and English (EN) documentation
ROOT_JA = Path('docs/web/ja')
ROOT_EN = Path('docs/web/en')

def count_headings(path):
    """Count H1 (#) and H2 (##) headings in Markdown or rST files."""
    txt = path.read_text(encoding='utf-8')
    h1 = 0
    h2 = 0
    
    if path.suffix == '.md':
        # Remove code blocks before counting headings
        clean_txt = re.sub(r'```.*?```', '', txt, flags=re.DOTALL)
        # Markdown headings starting with #
        h1 = len(re.findall(r'^\s*#\s+.+$', clean_txt, flags=re.MULTILINE))
        h2 = len(re.findall(r'^\s*##\s+.+$', clean_txt, flags=re.MULTILINE))
    elif path.suffix == '.rst':
        # rST headings (simple counting of underline patterns)
        # H1: =================, H2: -----------------
        h1 = len(re.findall(r'^\s*={3,}\s*$', txt, flags=re.MULTILINE))
        h2 = len(re.findall(r'^\s*-{3,}\s*$', txt, flags=re.MULTILINE))
        
    return h1, h2

def find_file_pairs():
    """Find files that exist in both JA and EN directories."""
    # Find all relevant documentation files in JA
    ja_files = set()
    for suffix in ['*.md', '*.rst']:
        ja_files |= {p.relative_to(ROOT_JA) for p in ROOT_JA.rglob(suffix)}
        
    # Find corresponding files in EN
    en_files = set()
    for suffix in ['*.md', '*.rst']:
        en_files |= {p.relative_to(ROOT_EN) for p in ROOT_EN.rglob(suffix)}
        
    # We check files that exist in BOTH (synchronization check)
    common = ja_files & en_files
    
    # Also find files that only exist in one language (optional warning)
    only_ja = ja_files - en_files
    only_en = en_files - ja_files
    
    return sorted(common), sorted(only_ja), sorted(only_en)

def main():
    """Main execution of the sync check."""
    common, only_ja, only_en = find_file_pairs()
    
    mismatches = []
    
    print(f"Starting ja/en synchronization check for {len(common)} paired files...")
    
    for rel in common:
        # Skip API reference, standard reference, legacy case studies, and tutorials
        rel_str = str(rel)
        if 'reference/' in rel_str or 'user_guide/tutorials/' in rel_str:
            continue
            
        ja_path = ROOT_JA / rel
        en_path = ROOT_EN / rel
        
        ja_h1, ja_h2 = count_headings(ja_path)
        en_h1, en_h2 = count_headings(en_path)
        
        if ja_h1 != en_h1 or ja_h2 != en_h2:
            mismatches.append({
                'file': str(rel),
                'ja': (ja_h1, ja_h2),
                'en': (en_h1, en_h2)
            })

    if only_ja:
        print(f"Warning: Files only in JA ({len(only_ja)}):")
        for f in only_ja:
            print(f"  - {f}")
            
    if only_en:
        print(f"Warning: Files only in EN ({len(only_en)}):")
        for f in only_en:
            print(f"  - {f}")

    if mismatches:
        print("\nError: Found heading mismatches between ja and en versions:")
        print("-" * 60)
        for m in mismatches:
            print(f"{m['file']}:")
            print(f"  JA: H1={m['ja'][0]}, H2={m['ja'][1]}")
            print(f"  EN: H1={m['en'][0]}, H2={m['en'][1]}")
        print("-" * 60)
        print(f"Total files with mismatches: {len(mismatches)}")
        sys.exit(1)
    else:
        print("\nSuccess: ja/en heading counts are consistent (Synchronized).")
        return 0

if __name__ == '__main__':
    main()
