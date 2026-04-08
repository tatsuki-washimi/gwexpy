#!/usr/bin/env python3
# scripts/check_external_links.py
import requests
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# root of documentation to scan
DOCS_ROOT = Path('docs/web')
# User agent to avoid some blocks
HEADERS = {'User-Agent': 'Mozilla/5.0 (GWexpy-Docs-CI)'}
TIMEOUT = 10
RETRIES = 2
CONCURRENCY = 8

# List of domains that are known to be unstable or should be excluded from Fail
WHITELIST = [
    'zenodo.org', # Placeholders/DOIs sometimes take time to resolve
]

def extract_links():
    """Extract http/https links from markdown and rST files."""
    links = set()
    files = list(DOCS_ROOT.rglob('*.md')) + list(DOCS_ROOT.rglob('*.rst'))
    for p in files:
        txt = p.read_text(encoding='utf-8')
        # Extract bare URLs (exclude trailing punctuation and brackets often used in BibTeX/text)
        found = re.findall(r'(https?://[^\s\)\>\]\"\'\`,{}]+)', txt)
        for url in found:
            # Strip trailing punctuation often caught in regex
            url = url.rstrip('.,')
            links.add((p, url))
    return links

def check_link(url):
    """Check availability of a URL with retries."""
    for attempt in range(RETRIES + 1):
        try:
            # Start with HEAD request for speed
            r = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if 200 <= r.status_code < 400:
                return True, r.status_code
            
            # Fallback to GET because some services (like Google Colab) may reject HEAD
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True, stream=True)
            if 200 <= r.status_code < 400:
                return True, r.status_code
                
            return False, r.status_code
        except Exception as e:
            last_err = str(e)
            
    return False, last_err

def main():
    """Main execution of the link check."""
    links = extract_links()
    total = len(links)
    print(f"Extracted {total} unique external links from {DOCS_ROOT}...")
    
    failures = []
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # Submit all check jobs
        future_to_url = {executor.submit(check_link, url): (path, url) for path, url in links}
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_url), 1):
            path, url = future_to_url[future]
            ok, result = future.result()
            
            if not ok:
                # Check if it's in Whitelist
                if any(domain in url for domain in WHITELIST):
                    print(f"[{i}/{total}] Skipped failure (Whitelisted): {url} -> {result}")
                else:
                    failures.append((path, url, result))
                    print(f"[{i}/{total}] FAILED: {url} -> {result}")
            else:
                if i % 10 == 0:
                    print(f"[{i}/{total}] OK: {url}")

    if failures:
        print("\nError: Found broken external links:")
        print("-" * 60)
        for p, u, s in failures:
            print(f"{p}: {u} -> {s}")
        print("-" * 60)
        print(f"Total failures: {len(failures)}")
        sys.exit(1)
    else:
        print("\nSuccess: All external links are healthy.")
        return 0

if __name__ == '__main__':
    main()
