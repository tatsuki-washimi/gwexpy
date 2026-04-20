#!/usr/bin/env python3
# scripts/run_quickstart_test.py
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_python_blocks(path):
    """Extract python code blocks from markdown files."""
    txt = path.read_text(encoding='utf-8')
    # Match markdown fenced code blocks: ```python ... ```
    blocks = re.findall(r'```python\n(.*?)```', txt, flags=re.DOTALL)
    return blocks

def run_code(code, timeout=60):
    """Run a piece of python code in a temporary file and return result."""
    # Prepend some standard imports or settings if needed
    # For CI, we use Agg backend for matplotlib to avoid GUI errors
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tmp:
        # Mock .show() so it doesn't block
        code = code.replace('.show()', '# .show() mocked')
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, '', f"Execution timed out after {timeout} seconds."
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    """Main execution of the quickstart code test."""
    parser = argparse.ArgumentParser(description="Test and verify code blocks in Quickstart documentation.")
    parser.add_argument('--lang', default='ja', choices=['ja', 'en'], help="Language sub-directory to test.")
    args = parser.parse_args()

    target_file = Path(f'docs/web/{args.lang}/user_guide/quickstart.md')
    if not target_file.exists():
        print(f"Error: {target_file} not found.")
        sys.exit(1)

    print(f"Testing code blocks in {target_file}...")
    blocks = extract_python_blocks(target_file)

    if not blocks:
        print("No python code blocks found to test.")
        return 0

    failures = []
    for i, block in enumerate(blocks, 1):
        # Heuristic: Skip blocks that only show installation commands (often starts with pip or !)
        first_line = block.strip().split('\n')[0]
        if 'pip install' in first_line or first_line.startswith('!'):
            print(f"[{i}/{len(blocks)}] Skipping installation/non-python block.")
            continue

        print(f"[{i}/{len(blocks)}] Executing code block...")
        rc, out, err = run_code(block)

        if rc != 0:
            failures.append({
                'index': i,
                'rc': rc,
                'stderr': err,
                'stdout': out
            })
            print(f"[{i}/{len(blocks)}] FAILED.")
        else:
            print(f"[{i}/{len(blocks)}] OK.")

    if failures:
        print("\nError: Quickstart code execution failed:")
        print("-" * 60)
        for f in failures:
            print(f"Block #{f['index']} (rc={f['rc']}):")
            print("STDERR:")
            print(f['stderr'])
            # Only show stdout if stderr is empty or short
            if not f['stderr']:
                print("STDOUT:")
                print(f['stdout'])
        print("-" * 60)
        print(f"Total blocks tested: {len(blocks)}, Total failures: {len(failures)}")
        sys.exit(1)
    else:
        print("\nSuccess: All python blocks in Quickstart executed successfully.")
        return 0

if __name__ == '__main__':
    main()
