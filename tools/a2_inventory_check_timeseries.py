#!/usr/bin/env python3
"""
A2 Inventory Check Tool for gwexpy/timeseries

This script performs:
1. AST extraction of all function/method definitions in gwexpy/timeseries/
2. Comparison with the ledger CSV (台帳)
3. Classification suspect detection (A2 heuristics)
4. Report generation

Usage:
    python -m tools.a2_inventory_check_timeseries \
        --csv tests/timeseries_all_defs_classified.csv \
        --package gwexpy/timeseries
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Patterns indicating numeric computation (A2 indicators)
A2_INDICATORS = frozenset([
    # numpy operations
    "np.", "numpy.", "np.fft", "np.linalg",
    # scipy operations
    "scipy.", "scipy.signal", "scipy.fft", "scipy.linalg",
    "signal.csd", "signal.psd", "signal.welch", "signal.coherence",
    # astropy
    "astropy.units", "u.Quantity",
    # FFT/spectral
    "fft", "rfft", "ifft", "irfft", "dft", "stft", "dct", "dst",
    # linear algebra
    "linalg", "matmul", "dot", "einsum", "svd", "eig",
    # math operations
    "convolve", "correlate", "filter", "resample", "decimate",
])

# Pattern for external numeric calls  
EXTERNAL_CALL_PATTERN = re.compile(
    r"(np\.|numpy\.|scipy\.|signal\.|linalg\.|fft)"
)

# Pattern for numeric operators
NUMERIC_OP_PATTERN = re.compile(r"[\+\-\*\/\%\@\*\*]")


@dataclass
class FunctionDef:
    """Represents a function/method definition extracted from AST."""
    qualified_name: str
    name: str
    class_name: str | None
    lineno: int
    is_async: bool
    decorators: list[str]
    file: str
    source_lines: tuple[int, int]  # (start, end)


@dataclass
class LedgerEntry:
    """Represents an entry in the ledger CSV."""
    qualified_name: str
    name: str
    class_name: str | None
    lineno: int
    is_async: bool
    decorators: list[str]
    file: str
    a2_flag: str  # "Yes" or "No"
    a2_sub: str   # "A2-a", "A2-b", etc.
    content_type: str  # "数値", "非数値"
    notes: str
    raw_row: dict[str, str]


@dataclass
class DiffResult:
    """Result of ledger-to-code comparison."""
    in_ledger_not_code: list[LedgerEntry] = field(default_factory=list)
    in_code_not_ledger: list[FunctionDef] = field(default_factory=list)
    file_mismatch: list[tuple[LedgerEntry, FunctionDef]] = field(default_factory=list)
    lineno_deviation: list[tuple[LedgerEntry, FunctionDef, int]] = field(default_factory=list)


@dataclass
class ClassificationSuspect:
    """A suspect classification entry."""
    qualified_name: str
    file: str
    lineno: int
    suspect_type: str
    reason: str


class FunctionExtractor(ast.NodeVisitor):
    """AST visitor to extract function and method definitions."""
    
    def __init__(self, file_path: str, module_prefix: str):
        self.file_path = file_path
        self.module_prefix = module_prefix
        self.functions: list[FunctionDef] = []
        self.current_class: str | None = None
        self.nesting_depth = 0
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node, is_async=True)
    
    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, 
                          is_async: bool = False) -> None:
        # Skip nested functions (nested depth > 0)
        if self.nesting_depth > 0:
            return
        
        # Extract decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                # Handle property.setter etc
                parts = []
                current = dec
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                parts.reverse()
                decorators.append(".".join(parts))
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)
        
        # Build qualified name
        if self.current_class:
            qualified_name = f"{self.module_prefix}.{self.current_class}.{node.name}"
        else:
            qualified_name = f"{self.module_prefix}.{node.name}"
        
        # Get end line number
        end_lineno = getattr(node, 'end_lineno', node.lineno)
        
        func_def = FunctionDef(
            qualified_name=qualified_name,
            name=node.name,
            class_name=self.current_class,
            lineno=node.lineno,
            is_async=is_async,
            decorators=decorators,
            file=self.file_path,
            source_lines=(node.lineno, end_lineno),
        )
        self.functions.append(func_def)
        
        # Recurse into function body but mark nesting depth
        self.nesting_depth += 1
        self.generic_visit(node)
        self.nesting_depth -= 1


def extract_functions_from_file(file_path: Path, package_root: Path) -> list[FunctionDef]:
    """Extract all function definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return []
    
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}")
        return []
    
    # Compute module prefix from path
    # Example: gwexpy/timeseries/_analysis.py -> gwexpy.timeseries._analysis
    # Example: gwexpy/timeseries/io/ats.py -> gwexpy.timeseries.io.ats
    rel_path = file_path.relative_to(package_root.parent.parent)  # relative to repo root
    module_parts = list(rel_path.with_suffix('').parts)
    if module_parts[-1] == '__init__':
        module_parts = module_parts[:-1]
    module_prefix = '.'.join(module_parts)
    
    # Make file path relative for comparison (relative to repo root)
    rel_file = str(file_path.relative_to(package_root.parent.parent))
    
    extractor = FunctionExtractor(rel_file, module_prefix)
    extractor.visit(tree)
    return extractor.functions


def scan_package(package_path: Path) -> list[FunctionDef]:
    """Scan all Python files in the package."""
    all_funcs: list[FunctionDef] = []
    package_root = package_path
    
    for py_file in package_path.rglob('*.py'):
        # Skip __pycache__ directories
        if '__pycache__' in str(py_file):
            continue
        funcs = extract_functions_from_file(py_file, package_root)
        all_funcs.extend(funcs)
    
    return all_funcs


def load_ledger(csv_path: Path) -> list[LedgerEntry]:
    """Load the ledger CSV."""
    entries: list[LedgerEntry] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse decorators (stored as string representation of list)
            dec_str = row.get('decorators', '[]')
            try:
                decorators = ast.literal_eval(dec_str) if dec_str else []
            except:
                decorators = []
            
            entry = LedgerEntry(
                qualified_name=row.get('qualified_name', ''),
                name=row.get('name', ''),
                class_name=row.get('class') if row.get('class') else None,
                lineno=int(row.get('lineno', 0)) if row.get('lineno') else 0,
                is_async=row.get('async', '').lower() == 'true',
                decorators=decorators,
                file=row.get('file', ''),
                a2_flag=row.get('A2該当', ''),
                a2_sub=row.get('A2細分類', ''),
                content_type=row.get('内容区分', ''),
                notes=row.get('備考', ''),
                raw_row=dict(row),
            )
            entries.append(entry)
    return entries


def compare_ledger_to_code(ledger: list[LedgerEntry], 
                           code_funcs: list[FunctionDef],
                           line_tolerance: int = 30) -> DiffResult:
    """Compare ledger entries to extracted code definitions."""
    result = DiffResult()
    
    # Create lookup maps
    code_by_qname: dict[str, FunctionDef] = {}
    for func in code_funcs:
        code_by_qname[func.qualified_name] = func
    
    ledger_qnames = set()
    for entry in ledger:
        ledger_qnames.add(entry.qualified_name)
    
    # Check each ledger entry
    for entry in ledger:
        if entry.qualified_name not in code_by_qname:
            result.in_ledger_not_code.append(entry)
        else:
            code_func = code_by_qname[entry.qualified_name]
            
            # Normalize file paths for comparison
            ledger_file = entry.file.replace('\\', '/')
            code_file = code_func.file.replace('\\', '/')
            
            # Check file path match (handle relative path differences)
            ledger_file_name = Path(ledger_file).name
            code_file_name = Path(code_file).name
            
            if ledger_file_name != code_file_name:
                result.file_mismatch.append((entry, code_func))
            
            # Check line number deviation
            deviation = abs(entry.lineno - code_func.lineno)
            if deviation > line_tolerance:
                result.lineno_deviation.append((entry, code_func, deviation))
    
    # Check for code not in ledger
    for func in code_funcs:
        if func.qualified_name not in ledger_qnames:
            result.in_code_not_ledger.append(func)
    
    return result


def get_function_source(file_path: str, start_line: int, end_line: int) -> str:
    """Extract function source code from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return ''.join(lines[start_line - 1:end_line])
    except:
        return ''


def check_a2_classification(ledger: list[LedgerEntry], 
                            code_funcs: list[FunctionDef],
                            repo_root: Path) -> list[ClassificationSuspect]:
    """Check A2 classification for suspects."""
    suspects: list[ClassificationSuspect] = []
    
    # Create code lookup
    code_by_qname = {f.qualified_name: f for f in code_funcs}
    
    for entry in ledger:
        func = code_by_qname.get(entry.qualified_name)
        if not func:
            continue
        
        # Load source code
        file_path = repo_root / func.file
        source = get_function_source(str(file_path), func.source_lines[0], func.source_lines[1])
        source_lower = source.lower()
        
        # Rule 1: content_type=非数値 but has numeric operations
        if entry.content_type == '非数値':
            numeric_indicators = []
            for indicator in ['numpy', 'scipy', 'np.', 'scipy.signal', 'fft', 'linalg', 
                              'welch', 'csd', 'psd', 'astropy.units']:
                if indicator.lower() in source_lower:
                    numeric_indicators.append(indicator)
            
            if numeric_indicators:
                suspects.append(ClassificationSuspect(
                    qualified_name=entry.qualified_name,
                    file=entry.file,
                    lineno=entry.lineno,
                    suspect_type="非数値 with numeric ops",
                    reason=f"Contains: {', '.join(numeric_indicators[:3])}"
                ))
        
        # Rule 2: A2該当=No but has significant numeric operations
        if entry.a2_flag == 'No':
            # Count numeric patterns
            numpy_count = source.count('np.') + source.count('numpy.')
            scipy_count = source.count('scipy.')
            
            # Check for arithmetic on arrays
            has_array_ops = any(op in source for op in ['+ ', '- ', '* ', '/ ', '** ', '@ '])
            
            # Specific numeric functions
            numeric_funcs = ['mean', 'std', 'var', 'sum', 'median', 'sqrt', 'exp', 'log',
                            'sin', 'cos', 'arctan', 'gradient', 'diff', 'cumsum',
                            'corrcoef', 'cov', 'dot', 'matmul']
            found_funcs = [f for f in numeric_funcs if f in source_lower]
            
            if (numpy_count >= 3 or scipy_count >= 1 or 
                (numpy_count >= 1 and len(found_funcs) >= 2)):
                suspects.append(ClassificationSuspect(
                    qualified_name=entry.qualified_name,
                    file=entry.file,
                    lineno=entry.lineno,
                    suspect_type="No A2 but has numeric ops",
                    reason=f"np calls={numpy_count}, scipy calls={scipy_count}, funcs={found_funcs[:3]}"
                ))
        
        # Rule 3: A2該当=Yes but thin wrapper (single external call)
        if entry.a2_flag == 'Yes':
            # Check if it's a very short function with mostly external calls
            lines = [l.strip() for l in source.split('\n') if l.strip() and not l.strip().startswith('#')]
            # Remove docstring lines (rough heuristic)
            non_doc_lines = []
            in_docstring = False
            for line in lines:
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    continue
                if not in_docstring:
                    non_doc_lines.append(line)
            
            # If function body is very short (2-3 lines) and just calls one method
            body_lines = [l for l in non_doc_lines if not l.startswith('def ') and not l.startswith('@')]
            if len(body_lines) <= 3:
                # Check if it's just a return statement with method call
                if any(l.startswith('return ') and '(' in l for l in body_lines):
                    method_calls = re.findall(r'\.([a-z_][a-z0-9_]*)\s*\(', source_lower)
                    if len(method_calls) == 1:
                        suspects.append(ClassificationSuspect(
                            qualified_name=entry.qualified_name,
                            file=entry.file,
                            lineno=entry.lineno,
                            suspect_type="A2 but thin wrapper",
                            reason=f"Short body ({len(body_lines)} lines), calls: {method_calls[0]}"
                        ))
    
    return suspects


def get_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return 'unknown'


def generate_diff_report(diff: DiffResult, output_path: Path, commit_hash: str) -> None:
    """Generate the inventory diff report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Timeseries A2 Inventory Diff Report\n\n")
        f.write(f"**対象 Commit Hash:** `{commit_hash}`\n\n")
        f.write("**実行方法:**\n```bash\n")
        f.write("python -m tools.a2_inventory_check_timeseries \\\n")
        f.write("    --csv tests/timeseries_all_defs_classified.csv \\\n")
        f.write("    --package gwexpy/timeseries\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        
        # Category 1: In ledger but not in code
        f.write("## 1. 台帳にあるが実コードに存在しない\n\n")
        if diff.in_ledger_not_code:
            f.write("| qualified_name | file | lineno | 推定原因 |\n")
            f.write("|----------------|------|--------|----------|\n")
            for entry in diff.in_ledger_not_code:
                reason = "削除/改名?"
                f.write(f"| `{entry.qualified_name}` | {entry.file} | {entry.lineno} | {reason} |\n")
        else:
            f.write("*なし*\n")
        f.write("\n")
        
        # Category 2: In code but not in ledger
        f.write("## 2. 実コードにあるが台帳にない（台帳の漏れ）\n\n")
        if diff.in_code_not_ledger:
            f.write("| qualified_name | file | lineno | 推定原因 |\n")
            f.write("|----------------|------|--------|----------|\n")
            for func in diff.in_code_not_ledger:
                reason = "新規追加?"
                f.write(f"| `{func.qualified_name}` | {func.file} | {func.lineno} | {reason} |\n")
        else:
            f.write("*なし*\n")
        f.write("\n")
        
        # Category 3: File path mismatch
        f.write("## 3. ファイルパス不一致\n\n")
        if diff.file_mismatch:
            f.write("| qualified_name | 台帳file | 実コードfile | 推定原因 |\n")
            f.write("|----------------|----------|--------------|----------|\n")
            for entry, func in diff.file_mismatch:
                reason = "移動/誤記?"
                f.write(f"| `{entry.qualified_name}` | {entry.file} | {func.file} | {reason} |\n")
        else:
            f.write("*なし*\n")
        f.write("\n")
        
        # Category 4: Line number deviation
        f.write("## 4. 行番号の大幅乖離（±30行超）\n\n")
        if diff.lineno_deviation:
            f.write("| qualified_name | 台帳lineno | 実コードlineno | 差分 |\n")
            f.write("|----------------|------------|----------------|------|\n")
            for entry, func, dev in diff.lineno_deviation:
                f.write(f"| `{entry.qualified_name}` | {entry.lineno} | {func.lineno} | {dev} |\n")
        else:
            f.write("*なし*\n")
        f.write("\n")
        
        # Summary
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write(f"- 台帳にあるが実コードに存在しない: {len(diff.in_ledger_not_code)} 件\n")
        f.write(f"- 実コードにあるが台帳にない: {len(diff.in_code_not_ledger)} 件\n")
        f.write(f"- ファイルパス不一致: {len(diff.file_mismatch)} 件\n")
        f.write(f"- 行番号の大幅乖離: {len(diff.lineno_deviation)} 件\n")


def generate_suspect_report(suspects: list[ClassificationSuspect], 
                           output_path: Path, commit_hash: str) -> None:
    """Generate the classification suspects report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Timeseries A2 Classification Suspects Report\n\n")
        f.write(f"**対象 Commit Hash:** `{commit_hash}`\n\n")
        f.write("**注意:** このレポートは機械的ヒューリスティックによる疑義リストであり、")
        f.write("断定的な結論ではありません。\n\n")
        f.write("---\n\n")
        
        # Group by suspect type
        by_type: dict[str, list[ClassificationSuspect]] = {}
        for s in suspects:
            by_type.setdefault(s.suspect_type, []).append(s)
        
        for suspect_type, items in by_type.items():
            f.write(f"## {suspect_type}\n\n")
            f.write("| qualified_name | file:lineno | 根拠 |\n")
            f.write("|----------------|-------------|------|\n")
            for item in items:
                f.write(f"| `{item.qualified_name}` | {item.file}:{item.lineno} | {item.reason} |\n")
            f.write("\n")
        
        # Summary
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write(f"- 疑義項目数合計: {len(suspects)} 件\n")
        for suspect_type, items in by_type.items():
            f.write(f"  - {suspect_type}: {len(items)} 件\n")


def update_ledger(ledger: list[LedgerEntry], 
                  diff: DiffResult,
                  code_funcs: list[FunctionDef],
                  output_path: Path) -> None:
    """Update ledger with missing entries and mark removed ones."""
    # Get fieldnames from first entry
    if not ledger:
        return
    
    fieldnames = list(ledger[0].raw_row.keys())
    if 'status' not in fieldnames:
        fieldnames.append('status')
    
    # Mark removed entries
    removed_qnames = {e.qualified_name for e in diff.in_ledger_not_code}
    
    # Prepare updated rows
    updated_rows = []
    for entry in ledger:
        row = dict(entry.raw_row)
        if entry.qualified_name in removed_qnames:
            row['status'] = 'removed'
        else:
            row['status'] = ''
        updated_rows.append(row)
    
    # Add new entries
    for func in diff.in_code_not_ledger:
        row = {
            'qualified_name': func.qualified_name,
            'name': func.name,
            'class': func.class_name or '',
            'lineno': str(func.lineno),
            'async': str(func.is_async),
            'decorators': str(func.decorators),
            'file': func.file,
            'A2該当': 'TBD',
            'A2細分類': 'TBD',
            '内容区分': 'TBD',
            '備考': 'auto-added',
            'status': 'new',
        }
        updated_rows.append(row)
    
    # Write updated CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(updated_rows)


def main():
    parser = argparse.ArgumentParser(
        description="A2 inventory check for gwexpy/timeseries"
    )
    parser.add_argument(
        '--csv', required=True,
        help='Path to the ledger CSV file'
    )
    parser.add_argument(
        '--package', required=True,
        help='Path to the package directory (e.g., gwexpy/timeseries)'
    )
    parser.add_argument(
        '--line-tolerance', type=int, default=30,
        help='Line number tolerance for deviation check (default: 30)'
    )
    parser.add_argument(
        '--output-dir', default='reports',
        help='Output directory for reports (default: reports)'
    )
    args = parser.parse_args()
    
    # Resolve paths
    repo_root = Path.cwd()
    csv_path = repo_root / args.csv
    package_path = repo_root / args.package
    output_dir = repo_root / args.output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== A2 Inventory Check ===")
    print(f"Ledger CSV: {csv_path}")
    print(f"Package: {package_path}")
    print()
    
    # Step 1: Extract functions from code
    print("Scanning package for function definitions...")
    code_funcs = scan_package(package_path)
    print(f"  Found {len(code_funcs)} function/method definitions")
    
    # Step 2: Load ledger
    print("Loading ledger CSV...")
    ledger = load_ledger(csv_path)
    print(f"  Loaded {len(ledger)} entries")
    
    # Step 3: Compare
    print("Comparing ledger to code...")
    diff = compare_ledger_to_code(ledger, code_funcs, args.line_tolerance)
    print(f"  In ledger but not code: {len(diff.in_ledger_not_code)}")
    print(f"  In code but not ledger: {len(diff.in_code_not_ledger)}")
    print(f"  File mismatch: {len(diff.file_mismatch)}")
    print(f"  Line deviation (>{args.line_tolerance}): {len(diff.lineno_deviation)}")
    
    # Step 4: Check A2 classification
    print("Checking A2 classifications...")
    suspects = check_a2_classification(ledger, code_funcs, repo_root)
    print(f"  Found {len(suspects)} classification suspects")
    
    # Step 5: Generate reports
    commit_hash = get_commit_hash()
    
    diff_report_path = output_dir / 'timeseries_a2_inventory_diff.md'
    print(f"Generating diff report: {diff_report_path}")
    generate_diff_report(diff, diff_report_path, commit_hash)
    
    suspect_report_path = output_dir / 'timeseries_a2_classification_suspects.md'
    print(f"Generating suspects report: {suspect_report_path}")
    generate_suspect_report(suspects, suspect_report_path, commit_hash)
    
    # Step 6: Update ledger if needed
    needs_update = diff.in_ledger_not_code or diff.in_code_not_ledger
    if needs_update:
        updated_csv_path = csv_path.with_suffix('.updated.csv')
        print(f"Updating ledger: {updated_csv_path}")
        update_ledger(ledger, diff, code_funcs, updated_csv_path)
        
        # Also generate a diff summary
        diff_summary_path = output_dir / 'timeseries_ledger_diff_summary.txt'
        with open(diff_summary_path, 'w') as f:
            f.write(f"Ledger update summary\n")
            f.write(f"=====================\n\n")
            f.write(f"Removed entries (marked status=removed): {len(diff.in_ledger_not_code)}\n")
            for e in diff.in_ledger_not_code:
                f.write(f"  - {e.qualified_name}\n")
            f.write(f"\nNew entries (status=new, A2=TBD): {len(diff.in_code_not_ledger)}\n")
            for func in diff.in_code_not_ledger:
                f.write(f"  - {func.qualified_name}\n")
    
    print()
    print("=== Done ===")


if __name__ == '__main__':
    main()
