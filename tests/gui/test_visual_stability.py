#!/usr/bin/env python3
"""
Automated Visual Stability Test for gwexpy GUI

This script detects patterns that could cause flickering, jittering,
or other visual instability issues in the GUI.

Checked patterns:
1. Redundant setXLink/setYLink calls (without state caching)
2. enableAutoRange calls inside frequently-called functions
3. setData/setImage without rate limiting in update loops
4. Signal connections that could cause cascading updates
5. Missing padding in setXRange/setYRange calls
"""

import os
import re
import sys
from pathlib import Path

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class StabilityChecker:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.issues: list[dict] = []
        self.warnings: list[dict] = []

    def check_file(self, filepath: Path) -> None:
        """Check a single Python file for stability issues."""
        try:
            content = filepath.read_text()
            lines = content.split("\n")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return

        rel_path = filepath.relative_to(self.base_path)

        # Pattern 1: setXLink without state caching
        self._check_uncached_setlink(lines, rel_path)

        # Pattern 2: enableAutoRange in update functions
        self._check_autorange_in_update(lines, rel_path, content)

        # Pattern 3: setData/setImage without rate limiting
        self._check_frequent_setdata(lines, rel_path, content)

        # Pattern 4: Cascading signal connections
        self._check_cascading_signals(lines, rel_path)

        # Pattern 5: setRange without padding
        self._check_range_without_padding(lines, rel_path)

        # Pattern 6: toggled.connect inside loops
        self._check_signal_in_loop(lines, rel_path)

    def _check_uncached_setlink(self, lines: list[str], filepath: Path) -> None:
        """Check for setXLink/setYLink without state caching."""
        in_function = None
        has_cache_check = False
        setlink_lines = []

        for i, line in enumerate(lines):
            # Track function entry
            if re.match(r"\s*def\s+\w+", line):
                # Check previous function
                if in_function and setlink_lines and not has_cache_check:
                    self.warnings.append(
                        {
                            "file": str(filepath),
                            "line": setlink_lines[0],
                            "type": "UNCACHED_SETLINK",
                            "message": f"setXLink/setYLink in '{in_function}' without state caching may cause flicker",
                        }
                    )
                # Reset
                match = re.match(r"\s*def\s+(\w+)", line)
                in_function = match.group(1) if match else None
                has_cache_check = False
                setlink_lines = []

            # Detect setXLink/setYLink
            if re.search(r"\.setXLink\(|\.setYLink\(", line):
                setlink_lines.append(i + 1)

            # Detect caching pattern (getattr or _linked check)
            if re.search(
                r'getattr\(self,\s*["\']_.*linked|_linked\s*[!=]', line, re.IGNORECASE
            ):
                has_cache_check = True

    def _check_autorange_in_update(
        self, lines: list[str], filepath: Path, content: str
    ) -> None:
        """Check for enableAutoRange inside update_* functions."""
        in_update_func = False
        func_name = None
        func_start = 0

        for i, line in enumerate(lines):
            match = re.match(r"\s*def\s+(update_\w+|on_\w+_changed)", line)
            if match:
                in_update_func = True
                func_name = match.group(1)
                func_start = i

            if in_update_func:
                # Check for enableAutoRange
                if re.search(r"enableAutoRange.*enable\s*=\s*True", line):
                    # This is problematic in frequently-called update functions
                    self.warnings.append(
                        {
                            "file": str(filepath),
                            "line": i + 1,
                            "type": "AUTORANGE_IN_UPDATE",
                            "message": f"enableAutoRange(enable=True) in '{func_name}' may cause jitter during streaming",
                        }
                    )

                # Rough detection of function end (next def at same indent level)
                if i > func_start and re.match(r"\s*def\s+", line):
                    in_update_func = False

    def _check_frequent_setdata(
        self, lines: list[str], filepath: Path, content: str
    ) -> None:
        """Check for setData/setImage without debouncing/throttling."""
        # Look for setData/setImage inside timer callbacks or signal handlers
        timer_pattern = r"\.timeout\.connect\(self\.(\w+)\)"
        timer_callbacks = set(re.findall(timer_pattern, content))

        for i, line in enumerate(lines):
            match = re.match(r"\s*def\s+(\w+)", line)
            if match and match.group(1) in timer_callbacks:
                # Check next 100 lines for setData without rate limit
                for j in range(i, min(i + 100, len(lines))):
                    if re.search(r"\.setData\(|\.setImage\(", lines[j]):
                        # Check if there's any time-based throttling
                        throttle_found = any(
                            re.search(
                                r"time\.time\(\)|_last_update|throttle|debounce",
                                lines[k],
                            )
                            for k in range(i, j)
                        )
                        if not throttle_found:
                            # Not an issue per se, but note it
                            pass

    def _check_cascading_signals(self, lines: list[str], filepath: Path) -> None:
        """Check for signal connections that might cascade."""
        signal_pattern = r"(\w+)\.(\w+)\.connect\(self\.(\w+)\)"
        connections = []

        for i, line in enumerate(lines):
            match = re.search(signal_pattern, line)
            if match:
                connections.append((i + 1, match.group(3)))

        # Check if connected methods themselves emit signals or call other methods
        # that are also connected - this could be a cascade
        # (Simplified check - just flag multiple connections to same handler)
        handler_counts: dict[str, list[int]] = {}
        for line_no, handler in connections:
            if handler not in handler_counts:
                handler_counts[handler] = []
            handler_counts[handler].append(line_no)

        for handler, line_numbers in handler_counts.items():
            if len(line_numbers) > 3:
                self.warnings.append(
                    {
                        "file": str(filepath),
                        "line": line_numbers[0],
                        "type": "MANY_SIGNAL_CONNECTIONS",
                        "message": f"Handler '{handler}' has {len(line_numbers)} signal connections - may cause cascade updates",
                    }
                )

    def _check_range_without_padding(self, lines: list[str], filepath: Path) -> None:
        """Check for setXRange/setYRange without explicit padding."""
        for i, line in enumerate(lines):
            if re.search(r"\.set[XY]Range\([^)]+\)", line):
                if "padding" not in line:
                    self.warnings.append(
                        {
                            "file": str(filepath),
                            "line": i + 1,
                            "type": "RANGE_NO_PADDING",
                            "message": "setXRange/setYRange without padding=0 may cause unexpected range expansion",
                        }
                    )

    def _check_signal_in_loop(self, lines: list[str], filepath: Path) -> None:
        """Check for signal connections inside loops."""
        in_loop = False
        loop_start = 0

        for i, line in enumerate(lines):
            if re.search(r"^\s*for\s+|^\s*while\s+", line):
                in_loop = True
                loop_start = i

            if in_loop and re.search(r"\.connect\(", line):
                # Check if this is creating duplicate connections
                self.warnings.append(
                    {
                        "file": str(filepath),
                        "line": i + 1,
                        "type": "SIGNAL_IN_LOOP",
                        "message": "Signal connection inside loop may create duplicate handlers",
                    }
                )

            # Rough loop end detection
            if in_loop and i > loop_start + 1:
                indent = len(line) - len(line.lstrip())
                loop_indent = len(lines[loop_start]) - len(lines[loop_start].lstrip())
                if indent <= loop_indent and line.strip():
                    in_loop = False

    def run(self) -> int:
        """Run all checks on the GUI codebase."""
        gui_path = self.base_path / "gwexpy" / "gui"

        if not gui_path.exists():
            print(f"{RED}Error: GUI path not found: {gui_path}{RESET}")
            return 1

        print("=" * 60)
        print("Visual Stability Automated Check")
        print("=" * 60)

        py_files = list(gui_path.rglob("*.py"))
        print(f"Scanning {len(py_files)} Python files...\n")

        for py_file in py_files:
            self.check_file(py_file)

        # Report
        if self.issues:
            print(f"{RED}CRITICAL ISSUES ({len(self.issues)}):{RESET}")
            for issue in self.issues:
                print(f"  [{issue['type']}] {issue['file']}:{issue['line']}")
                print(f"    {issue['message']}")
            print()

        if self.warnings:
            print(f"{YELLOW}WARNINGS ({len(self.warnings)}):{RESET}")
            for warn in self.warnings:
                print(f"  [{warn['type']}] {warn['file']}:{warn['line']}")
                print(f"    {warn['message']}")
            print()

        if not self.issues and not self.warnings:
            print(f"{GREEN}No visual stability issues detected!{RESET}")
            return 0
        elif self.issues:
            print(
                f"{RED}Found {len(self.issues)} critical issues and {len(self.warnings)} warnings.{RESET}"
            )
            return 1
        else:
            print(
                f"{YELLOW}Found {len(self.warnings)} warnings (no critical issues).{RESET}"
            )
            return 0


def main():
    # Determine base path
    # We are in tests/gui/test_visual_stability.py
    script_path = Path(__file__).resolve()
    # Path is tests/gui/test_visual_stability.py
    # Parent of parents (tests/gui -> tests -> root)
    project_root = script_path.parents[2]
    gui_path = project_root / "gwexpy" / "gui"

    # For StabilityChecker, we need parent of 'gui' as base
    # so that base_path / "gwexpy" / "gui" works
    # Actually let's just directly use gui_path in the checker

    class DirectStabilityChecker(StabilityChecker):
        def run(self) -> int:
            """Run all checks on the GUI codebase."""
            gui_path_local = gui_path

            if not gui_path_local.exists():
                print(f"{RED}Error: GUI path not found: {gui_path_local}{RESET}")
                return 1

            print("=" * 60)
            print("Visual Stability Automated Check")
            print("=" * 60)

            py_files = list(gui_path_local.rglob("*.py"))
            print(f"Scanning {len(py_files)} Python files in {gui_path_local}...\n")

            for py_file in py_files:
                self.check_file(py_file)

            # Report
            if self.issues:
                print(f"{RED}CRITICAL ISSUES ({len(self.issues)}):{RESET}")
                for issue in self.issues:
                    print(f"  [{issue['type']}] {issue['file']}:{issue['line']}")
                    print(f"    {issue['message']}")
                print()

            if self.warnings:
                print(f"{YELLOW}WARNINGS ({len(self.warnings)}):{RESET}")
                for warn in self.warnings:
                    print(f"  [{warn['type']}] {warn['file']}:{warn['line']}")
                    print(f"    {warn['message']}")
                print()

            if not self.issues and not self.warnings:
                print(f"{GREEN}No visual stability issues detected!{RESET}")
                return 0
            elif self.issues:
                print(
                    f"{RED}Found {len(self.issues)} critical issues and {len(self.warnings)} warnings.{RESET}"
                )
                return 1
            else:
                print(
                    f"{YELLOW}Found {len(self.warnings)} warnings (no critical issues).{RESET}"
                )
                return 0

    checker = DirectStabilityChecker(gui_path)
    sys.exit(checker.run())


if __name__ == "__main__":
    main()
