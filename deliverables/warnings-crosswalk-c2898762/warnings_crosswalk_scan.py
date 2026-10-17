#!/usr/bin/env python3
"""Dual inventory of warnings.warn call sites for GWexpy #781 crosswalk.

Universe: git-tracked gwexpy/**/*.py at pinned commit (396 files).
Outputs JSONL records and summary JSON.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

PIN_DEFAULT = "c2898762ca68e20a1a241fffb3c3262b0a27290b"
WARN_RE = re.compile(r"\bwarnings\.warn\s*\(", re.MULTILINE)
CATEGORY_RE = re.compile(r"\b([A-Za-z_]\w*Warning)\b")


@dataclass
class Site:
    scan: str
    path: str
    line: int
    end_line: int
    col: int
    call_id: str
    expr_snippet: str
    has_stacklevel_kw: bool | None  # None for text-only heuristics
    has_category_kw: bool | None
    category_hint: str | None
    stacklevel_text: bool  # A-style: 'stacklevel' token inside paren-balanced span
    category_text: bool  # A-style: Warning type token inside span
    notes: list[str] = field(default_factory=list)


def git_list_files(repo: Path, commit: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit, "--", "gwexpy"],
        text=True,
    )
    return sorted(line for line in out.splitlines() if line.endswith(".py"))


def git_show_file(repo: Path, commit: str, relpath: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{relpath}"],
        text=True,
    )


def paren_span(source: str, open_index: int) -> tuple[int, int] | None:
    """Return (start, end_exclusive) of balanced parens starting at source[open_index]=='('."""
    if open_index >= len(source) or source[open_index] != "(":
        return None
    depth = 0
    i = open_index
    in_str: str | None = None
    escape = False
    while i < len(source):
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
            i += 1
            continue
        if ch == "#":
            # skip to end of line
            nl = source.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return open_index, i + 1
        i += 1
    return None


def line_of_offset(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    last_nl = source.rfind("\n", 0, offset)
    col = offset if last_nl == -1 else offset - last_nl - 1
    return line, col


def scan_text_calls(source: str, relpath: str) -> list[Site]:
    sites: list[Site] = []
    for m in WARN_RE.finditer(source):
        open_paren = m.end() - 1
        span = paren_span(source, open_paren)
        if span is None:
            snippet = source[m.start() : m.start() + 80].replace("\n", " ")
            line, col = line_of_offset(source, m.start())
            sites.append(
                Site(
                    scan="A_text",
                    path=relpath,
                    line=line,
                    end_line=line,
                    col=col,
                    call_id=f"{relpath}:{line}:{col}",
                    expr_snippet=snippet,
                    has_stacklevel_kw=None,
                    has_category_kw=None,
                    category_hint=None,
                    stacklevel_text=False,
                    category_text=False,
                    notes=["unbalanced_parens"],
                )
            )
            continue
        start, end = span
        call_text = source[m.start() : end]
        inner = source[open_paren + 1 : end - 1]
        line, col = line_of_offset(source, m.start())
        end_line, _ = line_of_offset(source, end - 1)
        stacklevel_text = "stacklevel" in inner
        category_text = bool(CATEGORY_RE.search(inner))
        cat_m = CATEGORY_RE.search(inner)
        sites.append(
            Site(
                scan="A_text",
                path=relpath,
                line=line,
                end_line=end_line,
                col=col,
                call_id=f"{relpath}:{line}:{col}",
                expr_snippet=call_text.replace("\n", " ")[:240],
                has_stacklevel_kw=None,
                has_category_kw=None,
                category_hint=cat_m.group(1) if cat_m else None,
                stacklevel_text=stacklevel_text,
                category_text=category_text,
                notes=["multiline" if end_line > line else "single_line"],
            )
        )
    return sites


class WarnVisitor(ast.NodeVisitor):
    def __init__(self, relpath: str, source: str) -> None:
        self.relpath = relpath
        self.source = source
        self.sites: list[Site] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "warnings"
                and node.func.attr == "warn"
            ):
                self._record(node)
        elif isinstance(node.func, ast.Name) and node.func.id == "warn":
            # from warnings import warn — track separately
            self._record(node, imported_warn=True)
        self.generic_visit(node)

    def _record(self, node: ast.Call, imported_warn: bool = False) -> None:
        line = node.lineno
        col = node.col_offset
        end_line = getattr(node, "end_lineno", line) or line
        has_sl = any(k.arg == "stacklevel" for k in node.keywords)
        cat_kw = next((k for k in node.keywords if k.arg == "category"), None)
        has_cat = cat_kw is not None
        category_hint = None
        if cat_kw is not None and isinstance(cat_kw.value, ast.Name):
            category_hint = cat_kw.value.id
        if not has_cat and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                pass
            # category as second positional Name
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Name):
                if node.args[1].id.endswith("Warning"):
                    has_cat = True
                    category_hint = node.args[1].id
        snippet = ast.get_source_segment(self.source, node) or "warnings.warn(...)"
        notes = []
        if imported_warn:
            notes.append("imported_warn_name")
        if end_line > line:
            notes.append("multiline")
        self.sites.append(
            Site(
                scan="B_ast",
                path=self.relpath,
                line=line,
                end_line=end_line,
                col=col,
                call_id=f"{self.relpath}:{line}:{col}",
                expr_snippet=snippet.replace("\n", " ")[:240],
                has_stacklevel_kw=has_sl,
                has_category_kw=has_cat,
                category_hint=category_hint,
                stacklevel_text=has_sl,
                category_text=has_cat,
                notes=notes,
            )
        )


def scan_ast_calls(source: str, relpath: str) -> list[Site]:
    try:
        tree = ast.parse(source, filename=relpath)
    except SyntaxError as exc:
        return [
            Site(
                scan="B_ast",
                path=relpath,
                line=exc.lineno or 1,
                end_line=exc.lineno or 1,
                col=exc.offset or 0,
                call_id=f"{relpath}:syntax_error",
                expr_snippet="",
                has_stacklevel_kw=None,
                has_category_kw=None,
                category_hint=None,
                stacklevel_text=False,
                category_text=False,
                notes=[f"syntax_error:{exc.msg}"],
            )
        ]
    v = WarnVisitor(relpath, source)
    v.visit(tree)
    return v.sites


def naive_line_grep_count(source: str) -> int:
    return sum(1 for line in source.splitlines() if "warnings.warn" in line and not line.lstrip().startswith("#"))


def comment_string_only_matches(source: str, relpath: str) -> list[dict[str, Any]]:
    """Find warnings.warn mentions that are not live code (comments/docstrings)."""
    findings: list[dict[str, Any]] = []
    try:
        tokens = list(tokenize.tokenize(BytesIO(source.encode()).readline))
    except tokenize.TokenError:
        return findings
    code_spans = [
        (t.start, t.end)
        for t in tokens
        if t.type not in (
            tokenize.COMMENT,
            tokenize.STRING,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.ENCODING,
        )
    ]

    def in_code(pos: tuple[int, int]) -> bool:
        for start, end in code_spans:
            if start <= pos <= end:
                return True
        return False

    for m in WARN_RE.finditer(source):
        line, col = line_of_offset(source, m.start())
        if not in_code((line, col)):
            findings.append({"path": relpath, "line": line, "col": col, "kind": "non_code_token"})
    return findings


def site_key(s: Site) -> str:
    return f"{s.path}:{s.line}"


def classify_warn_line_window(
    repo: Path, commit: str, files: list[str]
) -> dict[str, Any]:
    """Replicate A ERR WF-01 / classify_warn.py (4-line window, line hit count)."""
    total = 0
    buckets: dict[tuple[bool, bool], int] = {}
    docstring_only: list[dict[str, Any]] = []
    for relpath in files:
        lines = git_show_file(repo, commit, relpath).splitlines()
        for i, line in enumerate(lines, 1):
            if "warnings.warn" not in line:
                continue
            total += 1
            chunk = "\n".join(lines[i - 1 : i + 3])
            has_cat = bool(
                re.search(r"warnings\.warn\([^,]+,\s*\n?\s*\w+Warning", chunk)
            )
            has_sl = "stacklevel" in chunk
            buckets[(has_cat, has_sl)] = buckets.get((has_cat, has_sl), 0) + 1
            if ":func:`warnings.warn`" in line or "Forwarded to" in line:
                docstring_only.append({"path": relpath, "line": i, "text": line.strip()})
    no_sl = sum(v for (cat, sl), v in buckets.items() if not sl)
    flat = {
        "no_category_no_stacklevel": buckets.get((False, False), 0),
        "no_category_has_stacklevel": buckets.get((False, True), 0),
        "has_category_no_stacklevel": buckets.get((True, False), 0),
        "has_category_has_stacklevel": buckets.get((True, True), 0),
    }
    return {
        "line_hits_total": total,
        "no_stacklevel_line_window": no_sl,
        "buckets": flat,
        "docstring_only_hits": docstring_only,
    }


def summarize(
    a_sites: list[Site],
    b_sites: list[Site],
    naive_line_hits: int,
    ghost_hits: list[dict[str, Any]],
    classify_warn: dict[str, Any],
) -> dict[str, Any]:
    a_keys = {site_key(s): s for s in a_sites}
    b_keys = {site_key(s): s for s in b_sites}

    a_only = sorted(set(a_keys) - set(b_keys))
    b_only = sorted(set(b_keys) - set(a_keys))

    a_no_sl = [s for s in a_sites if not s.stacklevel_text]
    b_no_sl = [s for s in b_sites if s.has_stacklevel_kw is False]

    # A audit buckets (100-call basis)
    a_buckets = {
        "no_category_no_stacklevel": sum(1 for s in a_sites if not s.category_text and not s.stacklevel_text),
        "no_category_has_stacklevel": sum(1 for s in a_sites if not s.category_text and s.stacklevel_text),
        "has_category_no_stacklevel": sum(1 for s in a_sites if s.category_text and not s.stacklevel_text),
        "has_category_has_stacklevel": sum(1 for s in a_sites if s.category_text and s.stacklevel_text),
    }

    cross_sl: list[dict[str, Any]] = []
    for key in sorted(set(a_keys) & set(b_keys)):
        a, b = a_keys[key], b_keys[key]
        if a.stacklevel_text != bool(b.has_stacklevel_kw):
            cross_sl.append(
                {
                    "key": key,
                    "A_stacklevel_text": a.stacklevel_text,
                    "B_has_stacklevel_kw": b.has_stacklevel_kw,
                    "snippet": a.expr_snippet[:160],
                }
            )

    return {
        "counts": {
            "A_text_calls": len(a_sites),
            "B_ast_warnings_warn_calls": len(b_sites),
            "B_ast_imported_warn_calls": sum(1 for s in b_sites if "imported_warn_name" in s.notes),
            "naive_line_grep_hits": naive_line_hits,
            "A_text_only_keys": len(a_only),
            "B_ast_only_keys": len(b_only),
            "A_no_stacklevel_text": len(a_no_sl),
            "B_no_stacklevel_kw": len(b_no_sl),
        },
        "A_audit_bucket_replica": a_buckets,
        "A_only_keys": a_only,
        "B_only_keys": b_only,
        "A_only_details": [asdict(a_keys[k]) for k in a_only],
        "B_only_details": [asdict(b_keys[k]) for k in b_only],
        "stacklevel_definition_mismatch": cross_sl,
        "ghost_non_code_hits": ghost_hits,
        "A_classify_warn_line_window_replica": classify_warn,
        "reconciliation_notes": {
            "count_100_vs_99": (
                "A/classify_warn counts lines containing the substring warnings.warn (100). "
                "One hit is a Sphinx docstring reference, not a live call."
            ),
            "count_59_vs_24": (
                "A 'missing stacklevel' uses stacklevel anywhere in a 4-line window from the "
                "hit line; it over-counts multiline calls whose stacklevel= appears later "
                "than three continuation lines. B/AST uses the full call and finds 24 sites "
                "without a stacklevel keyword."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--commit", default=PIN_DEFAULT)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    files = git_list_files(args.repo, args.commit)
    if len(files) != 396:
        print(f"warn: expected 396 files, got {len(files)}", file=sys.stderr)

    a_all: list[Site] = []
    b_all: list[Site] = []
    naive_total = 0
    ghosts: list[dict[str, Any]] = []

    for relpath in files:
        source = git_show_file(args.repo, args.commit, relpath)
        naive_total += naive_line_grep_count(source)
        ghosts.extend(comment_string_only_matches(source, relpath))
        a_all.extend(scan_text_calls(source, relpath))
        b_all.extend(scan_ast_calls(source, relpath))

    classify = classify_warn_line_window(args.repo, args.commit, files)
    summary = summarize(a_all, b_all, naive_total, ghosts, classify)
    summary["pin_commit"] = args.commit
    summary["file_count"] = len(files)

    args.jsonl.write_text("", encoding="utf-8")
    with args.jsonl.open("a", encoding="utf-8") as out:
        for s in a_all + b_all:
            out.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        out.write(json.dumps({"record_type": "summary", **summary}, ensure_ascii=False) + "\n")

    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    print("A buckets:", summary["A_audit_bucket_replica"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
