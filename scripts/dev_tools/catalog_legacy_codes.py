#!/usr/bin/env python3
"""
Legacy GWpy SampleCodes カタログ生成スクリプト

docs_internal/references/SampleCodes_GWpy/ 以下のすべての .py / .ipynb ファイルを
スキャンし、CATALOG.json と CATALOG_SUMMARY.md を生成する。

Python 標準ライブラリのみで動作（外部依存なし）。

使い方:
    python scripts/dev_tools/catalog_legacy_codes.py
    python scripts/dev_tools/catalog_legacy_codes.py --root docs_internal/references/SampleCodes_GWpy
    python scripts/dev_tools/catalog_legacy_codes.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 対象 API パターン
# ---------------------------------------------------------------------------

# GWpy API キーワード（クラス名 / メソッド名 / 属性名）
GWPY_CLASSES = frozenset(
    [
        "TimeSeries",
        "TimeSeriesDict",
        "FrequencySeries",
        "Spectrogram",
        "SpectrogramDict",
        "StateVector",
        "StateTimeSeries",
        "Segment",
        "SegmentList",
        "DataQualityFlag",
        "Plot",
        "TimeSeriesPlot",
        "EventTable",
        "Table",
    ]
)

GWPY_METHODS = frozenset(
    [
        "read",
        "fetch",
        "fetch_open_data",
        "asd",
        "psd",
        "spectrogram",
        "spectrogram2",
        "fft",
        "bandpass",
        "highpass",
        "lowpass",
        "notch",
        "whiten",
        "resample",
        "crop",
        "coherence",
        "transfer_function",
        "csd",
        "zpk",
    ]
)

# gwexpy API キーワード
GWEXPY_MODULES = frozenset(
    [
        "gwexpy",
        "from gwexpy",
        "import gwexpy",
    ]
)

GWEXPY_CLASSES = frozenset(
    [
        "BifrequencyMap",
        "Bruco",
        "CouplingFunction",
        "fit_series",
        "NInjA",
        "ScalarField",
        "VectorField",
        "TensorField",
    ]
)

# 外部依存として記録するパッケージ
TRACKED_DEPS = frozenset(
    [
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "numba",
        "iminuit",
        "h5py",
        "astropy",
        "lal",
        "lalsuite",
        "obspy",
        "seaborn",
        "sklearn",
        "scikit_learn",
        "torch",
        "tensorflow",
        "sympy",
        "nds2",
    ]
)

# トピック分類: ディレクトリ名 → (topic, subtopic) ヒント
TOPIC_KEYWORDS: dict[str, tuple[str, str]] = {
    # PEM (Physical Environment Monitor)
    "PEM": ("PEM", ""),
    "MAG": ("PEM", "MAG"),
    "MFS": ("PEM", "MAG/MFS"),
    "ACC": ("PEM", "ACC"),
    "MIC": ("PEM", "MIC"),
    "TMP": ("PEM", "TMP"),
    # Injection / Coupling
    "Injection": ("injection", ""),
    "InjectionAnalysis": ("injection", ""),
    "NInjA": ("injection", "NInjA"),
    "injection": ("injection", ""),
    "coupling": ("coupling", ""),
    "Coupling": ("coupling", ""),
    # Seismic / Earthquake
    "Lightning": ("lightning", ""),
    "lightning": ("lightning", ""),
    "Earthquake": ("earthquake", ""),
    "earthquake": ("earthquake", ""),
    "Seismic": ("seismic", ""),
    "seismic": ("seismic", ""),
    # VIS (Vibration Isolation System)
    "Violin": ("violin_mode", ""),
    "violin": ("violin_mode", ""),
    "Cryo": ("cryogenic", ""),
    "cryo": ("cryogenic", ""),
    "VIS": ("VIS", ""),
    "TypeA": ("VIS", "TypeA"),
    "TypeBp": ("VIS", "TypeBp"),
    "VISapps": ("VIS", "apps"),
    "VISsvn": ("VIS", "svn"),
    "typeapayload": ("VIS", "TypeA"),
    # Calibration
    "PowerOutage": ("power_outage", ""),
    "Calibration": ("calibration", ""),
    "calibration": ("calibration", ""),
    "CAL": ("calibration", ""),
    "cal-onsite": ("calibration", "onsite"),
    "Pcal": ("calibration", "Pcal"),
    # GW analysis / DetChar
    "GW": ("GW_analysis", ""),
    "CBC": ("GW_analysis", "CBC"),
    "Burst": ("GW_analysis", "Burst"),
    "lockloss": ("GW_analysis", "lockloss"),
    "locklost": ("GW_analysis", "lockloss"),
    "glitch": ("GW_analysis", "glitch"),
    "noiseb": ("GW_analysis", "noise_budget"),
    "insprange": ("GW_analysis", "range"),
    "hveto": ("GW_analysis", "hveto"),
    "DetChar": ("GW_analysis", "DetChar"),
    "OBS": ("GW_analysis", "observation"),
    # Noise analysis tools
    "bruco": ("noise_analysis", "bruco"),
    "Bruco": ("noise_analysis", "bruco"),
    "fscan": ("noise_analysis", "fscan"),
    "miyopy": ("noise_analysis", "miyopy"),
    # Optics / ASC (Alignment Sensing and Control)
    "aligonb": ("optics_ASC", ""),
    "ASC": ("optics_ASC", ""),
    "tfmodel": ("optics_ASC", "tf_model"),
    "finesse": ("optics_ASC", "finesse"),
    # MIF (Michelson Interferometer)
    "MIF": ("MIF", ""),
    "PySimpleGUI": ("MIF", "GUI"),
    # DGS (Digital control system)
    "DGS": ("DGS", ""),
    # DET (Detector)
    "DET": ("DET", ""),
    # DAQ / Data acquisition
    "DAQ": ("DAQ", ""),
    "EpicsDAQ": ("DAQ", "EPICS"),
    "ndscope": ("data_monitoring", "ndscope"),
    "pcas-controller": ("data_monitoring", "PCAS"),
    "djangoapp": ("data_monitoring", "django"),
    "netgpibdata": ("data_monitoring", "netgpib"),
    # Camera / GigE
    "gige": ("camera_DAQ", "GigE"),
    "GigE": ("camera_DAQ", "GigE"),
    "camlan": ("camera_DAQ", "camlan"),
    # Hardware control
    "stepmotor": ("hardware_control", "stepmotor"),
    "agilis": ("hardware_control", "agilis"),
    "picomotor": ("hardware_control", "picomotor"),
    # Guardian (Detector control sequencer)
    "guardian": ("guardian_control", ""),
    # UserApps (KAGRA user applications)
    "userapps": ("userapps", ""),
    # IO / Format conversion
    "IO": ("io", ""),
    "io": ("io", ""),
    "gbd2gwf": ("io", "format_conversion"),
    "Format": ("io", "format_conversion"),
    # Visualization
    "kagra-gif": ("visualization", "gif"),
    # Docs / Tutorials / Examples / Tests
    "tutorial": ("tutorial", ""),
    "example": ("example", ""),
    "test": ("test", ""),
    # Lib / vendor (not analysis code)
    "site-packages": ("lib_vendor", ""),
    ".python-environments": ("lib_vendor", ""),
}

# lib_vendor パターン（パス部分一致で判定）
LIB_VENDOR_PATH_PATTERNS: tuple[str, ...] = (
    "site-packages",
    "/.python-environments/",
    "/lib/python",
    "/_vendor/",
    "/emacs/.python-environments",
)

# パスキーワードによる追加トピック分類（TOPIC_KEYWORDS で拾えない場合の補完）
# (パス部分文字列, topic, subtopic) の優先順リスト
PATH_TOPIC_RULES: list[tuple[str, str, str]] = [
    # --- lib/vendor ---
    ("site-packages", "lib_vendor", ""),
    (".python-environments", "lib_vendor", ""),
    ("/lib/python", "lib_vendor", ""),
    ("/_vendor/", "lib_vendor", ""),
    # --- calibration ---
    ("/cal/", "calibration", ""),
    ("cal-onsite", "calibration", "onsite"),
    ("/Pcal/", "calibration", "Pcal"),
    # --- GW analysis ---
    ("lockloss", "GW_analysis", "lockloss"),
    ("locklost", "GW_analysis", "lockloss"),
    ("lock_loss", "GW_analysis", "lockloss"),
    ("glitch", "GW_analysis", "glitch"),
    ("insprange", "GW_analysis", "range"),
    ("hveto", "GW_analysis", "hveto"),
    ("noiseb", "GW_analysis", "noise_budget"),
    ("/OBS/", "GW_analysis", "observation"),
    ("/O3", "GW_analysis", "O3"),
    ("/O4", "GW_analysis", "O4"),
    # --- noise analysis ---
    ("bruco", "noise_analysis", "bruco"),
    ("fscan", "noise_analysis", "fscan"),
    ("miyopy", "noise_analysis", "miyopy"),
    # --- optics/ASC ---
    ("aligonb", "optics_ASC", ""),
    ("tfmodel", "optics_ASC", "tf_model"),
    ("260116_asc", "optics_ASC", "ASC"),
    ("finesse-gui", "optics_ASC", "finesse"),
    # --- MIF ---
    ("PySimpleGUI", "MIF", "GUI"),
    ("pysimplegui", "MIF", "GUI"),
    ("finesse", "optics_ASC", "finesse"),
    # --- VIS ---
    ("/TypeA/", "VIS", "TypeA"),
    ("/TypeBp/", "VIS", "TypeBp"),
    ("VISapps", "VIS", "apps"),
    ("VISsvn", "VIS", "svn"),
    ("typeapayload", "VIS", "TypeA"),
    # --- camera / GigE ---
    ("gige", "camera_DAQ", "GigE"),
    ("camlan", "camera_DAQ", "camlan"),
    # --- hardware control ---
    ("stepmotor", "hardware_control", "stepmotor"),
    ("agilis", "hardware_control", "agilis"),
    ("picomotor", "hardware_control", "picomotor"),
    # --- guardian ---
    ("guardian", "guardian_control", ""),
    # --- userapps ---
    ("userapps", "userapps", ""),
    # --- data monitoring ---
    ("ndscope", "data_monitoring", "ndscope"),
    ("pcas-controller", "data_monitoring", "PCAS"),
    ("djangoapp", "data_monitoring", "django"),
    ("netgpibdata", "data_monitoring", "netgpib"),
    # --- DAQ ---
    ("EpicsDAQ", "DAQ", "EPICS"),
    # --- seismic ---
    ("obspy", "seismic", "obspy"),
    # --- visualization ---
    ("kagra-gif", "visualization", "gif"),
]


# ---------------------------------------------------------------------------
# ファイル解析ユーティリティ
# ---------------------------------------------------------------------------


def _normalize_source(source: str) -> str:
    """コンテンツハッシュ計算用に空白・コメントを除去して正規化する。"""
    # 行コメント除去
    source = re.sub(r"#.*$", "", source, flags=re.MULTILINE)
    # 文字列リテラル内の日付・チャンネル名等を汎用化（引用符で囲まれた内容を除去）
    source = re.sub(r'"[^"]{1,80}"', '""', source)
    source = re.sub(r"'[^']{1,80}'", "''", source)
    # 空白を正規化
    source = re.sub(r"\s+", " ", source).strip()
    return source


def _content_hash(source: str) -> str:
    """正規化済みソースの SHA-1 ハッシュ（先頭12文字）を返す。"""
    normalized = _normalize_source(source)
    return hashlib.sha1(normalized.encode("utf-8", errors="replace")).hexdigest()[:12]


def _parse_imports_ast(source: str) -> tuple[list[str], bool]:
    """AST でインポートを解析。失敗した場合は ([], False) を返す。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], False

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
    return imports, True


def _parse_imports_regex(source: str) -> list[str]:
    """正規表現フォールバックでインポートを解析する。"""
    imports: list[str] = []
    for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", source, re.MULTILINE):
        imports.append(m.group(1))
    return imports


def _extract_imports(source: str) -> list[str]:
    """インポートリストを返す（AST 優先、失敗時は正規表現）。"""
    result, ok = _parse_imports_ast(source)
    if not ok:
        result = _parse_imports_regex(source)
    return result


def _find_gwpy_apis(source: str) -> list[str]:
    """ソースから GWpy API 使用状況を抽出する。"""
    found: set[str] = set()

    # クラス名の登場
    for cls in GWPY_CLASSES:
        if re.search(rf"\b{cls}\b", source):
            found.add(cls)

    # .method() パターン（クラス名と組み合わせ）
    for method in GWPY_METHODS:
        pattern = rf"\.{method}\s*\("
        if re.search(pattern, source):
            # どのクラスと組み合わせているか推測
            for cls in GWPY_CLASSES:
                if re.search(rf"\b{cls}\b.*\.{method}\s*\(", source, re.DOTALL):
                    found.add(f"{cls}.{method}")
                    break
            else:
                found.add(f".{method}")

    return sorted(found)


def _find_gwexpy_apis(source: str) -> list[str]:
    """ソースから gwexpy API 使用状況を抽出する。"""
    found: set[str] = set()

    # import 文での gwexpy 使用
    if re.search(r"\bimport gwexpy\b", source) or re.search(
        r"\bfrom gwexpy\b", source
    ):
        found.add("gwexpy")

    # gwexpy クラス / 関数
    for name in GWEXPY_CLASSES:
        if re.search(rf"\b{name}\b", source):
            found.add(name)

    return sorted(found)


def _find_external_deps(imports: list[str]) -> list[str]:
    """インポートリストから外部依存パッケージを特定する。"""
    found: set[str] = set()
    for imp in imports:
        top_level = imp.split(".")[0]
        if top_level in TRACKED_DEPS:
            found.add(top_level)
        # scipy.signal などサブモジュールも記録
        if len(imp.split(".")) > 1 and imp.split(".")[0] in TRACKED_DEPS:
            found.add(imp)
    return sorted(found)


def _extract_functions_ast(source: str) -> list[str]:
    """AST でトップレベル + クラス内の関数名を抽出する。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
    return functions


def _classify_topic(rel_path: str) -> tuple[str, str]:
    """パスのディレクトリ構造からトピック / サブトピックを推定する。

    優先順:
    1. lib_vendor パターン（最優先：分析コードではないため）
    2. TOPIC_KEYWORDS による完全一致 / 部分一致
    3. PATH_TOPIC_RULES によるパス文字列一致（大文字小文字無視）
    """
    p_lower = rel_path.lower()
    parts = Path(rel_path).parts

    # 1. lib_vendor の早期判定
    for pat in LIB_VENDOR_PATH_PATTERNS:
        if pat.lower() in p_lower:
            return "lib_vendor", ""

    # 2. TOPIC_KEYWORDS: 各パーツで完全一致 → 部分一致
    for part in parts:
        if part in TOPIC_KEYWORDS:
            return TOPIC_KEYWORDS[part]
        for kw, (t, s) in TOPIC_KEYWORDS.items():
            if kw.lower() in part.lower():
                return t, s

    # 3. PATH_TOPIC_RULES: パス全体の文字列一致
    for pat, t, s in PATH_TOPIC_RULES:
        if pat.lower() in p_lower:
            return t, s

    # 4. 個人ホームディレクトリ内の追加パターン
    # dotfiles / emacs 設定 → lib_vendor
    if any(k in p_lower for k in [".emacs.d", "/elisp/", "/dotfiles/", "/elpy"]):
        return "lib_vendor", "dotfiles"
    # medm → 制御パネル (Control Display Manager)
    if "/medm/" in p_lower:
        return "data_monitoring", "medm"
    # autobuild → CI/ビルド
    if "autobuild" in p_lower:
        return "misc_scripts", "autobuild"
    # kontrol → Python制御ライブラリ (VIS)
    if "/kontrol/" in p_lower:
        return "VIS", "kontrol"
    # camera/align → カメラ整列
    if any(k in p_lower for k in ["/align/", "/camera/"]):
        return "camera_DAQ", "align"

    return "unknown", ""


def _classify_topic_by_content(
    topic: str,
    subtopic: str,
    gwpy_apis: list[str],
    external_deps: list[str],
    functions_defined: list[str],
    lines_of_code: int,
) -> tuple[str, str]:
    """パス分類で unknown のままだった場合、コンテンツ情報で補完する。

    優先順:
    1. guardian フィルタ関数パターン
    2. nds2 → DAQ
    3. obspy → seismic
    4. GWpy API + DetChar クラス
    5. GWpy API + scipy.signal → noise_analysis
    6. GWpy API (その他) → GW_analysis
    7. astropy / scipy / matplotlib 使用 → GW_analysis
    8. 空/極小ファイル → misc_config
    """
    if topic != "unknown":
        return topic, subtopic

    gwpy = set(gwpy_apis)
    deps = set(external_deps)
    funcs = set(f.lower() for f in functions_defined)

    # guardian 制御フィルタ
    if any(f.startswith("filt_") for f in funcs) or "pre_exec" in funcs or "btn_click" in funcs:
        return "guardian_control", "filter"

    # nds2 → データ取得
    if "nds2" in deps:
        return "DAQ", "nds2"

    # obspy → 地震
    if "obspy" in deps:
        return "seismic", "obspy"

    # DetChar 系 GWpy クラス
    if "DataQualityFlag" in gwpy or "StateVector" in gwpy or "StateTimeSeries" in gwpy:
        return "GW_analysis", "DetChar"

    # フィッティング
    if "scipy.optimize" in deps and any(
        k in funcs for k in ["fitfunc", "fit", "nonlinear_fit", "residual"]
    ):
        return "GW_analysis", "fitting"

    # TimeSeries + scipy.signal → ノイズ解析
    if gwpy and "TimeSeries" in gwpy and "scipy.signal" in deps:
        return "noise_analysis", "timeseries"

    # その他 GWpy 使用
    if gwpy:
        return "GW_analysis", "misc"

    # astropy (天文時刻ライブラリ) → GW解析系
    if "astropy" in deps:
        return "GW_analysis", "astropy"

    # scipy + matplotlib/numpy → 汎用数値解析
    if "scipy" in deps and ("matplotlib" in deps or "numpy" in deps):
        return "GW_analysis", "scipy_analysis"

    # matplotlib → プロット
    if "matplotlib" in deps and lines_of_code > 10:
        return "GW_analysis", "plotting"

    # numpy のみ
    if "numpy" in deps:
        return "GW_analysis", "numpy_analysis"

    # 極小ファイル
    if lines_of_code <= 5:
        return "misc_config", ""

    # それでも不明 → misc_scripts (個人スクリプト)
    return "misc_scripts", ""


def _get_location(rel_path: str) -> str:
    """パスの先頭ディレクトリから作業環境を判別する。"""
    first = Path(rel_path).parts[0] if Path(rel_path).parts else ""
    if first in ("GoogleDrive", "k1ctr", "kmst"):
        return first
    return "unknown"


# ---------------------------------------------------------------------------
# .py ファイルの解析
# ---------------------------------------------------------------------------


def analyze_py(path: Path, root: Path) -> dict[str, Any]:
    """Python ファイルを解析してカタログエントリを返す。"""
    source = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(root))
    stat = path.stat()

    imports = _extract_imports(source)
    gwpy_apis = _find_gwpy_apis(source)
    gwexpy_apis = _find_gwexpy_apis(source)
    external_deps = _find_external_deps(imports)
    functions_defined = _extract_functions_ast(source)
    topic, subtopic = _classify_topic(rel)
    location = _get_location(rel)
    lines = source.count("\n") + 1

    # コンテンツ情報で unknown を補完
    topic, subtopic = _classify_topic_by_content(
        topic, subtopic, gwpy_apis, external_deps, functions_defined, lines
    )

    return {
        "path": rel,
        "location": location,
        "topic": topic,
        "subtopic": subtopic,
        "type": "py",
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
        "gwpy_apis": gwpy_apis,
        "gwexpy_apis": gwexpy_apis,
        "external_deps": external_deps,
        "functions_defined": functions_defined,
        "content_hash": _content_hash(source),
        "duplicate_group": None,  # 後で設定
        "lines_of_code": lines,
        "is_untitled": path.stem.lower().startswith("untitled"),
    }


# ---------------------------------------------------------------------------
# .ipynb ファイルの解析
# ---------------------------------------------------------------------------


def analyze_ipynb(path: Path, root: Path) -> dict[str, Any]:
    """Jupyter Notebook ファイルを解析してカタログエントリを返す。"""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        nb = {}

    rel = str(path.relative_to(root))
    stat = path.stat()
    topic, subtopic = _classify_topic(rel)
    location = _get_location(rel)

    # コードセルのソースを結合
    cells = nb.get("cells", [])
    code_parts: list[str] = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            code_parts.append(src)

    source = "\n".join(code_parts)
    lines = source.count("\n") + 1

    imports = _extract_imports(source)
    gwpy_apis = _find_gwpy_apis(source)
    gwexpy_apis = _find_gwexpy_apis(source)
    external_deps = _find_external_deps(imports)
    functions_defined = _extract_functions_ast(source)

    # コンテンツ情報で unknown を補完
    topic, subtopic = _classify_topic_by_content(
        topic, subtopic, gwpy_apis, external_deps, functions_defined, lines
    )

    return {
        "path": rel,
        "location": location,
        "topic": topic,
        "subtopic": subtopic,
        "type": "ipynb",
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
        "gwpy_apis": gwpy_apis,
        "gwexpy_apis": gwexpy_apis,
        "external_deps": external_deps,
        "functions_defined": functions_defined,
        "content_hash": _content_hash(source),
        "duplicate_group": None,
        "lines_of_code": lines,
        "is_untitled": path.stem.lower().startswith("untitled"),
    }


# ---------------------------------------------------------------------------
# 重複グループの割り当て
# ---------------------------------------------------------------------------


def assign_duplicate_groups(entries: list[dict[str, Any]]) -> None:
    """同じ content_hash を持つエントリに duplicate_group を設定する（in-place）。"""
    hash_to_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        hash_to_entries[entry["content_hash"]].append(entry)

    for h, group in hash_to_entries.items():
        if len(group) > 1:
            # グループ名: 最初のエントリのステムを使用
            representative = Path(group[0]["path"]).stem
            group_name = f"{representative}_{h[:6]}"
            for entry in group:
                entry["duplicate_group"] = group_name


# ---------------------------------------------------------------------------
# サマリ生成
# ---------------------------------------------------------------------------


def generate_summary(entries: list[dict[str, Any]], catalog_path: Path) -> str:
    """CATALOG_SUMMARY.md のテキストを生成する。"""
    total = len(entries)
    py_count = sum(1 for e in entries if e["type"] == "py")
    ipynb_count = sum(1 for e in entries if e["type"] == "ipynb")
    total_bytes = sum(e["size_bytes"] for e in entries)

    # 重複グループ統計
    groups: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        if e["duplicate_group"]:
            groups[e["duplicate_group"]].append(e["path"])
    dup_files = sum(len(v) for v in groups.values())
    unique_patterns = total - dup_files + len(groups)

    # gwexpy 使用ファイル
    gwexpy_files = [e for e in entries if e["gwexpy_apis"]]

    # ロケーション別カウント
    loc_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        loc_counts[e["location"]] += 1

    # トピック別カウント
    topic_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        topic_counts[e["topic"]] += 1

    # GWpy API 頻度
    api_freq: dict[str, int] = defaultdict(int)
    for e in entries:
        for api in e["gwpy_apis"]:
            api_freq[api] += 1

    # 外部依存頻度
    dep_freq: dict[str, int] = defaultdict(int)
    for e in entries:
        # トップレベルパッケージのみカウント
        for dep in e["external_deps"]:
            top = dep.split(".")[0]
            dep_freq[top] += 1

    # 重複グループ上位10件
    top_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    lines: list[str] = [
        "# Legacy GWpy SampleCodes カタログ サマリ",
        "",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"カタログ: `{catalog_path.name}`",
        "",
        "---",
        "",
        "## 全体統計",
        "",
        f"| 項目 | 値 |",
        f"|------|-----|",
        f"| 総ファイル数 | {total:,} |",
        f"| Python (.py) | {py_count:,} |",
        f"| Notebook (.ipynb) | {ipynb_count:,} |",
        f"| 総サイズ | {total_bytes / 1024 / 1024:.1f} MB |",
        f"| ユニークパターン（重複除去後推定） | ~{unique_patterns:,} |",
        f"| gwexpy 使用ファイル | {len(gwexpy_files):,} |",
        "",
        "---",
        "",
        "## 作業環境別ファイル数",
        "",
        "| 環境 | ファイル数 |",
        "|------|-----------|",
    ]
    for loc, cnt in sorted(loc_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {loc} | {cnt:,} |")

    lines += [
        "",
        "---",
        "",
        "## トピック別ファイル数",
        "",
        "| トピック | ファイル数 |",
        "|---------|-----------|",
    ]
    for topic, cnt in sorted(topic_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {topic} | {cnt:,} |")

    lines += [
        "",
        "---",
        "",
        "## 頻出 GWpy API（上位20件）",
        "",
        "| API | 使用ファイル数 |",
        "|-----|--------------|",
    ]
    for api, cnt in sorted(api_freq.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"| `{api}` | {cnt:,} |")

    lines += [
        "",
        "---",
        "",
        "## 外部依存（上位10件）",
        "",
        "| パッケージ | 使用ファイル数 |",
        "|-----------|--------------|",
    ]
    for dep, cnt in sorted(dep_freq.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"| `{dep}` | {cnt:,} |")

    lines += [
        "",
        "---",
        "",
        "## 重複グループ上位10件",
        "",
        "| グループ名 | ファイル数 | 代表例 |",
        "|-----------|----------|--------|",
    ]
    for group_name, paths in top_groups:
        lines.append(
            f"| {group_name} | {len(paths)} | `{Path(paths[0]).name}` |"
        )

    if gwexpy_files:
        lines += [
            "",
            "---",
            "",
            "## gwexpy 使用ファイル（最優先参照対象）",
            "",
            "| ファイル | gwexpy API |",
            "|---------|-----------|",
        ]
        for e in sorted(gwexpy_files, key=lambda x: len(x["gwexpy_apis"]), reverse=True)[
            :20
        ]:
            apis = ", ".join(e["gwexpy_apis"][:5])
            lines.append(f"| `{e['path']}` | {apis} |")

    lines += ["", "---", "", "*このファイルは自動生成されます（gitignore対象）*", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Legacy GWpy SampleCodes をスキャンして CATALOG.json / CATALOG_SUMMARY.md を生成する"
    )
    parser.add_argument(
        "--root",
        default="docs_internal/references/SampleCodes_GWpy",
        help="スキャン対象のルートディレクトリ（デフォルト: docs_internal/references/SampleCodes_GWpy）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="出力先ディレクトリ（デフォルト: --root と同じ）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="進捗を詳細表示する"
    )
    args = parser.parse_args()

    # リポジトリルートからの相対パスを解決
    repo_root = Path(__file__).resolve().parents[2]
    root = (repo_root / args.root).resolve()
    output_dir = (repo_root / args.output_dir).resolve() if args.output_dir else root

    if not root.exists():
        print(f"ERROR: ディレクトリが見つかりません: {root}", file=sys.stderr)
        return 1

    catalog_path = output_dir / "CATALOG.json"
    summary_path = output_dir / "CATALOG_SUMMARY.md"

    # ファイル収集
    py_files = sorted(root.rglob("*.py"))
    ipynb_files = sorted(root.rglob("*.ipynb"))
    all_files = py_files + ipynb_files
    total = len(all_files)

    if args.verbose:
        print(f"スキャン対象: {root}")
        print(f"  .py    : {len(py_files):,} ファイル")
        print(f"  .ipynb : {len(ipynb_files):,} ファイル")
        print(f"  合計   : {total:,} ファイル")

    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, path in enumerate(all_files, 1):
        if args.verbose and i % 100 == 0:
            print(f"  [{i}/{total}] 処理中...", end="\r")

        try:
            if path.suffix == ".py":
                entry = analyze_py(path, root)
            else:
                entry = analyze_ipynb(path, root)
            entries.append(entry)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            if args.verbose:
                print(f"\nWARN: {path}: {exc}", file=sys.stderr)

    if args.verbose:
        print(f"\n解析完了: {len(entries):,} エントリ（エラー: {len(errors)}）")

    # 重複グループ割り当て
    assign_duplicate_groups(entries)

    # CATALOG.json 出力
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"CATALOG.json → {catalog_path}")

    # CATALOG_SUMMARY.md 出力
    summary_text = generate_summary(entries, catalog_path)
    summary_path.write_text(summary_text, encoding="utf-8")

    if args.verbose:
        print(f"CATALOG_SUMMARY.md → {summary_path}")

    # 簡易サマリを常に表示
    py_count = sum(1 for e in entries if e["type"] == "py")
    ipynb_count = sum(1 for e in entries if e["type"] == "ipynb")
    gwexpy_count = sum(1 for e in entries if e["gwexpy_apis"])
    dup_groups = sum(
        1 for e in entries if e["duplicate_group"] is not None
    )

    print(f"完了: {len(entries):,} ファイル（.py: {py_count}, .ipynb: {ipynb_count}）")
    print(f"  gwexpy使用: {gwexpy_count} ファイル")
    print(f"  重複グループ参加ファイル: {dup_groups}")
    print(f"  出力: {catalog_path}")
    print(f"  出力: {summary_path}")

    if errors:
        print(f"\n警告: {len(errors)} ファイルでエラーが発生しました", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
