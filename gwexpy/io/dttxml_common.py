"""
Shared utilities for parsing dttxml (Diag GUI XML) files.
"""

from __future__ import annotations

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np

SUPPORTED_TS = {"TS"}
SUPPORTED_FREQ = {"PSD", "ASD", "FFT"}
SUPPORTED_MATRIX = {"TF", "STF", "CSD", "COH"}


def _load_external_parser() -> Optional[object]:
    # Security: This mechanism is primarily for internal legacy interop.
    # It is disabled by default and requires explicit environment opt-in.
    if os.environ.get("GWEXPY_ALLOW_DTTXML_EXEC", "0") != "1":
        return None

    # Limit search path to a fixed location relative to the package
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "dttxml_source.txt"
    
    if candidate.exists():
        warnings.warn(
            f"Loading external parser from {candidate} via exec(). "
            "Only use this with trusted source files.",
            UserWarning
        )
        try:
            ns: Dict[str, object] = {}
            # Use a more restricted compile if possible, but here we just need exec
            code = candidate.read_text()
            exec(compile(code, str(candidate), "exec"), ns, ns)
            return ns.get("dtt_read")
        except Exception as e:
            warnings.warn(f"Failed to load external parser: {e}")
            return None
    return None


def _normalize_entry(val):
    if isinstance(val, dict):
        data = np.asarray(val.get("data") or val.get("y") or val.get("value") or [])
        dt = val.get("dt") or val.get("delta_t")
        df = val.get("df") or val.get("delta_f")
        freqs = np.asarray(val.get("frequencies") or val.get("freq") or [])
        epoch = val.get("epoch") or val.get("t0") or val.get("start")
        unit = val.get("unit")
        return {"data": data, "dt": dt, "df": df, "frequencies": freqs, "epoch": epoch, "unit": unit}
    if isinstance(val, (list, tuple, np.ndarray)):
        return {
            "data": np.asarray(val),
            "dt": None,
            "df": None,
            "frequencies": np.array([]),
            "epoch": None,
            "unit": None,
        }
    return {"data": np.asarray([]), "dt": None, "df": None, "frequencies": np.array([]), "epoch": None, "unit": None}


def load_dttxml_products(source):
    """
    Load products from a dttxml file into a normalized mapping.
    """
    parser = _load_external_parser()
    if parser:
        try:
            parsed = parser(source)
            if isinstance(parsed, dict) and "results" in parsed:
                parsed = parsed["results"]
            normalized: Dict[str, dict] = {}
            if isinstance(parsed, dict):
                for prod, entries in parsed.items():
                    payload = {}
                    if isinstance(entries, dict):
                        for key, val in entries.items():
                            payload[key] = _normalize_entry(val)
                    normalized[str(prod).upper()] = payload
            if normalized:
                return normalized
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    tree = ET.parse(source)
    root = tree.getroot()
    normalized: Dict[str, dict] = {}
    for result in root.findall(".//result"):
        prod = (result.get("type") or result.get("name") or "").upper()
        payload = {}
        for ch in result.findall("channel"):
            name = ch.get("name") or ch.get("id") or f"ch{len(payload)}"
            unit = ch.get("unit")
            dt = ch.get("dt") or ch.get("sample") or ch.get("delta_t")
            epoch = ch.get("epoch")
            data = np.fromstring((ch.findtext("data") or ""), sep=",")
            payload[name] = {
                "data": data,
                "dt": float(dt) if dt else None,
                "df": None,
                "frequencies": np.array([]),
                "epoch": epoch,
                "unit": unit,
            }
        for pair in result.findall("pair"):
            row = pair.get("row") or pair.get("from")
            col = pair.get("col") or pair.get("to")
            unit = pair.get("unit")
            df = pair.get("df")
            epoch = pair.get("epoch")
            freqs = np.fromstring((pair.findtext("frequencies") or ""), sep=",")
            data = np.fromstring((pair.findtext("data") or ""), sep=",")
            payload[(row, col)] = {
                "data": data,
                "df": float(df) if df else None,
                "frequencies": freqs,
                "dt": None,
                "epoch": epoch,
                "unit": unit,
            }
        if payload:
            normalized[prod] = payload
    return normalized

