"""Generate the deterministic HDF5 provenance tutorial notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _cell_id(prefix: str, source: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id("md", source),
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id("cd", source),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


ROUNDTRIP_CODE = """\
import tempfile
from pathlib import Path

import numpy as np

import gwexpy

gwexpy.register_all()

values = np.array([0.0, 1.0, 0.5, -1.0], dtype=float)
metadata = {
    "channel": "K1:TEST-STRAIN",
    "analysis": {"sample_rate_hz": 4.0, "window": "hann"},
    "labels": ["synthetic", "tutorial"],
}
provenance = {
    "pipeline": {"name": "hdf5-provenance-tutorial", "version": 1},
    "inputs": ["deterministic synthetic samples"],
    "parameters": {"sample_rate_hz": 4.0, "calibration": "not applied"},
}

ts = gwexpy.TimeSeries(
    values,
    t0_ns=1234567890123456789,
    sample_rate=4.0,
    name="K1:TEST-STRAIN",
    unit="ct",
)
ts.metadata = metadata
ts.provenance = provenance

with tempfile.TemporaryDirectory() as temporary_directory:
    archive_path = Path(temporary_directory) / "provenance.h5"
    ts.write(archive_path, format="hdf5")
    restored = gwexpy.TimeSeries.read(archive_path, format="hdf5")

    assert restored.t0_gps_ns == 1234567890123456789
    assert restored.metadata == metadata
    assert restored.provenance == provenance
    np.testing.assert_array_equal(restored.value, values)

print(f"Restored exact GPS epoch: {restored.t0_gps_ns}")
print(f"Restored metadata keys: {sorted(restored.metadata)}")
print(f"Restored provenance keys: {sorted(restored.provenance)}")
"""


EN_CELLS = [
    md("""\
# HDF5 provenance with public GWexpy APIs

This tutorial keeps analysis context next to a `TimeSeries` while using only
the public GWexpy registration and HDF5 APIs. The example is deterministic,
offline, and uses four synthetic samples.
"""),
    md("""\
## Automatic sidecar storage

Call `gwexpy.register_all()` before using format registration. Then assign
JSON-safe mappings to `.metadata` and `.provenance` and use
`object.write(..., format="hdf5")` and the matching `.read(...,
format="hdf5")` API.

GWexpy automatically stores this extra state in one root HDF5 attribute named
`_gwexpy_sidecar_json_v1`. The attribute also carries the exact nanosecond
epoch state when it is available. Users must not manually edit this attribute;
edit `.metadata` or `.provenance` before writing instead.
"""),
    md("## Create a JSON-safe TimeSeries"),
    code(ROUNDTRIP_CODE),
    md("""\
## Verify the round trip

The final assertions check the exact integer `t0_gps_ns`, the complete
structured metadata mapping, the complete provenance mapping, and the sample
values after the HDF5 round trip.
"""),
]


JA_CELLS = [
    md("""\
# 公開 GWexpy API による HDF5 プロバナンス

このチュートリアルでは、公開されている登録処理と HDF5 API だけを使い、
`TimeSeries` の近くに解析コンテキストを保持します。
例は決定的かつオフラインで動作し、4 個の合成サンプルを使います。
"""),
    md("""\
## 自動サイドカー保存

フォーマット登録を使う前に `gwexpy.register_all()` を呼び出します。
その後、JSON として安全なマッピングを `.metadata` と `.provenance` に設定し、
公開 API の `object.write(..., format="hdf5")` と対応する
`.read(..., format="hdf5")` を使います。

GWexpy は、この追加状態を `_gwexpy_sidecar_json_v1` という名前の単一の
HDF5 ルート属性に自動保存します。
この属性には、利用可能な場合にナノ秒単位の時刻の状態も含まれます。
利用者がこの属性を手動で編集することはできません。
書き込む前に `.metadata` または `.provenance` を編集してください。
"""),
    md("## JSON として安全な TimeSeries を作る"),
    code(ROUNDTRIP_CODE),
    md("""\
## ラウンドトリップを検証する

最後のアサーションでは、HDF5 ラウンドトリップ後の整数
`t0_gps_ns`、構造化されたメタデータ全体、プロバナンス全体、サンプル値を
検証します。
"""),
]


def write_nb(cells: list[dict], path: Path) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Written: {path}")


if __name__ == "__main__":
    root = Path(__file__).parents[2]
    write_nb(
        EN_CELLS,
        root / "docs/web/en/user_guide/tutorials/case_hdf5_provenance.ipynb",
    )
    write_nb(
        JA_CELLS,
        root / "docs/web/ja/user_guide/tutorials/case_hdf5_provenance.ipynb",
    )
    print("Done.")
