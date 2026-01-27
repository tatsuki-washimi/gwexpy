# gwexpy リポジトリ品質分析レポート

**分析日**: 2026-01-27
**分析対象**: `/home/washimi/work/gwexpy`
**総合スコア**: 7.5/10

---

## エグゼクティブサマリー

gwexpy は、GWpy の拡張ライブラリとして**高い品質レベルを維持**しています。テストカバレッジは適切（テスト:本体コード比 = 1:2.9）で、CI/CD パイプラインは堅牢です。

しかし、以下の3つの面で改善機会があります：

1. **型安全性**: `from __future__ import annotations` の採用率が 33% に留まる
2. **型チェック**: MyPy が GUI/UI層を除外しており、ユーザー接点の安全性向上の余地あり
3. **レガシーコード**: 7つのモジュールで MyPy `ignore_errors=true` が設定されている

---

## 1. プロジェクト規模

| 指標 | 数値 |
|------|------|
| Pythonファイル数 | 318個 |
| クラス定義数 | 146個 |
| 関数定義数 | 1,857個 |
| 本体コード総行数 | 58,731行 |
| テストファイル数 | 208個 |
| テストコード総行数 | 20,217行 |
| ドキュメント行数 | 33,464行 |
| 依存パッケージ（基本） | 17個 |

**テスト:本体コード比**: 20,217 : 58,731 ≈ **1 : 2.9** ✓ 適切

---

## 2. コード品質分析

### 2.1 例外処理

**問題**: 過度に広い例外処理 - **2件**

```python
# gwexpy/gui/nds/cache.py (Line 112, 125)
except Exception as exc:
    logger.warning("Unexpected error disconnecting NDSThread: %s", exc)
```

**評価**: P2（中優先度）

**改善提案**:
```python
except (OSError, RuntimeError, AttributeError) as exc:
    logger.warning("Unexpected error disconnecting NDSThread: %s", exc)
```

---

### 2.2 空のブロック (`pass`)

**検出**: **0件** ✓

---

### 2.3 TODO/FIXME コメント

**検出**: **1件** (P2)

```
gwexpy/gui/ui/main_window.py: "Export first item for now (TODO: export all)"
```

**推奨**: GitHub Issues に移行

---

### 2.4 古いPythonバージョン対応

**検出**: **0件** ✓

Python 3.9+ のみサポート（統一的で良好）

---

## 3. 型安全性分析

### 3.1 `from __future__ import annotations` 採用率

```
採用: 106ファイル / 318ファイル = 33.3%
未採用: 212ファイル / 318ファイル = 66.7%
```

**評価**: P1（高優先度改善機会）

**未採用カテゴリ**:
- `gwexpy/detector/` (6ファイル)
- `gwexpy/spectrogram/io/` (複数)
- レガシーモジュール各所

**改善効果**:
- Python 3.9 との互換性向上
- 循環インポート回避
- 型チェック精度向上

---

### 3.2 型ヒント充実度

| 指標 | 数値 |
|------|------|
| 型ヒント付きファイル | 232/318 = 73.0% |
| 戻り値型ヒント数 | 930個 |

**評価**: 良好（P1 で 100% を目指す）

**良い例**:
```python
# gwexpy/timeseries/matrix_core.py
def _apply_timeseries_method(
    self: Any, method_name: str, *args: Any, **kwargs: Any
) -> Any:
    """Apply a TimeSeries method element-wise and rebuild a TimeSeriesMatrix."""
```

---

### 3.3 MyPy 設定と型チェック

**現在の設定** (`pyproject.toml`):
```toml
[tool.mypy]
ignore_missing_imports = true
check_untyped_defs = true
exclude = "gwexpy/gui/(ui|reference-dtt|test-data)/.*|tests/.*"
```

**問題点**:

1. **GUI UI層が除外** → ユーザー接点の型安全性盲点
2. **7つのモジュールで `ignore_errors = true`**:
   ```
   - gwexpy.timeseries.pipeline
   - gwexpy.timeseries.io.win
   - gwexpy.timeseries.io.tdms
   - gwexpy.types.mixin.mixin_legacy
   - gwexpy.types.mixin.signal_interop
   - gwexpy.types.axis_api
   - gwexpy.types.array3d
   ```

**評価**: P1（高優先度改善）

**段階的改善案**:

```toml
# Step 1: GUI UI層を除外から削除
exclude = "gwexpy/gui/reference-dtt/.*|gwexpy/gui/test-data/.*|tests/.*"

# Step 2: ignore_errors を段階的に削除
# 優先度順:
# 1. GUI/NDS (ユーザー接点)
# 2. types/series_matrix (コア)
# 3. レガシーモジュール
```

---

### 3.4 ワイルドカードインポート

**検出**: **10件** (noqa付き)

```python
# gwexpy/detector/io/__init__.py
from gwpy.detector.io import *  # noqa: F403
```

**評価**: P2（中優先度）

**理由**:
- Ruff は noqa で通過
- MyPy は警告の可能性あり
- 型チェック精度低下

**改善案**:
```python
# 明示的インポート
from gwpy.detector.io import Channel, ChannelList
__all__ = ["Channel", "ChannelList"]
```

---

## 4. テストとQA

### 4.1 テスト構造

**テストディレクトリ**: 27個

```
tests/
├── timeseries/    (時系列 - 多数のテスト)
├── types/         (型システム)
├── spectral/      (スペクトル)
├── frequencyseries/
├── spectrogram/
├── gui/           (GUI テスト - xvfb対応)
├── io/            (入出力)
└── ... その他22個
```

**テスト規模**: 208ファイル、20,217行コード

**評価**: P0（合格） ✓

---

### 4.2 CI/CD ワークフロー

| ワークフロー | 実装 | 詳細 | 評価 |
|-------------|------|------|------|
| **Test** | ✓ | Python 3.9-3.12 マトリックス、Ruff、pytest-cov | P0 ✓ |
| **CI** | ✓ | Ruff lint、MyPy（限定） | P1 改善機会 |
| **Docs** | ✓ | Sphinx、多言語対応、GitHub Pages | P0 ✓ |

**GUI テスト**:
- xvfb (X11 仮想フレームバッファ) 対応
- スクリーンショット自動キャプチャ（デバッグ用）

**評価**: P0 ✓

---

## 5. ドキュメント

### 5.1 構造

```
docs/
├── conf.py              (Sphinx設定)
├── index.rst            (ルート)
├── _build/              (生成済みHTML)
├── developers/          (開発者向け)
│   └── archive/plans/   (実装計画アーカイブ)
└── web/                 (Webドキュメント)
```

**統計**:
- Markdown/RST ファイル: 257個
- 総行数: 33,464行

**多言語対応**: ja, en ✓

**評価**: P0（良好） ✓

---

### 5.2 Docstring 充実度

**主要クラス**: Docstring あり ✓

```python
class TimeSeriesMatrixCoreMixin:
    """Core properties and application helpers for TimeSeriesMatrix."""
```

**評価**: P0（良好） ✓

---

## 6. 依存関係

### 6.1 基本依存 (17個)

```
astropy, gwpy, numpy, pandas, scipy, matplotlib,
lalsuite, gwdatafind, gwosc, gpstime, igwn-segments,
ligo.skymap, ligotimegps, dqsegdb2, dateparser, h5py,
typing_extensions
```

### 6.2 オプション依存セット (9個)

- `dev`: 開発ツール
- `gw`: 重力波解析
- `stats`: 統計・ML
- `fitting`: 最適化
- `astro`: 天文学
- `geophysics`: 地球物理学
- `audio`: 音声処理
- `bio`: 生物情報
- `interop`: ML フレームワーク

**評価**: P0（柔軟で適切） ✓

---

## 7. 最近の改善活動（CHANGELOG から）

### 完了済み（P1）

1. **例外処理の改善**
   - 広い `except Exception` を排除
   - **ただし 2件残存** (gui/nds/cache.py)

2. **型安全性拡張**
   - GUI層への型ヒント追加
   - TimeSeriesMatrix の Protocol ベース化

3. **CI 安定化**
   - `qtbot.waitForWindowShown()` → `waitExposed()`

---

## 総合評価

### 強み

| 項目 | スコア | 理由 |
|------|--------|------|
| テスト規模 | ★★★★★ | 20K行のテストコード |
| CI/CD | ★★★★★ | マトリックステスト、自動デプロイ |
| ドキュメント | ★★★★★ | Sphinx + 多言語 |
| 依存関係 | ★★★★★ | モジュール化、オプション依存 |
| コード品質 | ★★★★☆ | 例外処理 2件のみ |
| **型安全性** | ★★★☆☆ | **改善機会あり** |

### 改善機会（優先度付き）

---

## P1（高優先度 - 影響大、実装難度 中）

| # | 項目 | 現状 | 目標 | 工数 | 期待効果 |
|----|------|------|------|------|---------|
| 1 | `from __future__` 採用 | 33% | 100% | 中 | Python 3.9 互換性向上、型チェック精度向上 |
| 2 | MyPy GUI層除外削除 | 除外中 | チェック対象 | 大 | ユーザー接点の安全性向上 |
| 3 | ワイルドカード廃止 | 10件 | 0件 | 中 | 型チェック精度向上、保守性向上 |
| 4 | `ignore_errors` 削除 | 7モジュール | 0 | 大 | 型安全性の完全達成 |

---

## P2（中優先度 - 改善推奨）

| # | 項目 | 現状 | 推奨 | 工数 |
|----|------|------|------|------|
| 1 | 例外処理（残存） | 2件 | 修正 | 小 |
| 2 | TODO コメント | 1件 | Issues 化 | 小 |

---

## P3（低優先度 - 保守性向上）

- テストカバレッジ詳細分析（JSON レポート）
- CONTRIBUTING ガイド強化
- レガシーモジュール整理

---

## 実装ロードマップ（推奨）

### Week 1: 即時対応

```bash
# 1. gui/nds/cache.py の例外処理修正
# 2. TODO コメント → Issues 化
# 工数: 1時間
```

### Week 2-3: 型安全性拡張（高優先度）

```bash
# 1. from __future__ import annotations を全ファイルに追加
# 工数: 4-6時間（自動化可能）

# 2. ワイルドカードインポント → 明示的インポートに変更
# 工数: 3-4時間
```

### Week 4+: MyPy 強化（段階的）

```bash
# 1. GUI UI層の exclude 削除
# 工数: 4-8時間

# 2. ignore_errors モジュール の段階的型安全化
# 優先度: UI → types → legacy
# 工数: 8-16時間（段階的）
```

---

## 実装優先度マトリックス

```
          影響度
          ↑
    P1   │ [型安全性拡張]
          │ ┌─────────────┐
難度 →    │ │  GUI層除外  │
          │ │  ignore_err │
          │ └─────────────┘
    P2   │
          │     [残存例外]
    P3   │
          └─────────────→ 工数
```

---

## 結論

**総合スコア: 7.5/10** → **8.5/10 への改善可能**

gwexpy は**基礎が堅実**で、テスト・ドキュメント・CI/CD が充実しています。

**次のステップ**は、型安全性を 100% に引き上げることで、メンテナンス性とユーザー体験をさらに向上させることです。

**推奨スケジュール**: 4週間で全体的な型安全性達成が可能です。

---

## 付録: ファイル別問題リスト

### 即改善対象

```
gwexpy/gui/nds/cache.py
  - Line 112: except Exception as exc: [P2]
  - Line 125: except Exception as exc: [P2]
```

### 型ヒント未対応（段階改善）

```
gwexpy/detector/ (6ファイル)
  - from __future__ import annotations なし

gwexpy/spectrogram/io/ (複数)
  - ワイルドカードインポート多用
```

### MyPy ignore_errors 設定

```
gwexpy/timeseries/pipeline.py
gwexpy/timeseries/io/win.py
gwexpy/timeseries/io/tdms.py
gwexpy/types/mixin/mixin_legacy.py
gwexpy/types/mixin/signal_interop.py
gwexpy/types/axis_api.py
gwexpy/types/array3d.py
```

---

**分析完了日**: 2026-01-27
**次回レビュー推奨**: 2026-02-27（4週間後）
