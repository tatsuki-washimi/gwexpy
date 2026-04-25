# MetaData / MetaDataMatrix 監査・修正計画

**Issue:** #243 — Audit MetaData and MetaDataMatrix unit propagation and shape semantics
**作成日:** 2026-04-25
**対象ブランチ:** `claude/plan-metadata-work-nDtXw`

---

## 1. Objectives & Goals

Issue #243 のスコープに沿い、以下を達成する:

1. `MetaData` / `MetaDataMatrix` / `MetaDataDict` における **unit propagation** の正確性を確認・修正する。
2. セッターの **shape mismatch** をサイレントに隠す挙動を排除し、明示的な `ValueError` を発生させる。
3. **DataFrame/CSV ラウンドトリップ** で name/channel/unit/shape が完全に保持されることをテストで保証する。
4. **厳格モード vs 柔軟モード** の unit チェック方針を docstring に明示する。

---

## 2. 監査結果サマリー（現状の問題点）

### 2-A. MetaDataMatrix セッターがシェイプ不一致を隠す（バグ）

`metadata.py:621-628` / `639-646` / `656-659`:

```python
@names.setter
def names(self, value):
    value = np.asarray(value)
    if value.shape != self.shape:
        value = value.reshape(self.shape)  # ← サイレントリシェイプ！
```

`units.setter`、`channels.setter` も同様。
**問題:** サイズが合わない配列を渡しても `reshape` が例外を吸収し、要素の対応がずれたまま設定される可能性がある。
**修正方針:** サイズが一致しない場合は `ValueError` を上げる。
サイズが一致する場合のみ reshape を許可する（フラット配列 → 行列への変換は有用なため残す）。

### 2-B. floor_divide の unit 意味論が未ドキュメント

`metadata.py:38`:
```python
_UFUNC_MULT_DIV = {np.multiply, np.divide, np.floor_divide}
```

`floor_divide` は整数割り算だが、unit は `lhs_unit / rhs_unit` として伝播される。
これは物理的に正しい（次元は同じ）が、docstring に記載がない。
**修正方針:** `_UFUNC_MULT_DIV` の意味論を docstring に追記。コード変更は不要。

### 2-C. CSV ラウンドトリップのテストが存在しない

`MetaDataMatrix.write()` / `read()` および `MetaDataDict.write()` / `read()` のテストがない。
実装は `MetaData` コンストラクタが string unit を受け付けるため動作するはずだが、
`MetaDataDict.to_dataframe()` が astropy `Unit` オブジェクトをそのまま DataFrame に入れるため、
`to_csv()` 時に `str(unit)` が呼ばれる挙動に依存している。
**修正方針:** `to_dataframe()` 内で `unit` を明示的に `str()` 変換してから格納し、round-trip テストを追加。

### 2-D. MetaDataDict には names/units/channels セッターが存在しない

現在は getter のみ。Issue の "setters" チェックは主に `MetaDataMatrix` を指しているが、
`MetaDataDict` についても同様のセッターが有用である可能性を確認する。
**修正方針:** 今回は `MetaDataMatrix` セッターの修正に集中。`MetaDataDict` へのセッター追加は scope 外とし、
issue コメントで明示する。

### 2-E. `as_meta()` が `MetaData` 以外の入力でチャンネルを常に `self.channel` から引き継ぐ

`metadata.py:165-173`:
```python
def as_meta(self, obj):
    if isinstance(obj, MetaData):
        return obj
    return MetaData(name=self.name, channel=self.channel, unit=get_unit(obj))
```

number や Quantity を渡した時、channel は `self.channel` を引き継ぐ。
これは現状テスト済みの挙動だが、仕様として docstring に明記されていない。
**修正方針:** docstring に "name and channel are inherited from self" と明記。

---

## 3. 詳細ロードマップ

### Phase 1: MetaDataMatrix セッター修正（`metadata.py`）

**対象:** `metadata.py:619-659`

変更内容:
```python
@names.setter
def names(self, value):
    value = np.asarray(value)
    if value.size != self.size:
        raise ValueError(
            f"Cannot set names: got {value.size} values, expected {self.size} "
            f"(shape {self.shape})"
        )
    if value.shape != self.shape:
        value = value.reshape(self.shape)
    for m, name in zip(self.reshape(-1), value.reshape(-1)):
        m.name = name
```

`units.setter` と `channels.setter` も同様に修正。

### Phase 2: MetaDataDict.to_dataframe() で unit を明示文字列化

**対象:** `metadata.py:410-415`

```python
def to_dataframe(self):
    if pd is None:
        raise ImportError("pandas is required for to_dataframe()")
    data = [
        {**{k: (str(v) if k == "unit" else v) for k, v in entry.items()}, "key": key}
        for key, entry in self.items()
    ]
    df = pd.DataFrame(data).set_index("key")
    return df
```

### Phase 3: テスト追加

**ファイル:** `tests/types/test_metadata_fixes.py` または新規 `tests/types/test_metadata_roundtrip.py`

追加するテスト:

#### MetaDataMatrix セッター shape mismatch
```python
def test_metadatamatrix_names_setter_size_mismatch_raises():
    mat = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="names"):
        mat.names = np.array(["a", "b"])  # size 2 != 6

def test_metadatamatrix_units_setter_size_mismatch_raises():
    mat = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="units"):
        mat.units = np.array([u.m, u.s])  # size 2 != 6

def test_metadatamatrix_channels_setter_size_mismatch_raises():
    mat = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="channels"):
        mat.channels = np.array(["H1:X"])  # size 1 != 6
```

#### MetaDataMatrix セッター flat → matrix reshape (許可ケース)
```python
def test_metadatamatrix_names_setter_flat_array_reshapes():
    mat = MetaDataMatrix(shape=(2, 3))
    mat.names = np.array(["a", "b", "c", "d", "e", "f"])
    assert mat.names[0, 0] == "a"
    assert mat.names[1, 2] == "f"
```

#### MetaDataMatrix CSV ラウンドトリップ
```python
def test_metadatamatrix_csv_roundtrip(tmp_path):
    mat = MetaDataMatrix([
        [MetaData(name="a", unit=u.m, channel="H1:A"),
         MetaData(name="b", unit=u.s, channel="L1:B")],
        [MetaData(name="c", unit=u.Hz, channel="H1:C"),
         MetaData(name="d", unit=u.dimensionless_unscaled, channel="")],
    ])
    path = tmp_path / "meta.csv"
    mat.write(str(path))
    mat2 = MetaDataMatrix.read(str(path))
    assert mat2.shape == (2, 2)
    assert mat2[0, 0].name == "a"
    assert mat2[0, 0].unit.is_equivalent(u.m)
    assert mat2[1, 1].name == "d"
```

#### MetaDataDict CSV ラウンドトリップ
```python
def test_metadatadict_csv_roundtrip(tmp_path):
    mdd = MetaDataDict({
        "x": MetaData(name="x", unit=u.m, channel="H1:X"),
        "y": MetaData(name="y", unit=u.s, channel="L1:Y"),
    })
    path = tmp_path / "meta.csv"
    mdd.write(str(path))
    mdd2 = MetaDataDict.read(str(path))
    assert list(mdd2.keys()) == ["x", "y"]
    assert mdd2["x"].unit.is_equivalent(u.m)
    assert mdd2["y"].name == "y"
```

#### floor_divide unit 伝播
```python
def test_metadata_floor_divide_unit():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.s)
    out = np.floor_divide(m1, m2)
    assert isinstance(out, MetaData)
    assert out.unit.is_equivalent(u.m / u.s)
```

#### MetaDataMatrix ufunc shape mismatch
```python
def test_metadatamatrix_ufunc_shape_mismatch_raises():
    m1 = MetaDataMatrix(shape=(2, 2))
    m2 = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="[Ss]hape"):
        np.multiply(m1, m2)
```

### Phase 4: docstring 更新

**対象:** `metadata.py` の以下の docstring:

1. `MetaData.__array_ufunc__` — 厳格 vs 柔軟の単位チェック方針を追記
2. `MetaDataMatrix.names.setter` / `units.setter` / `channels.setter` — サイズ一致要件を明記
3. `MetaData.as_meta()` — name/channel 引き継ぎ仕様を明記
4. `_UFUNC_MULT_DIV` コメント — `floor_divide` 含む旨と unit 意味論を注記

---

## 4. テスト・検証計画

### 実行コマンド
```bash
conda run -n gwexpy pytest tests/types/test_metadata_ufuncs.py tests/types/test_metadata_fixes.py -v
conda run -n gwexpy ruff check gwexpy/types/metadata.py tests/types/
conda run -n gwexpy mypy gwexpy/types/metadata.py
```

### 確認項目チェックリスト（Issue #243 対照）

| チェック項目 | 実装状況 | 対応 |
|---|---|---|
| ufunc unit propagation (unary: abs, neg, sqrt, square, transcendental) | ✅ 実装済み | テスト強化のみ |
| ufunc unit propagation (binary: add, sub requires compatible) | ✅ 実装済み | テスト確認のみ |
| ufunc unit propagation (multiply, divide, floor_divide) | ✅ 実装済み | floor_divide テスト追加 |
| 互換性なし unit で add/sub は UnitConversionError | ✅ 実装済み | テスト確認のみ |
| names/units/channels セッターが shape mismatch を隠さない | ❌ 修正必要 | Phase 1 で修正 |
| shape mismatch エラーが明示的 | ❌ 修正必要 | Phase 1 で修正 |
| DataFrame/CSV round-trip (name, channel, unit, shape) | ⚠️ 実装はあるがテストなし | Phase 2-3 で対応 |
| docs: strict vs flexible unit behavior | ❌ 未記載 | Phase 4 で追記 |

---

## 5. モデル・スキル・工数見積もり

| 項目 | 詳細 |
|---|---|
| 推奨スキル | `setup_plan` → `fix_errors` → `run_tests` → `finalize_work` |
| 推奨モデル | claude-sonnet-4-6 (コード変更は小規模で集中的) |
| 工数見積もり | Phase 1-2: ~30分 / Phase 3: ~45分 / Phase 4: ~20分 |
| 物理レビュー要否 | 不要（unit 意味論の変更なし、セッター動作の堅牢化のみ） |
| `needs-physics-review` ラベル | 不要 |

---

## 6. ファイル変更一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `gwexpy/types/metadata.py` | 修正 | セッター shape mismatch 明示化、`to_dataframe()` unit 文字列化、docstring 追記 |
| `tests/types/test_metadata_fixes.py` | 追加 | セッター shape mismatch テスト、CSV ラウンドトリップテスト、floor_divide テスト |

---

## 7. 関連情報

- Issue #243: https://github.com/tatsuki-washimi/gwexpy/issues/243
- PR #240, #241: Core Data Model audit 前段
- `gwexpy/types/metadata.py`: 主要実装
- `tests/types/test_metadata_ufuncs.py`: 既存 ufunc テスト
- `tests/types/test_metadata_fixes.py`: 既存 fixes テスト
