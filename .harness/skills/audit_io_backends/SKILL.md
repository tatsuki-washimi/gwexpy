---
name: audit_io_backends
description: I/O backend（GWF/HDF5/Zarr/FrameL 等）の追加・変更時、フォーマット互換性・feature parity の監査時に使用する。リリース前 I/O 回帰チェックにも適用。
---

# Audit I/O Backends

I/O backend の追加・変更・リリース前 I/O 回帰を体系的に検証するスキル。

## When to Use

- GWF / HDF5 / Zarr / FrameL など I/O backend を新規追加または修正するとき
- フォーマット互換性・feature parity（read/write 対称性）を確認するとき
- リリース前の I/O 回帰監査（`public_io_contract.json` との整合確認）
- 過去の典型事例:
  - Issue #384: FrameL/GWF backend の多重フォーマット対応
  - Issue #356: File I/O hotfix（registry smoke で早期検出）

## 手順

### 1. Registry smoke — backend 登録の確認

```bash
conda run -n gwexpy python -c "import gwexpy; gwexpy.register_all()"
```

`ImportError` や登録漏れがあればここで即座に検出される。

### 2. Contract gate — public API 整合確認

```bash
python scripts/ci/run_gate.py io-contract
```

`tests/io/test_io_contract.py` と `tests/io/test_io_docs_contract_sync.py`、
`tests/segments/`、`tests/table/` を一括実行し、ホイールビルドの bytecode 混入も検査する。

### 3. Conformance gate — schema v3 整合確認

```bash
python scripts/ci/run_gate.py io-conformance
```

`tests/io_conformance/` 配下（contract schema v3・ write roundtrip・optional dep 動作など）を実行する。

### 4. Optional deps gate — h5py / zarr / framel 等

```bash
python scripts/ci/run_gate.py io-optional
```

`test_optional_deps.py`、`test_netcdf4_reader.py`、`test_tdms_reader.py`、
`test_audio_metadata.py`、`test_seismic_public_io.py` を実行する。

### 5. Zarr 固有 gate

```bash
python scripts/ci/run_gate.py io-zarr
```

`GWEXPY_ALLOW_ZARR=1` を設定して `test_zarr_reader.py` を実行する。

### 6. ネットワーク backend（必要時のみ）

ネットワーク依存 backend（NDS2 / Kerberos 等）を含む場合のみ実行する:

```bash
python scripts/ci/run_gate.py io-network-backend
```

`tests/io/`、`tests/nds/`、`tests/segments/` 等の `network` / `nds` マークを実行する。

### 7. 直接 pytest — ピンポイント調査

特定ファイルやサブ範囲を絞り込む場合:

```bash
conda run -n gwexpy pytest tests/io/ tests/io_conformance/
conda run -n gwexpy pytest tests/io/test_gwf.py -v
```

## Contract 照合

`docs/developers/contracts/public_io_contract.md` に記述された公開 API と
実装のずれは次のテストで検証する:

```bash
conda run -n gwexpy pytest tests/io/test_io_docs_contract_sync.py
```

このテストは `io-contract` gate にも含まれるため、gate 実行後に差分が出た場合の
ピンポイント再実行として使う。

## Out of Scope

- `docs-notebook` gate: docs/ノートブック同期 workflow が担当する
- `interop-contract` gate: interop workflow（GWpy 互換層）が担当する

これらは本スキルのスコープ外であり、本 skill からは呼び出さない。
