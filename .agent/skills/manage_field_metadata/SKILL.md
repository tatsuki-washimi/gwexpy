---
name: manage_field_metadata
description: 多次元フィールド（ScalarField等）の4D構造維持、ドメイン変換（時間・空間・周波数・波数）、および物理単位の整合性を管理する
---

# Manage Field Metadata

`gwexpy` における多次元フィールドデータ（`ScalarField` など）を扱う際、次元の欠落を防ぎ、物理的な整合性（ドメインと単位）を維持するための設計パターンを提供します。

## 1. 4次元構造の維持 (Metadata-Preserving Indexing)

スライス操作時、特定の次元が長さ1になっても軸を削除せず、常に4次元を維持することでメタデータの消失を防ぎます。

*   **実装パターン**:
    ```python
    def _force_4d_item(self, item):
        # 整数インデックスを slice(i, i+1) に変換して次元を維持
        new_item = list(item)
        for i, val in enumerate(new_item):
            if isinstance(val, int):
                new_item[i] = slice(val, val + 1)
        return tuple(new_item)
    ```

## 2. ドメイン変換と座標の更新

FFTやPSDによってドメインが変換される際（Time -> Frequency, Real -> K-space）、以下の4要素をセットで更新します。

1.  **データ値**: 変換アルゴリズムの適用。
2.  **軸座標 (Index)**: $1/(\Delta x)$ に基づく新しいサンプリング座標の生成。
3.  **軸名称 (Name)**: `t` -> `f`, `x` -> `kx` などのプレフィックス/名称変更。
4.  **ドメイン状態 (Domain State)**: `axis0_domain` や `space_domains` メタデータの更新。

## 3. 物理単位の伝播 (Spectral Unit Tracking)

変換後の単位を自動的に計算します。

*   **PSD (Density scaling)**: $[unit]^2 / [1/axis\_unit]$ (例: $V^2/Hz$)
*   **PSD (Spectrum scaling)**: $[unit]^2$
*   **Wavenumber**: $[axis\_unit]^{-1}$ (例: $1/m$)

## 4. 汎用軸処理のバリデーション

信号処理メソッド（`spectral_density` 等）は、対象軸が以下の条件を満たしているか確認する必要があります。

*   **等間隔性**: `AxisDescriptor.regular` であること。
*   **サイズ**: 変換に十分なデータ長があること（通常 2 以上）。
*   **現在のドメイン**: すでに変換済みでないか（例: 周波数ドメインに再び PSD をかけようとしていないか）。

## 知見の活用例
- `gwexpy/fields/scalar.py` の `spectral_density` メソッド
- `gwexpy/fields/signal.py` の `_validate_axis_for_spectral` 内部関数
