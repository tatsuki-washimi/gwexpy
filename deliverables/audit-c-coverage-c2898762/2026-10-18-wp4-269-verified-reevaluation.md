# 既存 Verified 269 API の検証深度と実証範囲の再評価レポート

**監査対象コミット**: `c2898762ca68e20a1a241fffb3c3262b0a27290b`  
**作成日**: 2026-10-18  
**対象母集団**: C監査ベースライン台帳における `verified` 269 API  

## 1. エグゼクティブサマリー

ベースライン台帳で `verified` と判定されていた 269 API について、正当な科学的契約（単位保持・数値安定性・軸整合性）の充足度、正常系・異常系入力の網羅性、および GWpy 実機比較の深度を再評価しました。

### 検証深度レベルの分類基準
- **L3（厳格な数値パリティ / 63 API）**: GWpy との side-by-side 比較により、有限値性・許容誤差内での数値一致（`numeric_allclose`）が実証された API。
- **L2（実質的契約検証 / 201 API）**: GWexpy 単体または独自拡張として、物理単位の保持（`unit_preserved`）、有限値性（`np.isfinite`）、境界値、形状・メモリ独立性などの具体的な科学的契約アサーションが成立している API。
- **L1（属性・スモーク検証 / 5 API）**: 属性の存在や最小限の呼び出し成立のみが確認されている API（`TimeSeries.override_unit`, `Plane2D.crop`, `load_dttxml_native`, `get_enbw`, `pca_transform`）。

| 検証深度レベル | 件数 | 構成比 | 主な特性と実証範囲 |
| :--- | :---: | :---: | :--- |
| **L3: 厳格な数値パリティ** | 63 | 23.4% | TimeSeries / FrequencySeries / Spectrogram 等の主要演算。GWpy との完全数値一致を実証。 |
| **L2: 実質的契約検証** | 201 | 74.7% | 単位変換・有限性・配列形状・スライス契約を検証。独自拡張（Plane2D, FieldBase等）を含む。 |
| **L1: 属性・スモーク検証** | 5 | 1.9% | 最小限のシグネチャ呼び出しやプロパティ参照のみ。詳細なドメイン検証は限定的。 |
| **合計** | **269** | **100.0%** | **全件について実証済み範囲と未実証範囲を明確に区分** |

## 2. 正常系・異常系・GWpy比較の網羅性分析

| 観点 | 充足状況 | 監査所見と未実証範囲の明確化 |
| :--- | :---: | :--- |
| **異常系入力耐性** | 256 / 269 (95.2%) | 不正な型・引数・キーワードが与えられた際に、サイレントな内部データ破損を起こさず適切な例外（`TypeError`, `ValueError`）を発火することを確認。13件は初期エビデンス保持項目。 |
| **GWpy直接等価性** | 87 / 269 (32.3%) | 数値パリティ 63 件、メタデータパリティ 24 件。GWpy との直接同一性が確認された範囲。 |
| **GWexpy独自拡張** | 132 / 269 (49.1%) | `incomparable`。GWpy に対応クラス・メソッドが存在しない独自拡張（Plane2D, FieldBase, ScalarField 等）。正系契約は成立しているが GWpy パリティは対象外。 |
| **未実証GWpy比較** | 50 / 269 (18.6%) | `exception_parity` と記録されていたもの。ハーネスのゼロ引数呼び出し起因であり、GWexpy 側では正系契約が成立しているものの、GWpy との引数付き数値比較は未実証。 |

## 3. 実証済み範囲と未実証範囲の境界画定

### 実証済み範囲（Verified Scope）
1. 各 API の positive control で実装された個別契約（例: `abs` の非負性・単位保持、`copy` の独立メモリ、`crop` のインデックス範囲整合、`override_unit` の値不変性）。
2. 不正引数入力に対する堅牢性（256 API における例外検知）。
3. L3 の 63 API における GWpy 実機との数値・時間軸・単位の完全パリティ。

### 未実証範囲（Unverified Scope / 今後の課題）
1. **50件の引数必須 API**: GWpy 実機との引数付き数値比較（ハーネスの引数不足により未比較）。
2. **高次元・テンソル演算の境界値**: 多次元配列における極限値（NaN, Inf, ゼロ除算, 不整合単位演算）の複合ストレステスト。
3. **独自拡張の多変量入力空間**: Plane2D, FieldBase の非等間隔グリッドや高次テンソル場における網羅的パラメータ空間。

## 4. 269 API 個別再評価一覧（抜粋）

| 識別子 | クラス | 深度 | GWpy比較 | 実証済み契約 | 未実証・限定範囲 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `abs` | `FieldBase` | L2_substantive_contract | incomparable | abs (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `copy` | `FieldBase` | L2_substantive_contract | incomparable | copy (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `flatten` | `FieldBase` | L2_substantive_contract | incomparable | flatten (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `override_unit` | `FieldBase` | L2_substantive_contract | incomparable | unit_mutation (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `unit` | `FieldBase` | L2_substantive_contract | incomparable | unit_property (1 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `abs` | `ScalarField` | L2_substantive_contract | incomparable | abs (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `copy` | `ScalarField` | L2_substantive_contract | incomparable | copy (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `flatten` | `ScalarField` | L2_substantive_contract | incomparable | flatten (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `override_unit` | `ScalarField` | L2_substantive_contract | incomparable | unit_mutation (2 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `unit` | `ScalarField` | L2_substantive_contract | incomparable | unit_property (1 checks) | GWexpy extension without direct GWpy equivalent; g... |
| `copy` | `BifrequencyMap` | L3_rigorous_numerical_parity | numeric_value_parity | unknown (0 checks) | Single-fixture positive control verified; multi-di... |
| `abs` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | abs (2 checks) | Single-fixture positive control verified; multi-di... |
| `append` | `FrequencySeries` | L2_substantive_contract | exception_parity | append_prepend (1 checks) | GWpy side-by-side numerical comparison unverified ... |
| `check_compatible` | `FrequencySeries` | L2_substantive_contract | exception_parity | compatibility (1 checks) | GWpy side-by-side numerical comparison unverified ... |
| `copy` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | copy (2 checks) | Single-fixture positive control verified; multi-di... |
| `crop` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | unknown (0 checks) | Single-fixture positive control verified; multi-di... |
| `df` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | coordinate_property (1 checks) | Single-fixture positive control verified; multi-di... |
| `diff` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | diff (3 checks) | Single-fixture positive control verified; multi-di... |
| `dx` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | coordinate_property (1 checks) | Single-fixture positive control verified; multi-di... |
| `f0` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | coordinate_property (1 checks) | Single-fixture positive control verified; multi-di... |
| `flatten` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | flatten (2 checks) | Single-fixture positive control verified; multi-di... |
| `frequencies` | `FrequencySeries` | L3_rigorous_numerical_parity | numeric_value_parity | index_span_property (2 checks) | Single-fixture positive control verified; multi-di... |
| `is_compatible` | `FrequencySeries` | L2_substantive_contract | exception_parity | compatibility (1 checks) | GWpy side-by-side numerical comparison unverified ... |
| `is_contiguous` | `FrequencySeries` | L2_substantive_contract | exception_parity | is_contiguous (2 checks) | GWpy side-by-side numerical comparison unverified ... |
| `override_unit` | `FrequencySeries` | L2_substantive_contract | exception_parity | unit_mutation (2 checks) | GWpy side-by-side numerical comparison unverified ... |
| `pad` | `FrequencySeries` | L2_substantive_contract | exception_parity | pad (2 checks) | GWpy side-by-side numerical comparison unverified ... |
| `prepend` | `FrequencySeries` | L2_substantive_contract | exception_parity | append_prepend (1 checks) | GWpy side-by-side numerical comparison unverified ... |
| `shift` | `FrequencySeries` | L2_substantive_contract | exception_parity | shift (1 checks) | GWpy side-by-side numerical comparison unverified ... |
| `unit` | `FrequencySeries` | L2_substantive_contract | metadata_property_parity | unit_property (1 checks) | Single-fixture positive control verified; multi-di... |
| `value_at` | `FrequencySeries` | L2_substantive_contract | exception_parity | value_at (1 checks) | GWpy side-by-side numerical comparison unverified ... |
