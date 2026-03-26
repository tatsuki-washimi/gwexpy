# ScalarField 信号処理拡張 (Refined Phase 2 & 3) 実装計画

**タイムスタンプ**: 2026-01-23 09:00 JST

## 目的・目標
`ScalarField` に対し、一般的かつ汎用的な周波数・波数ドメイン変換（Spectral Density推定）と、参照点を用いた相関解析機能を実装します。
内部実装ではメモリと速度を優先しつつ、ユーザーには `TimeSeries.psd()` のような直感的な API を提供します。

## 詳細なロードマップ（Phase単位）

### Phase 1: ノイズ生成・デモデータ (完了)
- [x] `gwexpy/noise/field.py` 実装。
- [x] `ScalarField.simulate()` クラスメソッド実装。
- [x] `scripts/verify_scalarfield_noise.py` による検証。

### Phase 2: 汎用的 Spectral Density 推定 (完了)
- [x] **汎用コア実装**: `compute_spectral_density(field, axis, method='welch', ...)`
    - 指定した軸（時間 0 or 空間 1-3）に対してスペクトル密度を計算する汎用ロジック。
    - メモリ効率を意識した内部実装。
- [x] **ユーザーAPI (Aliases)**:
    - `ScalarField.psd()`: `axis=0` (時間) 固定のエイリアス。
    - `ScalarField.csd()` (Cross Spectral Density): 参照点との比較用。
- [x] **全点変換のデフォルト化**:
    - まずは全空間点に対する変換を実装し、4D構造を維持した `ScalarField` を返す。
    - メモリ節約オプション（特定の空間ラインのみ変換など）は後続ステップで検討。

### Phase 3: 相関解析と参照点マッピング (完了)
- [x] **Coherence/Correlation with Ref Point**:
    - `field.coherence(ref_point, ...)`: 参照点と全空間点のコヒーレンスを計算し、空間 3D/2D マップ（`ScalarField`）として返す。
    - `field.xcorr(ref_point, ...)`: 同様に相互相関。
- [x] **帯域平均化**:
    - 指定した周波数帯域で平均した「コヒーレンスマップ（スカラーマップ）」を生成する機能。

### Phase 4: 検証とドキュメント (一部完了)
- [x] 物理整合性チェック（単位、Parsevalの定理、波数定義）。
- [x] pytest による自動検証。
- [ ] チュートリアルノートブックの作成。

## テスト・検証計画
1.  **汎用性の検証**: 時間軸だけでなく、空間軸（x, y, z）に対しても正しく $1/\text{m}$ (波数) への変換が行われるか。
2.  **既存APIとの比較**: `TimeSeries.psd()` と結果が一致するか。

## 使用モデル・推奨スキル・工数見積もり

- **推奨モデル**:
    - `Claude Opus 4.5 (Thinking)` (メイン実装・物理検証)
    - `GPT-5.2` (テスト・高速リファクタリング)

- **推奨スキル**:
    - `check_physics`
    - `estimate_effort`

- **工数見積もり**:
    - 合計: **100分**
    - Quota: **High**
