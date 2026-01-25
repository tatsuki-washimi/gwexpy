# ScalarField 信号処理拡張 計画書 (2026-01-23 09:05:00)

## 目的・目標
`ScalarField` に対し、一般的かつ汎用的な周波数・波数ドメイン変換（Spectral Density推定）と、参照点を用いた相関解析機能を実装します。
内部実装ではメモリと速度を優先しつつ、ユーザーには `TimeSeries.psd()` のような直感的な API を提供します。

## 詳細なロードマップ（Phase単位）

### Phase 1: ノイズ生成・デモデータ (完了)
- 状況: `gwexpy/noise/field.py` および `ScalarField.simulate()` 実装済み。スクリプト検証済み。

### Phase 2: 汎用的 Spectral Density 推定 (40分)
- **汎用コア実装**: `compute_spectral_density(field, axis, method='welch', ...)`
    - 指定した軸（時間 0 または空間 1-3）に対してスペクトル密度を計算。
- **ユーザーAPI**:
    - `psd()` (時間軸エイリアス)、`csd()` など。
- **全点変換**: 4D形状を維持しつつ、該当軸をドメイン変換した `ScalarField` を生成。

### Phase 3: 相関解析と参照点マッピング (40分)
- **コヒーレンス/相関マップ**:
    - `field.coherence(ref_point, ...)` など、参照点と全空間点の統計的関係を 3D/2D マップ化。
- **帯域平均**: 特定帯域の積分値（スカラー）マッピング対応。

### Phase 4: 検証とドキュメント (20分)
- 物理単位、Parsevalの定理、波数定義の整合チェック。
- ノートブック形式のチュートリアル作成。

## 使用モデルとリソース最適化

- **推奨モデル**:
    - **Claude Opus 4.5 (Thinking)**: コアロジックの実装、物理数学的整合性（FFT正規化、単位変換）の検証。
    - **GPT-5.2-Codex**: 大規模なテストケース生成、高速なリファクタリング。
    - **Gemini 3 Flash**: ドキュメント作成、ボイラープレートコード生成。

- **リソース管理戦略**:
    - 数学的検証には `check_physics` スキルを活用し、モデルの推論を補強する。
    - メモリ効率を優先するため、大規模データの処理には `numpy` の軸指定演算を最大限活用し、ループ処理を避ける。

## テスト・検証計画
- 121件の既存テストを維持しつつ、`tests/fields/test_scalarfield_signal.py` を新設。
- 時間軸/空間軸の両方でのスペクトル密度計算の正確性を、既知の解析解（正弦波、ホワイトノイズ）と比較。

## 工数見積もり
- **Estimated Total Time**: 100 minutes
- **Estimated Quota Consumption**: High
- **Breakdown**:
    - Phase 2: 40 mins (High Difficulty)
    - Phase 3: 40 mins (High Difficulty)
    - Phase 4: 20 mins (Medium Difficulty)
