# タイムスタンプ
- 2026-01-22 18:05 JST（update）

# 目的・目標
- `gwexpy.fields` に Scalar/Vector/Tensor Field を新設し、明示的な時間/周波数・位置/波数ドメインメタデータを軸ごとに保持する統一 Field API を実装する。
- 形状不変性（軸削除なし、2D/3D は長さ1軸で表現）と部分FFT時のドメイン伝播を保証し、物理単位とドメインの整合性を検証可能にする。
- ScalarField 時代の「4D」表記をユーザAPIから排除しつつ、Vector/Tensor 拡張を前提にコンポーネント軸を常に終端に保持する設計を定着させる。

# 詳細なロードマップ（Phase単位）
- **Phase 1: 構造リファクタ**
  - `gwexpy/fields/` パッケージを新設し、旧4Dフィールドクラスを ScalarField として移動/整理。ScalarFieldList/Dict → FieldList/FieldDict。公開 import を `gwexpy.fields` に統一。
  - 既存 `types/array4d.py` 等の低レベルは温存し、内部参照を更新。
  - CI 影響範囲のモジュール import 修正。
- **Phase 2: ベースクラスとドメインメタデータ**
  - `FieldBase` を追加し、`axis0_domain ∈ {time, frequency}` と `space_domain ∈ {position, wavenumber}^3` を保持。単位とドメインの整合チェック（time↔s, frequency↔Hz, position↔length, wavenumber↔1/length）を実装。
  - スライスで軸・メタデータを保持する挙動を固定化（長さ1 Quantity を維持）。
- **Phase 3: FFTとドメイン伝播**
  - 時間軸FFTで `time→frequency`、空間軸FFTで該当軸を `position→wavenumber` に更新。部分FFT対応。`fftshift` 等のシフト処理はドメインと独立に扱う。
  - ドメイン更新と単位の組み合わせをテストで網羅し、非変換軸のドメイン/単位が保持されることを確認。
- **Phase 4: ScalarField向け信号処理**
  - PSD(Welch), 相互相関, coherence, 時間遅延マップを ScalarField 用に移植/実装。Vector/Tensor は component/magnitude 経由でスカラー化してから利用。
- **Phase 5: プロットとドキュメント**
  - ドメイン対応ラベルのプロットヘルパーを ScalarField に提供。
  - `docs/examples`/`docs` に部分FFT例、ドメイン不変性、スライス挙動を追加。docstring も同期。

# テスト・検証計画
- `pytest` でドメイン伝播・部分FFT・スライス（軸保持/Quantity維持）・単位整合のケースを追加。
- `ruff` と `mypy` を全体実行し、フィールド新規APIに対する型整合を確認。
- 物理整合チェック: wavenumber 定義と単位逆数を `check_physics` スキルで確認予定。
- スライスによる軸長=1維持（例: z=0, kx=0）とメタデータ保持を明示検証。
- 2D/3D 退化ケース（長さ1軸含む）の FFT で軸数が変化しないことを確認。

# 使用モデル・推奨スキル・工数見積もり
- **推奨モデル**: コード中心かつ物理メタデータ厳格さが必要なため `GPT-5.2-Codex`（実装と型整合）、複雑な物理/FFT仕様検証には `Claude Opus 4.5 (Thinking)` をバックアップ。軽量レビュー・小修正には `Gemini 3 Flash`。
- **推奨スキル**: `lint`（ruff/mypy実行）、`check_physics`（ドメインと単位検証）、`sync_docs`（docstring/docs反映）、`wrap_up_gwexpy`（最終整備）、必要に応じ `visualize_fields`（描画確認）。
- **工数見積もり** (`estimate_effort` 規約):
  - Phase1-2: 35-45分（リファクタ＋メタデータ実装/単位チェック）
  - Phase3: 20-30分（FFTドメイン伝播とテスト）
  - Phase4: 20分（信号処理移植）
  - Phase5: 15分（プロット/ドキュメント）
  - 合計目安: 90-110分、クオータ消費は Medium-High（型/テスト/ドキュメント更新と複数ツール呼び出しのため）。

# 承認待ち
- 上記計画とモデル選択方針に問題なければ承認ください。承認後 Phase 1 から着手します。
