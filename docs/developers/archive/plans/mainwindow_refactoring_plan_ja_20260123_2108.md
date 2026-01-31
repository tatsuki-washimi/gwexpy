# MainWindowロジックのリファクタリングとモジュール化計画書 (2026-01-23 21:08:50)

**参照**: `docs/developers/plans/mainwindow_refactoring_plan_20260123.md`

## 1. 目的と目標
`gwexpy/gui/ui/main_window.py` 内の巨大な `MainWindow.update_graphs` メソッドをリファクタリングし、保守性と拡張性を向上させます。
*   **疎結合化**: 描画（Rendering）および信号注入（Excitation）ロジックを専用のクラスに分離します。
*   **抽象化**: メインループを変更することなく、多様な結果タイプ（Series, Spectrogram, SweptSine等）をサポート可能にします。
*   **DTT互換性**: 選択可能な正規化ロジック（標準 vs DTT互換）を導入します。
*   **保守性**: `main_window.py` のコード量を削減し、可読性を向上させます。

## 2. 詳細ロードマップ

### フェーズ 1: 共通信号ロジックの抽出
*   **目標**: 正規化およびENBW補正ロジックを実装。
*   **タスク**:
    *   `gwexpy/signal/normalization.py` の作成。
    *   `get_normalization_factor(window_name, n_fft, fs, mode='standard')` の実装。
    *   DTT分析結果に基づいた `detect_dtt_normalization_ratio(window)` の実装。

### フェーズ 2: Excitation Manager のモジュール化
*   **目標**: 信号生成と注入ロジックの抽出。
*   **タスク**:
    *   `gwexpy/gui/ui/excitation_manager.py` の作成。
    *   `ExcitationManager` クラスによる `GeneratorParams` の作成と `data_map` への信号注入。
    *   `main_window.py` の ~741-800行付近のロジックを移行。

### フェーズ 3: Plot Renderer のモジュール化
*   **目標**: あらゆるプロットタイプの描画ロジックを抽出。
*   **タスク**:
    *   `gwexpy/gui/ui/plot_renderer.py` の作成。
    *   `PlotRenderer` クラスの実装:
        *   `render_results(info_root, results)`: 高レベルエントリポイント。
        *   `render_trace(trace_item, result, config)`: 抽象ディスパッチャ。
        *   `_draw_series(...)`, `_draw_spectrogram(...)`: 低レベル描画。
    *   単位変換（dB, Phase, Magnitude）と座標変換（Log-Y）の処理。
    *   `main_window.py` の ~883-1143行付近のロジックを移行。

### フェーズ 4: MainWindow への統合
*   **目標**: 新しいマネージャーを使用するようにメインループをリファクタリング。
*   **タスク**:
    *   `MainWindow.__init__` で `ExcitationManager` と `PlotRenderer` を初期化。
    *   `MainWindow.update_graphs` を簡潔なパイプライン（`_collect_data_map`, `_check_stop_conditions`, `self.excitation_manager.inject`, `self.plot_renderer.render`）に書き換え。
    *   スレッド安全性とパフォーマンス（1サイクル20ms未満）を確保。

### フェーズ 5: 検証
*   **目標**: UI動作にデグレがないことを確認。
*   **タスク**:
    *   `pytest tests/gui` の実行。
    *   Spectrogramのログスケール描画の手動確認。
    *   DTT正規化切替の検証。

## 3. 使用モデルとリソース最適化

### 推奨モデル
*   **モデル**: `Claude Opus 4.5 (Thinking)`
*   **選定理由**: 1,800行を超える大規模な既存コードを正確に把握し、物理的な計算ロジック（正規化等）の整合性を保ちながら、複雑なクラス分割と抽象化を完遂するには、最高レベルの推論能力が必要であるため。

### リソース管理戦略
*   **現在の状況**: Claude Opus 4.5 (Thinking) のクオータが残り20%の状態。
*   **戦略**: 大規模なファイル編集を伴う本タスクではクオータを大量に消費するため、**翌日のクオータ回復を待ってから**作業に着手する。
*   **実行効率化**: 作業開始時には、この計画書と `MainWindow` の構造分析結果をコンテキストとして与え、バッチ的に実装を進めることで試行錯誤回数を減らす。

## 4. リスクと懸念事項
*   **パフォーマンス**: オブジェクト間通信のオーバーヘッド（Pythonでは無視できるレベルの想定）。
*   **状態の同期**: `PlotRenderer` が `MainWindow` ウィジェットの最新設定を常に参照できるようにする。
*   **正規化の不整合**: 正規化係数が二重に適用されたり、欠落したりしないよう厳密に管理する。
