# フェーズ1：信号正規化ロジック（DTT互換モード）の実装計画 (2026-01-23 21:16:30)

## 1. 目的
DTT (diaggui) と Scipy で異なる窓関数の正規化（ENBW）を統合的に管理し、ユーザーが用途に応じて選択できるようにします。これは MainWindow のリファクタリングにおける信号処理エンジンの基盤となります。

## 2. 実装内容

### モジュール: `gwexpy/signal/normalization.py`
以下の機能を実装する：
*   `get_enbw(window, fs, mode='standard')`: 
    *   `standard`: 標準的な $f_s \frac{\sum w^2}{(\sum w)^2}$。
    *   `dtt`: DTT定義の $f_s \frac{N}{(\sum w)^2}$。
*   `get_psd_normalization_factor(window, fs, mode='standard')`:
    *   PSD計算時に適用すべき係数を算出。
*   `convert_scipy_to_dtt(psd, window)`:
    *   `scipy.signal.welch` で得られたPSDをDTT互換の正規化に変換する便利関数。

### テスト: `tests/signal/test_normalization.py`
*   主要な窓関数（Hann, Flat-top, Boxcar）での理論値との比較。
*   DTT互換モードでの係数が期待通り（$f_s \frac{N}{(\sum w)^2}$）であることを確認。

## 3. 使用モデルとリソース
*   **モデル**: `Gemini 3 Flash`
*   **選定理由**: 独立した数学的モジュールの新規作成であり、クオータが100%残っているため迅速に実行可能。
*   **戦略**: 物理的な定義に基づいた純粋な関数として実装し、副作用を排除する。

## 4. 完了定義
*   `gwexpy/signal/normalization.py` が作成され、Linter (Ruff) をパスする。
*   テストが作成され、全項目パスする。
