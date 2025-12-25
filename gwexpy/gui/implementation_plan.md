# 実装計画 - NDSオンラインデータアクセス

`gwexpy/gui/` にNDS2オンラインデータ取得機能を実装します。
これは `ndscope`（`reference_ndscope/` にて参照可能）から関連部分を新しい `nds/` モジュールに移植し、PyQt GUIのフリーズを引き起こすことなく統合することを伴います。

## ユーザーレビュー必須事項

> [!IMPORTANT]
> - **フェーズ1** は **RAWデータのみ** および基本的なリングバッファに焦点を当てます。ズーム連動バックフィル（catch-up）やトレンド切り替えなどの高度な機能はフェーズ2に後回しとなります。
> - 指示に従い、認証機能（Kerberos/SASL）は省略します。
> - 新しい `nds/` モジュールは、実行時に `reference_ndscope` に依存しません。

## 変更案

### 1. 新規モジュール: `gwexpy/gui/nds/`

移植ロジックを含む新しいパッケージ `gwexpy/gui/nds/` を作成します。

#### [NEW] `gwexpy/gui/nds/__init__.py`
パッケージ化のための空ファイル。

#### [NEW] `gwexpy/gui/nds/util.py`
`reference_ndscope/util.py` からサーバー文字列解析ロジックを移植します。
- 関数: `parse_server_string(server: str) -> tuple[str, int]`
- **追加**: `gps_now()` を定義し、`gpstime` 利用不可時のフォールバック方針（システム時間等）を実装します。

#### [NEW] `gwexpy/gui/nds/buffer.py`
`reference_ndscope/data.py` から `DataBuffer` と `DataBufferDict` を移植します。
- `lookback` 制限付きのリングバッファを管理します。
- プロット用の `update_tarray()` およびデータアクセスを提供します。

#### [NEW] `gwexpy/gui/nds/nds_thread.py`
`reference_ndscope/nds.py` から `NDSThread` を移植します。
- **統一**: Qtバインディングは `qtpy` に統一して移植し、既存UIとはシグナル境界で連携します。
- `nds2-client` を使用してデータを取得します。
- `nds2.iterate` を使用した `online` コマンドを実装します。
- **認証ロジックなし。**

#### [NEW] `gwexpy/gui/nds/cache.py`
`reference_ndscope/cache.py` から `DataCache` を `NDSDataCache` として移植します。
- `NDSThread` を管理します。
- API:
    - `set_channels(channels: list)`
    - `online_start(trend='raw', lookback=30)`
    - `online_stop()`: スレッド停止と参照破棄を確実に保証します。
    - `reset()`: 内部バッファとスレッド辞書（tid多重起動防止）をクリアします。
- シグナル: `signal_data` (新しいデータペイロードをGUIに送信)。

### 2. GUI統合

#### [MODIFY] `gwexpy/gui/ui/tabs.py`
- Inputタブのデータソース選択に "NDS" オプションを追加します。
- "NDS Window (sec)" 入力フィールドを追加します。
- **注**: これらの設定は "Input" タブに配置することを想定しています。

#### [MODIFY] `gwexpy/gui/ui/graph_panel.py`
- NDSチャンネル名を手動入力できるように、チャンネル選択コンボボックスを編集可能 (`setEditable(True)`) にします。

#### [MODIFY] `gwexpy/gui/ui/main_window.py`
- `NDSDataCache` を初期化します。
- シグナルを接続します (`on_nds_data`)。
- `data_source == 'NDS'` を処理するように `Start`/`Pause`/`Resume`/`Abort` ロジックを更新します。
    - Start: Set channels -> `online_start`
    - Pause: `online_stop`
    - Resume: `online_start`
    - Abort: `online_stop` -> `reset`
- `update_graphs()` を更新します:
    - NDSモードの場合、ダミーデータ生成を完全にスキップします。
    - `NDSDataCache` バッファから最新データを取得します。
    - **方針A（時系列のみ）を採用**: フェーズ1では計算エンジン（`engine.compute`）を通さず、時系列のみを直接プロットします。
    - 各 `PlotDataItem` 生成直後に `setClipToView(True)` および `setDownsampling(auto=True, method='peak')` を適用し、最適化します。

## 検証計画

### 自動テスト
- 新しい `nds` モジュールをインポートし、構文エラーがないことを確認するように `smoke_test.py` を更新します。
- (オプション) 実際のNDS接続なしでバッファロジックをテストするための `test_nds_structure.py` を作成します。

### 手動検証
1. **セットアップ**: `NDSSERVER` 環境変数を設定します。
2. **起動**: `python pyaggui.py`
3. **設定**:
    - Inputタブ: "NDS" を選択、Window=30s に設定。
    - Resultタブ: 有効なNDSチャンネルを手動入力 (例: `K1:PEM-ACC_SEIS_IX_Gnd_X_OUT_DQ`)。
4. **アクション**:
    - **Start** をクリック。データが表示され更新されることを確認。
    - **Pause** をクリック。更新が停止することを確認。
    - **Resume** をクリック。更新が再開することを確認。
    - **Abort** をクリック。プロットがクリア/リセットされることを確認。
5. **安定性**: NDS接続の問題（無効なチャンネルやサーバーでシミュレート）発生時にUIがフリーズしないことを確認します。
