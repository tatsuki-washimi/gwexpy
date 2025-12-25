# チャンネルの扱いについて (Channel Handling Explanation)

`gwexpy` の GUI (`pyaggui`) におけるチャンネルの扱いについて、コードベース調査および実装に基づき解説します。

## 1. 概要
`gwexpy` および `pyaggui` において、チャンネルは基本的に **文字列（チャンネル名）** として扱われます。データ構造としては `gwpy.timeseries.TimeSeries` 等のオブジェクトが使用され、その `name` 属性や辞書のキーとして管理されます。

## 2. GUI におけるチャンネル選択
`pyaggui` では、DTT (`diaggui`) の設計思想に基づき、チャンネル選択は「計測対象の定義 (Measurement)」と「表示対象の選択 (Results)」に分かれています。

*   **Measurement タブ**:
    *   **役割**: ここで **計測対象（データ取得対象）とするチャンネル** を選択します。
    *   多数のチャンネル（最大96個など）を登録・管理することができます。
    *   ここで `Active` にチェックを入れたチャンネルのみが、SIM/NDSモードでのデータ取得対象となります。

*   **Result タブ (GraphPanel)**:
    *   **役割**: Measurement タブで登録されたチャンネルの中から、**グラフに描画したいチャンネル (Traces)** を選択します。
    *   各トレースには `A` (チャンネルA) と `B` (チャンネルB、伝達関数などの2チャンネル測定用) の入力欄があります。
    *   コンボボックスの選択肢には、**Measurement タブで Active になっているチャンネルのみ** が表示されます（即時連動）。

## 3. データソース別の処理フロー

### A. オンライン (NDS) - ※動作未検証 / リスト取得未実装
1.  **Measurement タブ** で計測したいチャンネル名を入力し、`Active` にします。
    *   現状、サーバーからチャンネル一覧を取得して選択肢にする機能はないため、手動入力が必要です。
2.  **Result タブ** のコンボボックスには、Measurement タブで Active なチャンネルのみが選択肢として現れます。
3.  「Start」ボタンを押すと、`start_animation` が **Measurement タブで Active な全チャンネル** に対してデータ取得要求 (`NDSDataCache`) を行います。

### B. ファイル (FILE) - XMLファイル
`diaggui` との互換性を重視し、LIGO light weight XML (`.xml`) ファイルについては以下の挙動となります。

1.  「Open...」で XML ファイルを読み込みます。
2.  ファイル内のパラメータ情報から、チャンネルリストおよび各チャンネルの **Active 状態** が読み取られます。
3.  これらが **Measurement タブ** に即座に反映（復元）されます。
4.  Measurement タブの更新に伴い、**Results タブ** のコンボボックスの選択肢も自動的に更新されます。
    *   つまり、ファイルに保存されていた「計測設定」が再現され、そこから選んでグラフを表示するというフローになります。

### C. ファイル (FILE) - その他 (GWF, miniseed, HDF5等)
**申し送り事項 (Future Work)**:
`.gwf`, `.miniseed`, `.h5` 等の形式については、現状読み込みは可能（レガシー挙動）ですが、MeasurementタブへのActive状態反映などのDTT互換機能は**未対応**です。
*   現状は、ファイルに含まれる全チャンネルが直接 Results タブの候補になる、あるいは Measurement タブを経由しない挙動となる場合があります。
*   将来的には、これらの形式も Measurement タブをマスターとする仕様へ統一することが望まれます。

### D. シミュレーション (SIM)
1.  **Measurement タブ** でチャンネル（例: `white_noise`）を Active にします。
2.  **Results タブ** の選択肢が連動して更新されます。
3.  「Start」ボタンを押すと、内部の `Engine` クラスが、Measurement タブで Active になっているチャンネルのデータのみを生成します。

## 4. DTT (diaggui) との整合性
2025/12/25 の改修により、`pyaggui` のチャンネルハンドリングは `diaggui` の設計思想（Measurement = Master / Results = View）に整合するようになりました。

*   **改善点**:
    *   以前は Results タブで直接チャンネル名を入力・選択していましたが、Measurement タブでの定義が必須（または優先）となるフローに変更されました。
    *   特に XML ファイル読み込み時に `Active` フラグも含めて復元されるようになったことで、DTT で保存した計測設定を忠実に再現可能となりました。

## 5. 内部実装のポイントと今後の課題
*   **`gwexpy/gui/ui/tabs.py`**: `Measurement` タブのチャンネル状態（名前、Activeフラグ）を管理するモデル (`channel_states`) を保持し、変更シグナル (`measure_callback`) を発行します。
*   **`gwexpy/gui/ui/main_window.py`**:
    *   `on_measurement_channel_changed`: Measurement タブからの変更通知を受け取り、Results タブのコンボボックスを更新します。
    *   XML読み込み時に `loaders.extract_xml_channels` を使用して Measurement タブの状態をプログラムから更新 (`set_all_channels`) します。
*   **技術的負債 / リファクタリング**:
    *   現在、XMLのパラメータ解析（Active状態読み取り）ロジックは `gwexpy/gui/io/loaders.py` 内に実装されています。
    *   本来このロジックは **`gwexpy/io/dttxml_common.py`** に実装すべきものです。
    *   **今後のタスク**: `loaders.py` 内の `extract_xml_channels` 相当の機能を `gwexpy/io/dttxml_common.py` に移動・統合し、`loaders.py` からはそれを呼び出すように修正する必要があります。
