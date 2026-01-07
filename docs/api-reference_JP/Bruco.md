# Bruco (Brute force coherence) 

`Bruco` は、ターゲットチャンネル（重力波チャンネルなど）と多数の補助チャンネルとの間の**「総当たりコヒーレンス（Brute force coherence）」**を計算し、ノイズ源を特定するためのツールです。
指定された周波数帯域、時間帯において、ターゲットと高いコヒーレンスを持つ補助チャンネルをランキング形式で提示し、そのノイズ寄与（Noise Projection）を推定します。

オリジナルのBruco実装は以下で公開されています。設計詳細やCLIの挙動を確認したい場合はこちらを参照してください。
- https://github.com/mikelovskij/bruco


## 主な機能

1. **バッチ処理による高速スキャン**: 数千のチャンネルを効率的に処理するために、指定サイズごとのバッチでデータを取得・計算します。
2. **トップNランク保持**: 全帯域での最大コヒーレンスだけでなく、**周波数ビンごとに**コヒーレンスの高いチャンネルを上位N個保持します。
3. **並列処理**: マルチプロセス (`concurrent.futures`) を用いてコヒーレンス計算を高速化します。
4. **ノイズプロジェクション**: 補助チャンネルがターゲットチャンネルのノイズにどれだけ寄与しているかを推定し、スペクトルとして描画します。
5. **HTMLレポート**: 解析結果をまとめたHTMLレポートと画像を生成します。

---

## 使い方

### 1. 初期化

まず、解析対象のターゲットチャンネルと、スキャンしたい補助チャンネルのリストを指定して `Bruco` インスタンスを作成します。

```python
from gwexpy.analysis.bruco import Bruco

# ターゲットチャンネル名
target = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"

# 補助チャンネルリスト (通常はNDSなどから取得)
aux_channels = [
    "K1:PEM-ACC_MC_TABLE_SENS_Z_OUT_DQ",
    "K1:PEM-MIC_BS_BOOTH_SENS_Z_OUT_DQ",
    "K1:IMC-MCL_SERVO_SUM_OUT_DQ",
    # ... 他多数
]

# オプション: 除外したいチャンネル（キャリブレーション信号など）
excluded = ["K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ", "K1:GRD-LSC_LOCK_STATE_N"]

# インスタンス作成
bruco = Bruco(target, aux_channels, excluded_channels=excluded)
```

### 2. 解析の実行 (Compute)

データソースに応じて、以下の3つのパターン（Case）を利用できます。

#### Case 1: 自動データ取得 (スタンダード)
チャンネル名と時間を指定するだけで、自動的にデータ（NDS2やフレームファイル）を取得・解析します。

```python
# 解析設定
start_gps = 1234567890
duration = 64  # 秒

# 計算実行
result = bruco.compute(
    start=start_gps,
    duration=duration,
    batch_size=50,     # 一度に取得するチャンネル数
    nproc=4            # 並列プロセス数
)
```

#### Case 2: データの直接指定 (マニュアル)
外部で用意した `TimeSeries` オブジェクトを使って解析します。シミュレーションデータや、特殊な方法で取得したデータを使う場合に便利です。
データに含まれる時刻情報（`t0`, `duration`）を使用するため、`start`, `duration` 引数は省略可能です。

**A. 辞書で渡す (全データロード済)**
```python
aux_dict = TimeSeriesDict.read(..., channels=my_channels)
result = bruco.compute(
    aux_data=aux_dict  # 辞書を渡す (start, durationは自動推定)
)
```

**B. ジェネレータで渡す (メモリ効率重視)**
大量のチャンネルを扱う場合、ジェネレータを使ってデータを受け渡すことができます。
※ジェネレータ使用時は推定が難しいため `start`, `duration` の指定が推奨されますが、`target_data` がある場合はそこから推定されます。

```python
def data_stream(channels):
    for ch in channels:
        yield TimeSeries.get(ch, start, end)

result = bruco.compute(
    start, duration,                   # ジェネレータの場合は指定推奨
    aux_data=data_stream(my_channels), # ジェネレータを渡す
    batch_size=100                     # 100個溜まるごとに並列解析してメモリ解放
)
```

#### Case 3: 自動取得 + 前処理 (ハイブリッド)
データ取得は `Bruco` に任せつつ、解析前にフィルタリングやチャンネル間の演算（掛け算など）を行いたい場合、コールバック関数を利用できます。

```python
def my_preprocessing(batch_data: TimeSeriesDict) -> TimeSeriesDict:
    # バッチごとのデータを受け取り、加工して返す
    for ch, ts in batch_data.items():
        batch_data[ch] = ts.highpass(10)  # 例: 10Hzハイパスフィルター
    return batch_data

result = bruco.compute(
    start, duration,
    preprocess_batch=my_preprocessing  # コールバック指定
)
```
これにより、「自動取得の利便性」と「カスタム処理の柔軟性」を両立しつつ、並列処理の恩恵も受けられます。

#### Case 4: 混合モード (NDS + マニュアル)
`Bruco` 初期化時に指定したチャンネル（自動取得）と、`compute()` で渡す `aux_data`（マニュアル）を **同時に** 解析することも可能です。
両方のデータソースが順番に処理され、結果は統合されます。

```python
# 1. 自動取得したいチャンネルで初期化
bruco = Bruco(target, ["K1:NDS-CHANNEL-1", ...])

# 2. 手動データの辞書を作成
manual_dict = TimeSeriesDict(...) 

# 3. 両方を指定して実行
# 注意: manual_dict の時間は start/duration と一致している必要があります。
result = bruco.compute(
    start, duration,
    aux_data=manual_dict
)
```
**注意**: この場合、 `aux_data` に含まれるデータの時刻（t0, duration）が `start`, `duration` で指定した解析区間を完全にカバーしていない場合、`ValueError` が発生します。

### 3. 結果の表示と保存

`compute()` は `BrucoResult` オブジェクトを返します。このオブジェクトを使って結果を可視化したり、レポートを作成したりできます。

#### ステップ 3.1 コヒーレンスのプロット
各周波数で最もコヒーレンスが高かったチャンネルを色分けして表示します。

```python
fig_coh = result.plot_coherence()
fig_coh.show()
```

デフォルトでは、**寄与度の高い上位チャンネル（Top-K）**ごとのコヒーレンススペクトルが表示されます。
以前の「Rank（ランク）」ごとの表示を行いたい場合は、`ranks=[0, 1, ...]` を指定してください。

#### ステップ 3.2 ノイズプロジェクションのプロット
ターゲットのASDと、各補助チャンネルからの寄与（Noise Projection）を重ねて表示します。
`asd` 引数（ブール値）を指定することで、ASD（デフォルト）またはPSDで表示できます。

```python
# ASDで表示 (デフォルト, asd=True)
fig_proj = result.plot_projection()
fig_proj.show()

# PSDで表示
fig_proj_psd = result.plot_projection(asd=False)
fig_proj_psd.show()
```

#### ステップ 3.3 HTMLレポート作成
結果をまとめたディレクトリを作成し、HTMLレポートを出力します。

```python
# 'bruco_report' ディレクトリに出力
result.generate_report(output_dir="bruco_report")
```

---

## アーキテクチャと注意点

- **データの取得**: `TimeSeriesDict.get()` を使用してデータを取得します。バッチ内で一部のチャンネル取得に失敗した場合、自動的に個別取得モードに切り替えて有効なチャンネルのみを解析します。
- **リサンプリング**: ターゲットと補助チャンネルのサンプリングレートが異なる場合、遅い方のレートに合わせて自動的にダウンサンプリングされます。
- **内部はPSD基準**: 解析の内部表現は PSD で統一し、ASD 表示は描画時のみ変換します。`coherence_threshold` は `asd=True` のとき振幅コヒーレンス基準で適用されます。
- **メモリ管理**: 非常に多くのチャンネルを扱う場合、`batch_size` と `block_size` を調整してメモリ使用量をコントロールしてください。

### Top-N更新のブロックサイズ

`BrucoResult` の Top-N 更新は、チャンネルをブロック単位で処理します。  
ブロックサイズは以下の順で決まります。

1. `block_size` 引数（`int` または `"auto"`）
2. `GWEXPY_BRUCO_BLOCK_SIZE` 環境変数（`int` または `"auto"`）
3. デフォルト `256`

`"auto"` の場合は `GWEXPY_BRUCO_BLOCK_BYTES` を使って以下の式で推定します。

```
max_cols = (block_bytes // (n_bins * 8)) - top_n
block_size = clamp(max_cols, 16, 1024)
```

目標の `block_size` を決めたい場合は以下を目安に設定します。

```
block_bytes ~= (top_n + block_size) * n_bins * 8
```

例: `n_bins=20000`, `top_n=5`, `block_size=256` の場合

```bash
export GWEXPY_BRUCO_BLOCK_SIZE=auto
export GWEXPY_BRUCO_BLOCK_BYTES=41760000
```

### ベンチマーク

`scripts/bruco_bench.py` で `update_batch` の簡易ベンチが実行できます。

```bash
python scripts/bruco_bench.py --n-bins 20000 --n-channels 300 --top-n 5 --block-size auto
```

参考値 (環境依存):

```
elapsed_s=0.153
ru_maxrss_kb=627808
block_size_resolved=414
```

## API リファレンス

### `Bruco`

**`__init__(self, target_channel: str, aux_channels: List[str], excluded_channels: List[str] = None)`**
- `target_channel`: 解析対象のメインチャンネル名。
- `aux_channels`: 比較対象の補助チャンネル名のリスト。
- `excluded_channels`: 解析から除外するチャンネル名のリスト。

**`compute(self, start=None, duration=None, fftlength=2.0, overlap=1.0, nproc=4, batch_size=100, top_n=5, block_size=None, ...) -> BrucoResult`**
- `start`: GPS開始時刻。データ（`target_data` または `aux_data`辞書）から推定できる場合は省略可能。
- `duration`: 解析データの長さ（秒）。推定できる場合は省略可能。
- `fftlength`: スペクトル計算のFFT長（秒）。
- `overlap`: オーバーラップ長（秒）。
- `nproc`: 並列計算に使用するプロセス数。
- `batch_size`: 一度にデータを取得するチャンネル数。
- `top_n`: 各周波数ビンで保持する上位チャンネル数。
- `block_size`: Top-N 更新のブロックサイズ（`int` または `"auto"`）。
- `target_data`: (`TimeSeries`) 事前に取得したターゲットデータ。
- `aux_data`: (`TimeSeriesDict` or `Iterable`) 事前に取得した補助チャンネルデータ。
- `preprocess_batch`: (`Callable`) バッチ前処理用コールバック関数。

### `BrucoResult`

**`plot_coherence(self, asd=True, coherence_threshold=0.0, channels=None, ranks=None)`**
- コヒーレンススペクトルをプロットします。
- **デフォルト動作**: `channels` も `ranks` も指定しない場合、寄与度の高い上位チャンネルを自動選択して描画します。
- `channels`: 特定のチャンネル名を指定して描画します。
- `ranks`: 特定のランク（0=最大）を指定して描画します（旧動作）。
- `asd=True`: 振幅コヒーレンス ($\sqrt{C_{xy}^2}$) を表示。
- `asd=False`: 二乗コヒーレンス ($C_{xy}^2$) を表示。
- `coherence_threshold`: 指定した値に閾値ラインを表示します（`asd` 設定に合わせて自動変換されます）。

**`plot_projection(self, asd=True, coherence_threshold=0.0, channels=None, ranks=None)`**
- ターゲットASDとノイズプロジェクションをプロットします。
- **デフォルト動作**: `channels` も `ranks` も指定しない場合、寄与度の高い上位チャンネルを自動選択して描画します。
- `channels`: 特定のチャンネル名を指定して描画します。
- `ranks`: 特定のランクを指定して描画します（旧動作）。
- `asd=True`: ASD (振幅スペクトル密度) で表示。 `False` で PSD。
- `coherence_threshold`: この値以下のコヒーレンスを持つ周波数の寄与を `NaN` としてマスクします（プロット上で途切れます）。

**`generate_report(self, output_dir="bruco_report", asd=True)`**
- 指定したディレクトリにレポート（HTML, PNG, CSV）を生成します。
