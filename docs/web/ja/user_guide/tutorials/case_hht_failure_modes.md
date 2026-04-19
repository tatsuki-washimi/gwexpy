# ケーススタディ: HHT 解析で失敗しやすいパターン

HHT は非定常・非線形信号の追跡に強力ですが、パラメータと解釈を誤ると STFT より不安定に見えることがあります。ここでは :doc:`advanced_hht` を補完する形で、重力波・振動解析で頻出する失敗モードを整理します。

## このケーススタディで扱うこと

- ノイズの強い信号で EMD をそのまま使ってモード混合を起こす失敗
- 端点効果で瞬時周波数を誤読する失敗
- IMF を物理モードと 1 対 1 対応だと思い込む失敗
- STFT と HHT を比較するときの公平な見方

## 前提

- 基本 API と SASI 例は :doc:`advanced_hht` を参照
- 時間周波数表現の比較は :doc:`time_frequency_comparison` も参照

## 失敗例 1: 標準 EMD だけで弱いトラックを追う

**症状**: 130 Hz 付近の成分が複数 IMF に分散し、瞬時周波数が激しく振れる。  
**典型原因**: 雑音レベルが高いのに `emd_method="emd"` を固定し、モード混合をそのまま受けている。

```python
hht_plain = ts.hht(
    emd_method="emd",
    output="spectrogram",
)
```

まずは EEMD/CEEMD 系へ切り替え、結果の安定性を比較します。

```python
hht_eemd = ts.hht(
    emd_method="eemd",
    eemd_trials=20,
    output="spectrogram",
)
```

**確認ポイント**:

- 同じ信号トラックが複数 IMF に割れていないか
- `eemd_trials` を増やしても主トラックの位置が大きく変わらないか
- 振幅の大きい雑音バーストだけを拾っていないか

## 失敗例 2: 端点効果を物理変化だと誤認する

ヒルベルト変換は端で不安定になりやすく、解析区間の最初と最後で瞬時周波数が跳ねることがあります。

```python
hht_spec = ts.hht(
    emd_method="eemd",
    eemd_trials=20,
    hilbert_kwargs={"pad": 256},
    output="spectrogram",
)
```

**実務上の扱い**:

- 端の数百サンプルは解釈対象から外す
- パディングあり/なしで主トラック位置が変わるか比べる
- 端点付近だけに現れる急変は物理解釈しない

## 失敗例 3: IMF をそのまま物理モード名で呼ぶ

**症状**: `IMF 2 = SASI`, `IMF 3 = convection` のように即断する。  
**典型原因**: EMD の分解結果を観測器・シミュレーション文脈と照合せずに命名している。

IMF はアルゴリズム上の分解単位であり、物理モードの直接ラベルではありません。少なくとも以下を確認してください。

1. 周波数帯と時間発展が理論期待と整合するか
2. 近傍 IMF を足し戻しても同じトラックが出るか
3. STFT や wavelet でも同じイベント時刻が見えるか

## 失敗例 4: STFT と HHT を片方だけ有利な条件で比較する

HHT は時間分解能で有利に見える一方、前処理と雑音感度の違いで見え方が簡単に変わります。比較時は次を揃えます。

- 同じ前処理済み `TimeSeries` を入力する
- STFT の窓長・オーバーラップを明記する
- HHT 側は `emd_method`, `eemd_trials`, `hilbert_kwargs` を明記する

```python
ts_white = ts.whiten(fftlength=1.0, overlap=0.5)
stft = ts_white.spectrogram2(fftlength=0.05, overlap=0.045)
hht = ts_white.hht(emd_method="eemd", eemd_trials=20, output="spectrogram")
```

STFT で見えないから HHT が正しい、あるいは HHT が荒れるから STFT が正しい、とは言えません。両者の失敗モードが違うだけです。

## 推奨ワークフロー

1. `advanced_hht` の SASI 例で基準挙動を確認する
2. ノイズが強い場合は最初から EEMD/CEEMD を候補に入れる
3. 端点効果を見るため padding と無視区間を設定する
4. IMF 単体ではなく近傍 IMF の合算と STFT 比較で解釈を固める
5. 物理モード名を付ける前に理論期待と照合する

## 関連ページ

- :doc:`advanced_hht`
- :doc:`case_glitch_analysis`
- :doc:`time_frequency_analysis_comparison`
- :doc:`time_frequency_comparison`
