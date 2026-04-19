# ケーススタディ: BruCo 解析で失敗しやすいパターン

BruCo は多数の補助チャンネルを横断的に走査できる一方で、設定を誤ると「高いコヒーレンスが出たのに原因切り分けに失敗する」状態に陥りやすい解析です。ここでは `advanced_bruco` の基本手順を前提に、実務で再発しやすい失敗モードを短い再現例つきで整理します。

## このケーススタディで扱うこと

- 線形相関だけを見て非線形結合を見落とす失敗
- 共通トレンドや低周波ドリフトを残したまま順位付けしてしまう失敗
- 「高コヒーレンス = 因果関係」と誤読してしまう失敗
- 上位候補を下流解析へ渡すときの確認手順

## 前提

- 基本 API と最小例は :doc:`advanced_bruco` を参照
- クラス仕様は :doc:`../../reference/Bruco` を参照

## 失敗例 1: 線形 BruCo だけで安心してしまう

**症状**: 既知の補助チャンネルが上位に出ない。代わりに周辺の環境チャンネルが弱く並ぶ。  
**典型原因**: 実際の混入経路が二乗項や積項を含むバイリニア結合なのに、元チャンネルだけを `aux_channels` に渡している。

```python
from gwexpy.analysis import Bruco

bruco = Bruco(target_channel=target.name, aux_channels=aux_names)
linear_result = bruco.compute(
    start=target.t0.value,
    duration=target.duration.value,
    fftlength=4.0,
    overlap=2.0,
    aux_data=aux_matrix,
)
linear_result.to_dataframe(ranks=[0, 1, 2])
```

この段階で順位が安定しない場合は、「補助チャンネルそのもの」ではなく「補助チャンネルから生成した仮説量」を試します。

```python
virtual_aux = aux_matrix.copy()
virtual_aux["ASC_X2"] = aux_matrix["ASC_X"] ** 2
virtual_aux["PEM_ACC_TIMES_MIC"] = aux_matrix["PEM_ACC"] * aux_matrix["PEM_MIC"]
```

**確認ポイント**:

- 仮想チャンネル追加後に上位順位が急変するか
- ターゲット ASD の線構造と候補チャンネルの周波数帯が一致するか
- 上位候補が物理的に同じ装置群へ集約されるか

## 失敗例 2: ドリフトや DC 成分を残して偽の上位候補を作る

**症状**: 低周波の大きなトレンドを持つチャンネルが常に上位に出る。  
**典型原因**: detrend や帯域制限なしでコヒーレンスを計算し、共通ドリフトを「因果的なノイズ源」と誤認している。

```python
preprocessed = {}
for name, ts in aux_matrix.items():
    preprocessed[name] = ts.detrend().highpass(5.0)

target_clean = target.detrend().highpass(5.0)
```

**防ぎたい誤読**:

- 地面振動や温度ドリフトの広帯域変動を、本命の混入チャネルだと判断すること
- 1 本のランキング表だけで commissioning action を決めること

## 失敗例 3: 高コヒーレンスを因果関係だとみなす

BruCo が返すのは「ターゲットと候補の結びつきの強さ」であって、単独で因果を証明するものではありません。共通の駆動源が別にある場合、無関係に見えるチャンネルでも高順位になり得ます。

**最低限の切り分け**:

1. 周波数帯が実際の問題帯域と一致するか確認する
2. 候補チャンネルを使って残差や差し引き後 ASD が改善するかを見る
3. 装置・センサ配置の文脈と矛盾しないか確認する

```python
top = linear_result.to_dataframe(ranks=[0]).iloc[0]
candidate = virtual_aux[top["channel"]]
residual = target_clean - candidate * 0.1
```

ここでは係数推定を単純化していますが、重要なのは「上位候補を使うとターゲットの問題帯域が本当に減るか」を見ることです。改善しないなら、順位表だけでは十分ではありません。

## 推奨ワークフロー

1. `advanced_bruco` の最小例で基準結果を得る
2. 対象帯域に合わせて detrend / bandpass / ホワイトニング の要否を決める
3. 線形候補で説明できない場合だけ仮想チャンネルを追加する
4. 上位候補は残差 ASD や後段のノイズ除去へ渡して妥当性確認する
5. 物理配置や既知の制御経路と照合する

## 関連ページ

- :doc:`advanced_bruco`
- :doc:`case_bruco_advanced`
- :doc:`case_bruco_ica_denoising`
- :doc:`case_noise_budget`
