新しい信号処理メソッド
========================

.. currentmodule:: gwexpy.fields

このガイドでは、GWpy の ``TimeSeries`` および ``FrequencySeries`` API を参考に :class:`ScalarField` と :class:`FieldDict` に追加された新しい信号処理メソッドについて説明します。

概要
----

3つの優先度レベルで **23個の新しいメソッド** を実装しました：

- **高優先度 (10個)**: 重力波データ解析に必須のコア信号処理機能
- **低優先度 (12個)**: ユーティリティと数学演算
- **中優先度 (9個)**: 高度な解析メソッド（将来の実装のため保留）

すべてのメソッドは :class:`ScalarField` と :class:`FieldDict`（全コンポーネントに操作を適用）の両方で利用可能です。

基本的な前処理
--------------

デトレンディング
~~~~~~~~~~~~~~~~

スペクトル解析を歪める可能性のある多項式トレンドを除去します::

    # 線形トレンドを除去
    detrended = field.detrend('linear')

    # DCオフセットのみを除去
    detrended = field.detrend('constant')

テーパリング
~~~~~~~~~~~~

FFTのリンギングアーティファクトを抑制するために窓関数を適用します::

    from astropy import units as u

    # 両端を1秒間テーパリング
    tapered = field.taper(duration=1.0*u.s)

    # 左側のみを100サンプルテーパリング
    tapered = field.taper(side='left', nsamples=100)

クロッピングとパディング
~~~~~~~~~~~~~~~~~~~~~~~~

時間セグメントを抽出、またはデータを拡張します::

    # 10秒から20秒のセグメントを抽出
    segment = field.crop(start=10*u.s, end=20*u.s)

    # 各端に100サンプルをゼロでパディング
    padded = field.pad(100)

    # エッジ値で非対称にパディング
    padded = field.pad((50, 150), mode='edge')

数学演算
--------

統計演算
~~~~~~~~

任意の軸に沿って統計量を計算します::

    # グローバル統計
    mean_val = field.mean()
    median_val = field.median()
    std_val = field.std()
    rms_val = field.rms()

    # 時間軸の統計（空間フィールドに縮約）
    time_mean = field.mean(axis=0)
    time_rms = field.rms(axis=0)

    # 空間統計
    x_profile = field.mean(axis=1)

要素ごとの演算
~~~~~~~~~~~~~~

::

    # 絶対値
    abs_field = field.abs()

    # 平方根
    sqrt_field = field.sqrt()

高度な信号処理
--------------

ホワイトニング
~~~~~~~~~~~~~~

振幅スペクトル密度を正規化してスペクトルを平坦化します::

    # 2秒セグメントを使用し、1秒オーバーラップでホワイトニング
    whitened = field.whiten(fftlength=2.0, overlap=1.0)

これは、色付きノイズに埋もれた弱い信号を強調するため、マッチドフィルタリング前の必須前処理です。

畳み込み
~~~~~~~~

時間領域でFIRフィルタを適用します::

    import numpy as np

    # シンプルなマッチドフィルタテンプレート
    template = np.array([1, 2, 3, 2, 1]) / 9.0
    matched = field.convolve(template, mode='same')

信号注入
~~~~~~~~

検出パイプラインのテスト用にシミュレーション信号を追加します::

    # シミュレーション平面波を作成
    signal = ScalarField.simulate('plane_wave',
                                   shape=(1000, 10, 10, 10),
                                   frequency=100,
                                   amplitude=1e-21)

    # スケーリング係数で注入
    injected = field.inject(signal, alpha=0.5)

フィルタリングメソッド
~~~~~~~~~~~~~~~~~~~~~~

Zero-Pole-Gain (ZPK) フィルタ::

    # カスタムIIRフィルタ
    zeros = [0]
    poles = [-1, -1+1j, -1-1j]
    gain = 1.0
    filtered = field.zpk(zeros, poles, gain)

クロススペクトル解析
--------------------

クロススペクトル密度
~~~~~~~~~~~~~~~~~~~~

異なるチャンネルまたは空間点間の関係を解析します::

    # 2つのフィールド間のCSD
    csd_result = field1.csd(field2, fftlength=2.0, overlap=1.0)

コヒーレンス
~~~~~~~~~~~~

各周波数での相関を示す周波数コヒーレンス（0-1の値）を計算します::

    # 相関するノイズ源を特定
    coh = field1.coherence(field2, fftlength=2.0, overlap=1.0)

    # 高いコヒーレンス（>0.8）は相関する信号/ノイズを示唆
    correlated_freqs = coh.value > 0.8

スペクトログラム
~~~~~~~~~~~~~~~~

時間-周波数表現を生成します::

    # 1秒ストライド、2秒FFT長のスペクトログラム
    spec = field.spectrogram(stride=1.0, fftlength=2.0, overlap=1.0)

時系列ユーティリティ
--------------------

互換性チェック
~~~~~~~~~~~~~~

フィールドを結合できるか検証します::

    if field1.is_compatible(field2):
        combined = field1.append(field2)

    if field1.is_contiguous(field2):
        # フィールドは時間的に隣接
        combined = field1.append(field2, gap='ignore')

連結
~~~~

時間セグメントを追加または前方結合します::

    # シンプルな追加
    combined = field1.append(field2)

    # ギャップをパディングで処理
    combined = field1.append(field2, gap='pad', pad=0.0)

    # 前方結合（field2がfield1の前に来る）
    combined = field1.prepend(field2)

値の抽出
~~~~~~~~

特定時刻のフィールド値を抽出します::

    # 単一時刻点（3D空間配列を返す）
    values_3d = field.value_at(5.0 * u.s)

    # 複数時刻点（ScalarFieldを返す）
    times = [1.0, 2.0, 3.0] * u.s
    subset = field.value_at(times)

FieldDict 操作
--------------

すべてのメソッドは、各コンポーネントに操作を適用することで :class:`FieldDict` でも機能します::

    from gwexpy.fields import FieldDict

    # FieldDictとしてベクトル場を作成
    vector_field = FieldDict({
        'x': Ex_field,
        'y': Ey_field,
        'z': Ez_field
    })

    # すべてのコンポーネントを前処理
    detrended = vector_field.detrend('linear')
    whitened = vector_field.whiten(fftlength=2.0)

    # コンポーネント間のクロススペクトル解析
    csd_xy = vector_field['x'].csd(vector_field['y'])
    coh_xy = vector_field['x'].coherence(vector_field['y'])

    # すべてのコンポーネントの統計演算
    rms_dict = vector_field.rms(axis=0)
    mean_dict = vector_field.mean(axis=0)

使用例
------

完全なワークフロー例
~~~~~~~~~~~~~~~~~~~~

典型的な重力波データ解析ワークフロー::

    import numpy as np
    from astropy import units as u
    from gwexpy.fields import ScalarField

    # 1. フィールドデータの読み込み/作成
    field = ScalarField(data, unit=u.m/u.s, axis0=times, ...)

    # 2. 前処理
    field = field.detrend('linear')        # トレンド除去
    field = field.taper(duration=1.0*u.s)  # エッジ効果の抑制
    field = field.highpass(10)             # 低周波除去

    # 3. マッチドフィルタリング用のホワイトニング
    whitened = field.whiten(fftlength=2.0, overlap=1.0)

    # 4. テンプレートでマッチドフィルタリング
    template = create_template()  # テンプレート関数
    matched = whitened.convolve(template, mode='same')

    # 5. 解析
    snr_map = matched.abs()
    peak_time = times[np.argmax(snr_map.value[:, 0, 0, 0])]

    # 6. スペクトル解析
    psd = field.psd(axis=0, fftlength=2.0)
    spec = field.spectrogram(stride=0.5, fftlength=1.0)

マルチ検出器解析
~~~~~~~~~~~~~~~~

複数の検出器からのデータを解析::

    # 複数検出器用のFieldDictを作成
    detectors = FieldDict({
        'H1': h1_field,  # LIGO Hanford
        'L1': l1_field,  # LIGO Livingston
        'V1': v1_field   # Virgo
    })

    # すべての検出器を同じように前処理
    detectors = detectors.detrend('linear')
    detectors = detectors.bandpass(30, 300)
    detectors = detectors.whiten(fftlength=4.0)

    # 検出器間のクロスコヒーレンス
    coh_HL = detectors['H1'].coherence(detectors['L1'], fftlength=4.0)
    coh_HV = detectors['H1'].coherence(detectors['V1'], fftlength=4.0)

    # すべての検出器にテスト信号を注入
    signal = ScalarField.simulate('plane_wave', ...)
    injected = detectors.inject(signal, alpha=1.0)

参照
----

- :doc:`/web/ja/reference/api/fields` - 完全なAPIリファレンス
- :doc:`/web/ja/examples/index` - その他の例
- `GWpy TimeSeries ドキュメント <https://gwpy.github.io/docs/stable/timeseries/>`_ - リファレンス実装
