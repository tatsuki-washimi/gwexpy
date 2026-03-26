# STLT 資料の整理・整理推奨リスト

`docs/developers/references/data-analysis/STLT/` 以下の資料を調査し、情報の重複や上位互換（新バージョン）の存在を確認しました。

## 概要：STLT (Short-Time Laplace Transform) とは

これらの資料群は、主に **重力波のリングダウン（ブラックホール準固有振動：QNM）** 解析を目的とした STLT の研究資料です。
通常のフーリエ変換（STFT）が実周波数軸（$j\omega$）のみを扱うのに対し、ラプラス変換では複素平面（$s = \sigma + j\omega$）上での解析を行うことで、**減衰振動（Damped Sinusoids）** の信号抽出やパラメータ推定（振動数・減衰時間）において有利であるとされています。

---

## 整理推奨リスト

以下のファイルは、より新しいバージョンや詳細な資料が存在するため、整理（削除またはアーカイブ）が可能です。

### 1. 土田氏 (Tsuchida) による発表資料

時系列で内容がアップデートされており、古いものは削除可能です。

- **削除対象**: `Tsuchida_JPS2021Spring@Tokyo(Remote).pdf` (2021年春)
- **削除対象**: `Tsuchida_JPS2021S_v2@Kobe(Remote).pdf` (2021年秋) - ファイルサイズ大(25MB)
- **残すべき**: `Tsuchida_JPS2022Spring.pdf` (2022年) - より新しいバージョンのため。

### 2. 神田氏 (Kanda) による発表資料

- **削除対象**: `17th_Kanda_LaplaceTr_KIW11_rev3.pdf` (古いリビジョン)
- **残すべき**: `JSP2024spring_21pW3-4_Kanda.pdf` (2024年) - 最新の研究状況が反映されています。

### 3. その他（要旨のみなど）

- **削除対象**: `TitleAbstract_2020S.pdf` - 2020年秋季学会の講演申込内容（タイトル・概要）のみであり、実質的な技術情報を含んでいません。

### 4. 最新資料（保持必須）

- **残すべき**: `f2f_2025_ishino.pdf` (2025年, Ishino) - 現在の最新の研究状況（パラメータ推定など）を示しています。

---

## 結論

以下のファイルを削除することで、ディレクトリ容量を削減し、最新の知見（2022年〜2025年の資料）にアクセスしやすくなります。

- `Tsuchida_JPS2021Spring@Tokyo(Remote).pdf`
- `Tsuchida_JPS2021S_v2@Kobe(Remote).pdf`
- `17th_Kanda_LaplaceTr_KIW11_rev3.pdf`
- `TitleAbstract_2020S.pdf`
