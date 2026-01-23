---
name: multimedia_analysis
description: 動画・音声ファイルの内容、メタデータ、およびストリーム情報の分析を行う
---

# Multimedia Analysis

動画 (Video) や音声 (Audio) ファイルを、プログラミング的アプローチとツール（ffmpeg等）を組み合わせて分析します。

## 1. メタデータとストリーム情報の取得
*   `ffmpeg` や `ffprobe` を `run_command` で実行し、コーデック、ビットレート、フレームレート、サンプリング周波数、持続時間等を確認します。
*   `ffprobe -v error -show_format -show_streams -of json <filename>`

## 2. 音声信号の分析
*   `librosa`, `scipy.io.wavfile`, `pydub` 等のライブラリを使用し、波形、スペクトログラム、RMSレベル、基本周波数等を抽出します。
*   **注意**: 必要に応じて `pip install` を提案・実行します。

## 3. 動画フレームの分析
*   `OpenCV (cv2)` や `MoviePy` を使用して、特定のタイムスタンプのフレームを抽出したり、動画像処理（輝度変化、動き検出）を行います。
*   抽出したフレームを画像として保存し、`view_file` で AI が視覚的に確認（メタ情報の読み取りなど）することを検討します（モデルが画像対応の場合）。

## 4. 文字起こし・要約（API / ライブラリ）
*   音声データからテキストを抽出する必要がある場合、ライブラリ（`SpeechRecognition` 等）や外部APIの利用を検討します。
