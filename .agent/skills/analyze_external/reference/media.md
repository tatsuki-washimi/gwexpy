# Multimedia Analysis

動画・音声ファイルの分析。

## 1. Metadata & Stream Info

`ffprobe` でコーデック、ビットレート、フレームレート、サンプリング周波数、時間長等を取得：

```bash
ffprobe -v error -show_format -show_streams -of json <filename>
```

## 2. Audio Signal Analysis

`librosa`、`scipy.io.wavfile`、`pydub` 等で波形・スペクトログラム・RMS・基本周波数を抽出。

必要に応じて `pip install` を提案。

## 3. Video Frame Analysis

`OpenCV (cv2)` / `MoviePy` でフレーム抽出・画像処理（輝度変化、動体検知等）。

抽出フレームを画像保存し、AI による視覚的検査も検討。

## 4. Transcription

音声からテキスト抽出が必要な場合：`SpeechRecognition` ライブラリまたは外部 API を検討。
