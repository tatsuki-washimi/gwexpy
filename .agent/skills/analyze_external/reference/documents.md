# Office Document Analysis

PDF、Word、Excel、PowerPoint からのデータ抽出・分析。

## PDF Analysis

- **Text**: `pdfminer.six` / `PyPDF2`
- **Tables**: `tabula-py` / `camelot-py`
- **Images/Metadata**: `PyMuPDF (fitz)`

## Excel (xlsx/csv)

- `pandas` (`read_excel`, `read_csv`) で統計分析
- `openpyxl` でシート構造・数式の取得

## Word (docx)

- `python-docx` で章構造・段落・表・スタイルを抽出

## PowerPoint (pptx)

- `python-pptx` でスライドタイトル・本文・ノート・シェイプテキストをスキャン

## Tips

- 大ファイルは中間フォーマット（.txt, .csv）に変換して `tmp/` に保存
- スキャンPDFには OCR（`Tesseract`, `EasyOCR`）を提案
