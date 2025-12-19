
import json

nb_path = 'examples/tutorial_TimeSeries_new.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
insert_idx = -1

# Find where to insert (Before Section 4)
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        src = "".join(cell['source'])
        if "## 4." in src:
            insert_idx = i
            break

if insert_idx == -1:
    insert_idx = len(cells)

# Create new cells
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### STLT (Short-Time Local Transform)\n",
            "\n",
            "STLT (Short-Time Local Transform) は、信号の時間変化に伴う局所的な構造（対称な2軸を持つ特徴量）を抽出するための変換です。\n",
            "`gwexpy` では、このような3Dデータ (時間 x 軸1 x 軸2) を扱うために `TimePlaneTransform` クラスを提供しています。\n",
            "\n",
            "以下は、`stlt` メソッドを使用して STLT を計算し、特定時刻のスライス (`Plane2D`) を抽出する例です。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# STLT の実行\n",
            "# stride: 時間ステップ, window: 解析ウィンドウ長\n",
            "stlt_result = data.stlt(stride='0.5s', window='2s')\n",
            "\n",
            "print(f\"Kind: {stlt_result.kind}\")\n",
            "print(f\"Shape: {stlt_result.shape} (Time x Axis1 x Axis2)\")\n",
            "print(f\"Time Axis: {len(stlt_result.times)} steps\")\n",
            "\n",
            "# 特定時刻 (t=5.0s) の平面を抽出\n",
            "plane_at_5s = stlt_result.at_time(5.0 * u.s)\n",
            "print(f\"Plane at 5.0s shape: {plane_at_5s.shape}\")\n",
            "\n",
            "# Plane2D としての振る舞いを確認\n",
            "print(f\"Axis 1: {plane_at_5s.axis1.name}\")\n",
            "print(f\"Axis 2: {plane_at_5s.axis2.name}\")"
        ]
    }
]

# Insert
for c in reversed(new_cells):
    cells.insert(insert_idx, c)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
