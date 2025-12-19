
import json

nb_path = 'examples/tutorial_TimeSeries_new.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the cell with the STLT code
target_source_fragment = "stlt_result = data.stlt"

for cell in cells:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if target_source_fragment in source_str:
            # Update the source to include data generation
            cell['source'] = [
                "# データの準備 (デモ用)\n",
                "import numpy as np\n",
                "from gwexpy.timeseries import TimeSeries\n",
                "from astropy import units as u\n",
                "\n",
                "t = np.linspace(0, 10, 1000)\n",
                "data = TimeSeries(np.sin(2 * np.pi * 1 * t), times=t*u.s, unit='V', name='Demo Data')\n",
                "\n",
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
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
