import json

nb_path = 'examples/intro_FrequencySeries.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

cells = nb['cells']
found = False
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "## 3. 周波数ドメインでの微積分" in source:
            found = True
            
            # Update Markdown
            new_source = [
                "## 3. 周波数ドメインでの微積分\n",
                "\n",
                "`differentiate()` および `integrate()` メソッドにより、周波数ドメインで微分・積分を行うことができます。\n",
                "引数 `order` で階数を指定できます（デフォルトは1）。\n",
                "これは「変位・速度・加速度」の変換（$(2 \\pi i f)^n$ の乗算・除算）を簡単に行うための機能です。"
            ]
            cell['source'] = new_source
            
            # Update Next Code Cell
            if i + 1 < len(cells):
                next_cell = cells[i+1]
                if next_cell['cell_type'] == 'code':
                    new_code = [
                        "# 変位 (m) -> 速度 (m/s) に微分 (order=1)\n",
                        "vel_spec = spec.differentiate()\n",
                        "\n",
                        "# 変位 (m) -> 加速度 (m/s^2) に 2回微分 (order=2)\n",
                        "accel_spec = spec.differentiate(order=2)\n",
                        "\n",
                        "# 積分も可能: 加速度 -> 速度\n",
                        "vel_from_accel = accel_spec.integrate()\n",
                        "\n",
                        "plot = Plot(spec.abs(), vel_spec.abs(), accel_spec.abs(), xscale='log', yscale='log', alpha=0.8)\n",
                        "ax = plot.gca()\n",
                        "ax.get_lines()[0].set_label('Displacement [m]')\n",
                        "ax.get_lines()[1].set_label('Velocity [m/s]')\n",
                        "ax.get_lines()[2].set_label('Acceleration [m/s^2]')\n",
                        "ax.legend()\n",
                        "ax.grid(True, which='both')\n",
                        "ax.set_title('Calculus in Frequency Domain')\n",
                        "ax.set_ylabel('Magnitude')\n",
                        "plot.show()"
                    ]
                    next_cell['source'] = new_code
                    # Clear outputs to avoid inconsistency (user will run it)
                    next_cell['outputs'] = []
                    next_cell['execution_count'] = None
            break

if found:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated.")
else:
    print("Section 3 not found.")
