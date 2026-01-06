import json

nb_path = 'examples/tutorial_Bruco.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Correct initialization: Bruco(target_channel, aux_channels)
# We pass an empty list for aux_channels to prevent auto-fetching, as we will inject data manually.

new_source = [
    "import gwexpy.analysis.bruco as bruco_module\n",
    "from gwexpy.analysis.bruco import Bruco\n",
    "\n",
    "# Brucoの初期化。\n",
    "# ここではデータは全て手動で渡すため、NDSからの自動取得チャンネルリスト(aux_channels)は空リストを指定します。\n",
    "bruco = Bruco(target_channel=target.name, aux_channels=[])\n",
    "\n",
    "# computeメソッドに target_data と aux_data を渡すことで、内部Fetchをスキップして解析を実行できます。\n",
    "result = bruco.compute(\n",
    "    start=0,\n",
    "    duration=int(duration),\n",
    "    fftlength=fftlength,\n",
    "    overlap=overlap,\n",
    "    nproc=1,\n",
    "    batch_size=2,\n",
    "    top_n=3,\n",
    "    spectrum=\"asd\",\n",
    "    target_data=target, # 事前に生成したターゲットデータ\n",
    "    aux_data=aux_dict   # 事前に生成した補助チャンネルデータの辞書\n",
    ")\n"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        # Matching purely on "bruco = Bruco" is safer if previous attempts modified it slightly
        if "bruco = Bruco(" in source_str:
            print(f"Found target cell (Source snippet: {source_str[:50]}...). Replacing content.")
            cell['source'] = new_source
            found = True
            break

if not found:
    print("Target cell not found!")
    exit(1)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
