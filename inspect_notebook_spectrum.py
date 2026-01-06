import json

nb_path = 'examples/tutorial_Bruco.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "bruco.compute" in source:
            print(f"--- Cell {i} (contains bruco.compute) ---")
            print(source)
            if "spectrum=" in source:
                print("WARNING: 'spectrum=' argument still found!")
            else:
                print("VERIFIED: 'spectrum=' argument NOT found.")
            print("\n")
