import json

nb_path = '/home/washimi/work/gwexpy/examples/fitting_demo.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with the incorrect import
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "from gwexpy.timeseries import FrequencySeries" in source:
            # Replace with correct import
            new_source = source.replace("from gwexpy.timeseries import FrequencySeries", 
                                      "from gwexpy.frequencyseries import FrequencySeries")
            cell['source'] = [line + "\n" for line in new_source.splitlines()]
            # Clean up newlines if splitlines adds them implicitly, 
            # but usually we want to preserve original structure.
            # actually splitlines removes \n, so appending \n is correct.
            # Check last line
            if not new_source.endswith('\n'):
                 cell['source'][-1] = cell['source'][-1].rstrip()
                 
            print("Fixed import in notebook.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
