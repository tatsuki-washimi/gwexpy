import json

nb_path = '/home/washimi/work/gwexpy/examples/fitting_demo.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last cell which should be the complex fitting demo
last_cell = nb['cells'][-1]

# Replace result_complex.plot() with result_complex.bode_plot()
source = "".join(last_cell['source'])
if "result_complex.plot()" in source:
    new_source = source.replace("result_complex.plot()", "result_complex.bode_plot()")
    last_cell['source'] = [line + "\n" for line in new_source.splitlines()]
    # Remove extra newline at end of last line if exists
    last_cell['source'][-1] = last_cell['source'][-1].rstrip()

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4, ensure_ascii=False)
    print("Notebook updated.")
else:
    print("bode_plot already used or pattern not found.")
