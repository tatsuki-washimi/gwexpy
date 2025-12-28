
import json

path = '/home/washimi/work/gwexpy/examples/tutorial_HHT_Analysis.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Change low-frequency description in markdown cell
# Markdown cell describing the signal is around entry index 3 or 4
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        new_source = source.replace('低周波正弦波成分（例：20 Hz）', '低周波正弦波成分（例：18.4 Hz）')
        if source != new_source:
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.splitlines()]
            # Ensure last line doesn't have extra newline if it didn't before, 
            # though splitlines/join might handle it. 
            # Actually nb format prefers list of lines ending with \n except maybe the last one.
            cell['source'] = [line if line.endswith('\n') else line + '\n' for line in new_source.splitlines()]

    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Change f_low = 20.0
        new_source = source.replace('f_low = 20.0', 'f_low = 18.4')
        if source != new_source:
            cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in new_source.splitlines()]
            # Also reset execution_count if needed, but and update outputs
            # Actually just change the code for now.
        
        # Change description text in cell 120
        new_source_desc = source.replace('低周波20 Hzの波', '低周波18.4 Hzの波')
        if source != new_source_desc:
             cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in new_source_desc.splitlines()]

# Specifically target the cell 120 description if it was in markdown
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        new_source = source.replace('低周波20 Hzの波', '低周波18.4 Hzの波')
        if source != new_source:
            cell['source'] = [line if line.endswith('\n') else line + '\n' for line in new_source.splitlines()]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
