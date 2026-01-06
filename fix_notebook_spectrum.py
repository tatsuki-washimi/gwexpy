import json

nb_path = 'examples/tutorial_Bruco.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Old source to find (approximate)
# spectrum="asd",

new_source = []

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        if any("spectrum=\"asd\"" in line for line in source_lines):
            print("Found target cell. Removing spectrum argument.")
            new_lines = []
            for line in source_lines:
                if "spectrum=\"asd\"" in line:
                    continue # Skip this line
                new_lines.append(line)
            cell['source'] = new_lines
            found = True
            break # Assume only one occurrence for now

if not found:
    print("Target cell with spectrum=\"asd\" not found!")
    # It might be spectrum='asd' or similar
    # Let's try flexible search if needed, but strict first.
    
    # Retry flexible
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "bruco.compute" in source_str and "spectrum=" in source_str:
                 print("Found target cell (flexible). Removing spectrum argument.")
                 new_lines = []
                 for line in cell['source']:
                     if "spectrum=" in line:
                         continue
                     new_lines.append(line)
                 cell['source'] = new_lines
                 found = True
                 break

if not found:
    print("Could not find spectrum argument to remove.")
    exit(1)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
