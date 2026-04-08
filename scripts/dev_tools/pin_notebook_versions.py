import json
import os
from pathlib import Path

# Target directories for tutorials
TUTORIAL_DIRS = [
    'docs/web/ja/user_guide/tutorials',
    'docs/web/en/user_guide/tutorials'
]

# Installation snippet to be inserted at the top of notebooks
INSTALL_CELL_CONTENT = [
    "# Install gwexpy with pinned versions of core dependencies for reproducibility on Colab\n",
    "%pip install -q \"gwexpy[all]\" \"gwpy<5.0.0\" \"numpy<2.0.0\" \"scipy<1.13.0\" \"astropy<7.0.0\""
]

def patch_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Check if the first cell is already an installation cell (heuristic)
    first_cell = nb.get('cells', [])[0] if nb.get('cells') else None
    
    is_already_present = False
    if first_cell and first_cell.get('cell_type') == 'code':
        source = "".join(first_cell.get('source', []))
        if '%pip install' in source or 'pip install' in source:
            # Update existing cell
            first_cell['source'] = [line + '\n' for line in INSTALL_CELL_CONTENT[:-1]] + [INSTALL_CELL_CONTENT[-1]]
            is_already_present = True

    if not is_already_present:
        # Create a new cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in INSTALL_CELL_CONTENT[:-1]] + [INSTALL_CELL_CONTENT[-1]]
        }
        nb['cells'].insert(0, new_cell)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n') # Ensure newline at end of file
    
    return not is_already_present

def main():
    modified_count = 0
    inserted_count = 0
    
    for d in TUTORIAL_DIRS:
        dp = Path(d)
        if not dp.exists():
            continue
            
        for nb_path in dp.glob('*.ipynb'):
            print(f"Patching {nb_path}...")
            was_inserted = patch_notebook(nb_path)
            if was_inserted:
                inserted_count += 1
            else:
                modified_count += 1
                
    print(f"Done. Inserted {inserted_count} new cells, modified {modified_count} existing cells.")

if __name__ == '__main__':
    main()
