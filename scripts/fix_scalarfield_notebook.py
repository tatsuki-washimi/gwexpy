#!/usr/bin/env python3
"""Fix ScalarField tutorial notebook to handle 4D-preserving slicing."""

import nbformat

# Read notebook
nb = nbformat.read("examples/tutorials/intro_ScalarField.ipynb", as_version=4)

# Find and fix cells with plotting issues
# Cell 8: FFT spectrum plot
cell_8_original = """# 周波数スペクトルをプロット（1点のx,y,zを選択）
spectrum = np.abs(field_freq[:, 0, 0, 0].value)
freqs = field_freq._axis0_index.value"""

cell_8_fixed = """# 周波数スペクトルをプロット（1点のx,y,zを選択）
# Note: スライスしても4D構造が維持されるため、squeeze()で次元を削減
spectrum = np.abs(field_freq[:, 0, 0, 0].value).squeeze()
freqs = field_freq._axis0_index.value"""

# Cell 10: IFFT comparison plot
cell_10_original = """# 逆FFT
field_reconstructed = field_freq.ifft_time()

# 元の信号と比較
original = field_time[:, 0, 0, 0].value
reconstructed = field_reconstructed[:, 0, 0, 0].value"""

cell_10_fixed = """# 逆FFT
field_reconstructed = field_freq.ifft_time()

# 元の信号と比較
# Note: 4D構造を維持するため、squeeze()で1次元に
original = field_time[:, 0, 0, 0].value.squeeze()
reconstructed = field_reconstructed[:, 0, 0, 0].value.squeeze()"""

# Cell 13: K-space spectrum plot
cell_13_original = """# K空間スペクトルをプロット
kx_spectrum = np.abs(field_kx[0, :, 0, 0].value)
kx_values = field_kx._axis1_index.value"""

cell_13_fixed = """# K空間スペクトルをプロット
# Note: squeeze()で次元削減
kx_spectrum = np.abs(field_kx[0, :, 0, 0].value).squeeze()
kx_values = field_kx._axis1_index.value"""

# Apply fixes
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        if cell_8_original in cell.source:
            cell.source = cell.source.replace(cell_8_original, cell_8_fixed)
            print(f"Fixed cell {i}: FFT spectrum plot")
        elif cell_10_original in cell.source:
            cell.source = cell.source.replace(cell_10_original, cell_10_fixed)
            print(f"Fixed cell {i}: IFFT comparison")
        elif cell_13_original in cell.source:
            cell.source = cell.source.replace(cell_13_original, cell_13_fixed)
            print(f"Fixed cell {i}: K-space spectrum plot")

# Write back
nbformat.write(nb, "examples/tutorials/intro_ScalarField.ipynb")
print("\n✓ Fixed: examples/tutorials/intro_ScalarField.ipynb")
