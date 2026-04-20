import os
import re

import nbformat


def patch_source(source):
    # 0. Clean up duplicate arguments from previous failed patches
    source = re.sub(r"overlap=0\.2, overlap=0\.5", "overlap=0.2", source)
    source = re.sub(r"overlap=0\.5, overlap=0\.2", "overlap=0.2", source)
    source = re.sub(r"overlap=0\.2, overlap=0\.2", "overlap=0.2", source)

    # 1. FFT parameters: Ensure noverlap < nfft for GBD
    # Find calls like .spectrogram(stride=0.5, fftlength=0.4) and replace intelligently
    if "stride=0.5, fftlength=0.4" in source:
        # If it also has overlap=0.5, it will fail.
        if "overlap=0.5" in source:
            source = source.replace("overlap=0.5", "overlap=0.2")
        elif "overlap" not in source:
            source = source.replace("fftlength=0.4", "fftlength=0.4, overlap=0.2")

    # 2. Colorbar fixes: Use explicit mappable
    # We use a regex to only match plt.colorbar() and plot.colorbar() that don't have mappable=
    def replace_colorbar(match):
        prefix = match.group(1)
        args = match.group(2)
        if "mappable=" in args:
            return match.group(0)
        mappable = 'mappable=plt.gca().get_images()[-1] if plt.gca().get_images() else plt.gca().collections[-1]'
        if args.strip():
            return f"{prefix}.colorbar({mappable}, {args}"
        else:
            return f"{prefix}.colorbar({mappable})"

    source = re.sub(r"(plt|plot)\.colorbar\((.*)\)", replace_colorbar, source)

    # 3. DTTXML parser fixes
    if "tf_prod = products[\"TF\"][0]" in source:
        source = source.replace("tf_prod = products[\"TF\"][0]", "tf_prod = list(products.get(\"TF\", {}).values())[0]")
    if "coh_prod = products[\"COH\"][0]" in source:
        source = source.replace("coh_prod = products[\"COH\"][0]", "coh_prod = list(products.get(\"COH\", {}).values())[0]")

    # 4. Physics Validation: RMS check
    if "ts.asd(fftlength=1.0, overlap=0.5)" in source and "window='rect'" not in source:
        source = source.replace("ts.asd(fftlength=1.0, overlap=0.5)", "ts.asd(fftlength=1.0, overlap=0.5, window='rect')")
    if "ts.psd(fftlength=1.0, overlap=0.5)" in source and "window='rect'" not in source:
        source = source.replace("ts.psd(fftlength=1.0, overlap=0.5)", "ts.psd(fftlength=1.0, overlap=0.5, window='rect')")

    # 5. Modal Analysis
    if "ax.scatter(fn, zeta * 100" in source and "for r" not in source:
        source = source.replace(
            "ax.scatter(fn, zeta * 100, c=colors, s=120, zorder=5, edgecolors=\"black\")",
            "for r, (f0, z) in enumerate(zip(np.atleast_1d(fn), np.atleast_1d(zeta))):\n    ax.scatter(f0, z*100, color=f\"C{r}\", s=120, zorder=5, edgecolors=\"black\")"
        )

    if "np.atleast_1d" in source and "import numpy as np" not in source:
         source = "import numpy as np\n" + source

    return source

def patch_notebook(path):
    print(f"Patching {path}...")
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    changed = False
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue

        old_source = cell.source
        new_source = patch_source(old_source)
        if old_source != new_source:
            cell.source = new_source
            changed = True

    if changed:
        with open(path, 'w') as f:
            nbformat.write(nb, f)
        print(f"  Saved changes to {path}")

if __name__ == "__main__":
    notebooks = []
    for root, dirs, files in os.walk("docs/web"):
        for f in files:
            if f.endswith(".ipynb"):
                notebooks.append(os.path.join(root, f))

    for nb in notebooks:
        patch_notebook(nb)
