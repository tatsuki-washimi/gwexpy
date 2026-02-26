import os
import glob
import subprocess

def run_notebooks(directory):
    notebooks = glob.glob(os.path.join(directory, "**", "*.ipynb"), recursive=True)
    # Ignore checkpoints and build dirs
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in nb and "_build" not in nb]
    
    print(f"Found {len(notebooks)} notebooks to execute.")
    for nb in notebooks:
        print(f"Executing {nb}...")
        subprocess.run([
            "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb
        ], check=False) # check=False to continue even if one notebook fails

if __name__ == "__main__":
    docs_dir = os.path.join("/home/washimi/work/gwexpy/docs")
    run_notebooks(docs_dir)
