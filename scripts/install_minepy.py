#!/usr/bin/env python3
"""
Install script for `minepy` on Python 3.11+
minepy currently lacks pre-built wheels for Python 3.11+ and its setup.py 
contains an outdated dependency on `pkg_resources` that breaks in modern pip.
This script downloads the source, patches it, recompiles the Cython extension,
and installs it into the current Python environment.

Requires: cython, numpy, scipy
"""

import os
import sys
import shutil
import urllib.request
import tarfile
import subprocess
import tempfile
from pathlib import Path


def main():
    print("=== minepy installer for Python 3.11+ ===")
    
    # 1. Check requirements
    try:
        import cython
        import numpy
        import scipy
    except ImportError as e:
        print(f"Error: Missing build requirement: {e.name}")
        print("Please install them first: pip install cython numpy scipy")
        sys.exit(1)

    url = "https://files.pythonhosted.org/packages/source/m/minepy/minepy-1.2.6.tar.gz"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tar_path = tmpdir_path / "minepy.tar.gz"
        
        # 2. Download
        print(f"Downloading minepy source from {url} ...")
        urllib.request.urlretrieve(url, tar_path)
        
        # 3. Extract
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner) 
            
            safe_extract(tar, tmpdir_path)

        minepy_src_dir = tmpdir_path / "minepy-1.2.6"
        
        # 4. Patch setup.py
        print("Patching setup.py for modern setuptools compatibility...")
        setup_py = minepy_src_dir / "setup.py"
        with open(setup_py, "r", encoding="utf-8") as f:
            setup_content = f.read()
            
        # Remove pkg_resources import which is not available in isolated builds
        setup_content = setup_content.replace(
            "from pkg_resources import get_platform", 
            "def get_platform(): return 'linux' if sys.platform.startswith('linux') else sys.platform"
        )
        # We also need import sys for the patched function
        setup_content = "import sys\n" + setup_content

        with open(setup_py, "w", encoding="utf-8") as f:
            f.write(setup_content)

        # 5. Cythonize
        print("Re-cythonizing C extensions for Python 3.11+ compatibility...")
        cython_executable = shutil.which("cython")
        if not cython_executable:
            # Fallback to python -m cython
            subprocess.run([sys.executable, "-m", "cython", "minepy/mine.pyx"], cwd=minepy_src_dir, check=True)
        else:
            subprocess.run([cython_executable, "minepy/mine.pyx"], cwd=minepy_src_dir, check=True)

        # 6. Install
        print("Installing minepy via pip...")
        env = os.environ.copy()
        # Bypass PIP_REQUIRE_VIRTUALENV if set so installation doesn't fail globally
        env["PIP_REQUIRE_VIRTUALENV"] = "false"
        
        res = subprocess.run(
            [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"], 
            cwd=minepy_src_dir, 
            env=env
        )
        
        if res.returncode == 0:
            print("\nSuccessfully installed minepy!")
            print("You can now compute Maximal Information Coefficient (MIC) in gwexpy.")
        else:
            print("\nError: minepy installation failed.")
            sys.exit(res.returncode)

if __name__ == "__main__":
    main()
