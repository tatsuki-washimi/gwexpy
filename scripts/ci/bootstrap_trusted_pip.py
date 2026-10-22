"""Upgrade a Python interpreter's bundled pip using a pinned wheel hash."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Optional

PIP_VERSION = "26.2.2"
PIP_WHEEL_SHA256 = "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
PIP_VENDOR_CERTIFI_VERSION = "2026.10.19"


def bootstrap(python: str) -> None:
    """Install the reviewed pip wheel, then verify pip and its vendored CA set."""
    env = os.environ.copy()
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    for key in tuple(env):
        if key.startswith("PIP_") and key not in {
            "PIP_CONFIG_FILE",
            "PIP_DISABLE_PIP_VERSION_CHECK",
        }:
            del env[key]

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False
    ) as requirements:
        requirements.write(f"pip=={PIP_VERSION} --hash=sha256:{PIP_WHEEL_SHA256}\n")
        requirements_path = requirements.name
    try:
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--require-hashes",
                "--index-url",
                "https://pypi.org/simple",
                "-r",
                requirements_path,
            ],
            check=True,
            env=env,
        )
    finally:
        os.unlink(requirements_path)
    verify = (
        "import pip, pip._vendor.certifi as certifi; "
        "from pip._vendor.packaging.version import Version; "
        f"assert pip.__version__ == {PIP_VERSION!r}, pip.__version__; "
        "assert Version(certifi.__version__) >= Version("
        f"{PIP_VENDOR_CERTIFI_VERSION!r}), certifi.__version__; "
        "print(f'pip={pip.__version__} vendored-certifi={certifi.__version__}')"
    )
    subprocess.run([python, "-c", verify], check=True, env=env)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter whose pip should be bootstrapped",
    )
    args = parser.parse_args(argv)
    bootstrap(args.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
