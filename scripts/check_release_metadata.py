#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def get_version_from_py():
    version_file = Path("gwexpy/_version.py")
    if not version_file.exists():
        print(f"Error: {version_file} not found")
        return None
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def get_version_from_cff():
    cff_file = Path("CITATION.cff")
    if not cff_file.exists():
        print(f"Warning: {cff_file} not found")
        return None
    for line in cff_file.read_text().splitlines():
        if not line.startswith("version:"):
            continue
        value = line.partition(":")[2].strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            end = value.find(quote, 1)
            if end != -1:
                parsed = value[1:end].strip()
                if not parsed:
                    print("Error: Malformed CITATION.cff version: empty value")
                    return ""
                return parsed
            print("Error: Malformed CITATION.cff version: unterminated quote")
            return ""
        parsed = value.split("#", 1)[0].strip()
        if not parsed:
            print("Error: Malformed CITATION.cff version: empty value")
            return ""
        return parsed
    return None


def get_version_from_zenodo():
    zenodo_file = Path(".zenodo.json")
    if not zenodo_file.exists():
        print(f"Warning: {zenodo_file} not found")
        return None
    try:
        data = json.loads(zenodo_file.read_text())
        return data.get("version")
    except Exception as e:
        print(f"Error parsing .zenodo.json: {e}")
        return None


def check_changelog(version):
    changelog_file = Path("CHANGELOG.md")
    if not changelog_file.exists():
        print(f"Warning: {changelog_file} not found")
        return True
    content = changelog_file.read_text()
    # Look for [X.Y.Z] or ## [X.Y.Z]
    pattern = rf"\[{re.escape(version)}\]"
    if re.search(pattern, content):
        return True
    print(f"Error: Version {version} not found in CHANGELOG.md")
    return False


def main():
    py_version = get_version_from_py()
    if not py_version:
        print("Error: Could not determine version from gwexpy/_version.py")
        sys.exit(1)

    print(f"Detected version: {py_version}")

    cff_version = get_version_from_cff()
    zenodo_version = get_version_from_zenodo()

    errors = 0

    if cff_version is not None and cff_version != py_version:
        print(f"Error: Version mismatch in CITATION.cff: {cff_version} != {py_version}")
        errors += 1
    elif cff_version == py_version:
        print("OK: CITATION.cff version matches.")
    else:
        print("Warning: CITATION.cff version not checked.")

    if zenodo_version and zenodo_version != py_version:
        print(
            f"Error: Version mismatch in .zenodo.json: {zenodo_version} != {py_version}"
        )
        errors += 1
    else:
        print("OK: .zenodo.json version matches.")

    if not check_changelog(py_version):
        errors += 1
    else:
        print("OK: CHANGELOG.md entry found.")

    if errors > 0:
        print(f"\nFound {errors} metadata consistency error(s).")
        sys.exit(1)

    print("\nMetadata consistency check passed!")


if __name__ == "__main__":
    main()
