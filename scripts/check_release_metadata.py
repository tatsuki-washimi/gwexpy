#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

QUOTED_VALUE_PATTERN = re.compile(r'(["\'])(.*?)\1\s*(?:#.*)?$')
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_version_from_py():
    version_file = Path("gwexpy/_version.py")
    if not version_file.exists():
        print(f"Error: {version_file} not found")
        return None
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def _parse_cff_scalar_value(value: str, *, source: str) -> str:
    value = value.strip()
    if value.startswith(("'", '"')):
        quoted_match = QUOTED_VALUE_PATTERN.fullmatch(value)
        if not quoted_match:
            if value.count(value[0]) < 2:
                print(f"Error: Malformed CITATION.cff {source}: unterminated quote")
            else:
                print(
                    f"Error: Malformed CITATION.cff {source}: "
                    "trailing content after quoted value"
                )
            return ""
        parsed = quoted_match.group(2).strip()
    else:
        parsed = value.split("#", 1)[0].strip()

    if not parsed:
        print(f"Error: Malformed CITATION.cff {source}: empty value")
        return ""
    return parsed


def _get_top_level_cff_value(key: str) -> str:
    cff_file = Path("CITATION.cff")
    if not cff_file.exists():
        print(f"Error: {cff_file} not found")
        return ""

    key_pattern = re.compile(
        rf'(?:{re.escape(key)}|"{re.escape(key)}"|\'{re.escape(key)}\')\s*:\s*(.*)$'
    )

    top_level_indent = None
    for line in cff_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in {"---", "..."}:
            continue
        indent = len(line) - len(line.lstrip())
        if top_level_indent is None:
            top_level_indent = indent
        if indent != top_level_indent:
            continue

        match = key_pattern.fullmatch(line.lstrip())
        if not match:
            continue

        return _parse_cff_scalar_value(match.group(1), source=key)

    print(f"Error: Could not parse top-level {key} from CITATION.cff")
    return ""


def get_version_from_cff():
    return _get_top_level_cff_value("version")


def get_date_from_cff():
    return _get_top_level_cff_value("date-released")


def get_version_from_zenodo():
    return _get_zenodo_field("version")


def get_date_from_zenodo():
    return _get_zenodo_field("publication_date")


def _get_zenodo_field(field_name):
    zenodo_file = Path(".zenodo.json")
    if not zenodo_file.exists():
        print(f"Error: {zenodo_file} not found")
        return ""
    try:
        data = json.loads(zenodo_file.read_text())
        value = data.get(field_name)
    except Exception as e:
        print(f"Error parsing .zenodo.json: {e}")
        return ""
    if not value:
        print(f"Error: .zenodo.json missing {field_name}")
        return ""
    return str(value)


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


def get_changelog_release_date(version):
    changelog_file = Path("CHANGELOG.md")
    if not changelog_file.exists():
        print(f"Error: {changelog_file} not found")
        return ""
    content = changelog_file.read_text()
    match = re.search(
        rf"^## \[{re.escape(version)}\]\s*-\s*(\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        content,
        flags=re.MULTILINE,
    )
    if not match:
        print(f"Error: Release date for version {version} not found in CHANGELOG.md")
        return ""
    return match.group(1)


def _check_date(label, value):
    if not DATE_PATTERN.fullmatch(value):
        print(f"Error: {label} release date is not YYYY-MM-DD: {value}")
        return False
    return True


def main():
    py_version = get_version_from_py()
    if not py_version:
        print("Error: Could not determine version from gwexpy/_version.py")
        sys.exit(1)

    print(f"Detected version: {py_version}")

    cff_version = get_version_from_cff()
    cff_date = get_date_from_cff()
    zenodo_version = get_version_from_zenodo()
    zenodo_date = get_date_from_zenodo()
    changelog_date = get_changelog_release_date(py_version)

    errors = 0

    if cff_version != py_version:
        print(f"Error: Version mismatch in CITATION.cff: {cff_version} != {py_version}")
        errors += 1
    else:
        print("OK: CITATION.cff version matches.")

    if zenodo_version != py_version:
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

    if not _check_date("CITATION.cff", cff_date):
        errors += 1
    if not _check_date(".zenodo.json", zenodo_date):
        errors += 1
    if not _check_date("CHANGELOG.md", changelog_date):
        errors += 1

    if cff_date and zenodo_date and cff_date != zenodo_date:
        print(
            f"Error: Release date mismatch: CITATION.cff {cff_date} != "
            f".zenodo.json {zenodo_date}"
        )
        errors += 1
    if cff_date and changelog_date and cff_date != changelog_date:
        print(
            f"Error: Release date mismatch: CITATION.cff {cff_date} != "
            f"CHANGELOG.md {changelog_date}"
        )
        errors += 1
    if (
        cff_date
        and zenodo_date
        and changelog_date
        and cff_date == zenodo_date == changelog_date
    ):
        print("OK: release dates match.")

    if errors > 0:
        print(f"\nFound {errors} metadata consistency error(s).")
        sys.exit(1)

    print("\nMetadata consistency check passed!")


if __name__ == "__main__":
    main()
