#!/usr/bin/env python3
"""Generate meta-refresh redirect stubs from the old docs/web publish paths to
the new docs_redesign site (Step F-3 design, Step F-4 cutover).

Run against a gh-pages branch checkout *after* the new site has already been
rsynced into place, since rsync --delete would otherwise wipe these stubs:

    python scripts/generate_docs_redirect_stubs.py /path/to/gh-pages-checkout
"""

import argparse
from pathlib import Path

SITE_ROOT = "https://tatsuki-washimi.github.io/gwexpy/"

# (old_en, old_ja, new_en, new_ja).
# - old_en/old_ja are relative to docs/web/en/ and docs/web/ja/ respectively.
# - new_en/new_ja are relative to docs/ (the /docs/ subdir kept per Step F-2).
REDIRECTS = [
    ("index.html", "index.html", "index.html", "ja/index.html"),
    ("guide/index.html", "guide/index.html", "index.html", "ja/index.html"),
    (
        "user_guide/quickstart.html",
        "user_guide/quickstart.html",
        "tutorials/quickstart.html",
        "ja/tutorials/quickstart.html",
    ),
    (
        "user_guide/getting_started.html",
        "user_guide/getting_started.html",
        "tutorials/getting_started.html",
        "ja/tutorials/getting_started.html",
    ),
    (
        "user_guide/installation.html",
        "user_guide/installation.html",
        "tutorials/installation.html",
        "ja/tutorials/installation.html",
    ),
    (
        "user_guide/troubleshooting.html",
        "user_guide/troubleshooting.html",
        "how-to/troubleshooting.html",
        "ja/how-to/troubleshooting.html",
    ),
    (
        "user_guide/migration_0.1.1.html",
        "user_guide/migration_0.1.1.html",
        "how-to/migration.html",
        "ja/how-to/migration.html",
    ),
    (
        "user_guide/interop.html",
        "user_guide/interop.html",
        "how-to/interop.html",
        "ja/how-to/interop.html",
    ),
    (
        "user_guide/io_formats.html",
        "user_guide/io_formats.html",
        "how-to/io_formats.html",
        "ja/how-to/io_formats.html",
    ),
    (
        "user_guide/time_utilities.html",
        "user_guide/time_utilities.html",
        "how-to/time_utilities.html",
        "ja/how-to/time_utilities.html",
    ),
    (
        "user_guide/cli.html",
        "user_guide/cli.html",
        "how-to/cli.html",
        "ja/how-to/cli.html",
    ),
    (
        "user_guide/gui.html",
        "user_guide/gui.html",
        "how-to/gui.html",
        "ja/how-to/gui.html",
    ),
    (
        "user_guide/scalarfield_slicing.html",
        "user_guide/scalarfield_slicing.html",
        "how-to/containers/scalarfield_slicing.html",
        "ja/how-to/containers/scalarfield_slicing.html",
    ),
    (
        "user_guide/architecture.html",
        "user_guide/architecture.html",
        "explanation/architecture.html",
        "ja/explanation/architecture.html",
    ),
    (
        "user_guide/physics_models.html",
        "user_guide/physics_models.html",
        "explanation/physics_models.html",
        "ja/explanation/physics_models.html",
    ),
    (
        "user_guide/numerical_stability.html",
        "user_guide/numerical_stability.html",
        "explanation/numerical_stability.html",
        "ja/explanation/numerical_stability.html",
    ),
    (
        "user_guide/prerequisites_and_conventions.html",
        "user_guide/prerequisites_and_conventions.html",
        "explanation/prerequisites_and_conventions.html",
        "ja/explanation/prerequisites_and_conventions.html",
    ),
    (
        "user_guide/roadmap.html",
        "user_guide/roadmap.html",
        "explanation/roadmap.html",
        "ja/explanation/roadmap.html",
    ),
    (
        "user_guide/validated_algorithms.html",
        "user_guide/validated_algorithms.html",
        "explanation/validated_algorithms.html",
        "ja/explanation/validated_algorithms.html",
    ),
    (
        "user_guide/verification_and_quality.html",
        "user_guide/verification_and_quality.html",
        "explanation/verification_and_quality.html",
        "ja/explanation/verification_and_quality.html",
    ),
    (
        "user_guide/gwexpy_for_gwpy_users_en.html",
        "user_guide/gwexpy_for_gwpy_users_ja.html",
        "explanation/gwexpy_for_gwpy_users.html",
        "ja/explanation/gwexpy_for_gwpy_users.html",
    ),
    (
        "reference/index.html",
        "reference/index.html",
        "reference/index.html",
        "ja/reference/index.html",
    ),
    (
        "reference/api/index.html",
        "reference/api/index.html",
        "reference/index.html",
        "ja/reference/index.html",
    ),
    (
        "user_guide/gwpy_added_api_index_en.html",
        "user_guide/gwpy_added_api_index_ja.html",
        "reference/gwpy_added_api.html",
        "ja/reference/gwpy_added_api.html",
    ),
    (
        "user_guide/glossary.html",
        "user_guide/glossary.html",
        "reference/index.html",
        "ja/reference/index.html",
    ),
    (
        "user_guide/citation.html",
        "user_guide/citation.html",
        "about/citation.html",
        "ja/about/citation.html",
    ),
    (
        "user_guide/license.html",
        "user_guide/license.html",
        "about/license.html",
        "ja/about/license.html",
    ),
    (
        "user_guide/changelog.html",
        "user_guide/changelog.html",
        "about/changelog.html",
        "ja/about/changelog.html",
    ),
    (
        "examples/index.html",
        "examples/index.html",
        "how-to/case-studies/index.html",
        "ja/how-to/case-studies/index.html",
    ),
]

STUB_TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={url}">
<link rel="canonical" href="{url}">
<title>Redirecting...</title>
</head>
<body>
<p>This page has moved. If you are not redirected automatically, <a href="{url}">click here</a>.</p>
</body>
</html>
"""


def write_stub(path: Path, url: str, lang: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STUB_TEMPLATE.format(lang=lang, url=url), encoding="utf-8")


def validate_redirect_targets(publish_dir: Path) -> None:
    """Ensure every redirect destination exists in the built site."""
    targets = {
        publish_dir / "docs" / target
        for redirect in REDIRECTS
        for target in redirect[2:]
    }
    missing = sorted(
        path.relative_to(publish_dir) for path in targets if not path.is_file()
    )
    if missing:
        missing_paths = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Redirect targets are missing from the built documentation:\n"
            f"{missing_paths}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publish_dir", type=Path, help="gh-pages branch checkout root")
    args = parser.parse_args()

    validate_redirect_targets(args.publish_dir)

    count = 0
    for old_en, old_ja, new_en, new_ja in REDIRECTS:
        write_stub(
            args.publish_dir / "docs/web/en" / old_en,
            SITE_ROOT + "docs/" + new_en,
            "en",
        )
        write_stub(
            args.publish_dir / "docs/web/ja" / old_ja,
            SITE_ROOT + "docs/" + new_ja,
            "ja",
        )
        count += 2

    print(f"Generated {count} redirect stubs under {args.publish_dir}.")


if __name__ == "__main__":
    main()
