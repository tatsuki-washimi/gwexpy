# Conda-forge Release Roadmap and Feedstock Onboarding

**Issue:** #294
**Branch:** `codex/wave4-294-conda-roadmap`
**Date:** 2026-04-28
**Author:** Codex

## Scope

This pass records the conda-forge onboarding plan that should follow the PyPI
release readiness work in #293. It does not publish a release, open a
`conda-forge/staged-recipes` PR, edit public install docs, or change package
runtime behavior.

The intended first submission is a single core `gwexpy` conda package. Pip
extras should not be promised as conda install groups in the initial recipe
because conda packages do not expose pip-style extras. Optional feature groups
can be documented later as explicit dependency sets or split outputs after the
core feedstock exists.

## External Policy Snapshot

Consulted sources on 2026-04-28:

- `conda-forge/staged-recipes` README:
  <https://github.com/conda-forge/staged-recipes>
- conda-forge feedstock model:
  <https://conda-forge.org/docs/maintainer/understanding_conda_forge/feedstocks/>
- conda-forge package maintenance guidance:
  <https://conda-forge.org/docs/maintainer/updating_pkgs/>
- conda-forge rerender guidance:
  <https://conda-forge.org/docs/how-to/basics/rerender/>
- conda-forge noarch Python knowledge base:
  <https://conda-forge.org/docs/maintainer/knowledge_base/#noarch-python>
- conda-forge first-recipe tutorial:
  <https://conda-forge.org/docs/tutorials/first-recipe/>

Important constraints from those sources:

- new packages start as a recipe under `conda-forge/staged-recipes/recipes`;
- a merged staged-recipes PR creates the feedstock and triggers the first build
  and upload;
- recipes may be generated with `grayskull`, but license, dependencies, tests,
  and metadata still require human review;
- conda-forge packages are immutable after upload, so release and build numbers
  must be deliberate;
- feedstock changes should normally be proposed through fork PRs and rerendered
  when recipe or feedstock configuration changes.

## Preconditions

Do not begin the staged-recipes submission until these gates are resolved:

1. #293 has produced a stable PyPI/source release or maintainers explicitly
   choose a source-archive-first conda submission.
2. The release version, `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`, and
   `gwexpy/_version.py` agree.
3. The source archive used by the recipe has a recorded SHA256 hash.
4. Maintainers confirm the initial feedstock maintainer list.
5. Remaining audit issues are closed, accepted as known limitations, or deferred
   as post-release follow-ups.

## Initial Recipe Model

Recommended first recipe shape:

```yaml
context:
  name: gwexpy
  version: "<released-version>"

package:
  name: ${{ name|lower }}
  version: ${{ version }}

source:
  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-${{ version }}.tar.gz
  sha256: "<pypi-sdist-sha256>"

build:
  noarch: python
  script: ${{ PYTHON }} -m pip install . -vv --no-deps --no-build-isolation
  number: 0
  entry_points:
    - gwexpy = gwexpy.cli:main

requirements:
  host:
    - python >=3.11
    - pip
    - setuptools >=68
    - wheel
  run:
    - python >=3.11
    - numpy >=1.23.2,<2.0.0
    - scipy >=1.10.0
    - astropy >=5.0
    - gwpy >=4.0.0,<5.0.0
    - pandas >=1.5.0
    - matplotlib-base >=3.5.0
    - typing-extensions
    - bottleneck
    - h5py
    - igwn-segments
    - ligotimegps
    - gpstime

tests:
  - python:
      imports:
        - gwexpy
      pip_check: true
      python_version:
        - "3.11.*"
        - "*"
  - script:
      - gwexpy --help
      - python -c "import gwexpy; gwexpy.register_all()"
```

`noarch: python` is the expected starting point because this repository ships
pure Python package code and delegates compiled or binary-heavy components to
dependencies. Confirm this during the staged-recipes build review. For the
recipe tests, keep `pip_check: true` and run the import test against the
minimum supported Python version plus the latest supported line that the
feedstock will exercise. In the current roadmap draft, that means `3.11.*` and
`*` in the conda-forge recipe syntax. If any future release adds compiled
extension modules, revisit the platform model before submitting.

The initial recipe should keep only the core `gwexpy` console script in
`build.entry_points`. The first PyPI package does not expose `gwexpy.gui` as a
console script because the GUI app is a post-release stabilization track. If the
GUI is added later, keep it in a separate output or add the GUI dependencies
explicitly.

## Core Dependency Map

The table below mirrors `pyproject.toml` `[project].dependencies`. Keep it in
sync with the source metadata before generating a recipe.

| PyPI dependency | Conda-forge package | Recipe policy | Notes |
| --- | --- | --- | --- |
| `numpy>=1.23.2,<2.0.0` | `numpy` | Preserve lower and upper bounds initially. | Revisit the `<2.0.0` cap only with runtime compatibility evidence. |
| `scipy>=1.10.0` | `scipy` | Preserve lower bound. | Core numerical dependency. |
| `astropy>=5.0` | `astropy` | Preserve lower bound. | Unit and astronomy foundation. |
| `gwpy>=4.0.0,<5.0.0` | `gwpy` | Preserve major-version cap. | GWpy 4.x is available as a noarch conda-forge package. |
| `pandas>=1.5.0` | `pandas` | Preserve lower bound. | DataFrame/table interoperability. |
| `matplotlib>=3.5.0` | `matplotlib-base` | Prefer `matplotlib-base` unless GUI backends are required. | Avoid pulling GUI stack into the core package. |
| `typing_extensions` | `typing-extensions` | Use conda package spelling. | Runtime typing backport. |
| `bottleneck` | `bottleneck` | Keep unpinned unless conda-forge lint suggests otherwise. | Optional acceleration used as a core dependency today. |
| `h5py` | `h5py` | Keep unpinned unless build/test evidence requires a floor. | HDF5 I/O foundation. |
| `igwn-segments` | `igwn-segments` | Use same package spelling. | Present on conda-forge. |
| `ligotimegps` | `ligotimegps` | Use same package spelling. | Present on conda-forge. |
| `gpstime` | `gpstime` | Use same package spelling. | Present on conda-forge. |

## Optional Dependency Policy

Initial conda-forge packaging should not mirror `gwexpy[all]`.

- Keep the first recipe focused on `project.dependencies`.
- Do not create split outputs until maintainers choose a concrete user story for
  optional conda installs.
- For binary-heavy GW dependencies such as NDS2/FrameLIB, document explicit
  conda installation guidance after the core package is live.
- Keep public docs saying that conda-forge install is unavailable until the
  package exists on the `conda-forge` channel.

Follow-up candidates after the core feedstock exists:

- optional dependency table for conda users;
- fresh conda environment smoke test job;
- public docs update that distinguishes `pip install gwexpy` from
  `conda install -c conda-forge gwexpy`;
- optional split-output decision record if users need conda-native feature
  bundles.

## Staged-Recipes Checklist

Before opening the external PR:

1. Build the PyPI sdist and wheel for the target release.
2. Record the PyPI sdist SHA256 hash.
3. Generate a candidate recipe with
   `grayskull pypi --use-v1-format --strict-conda-forge -o recipes gwexpy`
   or write the recipe manually from the model above.
4. Review license fields against `LICENSE` and package metadata.
5. Review every dependency against the core dependency map above.
6. Add import and CLI tests: `import gwexpy`, `gwexpy.register_all()`, and
   `gwexpy --help`.
7. Add one minimal data-structure smoke test only if it does not require network,
   optional backends, or test-only data files.
8. Run staged-recipes local build/lint when tooling is available.
9. Open the PR to `conda-forge/staged-recipes` and request Python review.

## Feedstock Maintenance Checklist

After staged-recipes merges:

1. Confirm that `gwexpy-feedstock` is created under `conda-forge`.
2. Confirm the first package upload appears on the `conda-forge` channel.
3. Run a fresh environment smoke test:
   `conda create -n gwexpy-release -c conda-forge gwexpy`.
4. Verify import, registry bootstrap, and CLI help in that environment.
5. Update public install docs only after the fresh conda install succeeds.
6. For future releases, update version, source URL, hash, and build number; reset
   build number to `0` when the upstream version changes.
7. Rerender the feedstock after recipe or feedstock configuration changes.

## Deferred Human Decisions

- Target release version and date.
- Whether the first conda submission must wait for a PyPI source distribution or
  can use a tagged GitHub source archive.
- Initial feedstock maintainers.
- Whether any optional dependencies should become split outputs.
- Whether the `numpy<2.0.0` cap remains necessary for the release being packaged.

## Validation

This PR adds a drift guard that parses `pyproject.toml` and ensures each core
runtime dependency is represented in the roadmap's core dependency table.

Recommended local checks:

```bash
rtk pytest -q tests/test_conda_forge_roadmap.py -p no:cacheprovider
rtk ruff check tests/test_conda_forge_roadmap.py
rtk ruff format --check tests/test_conda_forge_roadmap.py
rtk python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('docs/developers/plans/audit-manifest-294-conda-forge.yaml').read_text())"
rtk git diff --check
```
