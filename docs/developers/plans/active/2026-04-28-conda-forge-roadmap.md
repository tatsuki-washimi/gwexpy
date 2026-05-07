# Conda-forge Release Roadmap and Feedstock Onboarding

**Issue:** #294
**Branch:** `codex/wave4-294-conda-roadmap`
**Date:** 2026-04-28
**Author:** Codex

## Scope

This document records the conda-forge onboarding plan that follows the PyPI
release readiness work in #293 and tracks the live staged-recipes submission for
#294. The initial staged-recipes PR is now open and CI-green, but the package is
not yet live on the `conda-forge` channel. Do not update public conda install
docs until the feedstock exists, the package is published, and a fresh conda
install smoke test passes.

The intended first submission is a single core `gwexpy` conda package. Pip
extras should not be promised as conda install groups in the initial recipe
because conda packages do not expose pip-style extras. Optional feature groups
can be documented later as explicit dependency sets or split outputs after the
core feedstock exists.

## Current Status

Status as of 2026-05-07:

- PyPI `gwexpy==0.1.2` is the next hotfix target and has not yet been
  published or smoke-tested from PyPI.
- The staged-recipes PR is open and mergeable:
  <https://github.com/conda-forge/staged-recipes/pull/33169>.
- The staged-recipes PR was opened from the `0.1.1` PyPI sdist and must be
  refreshed to the `0.1.2` PyPI sdist before conda-forge merge.
- Local validation passed with `conda-smithy recipe-lint`, `rattler-build
  --render-only`, and `rattler-build ... --test native`.
- Remote staged-recipes CI is green: GitHub linter, conda-forge-linter, Azure
  aggregate job, linux_64, osx_64, and win_64 builds.
- `conda-forge/gwexpy-feedstock` does not exist yet.

Current blocker: publish and smoke-test `gwexpy==0.1.2`, then refresh
staged-recipes PR #33169 to the new PyPI sdist before continuing conda-forge
review. After merge, continue with feedstock creation checks, package
publication checks, and fresh conda install smoke testing.

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

## Submission Preconditions And Refresh Gates

The initial staged-recipes submission based on the `gwexpy==0.1.1` PyPI sdist
was opened only after these gates were resolved:

1. #293 produced a stable PyPI/source release suitable for the first
   conda-forge recipe submission.
2. The release version, `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`, and
   `gwexpy/_version.py` were aligned before publication.
3. The initial feedstock maintainer is `tatsuki-washimi`.
4. Remaining audit issues were closed, accepted as known limitations, or
   deferred as post-release follow-ups.

Before the staged-recipes PR can merge, it must still satisfy these
`gwexpy==0.1.2` refresh gates:

1. Refresh the recipe source from the current `0.1.1` PyPI sdist to the
   `0.1.2` PyPI sdist.
2. Record the `0.1.2` sdist SHA256 hash in staged-recipes PR #33169.
3. Rerun local and remote staged-recipes validation after the refresh.

## Initial Recipe Model

Recommended first recipe shape:

```yaml
context:
  name: gwexpy
  version: "0.1.2"
  python_min: "3.11"

package:
  name: ${{ name|lower }}
  version: ${{ version }}

source:
  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-${{ version }}.tar.gz
  sha256: "<pypi-sdist-sha256>"

build:
  number: 0
  noarch: python
  script: python -m pip install . -vv --no-deps --no-build-isolation
  python:
    entry_points:
      - gwexpy = gwexpy.cli:main

requirements:
  host:
    - python ${{ python_min }}.*
    - pip
    - setuptools >=68
    - wheel
  run:
    - python >=${{ python_min }}
    - numpy >=1.23.2,<2.0.0
    - scipy >=1.10.0
    - astropy-base >=5.0
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
      python_version: ${{ python_min }}.*
  - script:
      - python -m pip check
      - gwexpy --help
      - python -c "import gwexpy; gwexpy.register_all()"
      - python -c "import importlib.util; assert importlib.util.find_spec('gwexpy.gui') is None"
      - python -c "import importlib.util; assert importlib.util.find_spec('gwexpy.utils.sphinx') is None"
      - python -c "from gwexpy.timeseries import TimeSeries; ts = TimeSeries([1, 2, 3], dt=1); assert ts.shape == (3,)"
    requirements:
      run:
        - python
        - pip
```

`noarch: python` is the expected starting point because this repository ships
pure Python package code and delegates compiled or binary-heavy components to
dependencies. Confirm this during the staged-recipes build review. For the
recipe tests, keep `pip_check: true` and run the import test against the
minimum supported Python version. The script smoke test should use an
unconstrained `python` test requirement so staged-recipes CI also exercises the
current solver default. If any future release adds compiled extension modules,
revisit the platform model before submitting.

The initial recipe should keep only the core `gwexpy` console script in
`build.python.entry_points`. The first PyPI package does not expose
`gwexpy.gui`, a GUI console script, or a `gui` extra because the GUI app is a
post-release stabilization track. If the GUI is added later, keep it in a
separate output or add the GUI dependencies explicitly.

## Core Dependency Map

The table below mirrors `pyproject.toml` `[project].dependencies`. Keep it in
sync with the source metadata before generating a recipe.

| PyPI dependency | Conda-forge package | Recipe policy | Notes |
| --- | --- | --- | --- |
| `numpy>=1.23.2,<2.0.0` | `numpy` | Preserve lower and upper bounds initially. | Revisit the `<2.0.0` cap only with runtime compatibility evidence. |
| `scipy>=1.10.0` | `scipy` | Preserve lower bound. | Core numerical dependency. |
| `astropy>=5.0` | `astropy-base` | Preserve lower bound while avoiding optional astropy dependencies. | Unit and astronomy foundation. |
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

Completed evidence for the current external PR based on the `gwexpy==0.1.1`
sdist. This evidence does not close the required `gwexpy==0.1.2` refresh and
revalidation listed in the remaining work below.

1. Identified `gwexpy==0.1.2` as the target for the upcoming refresh after
   the PyPI hotfix release is published and smoke-tested.
2. Recorded the PyPI sdist SHA256 hash for the `0.1.1` source used by the
   current staged-recipes PR.
3. Wrote the v1 recipe manually from the model above for the current PR.
4. Reviewed license fields against `LICENSE.txt` and package metadata.
5. Reviewed every dependency against the core dependency map above.
6. Added import, CLI, `gwexpy.register_all()`, `pip check`, GUI/sphinx
   exclusion, and minimal `TimeSeries` smoke tests.
7. Ran staged-recipes local build/lint with `conda-smithy` and `rattler-build`
   against the `0.1.1` PR state.
8. Opened the PR to `conda-forge/staged-recipes`:
   <https://github.com/conda-forge/staged-recipes/pull/33169>.
9. Confirmed remote staged-recipes CI is green for the `0.1.1` PR state.

Remaining staged-recipes work:

1. Refresh staged-recipes PR #33169 from `gwexpy==0.1.1` to the
   `gwexpy==0.1.2` PyPI sdist, record the `0.1.2` SHA256 hash, and rerun
   local plus remote validation.
2. Wait for conda-forge reviewer/maintainer review.
3. Address any requested recipe changes without expanding the first package
   beyond the core, non-GUI surface.
4. Wait for staged-recipes merge.

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

- Reviewer approval and merge timing for
  <https://github.com/conda-forge/staged-recipes/pull/33169>.
- Whether any optional dependencies should become split outputs.
- Whether the `numpy<2.0.0` cap remains necessary for future releases.

## Validation

This PR adds a drift guard that parses `pyproject.toml` and ensures each core
runtime dependency is represented in the roadmap's core dependency table.

Recommended local checks:

```bash
rtk pytest -q tests/test_conda_forge_roadmap.py -p no:cacheprovider
rtk ruff check tests/test_conda_forge_roadmap.py
rtk ruff format --check tests/test_conda_forge_roadmap.py
rtk python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('docs/developers/plans/manifests/audit-manifest-294-conda-forge-staged-recipes-status.yaml').read_text())"
rtk git diff --check
```
