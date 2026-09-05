# Developer guide

Start here to change GWexpy or its documentation. For an analysis workflow,
use [Start Here](../tutorials/getting_started.md).

| Task | Source of guidance |
|---|---|
| Set up a checkout and submit a change | [Contributing](https://github.com/tatsuki-washimi/gwexpy/blob/main/CONTRIBUTING.md) |
| Understand containers and processing layers | [Architecture](../explanation/architecture.md) |
| Review numerical and metadata behavior | [GWpy compatibility policy](../explanation/gwpy_compatibility_policy.md) |
| Run validation | [Verification and quality](../explanation/verification_and_quality.md) |
| Find internal contracts and design decisions | [Developer sources](https://github.com/tatsuki-washimi/gwexpy/tree/main/docs/developers) |

## Build the public documentation

Use the project's `gwexpy` Conda environment with the documentation dependencies.
From a source checkout, prepare a fresh temporary directory outside the repository:

```bash
python scripts/prepare_public_docs.py /tmp/gwexpy-docs-preview/docs_redesign
python -m sphinx -b html /tmp/gwexpy-docs-preview/docs_redesign /tmp/gwexpy-docs-preview/html
python -m sphinx -b html -D language=ja /tmp/gwexpy-docs-preview/docs_redesign /tmp/gwexpy-docs-preview/html/ja
python scripts/check_public_docs.py /tmp/gwexpy-docs-preview/html
```

Use a new output directory on subsequent runs. Preparation reads canonical
notebook code and keeps execution output outside the checkout. English public
prose and Japanese gettext catalogs are maintained together. When changing a
notebook's executable cells, edit its canonical source and run the preparation
step; the public notebook supplies the narrative and translation identities.

## Studio

GWexpy Studio is a separate interface under development. It is not required to
complete these tutorials. A Studio learning path will be added after its
distribution and basic workflows can be reproduced. Until then, the downloadable
Python examples are the supported entry to the learning material.
