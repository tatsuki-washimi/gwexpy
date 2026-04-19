# Command-Line Interface (CLI)

## Overview

The GWexpy CLI is currently a **minimal placeholder interface** and should be treated as a **prototype-stage feature**. At present, GWexpy is intended to be used primarily through the **Python API** for interactive analysis and scripting.

## Current Status

The GWexpy CLI is in **early development / prototype stage**. The implemented behavior is currently limited to version display and a small informational help message. Advanced analysis workflows are not yet exposed as stable GWexpy subcommands, and the interface should not yet be treated as a finalized public workflow surface.

## Available Commands

### `gwexpy --version`

Display the installed GWexpy version:

```bash
gwexpy --version
```

Output:
```
gwexpy 0.1.0
```

### `gwexpy --help`

Show general help information:

```bash
gwexpy --help
```

## Using GWpy CLI

If you need GWpy's command-line tools, refer to the [GWpy Documentation](https://gwpy.readthedocs.io/en/latest/cli/) directly. GWexpy does not currently provide a documented compatibility layer that forwards the full GWpy CLI surface.

**Note:** For complex analysis pipelines, gravitational wave parameter estimation, and custom data processing, the **Python API** is recommended. See the [Getting Started](./getting_started.md) guide for API examples.

## Future Development

Future GWexpy releases will add specialized subcommands for:
- Data ingestion and validation
- Noise characterization
- Time-frequency analysis
- Event localization

For planned features and timeline, see the public [Roadmap](roadmap.md).

## Troubleshooting

If the `gwexpy` command is not found after installation, ensure that `gwexpy` was installed in the correct environment:

```bash
pip install -e .
```

Verify the installation:

```bash
python -c "import gwexpy; print(gwexpy.__version__)"
```
