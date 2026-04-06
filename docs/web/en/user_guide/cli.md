# Command-Line Interface (CLI)

## Overview

The GWexpy CLI provides command-line access to GWpy's pipeline functionality. Currently, GWexpy primarily uses the **Python API** for interactive analysis and scripting. The CLI serves as a lightweight frontend to common workflows.

## Current Status

The GWexpy CLI is in **active development**. Some commands are available, but most advanced analysis workflows are best performed using the **Python API**.

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

GWexpy re-exports GWpy's command-line tools. For detailed documentation on GWpy's CLI, see the [GWpy Documentation](https://gwpy.readthedocs.io/en/latest/cli/).

**Note:** For complex analysis pipelines, gravitational wave parameter estimation, and custom data processing, the **Python API** is recommended. See the [Getting Started](./getting_started.md) guide for API examples.

## Future Development

Future GWexpy releases will add specialized subcommands for:
- Data ingestion and validation
- Noise characterization
- Time-frequency analysis
- Event localization

For planned features and timeline, see the [Roadmap](https://github.com/tatsuki-washimi/gwexpy/issues).

## Troubleshooting

If the `gwexpy` command is not found after installation, ensure that `gwexpy` was installed in the correct environment:

```bash
pip install -e ".[gw]"  # Install with gravitational wave extras
```

Verify the installation:

```bash
python -c "import gwexpy; print(gwexpy.__version__)"
```
