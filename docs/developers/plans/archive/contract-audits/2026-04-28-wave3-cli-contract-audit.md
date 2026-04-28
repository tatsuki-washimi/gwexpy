# Wave 3 CLI Contract Audit

Date: 2026-04-28
Issue: #280, "Audit CLI command contracts and failure behavior"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #280 slice records deterministic CLI contracts that can be baselined
without invoking real data loading, plotting, network access, file creation, or
heavy GWpy command workflows:

- `gwexpy.cli.main()` version flag, placeholder banner, and unknown-command
  behavior;
- stdout, stderr, return-value, and exit-code behavior for the current
  placeholder command;
- representative GWpy CLI module re-export identity boundaries;
- the `gwexpy.cli.__all__` package export contract.

This slice does not change runtime CLI behavior. Behavior changes to command
dispatch, parser aliases, stderr handling, file writes, optional dependency
messages, or output schemas remain human-review items because they affect the
public command-line contract.

## Contracts Recorded

### Placeholder Entry Point

- `main(["--version"])` and `main(["-v"])` return `None`, write exactly
  `gwexpy <version>` plus a newline to stdout, and leave stderr empty.
- `main([])` and `main(["--help"])` return `None`, write the current
  placeholder informational banner to stdout, and leave stderr empty.
- The placeholder banner identifies GWexPy as an experimental gravitational
  wave analysis tool, reports the package version, and lists `spectrogram` and
  `spectrum` as planned subcommands.
- `main(["unknown"])` raises `SystemExit(1)`, writes
  `gwexpy: unknown command or option 'unknown'` plus a newline to stdout, and
  leaves stderr empty. This test records the current stdout behavior; it does
  not move errors to stderr.

### GWpy Re-Export Module Boundary

- `gwexpy.cli.gwpy_plot.main`, `create_parser`, and `parse_command_line` are
  the same objects as the corresponding `gwpy.cli.gwpy_plot` exports.
- Representative product classes are direct GWpy object re-exports:
  `SpectrumProduct`, `SpectrogramProduct`, and `QtransformProduct`.
- These identity tests intentionally avoid asserting complete GWpy parser
  behavior, plotting behavior, file output behavior, or deep command workflows.

### Package Export Contract

- `gwexpy.cli.__all__` currently contains only `["main", "__version__"]`.
- The package imports GWpy product names for compatibility, but those names are
  not part of the explicit star-import export list in this baseline.

## Stable Versus Experimental Surfaces

The stable surface recorded in this slice is narrow: the current placeholder
entry point, its exact stream/exit behavior, and the direct GWpy re-export
identity boundary for representative modules/classes.

The broader GWexPy CLI remains experimental. The current top-level `gwexpy`
command is a minimal placeholder, not a documented full GWpy CLI forwarding
layer. Subcommand modules proxy GWpy CLI objects today, but this audit does not
declare the complete GWpy command parser, aliases, defaults, file creation
behavior, overwrite policy, optional dependency handling, or output formatting
as stable GWexPy contracts.

## Docs Alignment

Existing user-facing CLI documentation describes the GWexPy CLI as minimal,
placeholder-like, and experimental. That aligns with the current implementation
and with this slice's test baseline. The docs do not promise complete GWpy CLI
forwarding, and this slice does not add that promise.

Future user docs should be updated only after humans decide which command names,
argument aliases, exit codes, stderr/stdout separation, file-output behavior,
optional dependency messages, and output formats are intentionally stable.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Add a unified GWexPy command dispatcher or forward top-level subcommands to
  GWpy products.
- Move unknown-command messages from stdout to stderr or change their wording.
- Change exit codes, return values, version aliases, or help/banner text.
- Define complete command names, argument aliases, parser defaults, or
  deprecation policy.
- Define machine-readable output schemas or structured logging behavior.
- Define file creation, overwrite, and dry-run contracts.
- Define optional dependency failure text, install hints, or backend fallback
  messages.
- Validate docs examples against complete parser behavior.
- Reclassify experimental/developer-facing CLI modules as stable public CLI
  commands.

## Follow-Up Slices For #280

1. Command inventory: enumerate top-level command names, GWpy proxy modules,
   aliases, and explicit unsupported commands.
2. Parser contract: compare documented examples against parser behavior,
   including defaults, deprecations, and help text.
3. Failure contract: decide stdout versus stderr, exit codes, error wording,
   optional dependency messages, and install hints.
4. Output contract: define text, image, table, and machine-readable output
   expectations without running heavy data workflows in unit tests.
5. File-system contract: audit file creation, overwrite behavior, output naming,
   temporary files, and dry-run expectations.
6. Stability policy: label commands and modules as stable, experimental, or
   developer-facing, with a human-reviewed change policy.

## Verification

Focused contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/cli/test_cli_contracts.py -p no:cacheprovider
```

Focused CLI regression check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/cli/test_cli_contracts.py tests/cli/test_main.py \
  tests/cli/test_gwpy_plot.py -p no:cacheprovider
```

Changed-file hygiene:

```bash
rtk ruff check tests/cli/test_cli_contracts.py
rtk ruff format --check tests/cli/test_cli_contracts.py
rtk python -c "import yaml; from pathlib import Path; yaml.safe_load(Path('docs/developers/plans/audit-manifest-280-cli-contracts.yaml').read_text())"
rtk git diff --check
```
