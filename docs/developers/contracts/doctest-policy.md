# Raw doctest policy

The raw module-doctest command remains the executable documentation baseline:

```text
PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python -m pytest --doctest-modules -q gwexpy/
```

Every example included in that baseline belongs to one of four categories.

### 1. Executable public examples

An example that is complete, deterministic, and presents a public API usage
path is executable documentation. It must pass under the supported environment.
Its output is part of the example contract and must be updated precisely when
the current public representation changes.

### 2. Illustrative fragments

A fragment that depends on application-owned objects, omitted setup, a live
session, or a result introduced by surrounding prose is illustrative rather
than executable. Mark it locally as **not a doctest** by rendering it as a
Python code block instead of using `>>>`, and state the local reason in the
docstring. An illustrative fragment must not be silently converted into a
module-wide skip.

### 3. Optional/external examples

Examples requiring an optional package, external service, detector data, or an
external file are optional/external. They are outside this lane. Their
availability belongs to the relevant optional or integration gate; this raw
doctest lane does not install dependencies or pretend that those examples are
core documentation. Render their usage as a reasoned `.. code-block:: python`
with no `>>>` prompts or `+SKIP` directives; no `+SKIP` is permitted in source.
State the package/file prerequisite plus the dedicated tests in the relevant
optional or integration gate that cover the boundary. Deterministic public
setup may remain executable before that block,
but the raw lane must not claim that optional integration was executed.

### 4. Stale repr/output and API/semantic examples

A complete example that fails because its repr/output, API, or semantics no
longer match the current implementation must be investigated against existing
tests and public API contracts. Correct stale repr/output exactly. A production
behavior change requires a failing regression test before implementation; a
documentation-only correction is preferred when the implementation is already
the intended behavior.

## Collection guardrails

The raw command must continue to collect `gwexpy/` normally. No blanket
exclusion is allowed: do not add a global pytest exclusion, a module-wide skip,
an `--ignore gwexpy` option, or a collection hook that hides doctests. Any
non-executable example must have a local documented reason and remain visible
in the rendered documentation. Repository source must contain no raw
`+SKIP` directive; local non-doctest code blocks are the required rendering for
optional, external, and schematic fragments.
