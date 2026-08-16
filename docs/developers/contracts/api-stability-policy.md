# API stability policy

[日本語版](api-stability-policy-ja.md)

This policy defines the stability labels used for the GWexpy public API.
It defines the meaning of a label; it does not make all of GWexpy stable or classify every API.

## Scope and labels

Apply a label at the smallest useful scope: a module, a symbol, or a behavior.
Record the label in the relevant API documentation and in release notes when a label is introduced, changed, or removed.

| Label | Definition and guarantee |
|---|---|
| **stable** | Public API. No breaking change without a documented deprecation notice and window, except explicitly documented emergency correctness or security cases. |
| **provisional** | Shipped and supported enough to use, but may change in a patch or minor release without a deprecation cycle. Every change still requires release-note and migration disclosure. |
| **experimental** | Explicit opt-in or research surface. It may change or be removed at any time. No compatibility promise. |

An unlabeled legacy API carries no implied new promise.
Its existing behavior must not be inferred to be stable, provisional, or experimental.

## Release outcomes

Deferred is a release outcome, not a stability tier or API classification.
It records that work did not ship in the release; it does not classify an API.
Deferred is used for a release decision, such as work postponed beyond the current release.
It must not be used as a fourth compatibility promise or as a substitute for an API stability label.

## Label changes

Graduation moves a surface to a stronger guarantee only after its contract and supporting evidence are documented.
Demotion records the reason, updates the API documentation, and discloses the migration impact in release notes.

No feature moves to stable until the corresponding contract audit is closed, or is explicitly deferred with recorded rationale and evidence.

Provisional and experimental labels do not waive correctness, security, or other applicable safety obligations.
The labels describe compatibility expectations; they do not remove the obligation to handle failures and unsafe conditions responsibly.
