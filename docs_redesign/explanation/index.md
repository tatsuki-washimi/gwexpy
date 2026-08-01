# Explanation

Understanding-oriented discussion of the ideas, design choices, and trade-offs
behind GWexpy. These pages are for reading, not for following step by step.

## Design and concepts

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Architecture & data flow
:link: architecture
:link-type: doc

How the container, analysis, and interoperability layers fit together.
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Prerequisites & conventions
:link: prerequisites_and_conventions
:link-type: doc

The assumptions, naming, and conventions GWexpy relies on.
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` GWexpy for GWpy Users
:link: gwexpy_for_gwpy_users
:link-type: doc

How GWexpy relates to, and differs from, GWpy.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem positioning
:link: ecosystem
:link-type: doc

Where GWexpy sits among GWpy, spicypy, GWDama, and the wider GW Python stack.
:::

:::{grid-item-card} {octicon}`milestone;1.5em;sd-mr-1` Roadmap
:link: roadmap
:link-type: doc

Where the project is heading.
:::
::::

## Theory and quality

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Physics models & theory
:link: physics_models
:link-type: doc

The mathematical foundations behind the analysis algorithms.
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Numerical stability
:link: numerical_stability
:link-type: doc

Where precision matters and how GWexpy protects it.
:::

:::{grid-item-card} {octicon}`verified;1.5em;sd-mr-1` Validated algorithms
:link: validated_algorithms
:link-type: doc

Validation assumptions and the evidence behind key algorithms.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Verification & quality
:link: verification_and_quality
:link-type: doc

How GWexpy is tested and kept correct.
:::
::::

```{toctree}
:hidden:
:caption: Design and concepts

architecture
prerequisites_and_conventions
gwexpy_for_gwpy_users
ecosystem
roadmap
```

```{toctree}
:hidden:
:caption: Theory and quality

physics_models
numerical_stability
validated_algorithms
verification_and_quality
```
