# About

Project information, licensing, and how to cite GWexpy.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` License
:link: license
:link-type: doc

The terms GWexpy is released under.
:::

:::{grid-item-card} {octicon}`quote;1.5em;sd-mr-1` Citation
:link: citation
:link-type: doc

How to cite GWexpy in your research.
:::

:::{grid-item-card} {octicon}`tag;1.5em;sd-mr-1` Changelog
:link: changelog
:link-type: doc

Release notes and version history.
:::

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Contributing
:link: https://github.com/tatsuki-washimi/gwexpy
:link-type: url

Issues and pull requests are welcome on GitHub.
:::
::::

## Contributing

GWexpy welcomes contributions. The short version: fork the repository, create a
branch for your change, and open a pull request against `main`. See
[CONTRIBUTING.md](https://github.com/tatsuki-washimi/gwexpy/blob/main/CONTRIBUTING.md)
for the full workflow, coding conventions, and test expectations, and the
[Code of Conduct](https://github.com/tatsuki-washimi/gwexpy/blob/main/CODE_OF_CONDUCT.md)
for community expectations.

To build this documentation site locally, run `sphinx-build -b html . _build/html`
from the `docs_redesign/` directory. The Japanese translation is delivered via
`gettext` catalogs under `docs_redesign/locales/`; to update them after an
English source change, run `sphinx-build -b gettext . _build/gettext` followed
by `sphinx-intl update -p _build/gettext -l ja`, then translate any new empty
entries in the resulting `.po` files.

```{toctree}
:hidden:

license
citation
changelog
known_limitations
```