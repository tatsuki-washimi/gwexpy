# GUI applications are maintained separately

GWexpy is an analysis library. It no longer contains the legacy `gwexpy.gui`
application in the source tree or in its distributions.

The earlier experimental interface was split into the independent
[pyaggui repository](https://github.com/tatsuki-washimi/pyaggui).
GWexpy Studio is a separate GUI application maintained in a separate repository;
it should not be confused with pyaggui or represented as part of GWexpy.

For programmatic signal processing, I/O, statistics and plotting, use
the supported GWexpy Python API. See the {doc}`main documentation <../index>`
for current examples.

This page remains as a migration pointer for links to older GWexpy GUI guides.
Historical documentation for the pre-split implementation is preserved in Git history.
