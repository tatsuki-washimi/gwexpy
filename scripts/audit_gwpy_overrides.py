#!/usr/bin/env python3
"""Build and validate the source/MRO GWpy override inventory for issue #639."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Any, get_args, get_origin

SCHEMA = "gwexpy-v023-gwpy-override-inventory-v1"
WORKER_SCHEMA = "gwexpy-v023-gwpy-override-oracle-v1"
SUPPORTED_GWPY = ("4.0.1", "4.0.2")
TERMINAL_STATES = ("fixed", "no-finding", "GWpy-fails", "GWexpy-only")
PROVISIONAL_STATES = ("unreviewed", "differential-required")
BEHAVIORAL_TERMINAL_STATES = frozenset({"fixed", "no-finding", "GWpy-fails"})
CONSTRUCTORS = frozenset({"__init__", "__new__"})
INTERNAL_CLASS_TOKENS = (
    "Mixin",
    "Base",
    "Core",
    "Protocol",
    "Interface",
    "MetaData",
)
ABSENT_FIXTURE = "__counterpart_absent__"
PENDING_FIXTURE = "__pending_differential__"
CASE_KEYS = frozenset(
    {
        "case_key",
        "comparator",
        "counterpart_present",
        "evidence",
        "fixture",
        "gwpy_version",
        "implementation_group",
        "issues",
        "member",
        "member_id",
        "observations",
        "owner",
        "public_class",
        "state",
    }
)
OPTIONAL_CASE_KEYS = frozenset({"compatibility_exception"})
NON_INTERSECTING_WINDOW_SAFETY = "non_intersecting_window_safety"
ALLOWED_COMPATIBILITY_EXCEPTIONS = frozenset({NON_INTERSECTING_WINDOW_SAFETY})
TIMESERIESDICT_READ_MEMBER_ID = "gwexpy.timeseries.collections.TimeSeriesDict/read"
TIMESERIESDICT_READ_PARITY_REFERENCE = (
    "tests/io/test_gwpy_override_terminal_io.py::test_timeseriesdict_read_matches_gwpy"
)
HDF5_AUTO_READ_REFERENCE = (
    "tests/io/test_hdf5_timeseries_family.py::test_native_hdf5_auto_read_matches_gwpy"
)
HDF5_AUTO_WRITE_REFERENCE = (
    "tests/io/test_hdf5_timeseries_family.py::test_native_hdf5_auto_write_matches_gwpy"
)
HDF5_AUTO_DICT_APPEND_REFERENCE = (
    "tests/io/test_hdf5_timeseries_family.py"
    "::test_timeseriesdict_auto_write_reconciles_existing_manifest"
)
HDF5_AUTO_ROUTE_REFERENCES = {
    "gwexpy.timeseries.timeseries.TimeSeries/read": (HDF5_AUTO_READ_REFERENCE,),
    "gwexpy.timeseries.timeseries.TimeSeries/write": (HDF5_AUTO_WRITE_REFERENCE,),
    "gwexpy.timeseries.collections.TimeSeriesDict/read": (HDF5_AUTO_READ_REFERENCE,),
    "gwexpy.timeseries.collections.TimeSeriesDict/write": (
        HDF5_AUTO_WRITE_REFERENCE,
        HDF5_AUTO_DICT_APPEND_REFERENCE,
    ),
}
NON_INTERSECTING_WINDOW_SAFETY_REFERENCE = (
    "tests/io/test_reader_start_end_contract.py"
    "::TestNonNanosecondGrids"
    "::test_window_before_data_with_non_ns_epoch_returns_metadata_empty"
)
NON_INTERSECTING_WINDOW_SAFETY_OBSERVATIONS = {
    "gwexpy": {
        "detail": (
            "GWexpy returns a zero-length key-preserving result for the dedicated "
            "non-intersecting native HDF5 window."
        ),
        "outcome": "return",
    },
    "gwpy": {
        "detail": (
            "GWpy 4.0.x returns nonempty outside-window samples for the dedicated "
            "non-intersecting native HDF5 window."
        ),
        "outcome": "return",
    },
}
AUDIT_OWNER = "v0.2.3-compatibility-audit"
IMPLEMENTATION_BASE = "a8085b71446d3ef3417a7e5b5ac8efb156368eac"
PUBLIC_ROOT_RULE = (
    "byte-sorted gwexpy Python paths; final literal top-level list/tuple __all__; "
    "all-source top-level class route index plus static vars(module) resolution; "
    "two-pass unique canonical-class-name lazy alias association; canonical GWexpy "
    "class identity; internal root exclusions"
)
MEMBER_WALK_RULE = (
    "first effective vars(owner) binding in the GWexpy MRO prefix before "
    "the first GWpy class; public callable/descriptors plus __new__/__init__"
)
PRISTINE_ORACLE_RULE = (
    "separate -I worker; sanitized PYTHONPATH/PYTHONHOME; no GWexpy import; "
    "exact GWpy 4.0.1/4.0.2"
)
UPSTREAM_DEPENDENCY_PROVENANCE = (
    "GWpy providers retain package-relative source/line; inherited "
    "NumPy/Astropy providers retain normalized provider, member, kind, "
    "descriptor, and signature without source path or resolved version. "
    "Matplotlib UNSET is normalized by singleton identity; five NumPy ndarray "
    "C descriptors bind exact reviewed unavailable/available signature variants."
)
EVIDENCE_TIMEOUT_SECONDS = 300
ORACLE_TIMEOUT_SECONDS = 120
SUBPROCESS_OUTPUT_LIMIT = 2_000_000
EXPECTED_EVIDENCE_CASES = 396
EVIDENCE_CHILD_ENV = "GWEXPY_OVERRIDE_EVIDENCE_CHILD"
SELF_EVIDENCE_PATH = "tests/test_gwpy_override_inventory.py"
SAFE_SELF_EVIDENCE_SELECTORS = frozenset(
    {
        "test_terminal_array_family_T_matches_gwpy",
        "test_terminal_plot_show_lifecycle_matches_gwpy",
    }
)


class InventoryError(RuntimeError):
    """Raised when inventory input or evidence fails closed validation."""


def _terminal_closure(
    state: str,
    fixture: str,
    reference: str,
    *,
    issues: Sequence[str] = ("#639", "#704"),
    pre_fix: str | None = None,
    comparator: str = "exact-result-and-metadata",
    observations: Mapping[str, Mapping[str, Any]] | None = None,
    pre_fix_observations: Mapping[str, Mapping[str, Any]] | None = None,
    additional_behavior: Sequence[str] = (),
    compatibility_exception: str | None = None,
) -> dict[str, Any]:
    """Create one immutable-by-convention terminal-evidence specification."""

    if state == "fixed" and not pre_fix:
        raise InventoryError(f"fixed closure lacks pre-fix evidence: {fixture}")
    if observations is None:
        observations = {
            "gwexpy": {"outcome": "return"},
            "gwpy": {"outcome": "return"},
        }
    if state == "fixed" and pre_fix_observations is None:
        pre_fix_observations = {
            "gwexpy": {"detail": pre_fix, "outcome": "mismatch"},
            "gwpy": {"outcome": "return"},
        }
    closure: dict[str, Any] = {
        "behavior": (reference, *additional_behavior),
        "comparator": comparator,
        "fixture": fixture,
        "issues": tuple(issues),
        "observations": copy.deepcopy(observations),
        "pre_fix": pre_fix,
        "pre_fix_observations": copy.deepcopy(pre_fix_observations),
        "state": state,
    }
    if compatibility_exception is not None:
        closure["compatibility_exception"] = compatibility_exception
    return closure


def _build_terminal_closures() -> dict[str, dict[str, Any]]:
    """Return the reviewed one-fixture closure for every GWpy-present row.

    Entries are deliberately registered by exact logical ``member_id`` rather
    than inferred from method names.  A newly exported class or override is
    therefore left provisional until it receives its own reviewed evidence.
    """

    closures: dict[str, dict[str, Any]] = {}

    def register(
        public_classes: str | Sequence[str],
        members: str | Sequence[str],
        closure: dict[str, Any],
    ) -> None:
        classes = (
            (public_classes,) if isinstance(public_classes, str) else public_classes
        )
        names = (members,) if isinstance(members, str) else members
        for public_class in classes:
            for member in names:
                member_id = f"{public_class}/{member}"
                if member_id in closures:
                    raise InventoryError(f"duplicate terminal closure: {member_id}")
                closures[member_id] = closure

    array_family = (
        "gwexpy.fields.scalar.ScalarField",
        "gwexpy.types.array.Array",
        "gwexpy.types.array3d.Array3D",
        "gwexpy.types.array4d.Array4D",
    )
    array2d_family = (
        "gwexpy.types.array2d.Array2D",
        "gwexpy.types.plane2d.Plane2D",
    )
    statistics_classes = (
        "gwexpy.fields.scalar.ScalarField",
        "gwexpy.frequencyseries.frequencyseries.FrequencySeries",
        "gwexpy.timeseries.timeseries.TimeSeries",
        "gwexpy.types.array.Array",
        "gwexpy.types.array2d.Array2D",
        "gwexpy.types.array3d.Array3D",
        "gwexpy.types.array4d.Array4D",
        "gwexpy.types.plane2d.Plane2D",
        "gwexpy.types.series.Series",
    )

    register(
        array_family,
        "__new__",
        _terminal_closure(
            "fixed",
            "gwpy-common-keyword-constructor",
            "tests/test_gwpy_constructor_terminal_compat.py::test_array_family_common_keyword_route_matches_gwpy",
            pre_fix=(
                "GWexpy constructor extensions occupied or hid GWpy constructor "
                "slots and rejected parent-supported calling forms."
            ),
            comparator="exact-constructor-result-and-binding",
        ),
    )
    register(
        array2d_family,
        "__new__",
        _terminal_closure(
            "fixed",
            "gwpy-full-positional-constructor-prefix",
            "tests/test_gwpy_constructor_terminal_compat.py::test_array2d_family_full_positional_prefix_matches_gwpy",
            pre_fix=(
                "GWexpy axis-name extensions occupied GWpy positional constructor "
                "slots and hid the remaining parent prefix."
            ),
            comparator="exact-constructor-result-and-binding",
        ),
    )
    register(
        "gwexpy.frequencyseries.frequencyseries.FrequencySeries",
        "__new__",
        _terminal_closure(
            "no-finding",
            "common-frequencyseries-constructor",
            "tests/test_gwpy_constructor_terminal_compat.py::test_frequencyseries_common_constructor_matches_gwpy",
            comparator="exact-constructor-result-and-binding",
        ),
    )
    register(
        "gwexpy.timeseries.timeseries.TimeSeries",
        "__new__",
        _terminal_closure(
            "fixed",
            "complete-gwpy-timeseries-constructor-prefix",
            "tests/timeseries/test_constructor_compat.py::test_constructor_every_gwpy_positional_prefix_matches",
            pre_fix=(
                "The override hid the GWpy positional prefix and normalized some "
                "parent-supported epochs before dispatch."
            ),
            comparator="exact-constructor-result-and-binding",
        ),
    )
    register(
        ("gwexpy.plot.plot.Plot", "gwexpy.plot.field.FieldPlot"),
        "__init__",
        _terminal_closure(
            "no-finding",
            "data-bearing-plot-constructor",
            "tests/test_gwpy_constructor_terminal_compat.py::test_plot_family_data_constructor_matches_gwpy",
            comparator="exact-constructor-result-and-binding",
        ),
    )
    register(
        "gwexpy.plot.skymap.SkyMap",
        "__init__",
        _terminal_closure(
            "fixed",
            "data-bearing-skymap-constructor",
            "tests/plot/test_skymap_gwpy_constructor_compat.py::test_skymap_accepts_parent_timeseries_constructor_surface",
            pre_fix=(
                "SkyMap injected an all-sky projection into a GWpy-valid "
                "data-bearing constructor and raised AttributeError."
            ),
            comparator="exact-constructor-result-and-binding",
        ),
    )

    register(
        statistics_classes,
        ("max", "mean", "median", "min", "std", "var"),
        _terminal_closure(
            "fixed",
            "finite-and-nonfinite-default-reduction",
            "tests/types/test_gwpy_stats_compat.py::test_shared_statistics_default_matches_gwpy_nonfinite_behavior",
            pre_fix=(
                "Shared reductions changed GWpy default non-finite behavior, "
                "result metadata, or public calling forms."
            ),
            comparator="exact-reduction-values-masks-and-metadata",
        ),
    )

    register(
        array_family,
        "T",
        _terminal_closure(
            "no-finding",
            "reverse-axis-property",
            "tests/test_gwpy_override_inventory.py::test_terminal_array_family_T_matches_gwpy",
            comparator="exact-axis-permutation-result",
        ),
    )
    register(
        array_family,
        "swapaxes",
        _terminal_closure(
            "fixed",
            "numeric-axis-swap",
            "tests/types/test_gwpy_override_terminal_compat.py::test_axis_api_numeric_swapaxes_matches_gwpy",
            pre_fix=(
                "The shared axis route copied same-axis results, rejected NumPy "
                "integer axes, and raised IndexError where GWpy raises AxisError."
            ),
            comparator="exact-axis-permutation-result",
        ),
    )
    register(
        array_family,
        "transpose",
        _terminal_closure(
            "fixed",
            "numeric-axis-transpose",
            "tests/types/test_gwpy_override_terminal_compat.py::test_axis_api_numeric_transpose_matches_gwpy",
            pre_fix=(
                "The shared axis route rejected GWpy-supported None and sequence "
                "arguments and changed identity, NumPy-integer, or failure outcomes."
            ),
            comparator="exact-axis-permutation-result",
        ),
    )
    register(
        array2d_family,
        "T",
        _terminal_closure(
            "no-finding",
            "array2d-transpose-property",
            "tests/types/test_gwpy_override_terminal_compat.py::test_array2d_T_keeps_gwpy_swapped_metadata_contract",
            comparator="exact-axis-permutation-result",
        ),
    )
    register(
        array2d_family,
        "swapaxes",
        _terminal_closure(
            "fixed",
            "array2d-numeric-axis-swap",
            "tests/types/test_gwpy_override_terminal_compat.py::test_array2d_numeric_swapaxes_matches_gwpy",
            pre_fix=(
                "Numeric Array2D-family permutations rewrote axis metadata instead "
                "of preserving the installed GWpy result."
            ),
            comparator="exact-axis-permutation-result",
        ),
    )
    register(
        array2d_family,
        "transpose",
        _terminal_closure(
            "fixed",
            "array2d-numeric-axis-transpose",
            "tests/types/test_gwpy_override_terminal_compat.py::test_array2d_numeric_transpose_matches_gwpy",
            pre_fix=(
                "Numeric Array2D-family permutations rewrote axis metadata instead "
                "of preserving the installed GWpy result."
            ),
            comparator="exact-axis-permutation-result",
        ),
    )
    register(
        "gwexpy.fields.scalar.ScalarField",
        "diff",
        _terminal_closure(
            "fixed",
            "numeric-finite-difference",
            "tests/types/test_gwpy_override_terminal_compat.py::test_scalarfield_diff_common_route_matches_gwpy",
            pre_fix=(
                "ScalarField.diff hid the GWpy finite-difference surface behind "
                "a field-comparison operation."
            ),
            comparator="exact-values-shape-unit-and-axes",
        ),
    )
    register(
        "gwexpy.frequencyseries.bifrequencymap.BifrequencyMap",
        "diagonal",
        _terminal_closure(
            "fixed",
            "numeric-diagonal-view",
            "tests/frequencyseries/test_bifrequencymap_gwpy_compat.py::test_diagonal_common_numeric_routes_match_gwpy",
            pre_fix=(
                "BifrequencyMap.diagonal selected its binned extension by default "
                "instead of the GWpy Array2D diagonal view."
            ),
            comparator="exact-values-metadata-and-view-sharing",
        ),
    )
    register(
        "gwexpy.frequencyseries.bifrequencymap.BifrequencyMap",
        "crop",
        _terminal_closure(
            "fixed",
            "single-axis-common-crop",
            "tests/frequencyseries/test_bifrequencymap_gwpy_compat.py::test_crop_common_route_matches_gwpy",
            pre_fix=(
                "BifrequencyMap.crop always selected two axes and exposed copy in "
                "a calling form that diverged from GWpy."
            ),
            comparator="exact-values-metadata-and-memory",
        ),
    )
    register(
        "gwexpy.frequencyseries.bifrequencymap.BifrequencyMap",
        "plot",
        _terminal_closure(
            "fixed",
            "common-bifrequency-plot-route",
            "tests/frequencyseries/test_bifrequencymap_gwpy_compat.py::test_plot_common_routes_match_gwpy_artist_source_and_label",
            pre_fix=(
                "BifrequencyMap.plot failed for unnamed data and changed the GWpy "
                "artist route, source orientation, or keyword ownership."
            ),
            comparator="exact-plot-outcome-and-artist-source",
        ),
    )

    plot_surface_references = {
        "gwexpy.plot.plot.Plot": "test_plot_set_matches_gwpy",
        "gwexpy.plot.field.FieldPlot": "test_fieldplot_set_matches_gwpy",
        "gwexpy.plot.skymap.SkyMap": "test_skymap_set_matches_gwpy",
    }
    for public_class, node in plot_surface_references.items():
        register(
            public_class,
            "set",
            _terminal_closure(
                "no-finding",
                "figure-property-update",
                f"tests/plot/test_gwpy_override_terminal_plot.py::{node}",
                comparator="exact-binding-outcome-and-figure-state",
            ),
        )
        register(
            public_class,
            "show",
            _terminal_closure(
                "fixed",
                "gwpy-positional-show-surface",
                "tests/test_gwpy_override_inventory.py::test_terminal_plot_show_lifecycle_matches_gwpy",
                pre_fix=(
                    "The close extension occupied a GWpy positional slot and the "
                    "default closed figures that GWpy leaves open."
                ),
                comparator="exact-show-binding-and-lifecycle",
            ),
        )

    image_rows = {
        "gwexpy.types.array2d.Array2D": {
            "imshow": "test_array2d_imshow_matches_gwpy",
            "pcolormesh": "test_array2d_pcolormesh_matches_gwpy",
        },
        "gwexpy.types.plane2d.Plane2D": {
            "imshow": "test_plane2d_imshow_matches_gwpy",
            "pcolormesh": "test_plane2d_pcolormesh_matches_gwpy",
        },
        "gwexpy.spectrogram.spectrogram.Spectrogram": {
            "imshow": "test_spectrogram_imshow_matches_gwpy",
            "pcolormesh": "test_spectrogram_pcolormesh_matches_gwpy",
        },
    }
    for public_class, members in image_rows.items():
        for member, node in members.items():
            register(
                public_class,
                member,
                _terminal_closure(
                    "no-finding",
                    "finite-and-nonfinite-image-artist",
                    f"tests/plot/test_gwpy_override_terminal_plot.py::{node}",
                    comparator="exact-binding-outcome-and-artist-source",
                ),
            )

    register(
        (
            "gwexpy.timeseries.timeseries.TimeSeries",
            "gwexpy.frequencyseries.frequencyseries.FrequencySeries",
            "gwexpy.spectrogram.spectrogram.Spectrogram",
            "gwexpy.timeseries.collections.TimeSeriesDict",
        ),
        "plot",
        _terminal_closure(
            "fixed",
            "supported-positional-plot-prefix",
            "tests/plot/test_gwpy_phase4_compat.py::test_every_supported_positional_prefix_matches_gwpy",
            issues=("#639", "#704", "#706"),
            pre_fix=(
                "The override signatures and positional routing displaced GWpy plot "
                "arguments or changed success and failure outcomes."
            ),
            comparator="exact-binding-and-plot-outcome",
        ),
    )

    io_rows: dict[str, dict[str, tuple[str, str]]] = {
        "gwexpy.timeseries.timeseries.TimeSeries": {
            "read": ("test_timeseries_read_matches_gwpy", "#700"),
            "write": ("test_timeseries_write_matches_gwpy", "#700"),
        },
        "gwexpy.timeseries.collections.TimeSeriesDict": {
            "read": ("test_timeseriesdict_read_matches_gwpy", "#611"),
            "write": ("test_timeseriesdict_write_matches_gwpy", "#611"),
        },
        "gwexpy.frequencyseries.frequencyseries.FrequencySeries": {
            "read": ("test_frequencyseries_read_matches_gwpy", "#701"),
            "write": ("test_frequencyseries_write_matches_gwpy", "#701"),
        },
    }
    for public_class, io_members in io_rows.items():
        for member, (node, issue) in io_members.items():
            member_id = f"{public_class}/{member}"
            is_non_intersecting_safety_exception = (
                member_id == TIMESERIESDICT_READ_MEMBER_ID
            )
            io_pre_fix = (
                f"The {member} override changed GWpy native routing, data, "
                "metadata, or failure outcomes for ordinary inputs."
            )
            io_observations: Mapping[str, Mapping[str, Any]] | None = None
            if is_non_intersecting_safety_exception:
                io_observations = NON_INTERSECTING_WINDOW_SAFETY_OBSERVATIONS
            additional_behavior = HDF5_AUTO_ROUTE_REFERENCES.get(member_id, ())
            if is_non_intersecting_safety_exception:
                additional_behavior += (NON_INTERSECTING_WINDOW_SAFETY_REFERENCE,)
            register(
                public_class,
                member,
                _terminal_closure(
                    "fixed",
                    f"native-{member}-route",
                    f"tests/io/test_gwpy_override_terminal_io.py::{node}",
                    issues=("#639", issue, "#704"),
                    pre_fix=io_pre_fix,
                    comparator="exact-io-outcome-payload-and-metadata",
                    observations=io_observations,
                    additional_behavior=additional_behavior,
                    compatibility_exception=(
                        NON_INTERSECTING_WINDOW_SAFETY
                        if is_non_intersecting_safety_exception
                        else None
                    ),
                ),
            )

    timeseries = "gwexpy.timeseries.timeseries.TimeSeries"
    timeseries_terminal = {
        "copy": (
            "no-finding",
            "array-copy-order-C",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_copy_orders_match_gwpy",
            None,
            "exact-values-metadata-binding-and-outcome",
            None,
            None,
        ),
        "crop": (
            "fixed",
            "third-positional-copy-rejection",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_crop_copy_is_keyword_only_like_gwpy",
            "GWexpy accepted a third positional copy argument that GWpy "
            "rejected with TypeError.",
            "exact-call-outcome-and-exception-class",
            {
                "gwexpy": {
                    "exception_class": "TypeError",
                    "outcome": "exception",
                },
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
        ),
        "append": (
            "fixed",
            "gwpy-append-parameter-layout",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_append_parameter_layout_matches_gwpy",
            "GWexpy exposed pad before gap in the public keyword-only parameter "
            "layout, unlike GWpy.",
            "exact-parameter-layout",
            None,
            None,
        ),
        "t0": (
            "fixed",
            "none-epoch-setter",
            "tests/timeseries/test_exact_gps_epoch.py"
            "::test_epoch_setters_accept_none_and_clear_exact_authority[t0]",
            "GWexpy rejected a GWpy-supported None t0 assignment with TypeError.",
            "exact-setter-outcome-and-metadata",
            None,
            {
                "gwexpy": {
                    "exception_class": "TypeError",
                    "outcome": "exception",
                },
                "gwpy": {"outcome": "return"},
            },
        ),
        "x0": (
            "fixed",
            "none-epoch-setter",
            "tests/timeseries/test_exact_gps_epoch.py"
            "::test_epoch_setters_accept_none_and_clear_exact_authority[x0]",
            "GWexpy rejected a GWpy-supported None x0 assignment with TypeError.",
            "exact-setter-outcome-and-metadata",
            None,
            {
                "gwexpy": {
                    "exception_class": "TypeError",
                    "outcome": "exception",
                },
                "gwpy": {"outcome": "return"},
            },
        ),
        "spectrogram": (
            "fixed",
            "gwpy-spectrogram-parameter-layout",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_spectrogram_parameter_layout_matches_gwpy",
            "GWexpy hid the public GWpy spectrogram parameter layout behind "
            "variadic arguments.",
            "exact-parameter-layout",
            None,
            None,
        ),
        "spectrogram2": (
            "fixed",
            "gwpy-spectrogram2-parameter-layout",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_spectrogram2_parameter_layout_matches_gwpy",
            "GWexpy hid the public GWpy spectrogram2 parameter layout behind "
            "variadic arguments.",
            "exact-parameter-layout",
            None,
            None,
        ),
    }
    for member, (
        state,
        fixture,
        reference,
        pre_fix,
        comparator,
        observations,
        pre_fix_observations,
    ) in timeseries_terminal.items():
        register(
            timeseries,
            member,
            _terminal_closure(
                state,
                fixture,
                reference,
                pre_fix=pre_fix,
                comparator=comparator,
                observations=observations,
                pre_fix_observations=pre_fix_observations,
            ),
        )

    register(
        timeseries,
        ("dt", "dx"),
        _terminal_closure(
            "no-finding",
            "cadence-set-copy-and-slice",
            "tests/timeseries/test_exact_gps_epoch.py::test_cadence_setters_synchronize_exact_interval_for_copy_and_slice",
            comparator="exact-cadence-property-outcome",
        ),
    )

    collection = "gwexpy.timeseries.collections.TimeSeriesDict"
    collection_terminal = {
        "append": (
            "fixed",
            "invalid-copy-key-mapping",
            "test_timeseriesdict_append_invalid_copy_key_matches_gwpy_without_mutation",
            "GWexpy removed an invalid copy key from the input mapping and "
            "suppressed the ValueError raised by GWpy.",
            "exact-call-outcome-mutation-and-exception-class",
            {
                "gwexpy": {
                    "exception_class": "ValueError",
                    "outcome": "exception",
                },
                "gwpy": {"exception_class": "ValueError", "outcome": "exception"},
            },
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"exception_class": "ValueError", "outcome": "exception"},
            },
        ),
        "crop": (
            "fixed",
            "third-positional-copy-rejection",
            "test_timeseriesdict_crop_copy_is_keyword_only_like_gwpy",
            "GWexpy accepted a third positional copy argument that GWpy "
            "rejected with TypeError.",
            "exact-call-outcome-and-exception-class",
            {
                "gwexpy": {
                    "exception_class": "TypeError",
                    "outcome": "exception",
                },
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
        ),
        "resample": (
            "no-finding",
            "mapping-numeric-resample",
            "test_timeseriesdict_numeric_resample_matches_gwpy",
            None,
            "exact-collection-values-metadata-and-mutation",
            None,
            None,
        ),
    }
    for member, (
        state,
        fixture,
        node,
        pre_fix,
        comparator,
        observations,
        pre_fix_observations,
    ) in collection_terminal.items():
        register(
            collection,
            member,
            _terminal_closure(
                state,
                fixture,
                f"tests/timeseries/test_gwpy_override_terminal_compat.py::{node}",
                pre_fix=pre_fix,
                comparator=comparator,
                observations=observations,
                pre_fix_observations=pre_fix_observations,
            ),
        )
    register(
        collection,
        "prepend",
        _terminal_closure(
            "fixed",
            "key-wise-mapping-prepend",
            "tests/timeseries/test_collections_batch.py::test_timeseriesdict_prepend_is_key_wise_like_gwpy",
            pre_fix=(
                "The collection override broadcast a single series instead of "
                "performing GWpy-compatible key-wise mapping prepend semantics."
            ),
            comparator="exact-collection-values-metadata-and-mutation",
        ),
    )

    signal_rows = {
        "heterodyne": (
            "test_heterodyne_matches_gwpy_values_and_metadata",
            "The override changed output units, phase failure classes, stride "
            "semantics, public binding, and exact output cadence.",
            ("#639", "#704"),
        ),
        "demodulate": (
            "test_demodulate_matches_gwpy_values_and_metadata",
            "The override changed binding, output units, Quantity semantics, and "
            "exact output cadence relative to GWpy.",
            ("#639", "#704"),
        ),
        "rms": (
            "test_rms_default_matches_gwpy_values_and_metadata",
            "The override omitted NaNs by default and changed naming, numerical, "
            "failure, and exact-time outcomes.",
            ("#451", "#639", "#704"),
        ),
    }
    for member, (node, pre_fix, issues) in signal_rows.items():
        register(
            timeseries,
            member,
            _terminal_closure(
                "fixed",
                f"default-{member}-route",
                f"tests/timeseries/test_gwpy_audit_signal_compat.py::{node}",
                issues=issues,
                pre_fix=pre_fix,
                comparator="exact-values-nonfinite-masks-and-metadata",
            ),
        )
    register(
        timeseries,
        "resample",
        _terminal_closure(
            "fixed",
            "gwpy-resample-parameter-layout",
            "tests/timeseries/test_gwpy_audit_signal_compat.py"
            "::test_resample_signature_keeps_gwpy_positional_layout",
            issues=("#639", "#703", "#704"),
            pre_fix=(
                "GWexpy hid the GWpy window, ftype, and n parameters behind "
                "variadic arguments."
            ),
            comparator="exact-parameter-layout",
        ),
    )

    register(
        "gwexpy.frequencyseries.frequencyseries.FrequencySeries",
        "ifft",
        _terminal_closure(
            "fixed",
            "large-epoch-high-rate-ifft",
            "tests/frequencyseries/test_ifft_gwpy_compat.py::test_ifft_default_preserves_parent_axis_shape_and_values",
            issues=("#639", "#703", "#704"),
            pre_fix=(
                "The override reconstructed cadence from absolute times, quantizing "
                "high-rate metadata and collapsing ten-megahertz cadence."
            ),
            comparator="exact-values-shape-dtype-and-axis-metadata",
        ),
    )
    for member in ("fft", "psd", "asd", "coherence"):
        register(
            timeseries,
            member,
            _terminal_closure(
                "fixed",
                "true-irregular-seconds",
                "tests/timeseries/test_spectral_gwpy_phase3_compat.py"
                "::test_true_irregular_axis_preserves_parent_failure_class"
                f"[{member}-seconds]",
                issues=("#639", "#703", "#704"),
                pre_fix=(
                    "Phase 3 classifies this fixture as GWpy-fails for upstream "
                    "support; the override inventory classifies the row as fixed "
                    "because GWexpy previously replaced GWpy's AttributeError "
                    "with ValueError."
                ),
                comparator="exact-failure-outcome-and-exception-class",
                observations={
                    "gwexpy": {
                        "exception_class": "AttributeError",
                        "outcome": "exception",
                    },
                    "gwpy": {
                        "exception_class": "AttributeError",
                        "outcome": "exception",
                    },
                },
                pre_fix_observations={
                    "gwexpy": {
                        "exception_class": "ValueError",
                        "outcome": "exception",
                    },
                    "gwpy": {
                        "exception_class": "AttributeError",
                        "outcome": "exception",
                    },
                },
            ),
        )
    register(
        timeseries,
        "csd",
        _terminal_closure(
            "fixed",
            "mixed-unit-default-csd",
            "tests/timeseries/test_spectral_gwpy_phase3_compat.py::test_csd_default_preserves_the_parent_result",
            issues=("#639", "#698", "#704"),
            pre_fix="The default CSD route changed the unit selected by GWpy.",
            comparator="exact-spectral-values-unit-and-metadata",
        ),
    )
    register(
        timeseries,
        "rayleigh_spectrogram",
        _terminal_closure(
            "fixed",
            "odd-recommended-overlap-rayleigh",
            "tests/timeseries/test_spectral_gwpy_phase3_compat.py::test_rayleigh_spectrogram_default_preserves_the_parent_result",
            issues=("#639", "#699", "#704"),
            pre_fix=(
                "The default Rayleigh route used a corrected segment selection "
                "instead of the installed GWpy selection."
            ),
            comparator="exact-spectral-values-unit-and-metadata",
        ),
    )
    register(
        timeseries,
        "transfer_function",
        _terminal_closure(
            "fixed",
            "steady-default-transfer",
            "tests/timeseries/test_spectral_gwpy_phase3_compat.py::test_steady_transfer_default_preserves_the_parent_result",
            issues=("#639", "#702", "#704"),
            pre_fix=(
                "The steady transfer route changed GWpy unit, name, channel, or "
                "epoch metadata and zero-denominator behavior."
            ),
            comparator="exact-spectral-values-nonfinite-masks-and-metadata",
        ),
    )

    return dict(sorted(closures.items(), key=lambda item: item[0].encode("utf-8")))


TERMINAL_CLOSURES = _build_terminal_closures()


def canonical_compact_json(value: Any) -> str:
    """Return the canonical compact JSON representation used for digests."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_manifest_json(value: Any) -> str:
    """Return the canonical checked-in JSON representation."""

    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_compact_json(value).encode("ascii")).hexdigest()


def walk_manifest_values(value: Any) -> Iterator[Any]:
    """Yield every recursive manifest value for hygiene checks."""

    yield value
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from walk_manifest_values(key)
            yield from walk_manifest_values(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_manifest_values(item)


def _fqname(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _stable_atom(value: Any) -> dict[str, Any]:
    """Normalize defaults and annotations without address-bearing repr()."""

    if value is inspect.Parameter.empty or value is inspect.Signature.empty:
        return {"kind": "empty"}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            token = "nan"
        elif value > 0:
            token = "+inf"
        else:
            token = "-inf"
        return {"kind": "float", "value": token}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "literal", "value": value}
    # Matplotlib 3.11 moved the singleton used in generated Artist.set
    # signatures. Its module location is not a change to the default value.
    sentinel_names = {
        "matplotlib.artist": "_UNSET",
        "matplotlib._api": "UNSET",
    }
    sentinel_module = type(value).__module__
    if type(value).__name__ == "_Unset" and sentinel_module in sentinel_names:
        sentinel_owner = sys.modules.get(sentinel_module)
        if sentinel_owner is not None and value is vars(sentinel_owner).get(
            sentinel_names[sentinel_module]
        ):
            return {"kind": "sentinel", "name": "matplotlib.UNSET"}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_stable_atom(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_stable_atom(item) for item in value]
        return {"kind": "frozenset", "items": sorted(items, key=canonical_compact_json)}
    if isinstance(value, type):
        return {"kind": "type", "name": _fqname(value)}
    if isinstance(value, types.ForwardRef) if hasattr(types, "ForwardRef") else False:
        return {"kind": "forward-reference", "value": str(value)}
    origin = get_origin(value)
    if origin is not None:
        return {
            "kind": "typing",
            "origin": _stable_atom(origin),
            "arguments": [_stable_atom(item) for item in get_args(value)],
        }
    module = type(value).__module__
    qualname = type(value).__qualname__
    name = getattr(value, "__name__", None)
    normalized: dict[str, Any] = {
        "kind": "opaque",
        "type": f"{module}.{qualname}",
    }
    if isinstance(name, str):
        normalized["name"] = name
    return normalized


def normalize_signature(value: Any) -> dict[str, Any]:
    """Normalize a callable signature structurally, or record error class only."""

    try:
        signature = inspect.signature(value, eval_str=False)
    except (TypeError, ValueError) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "parameters": [
            {
                "annotation": _stable_atom(parameter.annotation),
                "default": _stable_atom(parameter.default),
                "kind": parameter.kind.name,
                "name": parameter.name,
            }
            for parameter in signature.parameters.values()
        ],
        "return_annotation": _stable_atom(signature.return_annotation),
    }


def _descriptor_slots(raw: Any) -> list[str]:
    descriptor_mro = inspect.getmro(type(raw))
    slots = []
    for label, special in (
        ("get", "__get__"),
        ("set", "__set__"),
        ("delete", "__delete__"),
    ):
        if any(special in vars(owner) for owner in descriptor_mro):
            slots.append(label)
    return slots


def raw_binding_kind(raw: Any) -> str | None:
    """Classify a raw ``vars(owner)[name]`` binding without invoking it."""

    raw_type = type(raw)
    if any(
        owner.__name__ == "UnifiedReadWriteMethod" for owner in inspect.getmro(raw_type)
    ):
        return "unified-read-write"
    if isinstance(raw, classmethod):
        return "classmethod"
    if isinstance(raw, staticmethod):
        return "staticmethod"
    if isinstance(raw, property):
        return "property"
    if isinstance(raw, (types.FunctionType, types.BuiltinFunctionType)):
        return "function"
    if inspect.isclass(raw):
        return None
    if _descriptor_slots(raw):
        return "generic-descriptor"
    if callable(raw):
        return "callable"
    return None


def _callable_descriptor(raw: Any, kind: str) -> Any:
    if kind in {"classmethod", "staticmethod"}:
        return raw.__func__
    return raw


def _raw_type_call(raw: Any) -> Any | None:
    """Resolve an instance's ``__call__`` implementation from static MRO vars."""

    return next(
        (
            vars(owner)["__call__"]
            for owner in inspect.getmro(type(raw))
            if "__call__" in vars(owner)
        ),
        None,
    )


def _package_relative_source(path: str | None, *, gwpy_owned: bool) -> str | None:
    if not path or not gwpy_owned:
        return None
    parts = Path(path).parts
    try:
        index = parts.index("gwpy")
    except ValueError:
        return None
    return Path(*parts[index:]).as_posix()


def _source_reference(value: Any, repository: Path) -> dict[str, Any] | None:
    try:
        path_text = inspect.getsourcefile(value) or inspect.getfile(value)
        _, line = inspect.getsourcelines(value)
    except (OSError, TypeError):
        return None
    path = Path(path_text).resolve()
    try:
        relative = path.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return None
    return {"path": relative, "line": line}


def _descriptor_projection(raw: Any, kind: str, repository: Path) -> dict[str, Any]:
    if kind in {"property", "unified-read-write"}:
        accessors = []
        details: dict[str, Any] = {}
        for label, attribute in (("get", "fget"), ("set", "fset"), ("delete", "fdel")):
            accessor = vars(type(raw)).get(attribute, None)
            # property stores these on the instance; reading the C-level slot does
            # not execute the public member descriptor.
            try:
                accessor = object.__getattribute__(raw, attribute)
            except AttributeError:
                accessor = None
            if accessor is not None:
                accessors.append(label)
                details[label] = {
                    "signature": normalize_signature(accessor),
                    "source": _source_reference(accessor, repository),
                }
        return {"accessors": accessors, "details": details}
    if kind == "generic-descriptor":
        details = {}
        if callable(raw):
            raw_call = _raw_type_call(raw)
            details["call"] = {
                "signature": normalize_signature(raw),
                "source": _source_reference(raw, repository)
                or (
                    _source_reference(raw_call, repository)
                    if raw_call is not None
                    else None
                ),
            }
        return {"accessors": _descriptor_slots(raw), "details": details}
    target = _callable_descriptor(raw, kind)
    return {
        "accessors": [],
        "details": {
            "call": {
                "signature": normalize_signature(target),
                "source": _source_reference(target, repository),
            }
        },
    }


def _literal_all_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for statement in tree.body:
        value: ast.expr | None = None
        assigns_all = False
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            assigns_all = True
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__all__"
            and statement.value is not None
        ):
            assigns_all = True
            value = statement.value
        if not assigns_all:
            continue
        if isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            for item in value.elts
        ):
            # Python assignment replaces the previous object.  Mirroring that
            # semantics avoids retaining stale exports from an earlier literal.
            names = []
            for item in value.elts:
                assert isinstance(item, ast.Constant)
                assert isinstance(item.value, str)
                names.append(item.value)
        else:
            names = []
    return tuple(sorted(set(names), key=lambda item: item.encode("utf-8")))


def _module_name(source: Path, package_root: Path) -> str:
    relative = source.relative_to(package_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("gwexpy", *parts))


def _public_root_allowed(value: type[Any]) -> bool:
    name = value.__name__
    module = value.__module__
    if not module.startswith("gwexpy.") or name.startswith("_"):
        return False
    if any(token in name for token in INTERNAL_CLASS_TOKENS):
        return False
    if ".gui" in module or ".io." in module:
        return False
    return any(
        base.__module__.startswith("gwpy.") for base in inspect.getmro(value)[1:]
    )


def _select_unique_lazy_class(
    export_name: str, candidates: Sequence[type[Any]]
) -> type[Any]:
    """Select one static canonical-name route, failing closed otherwise."""

    unique: list[type[Any]] = []
    for candidate in candidates:
        if candidate.__name__ != export_name:
            continue
        if not any(candidate is existing for existing in unique):
            unique.append(candidate)
    if not unique:
        raise InventoryError(f"missing lazy class export route: {export_name}")
    if len(unique) != 1:
        routes = ", ".join(
            sorted((_fqname(item) for item in unique), key=lambda item: item.encode())
        )
        raise InventoryError(
            f"ambiguous lazy class export route: {export_name} ({routes})"
        )
    return unique[0]


def _looks_like_class_export(name: str) -> bool:
    """Distinguish unresolved class-style exports from functions and constants."""

    return bool(name) and name[0].isupper() and not name.isupper()


def discover_public_classes(
    repository: Path,
) -> list[tuple[type[Any], tuple[str, ...]]]:
    """Discover canonical public roots from literal explicit exports only."""

    package_root = repository / "gwexpy"
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    exports: dict[type[Any], set[str]] = {}
    export_modules: list[tuple[str, tuple[str, ...]]] = []
    class_routes: dict[str, set[str]] = {}
    paths = sorted(
        package_root.rglob("*.py"),
        key=lambda item: item.relative_to(repository).as_posix().encode("utf-8"),
    )
    for source in paths:
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as exc:
            raise InventoryError(
                f"cannot scan {source.relative_to(repository)}: {type(exc).__name__}"
            ) from exc
        module_name = _module_name(source, package_root)
        for statement in tree.body:
            if isinstance(statement, ast.ClassDef):
                class_routes.setdefault(statement.name, set()).add(module_name)
        names = _literal_all_names(tree)
        if not names:
            continue
        module_parts = module_name.split(".")
        if (
            "gui" in module_parts
            or "io" in module_parts
            or "__main__" in module_parts
            or any(part.startswith("_") for part in module_parts[1:])
        ):
            continue
        export_modules.append((module_name, names))

    snapshots: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
    classes_by_name: dict[str, list[type[Any]]] = {}
    missing = object()
    for module_name, names in export_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise InventoryError(
                f"cannot import explicit export module {module_name}: {type(exc).__name__}"
            ) from exc
        namespace = vars(module)
        snapshot = {name: namespace.get(name, missing) for name in names}
        snapshots.append((module_name, names, snapshot))
        # The public export can be lazy while its canonical class is imported
        # into another explicit module under a non-exported helper name.  Build
        # candidates from static namespace snapshots only; never call getattr.
        for value in namespace.values():
            if inspect.isclass(value):
                classes_by_name.setdefault(value.__name__, []).append(value)

    for module_name, names, snapshot in snapshots:
        for name in names:
            value = snapshot[name]
            if inspect.isclass(value):
                if _public_root_allowed(value):
                    exports.setdefault(value, set()).add(f"{module_name}:{name}")
                continue
            # A concrete non-class binding masks any same-named class visible in
            # another module, just as an effective MRO binding masks its bases.
            if value is not missing:
                continue
            routes = sorted(
                class_routes.get(name, ()), key=lambda item: item.encode("utf-8")
            )
            if not routes:
                if _looks_like_class_export(name):
                    raise InventoryError(f"missing lazy class export route: {name}")
                continue
            if len(routes) != 1:
                raise InventoryError(
                    f"ambiguous lazy class export route: {name} ({', '.join(routes)})"
                )
            candidate_module_name = routes[0]
            try:
                candidate_module = importlib.import_module(candidate_module_name)
            except Exception as exc:
                raise InventoryError(
                    "cannot import static class candidate module "
                    f"{candidate_module_name}: {type(exc).__name__}"
                ) from exc
            candidate = vars(candidate_module).get(name, missing)
            if not inspect.isclass(candidate):
                raise InventoryError(f"missing lazy class export route: {name}")
            candidates = [*classes_by_name.get(name, ()), candidate]
            value = _select_unique_lazy_class(name, candidates)
            if _public_root_allowed(value):
                exports.setdefault(value, set()).add(f"{module_name}:{name}")
    return [
        (value, tuple(sorted(paths, key=lambda item: item.encode("utf-8"))))
        for value, paths in sorted(
            exports.items(), key=lambda item: _fqname(item[0]).encode("utf-8")
        )
    ]


def _raw_alias_identity(
    owner: type[Any], raw: Any, kind: str
) -> tuple[str, str | None]:
    aliases = sorted(
        name
        for name, candidate in vars(owner).items()
        if candidate is raw and (not name.startswith("_") or name in CONSTRUCTORS)
    )
    canonical = aliases[0] if aliases else None
    identity = f"{_fqname(owner)}::{canonical or kind}::{kind}"
    alias_group = identity if len(aliases) > 1 else None
    return identity, alias_group


def extract_members_for_classes(
    classes: Sequence[tuple[type[Any], tuple[str, ...]]], repository: Path
) -> list[dict[str, Any]]:
    """Extract effective GWexpy bindings before the first GWpy MRO class."""

    members: list[dict[str, Any]] = []
    for public_class, exports in classes:
        mro = inspect.getmro(public_class)
        try:
            boundary = next(
                index
                for index, owner in enumerate(mro)
                if owner.__module__.startswith("gwpy.")
            )
        except StopIteration as exc:
            raise InventoryError(
                f"public root has no GWpy MRO class: {_fqname(public_class)}"
            ) from exc
        prefix = mro[:boundary]
        candidate_names = sorted(
            {
                name
                for owner in prefix
                for name in vars(owner)
                if not name.startswith("_") or name in CONSTRUCTORS
            },
            key=lambda item: item.encode("utf-8"),
        )
        for name in candidate_names:
            # Resolve the first binding before deciding whether it is callable.
            # A subclass ``name = None`` intentionally masks a later mixin method.
            effective_owner = next(owner for owner in mro if name in vars(owner))
            raw = vars(effective_owner)[name]
            kind = raw_binding_kind(raw)
            if kind is None or not effective_owner.__module__.startswith("gwexpy."):
                continue
            raw_identity, alias_group = _raw_alias_identity(effective_owner, raw, kind)
            if effective_owner is public_class:
                resolution = "direct"
            elif "Mixin" in effective_owner.__name__:
                resolution = "inherited-mixin"
            else:
                resolution = "inherited-gwexpy-base"
            member_id = f"{_fqname(public_class)}/{name}"
            source_target = _callable_descriptor(raw, kind)
            if kind in {"property", "unified-read-write"}:
                source_target = object.__getattribute__(raw, "fget") or effective_owner
            members.append(
                {
                    "alias_group": alias_group,
                    "constructor": name in CONSTRUCTORS,
                    "counterpart_class": _fqname(mro[boundary]),
                    "descriptor": _descriptor_projection(raw, kind, repository),
                    "effective_owner": _fqname(effective_owner),
                    "exports": list(exports),
                    "kind": kind,
                    "member": name,
                    "member_id": member_id,
                    "mro_prefix": [_fqname(owner) for owner in prefix],
                    "public_class": _fqname(public_class),
                    "raw_descriptor_identity": raw_identity,
                    "resolution": resolution,
                    "source": _source_reference(source_target, repository),
                }
            )
    return sorted(members, key=lambda item: item["member_id"].encode("utf-8"))


def build_source_population(repository: Path) -> dict[str, Any]:
    classes = discover_public_classes(repository)
    roots = []
    for value, exports in classes:
        mro = inspect.getmro(value)
        counterpart = next(
            owner for owner in mro if owner.__module__.startswith("gwpy.")
        )
        roots.append(
            {
                "counterpart_class": _fqname(counterpart),
                "exports": list(exports),
                "public_class": _fqname(value),
                "source": _source_reference(value, repository),
            }
        )
    members = extract_members_for_classes(classes, repository)
    population = {"public_roots": roots, "members": members}
    population["digest"] = digest_json(population)
    return population


def _resolve_qualname(module_name: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        namespace = vars(value)
        if component not in namespace:
            raise InventoryError(f"cannot resolve {module_name}.{qualname}")
        value = namespace[component]
    return value


def _split_fqname(name: str) -> tuple[str, str]:
    components = name.split(".")
    for index in range(len(components) - 1, 0, -1):
        module_name = ".".join(components[:index])
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        return module_name, ".".join(components[index:])
    raise InventoryError(f"cannot split importable name: {name}")


def _oracle_source(raw: Any, provider: type[Any]) -> dict[str, Any] | None:
    """Keep source provenance only for exactly pinned GWpy-owned providers."""

    if not provider.__module__.startswith("gwpy."):
        return None
    kind = raw_binding_kind(raw)
    target = _callable_descriptor(raw, kind or "callable")
    if kind in {"property", "unified-read-write"}:
        try:
            target = object.__getattribute__(raw, "fget") or provider
        except AttributeError:
            target = provider
    try:
        path = inspect.getsourcefile(target) or inspect.getfile(target)
        _, line = inspect.getsourcelines(target)
    except (OSError, TypeError):
        return None
    relative = _package_relative_source(path, gwpy_owned=True)
    return {"path": relative, "line": line} if relative is not None else None


def _oracle_callable_signature(raw: Any, provider: type[Any]) -> dict[str, Any]:
    """Bind the two reviewed inspection forms of five NumPy C descriptors."""

    observed = normalize_signature(raw)
    if _fqname(provider) != "numpy.ndarray":
        return observed
    import numpy as np

    name = getattr(raw, "__name__", None)
    if (
        provider is not np.ndarray
        or not isinstance(name, str)
        or raw is not vars(np.ndarray).get(name)
    ):
        return observed
    empty = inspect.Parameter.empty
    positional = "POSITIONAL_ONLY"
    keyword = "POSITIONAL_OR_KEYWORD"
    reduction = [
        ("axis", keyword, None),
        ("out", keyword, None),
        ("kwargs", "VAR_KEYWORD", empty),
    ]
    parameters: dict[str, list[tuple[str, str, Any]]] = {
        "max": reduction,
        "min": reduction,
        "swapaxes": [("axis1", positional, empty), ("axis2", positional, empty)],
        "transpose": [("axes", "VAR_POSITIONAL", empty)],
        "diagonal": [
            ("offset", keyword, 0),
            ("axis1", keyword, 0),
            ("axis2", keyword, 1),
        ],
    }
    if name not in parameters:
        return observed
    reviewed = {
        "available": True,
        "parameters": [
            {
                "annotation": {"kind": "empty"},
                "default": _stable_atom(default),
                "kind": parameter_kind,
                "name": parameter_name,
            }
            for parameter_name, parameter_kind, default in [
                ("self", positional, empty),
                *parameters[name],
            ]
        ],
        "return_annotation": {"kind": "empty"},
    }
    # NumPy 1 lacks these text signatures; recent NumPy 2 exposes them.
    # Retain both exact reviewed forms, and reject any different observation.
    variants = [{"available": False, "error": "ValueError"}, reviewed]
    if canonical_compact_json(observed) not in {
        canonical_compact_json(variant) for variant in variants
    }:
        raise InventoryError(f"unreviewed NumPy descriptor signature: {name}")
    return {"kind": "reviewed-native-signature", "variants": variants}


def _oracle_descriptor(raw: Any, kind: str, provider: type[Any]) -> dict[str, Any]:
    if kind in {"property", "unified-read-write"}:
        accessors = []
        details: dict[str, Any] = {}
        for label, attribute in (("get", "fget"), ("set", "fset"), ("delete", "fdel")):
            try:
                accessor = object.__getattribute__(raw, attribute)
            except AttributeError:
                accessor = None
            if accessor is None:
                continue
            accessors.append(label)
            details[label] = {
                "signature": normalize_signature(accessor),
                "source": _oracle_source(accessor, provider),
            }
        return {"accessors": accessors, "details": details}
    if kind == "generic-descriptor":
        details = {}
        if callable(raw):
            raw_call = _raw_type_call(raw)
            details["call"] = {
                "signature": _oracle_callable_signature(raw, provider),
                "source": _oracle_source(raw, provider)
                or (
                    _oracle_source(raw_call, provider) if raw_call is not None else None
                ),
            }
        return {"accessors": _descriptor_slots(raw), "details": details}
    return {
        "accessors": [],
        "details": {
            "call": {
                "signature": normalize_signature(_callable_descriptor(raw, kind)),
                "source": _oracle_source(raw, provider),
            }
        },
    }


def _counterpart_raw_identity(
    provider: type[Any], member: str, raw: Any, kind: str
) -> str:
    aliases = sorted(
        name
        for name, candidate in vars(provider).items()
        if candidate is raw and (not name.startswith("_") or name in CONSTRUCTORS)
    )
    canonical = aliases[0] if aliases else member
    # Member is included deliberately: two public names must not share evidence
    # only because an upstream class happens to alias the same descriptor object.
    return f"{_fqname(provider)}::{member}::{canonical}::{kind}"


def build_oracle_projection(
    expected_version: str, queries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Resolve pristine GWpy counterparts without importing GWexpy."""

    if any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules):
        raise InventoryError("oracle worker started with gwexpy imported")
    actual_version = distribution_version("gwpy")
    if actual_version != expected_version:
        raise InventoryError(
            f"oracle version mismatch: expected {expected_version}, got {actual_version}"
        )
    members = []
    for query in sorted(
        queries, key=lambda item: str(item["member_id"]).encode("utf-8")
    ):
        counterpart_name = str(query["counterpart_class"])
        module_name, qualname = _split_fqname(counterpart_name)
        counterpart = _resolve_qualname(module_name, qualname)
        member = str(query["member"])
        binding_owner = next(
            (owner for owner in inspect.getmro(counterpart) if member in vars(owner)),
            None,
        )
        raw = vars(binding_owner)[member] if binding_owner is not None else None
        kind = raw_binding_kind(raw) if binding_owner is not None else None
        provider = binding_owner if kind is not None else None
        result: dict[str, Any] = {
            "counterpart_class": counterpart_name,
            "descriptor": None,
            "kind": None,
            "member": member,
            "member_id": str(query["member_id"]),
            "present": provider is not None,
            "provider": None,
            "public_class": str(query["public_class"]),
            "raw_descriptor_identity": None,
            "source": None,
        }
        if provider is not None:
            assert kind is not None
            result.update(
                {
                    "descriptor": _oracle_descriptor(raw, kind, provider),
                    "kind": kind,
                    "provider": _fqname(provider),
                    "raw_descriptor_identity": _counterpart_raw_identity(
                        provider, member, raw, kind
                    ),
                    "source": _oracle_source(raw, provider),
                }
            )
        members.append(result)
    if any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules):
        raise InventoryError("oracle worker imported gwexpy")
    body = {
        "gwpy_version": actual_version,
        "isolation": {
            "cwd_matches_expected": True,
            "gwexpy_absent_at_end": True,
            "gwexpy_absent_at_start": True,
            "isolated_flag": bool(sys.flags.isolated),
            "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "members": members,
        "schema": WORKER_SCHEMA,
    }
    body["digest"] = digest_json(body)
    return body


def _worker_main() -> int:
    try:
        payload = decode_json_strict(sys.stdin.read())
        if not isinstance(payload, dict):
            raise InventoryError("oracle payload must be an object")
        if (
            set(payload)
            != {
                "expected_cwd",
                "expected_version",
                "queries",
                "schema",
            }
            or payload.get("schema") != WORKER_SCHEMA
        ):
            raise InventoryError("invalid oracle payload schema")
        expected_cwd = payload.get("expected_cwd")
        queries = payload.get("queries")
        query_keys = {
            "counterpart_class",
            "member",
            "member_id",
            "public_class",
        }
        if not isinstance(expected_cwd, str) or not isinstance(queries, list):
            raise InventoryError("invalid oracle payload fields")
        if any(
            not isinstance(query, dict)
            or set(query) != query_keys
            or not all(isinstance(value, str) and value for value in query.values())
            for query in queries
        ):
            raise InventoryError("malformed oracle input: invalid query schema")
        projection = build_oracle_projection(
            str(payload.get("expected_version")), queries
        )
        projection["isolation"]["cwd_matches_expected"] = (
            isinstance(expected_cwd, str) and os.getcwd() == expected_cwd
        )
        unsigned = {key: value for key, value in projection.items() if key != "digest"}
        projection["digest"] = digest_json(unsigned)
        sys.stdout.write(canonical_compact_json(projection) + "\n")
        return 0
    except InventoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        print(f"malformed oracle input: {type(exc).__name__}", file=sys.stderr)
        return 2


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise InventoryError(f"non-finite JSON constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise InventoryError(f"non-finite JSON float: {value}")
    return parsed


def decode_json_strict(value: str) -> Any:
    """Decode JSON with duplicate and all non-finite numbers rejected."""

    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_non_finite_constant,
        parse_float=_parse_finite_json_float,
    )


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        loaded = decode_json_strict(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot load manifest: {type(exc).__name__}") from exc
    if not isinstance(loaded, dict):
        raise InventoryError("manifest must be a JSON object")
    return loaded


def parse_oracle_arguments(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise InventoryError("oracle argument must be VERSION=EXECUTABLE")
        oracle_version, executable = value.split("=", 1)
        if oracle_version not in SUPPORTED_GWPY:
            raise InventoryError(f"unknown oracle version: {oracle_version}")
        if oracle_version in parsed:
            raise InventoryError(f"duplicate oracle version: {oracle_version}")
        if not executable:
            raise InventoryError(f"empty oracle executable for {oracle_version}")
        if executable == "@current":
            resolved = Path(sys.executable).resolve()
        else:
            located = shutil.which(executable)
            resolved = Path(located or executable).resolve()
        parsed[oracle_version] = str(resolved)
    if not parsed:
        raise InventoryError("at least one --oracle-python is required")
    return parsed


def _run_bounded_subprocess(
    command: Sequence[str],
    *,
    cwd: str | Path,
    environment: Mapping[str, str],
    input_text: str | None,
    timeout: int,
    label: str,
) -> subprocess.CompletedProcess[str]:
    """Run one shell-free child without retaining unbounded output in memory."""

    try:
        with (
            tempfile.TemporaryFile(mode="w+b") as stdout_file,
            tempfile.TemporaryFile(mode="w+b") as stderr_file,
        ):
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(environment),
                input=input_text,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=False,
                shell=False,
                timeout=timeout,
            )
            stdout_size = stdout_file.tell()
            stderr_size = stderr_file.tell()
            returned_stdout = completed.stdout or ""
            returned_stderr = completed.stderr or ""
            returned_size = len(returned_stdout.encode("utf-8")) + len(
                returned_stderr.encode("utf-8")
            )
            if stdout_size + stderr_size + returned_size > SUBPROCESS_OUTPUT_LIMIT:
                raise InventoryError(f"{label} exceeded output limit")
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read().decode("utf-8") or returned_stdout
            stderr = stderr_file.read().decode("utf-8") or returned_stderr
    except subprocess.TimeoutExpired as exc:
        raise InventoryError(f"{label} timed out") from exc
    except OSError as exc:
        raise InventoryError(f"cannot execute {label}: {type(exc).__name__}") from exc
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run_pristine_oracle(
    script: Path,
    oracle_version: str,
    executable: str,
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    queries = [
        {
            "counterpart_class": member["counterpart_class"],
            "member": member["member"],
            "member_id": member["member_id"],
            "public_class": member["public_class"],
        }
        for member in members
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="gwexpy-gwpy-oracle-") as temporary:
        payload = {
            "expected_cwd": temporary,
            "expected_version": oracle_version,
            "queries": queries,
            "schema": WORKER_SCHEMA,
        }
        completed = _run_bounded_subprocess(
            [executable, "-I", str(script.resolve()), "--oracle-worker"],
            cwd=temporary,
            environment=environment,
            input_text=canonical_compact_json(payload),
            timeout=ORACLE_TIMEOUT_SECONDS,
            label=f"oracle {oracle_version}",
        )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        message = stderr.strip()[-4000:] or f"exit {completed.returncode}"
        raise InventoryError(message)
    try:
        projection = decode_json_strict(stdout)
    except json.JSONDecodeError as exc:
        raise InventoryError("oracle stdout is not canonical JSON") from exc
    if not isinstance(projection, dict):
        raise InventoryError("oracle stdout must be a JSON object")
    if stdout != canonical_compact_json(projection) + "\n":
        raise InventoryError("oracle stdout is not canonical JSON")
    _validate_projection(oracle_version, projection)
    isolation = projection.get("isolation", {})
    if isolation != {
        "cwd_matches_expected": True,
        "gwexpy_absent_at_end": True,
        "gwexpy_absent_at_start": True,
        "isolated_flag": True,
        "no_user_site": True,
    }:
        raise InventoryError("oracle isolation contract failed")
    return projection


def _case_sort_key(case: Mapping[str, Any]) -> tuple[bytes, bytes, bytes, bytes]:
    return tuple(
        str(case[key]).encode("utf-8")
        for key in ("public_class", "member", "gwpy_version", "fixture")
    )  # type: ignore[return-value]


def _projection_without_digest(projection: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in projection.items() if key != "digest"}


def _implementation_groups(
    members: Sequence[Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], str | None]:
    by_member = {str(member["member_id"]): member for member in members}
    raw_keys: dict[tuple[str, str], tuple[str, str, str] | None] = {}
    for oracle_version, projection in projections.items():
        for observed in projection["members"]:
            key = (str(observed["member_id"]), oracle_version)
            if not observed["present"]:
                raw_keys[key] = None
                continue
            member = by_member[key[0]]
            raw_keys[key] = (
                str(member["raw_descriptor_identity"]),
                str(observed["raw_descriptor_identity"]),
                str(observed["provider"]),
            )
    canonical_keys = sorted(
        {key for key in raw_keys.values() if key is not None},
        key=canonical_compact_json,
    )
    labels = {
        key: f"implementation-{digest_json(list(key))[:16]}" for key in canonical_keys
    }
    return {
        member_version: labels.get(raw_key) if raw_key is not None else None
        for member_version, raw_key in raw_keys.items()
    }


def calculate_summary(
    cases: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate structural and workflow counts from the actual case states."""

    state_counts = {
        state: sum(case.get("state") == state for case in cases)
        for state in (*TERMINAL_STATES, *PROVISIONAL_STATES)
    }
    present_per_version = {
        oracle_version: sum(
            item.get("present") is True
            for item in projections[oracle_version]["members"]
        )
        for oracle_version in SUPPORTED_GWPY
    }
    absent_per_version = {
        oracle_version: len(members) - present_per_version[oracle_version]
        for oracle_version in SUPPORTED_GWPY
    }
    if len(set(present_per_version.values())) != 1:
        raise InventoryError("counterpart-present count differs by oracle version")
    if len(set(absent_per_version.values())) != 1:
        raise InventoryError("counterpart-absent count differs by oracle version")
    implementation_group_count = len(
        {
            case.get("implementation_group")
            for case in cases
            if case.get("implementation_group") is not None
        }
    )
    return {
        "cases": len(cases),
        "constructors": sum(member.get("constructor") is True for member in members),
        "counterpart_absent_per_version": next(iter(absent_per_version.values())),
        "counterpart_implementation_groups": implementation_group_count,
        "counterpart_present_per_version": next(iter(present_per_version.values())),
        "differential-required": state_counts["differential-required"],
        "fixed": state_counts["fixed"],
        "GWexpy-only": state_counts["GWexpy-only"],
        "GWpy-fails": state_counts["GWpy-fails"],
        "logical_members": len(members),
        "no-finding": state_counts["no-finding"],
        "public_roots": len({member.get("public_class") for member in members}),
        "unreviewed": state_counts["unreviewed"],
    }


def require_terminal_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    """Fail with stable counts when any provisional inventory cases remain."""

    counts = {
        state: sum(case.get("state") == state for case in cases)
        for state in PROVISIONAL_STATES
    }
    if any(counts.values()):
        raise InventoryError(
            "provisional states remain: "
            f"differential-required={counts['differential-required']}, "
            f"unreviewed={counts['unreviewed']}"
        )


def _manifest_policy() -> dict[str, Any]:
    return {
        "behavioral_owner": AUDIT_OWNER,
        "fixture_key": ["public_class", "member", "gwpy_version", "fixture"],
        "implementation_base": IMPLEMENTATION_BASE,
        "member_walk_rule": MEMBER_WALK_RULE,
        "oracle_versions": list(SUPPORTED_GWPY),
        "pristine_oracle_rule": PRISTINE_ORACLE_RULE,
        "provisional_states": list(PROVISIONAL_STATES),
        "public_root_rule": PUBLIC_ROOT_RULE,
        "terminal_states": list(TERMINAL_STATES),
        "upstream_dependency_provenance": UPSTREAM_DEPENDENCY_PROVENANCE,
    }


def _terminal_case_fields(
    closure: Mapping[str, Any], projection_digest: str
) -> dict[str, Any]:
    """Materialize a reviewed closure without sharing mutable manifest values."""

    state = str(closure["state"])
    behavior = [
        {"reference": str(reference)} for reference in closure.get("behavior", ())
    ]
    evidence: dict[str, Any] = {
        "behavior": behavior,
        "oracle_projection_digest": projection_digest,
    }
    if state == "fixed":
        reference = str(closure["behavior"][0])
        pre_fix_observations = copy.deepcopy(closure["pre_fix_observations"])
        pre_fix_observations["gwexpy"].setdefault("detail", str(closure["pre_fix"]))
        evidence.update(
            {
                "green_test": {"reference": reference},
                "pre_fix_mismatch": {
                    "gwexpy": pre_fix_observations["gwexpy"],
                    "gwpy": pre_fix_observations["gwpy"],
                    "reference": reference,
                },
            }
        )
    fields = {
        "comparator": {"name": str(closure["comparator"])},
        "evidence": evidence,
        "fixture": str(closure["fixture"]),
        "issues": list(closure["issues"]),
        "observations": copy.deepcopy(closure["observations"]),
        "owner": AUDIT_OWNER,
        "state": state,
    }
    if "compatibility_exception" in closure:
        fields["compatibility_exception"] = str(closure["compatibility_exception"])
    return fields


def build_manifest(
    population: Mapping[str, Any],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    members = list(population["members"])
    projection_maps = {
        oracle_version: {item["member_id"]: item for item in projection["members"]}
        for oracle_version, projection in projections.items()
    }
    groups = _implementation_groups(members, projections)
    cases = []
    for member in members:
        for oracle_version in SUPPORTED_GWPY:
            projection = projections[oracle_version]
            counterpart = projection_maps[oracle_version][member["member_id"]]
            fields: dict[str, Any]
            if counterpart["present"]:
                closure = TERMINAL_CLOSURES.get(str(member["member_id"]))
                if closure is None:
                    fields = {
                        "comparator": {"name": "pending"},
                        "evidence": {
                            "behavior": [],
                            "oracle_projection_digest": projection["digest"],
                        },
                        "fixture": PENDING_FIXTURE,
                        "issues": ["#639"],
                        "observations": {
                            "gwexpy": {"outcome": "pending"},
                            "gwpy": {"outcome": "pending"},
                        },
                        "owner": AUDIT_OWNER,
                        "state": "differential-required",
                    }
                else:
                    fields = _terminal_case_fields(closure, str(projection["digest"]))
            else:
                fields = {
                    "comparator": {"name": "counterpart-absence"},
                    "evidence": {
                        "behavior": [],
                        "oracle_projection_digest": projection["digest"],
                    },
                    "fixture": ABSENT_FIXTURE,
                    "issues": ["#639"],
                    "observations": {
                        "gwexpy": {
                            "kind": member["kind"],
                            "outcome": "attribute-present",
                        },
                        "gwpy": {"outcome": "attribute-absent"},
                    },
                    "owner": None,
                    "state": "GWexpy-only",
                }
            case: dict[str, Any] = {
                "case_key": "/".join(
                    (
                        str(member["public_class"]),
                        str(member["member"]),
                        oracle_version,
                        str(fields["fixture"]),
                    )
                ),
                "comparator": fields["comparator"],
                "counterpart_present": bool(counterpart["present"]),
                "evidence": fields["evidence"],
                "fixture": fields["fixture"],
                "gwpy_version": oracle_version,
                "implementation_group": groups[(member["member_id"], oracle_version)],
                "issues": fields["issues"],
                "member": member["member"],
                "member_id": member["member_id"],
                "observations": fields["observations"],
                "owner": fields["owner"],
                "public_class": member["public_class"],
                "state": fields["state"],
            }
            if "compatibility_exception" in fields:
                case["compatibility_exception"] = fields["compatibility_exception"]
            cases.append(case)
    cases.sort(key=_case_sort_key)
    manifest = {
        "cases": cases,
        "members": members,
        "oracle_projections": {
            oracle_version: projections[oracle_version]
            for oracle_version in SUPPORTED_GWPY
        },
        "policy": _manifest_policy(),
        "population_digest": population["digest"],
        "public_roots": list(population["public_roots"]),
        "schema": SCHEMA,
        "summary": calculate_summary(cases, members, projections),
    }
    return manifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def _validate_recursive_finite(value: Any) -> None:
    _require(
        not any(
            isinstance(item, float) and not math.isfinite(item)
            for item in walk_manifest_values(value)
        ),
        "manifest contains non-finite float",
    )


def _validate_projection(oracle_version: str, projection: Mapping[str, Any]) -> None:
    _require(projection.get("schema") == WORKER_SCHEMA, "invalid oracle schema")
    _require(
        projection.get("gwpy_version") == oracle_version,
        "oracle projection version mismatch",
    )
    expected_digest = digest_json(_projection_without_digest(projection))
    _require(
        projection.get("digest") == expected_digest, "oracle projection digest mismatch"
    )
    isolation = projection.get("isolation")
    _require(
        isolation
        == {
            "cwd_matches_expected": True,
            "gwexpy_absent_at_end": True,
            "gwexpy_absent_at_start": True,
            "isolated_flag": True,
            "no_user_site": True,
        },
        "invalid oracle isolation evidence",
    )
    members = projection.get("members")
    if not isinstance(members, list):
        raise InventoryError("oracle members must be a list")
    member_objects: list[dict[str, Any]] = []
    for item in members:
        if not isinstance(item, dict):
            raise InventoryError("oracle member must be an object")
        member_objects.append(item)
    ids = [item.get("member_id") for item in member_objects]
    _require(
        ids == sorted(ids, key=lambda item: str(item).encode("utf-8")),
        "oracle members are unsorted",
    )
    _require(len(ids) == len(set(ids)), "duplicate oracle member")
    for item in member_objects:
        present = item.get("present")
        _require(isinstance(present, bool), "oracle presence must be boolean")
        if present:
            _require(
                item.get("provider") is not None, "present counterpart lacks provider"
            )
            _require(item.get("kind") is not None, "present counterpart lacks kind")
            _require(
                item.get("raw_descriptor_identity") is not None,
                "present counterpart lacks raw identity",
            )
            provider = str(item["provider"])
            if provider.startswith("gwpy."):
                source = item.get("source")
                _require(
                    source is None or isinstance(source, dict),
                    "oracle source must be an object or null",
                )
                _require(
                    source is None or not str(source.get("path", "")).startswith("/"),
                    "absolute oracle source path",
                )
            else:
                _require(
                    item.get("source") is None,
                    "non-GWpy provider source must be normalized away",
                )
        else:
            for key in (
                "provider",
                "kind",
                "raw_descriptor_identity",
                "source",
                "descriptor",
            ):
                _require(
                    item.get(key) is None,
                    "absent counterpart contains provider evidence",
                )


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_test_reference_text(value: Any, label: str) -> None:
    _require(_is_nonempty_string(value), f"{label} reference must be a string")
    assert isinstance(value, str)
    _require(value == value.strip(), f"{label} reference is not canonical")
    path_text, separator, selector = value.partition("::")
    _require(separator == "::" and bool(selector.strip()), f"{label} selector missing")
    _require(selector == selector.strip(), f"{label} selector is not canonical")
    _require("\\" not in path_text, f"{label} reference is not POSIX-relative")
    relative = PurePosixPath(path_text)
    _require(
        not relative.is_absolute()
        and relative.as_posix() == path_text
        and relative.parts
        and relative.parts[0] == "tests"
        and relative.suffix == ".py"
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label} reference is not a canonical repo-relative test path",
    )
    repository = Path(__file__).resolve().parents[1]
    referenced = (repository / Path(*relative.parts)).resolve()
    try:
        referenced.relative_to(repository)
    except ValueError as exc:
        raise InventoryError(f"{label} reference escapes repository") from exc
    _require(referenced.is_file(), f"{label} reference file does not exist")


def _validate_test_reference(value: Any, label: str) -> None:
    _require(
        isinstance(value, dict) and set(value) == {"reference"},
        f"{label} reference schema mismatch",
    )
    _validate_test_reference_text(value["reference"], label)


def _validate_typed_observation(value: Any, label: str) -> None:
    _require(isinstance(value, dict), f"{label} observation must be an object")
    outcome = value.get("outcome")
    _require(
        _is_nonempty_string(outcome) and outcome != "pending",
        f"{label} observation outcome must be a concrete string",
    )
    exception_class = value.get("exception_class")
    if outcome == "exception":
        _require(
            _is_nonempty_string(exception_class),
            f"{label} exception_class must be a nonempty string",
        )
    elif "exception_class" in value:
        _require(
            _is_nonempty_string(exception_class),
            f"{label} exception_class must be a nonempty string",
        )


def _validate_terminal_observations(value: Any, label: str) -> None:
    _require(
        isinstance(value, dict) and set(value) == {"gwpy", "gwexpy"},
        f"{label} observation schema mismatch",
    )
    _validate_typed_observation(value["gwpy"], f"{label} gwpy")
    _validate_typed_observation(value["gwexpy"], f"{label} gwexpy")


def _validate_terminal_comparator(value: Any) -> None:
    _require(isinstance(value, dict), "terminal comparator must be an object")
    name = value.get("name")
    _require(
        _is_nonempty_string(name) and name != "pending",
        "terminal comparator name must be a concrete string",
    )
    assert isinstance(name, str)
    expected_keys = (
        {"name", "rtol", "atol"} if name.startswith("approximate") else {"name"}
    )
    _require(set(value) == expected_keys, "terminal comparator schema mismatch")
    if name.startswith("approximate"):
        for tolerance in ("rtol", "atol"):
            number = value[tolerance]
            _require(
                (type(number) is int and number >= 0)
                or (type(number) is float and math.isfinite(number) and number >= 0),
                f"terminal comparator {tolerance} must be finite and nonnegative",
            )


def _validate_terminal_issues(value: Any, *, fixed: bool) -> None:
    _require(
        isinstance(value, list)
        and all(_is_nonempty_string(issue) for issue in value)
        and "#639" in value,
        "behavioral terminal issues must contain #639 strings",
    )
    if fixed:
        _require(
            any(issue != "#639" for issue in value),
            "fixed case requires a specific issue reference beyond #639",
        )


def _validate_behavior_references(value: Any) -> None:
    _require(
        isinstance(value, list) and bool(value),
        "behavioral terminal lacks differential evidence",
    )
    for index, reference in enumerate(value):
        _validate_test_reference(reference, f"behavior[{index}]")


def _validate_fixed_evidence(evidence: Mapping[str, Any]) -> None:
    _require(
        set(evidence)
        == {
            "behavior",
            "green_test",
            "oracle_projection_digest",
            "pre_fix_mismatch",
        },
        "fixed evidence schema mismatch",
    )
    _validate_test_reference(evidence["green_test"], "green_test")
    mismatch = evidence["pre_fix_mismatch"]
    _require(
        isinstance(mismatch, dict) and set(mismatch) == {"reference", "gwpy", "gwexpy"},
        "pre_fix_mismatch schema mismatch",
    )
    _validate_test_reference_text(mismatch["reference"], "pre_fix_mismatch")
    _validate_typed_observation(mismatch["gwpy"], "pre_fix_mismatch gwpy")
    _validate_typed_observation(mismatch["gwexpy"], "pre_fix_mismatch gwexpy")
    _require(
        mismatch["gwpy"] != mismatch["gwexpy"],
        "pre_fix_mismatch observations must differ",
    )


def _validate_compatibility_exception(
    case: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    value = case["compatibility_exception"]
    _require(
        _is_nonempty_string(value),
        "compatibility_exception must be a nonempty string",
    )
    _require(
        value in ALLOWED_COMPATIBILITY_EXCEPTIONS,
        "unknown compatibility_exception",
    )
    _require(
        case.get("member_id") == TIMESERIESDICT_READ_MEMBER_ID
        and case.get("fixture") == "native-read-route",
        "compatibility_exception is only valid for TimeSeriesDict/read "
        "native-read-route",
    )
    _require(
        case.get("state") == "fixed",
        "compatibility_exception requires fixed state",
    )
    _require(
        case.get("counterpart_present") is True and observed.get("present") is True,
        "compatibility_exception requires a present counterpart",
    )
    issues = case.get("issues")
    _require(
        isinstance(issues, list) and "#611" in issues,
        "compatibility_exception requires issue #611",
    )
    evidence = case.get("evidence")
    behavior = evidence.get("behavior") if isinstance(evidence, dict) else None
    _require(
        isinstance(behavior, list)
        and behavior[:3]
        == [
            {"reference": TIMESERIESDICT_READ_PARITY_REFERENCE},
            {"reference": HDF5_AUTO_READ_REFERENCE},
            {"reference": NON_INTERSECTING_WINDOW_SAFETY_REFERENCE},
        ],
        "compatibility_exception lacks dedicated #611 safety evidence",
    )
    _require(
        case.get("observations") == NON_INTERSECTING_WINDOW_SAFETY_OBSERVATIONS,
        "compatibility_exception lacks dedicated #611 safety observations",
    )


def _validate_case(
    case: Mapping[str, Any],
    members: Mapping[str, Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
    projection_members: Mapping[str, Mapping[str, Mapping[str, Any]]],
    implementation_groups: Mapping[tuple[str, str], str | None],
) -> None:
    case_keys = set(case) if isinstance(case, dict) else set()
    _require(
        isinstance(case, dict)
        and CASE_KEYS <= case_keys
        and case_keys <= CASE_KEYS | OPTIONAL_CASE_KEYS,
        "case schema mismatch",
    )
    member_id = str(case.get("member_id"))
    _require(member_id in members, "orphan case member reference")
    member = members[member_id]
    oracle_version = str(case.get("gwpy_version"))
    _require(oracle_version in projections, "orphan case oracle reference")
    observed = projection_members[oracle_version][member_id]
    _require(
        case.get("public_class") == member["public_class"], "case public class mismatch"
    )
    _require(case.get("member") == member["member"], "case member mismatch")
    _require(
        case.get("counterpart_present") is observed["present"], "case presence mismatch"
    )
    _require(
        case.get("implementation_group")
        == implementation_groups[(member_id, oracle_version)],
        "case implementation group mismatch",
    )
    fixture = case.get("fixture")
    expected_key = "/".join(
        (
            str(member["public_class"]),
            str(member["member"]),
            oracle_version,
            str(fixture),
        )
    )
    _require(case.get("case_key") == expected_key, "case key mismatch")
    evidence = case.get("evidence")
    _require(isinstance(evidence, dict), "case evidence must be an object")
    _require(
        evidence.get("oracle_projection_digest")
        == projections[oracle_version]["digest"],
        "case oracle digest mismatch",
    )
    behavior = evidence.get("behavior")
    _require(isinstance(behavior, list), "behavior evidence must be a list")
    state = case.get("state")
    _require(isinstance(state, str), "case state must be a string")
    _require(state in {*TERMINAL_STATES, *PROVISIONAL_STATES}, "unknown case state")
    has_compatibility_exception = "compatibility_exception" in case
    if (
        any(
            item == NON_INTERSECTING_WINDOW_SAFETY_REFERENCE
            for item in walk_manifest_values(evidence)
        )
        and not has_compatibility_exception
    ):
        raise InventoryError(
            "dedicated #611 safety evidence requires compatibility_exception"
        )
    if (
        case.get("observations") == NON_INTERSECTING_WINDOW_SAFETY_OBSERVATIONS
        and not has_compatibility_exception
    ):
        raise InventoryError(
            "dedicated #611 safety observations require compatibility_exception"
        )
    if has_compatibility_exception:
        _validate_compatibility_exception(case, observed)
    if state == "GWexpy-only":
        _require(observed["present"] is False, "GWexpy-only counterpart is present")
        _require(fixture == ABSENT_FIXTURE, "GWexpy-only fixture mismatch")
        _require(
            case.get("comparator") == {"name": "counterpart-absence"},
            "GWexpy-only comparator mismatch",
        )
        _require(
            case.get("owner") is None, "GWexpy-only must not have behavioral owner"
        )
        _require(
            case.get("implementation_group") is None,
            "GWexpy-only cannot have implementation group",
        )
        _require(
            case.get("observations")
            == {
                "gwexpy": {"kind": member["kind"], "outcome": "attribute-present"},
                "gwpy": {"outcome": "attribute-absent"},
            },
            "GWexpy-only observation mismatch",
        )
        _require(case.get("issues") == ["#639"], "GWexpy-only issue mismatch")
        _require(
            evidence
            == {
                "behavior": [],
                "oracle_projection_digest": projections[oracle_version]["digest"],
            },
            "GWexpy-only evidence schema mismatch",
        )
    elif state == "differential-required":
        _require(observed["present"] is True, "pending differential lacks counterpart")
        _require(fixture == PENDING_FIXTURE, "pending differential fixture mismatch")
        _require(
            case.get("owner") == AUDIT_OWNER, "pending differential owner mismatch"
        )
        _require(case.get("issues") == ["#639"], "pending differential issue mismatch")
        _require(
            case.get("comparator") == {"name": "pending"},
            "pending differential comparator mismatch",
        )
        _require(
            case.get("observations")
            == {
                "gwexpy": {"outcome": "pending"},
                "gwpy": {"outcome": "pending"},
            },
            "pending differential observation mismatch",
        )
        _require(
            evidence
            == {
                "behavior": [],
                "oracle_projection_digest": projections[oracle_version]["digest"],
            },
            "pending differential evidence schema mismatch",
        )
        _require(
            case.get("implementation_group") is not None,
            "pending differential lacks implementation group",
        )
    elif state == "unreviewed":
        _require(case.get("owner") is not None, "unreviewed case lacks owner")
    elif state in BEHAVIORAL_TERMINAL_STATES:
        _require(observed["present"] is True, "behavioral terminal lacks counterpart")
        _require(
            _is_nonempty_string(fixture)
            and fixture not in {ABSENT_FIXTURE, PENDING_FIXTURE},
            "behavioral terminal fixture must be concrete",
        )
        _require(
            _is_nonempty_string(case.get("owner")),
            "behavioral terminal owner must be a nonempty string",
        )
        _validate_terminal_issues(case.get("issues"), fixed=state == "fixed")
        _validate_terminal_comparator(case.get("comparator"))
        _validate_terminal_observations(case.get("observations"), state)
        expected_evidence_keys = (
            {
                "behavior",
                "green_test",
                "oracle_projection_digest",
                "pre_fix_mismatch",
            }
            if state == "fixed"
            else {"behavior", "oracle_projection_digest"}
        )
        _require(
            set(evidence) == expected_evidence_keys,
            f"{state} evidence schema mismatch",
        )
        _validate_behavior_references(behavior)
        if state == "fixed":
            _validate_fixed_evidence(evidence)
        if state == "GWpy-fails":
            gwpy_observation = case["observations"]["gwpy"]
            _require(
                gwpy_observation.get("outcome") == "exception",
                "GWpy-fails lacks exception outcome",
            )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate schema, references, ordering, digests, and state evidence."""

    _validate_recursive_finite(manifest)
    _require(
        set(manifest)
        == {
            "cases",
            "members",
            "oracle_projections",
            "policy",
            "population_digest",
            "public_roots",
            "schema",
            "summary",
        },
        "inventory top-level schema mismatch",
    )
    _require(manifest.get("schema") == SCHEMA, "invalid inventory schema")
    policy = manifest.get("policy")
    _require(isinstance(policy, dict), "inventory policy must be an object")
    _require(set(policy) == set(_manifest_policy()), "policy schema mismatch")
    _require(
        policy.get("behavioral_owner") == AUDIT_OWNER,
        "behavioral owner policy mismatch",
    )
    _require(
        policy.get("upstream_dependency_provenance") == UPSTREAM_DEPENDENCY_PROVENANCE,
        "upstream dependency provenance policy mismatch",
    )
    _require(
        policy.get("implementation_base") == IMPLEMENTATION_BASE,
        "implementation base mismatch",
    )
    _require(
        policy.get("public_root_rule") == PUBLIC_ROOT_RULE, "public root rule mismatch"
    )
    _require(
        policy.get("member_walk_rule") == MEMBER_WALK_RULE, "member walk rule mismatch"
    )
    _require(
        policy.get("pristine_oracle_rule") == PRISTINE_ORACLE_RULE,
        "pristine oracle rule mismatch",
    )
    _require(
        policy.get("oracle_versions") == list(SUPPORTED_GWPY),
        "invalid oracle version policy",
    )
    _require(
        policy.get("terminal_states") == list(TERMINAL_STATES)
        and policy.get("provisional_states") == list(PROVISIONAL_STATES),
        "invalid state policy",
    )
    _require(
        policy.get("fixture_key")
        == ["public_class", "member", "gwpy_version", "fixture"],
        "invalid stable case key policy",
    )
    projections = manifest.get("oracle_projections")
    _require(isinstance(projections, dict), "oracle projections must be an object")
    _require(set(projections) == set(SUPPORTED_GWPY), "oracle projection set mismatch")
    for oracle_version in SUPPORTED_GWPY:
        _validate_projection(oracle_version, projections[oracle_version])
    roots = manifest.get("public_roots")
    members_list = manifest.get("members")
    cases = manifest.get("cases")
    _require(isinstance(roots, list), "public roots must be a list")
    _require(isinstance(members_list, list), "members must be a list")
    _require(isinstance(cases, list), "cases must be a list")
    _require(
        all(isinstance(item, dict) for item in roots),
        "public root must be an object",
    )
    _require(
        all(isinstance(item, dict) for item in members_list),
        "member must be an object",
    )
    root_ids = [item.get("public_class") for item in roots]
    _require(
        root_ids == sorted(root_ids, key=lambda item: str(item).encode("utf-8")),
        "public roots are unsorted",
    )
    _require(len(root_ids) == len(set(root_ids)), "duplicate public root")
    member_ids = [item.get("member_id") for item in members_list]
    _require(
        member_ids == sorted(member_ids, key=lambda item: str(item).encode("utf-8")),
        "members are unsorted",
    )
    _require(len(member_ids) == len(set(member_ids)), "duplicate member")
    members = {str(item["member_id"]): item for item in members_list}
    roots_by_id = {str(item["public_class"]): item for item in roots}
    for member in members_list:
        public_class = str(member.get("public_class"))
        _require(public_class in roots_by_id, "orphan member public root reference")
        _require(
            member.get("exports") == roots_by_id[public_class].get("exports"),
            "member export alias reference mismatch",
        )
        _require(
            member.get("counterpart_class")
            == roots_by_id[public_class].get("counterpart_class"),
            "member counterpart class mismatch",
        )
    projection_members = {
        oracle_version: {
            str(item["member_id"]): item
            for item in projections[oracle_version]["members"]
        }
        for oracle_version in SUPPORTED_GWPY
    }
    for oracle_version in SUPPORTED_GWPY:
        _require(
            set(projection_members[oracle_version]) == set(members),
            "oracle/member population mismatch",
        )
        for member_id, observed in projection_members[oracle_version].items():
            member = members[member_id]
            _require(
                observed.get("public_class") == member.get("public_class")
                and observed.get("member") == member.get("member")
                and observed.get("counterpart_class")
                == member.get("counterpart_class"),
                "oracle/source member reference mismatch",
            )
    implementation_groups = _implementation_groups(members_list, projections)
    for case in cases:
        case_keys = set(case) if isinstance(case, dict) else set()
        _require(
            isinstance(case, dict)
            and CASE_KEYS <= case_keys
            and case_keys <= CASE_KEYS | OPTIONAL_CASE_KEYS,
            "case schema mismatch",
        )
    _require(cases == sorted(cases, key=_case_sort_key), "cases are unsorted")
    case_keys = [case.get("case_key") for case in cases]
    _require(len(case_keys) == len(set(case_keys)), "duplicate case key")
    for case in cases:
        _validate_case(
            case,
            members,
            projections,
            projection_members,
            implementation_groups,
        )
    expected_pairs = {
        (member_id, oracle_version)
        for member_id in members
        for oracle_version in SUPPORTED_GWPY
    }
    actual_pairs = {
        (str(case["member_id"]), str(case["gwpy_version"])) for case in cases
    }
    _require(actual_pairs == expected_pairs, "missing or orphan version case")
    expected_population_digest = digest_json(
        {"public_roots": roots, "members": members_list}
    )
    _require(
        manifest.get("population_digest") == expected_population_digest,
        "population digest mismatch",
    )
    expected_summary = calculate_summary(cases, members_list, projections)
    _require(manifest.get("summary") == expected_summary, "summary mismatch")
    _require(
        not any(
            isinstance(value, str) and (value.startswith("/") or "0x" in value.lower())
            for value in walk_manifest_values(manifest)
        ),
        "manifest contains absolute path or address-bearing repr",
    )


def validate_source_population(
    manifest: Mapping[str, Any], population: Mapping[str, Any]
) -> None:
    _require(
        manifest.get("public_roots") == population.get("public_roots"),
        "public root population drift",
    )
    _require(
        manifest.get("members") == population.get("members"), "member population drift"
    )
    _require(
        manifest.get("population_digest") == population.get("digest"),
        "source/MRO digest drift",
    )


def validate_catalog_binding(
    manifest: Mapping[str, Any], population: Mapping[str, Any]
) -> None:
    """Require the checked artifact to match the reviewed code catalog exactly."""

    expected = build_manifest(population, manifest["oracle_projections"])
    _require(
        canonical_manifest_json(manifest) == canonical_manifest_json(expected),
        "manifest/catalog drift",
    )


def manifest_evidence_selectors(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every reviewed pytest selector in deterministic byte order."""

    selectors: set[str] = set()
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise InventoryError("manifest cases must be a list")
    for case in cases:
        try:
            evidence = case["evidence"]
            references = [
                *(entry["reference"] for entry in evidence["behavior"]),
            ]
            if "green_test" in evidence:
                references.append(evidence["green_test"]["reference"])
            if "pre_fix_mismatch" in evidence:
                references.append(evidence["pre_fix_mismatch"]["reference"])
        except (KeyError, TypeError) as exc:
            raise InventoryError("malformed evidence reference") from exc
        for reference in references:
            _validate_test_reference_text(reference, "evidence")
            assert isinstance(reference, str)
            path, _, selector = reference.partition("::")
            selector_name = selector.partition("[")[0]
            if (
                path == SELF_EVIDENCE_PATH
                and selector_name not in SAFE_SELF_EVIDENCE_SELECTORS
            ):
                raise InventoryError(
                    f"recursive inventory evidence selector is forbidden: {reference}"
                )
            selectors.add(reference)
    return tuple(sorted(selectors, key=lambda item: item.encode("utf-8")))


def validate_evidence_junit(path: Path) -> None:
    """Require the executed evidence set to be complete and entirely passing."""

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise InventoryError(
            f"cannot parse evidence JUnit: {type(exc).__name__}"
        ) from exc
    testcases = list(root.iter("testcase"))
    skipped = sum(any(child.tag == "skipped" for child in case) for case in testcases)
    failures = sum(any(child.tag == "failure" for child in case) for case in testcases)
    errors = sum(any(child.tag == "error" for child in case) for case in testcases)
    if len(testcases) != EXPECTED_EVIDENCE_CASES:
        raise InventoryError(
            "evidence execution expected "
            f"{EXPECTED_EVIDENCE_CASES} cases, got {len(testcases)}"
        )
    if skipped or failures or errors:
        raise InventoryError(
            "evidence execution is not entirely passing: "
            f"skipped={skipped}, failures={failures}, errors={errors}"
        )


def run_evidence_pytest(
    repository: Path,
    selectors: Sequence[str],
    *,
    execute: bool,
    python_executable: str | None = None,
) -> str:
    """Collect or execute evidence nodes through a bounded, shell-free argv."""

    if not selectors:
        raise InventoryError("manifest has no evidence selectors")
    if execute and os.environ.get(EVIDENCE_CHILD_ENV) == "1":
        raise InventoryError("recursive evidence execution is forbidden")
    command = [python_executable or sys.executable, "-m", "pytest"]
    if execute:
        command.append("-q")
        action = "execution"
    else:
        command.extend(("--collect-only", "-qq"))
        action = "collection"
    command.extend(("--maxfail=1", "--tb=short", "-p", "no:cacheprovider"))
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    environment[EVIDENCE_CHILD_ENV] = "1"
    with tempfile.TemporaryDirectory(prefix="gwexpy-override-evidence-") as temporary:
        junit = Path(temporary) / "evidence.xml"
        if execute:
            command.append(f"--junitxml={junit}")
        command.extend(("--", *selectors))
        if sum(len(item.encode("utf-8")) + 1 for item in command) > 100_000:
            raise InventoryError(f"evidence {action} argv exceeds limit")
        completed = _run_bounded_subprocess(
            command,
            cwd=repository,
            environment=environment,
            input_text=None,
            timeout=EVIDENCE_TIMEOUT_SECONDS,
            label=f"evidence {action}",
        )
        if execute and completed.returncode == 0:
            validate_evidence_junit(junit)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if detail:
            detail = f": {detail[-4000:]}"
        raise InventoryError(
            f"evidence {action} failed with exit {completed.returncode}{detail}"
        )
    return completed.stdout


def _refuse_behavioral_overwrite(path: Path) -> None:
    if not path.exists():
        return
    existing = load_json_strict(path)
    cases = existing.get("cases")
    if not isinstance(cases, list):
        raise InventoryError("refusing to overwrite malformed existing manifest")
    preserved = [
        case for case in cases if case.get("state") in BEHAVIORAL_TERMINAL_STATES
    ]
    if preserved:
        raise InventoryError(
            "refusing to overwrite existing fixed/no-finding/GWpy-fails behavioral evidence"
        )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--oracle-worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--oracle-python", action="append", default=[])
    parser.add_argument("--require-terminal", action="store_true")
    parser.add_argument("--execute-evidence", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.oracle_worker:
        if (
            args.manifest is not None
            or args.oracle_python
            or args.require_terminal
            or args.execute_evidence
        ):
            print("oracle worker accepts stdin only", file=sys.stderr)
            return 2
        return _worker_main()
    try:
        if args.manifest is None:
            raise InventoryError("--manifest is required")
        if args.require_terminal and not args.check:
            raise InventoryError("--require-terminal is check-only")
        if args.execute_evidence and not args.check:
            raise InventoryError("--execute-evidence is check-only")
        if args.execute_evidence and not args.require_terminal:
            raise InventoryError("--execute-evidence requires --require-terminal")
        oracles = parse_oracle_arguments(args.oracle_python)
        if args.write and set(oracles) != set(SUPPORTED_GWPY):
            raise InventoryError("--write requires exactly GWpy 4.0.1 and 4.0.2")
        repository = Path(__file__).resolve().parents[1]
        population = build_source_population(repository)
        current_projections = {
            oracle_version: run_pristine_oracle(
                Path(__file__), oracle_version, executable, population["members"]
            )
            for oracle_version, executable in sorted(oracles.items())
        }
        if args.write:
            _refuse_behavioral_overwrite(args.manifest)
            manifest = build_manifest(population, current_projections)
            validate_manifest(manifest)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                canonical_manifest_json(manifest), encoding="utf-8"
            )
            print(f"wrote {args.manifest}")
            return 0
        manifest = load_json_strict(args.manifest)
        if args.manifest.read_text(encoding="utf-8") != canonical_manifest_json(
            manifest
        ):
            raise InventoryError("manifest serialization is not canonical JSON")
        validate_manifest(manifest)
        validate_source_population(manifest, population)
        for oracle_version, projection in current_projections.items():
            if manifest["oracle_projections"][oracle_version] != projection:
                raise InventoryError(
                    f"oracle projection drift for GWpy {oracle_version}"
                )
        validate_catalog_binding(manifest, population)
        if args.require_terminal:
            require_terminal_cases(manifest["cases"])
        if args.require_terminal:
            selectors = manifest_evidence_selectors(manifest)
            for executable in oracles.values():
                run_evidence_pytest(
                    repository,
                    selectors,
                    execute=False,
                    python_executable=executable,
                )
            print(f"evidence collection passed: selectors={len(selectors)}")
            if args.execute_evidence:
                for executable in oracles.values():
                    evidence_output = run_evidence_pytest(
                        repository,
                        selectors,
                        execute=True,
                        python_executable=executable,
                    )
                    sys.stdout.write(evidence_output)
                print(
                    "evidence execution passed: "
                    f"selectors={len(selectors)}, cases={EXPECTED_EVIDENCE_CASES}"
                )
        print(
            "inventory check passed: "
            f"members={len(manifest['members'])}, "
            f"gwpy={','.join(sorted(current_projections))}"
        )
        return 0
    except InventoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        print(f"malformed input: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
