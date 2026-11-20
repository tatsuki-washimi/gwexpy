"""Tests for project-wide HDF5 schema versioning and unknown-field policy (#658)."""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from gwexpy.io.schema import (
    SCHEMA_ATTR,
    SCHEMA_ATTRS,
    SCHEMA_VERSION_ATTR,
    classify_unknown_attrs,
    read_and_validate_schema,
    write_schema_attrs,
)


class TestSchemaAttrsRoundtrip:
    def test_write_and_read_valid_schema(self, tmp_path):
        path = tmp_path / "schema.h5"
        with h5py.File(path, "w") as h5f:
            write_schema_attrs(h5f, "gwexpy.segment_table", 2)

        with h5py.File(path, "r") as h5f:
            assert h5f.attrs[SCHEMA_ATTR] == "gwexpy.segment_table"
            assert int(h5f.attrs[SCHEMA_VERSION_ATTR]) == 2
            version = read_and_validate_schema(
                h5f, "gwexpy.segment_table", max_supported_version=2
            )
            assert version == 2
            assert isinstance(version, int)

    def test_older_version_within_max_supported_succeeds(self, tmp_path):
        path = tmp_path / "older.h5"
        with h5py.File(path, "w") as h5f:
            write_schema_attrs(h5f, "gwexpy.field", 1)

        with h5py.File(path, "r") as h5f:
            version = read_and_validate_schema(
                h5f, "gwexpy.field", max_supported_version=3
            )
            assert version == 1

    def test_missing_version_returns_none_with_legacy_warning(self, tmp_path):
        path = tmp_path / "legacy.h5"
        with h5py.File(path, "w") as h5f:
            h5f.create_dataset("data", data=[1.0, 2.0])

        with h5py.File(path, "r") as h5f:
            with pytest.warns(UserWarning, match="legacy"):
                version = read_and_validate_schema(
                    h5f, "gwexpy.segment_table", max_supported_version=1
                )
            assert version is None

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                version_quiet = read_and_validate_schema(
                    h5f,
                    "gwexpy.segment_table",
                    max_supported_version=1,
                    warn_on_legacy=False,
                )
            assert version_quiet is None
            assert caught == []

    def test_newer_version_raises_value_error_citing_versions(self, tmp_path):
        path = tmp_path / "future.h5"
        with h5py.File(path, "w") as h5f:
            write_schema_attrs(h5f, "gwexpy.segment_table", 5)

        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match=r"version 5.*maximum supported.*2"):
                read_and_validate_schema(
                    h5f, "gwexpy.segment_table", max_supported_version=2
                )

    def test_schema_mismatch_raises_value_error(self, tmp_path):
        path = tmp_path / "mismatch.h5"
        with h5py.File(path, "w") as h5f:
            write_schema_attrs(h5f, "gwexpy.other", 1)

        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="Schema mismatch"):
                read_and_validate_schema(
                    h5f, "gwexpy.segment_table", max_supported_version=1
                )

    def test_version_without_schema_name_raises_value_error(self, tmp_path):
        path = tmp_path / "no_name.h5"
        with h5py.File(path, "w") as h5f:
            h5f.attrs[SCHEMA_VERSION_ATTR] = 1

        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="Missing 'gwexpy_schema'"):
                read_and_validate_schema(
                    h5f, "gwexpy.segment_table", max_supported_version=1
                )

    def test_bytes_schema_name_is_accepted(self, tmp_path):
        path = tmp_path / "bytes_schema.h5"
        with h5py.File(path, "w") as h5f:
            h5f.attrs[SCHEMA_ATTR] = b"gwexpy.segment_table"
            h5f.attrs[SCHEMA_VERSION_ATTR] = np.int32(1)

        with h5py.File(path, "r") as h5f:
            version = read_and_validate_schema(
                h5f, "gwexpy.segment_table", max_supported_version=1
            )
            assert version == 1

    @pytest.mark.parametrize(
        "bad_version",
        [True, False, np.bool_(True), 1.5, "1", 0, -1],
    )
    def test_invalid_version_in_file_raises_value_error(self, tmp_path, bad_version):
        path = tmp_path / "bad_ver.h5"
        with h5py.File(path, "w") as h5f:
            h5f.attrs[SCHEMA_ATTR] = "gwexpy.segment_table"
            h5f.attrs[SCHEMA_VERSION_ATTR] = bad_version

        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="Invalid 'gwexpy_schema_version'"):
                read_and_validate_schema(
                    h5f, "gwexpy.segment_table", max_supported_version=5
                )

    @pytest.mark.parametrize(
        "bad_version",
        [True, False, np.bool_(True), 1.5, "1", 0, -2, None],
    )
    def test_write_rejects_invalid_version(self, tmp_path, bad_version):
        path = tmp_path / "bad_write.h5"
        with h5py.File(path, "w") as h5f:
            with pytest.raises(ValueError, match="Invalid 'gwexpy_schema_version'"):
                write_schema_attrs(h5f, "gwexpy.segment_table", bad_version)

    @pytest.mark.parametrize("bad_schema", ["", "   ", 123, None])
    def test_write_and_read_reject_invalid_schema_name(self, tmp_path, bad_schema):
        path = tmp_path / "bad_name.h5"
        with h5py.File(path, "w") as h5f:
            with pytest.raises(ValueError, match="schema_name"):
                write_schema_attrs(h5f, bad_schema, 1)
            with pytest.raises(ValueError, match="expected_schema"):
                read_and_validate_schema(h5f, bad_schema, 1)


class TestClassifyUnknownAttrs:
    def test_preserves_external_attrs_and_warns_on_unknown_gwexpy(self):
        attrs = {
            "gwexpy_schema": "gwexpy.segment_table",
            "gwexpy_schema_version": 1,
            "gwexpy_future_flag": True,
            "_gwexpy_internal": "abc",
            "gwdama_author": "KAGRA",
            "instrument": "K1",
        }
        with pytest.warns(UserWarning, match="Ignoring unknown GWexpy attributes") as w:
            external = classify_unknown_attrs(attrs, known_keys=SCHEMA_ATTRS)

        assert len(w) == 1
        msg = str(w[0].message)
        assert "gwexpy_future_flag" in msg
        assert "_gwexpy_internal" in msg
        assert external == {"gwdama_author": "KAGRA", "instrument": "K1"}

    def test_strict_mode_rejects_unknown_gwexpy_attrs(self):
        attrs = {
            "gwexpy_schema": "gwexpy.segment_table",
            "gwexpy_unexpected": 42,
            "external_note": "keep",
        }
        with pytest.raises(
            ValueError, match="Unknown GWexpy attributes.*gwexpy_unexpected"
        ):
            classify_unknown_attrs(attrs, known_keys={"gwexpy_schema"}, strict=True)

    def test_strict_mode_allows_external_attrs_when_no_unknown_gwexpy(self):
        attrs = {
            "gwexpy_schema": "gwexpy.segment_table",
            "external_note": "keep",
            "custom_id": 7,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            external = classify_unknown_attrs(
                attrs, known_keys={"gwexpy_schema"}, strict=True
            )
        assert caught == []
        assert external == {"external_note": "keep", "custom_id": 7}

    def test_custom_prefixes_and_hdf5_roundtrip(self, tmp_path):
        src = tmp_path / "src.h5"
        dst = tmp_path / "dst.h5"
        with h5py.File(src, "w") as h5f:
            write_schema_attrs(h5f, "gwexpy.segment_table", 1)
            h5f.attrs["ext_tag"] = "observing-run"
            h5f.attrs["ext_count"] = 12
            h5f.attrs["custom_internal"] = "drop"

        with h5py.File(src, "r") as h5f:
            read_and_validate_schema(h5f, "gwexpy.segment_table", 1)
            with pytest.warns(UserWarning, match="custom_internal"):
                external = classify_unknown_attrs(
                    h5f.attrs,
                    known_keys=SCHEMA_ATTRS,
                    gwexpy_prefixes=("gwexpy_", "_gwexpy_", "custom_"),
                )

        with h5py.File(dst, "w") as h5f:
            write_schema_attrs(h5f, "gwexpy.segment_table", 1)
            for k, v in external.items():
                h5f.attrs[k] = v

        with h5py.File(dst, "r") as h5f:
            assert h5f.attrs["ext_tag"] == "observing-run"
            assert int(h5f.attrs["ext_count"]) == 12
            assert "custom_internal" not in h5f.attrs
