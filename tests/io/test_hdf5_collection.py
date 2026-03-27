"""Tests for gwexpy/io/hdf5_collection.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import pytest

from gwexpy.io.hdf5_collection import (
    KEYMAP_ATTR,
    LAYOUT_DATASET,
    LAYOUT_GROUP,
    ORDER_ATTR,
    detect_hdf5_layout,
    ensure_hdf5_file,
    normalize_layout,
    read_hdf5_keymap,
    read_hdf5_order,
    safe_hdf5_key,
    unique_hdf5_key,
    write_hdf5_manifest,
)


# ---------------------------------------------------------------------------
# safe_hdf5_key
# ---------------------------------------------------------------------------


class TestSafeHdf5Key:
    def test_basic(self):
        assert safe_hdf5_key("hello") == "hello"

    def test_slash_replaced(self):
        key = safe_hdf5_key("H1:GDS/STRAIN")
        assert "/" not in key

    def test_empty_returns_default(self):
        assert safe_hdf5_key("", default="item") == "item"

    def test_whitespace_returns_default(self):
        assert safe_hdf5_key("   ") == "item"

    def test_special_chars_replaced(self):
        key = safe_hdf5_key("H1:PSL PWR@100")
        assert " " not in key


# ---------------------------------------------------------------------------
# unique_hdf5_key
# ---------------------------------------------------------------------------


class TestUniqueHdf5Key:
    def test_first_time_no_suffix(self):
        used: set[str] = set()
        result = unique_hdf5_key("ch", used=used)
        assert result == "ch"
        assert "ch" in used

    def test_collision_adds_suffix(self):
        used = {"ch"}
        result = unique_hdf5_key("ch", used=used)
        assert result == "ch__1"

    def test_multiple_collisions(self):
        used = {"ch", "ch__1", "ch__2"}
        result = unique_hdf5_key("ch", used=used)
        assert result == "ch__3"


# ---------------------------------------------------------------------------
# detect_hdf5_layout
# ---------------------------------------------------------------------------


class TestDetectHdf5Layout:
    def test_empty_file_returns_none(self, tmp_path):
        fp = tmp_path / "empty.h5"
        with h5py.File(fp, "w") as h5f:
            result = detect_hdf5_layout(h5f)
        assert result is None

    def test_datasets_returns_dataset_layout(self, tmp_path):
        fp = tmp_path / "ds.h5"
        with h5py.File(fp, "w") as h5f:
            h5f.create_dataset("a", data=[1, 2, 3])
            h5f.create_dataset("b", data=[4, 5, 6])
            result = detect_hdf5_layout(h5f)
        assert result == LAYOUT_DATASET

    def test_groups_returns_group_layout(self, tmp_path):
        fp = tmp_path / "grp.h5"
        with h5py.File(fp, "w") as h5f:
            h5f.create_group("grp_a")
            h5f.create_group("grp_b")
            result = detect_hdf5_layout(h5f)
        assert result == LAYOUT_GROUP

    def test_mixed_returns_none(self, tmp_path):
        fp = tmp_path / "mixed.h5"
        with h5py.File(fp, "w") as h5f:
            h5f.create_dataset("ds", data=[1])
            h5f.create_group("grp")
            result = detect_hdf5_layout(h5f)
        assert result is None


# ---------------------------------------------------------------------------
# write_hdf5_manifest & read_hdf5_keymap / read_hdf5_order
# ---------------------------------------------------------------------------


class TestWriteReadManifest:
    def test_write_and_read_keymap(self, tmp_path):
        fp = tmp_path / "test.h5"
        keymap = {"H1:A": "H1_A", "H1:B": "H1_B"}
        with h5py.File(fp, "w") as h5f:
            write_hdf5_manifest(h5f, kind="Test", layout=LAYOUT_DATASET, keymap=keymap, order=["H1:A", "H1:B"])
            result = read_hdf5_keymap(h5f)
        assert result == keymap

    def test_write_and_read_order(self, tmp_path):
        fp = tmp_path / "test.h5"
        order = ["H1:B", "H1:A"]
        with h5py.File(fp, "w") as h5f:
            write_hdf5_manifest(h5f, kind="Test", layout=LAYOUT_DATASET, keymap={}, order=order)
            result = read_hdf5_order(h5f)
        assert result == order

    def test_read_keymap_missing_attr_returns_empty(self, tmp_path):
        fp = tmp_path / "test.h5"
        with h5py.File(fp, "w") as h5f:
            result = read_hdf5_keymap(h5f)
        assert result == {}

    def test_read_order_missing_attr_returns_empty(self, tmp_path):
        fp = tmp_path / "test.h5"
        with h5py.File(fp, "w") as h5f:
            result = read_hdf5_order(h5f)
        assert result == []

    def test_read_keymap_bytes_value(self, tmp_path):
        """Handle bytes-encoded KEYMAP_ATTR."""
        fp = tmp_path / "test.h5"
        keymap = {"a": "A"}
        with h5py.File(fp, "w") as h5f:
            # Write as bytes to trigger the bytes branch
            h5f.attrs[KEYMAP_ATTR] = json.dumps(keymap).encode("utf-8")
            result = read_hdf5_keymap(h5f)
        assert result == keymap

    def test_read_order_bytes_value(self, tmp_path):
        """Handle bytes-encoded ORDER_ATTR."""
        fp = tmp_path / "test.h5"
        order = ["x", "y"]
        with h5py.File(fp, "w") as h5f:
            h5f.attrs[ORDER_ATTR] = json.dumps(order).encode("utf-8")
            result = read_hdf5_order(h5f)
        assert result == order

    def test_read_keymap_invalid_json_returns_empty(self, tmp_path):
        fp = tmp_path / "test.h5"
        with h5py.File(fp, "w") as h5f:
            h5f.attrs[KEYMAP_ATTR] = "not valid json {{"
            result = read_hdf5_keymap(h5f)
        assert result == {}

    def test_read_order_invalid_json_returns_empty(self, tmp_path):
        fp = tmp_path / "test.h5"
        with h5py.File(fp, "w") as h5f:
            h5f.attrs[ORDER_ATTR] = "[[["
            result = read_hdf5_order(h5f)
        assert result == []

    def test_read_keymap_non_dict_returns_empty(self, tmp_path):
        """If JSON is valid but not a dict, return empty."""
        fp = tmp_path / "test.h5"
        with h5py.File(fp, "w") as h5f:
            h5f.attrs[KEYMAP_ATTR] = json.dumps([1, 2, 3])
            result = read_hdf5_keymap(h5f)
        assert result == {}

    def test_read_order_non_list_returns_empty(self, tmp_path):
        """If JSON is valid but not a list, return empty."""
        fp = tmp_path / "test.h5"
        with h5py.File(fp, "w") as h5f:
            h5f.attrs[ORDER_ATTR] = json.dumps({"a": "b"})
            result = read_hdf5_order(h5f)
        assert result == []


# ---------------------------------------------------------------------------
# ensure_hdf5_file
# ---------------------------------------------------------------------------


class TestEnsureHdf5File:
    def test_creates_new_file(self, tmp_path):
        fp = tmp_path / "new.h5"
        with ensure_hdf5_file(fp) as h5f:
            assert isinstance(h5f, h5py.File)

    def test_overwrite_mode(self, tmp_path):
        fp = tmp_path / "existing.h5"
        # Create the file first
        with h5py.File(fp, "w") as h5f:
            h5f.create_dataset("old_data", data=[1, 2, 3])
        # Open with overwrite=True (should use "w" mode)
        with ensure_hdf5_file(fp, overwrite=True) as h5f:
            assert "old_data" not in h5f  # overwritten


# ---------------------------------------------------------------------------
# normalize_layout
# ---------------------------------------------------------------------------


class TestNormalizeLayout:
    def test_none_returns_dataset(self):
        assert normalize_layout(None) == LAYOUT_DATASET

    def test_dataset_aliases(self):
        for alias in ["gwpy", "dataset", "dataset-per-entry"]:
            assert normalize_layout(alias) == LAYOUT_DATASET

    def test_group_aliases(self):
        for alias in ["group", "group-per-entry", "legacy"]:
            assert normalize_layout(alias) == LAYOUT_GROUP

    def test_case_insensitive(self):
        assert normalize_layout("Dataset") == LAYOUT_DATASET
        assert normalize_layout("GROUP") == LAYOUT_GROUP

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown HDF5 layout"):
            normalize_layout("invalid_layout")
