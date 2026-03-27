"""Tests for gwexpy/io/collection_dir.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwexpy.io.collection_dir import (
    MANIFEST_NAME,
    _safe_stem,
    iter_collection_dir_entries,
    read_collection_dir,
    write_collection_dir,
)


# ---------------------------------------------------------------------------
# _safe_stem
# ---------------------------------------------------------------------------


class TestSafeStem:
    def test_basic(self):
        assert _safe_stem("hello", default="item") == "hello"

    def test_slash_replaced(self):
        result = _safe_stem("H1:GDS/STRAIN", default="item")
        assert "/" not in result

    def test_empty_returns_default(self):
        assert _safe_stem("", default="item") == "item"

    def test_whitespace_only_returns_default(self):
        assert _safe_stem("   ", default="item") == "item"

    def test_special_chars_replaced(self):
        result = _safe_stem("H1:PSL PWR@100", default="item")
        assert " " not in result
        assert "@" not in result


# ---------------------------------------------------------------------------
# write_collection_dir
# ---------------------------------------------------------------------------


def _dummy_writer(value, filepath, fmt):
    """Write value as plain text."""
    filepath.write_text(str(value))


class TestWriteCollectionDir:
    def test_basic_write(self, tmp_path):
        entries = [("key1", "val1"), ("key2", "val2")]
        dp = write_collection_dir(
            tmp_path / "out",
            kind="Test",
            entry_format="txt",
            entries=entries,
            writer=_dummy_writer,
        )
        assert dp.is_dir()
        manifest = json.loads((dp / MANIFEST_NAME).read_text())
        assert manifest["kind"] == "Test"
        assert len(manifest["entries"]) == 2

    def test_directory_created(self, tmp_path):
        dp = write_collection_dir(
            tmp_path / "subdir" / "out",
            kind="Test",
            entry_format="txt",
            entries=[("a", "1")],
            writer=_dummy_writer,
        )
        assert dp.is_dir()

    def test_existing_file_raises_not_a_directory(self, tmp_path):
        f = tmp_path / "myfile.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            write_collection_dir(
                f,
                kind="Test",
                entry_format="txt",
                entries=[],
                writer=_dummy_writer,
            )

    def test_non_empty_dir_raises_without_overwrite(self, tmp_path):
        dp = tmp_path / "out"
        dp.mkdir()
        (dp / "existing.txt").write_text("x")
        with pytest.raises(FileExistsError):
            write_collection_dir(
                dp,
                kind="Test",
                entry_format="txt",
                entries=[("a", "1")],
                writer=_dummy_writer,
            )

    def test_overwrite_replaces_existing(self, tmp_path):
        dp = tmp_path / "out"
        dp.mkdir()
        (dp / "existing.txt").write_text("x")
        result = write_collection_dir(
            dp,
            kind="Test",
            entry_format="txt",
            entries=[("a", "val_a")],
            writer=_dummy_writer,
            overwrite=True,
        )
        assert result == dp

    def test_meta_getter_included(self, tmp_path):
        def meta_getter(value):
            return {"length": len(value)}

        entries = [("ch1", "hello")]
        dp = write_collection_dir(
            tmp_path / "out",
            kind="Test",
            entry_format="txt",
            entries=entries,
            writer=_dummy_writer,
            meta_getter=meta_getter,
        )
        manifest = json.loads((dp / MANIFEST_NAME).read_text())
        assert manifest["entries"][0].get("meta") == {"length": 5}

    def test_meta_getter_empty_meta_not_included(self, tmp_path):
        def meta_getter(value):
            return {}  # empty dict → not included

        entries = [("ch1", "hello")]
        dp = write_collection_dir(
            tmp_path / "out",
            kind="Test",
            entry_format="txt",
            entries=entries,
            writer=_dummy_writer,
            meta_getter=meta_getter,
        )
        manifest = json.loads((dp / MANIFEST_NAME).read_text())
        assert "meta" not in manifest["entries"][0]

    def test_returns_path(self, tmp_path):
        result = write_collection_dir(
            tmp_path / "out",
            kind="Test",
            entry_format="csv",
            entries=[],
            writer=_dummy_writer,
        )
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# iter_collection_dir_entries
# ---------------------------------------------------------------------------


def _make_dir_with_manifest(tmp_path, kind="Test", entry_format="txt", entries=None):
    """Create a directory with a manifest and entry files."""
    dp = tmp_path / "col"
    dp.mkdir()
    entries = entries or [("a", "va"), ("b", "vb")]
    manifest_entries = []
    for key, val in entries:
        fn = f"{key}.{entry_format}"
        (dp / fn).write_text(str(val))
        manifest_entries.append({"key": key, "filename": fn})
    manifest = {
        "version": 1,
        "kind": kind,
        "entry_format": entry_format,
        "entries": manifest_entries,
    }
    (dp / MANIFEST_NAME).write_text(json.dumps(manifest))
    return dp


class TestIterCollectionDirEntries:
    def test_reads_manifest(self, tmp_path):
        dp = _make_dir_with_manifest(tmp_path)
        fmt, pairs = iter_collection_dir_entries(dp)
        assert fmt == "txt"
        assert len(pairs) == 2

    def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            iter_collection_dir_entries(f)

    def test_kind_mismatch_raises(self, tmp_path):
        dp = _make_dir_with_manifest(tmp_path, kind="RealKind")
        with pytest.raises(ValueError, match="kind mismatch"):
            iter_collection_dir_entries(dp, expected_kind="WrongKind")

    def test_format_mismatch_raises(self, tmp_path):
        dp = _make_dir_with_manifest(tmp_path, entry_format="txt")
        with pytest.raises(ValueError, match="format mismatch"):
            iter_collection_dir_entries(dp, entry_format="csv")

    def test_meta_non_dict_falls_back_to_empty(self, tmp_path):
        dp = tmp_path / "col"
        dp.mkdir()
        fn = "item.txt"
        (dp / fn).write_text("v")
        manifest = {
            "version": 1,
            "kind": "Test",
            "entry_format": "txt",
            "entries": [{"key": "x", "filename": fn, "meta": "not_a_dict"}],
        }
        (dp / MANIFEST_NAME).write_text(json.dumps(manifest))
        fmt, pairs = iter_collection_dir_entries(dp)
        _, _, meta = pairs[0]
        assert meta == {}  # non-dict meta falls back to {}

    def test_no_manifest_infers_from_csv(self, tmp_path):
        dp = tmp_path / "col"
        dp.mkdir()
        (dp / "channel_a.csv").write_text("1,2,3")
        (dp / "channel_b.csv").write_text("4,5,6")
        fmt, pairs = iter_collection_dir_entries(dp)
        assert fmt == "csv"
        assert len(pairs) == 2

    def test_no_manifest_no_csv_raises(self, tmp_path):
        dp = tmp_path / "col"
        dp.mkdir()
        (dp / "random.json").write_text("{}")
        with pytest.raises(FileNotFoundError):
            iter_collection_dir_entries(dp)

    def test_no_manifest_with_entry_format(self, tmp_path):
        dp = tmp_path / "col"
        dp.mkdir()
        (dp / "item.txt").write_text("data")
        fmt, pairs = iter_collection_dir_entries(dp, entry_format="txt")
        assert fmt == "txt"
        assert len(pairs) == 1

    def test_no_manifest_format_no_match_raises(self, tmp_path):
        dp = tmp_path / "col"
        dp.mkdir()
        (dp / "item.txt").write_text("data")
        with pytest.raises(FileNotFoundError):
            iter_collection_dir_entries(dp, entry_format="csv")


# ---------------------------------------------------------------------------
# read_collection_dir
# ---------------------------------------------------------------------------


class TestReadCollectionDir:
    def test_basic_read(self, tmp_path):
        dp = _make_dir_with_manifest(tmp_path, entries=[("a", "hello")])
        fmt, result = read_collection_dir(
            dp,
            expected_kind=None,
            entry_format=None,
            reader=lambda path, fmt: path.read_text(),
        )
        assert fmt == "txt"
        assert len(result) == 1
        key, val, meta = result[0]
        assert key == "a"
        assert val == "hello"
