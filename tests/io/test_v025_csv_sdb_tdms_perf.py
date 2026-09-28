"""F2 CSV parser and direct-writer performance contracts."""

from __future__ import annotations

import ast
import inspect
import io
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io import csv_enhanced
from gwexpy.timeseries.io.csv_config import ColumnSpec, CSVFormatConfig


def _numeric_csv(path: Path, *, rows: int, channels: int) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index in range(rows):
            values = ",".join(str(index + channel) for channel in range(channels))
            stream.write(f"{index},{values}\n")


def test_plain_all_column_parser_uses_bounded_numpy_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "all.csv"
    _numeric_csv(path, rows=32, channels=3)
    monkeypatch.setattr(csv_enhanced, "_MAX_CSV_MATRIX_CHUNK_BYTES", 128)
    monkeypatch.setattr(csv_enhanced, "_try_plain_numeric_file", lambda *_: None)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("whole-file read_text used"),
    )
    converted_chunks: list[int] = []
    original = csv_enhanced._convert_numeric_chunk

    def record_chunk(rows: list[list[str]], lines: list[int], width: int) -> np.ndarray:
        result = original(rows, lines, width)
        converted_chunks.append(result.nbytes)
        return result

    monkeypatch.setattr(csv_enhanced, "_convert_numeric_chunk", record_chunk)
    result = csv_enhanced.read_timeseriesdict_csv(path)

    assert len(converted_chunks) > 1
    assert max(converted_chunks) <= 128
    assert list(result) == ["ch1", "ch2", "ch3"]
    np.testing.assert_array_equal(result["ch3"].value, np.arange(32) + 2)


def test_plain_numeric_file_uses_numpy_reader_and_keeps_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "plain.csv"
    _numeric_csv(path, rows=32, channels=3)
    calls = 0
    original = csv_enhanced.np.loadtxt

    def count_loadtxt(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(csv_enhanced.np, "loadtxt", count_loadtxt)
    result = csv_enhanced.read_timeseriesdict_csv(path)
    assert calls == 1
    np.testing.assert_array_equal(result["ch3"].value, np.arange(32) + 2)
    np.testing.assert_array_equal(result["ch3"].times.value, np.arange(32))


def test_plain_numeric_file_scans_successful_input_once() -> None:
    class CountingStream:
        def __init__(self) -> None:
            self.source = io.StringIO("0,1\n1,2\n2,3\n")
            self.iterations = 0

        def seekable(self) -> bool:
            return True

        def tell(self) -> int:
            return self.source.tell()

        def seek(self, *args: int) -> int:
            return self.source.seek(*args)

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            return iter(self.source)

    stream = CountingStream()
    parsed = csv_enhanced._try_plain_numeric_file(stream, CSVFormatConfig())
    assert parsed is not None
    _, matrix, _, times, lines, width = parsed
    assert stream.iterations == 1
    assert width == 2
    assert lines == [1, 2, 3]
    assert times == {0: ["0", "1", "2"]}
    np.testing.assert_array_equal(matrix, [[0, 1], [1, 2], [2, 3]])


def test_selected_parser_retains_only_requested_values(tmp_path: Path) -> None:
    path = tmp_path / "selected.csv"
    _numeric_csv(path, rows=100, channels=12)
    cfg = csv_enhanced.CSVFormatConfig()

    metadata, matrix, selected, timestamps, lines, width = (
        csv_enhanced._read_numeric_rows(
            path, cfg, channels=["ch7"], start=None, end=None
        )
    )

    assert metadata == {}
    assert matrix is None
    assert list(selected) == [7]
    assert selected[7].size == 100
    assert list(timestamps) == [0]
    assert len(timestamps[0]) == len(lines) == 100
    assert width == 13
    result = csv_enhanced.read_timeseriesdict_csv(path, channels=["ch7"])
    assert list(result) == ["ch7"]
    np.testing.assert_array_equal(result["ch7"].value, np.arange(100) + 6)


def test_large_selected_parser_materializes_one_payload_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large-selected.csv"
    _numeric_csv(path, rows=65536, channels=16)
    parser_code = csv_enhanced._read_numeric_rows.__code__
    observed: dict[str, int | bool] = {}

    def inspect_return(frame: object, event: str, _arg: object) -> None:
        if event != "return" or getattr(frame, "f_code", None) is not parser_code:
            return
        locals_ = frame.f_locals  # type: ignore[attr-defined]
        observed.update(
            selected_columns=len(locals_["selected_values"]),
            selected_values=sum(len(v) for v in locals_["selected_values"].values()),
            converted_values=sum(v.size for v in locals_["selected"].values()),
            buffered_full_rows=len(locals_["chunk_rows"]),
            buffered_matrices=len(locals_["matrix_chunks"]),
            full_matrix=locals_["matrix"] is not None,
        )

    sys.setprofile(inspect_return)
    try:
        result = csv_enhanced.read_timeseriesdict_csv(path, channels=["ch1"])
    finally:
        sys.setprofile(None)
    assert observed == {
        "selected_columns": 1,
        "selected_values": 65536,
        "converted_values": 65536,
        "buffered_full_rows": 0,
        "buffered_matrices": 0,
        "full_matrix": False,
    }
    assert list(result) == ["ch1"]


@pytest.mark.parametrize("selected", [None, ["ch1"]])
def test_later_unselected_numeric_fault_precedes_timestamp_fault(
    tmp_path: Path, selected: list[str] | None
) -> None:
    path = tmp_path / "fault.csv"
    path.write_text("0,1,2\n2,3,4\n3,5,bad\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        csv_enhanced.read_timeseriesdict_csv(path, channels=selected)
    assert str(caught.value) == "CSV line 3 contains non-numeric data"


def test_selected_parser_reports_unselected_width_fault(tmp_path: Path) -> None:
    path = tmp_path / "width.csv"
    path.write_text("0,1,2\n1,3,4\n2,5\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        csv_enhanced.read_timeseriesdict_csv(path, channels=["ch1"])
    assert str(caught.value) == "CSV line 3 has 2 columns; expected 3"


@pytest.mark.parametrize(
    "row",
    [
        '0,"1,2",bad',
        "0,1,2bad",
        "0,1,2e",
        "0,1,1.2.3",
        "0,1,1nan",
        "0,1#not-a-comment",
    ],
)
def test_bulk_parser_rejects_partial_numeric_fields(row: str) -> None:
    with pytest.raises(ValueError, match="CSV line 1 contains non-numeric data"):
        csv_enhanced.read_timeseriesdict_csv(io.StringIO(row + "\n"))


def test_wide_row_converts_in_bounded_temporary_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csv_enhanced, "_MAX_CSV_MATRIX_CHUNK_BYTES", 128)
    converted_chunks: list[int] = []
    original = csv_enhanced._convert_numeric_chunk

    def record_chunk(rows: list[list[str]], lines: list[int], width: int) -> np.ndarray:
        result = original(rows, lines, width)
        converted_chunks.append(result.nbytes)
        return result

    monkeypatch.setattr(csv_enhanced, "_convert_numeric_chunk", record_chunk)
    row = list(map(str, range(21)))
    _, matrix, _, _, _, width = csv_enhanced._read_numeric_rows(
        io.StringIO(",".join(row) + "\n"),
        CSVFormatConfig(),
        channels=None,
        start=None,
        end=None,
    )
    assert width == 21
    assert matrix is not None
    np.testing.assert_array_equal(matrix[0], np.arange(21))
    assert converted_chunks and max(converted_chunks) <= 128


def test_read_only_text_stream_preserves_old_reader_contract() -> None:
    class ReadOnlyStream(io.TextIOBase):
        def read(self, size: int = -1) -> str:
            return "0,1\n1,2\n"

    result = csv_enhanced.read_timeseriesdict_csv(ReadOnlyStream())
    np.testing.assert_array_equal(result["ch1"].value, [1, 2])


def test_unbuffered_registry_style_file_keeps_caller_ownership(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text("0,1\n1,2\n", encoding="utf-8")
    with path.open("rb", buffering=0) as raw:
        result = csv_enhanced.read_timeseriesdict_csv(raw)
        assert not raw.closed
        assert raw.tell() == path.stat().st_size
        np.testing.assert_array_equal(result["ch1"].value, [1, 2])


def test_public_csv_read_keeps_cr_only_newlines(tmp_path: Path) -> None:
    path = tmp_path / "cr-only.csv"
    path.write_bytes(b"0,1\r1,2\r2,3\r")
    result = TimeSeriesDict.read(path, format="csv")
    np.testing.assert_array_equal(result["ch1"].value, [1, 2, 3])
    np.testing.assert_array_equal(result["ch1"].times.value, [0, 1, 2])


def test_negative_configured_data_index_keeps_last_column() -> None:
    config = CSVFormatConfig(columns=[ColumnSpec("last", -1)])
    result = csv_enhanced.read_timeseriesdict_csv(
        io.StringIO("1,2,3\n4,5,6\n"), config=config
    )
    np.testing.assert_array_equal(result["last"].value, [3, 6])


@pytest.mark.parametrize("selected", [None, ["ch1"]])
def test_single_textual_nat_row_keeps_header_only_result(
    tmp_path: Path, selected: list[str] | None
) -> None:
    path = tmp_path / "header-only.csv"
    path.write_text("NaT,bad\n", encoding="utf-8")

    assert not csv_enhanced.read_timeseriesdict_csv(path, channels=selected)


@pytest.mark.parametrize("explicit", [False, True])
def test_frequency_csv_keeps_nonuniform_axis_and_native_metadata(
    tmp_path: Path, explicit: bool
) -> None:
    path = tmp_path / "frequency.csv"
    path.write_text("# unit=V\n1,10\n2,20\n4,30\n", encoding="utf-8")
    kwargs = {"format": "csv"} if explicit else {}

    spectrum = FrequencySeries.read(path, **kwargs)

    np.testing.assert_array_equal(spectrum.frequencies.value, [1, 2, 4])
    np.testing.assert_array_equal(spectrum.value, [10, 20, 30])
    assert spectrum.epoch is None
    assert str(spectrum.unit) == ""


@pytest.mark.parametrize("values", [[], [1.0, 2.0, 3.0]])
def test_direct_writer_streams_identical_bytes(
    values: list[float], tmp_path: Path
) -> None:
    series = TimeSeries(values, t0=12.5, dt=0.25, name="sensor", unit="m")

    class RecordingStream(io.StringIO):
        largest_write = 0

        def write(self, value: str) -> int:
            self.largest_write = max(self.largest_write, len(value))
            return super().write(value)

    stream = RecordingStream()
    returned = csv_enhanced.write_timeseries_csv(series, stream)
    header = (
        "# gwexpy.timeseries.csv v1\n"
        "# name=sensor\n"
        "# unit=m\n"
        "# t0=1.250000000000000000e+01\n"
        "# dt=2.500000000000000000e-01\n"
    )
    rows = [
        f"{float(timestamp):.18e},{float(value):.18e}"
        for timestamp, value in zip(series.times.value, series.value, strict=False)
    ]
    expected = header + "\n".join(rows) + "\n"

    assert returned is stream
    assert stream.getvalue().encode("utf-8") == expected.encode("utf-8")
    assert stream.largest_write < len(expected)

    path = tmp_path / "direct.csv"
    assert csv_enhanced.write_timeseries_csv(series, path) == path
    assert path.read_bytes() == expected.encode("utf-8")


def test_direct_writer_has_no_output_sized_list_or_join() -> None:
    source = inspect.getsource(csv_enhanced.write_timeseries_csv)
    body = ast.parse(source)

    assert not any(isinstance(node, ast.ListComp) for node in ast.walk(body))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        for node in ast.walk(body)
    )
