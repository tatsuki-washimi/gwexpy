import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries._gwf_io import _resolve_gwf_format

CHANNEL = "X1:ISSUE594_MAIN"
AUX_CHANNEL = "X1:ISSUE594_AUX"
SAMPLE_RATE = 4.0
T0 = 1000.0
EXPECTED_VALUES = {
    CHANNEL: np.arange(8.0),
    AUX_CHANNEL: np.arange(8.0) + 10.0,
}


@pytest.fixture
def synthetic_ffl(tmp_path: Path) -> dict[str, object]:
    frames_dir = tmp_path / "frames"
    lists_dir = tmp_path / "lists"
    nested_dir = lists_dir / "nested"
    frames_dir.mkdir()
    nested_dir.mkdir(parents=True)

    frame_paths: list[Path] = []
    for index, start in enumerate((T0, T0 + 1.0)):
        values = np.arange(index * 4, (index + 1) * 4, dtype=np.float64)
        series = TimeSeries(
            values,
            sample_rate=SAMPLE_RATE,
            t0=start,
            unit="V",
            channel=CHANNEL,
            name=CHANNEL,
        )
        aux = TimeSeries(
            values + 10.0,
            sample_rate=SAMPLE_RATE,
            t0=start,
            unit="A",
            channel=AUX_CHANNEL,
            name=AUX_CHANNEL,
        )
        frame_path = frames_dir / f"synthetic-{index}.gwf"
        try:
            TimeSeriesDict({CHANNEL: series, AUX_CHANNEL: aux}).write(
                frame_path, format="gwf"
            )
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
            pytest.skip(
                "optional GWF backend unavailable for synthetic fixture: "
                f"{type(exc).__name__}: {exc}"
            )
        frame_paths.append(frame_path)

    direct_ffl = lists_dir / "direct.ffl"
    direct_ffl.write_text(
        "\n".join(
            (
                f"../frames/{frame_paths[1].name} 1001 1 0 0",
                f"../frames/{frame_paths[0].name} 1000 1 0 0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    inner_ffl = nested_dir / "inner.ffl"
    inner_ffl.write_text(
        "\n".join(
            (
                f"../../frames/{frame_paths[0].name} 1000 1 0 0",
                f"../../frames/{frame_paths[1].name} 1001 1 0 0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    outer_ffl = lists_dir / "outer.ffl"
    outer_ffl.write_text("nested/inner.ffl 1000 2\n", encoding="utf-8")

    return {
        "frames": frame_paths,
        "direct": direct_ffl,
        "nested": outer_ffl,
        "expected_paths": [path.resolve() for path in frame_paths],
    }


def _run_fresh_gwexpy_read(
    source: Path,
    tmp_path: Path,
    *,
    kind: str,
    explicit_format: bool,
) -> dict[str, object]:
    script = r"""
import json
import sys

from gwexpy.timeseries import TimeSeries, TimeSeriesDict

source = sys.argv[1]
channel = sys.argv[2]
aux_channel = sys.argv[3]
kind = sys.argv[4]
explicit_format = sys.argv[5] == "1"
kwargs = {"format": "gwf"} if explicit_format else {}

if kind == "timeseries":
    result = TimeSeries.read(source, channel, **kwargs)
    series_by_name = {channel: result}
else:
    result = TimeSeriesDict.read(source, [channel, aux_channel], **kwargs)
    series_by_name = dict(result)

def metadata(series):
    return {
        "values": series.value.tolist(),
        "t0": float(series.t0.value),
        "span": [float(series.span[0]), float(series.span[1])],
        "dt": float(series.dt.value),
        "sample_rate": float(series.sample_rate.value),
        "unit": str(series.unit),
        "channel": None if series.channel is None else str(series.channel),
        "name": None if series.name is None else str(series.name),
    }

print(json.dumps({name: metadata(series) for name, series in series_by_name.items()}))
"""
    env = dict(__import__("os").environ)
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    env["XDG_CACHE_HOME"] = str(tmp_path / "xdg-cache")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(source),
            CHANNEL,
            AUX_CHANNEL,
            kind,
            "1" if explicit_format else "0",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "fresh GWexpy subprocess failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "fresh GWexpy subprocess did not emit JSON\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
            f"error: {exc}"
        )


@pytest.mark.parametrize("kind", ["timeseries", "timeseriesdict"])
@pytest.mark.parametrize("explicit_format", [True, False])
def test_gwexpy_reads_ffl_in_fresh_process_with_both_format_modes(
    synthetic_ffl: dict[str, object],
    tmp_path: Path,
    kind: str,
    explicit_format: bool,
):
    result = _run_fresh_gwexpy_read(
        synthetic_ffl["direct"],
        tmp_path,
        kind=kind,
        explicit_format=explicit_format,
    )

    assert set(result) == (
        {CHANNEL} if kind == "timeseries" else {CHANNEL, AUX_CHANNEL}
    )
    for channel, series in result.items():
        expected = EXPECTED_VALUES[channel]
        assert series["values"] == expected.tolist()
        assert series["t0"] == pytest.approx(T0)
        assert series["span"] == pytest.approx([T0, T0 + 2.0])
        assert series["dt"] == pytest.approx(1.0 / SAMPLE_RATE)
        assert series["sample_rate"] == pytest.approx(SAMPLE_RATE)
        assert series["unit"] == ("V" if channel == CHANNEL else "A")
        assert series["channel"] == channel
        assert series["name"] == channel


def test_ffl_format_resolution_selects_gwf_for_explicit_and_implicit_paths(
    synthetic_ffl: dict[str, object],
):
    source = synthetic_ffl["direct"]
    assert _resolve_gwf_format(source, None) == "gwf"
    assert _resolve_gwf_format(source, "gwf") == "gwf"


def test_ffl_expansion_normalizes_relative_paths_without_reordering_entries(
    synthetic_ffl: dict[str, object],
):
    from gwexpy.timeseries._gwf_io import _expand_gwf_source

    expected_paths = synthetic_ffl["expected_paths"]
    assert _expand_gwf_source(synthetic_ffl["direct"]) == [
        expected_paths[1],
        expected_paths[0],
    ]


@pytest.mark.parametrize("kind", ["timeseries", "timeseriesdict"])
def test_nested_ffl_reads_relative_entries_in_a_fresh_process(
    synthetic_ffl: dict[str, object],
    tmp_path: Path,
    kind: str,
):
    result = _run_fresh_gwexpy_read(
        synthetic_ffl["nested"],
        tmp_path,
        kind=kind,
        explicit_format=True,
    )

    assert set(result) == (
        {CHANNEL} if kind == "timeseries" else {CHANNEL, AUX_CHANNEL}
    )
    for channel, series in result.items():
        assert series["values"] == EXPECTED_VALUES[channel].tolist()
        assert series["t0"] == pytest.approx(T0)
        assert series["span"] == pytest.approx([T0, T0 + 2.0])


def test_ffl_cycle_error_contains_deterministic_include_chain(tmp_path: Path):
    first = tmp_path / "first.ffl"
    second = tmp_path / "second.ffl"
    first.write_text("second.ffl 0 0\n", encoding="utf-8")
    second.write_text("first.ffl 0 0\n", encoding="utf-8")

    script = r"""
import json
import sys

from gwexpy.timeseries import TimeSeriesDict

try:
    TimeSeriesDict.read(sys.argv[1], ["X1:ISSUE594_MAIN"], format="gwf")
except ValueError as exc:
    print(json.dumps({"error": str(exc)}))
else:
    print(json.dumps({"error": None}))
"""
    env = dict(__import__("os").environ)
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    env["XDG_CACHE_HOME"] = str(tmp_path / "xdg-cache")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(first)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["error"] is not None

    message = payload["error"]
    assert "FFL include cycle" in message
    first_path = str(first.resolve())
    second_path = str(second.resolve())
    assert f"{first_path} -> {second_path} -> {first_path}" in message


def test_ffl_entry_shapes_match_the_documented_contract(tmp_path: Path):
    """Pin the accepted entry shapes, and keep the GWpy attribution honest.

    The expansion docstring claims the one-field and five-field path-first
    forms are also accepted by GWpy's cache reader, while the nested
    three-field form is a GWexpy extension that GWpy rejects. Assert both
    halves so the claim cannot rot, and pin the LAL-layout fail-closed case.
    """
    from gwpy.io.cache import read_cache_entry

    from gwexpy.timeseries._gwf_io import _expand_gwf_source

    frame = tmp_path / "X-Y-0-1.gwf"
    frame.touch()
    child = tmp_path / "child.ffl"
    child.write_text(f"{frame.name} 0 1 0 0\n", encoding="utf-8")

    def expand(line: str) -> list[Path]:
        parent = tmp_path / "parent.ffl"
        parent.write_text(line + "\n", encoding="utf-8")
        return _expand_gwf_source(parent)

    # Shapes GWpy's cache reader accepts too.
    assert expand(frame.name) == [frame.resolve()]
    assert expand(f"{frame.name} 0 1 0 0") == [frame.resolve()]
    assert read_cache_entry(str(frame)) == str(frame)
    assert read_cache_entry(f"{frame} 0 1 0 0") == str(frame)

    # Nested three-field entries are a GWexpy extension; GWpy rejects them.
    assert expand(f"{child.name} 0 1") == [frame.resolve()]
    with pytest.raises(ValueError):
        read_cache_entry(f"{child} 0 1")

    # A five-field LAL cache line puts the path last. Fail closed instead of
    # treating the observatory field as a frame path.
    with pytest.raises(ValueError, match=r"expected a \.gwf path"):
        expand(f"X Y 0 1 {frame}")


def test_existing_gwf_list_route_remains_available(synthetic_ffl: dict[str, object]):
    frames = synthetic_ffl["frames"]
    result = TimeSeries.read(frames, CHANNEL, format="gwf")

    np.testing.assert_allclose(result.value, EXPECTED_VALUES[CHANNEL])
    assert float(result.t0.value) == pytest.approx(T0)
    assert float(result.dt.value) == pytest.approx(1.0 / SAMPLE_RATE)
    assert result.unit == "V"
    assert str(result.channel) == CHANNEL
    assert result.name == CHANNEL
