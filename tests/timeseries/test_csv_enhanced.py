"""Tests for the enhanced CSV reader (csv_enhanced.py / csv_config.py)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries.io.csv_config import CSVFormatConfig, ColumnSpec
from gwexpy.timeseries.io.csv_enhanced import (
    _detect_delimiter,
    _detect_skip_rows,
    _resample_uniform,
    read_timeseriesdict_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_csv(tmp_path: Path, content: str, name: str = "data.csv") -> Path:
    """Write content to a temp CSV file and return the path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# CSVFormatConfig
# ---------------------------------------------------------------------------


class TestCSVFormatConfig:
    def test_from_dict_basic(self):
        d = {
            "format": {"delimiter": "\t", "encoding": "utf-8"},
            "columns": [
                {"name": "acc_x", "index": 0, "unit": "m/s^2", "role": "data"},
            ],
        }
        cfg = CSVFormatConfig.from_dict(d)
        assert cfg.delimiter == "\t"
        assert len(cfg.columns) == 1
        assert cfg.columns[0].name == "acc_x"
        assert cfg.columns[0].unit == "m/s^2"

    def test_from_dict_defaults(self):
        cfg = CSVFormatConfig.from_dict({})
        assert cfg.delimiter == ","
        assert cfg.columns == []
        assert cfg.encoding == "utf-8"

    def test_from_dict_time_component(self):
        d = {
            "columns": [
                {"name": "year", "index": 0, "role": "time_component",
                 "time_component": "year"},
                {"name": "val", "index": 1, "role": "data"},
            ]
        }
        cfg = CSVFormatConfig.from_dict(d)
        tc = cfg.columns[0]
        assert tc.role == "time_component"
        assert tc.time_component == "year"

    def test_from_json(self, tmp_path):
        import json
        d = {"format": {"delimiter": ";"}, "columns": []}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(d))
        cfg = CSVFormatConfig.from_json(p)
        assert cfg.delimiter == ";"

    def test_from_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            format:
              delimiter: ","
              timezone: "UTC"
            columns:
              - name: ch0
                index: 1
                unit: "V"
                role: data
        """)
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content)
        cfg = CSVFormatConfig.from_yaml(p)
        assert cfg.timezone == "UTC"
        assert cfg.columns[0].unit == "V"

    def test_scale_factor_default(self):
        d = {"columns": [{"name": "x", "index": 0}]}
        cfg = CSVFormatConfig.from_dict(d)
        assert cfg.columns[0].scale_factor == 1.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


class TestDetectDelimiter:
    def test_comma(self):
        assert _detect_delimiter("1.0,2.0,3.0\n4.0,5.0,6.0") == ","

    def test_tab(self):
        assert _detect_delimiter("1.0\t2.0\t3.0\n4.0\t5.0\t6.0") == "\t"


class TestDetectSkipRows:
    def test_no_header(self):
        lines = ["1.0,2.0", "3.0,4.0"]
        assert _detect_skip_rows(lines, ",", "#") == 0

    def test_with_comment_lines(self):
        lines = ["# comment", "# another", "1.0,2.0", "3.0,4.0"]
        assert _detect_skip_rows(lines, ",", "#") == 2

    def test_with_text_header(self):
        lines = ["time,value", "0.0,1.0", "1.0,2.0"]
        # First line is not numeric → skip 1
        assert _detect_skip_rows(lines, ",", "#") == 1


class TestResampleUniform:
    def test_interpolate(self):
        # Non-uniform times
        times = np.array([0.0, 0.1, 0.3, 0.4, 0.6])
        values = times * 2.0
        new_t, new_v = _resample_uniform(times, values, sample_rate=10.0)
        # uniform dt should be 0.1
        np.testing.assert_allclose(np.diff(new_t), 0.1, atol=1e-10)

    def test_asfreq(self):
        times = np.array([0.0, 0.1, 0.3])
        values = np.array([1.0, 2.0, 3.0])
        new_t, new_v = _resample_uniform(times, values, 10.0, method="asfreq")
        assert len(new_t) > 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown resample method"):
            _resample_uniform(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                              1.0, method="bad")


# ---------------------------------------------------------------------------
# read_timeseriesdict_csv
# ---------------------------------------------------------------------------


class TestReadCSVAutoDetect:
    """Tests with auto-detection (no config)."""

    def test_simple_two_column(self, tmp_path):
        p = write_csv(tmp_path, """\
            0.0,1.0
            1.0,2.0
            2.0,3.0
        """)
        tsd = read_timeseriesdict_csv(p)
        assert "ch1" in tsd
        np.testing.assert_allclose(tsd["ch1"].value, [1.0, 2.0, 3.0])

    def test_multiple_data_columns(self, tmp_path):
        p = write_csv(tmp_path, """\
            0.0,1.0,10.0
            1.0,2.0,20.0
            2.0,3.0,30.0
        """)
        tsd = read_timeseriesdict_csv(p)
        assert "ch1" in tsd
        assert "ch2" in tsd

    def test_empty_file_returns_empty_dict(self, tmp_path):
        p = write_csv(tmp_path, "")
        tsd = read_timeseriesdict_csv(p)
        assert len(tsd) == 0

    def test_comment_lines_skipped(self, tmp_path):
        p = write_csv(tmp_path, """\
            # header comment
            0.0,1.0
            1.0,2.0
        """)
        tsd = read_timeseriesdict_csv(p)
        assert "ch1" in tsd
        assert len(tsd["ch1"]) == 2


class TestReadCSVWithConfig:
    """Tests with explicit CSVFormatConfig."""

    def test_named_columns(self, tmp_path):
        p = write_csv(tmp_path, "0.0,1.5\n1.0,2.5\n2.0,3.5\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="acc_x", column_index=1, unit="m/s^2", role="data"),
            ]
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        assert "acc_x" in tsd
        np.testing.assert_allclose(tsd["acc_x"].value, [1.5, 2.5, 3.5])

    def test_unit_applied(self, tmp_path):
        p = write_csv(tmp_path, "0.0,5.0\n1.0,6.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="pressure", column_index=1, unit="Pa", role="data"),
            ]
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        assert tsd["pressure"].unit == u.Pa

    def test_scale_factor(self, tmp_path):
        p = write_csv(tmp_path, "0.0,1000.0\n1.0,2000.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="val", column_index=1, scale_factor=0.001, role="data"),
            ]
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        np.testing.assert_allclose(tsd["val"].value, [1.0, 2.0])

    def test_channel_filter(self, tmp_path):
        p = write_csv(tmp_path, "0.0,1.0,10.0\n1.0,2.0,20.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="ch_a", column_index=1, role="data"),
                ColumnSpec(name="ch_b", column_index=2, role="data"),
            ]
        )
        tsd = read_timeseriesdict_csv(p, config=cfg, channels=["ch_a"])
        assert "ch_a" in tsd
        assert "ch_b" not in tsd

    def test_skip_role_excluded(self, tmp_path):
        p = write_csv(tmp_path, "0.0,999.0,1.0\n1.0,888.0,2.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="ignored", column_index=1, role="skip"),
                ColumnSpec(name="val", column_index=2, role="data"),
            ]
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        assert "val" in tsd
        assert "ignored" not in tsd

    def test_config_from_dict(self, tmp_path):
        p = write_csv(tmp_path, "0.0,42.0\n1.0,43.0\n")
        d = {
            "columns": [
                {"name": "time", "index": 0, "role": "time"},
                {"name": "temp", "index": 1, "unit": "K", "role": "data"},
            ]
        }
        tsd = read_timeseriesdict_csv(p, config=d)
        assert "temp" in tsd

    def test_config_from_yaml(self, tmp_path):
        import textwrap
        yaml_content = textwrap.dedent("""\
            columns:
              - name: time
                index: 0
                role: time
              - name: voltage
                index: 1
                unit: V
                role: data
        """)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml_content)
        p = write_csv(tmp_path, "0.0,3.3\n1.0,3.4\n")
        tsd = read_timeseriesdict_csv(p, config=cfg_path)
        assert "voltage" in tsd

    def test_resample_uniform(self, tmp_path):
        # Non-uniform times: 0, 0.1, 0.3
        p = write_csv(tmp_path, "0.0,1.0\n0.1,2.0\n0.3,3.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="val", column_index=1, role="data"),
            ],
            sample_rate=10.0,
            resample_method="interpolate",
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        # dt should be ~0.1
        assert abs(tsd["val"].dt.value - 0.1) < 0.01


class TestReadCSVTimestampReconstruction:
    """Tests for multi-column timestamp reconstruction."""

    def test_year_month_day_columns(self, tmp_path):
        # Columns: year, month, day, hour, minute, second, value
        p = write_csv(tmp_path, """\
            2021,1,15,12,0,0.0,1.5
            2021,1,15,12,0,1.0,2.5
        """)
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="year", column_index=0, role="time_component",
                           time_component="year"),
                ColumnSpec(name="month", column_index=1, role="time_component",
                           time_component="month"),
                ColumnSpec(name="day", column_index=2, role="time_component",
                           time_component="day"),
                ColumnSpec(name="hour", column_index=3, role="time_component",
                           time_component="hour"),
                ColumnSpec(name="minute", column_index=4, role="time_component",
                           time_component="minute"),
                ColumnSpec(name="second", column_index=5, role="time_component",
                           time_component="second"),
                ColumnSpec(name="val", column_index=6, role="data"),
            ],
            timezone="UTC",
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        assert "val" in tsd
        assert len(tsd["val"]) == 2
        np.testing.assert_allclose(tsd["val"].value, [1.5, 2.5])

    def test_timezone_required_for_time_components(self, tmp_path):
        p = write_csv(tmp_path, "2021,1,15,1.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="year", column_index=0, role="time_component",
                           time_component="year"),
                ColumnSpec(name="month", column_index=1, role="time_component",
                           time_component="month"),
                ColumnSpec(name="day", column_index=2, role="time_component",
                           time_component="day"),
                ColumnSpec(name="val", column_index=3, role="data"),
            ],
            timezone=None,
        )
        with pytest.raises(ValueError, match="timezone is required"):
            read_timeseriesdict_csv(p, config=cfg)
