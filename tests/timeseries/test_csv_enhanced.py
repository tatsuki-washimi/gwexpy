"""Tests for the enhanced CSV reader (csv_enhanced.py / csv_config.py)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries.io.csv_config import ColumnSpec, CSVFormatConfig
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
                {
                    "name": "year",
                    "index": 0,
                    "role": "time_component",
                    "time_component": "year",
                },
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
            _resample_uniform(
                np.array([0.0, 1.0]), np.array([0.0, 1.0]), 1.0, method="bad"
            )


# ---------------------------------------------------------------------------
# read_timeseriesdict_csv
# ---------------------------------------------------------------------------


class TestReadCSVAutoDetect:
    """Tests with auto-detection (no config)."""

    def test_simple_two_column(self, tmp_path):
        p = write_csv(
            tmp_path,
            """\
            0.0,1.0
            1.0,2.0
            2.0,3.0
        """,
        )
        tsd = read_timeseriesdict_csv(p)
        assert "ch1" in tsd
        np.testing.assert_allclose(tsd["ch1"].value, [1.0, 2.0, 3.0])

    def test_multiple_data_columns(self, tmp_path):
        p = write_csv(
            tmp_path,
            """\
            0.0,1.0,10.0
            1.0,2.0,20.0
            2.0,3.0,30.0
        """,
        )
        tsd = read_timeseriesdict_csv(p)
        assert "ch1" in tsd
        assert "ch2" in tsd

    def test_empty_file_returns_empty_dict(self, tmp_path):
        p = write_csv(tmp_path, "")
        tsd = read_timeseriesdict_csv(p)
        assert len(tsd) == 0

    def test_comment_lines_skipped(self, tmp_path):
        p = write_csv(
            tmp_path,
            """\
            # header comment
            0.0,1.0
            1.0,2.0
        """,
        )
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
            ],
            time_scale="gps",
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
            ],
            time_scale="gps",
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        assert tsd["pressure"].unit == u.Pa

    def test_scale_factor(self, tmp_path):
        p = write_csv(tmp_path, "0.0,1000.0\n1.0,2000.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="val", column_index=1, scale_factor=0.001, role="data"),
            ],
            time_scale="gps",
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
            ],
            time_scale="gps",
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
            ],
            time_scale="gps",
        )
        tsd = read_timeseriesdict_csv(p, config=cfg)
        assert "val" in tsd
        assert "ignored" not in tsd

    def test_config_from_dict(self, tmp_path):
        p = write_csv(tmp_path, "0.0,42.0\n1.0,43.0\n")
        d = {
            "format": {"time_scale": "gps"},
            "columns": [
                {"name": "time", "index": 0, "role": "time"},
                {"name": "temp", "index": 1, "unit": "K", "role": "data"},
            ],
        }
        tsd = read_timeseriesdict_csv(p, config=d)
        assert "temp" in tsd

    def test_config_from_yaml(self, tmp_path):
        import textwrap

        yaml_content = textwrap.dedent("""\
            format:
              time_scale: "gps"
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

    def test_reject_resample_irregular_source(self, tmp_path):
        # A missing source timestamp must not be repaired by interpolation.
        p = write_csv(tmp_path, "0.0,1.0\n0.1,2.0\n0.3,3.0\n")
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="val", column_index=1, role="data"),
            ],
            sample_rate=10.0,
            resample_method="interpolate",
            time_scale="gps",
        )
        with pytest.raises(ValueError, match="CSV timestamp"):
            read_timeseriesdict_csv(p, config=cfg)


class TestReadCSVTimestampReconstruction:
    """Tests for multi-column timestamp reconstruction."""

    def test_year_month_day_columns(self, tmp_path):
        # Columns: year, month, day, hour, minute, second, value
        p = write_csv(
            tmp_path,
            """\
            2021,1,15,12,0,0.0,1.5
            2021,1,15,12,0,1.0,2.5
        """,
        )
        cfg = CSVFormatConfig(
            columns=[
                ColumnSpec(
                    name="year",
                    column_index=0,
                    role="time_component",
                    time_component="year",
                ),
                ColumnSpec(
                    name="month",
                    column_index=1,
                    role="time_component",
                    time_component="month",
                ),
                ColumnSpec(
                    name="day",
                    column_index=2,
                    role="time_component",
                    time_component="day",
                ),
                ColumnSpec(
                    name="hour",
                    column_index=3,
                    role="time_component",
                    time_component="hour",
                ),
                ColumnSpec(
                    name="minute",
                    column_index=4,
                    role="time_component",
                    time_component="minute",
                ),
                ColumnSpec(
                    name="second",
                    column_index=5,
                    role="time_component",
                    time_component="second",
                ),
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
                ColumnSpec(
                    name="year",
                    column_index=0,
                    role="time_component",
                    time_component="year",
                ),
                ColumnSpec(
                    name="month",
                    column_index=1,
                    role="time_component",
                    time_component="month",
                ),
                ColumnSpec(
                    name="day",
                    column_index=2,
                    role="time_component",
                    time_component="day",
                ),
                ColumnSpec(name="val", column_index=3, role="data"),
            ],
            timezone=None,
        )
        with pytest.raises(ValueError, match="timezone is required"):
            read_timeseriesdict_csv(p, config=cfg)


class TestCSVNumericTimeScaleAndUnit:
    """Tests for #634: explicit time_scale= and time_unit= on numeric CSV."""

    def _numeric_cfg(self, **fmt_kwargs) -> CSVFormatConfig:
        return CSVFormatConfig(
            columns=[
                ColumnSpec(name="time", column_index=0, role="time"),
                ColumnSpec(name="val", column_index=1, role="data"),
            ],
            **fmt_kwargs,
        )

    def test_configured_numeric_time_missing_scale_fails_closed(self, tmp_path):
        p = write_csv(tmp_path, "0.0,1.0\n1.0,2.0\n")
        cfg = self._numeric_cfg()
        with pytest.raises(ValueError, match="time_scale is required"):
            read_timeseriesdict_csv(p, config=cfg)

    def test_autodetect_missing_time_scale_warns_and_assumes_gps(self, tmp_path):
        p = write_csv(tmp_path, "1400000000.0,1.0\n1400000001.0,2.0\n")
        with pytest.warns(DeprecationWarning, match="time_scale="):
            tsd = read_timeseriesdict_csv(p)
        assert float(tsd["ch1"].t0.value) == pytest.approx(1400000000.0)
        assert tsd._gwexpy_io["time_scale"] == "gps"
        assert tsd._gwexpy_io["epoch_source"] == "csv_gps_assumed"

    def test_gps_scale_with_millisecond_time_unit(self, tmp_path):
        p = write_csv(
            tmp_path,
            "1400000000000,1.0\n1400000000100,2.0\n1400000000200,3.0\n",
        )
        cfg = self._numeric_cfg(time_scale="gps", time_unit="ms")
        tsd = read_timeseriesdict_csv(p, config=cfg)
        ts = tsd["val"]
        assert float(ts.t0.value) == pytest.approx(1400000000.0)
        assert float(ts.dt.value) == pytest.approx(0.1)
        assert ts.time_reference == "absolute"
        assert tsd._gwexpy_io["time_scale"] == "gps"
        assert tsd._gwexpy_io["time_unit"] == "ms"
        assert tsd._gwexpy_io["epoch_source"] == "csv_gps"

    def test_unix_scale_converts_with_leap_seconds_and_unit(self, tmp_path):
        from astropy.time import Time

        # 2023-11-14T22:13:20 UTC = Unix 1700000000.0 -> GPS = 1384035218.0 (18 leap s)
        # 2016-12-30T00:00:00 UTC = Unix 1483056000.0 -> 17 leap s
        p_2023 = write_csv(
            tmp_path,
            "1700000000000,1.0\n1700000001000,2.0\n",
            name="u2023.csv",
        )
        tsd_2023 = read_timeseriesdict_csv(
            p_2023, config=self._numeric_cfg(time_scale="unix", time_unit="ms")
        )
        expected_2023 = float(Time(1700000000, format="unix").to_value("gps"))
        assert float(tsd_2023["val"].t0.value) == pytest.approx(expected_2023)
        assert float(tsd_2023["val"].dt.value) == pytest.approx(1.0)
        assert tsd_2023._gwexpy_io["time_scale"] == "unix"
        assert tsd_2023._gwexpy_io["time_unit"] == "ms"
        assert tsd_2023._gwexpy_io["epoch_source"] == "csv_unix"

        p_2016 = write_csv(
            tmp_path,
            "1483056000.0,1.0\n1483056001.0,2.0\n",
            name="u2016.csv",
        )
        tsd_2016 = read_timeseriesdict_csv(
            p_2016, config=self._numeric_cfg(time_scale="unix")
        )
        expected_2016 = float(Time(1483056000, format="unix").to_value("gps"))
        assert float(tsd_2016["val"].t0.value) == pytest.approx(expected_2016)
        assert (expected_2023 - 1700000000.0) - (
            expected_2016 - 1483056000.0
        ) == pytest.approx(1.0)

    def test_unix_fractional_seconds_use_astropy_conversion(self):
        from decimal import Decimal

        from astropy.time import Time

        from gwexpy.timeseries.io.csv_enhanced import _convert_unix_decimals_to_gps

        samples = [Decimal("1483228799.25"), Decimal("1700000000.123456789")]
        converted = _convert_unix_decimals_to_gps(samples)
        for sample, actual in zip(samples, converted, strict=True):
            whole = int(sample)
            expected = Time(whole, float(sample - whole), format="unix").to_value(
                "gps", "decimal"
            )
            assert abs(actual - expected) < Decimal("0.000000001")

    def test_unix_regular_fractional_grid_on_leap_day(self, tmp_path):
        from astropy.time import Time

        p = write_csv(
            tmp_path,
            "1483228798.25,1.0\n1483228798.50,2.0\n1483228798.75,3.0\n",
        )
        ts = read_timeseriesdict_csv(p, config=self._numeric_cfg(time_scale="unix"))[
            "val"
        ]
        expected = Time(
            [1483228798, 1483228798, 1483228798],
            [0.25, 0.50, 0.75],
            format="unix",
        ).to_value("gps", "decimal")
        assert float(ts.dt.value) == pytest.approx(
            float((expected[-1] - expected[0]) / 2), rel=0, abs=2e-9
        )
        assert float(ts.t0.value) == pytest.approx(float(expected[0]), rel=0, abs=3e-7)

    @pytest.mark.parametrize("resample", [None, 2.0])
    def test_unix_leap_boundary_rejects_irregular_gps_grid(self, tmp_path, resample):
        p = write_csv(
            tmp_path,
            "1483228799,1.0\n1483228800,2.0\n1483228801,3.0\n",
        )
        with pytest.raises(ValueError, match="regular|irregular|cadence"):
            read_timeseriesdict_csv(
                p, config=self._numeric_cfg(time_scale="unix"), resample=resample
            )

    def test_relative_scale_with_and_without_epoch(self, tmp_path):
        import datetime as dt

        p = write_csv(tmp_path, "0,1.0\n100,2.0\n200,3.0\n")

        # Without epoch -> Plan I relative TimeSeries
        tsd_rel = read_timeseriesdict_csv(
            p, config=self._numeric_cfg(time_scale="relative", time_unit="ms")
        )
        assert tsd_rel["val"].time_reference == "relative"
        assert float(tsd_rel["val"].t0.value) == pytest.approx(0.0)
        assert float(tsd_rel["val"].dt.value) == pytest.approx(0.1)
        assert tsd_rel._gwexpy_io["time_scale"] == "relative"
        assert tsd_rel._gwexpy_io["time_unit"] == "ms"
        assert tsd_rel._gwexpy_io["epoch_source"] == "relative"

        # With numeric epoch -> absolute GPS TimeSeries
        tsd_abs = read_timeseriesdict_csv(
            p,
            config=self._numeric_cfg(time_scale="relative", time_unit="ms"),
            epoch=1400000000.0,
        )
        assert tsd_abs["val"].time_reference == "absolute"
        assert float(tsd_abs["val"].t0.value) == pytest.approx(1400000000.0)
        assert float(tsd_abs["val"].dt.value) == pytest.approx(0.1)
        assert tsd_abs._gwexpy_io["epoch_source"] == "user"

        # With naive datetime epoch + timezone
        tsd_dt = read_timeseriesdict_csv(
            p,
            config=self._numeric_cfg(time_scale="relative", time_unit="ms"),
            epoch=dt.datetime(2024, 1, 1, 9, 0, 0),
            timezone="Asia/Tokyo",
        )
        assert tsd_dt["val"].time_reference == "absolute"
        assert float(tsd_dt["val"].t0.value) == pytest.approx(1388102418.0)

    def test_kwargs_override_config_and_public_dict_read(self, tmp_path):
        from gwexpy.timeseries import TimeSeriesDict

        p = write_csv(tmp_path, "1700000000,1.0\n1700000001,2.0\n")
        cfg = self._numeric_cfg(time_scale="gps", time_unit="ms")
        tsd = TimeSeriesDict.read(
            p,
            format="csv",
            config=cfg,
            time_scale="unix",
            time_unit="s",
        )
        assert tsd._gwexpy_io["time_scale"] == "unix"
        assert tsd._gwexpy_io["time_unit"] == "s"
        assert float(tsd["val"].dt.value) == pytest.approx(1.0)

    def test_invalid_scale_unit_and_epoch_combinations_fail_closed(self, tmp_path):
        p = write_csv(tmp_path, "0,1.0\n1,2.0\n")
        with pytest.raises(ValueError, match="Unsupported CSV time_scale"):
            read_timeseriesdict_csv(p, time_scale="tai")
        with pytest.raises(ValueError, match="CSV time_unit"):
            read_timeseriesdict_csv(p, time_scale="gps", time_unit="m")
        with pytest.raises(ValueError, match="epoch cannot be combined"):
            read_timeseriesdict_csv(p, time_scale="gps", epoch=100.0)

    def test_multi_source_forwards_time_scale_and_unit(self, tmp_path):
        p1 = write_csv(tmp_path, "0,1.0\n100,2.0\n", name="p1.csv")
        p2 = write_csv(tmp_path, "200,3.0\n300,4.0\n", name="p2.csv")
        cfg = self._numeric_cfg(time_scale="relative", time_unit="ms")
        tsd = read_timeseriesdict_csv([p1, p2], config=cfg, epoch=1000.0)
        ts = tsd["val"]
        assert len(ts) == 4
        assert float(ts.t0.value) == pytest.approx(1000.0)
        assert float(ts.dt.value) == pytest.approx(0.1)
        assert tsd._gwexpy_io["n_sources"] == 2
        assert tsd._gwexpy_io["time_scale"] == "relative"
        assert tsd._gwexpy_io["time_unit"] == "ms"
        assert tsd._gwexpy_io["epoch_source"] == "user"
