"""Public reader regressions for synthetic DiagGUI XML products (issue #726)."""

from __future__ import annotations

import base64
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesMatrix,
)
from gwexpy.io.dttxml_common import HAS_DTTXML, load_dttxml_products
from gwexpy.timeseries import TimeSeriesDict

_EPOCH = 1_234_567_890.25
_F0 = 17.5
_DF = 2.5
_ASD_CHANNEL = "K1:TEST-ASD"
_INPUT_CHANNEL = "K1:TEST-INPUT"
_OUTPUT_CHANNELS = ("K1:TEST-OUTPUT-B", "K1:TEST-OUTPUT-A")
_ASD_VALUES = np.array([1.25, 2.5, 5.0, 10.0], dtype=np.float32)
_TF_VALUES = np.array(
    [1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 4.0j, -2.0 - 1.5j], dtype=np.complex64
)
_TF_SECOND_VALUES = np.array(
    [0.5 - 1.0j, 2.0 + 3.0j, -1.5 + 0.5j, 4.0 - 2.0j], dtype=np.complex64
)
_TS_CHANNEL = "K1:TEST-TIME"
_TS_VALUES = np.array([1.5, -2.0, 0.25, 8.0], dtype=np.float32)
_TS_DT = 0.125
_EMBEDDED_FREQUENCIES = np.array([10.0, 11.0, 13.5, 17.0], dtype=np.float32)


def _add_product(
    root: ET.Element,
    *,
    result_index: int,
    product_type: str,
    params: dict[str, str],
    values: np.ndarray,
    dtype: str,
    dims: tuple[int, ...],
    time_params: dict[str, str] | None = None,
) -> None:
    result = ET.SubElement(
        root, "LIGO_LW", {"Name": f"Result[{result_index}]", "Type": product_type}
    )
    for name, value in params.items():
        ET.SubElement(result, "Param", {"Name": name}).text = value
    for name, value in (time_params or {}).items():
        ET.SubElement(result, "Time", {"Name": name}).text = value
    array = ET.SubElement(result, "Array", {"Type": dtype})
    for dim in dims:
        ET.SubElement(array, "Dim").text = str(dim)
    stream = ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"})
    storage_dtype = {
        "float": "<f4",
        "double": "<f8",
        "floatComplex": "<c8",
        "doubleComplex": "<c16",
    }[dtype]
    raw = np.asarray(values, dtype=storage_dtype).tobytes()
    stream.text = base64.b64encode(raw).decode("ascii")


@pytest.fixture
def synthetic_diaggui_xml(tmp_path):
    """Build a public-format XML sample without relying on private DTT data."""
    root = ET.Element("LIGO_LW")

    # Spectrum subtype 1 is PSD in the external parser and is normalized as ASD.
    # BUnit is consumed by gwexpy's native parser; the dttxml package ignores it.
    _add_product(
        root,
        result_index=0,
        product_type="Spectrum",
        params={
            "Subtype": "1",
            "M": "1",
            "N": str(len(_ASD_VALUES)),
            "f0": str(_F0),
            "df": str(_DF),
            "ChannelA": _ASD_CHANNEL,
            "BUnit": "m",
        },
        values=_ASD_VALUES,
        dtype="float",
        dims=(1, len(_ASD_VALUES)),
        time_params={"t0": str(_EPOCH)},
    )

    # TransferFunction subtype 0 is the external DiagGUI representation for
    # complex Y samples. Keep this as the sole TF source in the fixture.
    transfer_values = np.stack((_TF_VALUES, _TF_SECOND_VALUES))
    transfer_params = {
        "Subtype": "0",
        "M": "2",
        "N": str(len(_TF_VALUES)),
        "f0": str(_F0),
        "df": str(_DF),
        "ChannelA": _INPUT_CHANNEL,
        "ChannelB[0]": _OUTPUT_CHANNELS[0],
        "ChannelB[1]": _OUTPUT_CHANNELS[1],
        "BUnit": "m/count",
    }
    _add_product(
        root,
        result_index=1,
        product_type="TransferFunction",
        params=transfer_params,
        values=transfer_values,
        dtype="floatComplex",
        dims=(2, len(_TF_VALUES)),
        time_params={"t0": str(_EPOCH)},
    )
    _add_product(
        root,
        result_index=2,
        product_type="TimeSeries",
        params={
            "Subtype": "0",
            "N": str(len(_TS_VALUES)),
            "dt": str(_TS_DT),
            "Channel": _TS_CHANNEL,
        },
        values=_TS_VALUES,
        dtype="float",
        dims=(len(_TS_VALUES),),
        time_params={"t0": str(_EPOCH)},
    )

    path = tmp_path / "synthetic_diaggui.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.fixture
def embedded_frequency_diaggui_xml(tmp_path):
    """Model the one-dimensional (f, Y) TF/COH layout in a real DTT export."""
    root = ET.Element("LIGO_LW")
    tf_rows = np.stack((_TF_VALUES, _TF_SECOND_VALUES))
    coherence_rows = np.array(
        [[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]], dtype=np.float32
    )
    common = {
        "M": "2",
        "N": str(len(_EMBEDDED_FREQUENCIES)),
        "f0": "0",
        "df": "0",
        "ChannelA": _INPUT_CHANNEL,
        "ChannelB[0]": _OUTPUT_CHANNELS[0],
        "ChannelB[1]": _OUTPUT_CHANNELS[1],
    }
    _add_product(
        root,
        result_index=0,
        product_type="TransferFunction",
        params={"Subtype": "3", **common},
        values=np.concatenate(
            (_EMBEDDED_FREQUENCIES.astype(np.complex64), tf_rows.ravel())
        ),
        dtype="floatComplex",
        dims=(3 * len(_EMBEDDED_FREQUENCIES),),
        time_params={"t0": str(_EPOCH)},
    )
    _add_product(
        root,
        result_index=1,
        product_type="TransferFunction",
        params={"Subtype": "5", **common},
        values=np.concatenate((_EMBEDDED_FREQUENCIES, coherence_rows.ravel())),
        dtype="float",
        dims=(3 * len(_EMBEDDED_FREQUENCIES),),
        time_params={"t0": str(_EPOCH)},
    )
    path = tmp_path / "embedded_frequency_diaggui.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.fixture
def rounded_uniform_embedded_diaggui_xml(tmp_path):
    """Use float32 storage for a uniform axis with small rounding differences."""
    root = ET.Element("LIGO_LW")
    frequencies = np.arange(100, dtype=np.float32) / 10
    values = np.ones(frequencies.size, dtype=np.complex64)
    _add_product(
        root,
        result_index=0,
        product_type="TransferFunction",
        params={
            "Subtype": "3",
            "M": "1",
            "N": str(frequencies.size),
            "f0": "0",
            "df": "0",
            "ChannelA": _INPUT_CHANNEL,
            "ChannelB[0]": _OUTPUT_CHANNELS[0],
        },
        values=np.concatenate((frequencies.astype(np.complex64), values)),
        dtype="floatComplex",
        dims=(2 * frequencies.size,),
        time_params={"t0": str(_EPOCH)},
    )
    path = tmp_path / "rounded_uniform_embedded_diaggui.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path, frequencies


@pytest.mark.skipif(not HAS_DTTXML, reason="requires the optional dttxml parser")
def test_external_loader_retains_uniform_embedded_df_contract(
    rounded_uniform_embedded_diaggui_xml,
):
    path, raw_frequencies = rounded_uniform_embedded_diaggui_xml
    series = load_dttxml_products(path, native=False)["TF"][
        (_OUTPUT_CHANNELS[0], _INPUT_CHANNEL)
    ]
    assert isinstance(series, FrequencySeries)
    assert series.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(series.value, np.ones(100, dtype=np.complex64))
    assert float(series.df.value) == pytest.approx(float(raw_frequencies[1]))
    # The existing helper builds a linear axis for uniform (f, Y) layouts.
    # float32 rounding can make its bins differ slightly from the raw row.
    np.testing.assert_allclose(
        series.frequencies.value, raw_frequencies, rtol=0, atol=1e-6
    )


@pytest.mark.skipif(not HAS_DTTXML, reason="requires the optional dttxml parser")
def test_external_loader_preserves_small_cumulative_frequency_drift(
    rounded_uniform_embedded_diaggui_xml,
):
    path, _ = rounded_uniform_embedded_diaggui_xml
    increments = np.where(np.arange(99) % 2 == 0, 1.0, 1.0625).astype(np.float32)
    frequencies = np.concatenate(
        (
            np.array([1_000_000.0], dtype=np.float32),
            np.float32(1_000_000.0) + np.cumsum(increments, dtype=np.float32),
        )
    )
    tree = ET.parse(path)
    stream = tree.find(".//Stream")
    assert stream is not None and stream.text is not None
    words = np.frombuffer(base64.b64decode(stream.text), dtype="<c8").copy()
    words[: frequencies.size] = frequencies.astype(np.complex64)
    stream.text = base64.b64encode(words.tobytes()).decode("ascii")
    tree.write(path)

    series = load_dttxml_products(path, native=False)["TF"][
        (_OUTPUT_CHANNELS[0], _INPUT_CHANNEL)
    ]
    np.testing.assert_array_equal(series.frequencies.value, frequencies)
    assert getattr(series, "df", None) is None


@pytest.mark.skipif(not HAS_DTTXML, reason="requires the optional dttxml parser")
def test_external_loader_preserves_embedded_irregular_frequencies(
    embedded_frequency_diaggui_xml,
):
    products = load_dttxml_products(embedded_frequency_diaggui_xml, native=False)
    pair = (_OUTPUT_CHANNELS[0], _INPUT_CHANNEL)
    for product, expected in (
        ("TF", _TF_VALUES),
        ("COH", np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)),
    ):
        series = products[product][pair]
        assert isinstance(series, FrequencySeries)
        np.testing.assert_array_equal(series.value, expected)
        assert series.dtype == expected.dtype
        np.testing.assert_array_equal(series.frequencies.value, _EMBEDDED_FREQUENCIES)
        assert float(series.epoch.value) == pytest.approx(_EPOCH)
        assert str(series.unit) == ""


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_embedded_frequency_tf_matrix_preserves_axis_phase_and_pairs(
    embedded_frequency_diaggui_xml, native
):
    matrix = FrequencySeriesMatrix.read(
        embedded_frequency_diaggui_xml,
        format="xml.diaggui",
        products="TF",
        native=native,
    )
    assert matrix.shape == (2, 1, len(_EMBEDDED_FREQUENCIES))
    assert matrix.dtype == np.dtype(np.complex64)
    assert list(matrix.rows) == sorted(_OUTPUT_CHANNELS)
    assert list(matrix.cols) == [_INPUT_CHANNEL]
    np.testing.assert_array_equal(matrix.frequencies.value, _EMBEDDED_FREQUENCIES)
    assert matrix.epoch == pytest.approx(_EPOCH)
    for row, expected in zip(
        _OUTPUT_CHANNELS, (_TF_VALUES, _TF_SECOND_VALUES), strict=True
    ):
        series = matrix[row, _INPUT_CHANNEL]
        np.testing.assert_array_equal(series.value, expected)
        np.testing.assert_array_equal(series.value.imag, expected.imag)
        np.testing.assert_array_equal(np.angle(series.value), np.angle(expected))
        np.testing.assert_array_equal(series.frequencies.value, _EMBEDDED_FREQUENCIES)
        assert str(series.unit) == ""


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_embedded_frequency_coherence_matrix_preserves_axis_and_pairs(
    embedded_frequency_diaggui_xml, native
):
    matrix = FrequencySeriesMatrix.read(
        embedded_frequency_diaggui_xml,
        format="xml.diaggui",
        products="COH",
        native=native,
    )
    assert matrix.shape == (2, 1, len(_EMBEDDED_FREQUENCIES))
    assert matrix.dtype == np.dtype(np.float32)
    assert list(matrix.rows) == sorted(_OUTPUT_CHANNELS)
    assert list(matrix.cols) == [_INPUT_CHANNEL]
    np.testing.assert_array_equal(matrix.frequencies.value, _EMBEDDED_FREQUENCIES)
    assert matrix.epoch == pytest.approx(_EPOCH)
    np.testing.assert_array_equal(
        matrix[_OUTPUT_CHANNELS[0], _INPUT_CHANNEL].value,
        np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        matrix[_OUTPUT_CHANNELS[1], _INPUT_CHANNEL].value,
        np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32),
    )


@pytest.mark.skipif(not HAS_DTTXML, reason="requires the optional dttxml parser")
def test_external_loader_keeps_frequencyseries_values(synthetic_diaggui_xml):
    """The directly imported loader's native=False value type predates #726."""
    tree = ET.parse(synthetic_diaggui_xml)
    coherence = np.array([0.2, 0.5, 0.75, 0.9], dtype=np.float32)
    cross_spectrum = np.array(
        [1 + 0.5j, 2 - 0.25j, -3 + 4j, 0.125 - 0.5j], dtype=np.complex64
    )
    for result_index, subtype, values, dtype in (
        (3, "3", coherence, "float"),
        (4, "2", cross_spectrum, "floatComplex"),
    ):
        _add_product(
            tree.getroot(),
            result_index=result_index,
            product_type="Spectrum",
            params={
                "Subtype": subtype,
                "M": "1",
                "N": str(len(values)),
                "f0": str(_F0),
                "df": str(_DF),
                "ChannelA": _INPUT_CHANNEL,
                "ChannelB[0]": _OUTPUT_CHANNELS[0],
            },
            values=values,
            dtype=dtype,
            dims=(1, len(values)),
            time_params={"t0": str(_EPOCH)},
        )
    tree.write(synthetic_diaggui_xml)
    products = load_dttxml_products(synthetic_diaggui_xml, native=False)

    assert products["PSD"] is products["ASD"]
    for product in ("PSD", "ASD"):
        series = products[product][_ASD_CHANNEL]
        assert isinstance(series, FrequencySeries)
        assert series.name == _ASD_CHANNEL
        np.testing.assert_array_equal(series.value, _ASD_VALUES)
        np.testing.assert_allclose(
            series.frequencies.value, _F0 + _DF * np.arange(len(_ASD_VALUES))
        )

    for pair, expected in zip(
        ((_OUTPUT_CHANNELS[0], _INPUT_CHANNEL), (_OUTPUT_CHANNELS[1], _INPUT_CHANNEL)),
        (_TF_VALUES, _TF_SECOND_VALUES),
        strict=True,
    ):
        series = products["TF"][pair]
        assert isinstance(series, FrequencySeries)
        assert series.name == str(pair)
        np.testing.assert_array_equal(series.value, expected)
        np.testing.assert_allclose(
            series.frequencies.value, _F0 + _DF * np.arange(len(expected))
        )

    pair = (_OUTPUT_CHANNELS[0], _INPUT_CHANNEL)
    for product, expected in (("COH", coherence), ("CSD", cross_spectrum)):
        series = products[product][pair]
        assert isinstance(series, FrequencySeries)
        assert series.name == str(pair)
        np.testing.assert_array_equal(series.value, expected)

    assert isinstance(products["TS"][_TS_CHANNEL], dict)


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_asd_frequencyseriesdict_preserves_data_and_metadata(
    synthetic_diaggui_xml, native
):
    result = FrequencySeriesDict.read(
        synthetic_diaggui_xml, format="xml.diaggui", products="ASD", native=native
    )

    assert list(result) == [_ASD_CHANNEL]
    series = result[_ASD_CHANNEL]
    assert isinstance(series, FrequencySeries)
    np.testing.assert_array_equal(series.value, _ASD_VALUES)
    assert series.dtype == _ASD_VALUES.dtype
    assert len(series) == len(_ASD_VALUES)
    assert series.name == _ASD_CHANNEL
    assert str(series.channel) == _ASD_CHANNEL
    assert float(series.f0.value) == pytest.approx(_F0)
    assert float(series.df.value) == pytest.approx(_DF)
    np.testing.assert_allclose(
        series.frequencies.value,
        _F0 + _DF * np.arange(len(_ASD_VALUES)),
    )
    assert float(series.epoch.value) == pytest.approx(_EPOCH)
    if native:
        assert str(series.unit) == "m"


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_asd_frequencyseries_preserves_data_and_metadata(synthetic_diaggui_xml, native):
    result = FrequencySeries.read(
        synthetic_diaggui_xml, format="xml.diaggui", products="ASD", native=native
    )

    assert isinstance(result, FrequencySeries)
    np.testing.assert_array_equal(result.value, _ASD_VALUES)
    assert result.dtype == _ASD_VALUES.dtype
    assert len(result) == len(_ASD_VALUES)
    assert result.name == _ASD_CHANNEL
    assert str(result.channel) == _ASD_CHANNEL
    assert float(result.f0.value) == pytest.approx(_F0)
    assert float(result.df.value) == pytest.approx(_DF)
    np.testing.assert_allclose(
        result.frequencies.value,
        _F0 + _DF * np.arange(len(_ASD_VALUES)),
    )
    assert float(result.epoch.value) == pytest.approx(_EPOCH)
    if native:
        assert str(result.unit) == "m"


@pytest.mark.parametrize(
    ("product_type", "subtype", "array_type", "product", "values"),
    [
        (
            "Spectrum",
            "1",
            "double",
            "ASD",
            np.array([1.25, 2.5, 5.0], dtype=np.float64),
        ),
        (
            "Spectrum",
            "2",
            "doubleComplex",
            "CSD",
            np.array([1 + 2j, -0.5 + 0.25j, 3 - 4j], dtype=np.complex128),
        ),
        (
            "TransferFunction",
            "0",
            "doubleComplex",
            "TF",
            np.array([1 + 2j, -0.5 + 0.25j, 3 - 4j], dtype=np.complex128),
        ),
    ],
)
def test_native_reader_preserves_double_precision_storage(
    tmp_path, product_type, subtype, array_type, product, values
):
    """Supported products preserve their declared on-disk precision."""
    root = ET.Element("LIGO_LW")
    _add_product(
        root,
        result_index=0,
        product_type=product_type,
        params={
            "Subtype": subtype,
            "M": "1",
            "N": str(len(values)),
            "f0": str(_F0),
            "df": str(_DF),
            "ChannelA": _INPUT_CHANNEL,
            "ChannelB[0]": _OUTPUT_CHANNELS[0],
        },
        values=values,
        dtype=array_type,
        dims=(1, len(values)),
        time_params={"t0": str(_EPOCH)},
    )
    path = tmp_path / f"{product}_{array_type}.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    products = load_dttxml_products(path, native=True)
    key = _INPUT_CHANNEL if product == "ASD" else (_OUTPUT_CHANNELS[0], _INPUT_CHANNEL)
    payload = products[product][key]
    assert payload["data"].dtype == values.dtype
    np.testing.assert_array_equal(payload["data"], values)
    np.testing.assert_allclose(
        payload["frequencies"], _F0 + _DF * np.arange(len(values))
    )
    assert payload["epoch"] == pytest.approx(_EPOCH)

    if product == "ASD":
        series = FrequencySeries.read(
            path, format="xml.diaggui", products=product, native=True
        )
    else:
        matrix = FrequencySeriesMatrix.read(
            path, format="xml.diaggui", products=product, native=True
        )
        assert matrix.shape == (1, 1, len(values))
        assert list(matrix.rows) == [_OUTPUT_CHANNELS[0]]
        assert list(matrix.cols) == [_INPUT_CHANNEL]
        series = matrix[_OUTPUT_CHANNELS[0], _INPUT_CHANNEL]
    assert series.dtype == values.dtype
    np.testing.assert_array_equal(series.value, values)
    np.testing.assert_allclose(series.frequencies.value, payload["frequencies"])
    assert float(series.epoch.value) == pytest.approx(_EPOCH)


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_tf_frequencyseriesmatrix_preserves_complex_pairs_and_axes(
    synthetic_diaggui_xml, native
):
    result = FrequencySeriesMatrix.read(
        synthetic_diaggui_xml,
        format="xml.diaggui",
        products="TF",
        native=native,
    )

    assert result.dtype == np.dtype(np.complex64)
    assert list(result.rows.keys()) == sorted(_OUTPUT_CHANNELS)
    assert list(result.cols.keys()) == [_INPUT_CHANNEL]
    assert result.shape == (2, 1, len(_TF_VALUES))
    expected_by_row = {
        _OUTPUT_CHANNELS[0]: _TF_VALUES,
        _OUTPUT_CHANNELS[1]: _TF_SECOND_VALUES,
    }
    for row, expected in expected_by_row.items():
        actual = result[row, _INPUT_CHANNEL]
        np.testing.assert_allclose(actual.value, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            actual.value.real, expected.real, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            actual.value.imag, expected.imag, rtol=1e-6, atol=1e-7
        )
        assert actual.dtype == np.dtype(np.complex64)
        assert len(actual) == len(expected)
        assert float(actual.f0.value) == pytest.approx(_F0)
        assert float(actual.df.value) == pytest.approx(_DF)
        np.testing.assert_allclose(
            actual.frequencies.value,
            _F0 + _DF * np.arange(len(expected)),
        )
    assert float(result.f0.value) == pytest.approx(_F0)
    assert float(result.df.value) == pytest.approx(_DF)
    np.testing.assert_allclose(
        result.frequencies.value,
        _F0 + _DF * np.arange(len(_TF_VALUES)),
    )
    assert result.epoch == pytest.approx(_EPOCH)
    if native:
        assert str(result.units[0, 0]) == "m / ct"


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_asd_unit_override_is_applied_for_both_parsers(synthetic_diaggui_xml, native):
    result = FrequencySeriesDict.read(
        synthetic_diaggui_xml,
        format="xml.diaggui",
        products="ASD",
        native=native,
        unit="m",
    )

    series = result[_ASD_CHANNEL]
    assert str(series.unit) == "m"
    np.testing.assert_array_equal(series.value, _ASD_VALUES)


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_tf_unit_override_preserves_values_and_frequency_axis(
    synthetic_diaggui_xml, native
):
    result = FrequencySeriesMatrix.read(
        synthetic_diaggui_xml,
        format="xml.diaggui",
        products="TF",
        native=native,
        unit="m/count",
    )

    assert str(result.units[0, 0]) == "m / ct"
    assert result.dtype == np.dtype(np.complex64)
    assert list(result.rows.keys()) == sorted(_OUTPUT_CHANNELS)
    assert list(result.cols.keys()) == [_INPUT_CHANNEL]
    assert float(result.f0.value) == pytest.approx(_F0)
    assert float(result.df.value) == pytest.approx(_DF)
    np.testing.assert_allclose(
        result.frequencies.value,
        _F0 + _DF * np.arange(len(_TF_VALUES)),
    )
    assert result.epoch == pytest.approx(_EPOCH)
    expected_by_row = {
        _OUTPUT_CHANNELS[0]: _TF_VALUES,
        _OUTPUT_CHANNELS[1]: _TF_SECOND_VALUES,
    }
    for row, expected in expected_by_row.items():
        actual = result[row, _INPUT_CHANNEL]
        assert actual.dtype == np.dtype(np.complex64)
        np.testing.assert_array_equal(actual.value, expected)


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_tf_pairs_filter_selects_requested_label_mapping(synthetic_diaggui_xml, native):
    selected_pair = (_OUTPUT_CHANNELS[1], _INPUT_CHANNEL)
    result = FrequencySeriesMatrix.read(
        synthetic_diaggui_xml,
        format="xml.diaggui",
        products="TF",
        native=native,
        pairs=[selected_pair],
    )

    assert list(result.rows.keys()) == [selected_pair[0]]
    assert list(result.cols.keys()) == [selected_pair[1]]
    assert result.shape == (1, 1, len(_TF_SECOND_VALUES))
    assert result.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(
        result[selected_pair[0], selected_pair[1]].value, _TF_SECOND_VALUES
    )


@pytest.mark.skipif(
    not HAS_DTTXML, reason="TS parsing requires the optional dttxml parser"
)
def test_timeseriesdict_reads_synthetic_dttxml_product(synthetic_diaggui_xml):
    result = TimeSeriesDict.read(
        synthetic_diaggui_xml, format="xml.diaggui", products="TS"
    )

    assert list(result) == [_TS_CHANNEL]
    series = result[_TS_CHANNEL]
    np.testing.assert_array_equal(series.value, _TS_VALUES)
    assert series.dtype == _TS_VALUES.dtype
    assert len(series) == len(_TS_VALUES)
    assert series.name == _TS_CHANNEL
    assert str(series.channel) == _TS_CHANNEL
    assert float(series.t0.value) == pytest.approx(_EPOCH)
    assert float(series.dt.value) == pytest.approx(_TS_DT)


def test_tf_matrix_read_without_tf_product_raises_value_error(tmp_path):
    path = tmp_path / "no_tf.xml"
    path.write_text("<LIGO_LW />")

    with pytest.raises(ValueError):
        FrequencySeriesMatrix.read(
            path,
            format="xml.diaggui",
            products="TF",
            native=True,
        )
