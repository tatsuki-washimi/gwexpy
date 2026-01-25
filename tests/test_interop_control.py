import numpy as np
import pytest


@pytest.mark.requires("control")
def test_control_frd_roundtrip():
    pytest.importorskip("control")

    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.interop.control_ import from_control_frd, to_control_frd

    fs = FrequencySeries([1 + 0j, 2 + 0j, 3 + 0j], f0=1, df=1)
    frd = to_control_frd(fs, frequency_unit="rad/s")

    restored = from_control_frd(FrequencySeries, frd, frequency_unit="Hz")
    assert isinstance(restored, FrequencySeries)
    assert np.allclose(restored.value, fs.value)


@pytest.mark.requires("control")
def test_control_response_interop():
    control = pytest.importorskip("control")
    from gwexpy.timeseries import TimeSeries, TimeSeriesDict

    # 1. SISO Case
    # Simple first-order system: x' = -x + u, y = x
    # This should return a single TimeSeries
    t = np.linspace(0, 1, 101)  # dt = 0.01
    u = np.ones_like(t)
    sys_siso = control.TransferFunction([1], [1, 1])
    res_siso = control.forced_response(sys_siso, T=t, U=u)

    # Convert via TimeSeries.from_control
    ts = TimeSeries.from_control(res_siso)
    assert isinstance(ts, TimeSeries)
    assert ts.dt.value == 0.01
    assert len(ts) == 101
    assert np.allclose(ts.times.value, t)
    assert ts.name in ["output", "y[0]"]  # Default name for SISO

    # 2. MIMO Case
    # Two independent systems: y1 = x1, y2 = x2
    sys_mimo = control.ss(
        [[-1, 0], [0, -2]], [[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0, 0], [0, 0]]
    )
    u_mimo = np.vstack([u, u])
    res_mimo = control.forced_response(sys_mimo, T=t, U=u_mimo)

    # Convert via TimeSeries.from_control (should return TimeSeriesDict because noutputs > 1)
    tdict = TimeSeries.from_control(res_mimo)
    assert isinstance(tdict, TimeSeriesDict)
    assert len(tdict) == 2
    # Names might be output_0/1 or y[0]/y[1] depending on control version
    names = list(tdict.keys())
    assert "output_0" in names or "y[0]" in names
    assert "output_1" in names or "y[1]" in names

    first_name = names[0]
    assert isinstance(tdict[first_name], TimeSeries)
    assert np.allclose(tdict[first_name].value, res_mimo.outputs[0])

    # Convert via TimeSeriesDict.from_control
    tdict2 = TimeSeriesDict.from_control(res_mimo)
    assert isinstance(tdict2, TimeSeriesDict)
    assert len(tdict2) == 2

    # 3. MIMO with labels
    sys_labeled = control.ss(
        [[-1, 0], [0, -2]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[0, 0], [0, 0]],
        outputs=["chanA", "chanB"],
    )
    res_labeled = control.forced_response(sys_labeled, T=t, U=u_mimo)
    tdict_labeled = TimeSeriesDict.from_control(res_labeled)
    assert "chanA" in tdict_labeled
    assert "chanB" in tdict_labeled
    assert tdict_labeled["chanA"].name == "chanA"

    # 4. Test **kwargs (e.g., unit)
    ts_unit = TimeSeries.from_control(res_siso, unit="m")
    assert ts_unit.unit == "m"

    tdict_unit = TimeSeriesDict.from_control(res_labeled, unit="m")
    assert tdict_unit["chanA"].unit == "m"
    assert tdict_unit["chanB"].unit == "m"
