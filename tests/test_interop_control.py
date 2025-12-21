import numpy as np
import pytest


@pytest.mark.requires("control")
def test_control_frd_roundtrip():
    pytest.importorskip("control")

    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.interop.control_ import to_control_frd, from_control_frd

    fs = FrequencySeries([1 + 0j, 2 + 0j, 3 + 0j], f0=1, df=1)
    frd = to_control_frd(fs, frequency_unit="rad/s")

    restored = from_control_frd(FrequencySeries, frd, frequency_unit="Hz")
    assert isinstance(restored, FrequencySeries)
    assert np.allclose(restored.value, fs.value)
