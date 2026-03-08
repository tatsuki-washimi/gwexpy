from __future__ import annotations

import warnings

from gwexpy.analysis import response


def test_estimate_response_function_wrapper_calls_compute(monkeypatch):
    witness = object()
    target = object()
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_compute(self, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(response.ResponseFunctionAnalysis, "compute", fake_compute)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = response.estimate_response_function(
            witness=witness,
            target=target,
            fftlength=8.0,
            overlap=2.0,
            custom_kw=123,
        )

    assert out is sentinel
    assert captured["witness"] is witness
    assert captured["target"] is target
    assert captured["fftlength"] == 8.0
    assert captured["overlap"] == 2.0
    assert captured["custom_kw"] == 123
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
