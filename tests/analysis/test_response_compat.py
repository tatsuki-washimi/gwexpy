from __future__ import annotations

import warnings
from typing import Any, cast

import gwexpy.analysis as analysis
from gwexpy.analysis import response


def test_estimate_response_function_wrapper_calls_compute(monkeypatch):
    witness = cast(Any, object())
    target = cast(Any, object())
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


def test_analysis_package_exports_response_symbols():
    """analysis パッケージが Response 系の公開シンボルを再エクスポートする。"""
    assert analysis.ResponseFunctionResult is response.ResponseFunctionResult
    assert analysis.ResponseFunctionAnalysis is response.ResponseFunctionAnalysis
    assert analysis.estimate_response_function is response.estimate_response_function
    assert analysis.detect_step_segments is response.detect_step_segments


def test_analysis_package_all_includes_response_symbols():
    """analysis.__all__ に Response 系の公開シンボルが含まれる。"""
    exported = set(analysis.__all__)
    assert "ResponseFunctionResult" in exported
    assert "ResponseFunctionAnalysis" in exported
    assert "estimate_response_function" in exported
    assert "detect_step_segments" in exported
