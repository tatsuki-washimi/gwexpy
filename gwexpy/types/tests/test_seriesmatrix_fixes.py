#!/usr/bin/env python
"""
Unit tests for SeriesMatrix key fixes:
1. _value internal storage unification
2. xarray/duration double-unit prevention
3. Quantity input unit defaults
4. Safe channel generation
"""

import numpy as np
from astropy import units as u
import pytest
from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.metadata import MetaDataMatrix, MetaData
from gwpy.detector import Channel

def test_value_internal_storage():
    """Test that _value is internal storage for shape references."""
    print("=" * 60)
    print("Test 1: _value internal storage")
    print("=" * 60)
    
    # Create a simple SeriesMatrix
    data = np.random.randn(2, 3, 10)
    xindex = np.linspace(0, 10, 10)
    
    sm = SeriesMatrix(data, xindex=xindex)
    
    # Check _value exists and has correct shape
    assert hasattr(sm, '_value'), "SeriesMatrix should have _value attribute"
    assert sm._value.shape == (2, 3, 10), f"_value shape incorrect: {sm._value.shape}"
    
    # Check that shape3D uses N_samples correctly
    assert sm.shape3D == (2, 3, 10), f"shape3D incorrect: {sm.shape3D}"
    assert sm.N_samples == 10, f"N_samples should be 10, got {sm.N_samples}"
    
    print(f"✓ _value shape: {sm._value.shape}")
    print(f"✓ shape3D: {sm.shape3D}")
    print(f"✓ N_samples: {sm.N_samples}")
    print("PASS\n")


def test_xarray_duration_no_double_units():
    """Test that xarray and duration don't double-apply units."""
    print("=" * 60)
    print("Test 2: xarray/duration no double-unit multiplication")
    print("=" * 60)
    
    # Create SeriesMatrix with frequency xindex (already as Quantity)
    from gwpy.types.index import Index
    
    # Create frequency index [0, 1, 2, ..., 9] Hz
    xindex = u.Quantity(np.linspace(0, 9, 10), 'Hz')
    data = np.random.randn(2, 2, 10)
    
    sm = SeriesMatrix(data, xindex=xindex)
    
    # xarray should return xindex directly (not multiply by xunit again)
    xarray = sm.xarray
    print(f"xindex: {sm.xindex}")
    print(f"xarray: {xarray}")
    print(f"xarray unit: {xarray.unit if hasattr(xarray, 'unit') else 'N/A'}")
    
    # Check units
    assert hasattr(xarray, 'unit'), "xarray should be a Quantity"
    assert xarray.unit == u.Hz, f"xarray unit should be Hz, got {xarray.unit}"
    
    # Duration should be last - first, not last*xunit - first*xunit
    duration = sm.duration
    print(f"duration: {duration}")
    print(f"duration value: {duration.value}")
    print(f"duration unit: {duration.unit if hasattr(duration, 'unit') else 'N/A'}")
    
    assert hasattr(duration, 'unit'), "duration should be a Quantity"
    assert duration.unit == u.Hz, f"duration unit should be Hz, got {duration.unit}"
    # duration should be 9 Hz (last - first in frequency space)
    assert np.isclose(duration.value, 9.0), f"duration value should be ~9, got {duration.value}"
    
    print("✓ xarray has correct unit (Hz, not Hz²)")
    print("✓ duration has correct unit (Hz, not Hz²)")
    print("✓ duration value is correct (9 Hz)")
    print("PASS\n")


def test_quantity_input_unit_defaults():
    """Test that Quantity input unit becomes default per-element unit."""
    print("=" * 60)
    print("Test 3: Quantity input unit defaults per-element")
    print("=" * 60)
    
    # Create data as Quantity with power units
    data_values = np.random.randn(2, 3, 10)
    data_quantity = u.Quantity(data_values, 'W')  # Watts
    
    # Create SeriesMatrix without explicit units argument
    # Should use data.unit as default for all cells
    xindex = np.linspace(0, 10, 10)
    sm = SeriesMatrix(data_quantity, xindex=xindex)
    
    # Check that all element units are set to Watts
    for i in range(2):
        for j in range(3):
            unit = sm.meta[i, j].unit
            print(f"Cell ({i},{j}) unit: {unit}")
            assert unit == u.W, f"Cell ({i},{j}) should have unit W, got {unit}"
    
    print("✓ All cells have Watts unit from input Quantity")
    print("PASS\n")


def test_xindex_quantity_preserves_unit():
    """Test that passing xindex as Quantity preserves it."""
    print("=" * 60)
    print("Test 4: xindex Quantity unit preservation")
    print("=" * 60)
    
    # Create xindex as Quantity (e.g., time or frequency)
    xindex_quantity = u.Quantity([0, 1, 2, 3, 4], 's')  # seconds
    data = np.random.randn(2, 2, 5)
    
    sm = SeriesMatrix(data, xindex=xindex_quantity)
    
    # xindex should preserve the Quantity
    print(f"Input xindex: {xindex_quantity}")
    print(f"SeriesMatrix.xindex: {sm.xindex}")
    
    assert hasattr(sm.xindex, 'unit'), "xindex should be a Quantity"
    assert sm.xindex.unit == u.s, f"xindex unit should be s, got {sm.xindex.unit}"
    
    # xarray should also have seconds unit (not seconds²)
    xarray = sm.xarray
    assert xarray.unit == u.s, f"xarray unit should be s, got {xarray.unit}"
    
    print("✓ xindex preserves Quantity unit (seconds)")
    print("✓ xarray has correct unit (seconds, not seconds²)")
    print("PASS\n")


def test_basic_import_and_instantiation():
    """Basic smoke test: ensure SeriesMatrix still instantiates."""
    print("=" * 60)
    print("Test 0: Basic instantiation")
    print("=" * 60)
    
    data = np.random.randn(3, 4, 20)
    xindex = np.arange(20, dtype=float)
    
    sm = SeriesMatrix(data, xindex=xindex, name="test_matrix")
    
    assert sm.shape == (3, 4, 20), f"Shape incorrect: {sm.shape}"
    assert sm.N_samples == 20, f"N_samples incorrect: {sm.N_samples}"
    assert sm.name == "test_matrix", f"Name incorrect: {sm.name}"
    
    print(f"✓ Created SeriesMatrix with shape {sm.shape}")
    print(f"✓ N_samples = {sm.N_samples}")
    print("PASS\n")


def test_array_ufunc_comparison_returns_bool():
    """Ensure comparison ufunc results use boolean dtype."""
    data = np.arange(6, dtype=float).reshape(1, 2, 3)
    xindex = np.array([0.0, 1.0, 2.0])
    sm = SeriesMatrix(data, xindex=xindex)

    result = sm < sm

    assert result.value.dtype == np.bool_, f"Comparison dtype should be bool, got {result.value.dtype}"
    assert not result.value.any(), "sm < sm should yield all False"


def test_array_ufunc_quantity_broadcast_shapes():
    """Quantity は (0D|1D|2D|3D) を明示ルールでブロードキャストして演算できる"""
    data = np.ones((2, 3, 4), dtype=float)
    xindex = np.arange(4, dtype=float)
    sm = SeriesMatrix(data, xindex=xindex, units=np.full((2, 3), u.m))

    # 0D: 全要素に一様適用
    r0 = sm * (2 * u.s)
    assert np.allclose(r0.value, 2.0)
    assert r0.meta[0, 0].unit == (u.m * u.s)

    # 1D: サンプル軸 (N_samples,) のみ許可
    q1 = u.Quantity(np.arange(4, dtype=float), u.s)
    r1 = sm * q1
    assert np.allclose(r1.value[0, 0], q1.value)
    assert np.allclose(r1.value[1, 2], q1.value)
    assert r1.meta[1, 2].unit == (u.m * u.s)

    # 2D: (Nrow,Ncol) のみ許可（サンプル軸は一様）
    q2 = u.Quantity(np.arange(6, dtype=float).reshape(2, 3), u.s)
    r2 = sm * q2
    assert np.allclose(r2.value[:, :, 0], q2.value)
    assert np.allclose(r2.value[:, :, -1], q2.value)
    assert r2.meta[0, 2].unit == (u.m * u.s)

    # 3D: (Nrow,Ncol,Nsample) のみ許可（完全に要素対応）
    q3 = u.Quantity(np.arange(24, dtype=float).reshape(2, 3, 4), u.s)
    r3 = sm * q3
    assert np.allclose(r3.value, q3.value)
    assert r3.meta[0, 0].unit == (u.m * u.s)


def test_array_ufunc_quantity_broadcast_invalid_shapes_raise():
    """Quantity の形状が規約に合わない場合は例外"""
    data = np.ones((2, 3, 4), dtype=float)
    xindex = np.arange(4, dtype=float)
    sm = SeriesMatrix(data, xindex=xindex)

    with pytest.raises(ValueError):
        _ = sm * u.Quantity(np.arange(5, dtype=float), u.s)  # 1D but length != N_samples

    with pytest.raises(ValueError):
        _ = sm * u.Quantity(np.ones((2, 2), dtype=float), u.s)  # 2D but shape != (Nrow,Ncol)

    with pytest.raises(ValueError):
        _ = sm * u.Quantity(np.ones((2, 3, 4, 1), dtype=float), u.s)  # 4D is unsupported


def test_is_contiguous_without_unit_xindex():
    """is_contiguous should work even when xindex has no unit."""
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([3.0, 4.0, 5.0])
    sm1 = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x1)
    sm2 = SeriesMatrix(np.ones((1, 1, 3)), xindex=x2)

    assert sm1.is_contiguous(sm2) == 1
    assert sm2.is_contiguous(sm1) == -1


def test_append_unitless_xindex():
    """append should support xindex without unit attribute."""
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([3.0, 4.0, 5.0])
    sm1 = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x1)
    sm2 = SeriesMatrix(np.ones((1, 1, 3)), xindex=x2)

    combined = sm1.append_exact(sm2)

    assert combined.shape == (1, 1, 6)
    assert np.array_equal(combined.xindex, np.concatenate([x1, x2]))


def test_logical_ufunc_skips_meta():
    """logical ufunc should not try to combine MetaData (avoid TypeError)."""
    sm = SeriesMatrix(np.arange(6).reshape(2, 1, 3), xindex=np.arange(3))
    out = np.logical_and(sm, sm)
    assert out.value.dtype == np.bool_
    assert out.shape == sm.shape


def test_logical_not_bool_output():
    """logical_notもメタに触らずbool出力を返す"""
    sm = SeriesMatrix(np.arange(6).reshape(2, 1, 3), xindex=np.arange(3))
    out = np.logical_not(sm)
    assert out.value.dtype == np.bool_
    assert out.shape == sm.shape


def test_update_resize_false_behaviour():
    """update は append(resize=False) 相当（末尾に新データを付けて古い分を左に押し出す）"""
    base = SeriesMatrix(np.arange(4).reshape(1, 1, 4), xindex=np.arange(4))
    new = SeriesMatrix(np.array([[[10, 11]]]), xindex=np.array([4, 5]))

    out = base.update(new, inplace=False)
    assert out.shape == (1, 1, 4)
    assert np.array_equal(out.value.flatten(), np.array([2, 3, 10, 11]))
    assert np.array_equal(np.asarray(out.xindex), np.array([2, 3, 4, 5]))


def test_diff_on_sample_axis():
    """diff はサンプル軸で shape が1サンプル短くなる"""
    data = np.arange(12).reshape(1, 1, 12)
    sm = SeriesMatrix(data, xindex=np.arange(12))
    out = sm.diff(n=2)
    assert out.shape == (1, 1, 10)
    assert np.allclose(out.value[0, 0, :3], np.array([0, 0, 0]))
    # xindex は2ステップ進む（長さが2減る）
    assert np.array_equal(np.asarray(out.xindex), np.arange(2, 12))


def test_value_at_returns_matrix_slice():
    """value_at は (Nrow,Ncol) 行列を返す"""
    data = np.array([[[1, 2, 3], [4, 5, 6]]])
    xindex = np.array([0, 1, 2])
    sm = SeriesMatrix(data, xindex=xindex)
    val = sm.value_at(1)
    assert val.shape == (1, 2)
    assert np.array_equal(val, np.array([[2, 5]]))


def test_pad_regular_xindex():
    """pad は正則 xindex を前提にサンプル軸方向に拡張する"""
    data = np.ones((1, 1, 3))
    sm = SeriesMatrix(data, xindex=np.arange(3))
    out = sm.pad(2, constant_values=0)
    assert out.shape == (1, 1, 7)
    # 追加された両端は0、元データは中央に残る
    assert np.array_equal(out.value[0, 0, :2], np.array([0, 0]))
    assert np.array_equal(out.value[0, 0, 2:5], np.ones(3))
    assert np.array_equal(out.value[0, 0, 5:], np.array([0, 0]))
    # xindex が dx=1 で前後に拡張される
    assert np.array_equal(np.asarray(out.xindex), np.arange(-2, 5))


def test_shift_moves_xindex():
    """shift は xindex を平行移動させる"""
    data = np.zeros((1, 1, 3))
    sm = SeriesMatrix(data, xindex=np.array([0.0, 1.0, 2.0]))
    sm.shift(5.0)
    assert np.array_equal(np.asarray(sm.xindex), np.array([5.0, 6.0, 7.0]))
    # _x0 も更新されているはず
    assert sm.x0 == 5.0


def test_copy_is_independent():
    """copy したオブジェクトを変更しても元に影響しない"""
    data = np.array([[[1.0, 2.0]]])
    sm = SeriesMatrix(data, xindex=np.array([0.0, 1.0]), rows={"r0": {"name": "r0"}}, cols={"c0": {"name": "c0"}})
    cp = sm.copy()

    cp.value[0, 0, 0] = 99.0
    cp.meta[0, 0]["name"] = "changed"
    cp.rows["r0"]["name"] = "row_changed"

    assert sm.value[0, 0, 0] == 1.0
    assert sm.meta[0, 0].name != "changed"
    assert sm.rows["r0"].name == "r0"


def test_step_returns_plot():
    """step は例外なく Plot を返す（描画自体はスモーク）"""
    data = np.arange(6).reshape(1, 2, 3)
    sm = SeriesMatrix(data, xindex=np.array([0.0, 1.0, 2.0]))
    plot_obj = sm.step(where="post")
    assert plot_obj is not None


def test_xindex_setter_length_check_and_cache_reset():
    """xindex setter が長さチェックし、キャッシュをクリアする"""
    data = np.arange(6).reshape(1, 1, 6)
    sm = SeriesMatrix(data, xindex=np.arange(6))
    # キャッシュ生成
    dx_before = sm.dx
    assert dx_before == 1
    # 長さ不一致はエラー
    with pytest.raises(ValueError):
        sm.xindex = np.arange(5)
    # 長さ一致なら更新され、_dx がリセットされる
    sm.xindex = np.arange(10, 16)
    assert sm.x0 == 10
    assert sm.dx == 1


def test_hdf5_io_roundtrip(tmp_path):
    """HDF5 read/write の往復でメタと値が保持される"""
    # データとメタを用意
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    xindex = u.Quantity([0, 1], "s")
    units = np.array([[u.V, u.V]], dtype=object)
    rows = {"r0": {"name": "r0", "unit": "m"}}
    cols = {"c0": {"name": "c0", "unit": "kg"}, "c1": {"name": "c1", "unit": "kg"}}

    sm = SeriesMatrix(
        data,
        xindex=xindex,
        units=units,
        rows=rows,
        cols=cols,
        name="test_sm",
        epoch=123.0,
        attrs={"note": "io_roundtrip"},
    )

    path = tmp_path / "sm.h5"
    sm.write(path)
    sm2 = SeriesMatrix.read(path)

    assert np.allclose(sm2.value, data)
    assert np.array_equal(np.asarray(sm2.xindex), np.asarray(xindex.value))
    assert getattr(sm2.xindex, "unit", None) == xindex.unit
    assert sm2.meta[0, 0].unit == u.V
    assert sm2.rows["r0"].unit == u.m
    assert sm2.cols["c1"].name == "c1"
    assert sm2.name == "test_sm"
    assert sm2.epoch == 123.0
    assert sm2.attrs.get("note") == "io_roundtrip"


def test_channel_mismatch_raises():
    """channels引数とmeta.channelが不一致ならエラーを投げる"""
    data = np.zeros((1, 1, 3))
    xindex = np.arange(3)
    meta_arr = np.empty((1, 1), dtype=object)
    meta_arr[0, 0] = MetaData(channel=Channel("A"), unit=u.dimensionless_unscaled, name="m")
    meta = MetaDataMatrix(meta_arr)

    with pytest.raises(ValueError):
        SeriesMatrix(data, xindex=xindex, meta=meta, channels=[["B"]])


def test_channel_string_matches_channel_object():
    """channelsに文字列、metaにChannelオブジェクトで同名なら許容される"""
    data = np.zeros((1, 1, 3))
    xindex = np.arange(3)
    meta_arr = np.empty((1, 1), dtype=object)
    meta_arr[0, 0] = MetaData(channel=Channel("A"), unit=u.dimensionless_unscaled, name="m")
    meta = MetaDataMatrix(meta_arr)

    sm = SeriesMatrix(data, xindex=xindex, meta=meta, channels=[["A"]])
    assert str(sm.meta[0, 0].channel.name) == "A"


def test_append_gap_and_pad_unitless():
    """単位なしxindexでgap/pad付きappendが正しくパディングとインデックスを生成する"""
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([5.0, 6.0, 7.0])
    sm1 = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x1)
    sm2 = SeriesMatrix(np.ones((1, 1, 3)), xindex=x2)

    combined = sm1.append_exact(sm2, pad=0.0, gap=3.0)

    assert combined.shape == (1, 1, 8)
    assert np.array_equal(combined.xindex, np.array([0., 1., 2., 3., 4., 5., 6., 7.]))
    # padding部分は0、それ以外は元の値
    expected = np.concatenate([np.zeros(3), np.zeros(2), np.ones(3)])
    assert np.array_equal(combined.value.flatten(), expected)


def test_getitem_preserves_3d_on_scalar_sample():
    """サンプル軸を整数指定しても3次元を維持する"""
    data = np.arange(24).reshape(2, 3, 4)
    xindex = np.arange(4)
    sm = SeriesMatrix(data, xindex=xindex)

    sub = sm[:, :, 1]
    assert sub.shape == (2, 3, 1)
    assert len(sub.xindex) == 1


def test_is_compatible_converts_units():
    """xindexが同値でも単位が異なる場合に互換とみなす"""
    x_s = np.array([0.0, 1.0, 2.0]) * u.s
    x_ms = np.array([0.0, 1000.0, 2000.0]) * u.ms
    sm_s = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x_s)
    sm_ms = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x_ms)
    assert sm_s.is_compatible(sm_ms)


def test_is_compatible_raises_on_incompatible_units():
    """変換不能な単位なら互換エラー"""
    x_s = np.array([0.0, 1.0, 2.0]) * u.s
    x_hz = np.array([0.0, 1.0, 2.0]) * u.Hz
    sm_s = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x_s)
    sm_hz = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x_hz)
    with pytest.raises(ValueError):
        sm_s.is_compatible(sm_hz)


def test_sign_preserves_unit_and_meta():
    data = np.array([[[ -1.0, 2.0 ]]])
    xindex = np.array([0.0, 1.0])
    sm = SeriesMatrix(data, xindex=xindex, units=[[u.m]])
    out = np.sign(sm)
    assert np.array_equal(out.value, [[[-1., 1.]]])
    assert out.meta[0,0].unit == u.m


def test_floor_divide_mod_passthrough_meta():
    data = np.array([[[5.0, 6.0]]])
    xindex = np.array([0.0, 1.0])
    sm = SeriesMatrix(data, xindex=xindex, units=[[u.s]])
    out_fd = np.floor_divide(sm, 2)
    out_mod = np.mod(sm, 2)
    assert out_fd.meta[0,0].unit == u.s
    assert out_mod.meta[0,0].unit == u.s
    assert np.array_equal(out_fd.value, [[[2., 3.]]])
    assert np.array_equal(out_mod.value, [[[1., 0.]]])


def test_clip_passthrough_meta():
    data = np.array([[[ -1.0, 0.5, 2.0 ]]])
    xindex = np.array([0.0, 1.0, 2.0])
    sm = SeriesMatrix(data, xindex=xindex, units=[[u.Hz]])
    out = np.clip(sm, 0.0, 1.0)
    assert out.meta[0,0].unit == u.Hz
    assert np.array_equal(out.value, [[[0.0, 0.5, 1.0]]])


def test_isclose_bool_output():
    a = SeriesMatrix(np.array([[[1.0, 2.0]]]), xindex=np.array([0.0, 1.0]))
    b = SeriesMatrix(np.array([[[1.0, 2.1]]]), xindex=np.array([0.0, 1.0]))
    out = np.isclose(a, b, atol=0.05)
    assert out.value.dtype == np.bool_
    assert np.array_equal(out.value, [[[True, False]]])


def test_is_compatible_gwpy_vs_exact():
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([0.0, 1.0, 2.0])
    sm1 = SeriesMatrix(np.zeros((1, 1, 3)), xindex=x1, units=[[u.m]])
    sm2 = SeriesMatrix(np.ones((1, 1, 3)), xindex=x2, units=[[u.m]])
    assert sm1.is_compatible(sm2)
    assert sm1.is_compatible_exact(sm2)


def test_is_contiguous_gwpy_allows_different_length():
    sm1 = SeriesMatrix(np.zeros((1, 1, 3)), xindex=np.array([0.0, 1.0, 2.0]))
    sm2 = SeriesMatrix(np.ones((1, 1, 2)), xindex=np.array([3.0, 4.0]))
    assert sm1.is_contiguous(sm2) == 1


def test_append_gwpy_pad_gap_option():
    sm1 = SeriesMatrix(np.zeros((1, 1, 2)), xindex=np.array([0.0, 1.0]))
    sm2 = SeriesMatrix(np.ones((1, 1, 2)), xindex=np.array([4.0, 5.0]))
    out = sm1.append(sm2, gap='pad', pad=0.0)
    assert out.shape == (1, 1, 6)
    # should have zeros padding between
    assert np.array_equal(out.value.flatten(), np.array([0., 0., 0., 0., 1., 1.]))

if __name__ == "__main__":
    try:
        test_basic_import_and_instantiation()
        test_value_internal_storage()
        test_quantity_input_unit_defaults()
        test_xindex_quantity_preserves_unit()
        test_xarray_duration_no_double_units()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
