from gwexpy.timeseries import TimeSeries


def test_timeseries_attr_preservation():
    print("Testing TimeSeries attribute preservation...")

    # 1. Create a TimeSeries and set a custom attribute
    ts = TimeSeries([1, 2, 3, 4, 5], sample_rate=1, unit="V")
    ts._gwex_custom_attr = "preserved"
    ts.other_attr = "lost"  # Standard attributes might be lost without registry

    print(
        f"Original: _gwex_custom_attr = {getattr(ts, '_gwex_custom_attr', 'MISSING')}"
    )

    # 2. Slice operation
    ts_slice = ts[1:4]

    print(f"Slice type: {type(ts_slice)}")
    print(
        f"Slice: _gwex_custom_attr = {getattr(ts_slice, '_gwex_custom_attr', 'MISSING')}"
    )

    if getattr(ts_slice, "_gwex_custom_attr", None) == "preserved":
        print("PASS: Slicing preserved _gwex_ attribute")
    else:
        print("FAIL: Slicing lost _gwex_ attribute")
        exit(1)

    # 3. Arithmetic operation (creates new instance)
    ts_math = ts * 2
    print(
        f"Math: _gwex_custom_attr = {getattr(ts_math, '_gwex_custom_attr', 'MISSING')}"
    )

    if getattr(ts_math, "_gwex_custom_attr", None) == "preserved":
        print("PASS: Arithmetic preserved _gwex_ attribute")
    else:
        print("FAIL: Arithmetic lost _gwex_ attribute")
        exit(1)


if __name__ == "__main__":
    try:
        test_timeseries_attr_preservation()
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
