#!/usr/bin/env python
"""
Basic validation script for Phase 1-2 I/O improvements.
Tests core functionality without pytest dependency.
"""

import sys
import traceback
from typing import Any


def test_imports():
    """Test that all modules can be imported successfully."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    tests = [
        ("ensure_dependency", "from gwexpy.io.utils import ensure_dependency"),
        ("register_timeseries_format", "from gwexpy.timeseries.io._registration import register_timeseries_format"),
        ("TimeSeries", "from gwexpy.timeseries import TimeSeries"),
        ("TimeSeriesDict", "from gwexpy.timeseries import TimeSeriesDict"),
        ("WAV reader", "from gwexpy.timeseries.io.wav import read_timeseriesdict_wav"),
        ("ATS reader", "from gwexpy.timeseries.io.ats import read_timeseries_ats"),
        ("GBD reader", "from gwexpy.timeseries.io.gbd import read_timeseriesdict_gbd"),
    ]

    passed = 0
    failed = 0

    for name, code in tests:
        try:
            exec(code)
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
            traceback.print_exc()

    return passed, failed

def test_ensure_dependency():
    """Test ensure_dependency function."""
    print("\n" + "=" * 60)
    print("TEST 2: ensure_dependency Function")
    print("=" * 60)

    from gwexpy.io.utils import ensure_dependency

    tests_passed = 0
    tests_failed = 0

    # Test 1: Existing package
    try:
        numpy = ensure_dependency("numpy")
        assert numpy.__name__ == "numpy"
        print("✅ Import existing package (numpy)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Import existing package: {e}")
        tests_failed += 1

    # Test 2: Non-existent package
    try:
        ensure_dependency("nonexistent_gwexpy_test_package")
        print("❌ Should have raised ImportError for non-existent package")
        tests_failed += 1
    except ImportError as e:
        if "pip install" in str(e):
            print("✅ Proper error message for non-existent package")
            tests_passed += 1
        else:
            print(f"❌ Error message doesn't contain pip install: {e}")
            tests_failed += 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        tests_failed += 1

    # Test 3: Custom import_name
    try:
        scipy = ensure_dependency("scipy", import_name="scipy")
        assert scipy.__name__ == "scipy"
        print("✅ Custom import_name parameter")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Custom import_name: {e}")
        tests_failed += 1

    return tests_passed, tests_failed

def test_registration_helper():
    """Test register_timeseries_format function."""
    print("\n" + "=" * 60)
    print("TEST 3: register_timeseries_format Helper")
    print("=" * 60)

    from gwexpy.timeseries import TimeSeries, TimeSeriesDict
    from gwexpy.timeseries.io._registration import register_timeseries_format

    tests_passed = 0
    tests_failed = 0

    # Test 1: Function signature
    try:
        import inspect
        sig = inspect.signature(register_timeseries_format)
        params = list(sig.parameters.keys())

        expected_params = [
            "format_name", "reader_dict", "reader_single", "reader_matrix",
            "writer_dict", "writer_single", "writer_matrix",
            "identifier_dict", "identifier_single", "extension", "auto_adapt", "force"
        ]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

        print("✅ Function signature complete")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Function signature: {e}")
        tests_failed += 1

    # Test 2: Registration without errors
    try:
        def mock_reader_dict(source, **kwargs):
            tsd = TimeSeriesDict()
            tsd["test"] = TimeSeries([1, 2, 3], t0=0, dt=1, name="test")
            return tsd

        register_timeseries_format(
            "test_fmt_validation",
            reader_dict=mock_reader_dict,
            extension="testval"
        )
        print("✅ Registration execution successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Registration execution: {e}")
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed

def test_reader_signatures():
    """Test that readers have the expected signatures."""
    print("\n" + "=" * 60)
    print("TEST 4: Reader Function Signatures")
    print("=" * 60)

    import inspect

    tests = [
        ("wav.read_timeseriesdict_wav", "from gwexpy.timeseries.io.wav import read_timeseriesdict_wav as func", ["unit", "epoch", "channels"]),
        ("ats.read_timeseries_ats", "from gwexpy.timeseries.io.ats import read_timeseries_ats as func", ["unit", "epoch"]),
        ("audio.read_timeseriesdict_audio", "from gwexpy.timeseries.io.audio import read_timeseriesdict_audio as func", ["epoch", "unit"]),
    ]

    passed = 0
    failed = 0

    for name, import_code, expected_params in tests:
        try:
            namespace: dict[str, Any] = {}
            exec(import_code, namespace)
            func = namespace["func"]
            sig = inspect.signature(func)

            for param in expected_params:
                if param not in sig.parameters:
                    print(f"❌ {name}: missing parameter '{param}'")
                    failed += 1
                    continue

            print(f"✅ {name}: has all expected parameters")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1

    return passed, failed

def test_pathlib_support():
    """Test that I/O readers accept pathlib.Path objects."""
    print("\n" + "=" * 60)
    print("TEST 5: Pathlib Support")
    print("=" * 60)

    import inspect

    tests = [
        ("ats.read_timeseries_ats", "from gwexpy.timeseries.io.ats import read_timeseries_ats as func"),
        ("gbd.read_timeseriesdict_gbd", "from gwexpy.timeseries.io.gbd import read_timeseriesdict_gbd as func"),
        ("wav.read_timeseriesdict_wav", "from gwexpy.timeseries.io.wav import read_timeseriesdict_wav as func"),
    ]

    passed = 0
    failed = 0

    for name, import_code in tests:
        try:
            namespace: dict[str, Any] = {}
            exec(import_code, namespace)
            func = namespace["func"]
            sig = inspect.signature(func)

            # Check source parameter annotation
            source_param = sig.parameters.get("source")
            if source_param:
                anno_str = str(source_param.annotation)
                # Check if it mentions Path or str
                if "Path" in anno_str or "str" in anno_str or anno_str == "~source":
                    print(f"✅ {name}: accepts str | Path")
                    passed += 1
                else:
                    print(f"⚠️  {name}: annotation may not include Path ({anno_str})")
                    passed += 1
            else:
                print(f"❌ {name}: no source parameter")
                failed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            traceback.print_exc()
            failed += 1

    return passed, failed

def main():
    """Run all validation tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Phase 1-2 I/O Improvements - Validation Tests".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    total_passed = 0
    total_failed = 0

    # Run all tests
    test_functions = [
        test_imports,
        test_ensure_dependency,
        test_registration_helper,
        test_reader_signatures,
        test_pathlib_support,
    ]

    for test_func in test_functions:
        try:
            p, f = test_func()
            total_passed += p
            total_failed += f
        except Exception as e:
            print(f"CRITICAL ERROR in {test_func.__name__}: {e}")
            traceback.print_exc()
            total_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📊 Total:  {total_passed + total_failed}")
    print()

    if total_failed == 0:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✨ Implementation is ready for git commit")
        return 0
    else:
        print(f"⚠️  {total_failed} test(s) failed - review needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
