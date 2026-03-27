"""Tests for gwexpy/interop/_registry.py."""
from __future__ import annotations

import pytest

from gwexpy.interop._registry import ConverterRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test by saving and restoring registry state."""
    saved_c = dict(ConverterRegistry._constructors)
    saved_v = dict(ConverterRegistry._converters)
    yield
    ConverterRegistry._constructors.clear()
    ConverterRegistry._constructors.update(saved_c)
    ConverterRegistry._converters.clear()
    ConverterRegistry._converters.update(saved_v)


class TestRegisterGetConstructor:
    def test_register_and_get(self):
        ConverterRegistry.register_constructor("MyClass", list)
        assert ConverterRegistry.get_constructor("MyClass") is list

    def test_get_missing_raises_key_error(self):
        with pytest.raises(KeyError, match="not registered"):
            ConverterRegistry.get_constructor("_NoSuchClass_")

    def test_error_message_lists_available(self):
        ConverterRegistry.register_constructor("AvailableClass", dict)
        with pytest.raises(KeyError, match="AvailableClass"):
            ConverterRegistry.get_constructor("_NoSuchClass_")

    def test_overwrite_same_class_no_warning(self):
        ConverterRegistry.register_constructor("MyClass", list)
        # Re-registering with the same class should not warn
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConverterRegistry.register_constructor("MyClass", list)
        assert not any("overwriting" in str(x.message) for x in w)

    def test_overwrite_different_class_warns(self):
        ConverterRegistry.register_constructor("MyClass", list)
        with pytest.warns(UserWarning, match="overwriting"):
            ConverterRegistry.register_constructor("MyClass", dict)
        assert ConverterRegistry.get_constructor("MyClass") is dict

    def test_has_constructor_true(self):
        ConverterRegistry.register_constructor("MyClass", set)
        assert ConverterRegistry.has_constructor("MyClass") is True

    def test_has_constructor_false(self):
        assert ConverterRegistry.has_constructor("_NoSuchClass_xyz_") is False


class TestRegisterGetConverter:
    def test_register_and_get(self):
        fn = lambda x: x
        ConverterRegistry.register_converter("my_conv", fn)
        assert ConverterRegistry.get_converter("my_conv") is fn

    def test_get_missing_raises_key_error(self):
        with pytest.raises(KeyError, match="not registered"):
            ConverterRegistry.get_converter("_no_such_converter_")

    def test_overwrite_different_func_warns(self):
        fn1 = lambda x: x
        fn2 = lambda x: x * 2
        ConverterRegistry.register_converter("conv", fn1)
        with pytest.warns(UserWarning, match="overwriting"):
            ConverterRegistry.register_converter("conv", fn2)
        assert ConverterRegistry.get_converter("conv") is fn2

    def test_has_converter_true(self):
        ConverterRegistry.register_converter("my_conv", lambda x: x)
        assert ConverterRegistry.has_converter("my_conv") is True

    def test_has_converter_false(self):
        assert ConverterRegistry.has_converter("_no_such_xyz_") is False
