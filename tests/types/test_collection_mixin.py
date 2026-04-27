from __future__ import annotations

from collections import OrderedDict, UserList

import pytest

from gwexpy.types.mixin._collection_mixin import (
    DictMapMixin,
    ListMapMixin,
    _make_dict_map_method,
    _make_dict_plain_method,
    _make_list_map_method,
    _make_list_plain_method,
)


class _Item:
    def __init__(self, value: int):
        self.value = value

    def scaled(self, factor: int = 1) -> _Item:
        return _Item(self.value * factor)

    def scalar(self) -> int:
        return self.value

    def fail(self) -> None:
        raise RuntimeError(f"boom {self.value}")


class _DictCollection(DictMapMixin, OrderedDict[str, _Item]):
    scaled = _make_dict_map_method("scaled")
    scalar = _make_dict_plain_method("scalar")
    fail = _make_dict_plain_method("fail")


class _DictWithOrderedResult(DictMapMixin, OrderedDict[str, _Item]):
    scaled = _make_dict_map_method(
        "scaled",
        result_class_path="collections.OrderedDict",
    )


class _ListCollection(ListMapMixin, list[_Item]):
    scaled = _make_list_map_method("scaled")
    scalar = _make_list_plain_method("scalar")
    fail = _make_list_plain_method("fail")


class _ListWithPlainResult(ListMapMixin, list[_Item]):
    scaled = _make_list_map_method("scaled", result_class_path="builtins.list")


class _UserListResult(UserList):
    pass


class _ListWithCustomResult(ListMapMixin, UserList):
    scaled = _make_list_map_method(
        "scaled",
        result_class_path="tests.types.test_collection_mixin._UserListResult",
    )


def test_dict_map_helper_preserves_order_and_returns_collection_type():
    collection = _DictCollection([("b", _Item(2)), ("a", _Item(1))])

    result = collection.scaled(10)

    assert isinstance(result, _DictCollection)
    assert list(result) == ["b", "a"]
    assert [item.value for item in result.values()] == [20, 10]
    assert [item.value for item in collection.values()] == [2, 1]


def test_dict_plain_helper_returns_native_dict_and_propagates_errors():
    collection = _DictCollection([("b", _Item(2)), ("a", _Item(1))])

    result = collection.scalar()

    assert type(result) is dict
    assert list(result) == ["b", "a"]
    assert result == {"b": 2, "a": 1}

    with pytest.raises(RuntimeError, match="boom 2"):
        collection.fail()


def test_dict_map_helper_can_return_configured_result_class():
    collection = _DictWithOrderedResult([("b", _Item(2)), ("a", _Item(1))])

    result = collection.scaled(3)

    assert type(result) is OrderedDict
    assert list(result) == ["b", "a"]
    assert [item.value for item in result.values()] == [6, 3]


def test_list_map_helper_preserves_order_and_returns_collection_type():
    collection = _ListCollection([_Item(2), _Item(1)])

    result = collection.scaled(10)

    assert isinstance(result, _ListCollection)
    assert [item.value for item in result] == [20, 10]
    assert [item.value for item in collection] == [2, 1]


def test_list_plain_helper_returns_native_list_and_propagates_errors():
    collection = _ListCollection([_Item(2), _Item(1)])

    result = collection.scalar()

    assert type(result) is list
    assert result == [2, 1]

    with pytest.raises(RuntimeError, match="boom 2"):
        collection.fail()


def test_list_map_helper_can_return_configured_result_class():
    collection = _ListWithPlainResult([_Item(2), _Item(1)])

    result = collection.scaled(3)

    assert type(result) is list
    assert [item.value for item in result] == [6, 3]


def test_list_map_helper_can_return_custom_imported_result_class():
    collection = _ListWithCustomResult([_Item(2), _Item(1)])

    result = collection.scaled(3)

    assert isinstance(result, _UserListResult)
    assert [item.value for item in result] == [6, 3]
