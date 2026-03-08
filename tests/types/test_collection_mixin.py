from collections import UserList

from gwexpy.types.mixin._collection_mixin import ListMapMixin, _make_list_map_method


class _Item:
    def __init__(self, value: int):
        self.value = value

    def bump(self, amount: int = 1):
        return self.value + amount


class _UserListResult(UserList):
    pass


class _Container(ListMapMixin, UserList):
    bump = _make_list_map_method(
        "bump",
        result_class_path="tests.types.test_collection_mixin._UserListResult",
    )


class _ContainerNoPath(ListMapMixin, UserList):
    bump = _make_list_map_method("bump")


def test_make_list_map_method_supports_userlist_result_class_path():
    container = _Container([_Item(1), _Item(2)])

    out = container.bump(amount=3)

    assert isinstance(out, _UserListResult)
    assert out == [4, 5]


def test_make_list_map_method_supports_userlist_without_result_path():
    container = _ContainerNoPath([_Item(3), _Item(4)])

    out = container.bump(amount=2)

    assert isinstance(out, _ContainerNoPath)
    assert out == [5, 6]
