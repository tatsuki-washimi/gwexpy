"""Collections for field objects in the unified `gwexpy.fields` namespace."""

from ..types.field4d_collections import Field4DDict, Field4DList

__all__ = ["FieldList", "FieldDict"]


class FieldList(Field4DList):
    """List-like collection for `ScalarField` objects with field-aware batch ops."""

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields, returning FieldList."""
        return FieldList([f.fft_time(**kwargs) for f in self])

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields, returning FieldList."""
        return FieldList([f.ifft_time(**kwargs) for f in self])

    def fft_space_all(self, axes=None, **kwargs):
        """Apply fft_space to all fields, returning FieldList."""
        return FieldList([f.fft_space(axes=axes, **kwargs) for f in self])

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space to all fields, returning FieldList."""
        return FieldList([f.ifft_space(axes=axes, **kwargs) for f in self])


class FieldDict(Field4DDict):
    """Dict-like collection for `ScalarField` objects with field-aware batch ops."""

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields, returning FieldDict."""
        return FieldDict({k: v.fft_time(**kwargs) for k, v in self.items()})

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields, returning FieldDict."""
        return FieldDict({k: v.ifft_time(**kwargs) for k, v in self.items()})

    def fft_space_all(self, axes=None, **kwargs):
        """Apply fft_space to all fields, returning FieldDict."""
        return FieldDict({k: v.fft_space(axes=axes, **kwargs) for k, v in self.items()})

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space to all fields, returning FieldDict."""
        return FieldDict({k: v.ifft_space(axes=axes, **kwargs) for k, v in self.items()})
