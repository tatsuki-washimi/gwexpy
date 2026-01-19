"""Collections for Field4D objects."""

from .field4d import Field4D

__all__ = ["Field4DList", "Field4DDict"]


class Field4DList(list):
    """A list of Field4D objects with batch operations.

    This class provides convenient methods for applying operations
    to all Field4D objects in the list.

    Parameters
    ----------
    items : iterable of Field4D, optional
        Initial items for the list.
    validate : bool, optional
        If True, validate that all items are Field4D instances
        and have compatible metadata. Default is False.

    Examples
    --------
    >>> from gwexpy.types import Field4D, Field4DList
    >>> import numpy as np
    >>> fields = [Field4D(np.random.randn(10, 4, 4, 4)) for _ in range(3)]
    >>> flist = Field4DList(fields)
    >>> freq_fields = flist.fft_time_all()
    """

    def __init__(self, items=None, validate=False):
        if items is None:
            items = []
        super().__init__(items)
        if validate:
            self._validate()

    def _validate(self):
        """Validate that all items are Field4D with compatible metadata."""
        if not self:
            return

        first = self[0]
        if not isinstance(first, Field4D):
            raise TypeError(f"Expected Field4D, got {type(first)}")

        ref_unit = first.unit
        ref_axis_names = first.axis_names
        ref_axis0_domain = first.axis0_domain
        ref_space_domains = first.space_domains

        for i, item in enumerate(self[1:], 1):
            if not isinstance(item, Field4D):
                raise TypeError(f"Item {i}: Expected Field4D, got {type(item)}")
            if item.unit != ref_unit:
                raise ValueError(
                    f"Item {i}: Inconsistent unit. "
                    f"Expected {ref_unit}, got {item.unit}"
                )
            if item.axis_names != ref_axis_names:
                raise ValueError(
                    f"Item {i}: Inconsistent axis_names. "
                    f"Expected {ref_axis_names}, got {item.axis_names}"
                )
            if item.axis0_domain != ref_axis0_domain:
                raise ValueError(
                    f"Item {i}: Inconsistent axis0_domain. "
                    f"Expected {ref_axis0_domain}, got {item.axis0_domain}"
                )
            if item.space_domains != ref_space_domains:
                raise ValueError(
                    f"Item {i}: Inconsistent space_domains. "
                    f"Expected {ref_space_domains}, got {item.space_domains}"
                )

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.fft_time().

        Returns
        -------
        Field4DList
            List of transformed fields.
        """
        return Field4DList([f.fft_time(**kwargs) for f in self])

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.ifft_time().

        Returns
        -------
        Field4DList
            List of transformed fields.
        """
        return Field4DList([f.ifft_time(**kwargs) for f in self])

    def fft_space_all(self, **kwargs):
        """Apply fft_space to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.fft_space().

        Returns
        -------
        Field4DList
            List of transformed fields.
        """
        return Field4DList([f.fft_space(**kwargs) for f in self])

    def ifft_space_all(self, **kwargs):
        """Apply ifft_space to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.ifft_space().

        Returns
        -------
        Field4DList
            List of transformed fields.
        """
        return Field4DList([f.ifft_space(**kwargs) for f in self])


class Field4DDict(dict):
    """A dictionary of Field4D objects with batch operations.

    This class provides convenient methods for applying operations
    to all Field4D objects in the dictionary.

    Parameters
    ----------
    items : dict or iterable of (key, value), optional
        Initial items for the dictionary.
    validate : bool, optional
        If True, validate that all values are Field4D instances
        and have compatible metadata. Default is False.

    Examples
    --------
    >>> from gwexpy.types import Field4D, Field4DDict
    >>> import numpy as np
    >>> fields = {
    ...     'Ex': Field4D(np.random.randn(10, 4, 4, 4)),
    ...     'Ey': Field4D(np.random.randn(10, 4, 4, 4)),
    ... }
    >>> fdict = Field4DDict(fields)
    >>> freq_fields = fdict.fft_time_all()
    """

    def __init__(self, items=None, validate=False):
        if items is None:
            items = {}
        super().__init__(items)
        if validate:
            self._validate()

    def _validate(self):
        """Validate that all values are Field4D with compatible metadata."""
        if not self:
            return

        values = list(self.values())
        first = values[0]
        if not isinstance(first, Field4D):
            raise TypeError(f"Expected Field4D, got {type(first)}")

        ref_unit = first.unit
        ref_axis_names = first.axis_names
        ref_axis0_domain = first.axis0_domain
        ref_space_domains = first.space_domains

        for key, item in list(self.items())[1:]:
            if not isinstance(item, Field4D):
                raise TypeError(f"Key '{key}': Expected Field4D, got {type(item)}")
            if item.unit != ref_unit:
                raise ValueError(
                    f"Key '{key}': Inconsistent unit. "
                    f"Expected {ref_unit}, got {item.unit}"
                )
            if item.axis_names != ref_axis_names:
                raise ValueError(
                    f"Key '{key}': Inconsistent axis_names. "
                    f"Expected {ref_axis_names}, got {item.axis_names}"
                )
            if item.axis0_domain != ref_axis0_domain:
                raise ValueError(
                    f"Key '{key}': Inconsistent axis0_domain. "
                    f"Expected {ref_axis0_domain}, got {item.axis0_domain}"
                )
            if item.space_domains != ref_space_domains:
                raise ValueError(
                    f"Key '{key}': Inconsistent space_domains. "
                    f"Expected {ref_space_domains}, got {item.space_domains}"
                )

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.fft_time().

        Returns
        -------
        Field4DDict
            Dictionary of transformed fields.
        """
        return Field4DDict({k: v.fft_time(**kwargs) for k, v in self.items()})

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.ifft_time().

        Returns
        -------
        Field4DDict
            Dictionary of transformed fields.
        """
        return Field4DDict({k: v.ifft_time(**kwargs) for k, v in self.items()})

    def fft_space_all(self, **kwargs):
        """Apply fft_space to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.fft_space().

        Returns
        -------
        Field4DDict
            Dictionary of transformed fields.
        """
        return Field4DDict({k: v.fft_space(**kwargs) for k, v in self.items()})

    def ifft_space_all(self, **kwargs):
        """Apply ifft_space to all fields.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Field4D.ifft_space().

        Returns
        -------
        Field4DDict
            Dictionary of transformed fields.
        """
        return Field4DDict({k: v.ifft_space(**kwargs) for k, v in self.items()})
