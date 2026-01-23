"""Base class for field objects with domain-aware axis metadata."""

from __future__ import annotations

from typing import Any

from astropy import units as u

from ..types.array4d import Array4D

__all__ = ["FieldBase"]


class FieldBase(Array4D):
    """Base class providing axis domain metadata handling for field objects.

    Maintains the invariant of four coordinate axes `(axis0, x, y, z)` and
    explicit domain labels for temporal and spatial axes.
    """

    _metadata_slots = Array4D._metadata_slots + (
        "_axis0_domain",
        "_space_domains",
        "_axis0_offset",  # Preserved during fft_time for ifft_time reconstruction
    )

    # Axis name conventions
    _TIME_AXIS_NAME = "t"
    _FREQ_AXIS_NAME = "f"
    _REAL_AXIS_NAMES = ("x", "y", "z")
    _K_AXIS_NAMES = ("kx", "ky", "kz")

    _axis0_index: Any
    _axis1_index: Any
    _axis2_index: Any
    _axis3_index: Any
    _space_domains: dict[str, str]

    def __new__(
        cls,
        data,
        unit=None,
        axis0=None,
        axis1=None,
        axis2=None,
        axis3=None,
        axis_names=None,
        axis0_domain: str = "time",
        space_domain: str | dict[str, str] = "real",
        **kwargs,
    ):
        # Set default axis names based on domain
        if axis_names is None:
            time_name = (
                cls._TIME_AXIS_NAME
                if axis0_domain == "time"
                else cls._FREQ_AXIS_NAME
            )

            if isinstance(space_domain, dict):
                space_names = ["x", "y", "z"]  # defaults when dict provided
            elif space_domain == "k":
                space_names = list(cls._K_AXIS_NAMES)
            else:
                space_names = list(cls._REAL_AXIS_NAMES)
            axis_names = [time_name] + space_names

        obj = super().__new__(
            cls,
            data,
            unit=unit,
            axis0=axis0,
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            axis_names=axis_names,
            **kwargs,
        )

        # Set domain states
        if axis0_domain not in ("time", "frequency"):
            raise ValueError(
                f"axis0_domain must be 'time' or 'frequency', got '{axis0_domain}'"
            )
        obj._axis0_domain = axis0_domain

        # Handle space_domain: str -> all same, dict -> per-axis
        if isinstance(space_domain, str):
            if space_domain not in ("real", "k"):
                raise ValueError(
                    f"space_domain must be 'real' or 'k', got '{space_domain}'"
                )
            obj._space_domains = {
                obj._axis1_name: space_domain,
                obj._axis2_name: space_domain,
                obj._axis3_name: space_domain,
            }
        elif isinstance(space_domain, dict):
            for name, dom in space_domain.items():
                if dom not in ("real", "k"):
                    raise ValueError(
                        f"space_domain values must be 'real' or 'k', "
                        f"got '{dom}' for '{name}'"
                    )
            obj._space_domains = dict(space_domain)
        else:
            raise TypeError(
                f"space_domain must be str or dict, got {type(space_domain)}"
            )

        obj._validate_domain_units()

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return

        # Copy domains from parent if available
        if getattr(self, "_axis0_domain", None) is None:
            self._axis0_domain = getattr(obj, "_axis0_domain", "time")

        parent_domains = getattr(obj, "_space_domains", None)
        if parent_domains is not None:
            # Check if axis names match. If they don't, we need to map keys.
            # But usually if we just copied names in super().__array_finalize__, 
            # they should match.
            self._space_domains = dict(parent_domains)
        elif getattr(self, "_space_domains", None) is None:
            self._space_domains = {
                self._axis1_name: "real",
                self._axis2_name: "real",
                self._axis3_name: "real",
            }

        if getattr(self, "_axis0_offset", None) is None:
            self._axis0_offset = getattr(obj, "_axis0_offset", None)

        self._validate_domain_units()

    @property
    def axis0_domain(self):
        """Domain of axis 0: 'time' or 'frequency'."""
        return self._axis0_domain

    @property
    def space_domains(self):
        """Mapping of spatial axis names to domains."""
        return dict(self._space_domains)

    # ------------------------------------------------------------------
    # Domain/unit validation
    # ------------------------------------------------------------------
    def _validate_domain_units(self) -> None:
        """Validate that axis units are consistent with declared domains."""

        # Axis 0: time or frequency
        axis0 = getattr(self, "_axis0_index", None)
        if isinstance(axis0, u.Quantity):
            axis0_unit = axis0.unit
            if axis0_unit != u.dimensionless_unscaled:
                expected = u.s if self._axis0_domain == "time" else 1 / u.s
                if not axis0_unit.is_equivalent(expected):
                    raise ValueError(
                        f"Axis0 domain '{self._axis0_domain}' expects units "
                        f"equivalent to {expected}, got {axis0_unit}"
                    )

        # Spatial axes: position or wavenumber
        spatial_axes = [
            (self._axis1_name, getattr(self, "_axis1_index", None)),
            (self._axis2_name, getattr(self, "_axis2_index", None)),
            (self._axis3_name, getattr(self, "_axis3_index", None)),
        ]
        for name, axis in spatial_axes:
            domain = self._space_domains.get(name)
            if domain is None or not isinstance(axis, u.Quantity):
                continue
            axis_unit = axis.unit
            if axis_unit == u.dimensionless_unscaled:
                continue
            expected = u.m if domain == "real" else 1 / u.m
            if not axis_unit.is_equivalent(expected):
                raise ValueError(
                    f"Spatial axis '{name}' domain '{domain}' expects units "
                    f"equivalent to {expected}, got {axis_unit}"
                )
