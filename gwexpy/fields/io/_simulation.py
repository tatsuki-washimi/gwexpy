"""Lazy Field I/O registry integration for simulation formats."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._registration import register_field_format


def _select_single(
    names: dict[str, int], components: int, *, aliases: tuple[str, ...] = ()
) -> str:
    matches = [name for name, count in names.items() if count == components]
    if len(matches) > 1:
        matches = [name for name in matches if name not in aliases]
    if len(matches) == 1:
        return matches[0]
    kind = "scalar" if components == 1 else "vector"
    raise ValueError(
        f"Specify a {kind} field name. Compatible fields: {matches}. "
        f"Available fields: {list(names)}."
    )


def _openfoam_names(source: Any, options: dict[str, Any]) -> dict[str, int]:
    from gwexpy.interop._openfoam.convert import list_fields_openfoam
    from gwexpy.interop._openfoam.model import FoamSnapshot

    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        if not source:
            raise ValueError("Empty OpenFOAM source sequence.")
        source = source[0]
    if isinstance(source, FoamSnapshot):
        return {
            name: 3 if field.is_vector else 1 for name, field in source.fields.items()
        }
    discovery_options = {
        key: options[key]
        for key in ("time", "region", "skip_zero_time")
        if key in options
    }
    return {
        name: 3 if "vector" in (info["class"] or "").lower() else 1
        for name, info in list_fields_openfoam(source, **discovery_options).items()
    }


def _read_openfoam_scalar(source: Any, name: str | None = None, **kwargs: Any) -> Any:
    from gwexpy.interop._openfoam.convert import read_scalarfield_openfoam

    if name is None:
        name = _select_single(_openfoam_names(source, kwargs), 1)
    return read_scalarfield_openfoam(source, name, **kwargs)


def _read_openfoam_vector(source: Any, name: str | None = None, **kwargs: Any) -> Any:
    from gwexpy.interop._openfoam.convert import read_vectorfield_openfoam

    if name is None:
        name = _select_single(_openfoam_names(source, kwargs), 3)
    return read_vectorfield_openfoam(source, name, **kwargs)


def _read_openfoam_dict(
    source: Any, names: Sequence[str] | str | None = None, **kwargs: Any
) -> Any:
    from gwexpy.interop._openfoam.convert import read_fielddict_openfoam

    field_names = [names] if isinstance(names, str) else names
    return read_fielddict_openfoam(
        source, fields=field_names, _group_vectors=True, **kwargs
    )


def _list_openfoam(source: Any, **kwargs: Any) -> list[str]:
    return list(_openfoam_names(source, kwargs))


def _make_exodus_handlers(default_format: str) -> dict[str, Any]:
    def _read_scalar(source: Any, name: str | None = None, **kwargs: Any) -> Any:
        from gwexpy.interop._exodus.convert import read_scalarfield_exodus

        if name is None:
            name = _select_single(_metadata_names(source, kwargs), 1)
        return read_scalarfield_exodus(source, name, format=default_format, **kwargs)

    def _read_vector(source: Any, name: str | None = None, **kwargs: Any) -> Any:
        from gwexpy.interop._exodus.convert import read_vectorfield_exodus

        if name is None:
            name = _select_single(_metadata_names(source, kwargs), 3)
        return read_vectorfield_exodus(source, name, format=default_format, **kwargs)

    def _read_dict(
        source: Any, names: Sequence[str] | str | None = None, **kwargs: Any
    ) -> Any:
        from gwexpy.interop._exodus.convert import read_fielddict_exodus

        var_names = [names] if isinstance(names, str) else names
        return read_fielddict_exodus(
            source,
            variables=var_names,
            format=default_format,
            _public_selection=True,
            **kwargs,
        )

    def _metadata_names(source: Any, options: dict[str, Any]) -> dict[str, int]:
        from gwexpy.interop._exodus.convert import list_fields_exodus
        from gwexpy.interop._exodus.model import ExodusSnapshot
        from gwexpy.interop._exodus.profiles import (
            find_vector_components,
            get_exodus_profile,
        )

        profile = get_exodus_profile(default_format)
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            if not source:
                raise ValueError("Empty EXODUS source sequence.")
            source = source[0]
        if isinstance(source, ExodusSnapshot):
            names = dict.fromkeys(source.point_data, 1)
            for block in source.cell_data.values():
                names.update(dict.fromkeys(block, 1))
        else:
            metadata_options = {
                key: options[key]
                for key in ("unit_system", "backend")
                if key in options
            }
            metadata = list_fields_exodus(
                source, format=default_format, **metadata_options
            )
            names = {
                name: 1
                for name, info in metadata.items()
                if options.get("association") in (None, info["association"])
            }
        available = list(names)
        resolved = set()
        velocity = profile.find_vector_components(available)
        if velocity is not None:
            names.setdefault("velocity", 3)
            resolved.add(velocity)
        for name in tuple(names):
            if name.lower().endswith("x"):
                base = name[:-1].removesuffix("_")
                components = find_vector_components(profile, available, base)
                if components is not None and components not in resolved:
                    names.setdefault(base, 3)
                    resolved.add(components)
        return names

    def _lister(source: Any, **kwargs: Any) -> list[str]:
        return list(_metadata_names(source, kwargs))

    return {
        "reader_scalar": _read_scalar,
        "reader_vector": _read_vector,
        "reader_dict": _read_dict,
        "lister": _lister,
    }


def _read_specfem3d_scalar(source: Any, name: str | None = None, **kwargs: Any) -> Any:
    from gwexpy.interop._specfem3d.convert import read_scalarfield_specfem3d

    if name is None:
        name = _select_single(
            _specfem_names(source, kwargs.get("association")), 1, aliases=("density",)
        )
    return read_scalarfield_specfem3d(source, name, **kwargs)


def _read_specfem3d_vector(source: Any, name: str | None = None, **kwargs: Any) -> Any:
    from gwexpy.interop._specfem3d.convert import read_vectorfield_specfem3d

    if name is None:
        name = _select_single(_specfem_names(source), 3)
    return read_vectorfield_specfem3d(source, name, **kwargs)


def _read_specfem3d_dict(
    source: Any, names: Sequence[str] | str | None = None, **kwargs: Any
) -> Any:
    from gwexpy.interop._specfem3d.convert import read_fielddict_specfem3d

    field_names = [names] if isinstance(names, str) else names
    return read_fielddict_specfem3d(
        source, fields=field_names, _group_vectors=True, **kwargs
    )


def _specfem_names(source: Any, association: Any = None) -> dict[str, int]:
    from gwexpy.interop._specfem3d.convert import _discover_specfem_fields

    return _discover_specfem_fields(source, association=association)


def _list_specfem3d(source: Any, **kwargs: Any) -> list[str]:
    return list(_specfem_names(source, kwargs.get("association")))


def register_simulation_field_formats() -> None:
    """Register OpenFOAM, EXODUS II, and SPECFEM3D formats on Field I/O."""
    register_field_format(
        "openfoam",
        reader_scalar=_read_openfoam_scalar,
        reader_vector=_read_openfoam_vector,
        reader_dict=_read_openfoam_dict,
        lister=_list_openfoam,
        aliases=("foam", "vtk.openfoam"),
    )
    for exo_fmt in ("exo.flow3d", "exo"):
        register_field_format(exo_fmt, **_make_exodus_handlers(exo_fmt))
    register_field_format(
        "specfem3d",
        reader_scalar=_read_specfem3d_scalar,
        reader_vector=_read_specfem3d_vector,
        reader_dict=_read_specfem3d_dict,
        lister=_list_specfem3d,
        aliases=("h5.specfem3d", "vtk.specfem3d"),
    )


register_simulation_field_formats()
