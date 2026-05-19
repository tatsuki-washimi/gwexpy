"""Deterministic generator specs for the IO conformance harness."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GENERATOR_SPECS", "GeneratorSpec", "iter_generator_specs"]


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    """Describe one conformance generator module."""

    name: str
    module_name: str
    entrypoint: str = "generate"


GENERATOR_SPECS: tuple[GeneratorSpec, ...] = (
    GeneratorSpec(
        name="csv_txt",
        module_name="tests.io_conformance.generators.csv_txt",
    ),
    GeneratorSpec(
        name="audio",
        module_name="tests.io_conformance.generators.audio",
    ),
    GeneratorSpec(
        name="hdf5",
        module_name="tests.io_conformance.generators.hdf5",
    ),
    GeneratorSpec(
        name="hdf_ndscope",
        module_name="tests.io_conformance.generators.hdf_ndscope",
    ),
    GeneratorSpec(
        name="gwf",
        module_name="tests.io_conformance.generators.gwf",
    ),
)


def iter_generator_specs() -> tuple[GeneratorSpec, ...]:
    """Return the fixed generator set in a deterministic order."""

    return GENERATOR_SPECS
