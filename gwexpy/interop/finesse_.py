"""
gwexpy.interop.finesse_
------------------------

Interoperability with Finesse 3 interferometer simulation library.

Provides conversion from Finesse's ``FrequencyResponseSolution`` and
``NoiseProjectionSolution`` to GWexpy FrequencySeries types.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional

__all__ = ["from_finesse_frequency_response", "from_finesse_noise"]


def from_finesse_frequency_response(
    cls: type,
    sol: Any,
    *,
    output: Any | None = None,
    input_dof: Any | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create FrequencySeries or FrequencySeriesMatrix from a Finesse 3
    ``FrequencyResponseSolution``.

    Parameters
    ----------
    cls : type
        The FrequencySeries (or FrequencySeriesDict) class to instantiate.
    sol : finesse.analysis.actions.lti.FrequencyResponseSolution
        The frequency response solution from a Finesse 3 simulation.
    output : str or object, optional
        Output DOF name to extract. If *None* and *input_dof* is also *None*,
        all output/input pairs are returned.
    input_dof : str or object, optional
        Input DOF name to extract. Must be combined with *output* to select
        a single transfer function.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the resulting data. Finesse solutions do not carry
        astropy units, so this must be supplied by the caller if physical
        units are desired.

    Returns
    -------
    FrequencySeries
        When a single (output, input_dof) pair is selected.
    FrequencySeriesMatrix
        When multiple output/input pairs exist and no specific pair is
        selected, or when only *output* is given (all inputs for that output).
    FrequencySeriesDict
        When called from ``FrequencySeriesDict.from_finesse_frequency_response``
        (cls is a dict type).

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> # sol = model.run(...)  # Finesse 3 simulation
    >>> fs = FrequencySeries.from_finesse_frequency_response(
    ...     sol, output="DARM", input_dof="EX_drive"
    ... )
    """
    require_optional("finesse")

    freqs = np.asarray(sol.f, dtype=np.float64)

    # Determine if caller wants dict-mode
    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
    want_dict = FrequencySeriesDict is not None and (
        cls is FrequencySeriesDict or _is_subclass_safe(cls, FrequencySeriesDict)
    )

    # Single transfer function
    if output is not None and input_dof is not None:
        data = np.asarray(sol[output, input_dof])
        name = f"{output} -> {input_dof}"
        FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
        return FrequencySeries(data, frequencies=freqs, name=name, unit=unit)

    outputs = [str(o) for o in sol.outputs]
    inputs = [str(i) for i in sol.inputs]

    # Single output, all inputs
    if output is not None:
        pairs = [(output, inp) for inp in sol.inputs]
        return _build_frequency_response_collection(
            cls,
            sol,
            freqs,
            pairs,
            outputs=[str(output)],
            inputs=inputs,
            unit=unit,
            want_dict=want_dict,
        )

    # All outputs × all inputs
    pairs = [(out, inp) for out in sol.outputs for inp in sol.inputs]

    # Single pair → single FrequencySeries
    if len(pairs) == 1 and not want_dict:
        out, inp = pairs[0]
        data = np.asarray(sol[out, inp])
        name = f"{out} -> {inp}"
        FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
        return FrequencySeries(data, frequencies=freqs, name=name, unit=unit)

    return _build_frequency_response_collection(
        cls,
        sol,
        freqs,
        pairs,
        outputs=outputs,
        inputs=inputs,
        unit=unit,
        want_dict=want_dict,
    )


def _build_frequency_response_collection(
    cls: type,
    sol: Any,
    freqs: np.ndarray,
    pairs: list[tuple[Any, Any]],
    *,
    outputs: list[str],
    inputs: list[str],
    unit: Any | None,
    want_dict: bool,
) -> Any:
    """Build a FrequencySeriesMatrix or FrequencySeriesDict from DOF pairs."""
    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")

    if want_dict:
        FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
        result = FrequencySeriesDict()
        for out, inp in pairs:
            data = np.asarray(sol[out, inp])
            key = f"{out} -> {inp}"
            result[key] = FrequencySeries(data, frequencies=freqs, name=key, unit=unit)
        return result

    # Build as FrequencySeriesMatrix
    FrequencySeriesMatrix = ConverterRegistry.get_constructor("FrequencySeriesMatrix")

    n_out = len(outputs)
    n_in = len(inputs)
    n_freq = len(freqs)

    matrix_data = np.empty((n_out, n_in, n_freq), dtype=complex)
    channel_names = np.empty((n_out, n_in), dtype=object)

    for i, out_dof in enumerate(outputs):
        for j, in_dof in enumerate(inputs):
            matrix_data[i, j, :] = np.asarray(sol[out_dof, in_dof])
            channel_names[i, j] = f"{out_dof} -> {in_dof}"

    sol_name = getattr(sol, "name", None)

    return FrequencySeriesMatrix(
        matrix_data,
        frequencies=freqs,
        channel_names=channel_names,
        unit=unit,
        name=str(sol_name) if sol_name else None,
    )


def from_finesse_noise(
    cls: type,
    sol: Any,
    *,
    output: Any | None = None,
    noise: str | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create FrequencySeries or FrequencySeriesDict from a Finesse 3
    ``NoiseProjectionSolution``.

    Parameters
    ----------
    cls : type
        The FrequencySeries (or FrequencySeriesDict) class to instantiate.
    sol : finesse.analysis.actions.noise.NoiseProjectionSolution
        The noise projection solution from a Finesse 3 simulation.
    output : str or object, optional
        Output node name to extract. If *None*, all output nodes are included.
    noise : str, optional
        Specific noise source name to extract. If *None*, all noise sources
        are included.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the resulting data (e.g., ``"m/sqrt(Hz)"``).

    Returns
    -------
    FrequencySeries
        When a single output and noise source are both specified.
    FrequencySeriesDict
        When multiple outputs or noise sources are present.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> # sol = model.run(...)  # Finesse 3 noise simulation
    >>> fs = FrequencySeries.from_finesse_noise(
    ...     sol, output="nDARMout", noise="laser_freq"
    ... )
    """
    require_optional("finesse")

    freqs = np.asarray(sol.f, dtype=np.float64)
    noises = list(sol.noises)

    # Resolve output nodes
    if output is not None:
        output_nodes = [output]
    else:
        output_nodes = list(sol.output_nodes)

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")

    # Single output + single noise source → FrequencySeries
    if len(output_nodes) == 1 and noise is not None:
        out_node = output_nodes[0]
        noise_idx = noises.index(noise)
        data = np.asarray(sol.out[out_node][:, noise_idx], dtype=np.float64)
        name = f"{out_node}: {noise}"
        return FrequencySeries(data, frequencies=freqs, name=name, unit=unit)

    # Build a FrequencySeriesDict for multiple entries
    result = FrequencySeriesDict()

    for out_node in output_nodes:
        out_data = np.asarray(sol.out[out_node])

        if noise is not None:
            noise_idx = noises.index(noise)
            data = np.asarray(out_data[:, noise_idx], dtype=np.float64)
            key = f"{out_node}: {noise}"
            result[key] = FrequencySeries(data, frequencies=freqs, name=key, unit=unit)
        else:
            for n_idx, n_name in enumerate(noises):
                data = np.asarray(out_data[:, n_idx], dtype=np.float64)
                key = f"{out_node}: {n_name}"
                result[key] = FrequencySeries(
                    data, frequencies=freqs, name=key, unit=unit
                )

    # If only one entry in the dict and cls is FrequencySeries, unwrap
    if len(result) == 1 and not (
        cls is FrequencySeriesDict or _is_subclass_safe(cls, FrequencySeriesDict)
    ):
        return next(iter(result.values()))

    return result


def _is_subclass_safe(cls: type, parent: type) -> bool:
    """Check issubclass without raising on non-class inputs."""
    try:
        return issubclass(cls, parent)
    except TypeError:
        return False
