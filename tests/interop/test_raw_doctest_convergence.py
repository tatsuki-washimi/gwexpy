import ast
import doctest
import importlib
import inspect

from docutils import nodes
from docutils.core import publish_doctree

import gwexpy.interop

FRAGMENTS = {
    ("emg3d_", "from_emg3d_field"): {
        "prerequisite": "pre-existing emg3d Field",
        "requirements": ("pre-existing emg3d Field",),
        "symbols": ("from_emg3d_field",),
    },
    ("finesse_", "from_finesse_frequency_response"): {
        "prerequisite": "live Finesse 3 model run",
        "requirements": ("live Finesse 3 model run",),
        "symbols": ("from_finesse_frequency_response",),
    },
    ("finesse_", "from_finesse_noise"): {
        "prerequisite": "live Finesse 3 model run",
        "requirements": ("live Finesse 3 model run",),
        "symbols": ("from_finesse_noise",),
    },
    ("meep_", "from_meep_hdf5"): {
        "prerequisite": "optional h5py package",
        "requirements": ("optional h5py package", "external Meep HDF5"),
        "symbols": ("from_meep_hdf5",),
    },
    ("mt_", "to_mth5"): {
        "prerequisite": "external MTH5 file",
        "requirements": ("external MTH5 file", "filesystem state"),
        "symbols": ("to_mth5",),
    },
    ("mt_", "from_mth5"): {
        "prerequisite": "external MTH5 file",
        "requirements": ("external MTH5 file", "filesystem state"),
        "symbols": ("from_mth5",),
    },
    ("openems_", "from_openems_hdf5"): {
        "prerequisite": "optional h5py package",
        "requirements": ("optional h5py package", "external openEMS", "HDF5 dump"),
        "symbols": ("from_openems_hdf5",),
    },
    ("pycbc_", "from_pycbc_timeseries"): {
        "prerequisite": "optional PyCBC package",
        "requirements": ("optional PyCBC package",),
        "symbols": ("from_pycbc_timeseries",),
    },
    ("pycbc_", "to_pycbc_timeseries"): {
        "prerequisite": "optional PyCBC package",
        "requirements": ("optional PyCBC package",),
        "symbols": ("to_pycbc_timeseries",),
    },
    ("pycbc_", "from_pycbc_frequencyseries"): {
        "prerequisite": "optional PyCBC package",
        "requirements": ("optional PyCBC package",),
        "symbols": ("from_pycbc_frequencyseries",),
    },
    ("pycbc_", "to_pycbc_frequencyseries"): {
        "prerequisite": "optional PyCBC package",
        "requirements": ("optional PyCBC package",),
        "symbols": ("to_pycbc_frequencyseries",),
    },
    ("pyspice_", "from_pyspice_transient"): {
        "prerequisite": "optional PySpice package",
        "requirements": ("optional PySpice package", "live simulator"),
        "symbols": ("from_pyspice_transient",),
    },
    ("pyspice_", "from_pyspice_ac"): {
        "prerequisite": "optional PySpice package",
        "requirements": ("optional PySpice package", "live simulator"),
        "symbols": ("from_pyspice_ac",),
    },
    ("pyspice_", "from_pyspice_noise"): {
        "prerequisite": "optional PySpice package",
        "requirements": ("optional PySpice package", "live simulator"),
        "symbols": ("from_pyspice_noise",),
    },
    ("pyspice_", "from_pyspice_distortion"): {
        "prerequisite": "optional PySpice package",
        "requirements": ("optional PySpice package", "live simulator"),
        "symbols": ("from_pyspice_distortion",),
    },
    ("skrf_", "from_skrf_network"): {
        "prerequisite": "optional scikit-rf package",
        "requirements": ("optional scikit-rf package", "external", "Touchstone file"),
        "symbols": ("from_skrf_network",),
    },
    ("skrf_", "to_skrf_network"): {
        "prerequisite": "optional scikit-rf package",
        "requirements": ("optional scikit-rf package",),
        "symbols": ("to_skrf_network",),
    },
    ("skrf_", "from_skrf_impulse_response"): {
        "prerequisite": "optional scikit-rf package",
        "requirements": ("optional scikit-rf package", "external", "Touchstone file"),
        "symbols": ("from_skrf_impulse_response",),
    },
    ("skrf_", "from_skrf_step_response"): {
        "prerequisite": "optional scikit-rf package",
        "requirements": ("optional scikit-rf package", "external", "Touchstone file"),
        "symbols": ("from_skrf_step_response",),
    },
}


def _function_doc(module_name: str, function_name: str) -> str:
    module = importlib.import_module(f"gwexpy.interop.{module_name}")
    function = getattr(module, function_name)
    return inspect.getdoc(function) or ""


def _python_blocks(docstring: str) -> list[nodes.literal_block]:
    doctree = publish_doctree(docstring)
    return [
        node
        for node in doctree.findall(nodes.literal_block)
        if "python" in node.get("classes", [])
    ]


def _preceding_prose(block: nodes.literal_block) -> str:
    assert block.parent is not None
    index = block.parent.index(block)
    return "\n".join(
        node.astext()
        for node in block.parent.children[:index]
        if isinstance(node, (nodes.paragraph, nodes.title))
    )


def test_illustrative_fragments_are_regular_python_blocks_with_bound_prerequisites():
    parser = doctest.DocTestParser()

    for (module_name, function_name), contract in FRAGMENTS.items():
        docstring = _function_doc(module_name, function_name)
        blocks = _python_blocks(docstring)
        assert blocks, (module_name, function_name)
        assert parser.get_examples(docstring) == []
        assert "+SKIP" not in docstring

        for block in blocks:
            assert contract["prerequisite"] in _preceding_prose(block)
            ast.parse(block.astext())


def test_fragment_prerequisites_include_required_dependency_and_external_state_words():
    for (module_name, function_name), contract in FRAGMENTS.items():
        docstring = _function_doc(module_name, function_name)
        for requirement in contract["requirements"]:
            assert requirement in docstring, (module_name, function_name, requirement)


def test_documented_converter_symbols_resolve_from_the_public_facade():
    facade = importlib.import_module("gwexpy.interop")

    for (module_name, function_name), contract in FRAGMENTS.items():
        block_source = "\n".join(
            block.astext()
            for block in _python_blocks(_function_doc(module_name, function_name))
        )
        tree = ast.parse(block_source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "gwexpy.interop"
            for alias in node.names
        }
        for symbol in contract["symbols"]:
            assert symbol in imported
            assert getattr(facade, symbol) is not None


def test_pycbc_examples_bind_to_the_public_facade_identity():
    implementation = importlib.import_module("gwexpy.interop.pycbc_")
    facade = importlib.import_module("gwexpy.interop")

    for name in (
        "from_pycbc_timeseries",
        "to_pycbc_timeseries",
        "from_pycbc_frequencyseries",
        "to_pycbc_frequencyseries",
    ):
        assert getattr(facade, name) is getattr(implementation, name)
