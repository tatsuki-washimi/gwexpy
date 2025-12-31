
modules_to_inspect = [
    "gwpy.table.tests",
    "gwpy.table.tests.test_gravityspy",
    "gwpy.table.tests.test_io_gstlal",
    "gwpy.table.tests.test_io_ligolw",
    "gwpy.table.tests.test_io_pycbc",
    "gwpy.table.tests.test_table",
]

import importlib

for mod_name in modules_to_inspect:
    try:
        mod = importlib.import_module(mod_name)
        exports = [n for n in dir(mod) if not n.startswith("_")]
        print(f"--- {mod_name} ---")
        print(exports)
    except ImportError as e:
        print(f"--- {mod_name} ---")
        print(f"SKIPPED due to ImportError: {e}")
    except Exception as e:
        print(f"--- {mod_name} ---")
        print(f"SKIPPED due to Error: {e}")
