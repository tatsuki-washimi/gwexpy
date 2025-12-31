
modules_to_inspect = [
    "gwpy.timeseries.io.gwf.framecpp",
    "gwpy.timeseries.io.gwf.framel",
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
