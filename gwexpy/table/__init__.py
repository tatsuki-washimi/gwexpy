# Dynamic import from gwpy (PEP 562)
import gwpy.table


def __getattr__(name):
    return getattr(gwpy.table, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.table)))
