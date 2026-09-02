#!/usr/bin/env python3
"""Build and validate the source/MRO GWpy override inventory for issue #639."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from collections.abc import Iterator, Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, get_args, get_origin

SCHEMA = "gwexpy-v023-gwpy-override-inventory-v1"
WORKER_SCHEMA = "gwexpy-v023-gwpy-override-oracle-v1"
SUPPORTED_GWPY = ("4.0.1", "4.0.2")
TERMINAL_STATES = ("fixed", "no-finding", "GWpy-fails", "GWexpy-only")
PROVISIONAL_STATES = ("unreviewed", "differential-required")
BEHAVIORAL_TERMINAL_STATES = frozenset({"fixed", "no-finding", "GWpy-fails"})
CONSTRUCTORS = frozenset({"__init__", "__new__"})
INTERNAL_CLASS_TOKENS = (
    "Mixin",
    "Base",
    "Core",
    "Protocol",
    "Interface",
    "MetaData",
)
ABSENT_FIXTURE = "__counterpart_absent__"
PENDING_FIXTURE = "__pending_differential__"
AUDIT_OWNER = "v0.2.3-compatibility-audit"
IMPLEMENTATION_BASE = "a8085b71446d3ef3417a7e5b5ac8efb156368eac"
PUBLIC_ROOT_RULE = (
    "byte-sorted gwexpy Python paths; literal top-level list/tuple __all__; "
    "static vars(module) exports plus two-pass unique canonical-class-name lazy "
    "alias association; canonical GWexpy class identity; internal root exclusions"
)
MEMBER_WALK_RULE = (
    "first effective vars(owner) binding in the GWexpy MRO prefix before "
    "the first GWpy class; public callable/descriptors plus __new__/__init__"
)
PRISTINE_ORACLE_RULE = (
    "separate -I worker; sanitized PYTHONPATH/PYTHONHOME; no GWexpy import; "
    "exact GWpy 4.0.1/4.0.2"
)
UPSTREAM_DEPENDENCY_PROVENANCE = (
    "GWpy providers retain package-relative source/line; inherited "
    "NumPy/Astropy providers retain normalized provider, member, kind, "
    "descriptor, and signature without source path or resolved version."
)


class InventoryError(RuntimeError):
    """Raised when inventory input or evidence fails closed validation."""


def canonical_compact_json(value: Any) -> str:
    """Return the canonical compact JSON representation used for digests."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_manifest_json(value: Any) -> str:
    """Return the canonical checked-in JSON representation."""

    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_compact_json(value).encode("ascii")).hexdigest()


def walk_manifest_values(value: Any) -> Iterator[Any]:
    """Yield every recursive manifest value for hygiene checks."""

    yield value
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from walk_manifest_values(key)
            yield from walk_manifest_values(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_manifest_values(item)


def _fqname(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _stable_atom(value: Any) -> dict[str, Any]:
    """Normalize defaults and annotations without address-bearing repr()."""

    if value is inspect.Parameter.empty or value is inspect.Signature.empty:
        return {"kind": "empty"}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            token = "nan"
        elif value > 0:
            token = "+inf"
        else:
            token = "-inf"
        return {"kind": "float", "value": token}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "literal", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_stable_atom(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_stable_atom(item) for item in value]
        return {"kind": "frozenset", "items": sorted(items, key=canonical_compact_json)}
    if isinstance(value, type):
        return {"kind": "type", "name": _fqname(value)}
    if isinstance(value, types.ForwardRef) if hasattr(types, "ForwardRef") else False:
        return {"kind": "forward-reference", "value": str(value)}
    origin = get_origin(value)
    if origin is not None:
        return {
            "kind": "typing",
            "origin": _stable_atom(origin),
            "arguments": [_stable_atom(item) for item in get_args(value)],
        }
    module = type(value).__module__
    qualname = type(value).__qualname__
    name = getattr(value, "__name__", None)
    normalized: dict[str, Any] = {
        "kind": "opaque",
        "type": f"{module}.{qualname}",
    }
    if isinstance(name, str):
        normalized["name"] = name
    return normalized


def normalize_signature(value: Any) -> dict[str, Any]:
    """Normalize a callable signature structurally, or record error class only."""

    try:
        signature = inspect.signature(value, eval_str=False)
    except (TypeError, ValueError) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "parameters": [
            {
                "annotation": _stable_atom(parameter.annotation),
                "default": _stable_atom(parameter.default),
                "kind": parameter.kind.name,
                "name": parameter.name,
            }
            for parameter in signature.parameters.values()
        ],
        "return_annotation": _stable_atom(signature.return_annotation),
    }


def _descriptor_slots(raw: Any) -> list[str]:
    descriptor_mro = inspect.getmro(type(raw))
    slots = []
    for label, special in (
        ("get", "__get__"),
        ("set", "__set__"),
        ("delete", "__delete__"),
    ):
        if any(special in vars(owner) for owner in descriptor_mro):
            slots.append(label)
    return slots


def raw_binding_kind(raw: Any) -> str | None:
    """Classify a raw ``vars(owner)[name]`` binding without invoking it."""

    raw_type = type(raw)
    if any(
        owner.__name__ == "UnifiedReadWriteMethod" for owner in inspect.getmro(raw_type)
    ):
        return "unified-read-write"
    if isinstance(raw, classmethod):
        return "classmethod"
    if isinstance(raw, staticmethod):
        return "staticmethod"
    if isinstance(raw, property):
        return "property"
    if isinstance(raw, (types.FunctionType, types.BuiltinFunctionType)):
        return "function"
    if inspect.isclass(raw):
        return None
    if _descriptor_slots(raw):
        return "generic-descriptor"
    if callable(raw):
        return "callable"
    return None


def _callable_descriptor(raw: Any, kind: str) -> Any:
    if kind in {"classmethod", "staticmethod"}:
        return raw.__func__
    return raw


def _package_relative_source(path: str | None, *, gwpy_owned: bool) -> str | None:
    if not path or not gwpy_owned:
        return None
    parts = Path(path).parts
    try:
        index = parts.index("gwpy")
    except ValueError:
        return None
    return Path(*parts[index:]).as_posix()


def _source_reference(value: Any, repository: Path) -> dict[str, Any] | None:
    try:
        path_text = inspect.getsourcefile(value) or inspect.getfile(value)
        _, line = inspect.getsourcelines(value)
    except (OSError, TypeError):
        return None
    path = Path(path_text).resolve()
    try:
        relative = path.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return None
    return {"path": relative, "line": line}


def _descriptor_projection(raw: Any, kind: str, repository: Path) -> dict[str, Any]:
    if kind in {"property", "unified-read-write"}:
        accessors = []
        details: dict[str, Any] = {}
        for label, attribute in (("get", "fget"), ("set", "fset"), ("delete", "fdel")):
            accessor = vars(type(raw)).get(attribute, None)
            # property stores these on the instance; reading the C-level slot does
            # not execute the public member descriptor.
            try:
                accessor = object.__getattribute__(raw, attribute)
            except AttributeError:
                accessor = None
            if accessor is not None:
                accessors.append(label)
                details[label] = {
                    "signature": normalize_signature(accessor),
                    "source": _source_reference(accessor, repository),
                }
        return {"accessors": accessors, "details": details}
    if kind == "generic-descriptor":
        return {"accessors": _descriptor_slots(raw), "details": {}}
    target = _callable_descriptor(raw, kind)
    return {
        "accessors": [],
        "details": {
            "call": {
                "signature": normalize_signature(target),
                "source": _source_reference(target, repository),
            }
        },
    }


def _literal_all_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for statement in tree.body:
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__all__"
        ):
            value = statement.value
        if isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            for item in value.elts
        ):
            names.extend(item.value for item in value.elts)
    return tuple(sorted(set(names), key=lambda item: item.encode("utf-8")))


def _module_name(source: Path, package_root: Path) -> str:
    relative = source.relative_to(package_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("gwexpy", *parts))


def _public_root_allowed(value: type[Any]) -> bool:
    name = value.__name__
    module = value.__module__
    if not module.startswith("gwexpy.") or name.startswith("_"):
        return False
    if any(token in name for token in INTERNAL_CLASS_TOKENS):
        return False
    if ".gui" in module or ".io." in module:
        return False
    return any(
        base.__module__.startswith("gwpy.") for base in inspect.getmro(value)[1:]
    )


def _select_unique_lazy_class(
    export_name: str, candidates: Sequence[type[Any]]
) -> type[Any]:
    """Select one static canonical-name route, failing closed otherwise."""

    unique: list[type[Any]] = []
    for candidate in candidates:
        if candidate.__name__ != export_name:
            continue
        if not any(candidate is existing for existing in unique):
            unique.append(candidate)
    if not unique:
        raise InventoryError(f"missing lazy class export route: {export_name}")
    if len(unique) != 1:
        routes = ", ".join(
            sorted((_fqname(item) for item in unique), key=lambda item: item.encode())
        )
        raise InventoryError(
            f"ambiguous lazy class export route: {export_name} ({routes})"
        )
    return unique[0]


def discover_public_classes(
    repository: Path,
) -> list[tuple[type[Any], tuple[str, ...]]]:
    """Discover canonical public roots from literal explicit exports only."""

    package_root = repository / "gwexpy"
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    exports: dict[type[Any], set[str]] = {}
    export_modules: list[tuple[str, tuple[str, ...]]] = []
    paths = sorted(
        package_root.rglob("*.py"),
        key=lambda item: item.relative_to(repository).as_posix().encode("utf-8"),
    )
    for source in paths:
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as exc:
            raise InventoryError(
                f"cannot scan {source.relative_to(repository)}: {type(exc).__name__}"
            ) from exc
        names = _literal_all_names(tree)
        if not names:
            continue
        module_name = _module_name(source, package_root)
        module_parts = module_name.split(".")
        if (
            "gui" in module_parts
            or "io" in module_parts
            or "__main__" in module_parts
            or any(part.startswith("_") for part in module_parts[1:])
        ):
            continue
        export_modules.append((module_name, names))

    snapshots: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
    classes_by_name: dict[str, list[type[Any]]] = {}
    missing = object()
    for module_name, names in export_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise InventoryError(
                f"cannot import explicit export module {module_name}: {type(exc).__name__}"
            ) from exc
        namespace = vars(module)
        snapshot = {name: namespace.get(name, missing) for name in names}
        snapshots.append((module_name, names, snapshot))
        for value in snapshot.values():
            if inspect.isclass(value):
                classes_by_name.setdefault(value.__name__, []).append(value)

    for module_name, names, snapshot in snapshots:
        for name in names:
            value = snapshot[name]
            if inspect.isclass(value):
                if _public_root_allowed(value):
                    exports.setdefault(value, set()).add(f"{module_name}:{name}")
                continue
            candidates = classes_by_name.get(name)
            if candidates is None:
                continue
            value = _select_unique_lazy_class(name, candidates)
            if _public_root_allowed(value):
                exports.setdefault(value, set()).add(f"{module_name}:{name}")
    return [
        (value, tuple(sorted(paths, key=lambda item: item.encode("utf-8"))))
        for value, paths in sorted(
            exports.items(), key=lambda item: _fqname(item[0]).encode("utf-8")
        )
    ]


def _raw_alias_identity(
    owner: type[Any], raw: Any, kind: str
) -> tuple[str, str | None]:
    aliases = sorted(
        name
        for name, candidate in vars(owner).items()
        if candidate is raw and (not name.startswith("_") or name in CONSTRUCTORS)
    )
    canonical = aliases[0] if aliases else None
    identity = f"{_fqname(owner)}::{canonical or kind}::{kind}"
    alias_group = identity if len(aliases) > 1 else None
    return identity, alias_group


def extract_members_for_classes(
    classes: Sequence[tuple[type[Any], tuple[str, ...]]], repository: Path
) -> list[dict[str, Any]]:
    """Extract effective GWexpy bindings before the first GWpy MRO class."""

    members: list[dict[str, Any]] = []
    for public_class, exports in classes:
        mro = inspect.getmro(public_class)
        try:
            boundary = next(
                index
                for index, owner in enumerate(mro)
                if owner.__module__.startswith("gwpy.")
            )
        except StopIteration as exc:
            raise InventoryError(
                f"public root has no GWpy MRO class: {_fqname(public_class)}"
            ) from exc
        prefix = mro[:boundary]
        candidate_names = sorted(
            {
                name
                for owner in prefix
                for name in vars(owner)
                if not name.startswith("_") or name in CONSTRUCTORS
            },
            key=lambda item: item.encode("utf-8"),
        )
        for name in candidate_names:
            # Resolve the first binding before deciding whether it is callable.
            # A subclass ``name = None`` intentionally masks a later mixin method.
            effective_owner = next(owner for owner in mro if name in vars(owner))
            raw = vars(effective_owner)[name]
            kind = raw_binding_kind(raw)
            if kind is None or not effective_owner.__module__.startswith("gwexpy."):
                continue
            raw_identity, alias_group = _raw_alias_identity(effective_owner, raw, kind)
            if effective_owner is public_class:
                resolution = "direct"
            elif "Mixin" in effective_owner.__name__:
                resolution = "inherited-mixin"
            else:
                resolution = "inherited-gwexpy-base"
            member_id = f"{_fqname(public_class)}/{name}"
            source_target = _callable_descriptor(raw, kind)
            if kind in {"property", "unified-read-write"}:
                source_target = object.__getattribute__(raw, "fget") or effective_owner
            members.append(
                {
                    "alias_group": alias_group,
                    "constructor": name in CONSTRUCTORS,
                    "counterpart_class": _fqname(mro[boundary]),
                    "descriptor": _descriptor_projection(raw, kind, repository),
                    "effective_owner": _fqname(effective_owner),
                    "exports": list(exports),
                    "kind": kind,
                    "member": name,
                    "member_id": member_id,
                    "mro_prefix": [_fqname(owner) for owner in prefix],
                    "public_class": _fqname(public_class),
                    "raw_descriptor_identity": raw_identity,
                    "resolution": resolution,
                    "source": _source_reference(source_target, repository),
                }
            )
    return sorted(members, key=lambda item: item["member_id"].encode("utf-8"))


def build_source_population(repository: Path) -> dict[str, Any]:
    classes = discover_public_classes(repository)
    roots = []
    for value, exports in classes:
        mro = inspect.getmro(value)
        counterpart = next(
            owner for owner in mro if owner.__module__.startswith("gwpy.")
        )
        roots.append(
            {
                "counterpart_class": _fqname(counterpart),
                "exports": list(exports),
                "public_class": _fqname(value),
                "source": _source_reference(value, repository),
            }
        )
    members = extract_members_for_classes(classes, repository)
    population = {"public_roots": roots, "members": members}
    population["digest"] = digest_json(population)
    return population


def _resolve_qualname(module_name: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        namespace = vars(value)
        if component not in namespace:
            raise InventoryError(f"cannot resolve {module_name}.{qualname}")
        value = namespace[component]
    return value


def _split_fqname(name: str) -> tuple[str, str]:
    components = name.split(".")
    for index in range(len(components) - 1, 0, -1):
        module_name = ".".join(components[:index])
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        return module_name, ".".join(components[index:])
    raise InventoryError(f"cannot split importable name: {name}")


def _oracle_source(raw: Any, provider: type[Any]) -> dict[str, Any] | None:
    """Keep source provenance only for exactly pinned GWpy-owned providers."""

    if not provider.__module__.startswith("gwpy."):
        return None
    kind = raw_binding_kind(raw)
    target = _callable_descriptor(raw, kind or "callable")
    if kind in {"property", "unified-read-write"}:
        try:
            target = object.__getattribute__(raw, "fget") or provider
        except AttributeError:
            target = provider
    try:
        path = inspect.getsourcefile(target) or inspect.getfile(target)
        _, line = inspect.getsourcelines(target)
    except (OSError, TypeError):
        return None
    relative = _package_relative_source(path, gwpy_owned=True)
    return {"path": relative, "line": line} if relative is not None else None


def _oracle_descriptor(raw: Any, kind: str, provider: type[Any]) -> dict[str, Any]:
    if kind in {"property", "unified-read-write"}:
        accessors = []
        details: dict[str, Any] = {}
        for label, attribute in (("get", "fget"), ("set", "fset"), ("delete", "fdel")):
            try:
                accessor = object.__getattribute__(raw, attribute)
            except AttributeError:
                accessor = None
            if accessor is None:
                continue
            accessors.append(label)
            details[label] = {
                "signature": normalize_signature(accessor),
                "source": _oracle_source(accessor, provider),
            }
        return {"accessors": accessors, "details": details}
    if kind == "generic-descriptor":
        return {"accessors": _descriptor_slots(raw), "details": {}}
    return {
        "accessors": [],
        "details": {
            "call": {
                "signature": normalize_signature(_callable_descriptor(raw, kind)),
                "source": _oracle_source(raw, provider),
            }
        },
    }


def _counterpart_raw_identity(
    provider: type[Any], member: str, raw: Any, kind: str
) -> str:
    aliases = sorted(
        name
        for name, candidate in vars(provider).items()
        if candidate is raw and (not name.startswith("_") or name in CONSTRUCTORS)
    )
    canonical = aliases[0] if aliases else member
    # Member is included deliberately: two public names must not share evidence
    # only because an upstream class happens to alias the same descriptor object.
    return f"{_fqname(provider)}::{member}::{canonical}::{kind}"


def build_oracle_projection(
    expected_version: str, queries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Resolve pristine GWpy counterparts without importing GWexpy."""

    if any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules):
        raise InventoryError("oracle worker started with gwexpy imported")
    actual_version = distribution_version("gwpy")
    if actual_version != expected_version:
        raise InventoryError(
            f"oracle version mismatch: expected {expected_version}, got {actual_version}"
        )
    members = []
    for query in sorted(
        queries, key=lambda item: str(item["member_id"]).encode("utf-8")
    ):
        counterpart_name = str(query["counterpart_class"])
        module_name, qualname = _split_fqname(counterpart_name)
        counterpart = _resolve_qualname(module_name, qualname)
        member = str(query["member"])
        binding_owner = next(
            (owner for owner in inspect.getmro(counterpart) if member in vars(owner)),
            None,
        )
        raw = vars(binding_owner)[member] if binding_owner is not None else None
        kind = raw_binding_kind(raw) if binding_owner is not None else None
        provider = binding_owner if kind is not None else None
        result: dict[str, Any] = {
            "counterpart_class": counterpart_name,
            "descriptor": None,
            "kind": None,
            "member": member,
            "member_id": str(query["member_id"]),
            "present": provider is not None,
            "provider": None,
            "public_class": str(query["public_class"]),
            "raw_descriptor_identity": None,
            "source": None,
        }
        if provider is not None:
            assert kind is not None
            result.update(
                {
                    "descriptor": _oracle_descriptor(raw, kind, provider),
                    "kind": kind,
                    "provider": _fqname(provider),
                    "raw_descriptor_identity": _counterpart_raw_identity(
                        provider, member, raw, kind
                    ),
                    "source": _oracle_source(raw, provider),
                }
            )
        members.append(result)
    if any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules):
        raise InventoryError("oracle worker imported gwexpy")
    body = {
        "gwpy_version": actual_version,
        "isolation": {
            "cwd_matches_expected": True,
            "gwexpy_absent_at_end": True,
            "gwexpy_absent_at_start": True,
            "isolated_flag": bool(sys.flags.isolated),
            "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "members": members,
        "schema": WORKER_SCHEMA,
    }
    body["digest"] = digest_json(body)
    return body


def _worker_main() -> int:
    try:
        payload = json.load(
            sys.stdin,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite_constant,
        )
        if not isinstance(payload, dict):
            raise InventoryError("oracle payload must be an object")
        if (
            set(payload)
            != {
                "expected_cwd",
                "expected_version",
                "queries",
                "schema",
            }
            or payload.get("schema") != WORKER_SCHEMA
        ):
            raise InventoryError("invalid oracle payload schema")
        expected_cwd = payload.get("expected_cwd")
        projection = build_oracle_projection(
            str(payload.get("expected_version")), payload.get("queries", [])
        )
        projection["isolation"]["cwd_matches_expected"] = (
            isinstance(expected_cwd, str) and os.getcwd() == expected_cwd
        )
        unsigned = {key: value for key, value in projection.items() if key != "digest"}
        projection["digest"] = digest_json(unsigned)
        sys.stdout.write(canonical_compact_json(projection) + "\n")
        return 0
    except (InventoryError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise InventoryError(f"non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot load manifest: {type(exc).__name__}") from exc
    if not isinstance(loaded, dict):
        raise InventoryError("manifest must be a JSON object")
    return loaded


def parse_oracle_arguments(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise InventoryError("oracle argument must be VERSION=EXECUTABLE")
        oracle_version, executable = value.split("=", 1)
        if oracle_version not in SUPPORTED_GWPY:
            raise InventoryError(f"unknown oracle version: {oracle_version}")
        if oracle_version in parsed:
            raise InventoryError(f"duplicate oracle version: {oracle_version}")
        if not executable:
            raise InventoryError(f"empty oracle executable for {oracle_version}")
        parsed[oracle_version] = (
            sys.executable if executable == "@current" else executable
        )
    if not parsed:
        raise InventoryError("at least one --oracle-python is required")
    return parsed


def run_pristine_oracle(
    script: Path,
    oracle_version: str,
    executable: str,
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    queries = [
        {
            "counterpart_class": member["counterpart_class"],
            "member": member["member"],
            "member_id": member["member_id"],
            "public_class": member["public_class"],
        }
        for member in members
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="gwexpy-gwpy-oracle-") as temporary:
        payload = {
            "expected_cwd": temporary,
            "expected_version": oracle_version,
            "queries": queries,
            "schema": WORKER_SCHEMA,
        }
        try:
            completed = subprocess.run(
                [executable, "-I", str(script.resolve()), "--oracle-worker"],
                cwd=temporary,
                env=environment,
                input=canonical_compact_json(payload),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise InventoryError(
                f"cannot execute oracle {oracle_version}: {type(exc).__name__}"
            ) from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"exit {completed.returncode}"
        raise InventoryError(message)
    try:
        projection = json.loads(
            completed.stdout,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise InventoryError("oracle stdout is not canonical JSON") from exc
    if completed.stdout != canonical_compact_json(projection) + "\n":
        raise InventoryError("oracle stdout is not canonical JSON")
    isolation = projection.get("isolation", {})
    if isolation != {
        "cwd_matches_expected": True,
        "gwexpy_absent_at_end": True,
        "gwexpy_absent_at_start": True,
        "isolated_flag": True,
        "no_user_site": True,
    }:
        raise InventoryError("oracle isolation contract failed")
    return projection


def _case_sort_key(case: Mapping[str, Any]) -> tuple[bytes, bytes, bytes, bytes]:
    return tuple(
        str(case[key]).encode("utf-8")
        for key in ("public_class", "member", "gwpy_version", "fixture")
    )  # type: ignore[return-value]


def _projection_without_digest(projection: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in projection.items() if key != "digest"}


def _implementation_groups(
    members: Sequence[Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], str | None]:
    by_member = {str(member["member_id"]): member for member in members}
    raw_keys: dict[tuple[str, str], tuple[str, str, str] | None] = {}
    for oracle_version, projection in projections.items():
        for observed in projection["members"]:
            key = (str(observed["member_id"]), oracle_version)
            if not observed["present"]:
                raw_keys[key] = None
                continue
            member = by_member[key[0]]
            raw_keys[key] = (
                str(member["raw_descriptor_identity"]),
                str(observed["raw_descriptor_identity"]),
                str(observed["provider"]),
            )
    canonical_keys = sorted(
        {key for key in raw_keys.values() if key is not None},
        key=canonical_compact_json,
    )
    labels = {
        key: f"implementation-{digest_json(list(key))[:16]}" for key in canonical_keys
    }
    return {
        member_version: labels.get(raw_key) if raw_key is not None else None
        for member_version, raw_key in raw_keys.items()
    }


def calculate_summary(
    cases: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate structural and workflow counts from the actual case states."""

    state_counts = {
        state: sum(case.get("state") == state for case in cases)
        for state in (*TERMINAL_STATES, *PROVISIONAL_STATES)
    }
    present_per_version = {
        oracle_version: sum(
            item.get("present") is True
            for item in projections[oracle_version]["members"]
        )
        for oracle_version in SUPPORTED_GWPY
    }
    absent_per_version = {
        oracle_version: len(members) - present_per_version[oracle_version]
        for oracle_version in SUPPORTED_GWPY
    }
    if len(set(present_per_version.values())) != 1:
        raise InventoryError("counterpart-present count differs by oracle version")
    if len(set(absent_per_version.values())) != 1:
        raise InventoryError("counterpart-absent count differs by oracle version")
    implementation_group_count = len(
        {
            case.get("implementation_group")
            for case in cases
            if case.get("implementation_group") is not None
        }
    )
    return {
        "cases": len(cases),
        "constructors": sum(member.get("constructor") is True for member in members),
        "counterpart_absent_per_version": next(iter(absent_per_version.values())),
        "counterpart_implementation_groups": implementation_group_count,
        "counterpart_present_per_version": next(iter(present_per_version.values())),
        "differential-required": state_counts["differential-required"],
        "fixed": state_counts["fixed"],
        "GWexpy-only": state_counts["GWexpy-only"],
        "GWpy-fails": state_counts["GWpy-fails"],
        "logical_members": len(members),
        "no-finding": state_counts["no-finding"],
        "public_roots": len({member.get("public_class") for member in members}),
        "unreviewed": state_counts["unreviewed"],
    }


def _manifest_policy() -> dict[str, Any]:
    return {
        "behavioral_owner": AUDIT_OWNER,
        "fixture_key": ["public_class", "member", "gwpy_version", "fixture"],
        "implementation_base": IMPLEMENTATION_BASE,
        "member_walk_rule": MEMBER_WALK_RULE,
        "oracle_versions": list(SUPPORTED_GWPY),
        "pristine_oracle_rule": PRISTINE_ORACLE_RULE,
        "provisional_states": list(PROVISIONAL_STATES),
        "public_root_rule": PUBLIC_ROOT_RULE,
        "terminal_states": list(TERMINAL_STATES),
        "upstream_dependency_provenance": UPSTREAM_DEPENDENCY_PROVENANCE,
    }


def build_manifest(
    population: Mapping[str, Any],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    members = list(population["members"])
    projection_maps = {
        oracle_version: {item["member_id"]: item for item in projection["members"]}
        for oracle_version, projection in projections.items()
    }
    groups = _implementation_groups(members, projections)
    cases = []
    for member in members:
        for oracle_version in SUPPORTED_GWPY:
            projection = projections[oracle_version]
            counterpart = projection_maps[oracle_version][member["member_id"]]
            if counterpart["present"]:
                fixture = PENDING_FIXTURE
                state = "differential-required"
                comparator = {"name": "pending"}
                observations = {
                    "gwexpy": {"outcome": "pending"},
                    "gwpy": {"outcome": "pending"},
                }
                owner: str | None = AUDIT_OWNER
            else:
                fixture = ABSENT_FIXTURE
                state = "GWexpy-only"
                comparator = {"name": "counterpart-absence"}
                observations = {
                    "gwexpy": {
                        "kind": member["kind"],
                        "outcome": "attribute-present",
                    },
                    "gwpy": {"outcome": "attribute-absent"},
                }
                owner = None
            case = {
                "case_key": "/".join(
                    (
                        str(member["public_class"]),
                        str(member["member"]),
                        oracle_version,
                        fixture,
                    )
                ),
                "comparator": comparator,
                "counterpart_present": bool(counterpart["present"]),
                "evidence": {
                    "behavior": [],
                    "oracle_projection_digest": projection["digest"],
                },
                "fixture": fixture,
                "gwpy_version": oracle_version,
                "implementation_group": groups[(member["member_id"], oracle_version)],
                "issues": ["#639"],
                "member": member["member"],
                "member_id": member["member_id"],
                "observations": observations,
                "owner": owner,
                "public_class": member["public_class"],
                "state": state,
            }
            cases.append(case)
    cases.sort(key=_case_sort_key)
    manifest = {
        "cases": cases,
        "members": members,
        "oracle_projections": {
            oracle_version: projections[oracle_version]
            for oracle_version in SUPPORTED_GWPY
        },
        "policy": _manifest_policy(),
        "population_digest": population["digest"],
        "public_roots": list(population["public_roots"]),
        "schema": SCHEMA,
        "summary": calculate_summary(cases, members, projections),
    }
    return manifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def _validate_projection(oracle_version: str, projection: Mapping[str, Any]) -> None:
    _require(projection.get("schema") == WORKER_SCHEMA, "invalid oracle schema")
    _require(
        projection.get("gwpy_version") == oracle_version,
        "oracle projection version mismatch",
    )
    expected_digest = digest_json(_projection_without_digest(projection))
    _require(
        projection.get("digest") == expected_digest, "oracle projection digest mismatch"
    )
    isolation = projection.get("isolation")
    _require(
        isolation
        == {
            "cwd_matches_expected": True,
            "gwexpy_absent_at_end": True,
            "gwexpy_absent_at_start": True,
            "isolated_flag": True,
            "no_user_site": True,
        },
        "invalid oracle isolation evidence",
    )
    members = projection.get("members")
    _require(isinstance(members, list), "oracle members must be a list")
    ids = [item.get("member_id") for item in members]
    _require(
        ids == sorted(ids, key=lambda item: str(item).encode("utf-8")),
        "oracle members are unsorted",
    )
    _require(len(ids) == len(set(ids)), "duplicate oracle member")
    for item in members:
        present = item.get("present")
        _require(isinstance(present, bool), "oracle presence must be boolean")
        if present:
            _require(
                item.get("provider") is not None, "present counterpart lacks provider"
            )
            _require(item.get("kind") is not None, "present counterpart lacks kind")
            _require(
                item.get("raw_descriptor_identity") is not None,
                "present counterpart lacks raw identity",
            )
            provider = str(item["provider"])
            if provider.startswith("gwpy."):
                source = item.get("source")
                _require(
                    source is None or not str(source.get("path", "")).startswith("/"),
                    "absolute oracle source path",
                )
            else:
                _require(
                    item.get("source") is None,
                    "non-GWpy provider source must be normalized away",
                )
        else:
            for key in (
                "provider",
                "kind",
                "raw_descriptor_identity",
                "source",
                "descriptor",
            ):
                _require(
                    item.get(key) is None,
                    "absent counterpart contains provider evidence",
                )


def _validate_case(
    case: Mapping[str, Any],
    members: Mapping[str, Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
    projection_members: Mapping[str, Mapping[str, Mapping[str, Any]]],
    implementation_groups: Mapping[tuple[str, str], str | None],
) -> None:
    member_id = str(case.get("member_id"))
    _require(member_id in members, "orphan case member reference")
    member = members[member_id]
    oracle_version = str(case.get("gwpy_version"))
    _require(oracle_version in projections, "orphan case oracle reference")
    observed = projection_members[oracle_version][member_id]
    _require(
        case.get("public_class") == member["public_class"], "case public class mismatch"
    )
    _require(case.get("member") == member["member"], "case member mismatch")
    _require(
        case.get("counterpart_present") is observed["present"], "case presence mismatch"
    )
    _require(
        case.get("implementation_group")
        == implementation_groups[(member_id, oracle_version)],
        "case implementation group mismatch",
    )
    fixture = str(case.get("fixture"))
    expected_key = "/".join(
        (str(member["public_class"]), str(member["member"]), oracle_version, fixture)
    )
    _require(case.get("case_key") == expected_key, "case key mismatch")
    evidence = case.get("evidence")
    _require(isinstance(evidence, dict), "case evidence must be an object")
    _require(
        evidence.get("oracle_projection_digest")
        == projections[oracle_version]["digest"],
        "case oracle digest mismatch",
    )
    behavior = evidence.get("behavior")
    _require(isinstance(behavior, list), "behavior evidence must be a list")
    state = str(case.get("state"))
    _require(state in {*TERMINAL_STATES, *PROVISIONAL_STATES}, "unknown case state")
    if state == "GWexpy-only":
        _require(observed["present"] is False, "GWexpy-only counterpart is present")
        _require(fixture == ABSENT_FIXTURE, "GWexpy-only fixture mismatch")
        _require(
            case.get("comparator") == {"name": "counterpart-absence"},
            "GWexpy-only comparator mismatch",
        )
        _require(
            case.get("owner") is None, "GWexpy-only must not have behavioral owner"
        )
        _require(
            case.get("implementation_group") is None,
            "GWexpy-only cannot have implementation group",
        )
        _require(
            case.get("observations")
            == {
                "gwexpy": {"kind": member["kind"], "outcome": "attribute-present"},
                "gwpy": {"outcome": "attribute-absent"},
            },
            "GWexpy-only observation mismatch",
        )
        _require(case.get("issues") == ["#639"], "GWexpy-only issue mismatch")
        _require(
            evidence
            == {
                "behavior": [],
                "oracle_projection_digest": projections[oracle_version]["digest"],
            },
            "GWexpy-only evidence schema mismatch",
        )
    elif state == "differential-required":
        _require(observed["present"] is True, "pending differential lacks counterpart")
        _require(fixture == PENDING_FIXTURE, "pending differential fixture mismatch")
        _require(
            case.get("owner") == AUDIT_OWNER, "pending differential owner mismatch"
        )
        _require(case.get("issues") == ["#639"], "pending differential issue mismatch")
        _require(
            case.get("comparator") == {"name": "pending"},
            "pending differential comparator mismatch",
        )
        _require(
            case.get("observations")
            == {
                "gwexpy": {"outcome": "pending"},
                "gwpy": {"outcome": "pending"},
            },
            "pending differential observation mismatch",
        )
        _require(
            evidence
            == {
                "behavior": [],
                "oracle_projection_digest": projections[oracle_version]["digest"],
            },
            "pending differential evidence schema mismatch",
        )
        _require(
            case.get("implementation_group") is not None,
            "pending differential lacks implementation group",
        )
    elif state == "unreviewed":
        _require(case.get("owner") is not None, "unreviewed case lacks owner")
    elif state in BEHAVIORAL_TERMINAL_STATES:
        _require(observed["present"] is True, "behavioral terminal lacks counterpart")
        _require(
            fixture not in {ABSENT_FIXTURE, PENDING_FIXTURE},
            "behavioral terminal has reserved fixture",
        )
        _require(bool(behavior), "behavioral terminal lacks differential evidence")
        _require(case.get("owner") is not None, "behavioral terminal lacks owner")
        _require(
            case.get("comparator", {}).get("name") not in {None, "pending"},
            "behavioral terminal lacks comparator",
        )
        if state == "fixed":
            issues = case.get("issues")
            _require(
                isinstance(issues, list)
                and "#639" in issues
                and any(
                    isinstance(issue, str) and bool(issue.strip()) and issue != "#639"
                    for issue in issues
                ),
                "fixed case requires a specific issue reference beyond #639",
            )
            _require(
                bool(evidence.get("pre_fix_mismatch")),
                "fixed case lacks pre-fix mismatch",
            )
            _require(bool(evidence.get("green_test")), "fixed case lacks green test")
        if state == "GWpy-fails":
            gwpy_observation = case.get("observations", {}).get("gwpy", {})
            _require(
                gwpy_observation.get("outcome") == "exception",
                "GWpy-fails lacks exception outcome",
            )
            _require(
                bool(gwpy_observation.get("exception_class")),
                "GWpy-fails lacks exception class",
            )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate schema, references, ordering, digests, and state evidence."""

    _require(
        set(manifest)
        == {
            "cases",
            "members",
            "oracle_projections",
            "policy",
            "population_digest",
            "public_roots",
            "schema",
            "summary",
        },
        "inventory top-level schema mismatch",
    )
    _require(manifest.get("schema") == SCHEMA, "invalid inventory schema")
    policy = manifest.get("policy")
    _require(isinstance(policy, dict), "inventory policy must be an object")
    _require(set(policy) == set(_manifest_policy()), "policy schema mismatch")
    _require(
        policy.get("behavioral_owner") == AUDIT_OWNER,
        "behavioral owner policy mismatch",
    )
    _require(
        policy.get("upstream_dependency_provenance") == UPSTREAM_DEPENDENCY_PROVENANCE,
        "upstream dependency provenance policy mismatch",
    )
    _require(
        policy.get("implementation_base") == IMPLEMENTATION_BASE,
        "implementation base mismatch",
    )
    _require(
        policy.get("public_root_rule") == PUBLIC_ROOT_RULE, "public root rule mismatch"
    )
    _require(
        policy.get("member_walk_rule") == MEMBER_WALK_RULE, "member walk rule mismatch"
    )
    _require(
        policy.get("pristine_oracle_rule") == PRISTINE_ORACLE_RULE,
        "pristine oracle rule mismatch",
    )
    _require(
        policy.get("oracle_versions") == list(SUPPORTED_GWPY),
        "invalid oracle version policy",
    )
    _require(
        policy.get("terminal_states") == list(TERMINAL_STATES)
        and policy.get("provisional_states") == list(PROVISIONAL_STATES),
        "invalid state policy",
    )
    _require(
        policy.get("fixture_key")
        == ["public_class", "member", "gwpy_version", "fixture"],
        "invalid stable case key policy",
    )
    projections = manifest.get("oracle_projections")
    _require(isinstance(projections, dict), "oracle projections must be an object")
    _require(set(projections) == set(SUPPORTED_GWPY), "oracle projection set mismatch")
    for oracle_version in SUPPORTED_GWPY:
        _validate_projection(oracle_version, projections[oracle_version])
    roots = manifest.get("public_roots")
    members_list = manifest.get("members")
    cases = manifest.get("cases")
    _require(isinstance(roots, list), "public roots must be a list")
    _require(isinstance(members_list, list), "members must be a list")
    _require(isinstance(cases, list), "cases must be a list")
    root_ids = [item.get("public_class") for item in roots]
    _require(
        root_ids == sorted(root_ids, key=lambda item: str(item).encode("utf-8")),
        "public roots are unsorted",
    )
    _require(len(root_ids) == len(set(root_ids)), "duplicate public root")
    member_ids = [item.get("member_id") for item in members_list]
    _require(
        member_ids == sorted(member_ids, key=lambda item: str(item).encode("utf-8")),
        "members are unsorted",
    )
    _require(len(member_ids) == len(set(member_ids)), "duplicate member")
    members = {str(item["member_id"]): item for item in members_list}
    roots_by_id = {str(item["public_class"]): item for item in roots}
    for member in members_list:
        public_class = str(member.get("public_class"))
        _require(public_class in roots_by_id, "orphan member public root reference")
        _require(
            member.get("exports") == roots_by_id[public_class].get("exports"),
            "member export alias reference mismatch",
        )
        _require(
            member.get("counterpart_class")
            == roots_by_id[public_class].get("counterpart_class"),
            "member counterpart class mismatch",
        )
    projection_members = {
        oracle_version: {
            str(item["member_id"]): item
            for item in projections[oracle_version]["members"]
        }
        for oracle_version in SUPPORTED_GWPY
    }
    for oracle_version in SUPPORTED_GWPY:
        _require(
            set(projection_members[oracle_version]) == set(members),
            "oracle/member population mismatch",
        )
        for member_id, observed in projection_members[oracle_version].items():
            member = members[member_id]
            _require(
                observed.get("public_class") == member.get("public_class")
                and observed.get("member") == member.get("member")
                and observed.get("counterpart_class")
                == member.get("counterpart_class"),
                "oracle/source member reference mismatch",
            )
    implementation_groups = _implementation_groups(members_list, projections)
    _require(cases == sorted(cases, key=_case_sort_key), "cases are unsorted")
    case_keys = [case.get("case_key") for case in cases]
    _require(len(case_keys) == len(set(case_keys)), "duplicate case key")
    for case in cases:
        _validate_case(
            case,
            members,
            projections,
            projection_members,
            implementation_groups,
        )
    expected_pairs = {
        (member_id, oracle_version)
        for member_id in members
        for oracle_version in SUPPORTED_GWPY
    }
    actual_pairs = {
        (str(case["member_id"]), str(case["gwpy_version"])) for case in cases
    }
    _require(actual_pairs == expected_pairs, "missing or orphan version case")
    expected_population_digest = digest_json(
        {"public_roots": roots, "members": members_list}
    )
    _require(
        manifest.get("population_digest") == expected_population_digest,
        "population digest mismatch",
    )
    expected_summary = calculate_summary(cases, members_list, projections)
    _require(manifest.get("summary") == expected_summary, "summary mismatch")
    _require(
        not any(
            isinstance(value, str) and (value.startswith("/") or "0x" in value.lower())
            for value in walk_manifest_values(manifest)
        ),
        "manifest contains absolute path or address-bearing repr",
    )


def validate_source_population(
    manifest: Mapping[str, Any], population: Mapping[str, Any]
) -> None:
    _require(
        manifest.get("public_roots") == population.get("public_roots"),
        "public root population drift",
    )
    _require(
        manifest.get("members") == population.get("members"), "member population drift"
    )
    _require(
        manifest.get("population_digest") == population.get("digest"),
        "source/MRO digest drift",
    )


def _refuse_behavioral_overwrite(path: Path) -> None:
    if not path.exists():
        return
    existing = load_json_strict(path)
    cases = existing.get("cases")
    if not isinstance(cases, list):
        raise InventoryError("refusing to overwrite malformed existing manifest")
    preserved = [
        case for case in cases if case.get("state") in BEHAVIORAL_TERMINAL_STATES
    ]
    if preserved:
        raise InventoryError(
            "refusing to overwrite existing fixed/no-finding/GWpy-fails behavioral evidence"
        )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--oracle-worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--oracle-python", action="append", default=[])
    parser.add_argument("--require-terminal", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.oracle_worker:
        if args.manifest is not None or args.oracle_python or args.require_terminal:
            print("oracle worker accepts stdin only", file=sys.stderr)
            return 2
        return _worker_main()
    try:
        if args.manifest is None:
            raise InventoryError("--manifest is required")
        if args.require_terminal and not args.check:
            raise InventoryError("--require-terminal is check-only")
        oracles = parse_oracle_arguments(args.oracle_python)
        if args.write and set(oracles) != set(SUPPORTED_GWPY):
            raise InventoryError("--write requires exactly GWpy 4.0.1 and 4.0.2")
        repository = Path(__file__).resolve().parents[1]
        population = build_source_population(repository)
        current_projections = {
            oracle_version: run_pristine_oracle(
                Path(__file__), oracle_version, executable, population["members"]
            )
            for oracle_version, executable in sorted(oracles.items())
        }
        if args.write:
            _refuse_behavioral_overwrite(args.manifest)
            manifest = build_manifest(population, current_projections)
            validate_manifest(manifest)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                canonical_manifest_json(manifest), encoding="utf-8"
            )
            print(f"wrote {args.manifest}")
            return 0
        manifest = load_json_strict(args.manifest)
        if args.manifest.read_text(encoding="utf-8") != canonical_manifest_json(
            manifest
        ):
            raise InventoryError("manifest serialization is not canonical JSON")
        validate_manifest(manifest)
        validate_source_population(manifest, population)
        for oracle_version, projection in current_projections.items():
            if manifest["oracle_projections"][oracle_version] != projection:
                raise InventoryError(
                    f"oracle projection drift for GWpy {oracle_version}"
                )
        if args.require_terminal:
            counts = {
                state: sum(case["state"] == state for case in manifest["cases"])
                for state in PROVISIONAL_STATES
            }
            if any(counts.values()):
                raise InventoryError(
                    "provisional states remain: "
                    f"differential-required={counts['differential-required']}, "
                    f"unreviewed={counts['unreviewed']}"
                )
        print(
            "inventory check passed: "
            f"members={len(manifest['members'])}, "
            f"gwpy={','.join(sorted(current_projections))}"
        )
        return 0
    except InventoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
