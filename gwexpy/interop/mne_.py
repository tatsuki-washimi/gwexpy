from __future__ import annotations

import errno
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index
from types import MethodType
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from astropy import units as u
from gwpy.time import LIGOTimeGPS

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional
from ._time import datetime_utc_to_gps, gps_to_datetime_utc

__all__ = [
    "to_mne_rawarray",
    "from_mne_raw",
    "to_mne",
    "from_mne",
]

# meas_date is a Python datetime (microsecond resolution); allow ~1us of
# round-trip slack when comparing it against a TimeSeries epoch (GPS seconds).
_MEAS_DATE_TOLERANCE_S = 1e-6
_GWEX_T0_GPS_NS_ATTR = "_gwex_t0_gps_ns"
_GWEX_CHANNEL_T0_GPS_NS_ATTR = "_gwex_channel_t0_gps_ns"
_GWEX_CHANNEL_DT_GPS_NS_ATTR = "_gwex_channel_dt_gps_ns"
_GWEX_MEAS_DATE_ATTR = "_gwex_exact_meas_date"

if TYPE_CHECKING:

    class _TimeSeriesDictLike(Protocol):
        def __setitem__(self, key: Any, value: Any) -> None: ...
        def to_matrix(self) -> Any: ...


def _infer_sfreq_hz(ts: Any) -> float:
    # 1. Try sample_rate / dt
    sample_rate = getattr(ts, "sample_rate", None)
    if sample_rate is not None:
        try:
            return float(sample_rate.to("Hz").value)
        except (AttributeError, TypeError):
            return float(getattr(sample_rate, "value", sample_rate))

    dt = getattr(ts, "dt", None)
    if dt is not None:
        try:
            dt_s = float(dt.to("s").value)
        except (AttributeError, TypeError):
            dt_s = float(getattr(dt, "value", dt))
        if dt_s == 0:
            # Maybe infinite sampling rate or something?
            # Or just bad metadata.
            pass
        else:
            return 1.0 / dt_s

    # 2. Try to infer from frequencies (for FrequencySeries)
    freqs = getattr(ts, "frequencies", None)
    if freqs is not None:
        # Assuming baseband: sfreq = 2 * max_freq
        # Or if available, df
        # frequencies.value usually array
        try:
            f_arr = freqs.value if hasattr(freqs, "value") else freqs
            if len(f_arr) > 0:
                # Use Nyquist assumption?
                # FrequencySeries usually goes up to Nyquist = sfreq / 2
                return float(f_arr[-1]) * 2.0
        except (TypeError, ValueError, AttributeError):
            pass

    # 3. Try to infer from times (for Spectrogram if dt is missing but times is array)
    times = getattr(ts, "times", None)
    if times is not None:
        try:
            t_arr = times.value if hasattr(times, "value") else times
            if len(t_arr) > 1:
                return 1.0 / float(t_arr[1] - t_arr[0])
        except (TypeError, ValueError, AttributeError, IndexError):
            pass

    # Fallback to None if not strict? MNE requires sfreq.
    # Raise error if we can't find it.
    raise ValueError(
        "Cannot infer sampling frequency (missing sample_rate/dt/frequencies/times)"
    )


def _default_ch_name(ts: Any, *, fallback: str) -> str:
    # ... existing implementation ...
    name = getattr(ts, "name", None)
    if isinstance(name, str) and name:
        return name
    channel = getattr(ts, "channel", None)
    if channel is not None:
        return str(channel)
    return fallback


def _t0_seconds(ts: Any) -> float:
    """Return a TimeSeries-like object's epoch as GPS seconds (float), or 0.0."""
    exact_ns = getattr(ts, "t0_gps_ns", None)
    if exact_ns is not None:
        return index(exact_ns) / 1e9
    t0 = getattr(ts, "t0", None)
    if t0 is None:
        return 0.0
    value = t0.value if hasattr(t0, "value") else t0
    return float(value)


def _t0_ns(ts: Any) -> int:
    """Exact-comparison representation of ``ts.t0``, as integer nanoseconds.

    Used for channel-to-channel epoch comparisons: converting through
    ``LIGOTimeGPS`` and comparing integer nanoseconds avoids any tolerance
    proportional to ``dt`` (see #493 -- a ``dt``-scaled tolerance let
    differently-timed channels silently stack together).
    """
    exact_ns = getattr(ts, "_gwex_t0_gps_ns", None)
    if exact_ns is None:
        exact_ns = getattr(ts, "t0_gps_ns", None)
    if exact_ns is not None:
        return index(exact_ns)
    return LIGOTimeGPS(_t0_seconds(ts)).ns()


def _strict_exact_ns(value: Any) -> int:
    """Validate metadata as an integer GPS-nanosecond value."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("exact GPS metadata must be an integer nanosecond value")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(
            "exact GPS metadata must be an integer nanosecond value"
        ) from exc


def _exact_authority(ts: Any) -> int | None:
    value = getattr(ts, "_gwex_t0_gps_ns", None)
    return None if value is None else _strict_exact_ns(value)


def _exact_dt_authority(ts: Any) -> int | None:
    value = getattr(ts, "_gwex_dt_gps_ns", None)
    return None if value is None else _strict_exact_ns(value)


def _raw_exact_epochs(raw: Any) -> dict[str, int]:
    """Return exact per-channel epochs after validating private metadata."""
    if hasattr(raw, _GWEX_MEAS_DATE_ATTR) and getattr(
        raw, _GWEX_MEAS_DATE_ATTR
    ) != raw.info.get("meas_date"):
        raise ValueError("official meas_date conflicts with exact GPS metadata")

    raw_global = getattr(raw, _GWEX_T0_GPS_NS_ATTR, None)
    raw_mapping = getattr(raw, _GWEX_CHANNEL_T0_GPS_NS_ATTR, None)
    if raw_mapping is not None and not isinstance(raw_mapping, Mapping):
        raise TypeError(
            "exact GPS metadata must map channel names to integer nanoseconds"
        )
    epochs = (
        {str(name): _strict_exact_ns(value) for name, value in raw_mapping.items()}
        if raw_mapping is not None
        else {}
    )
    if raw_global is not None:
        global_ns = _strict_exact_ns(raw_global)
        if epochs and any(value != global_ns for value in epochs.values()):
            raise ValueError("conflicting exact GPS metadata")
        for name in raw.ch_names:
            epochs.setdefault(name, global_ns)
    return epochs


def _raw_exact_dt(raw: Any, name: str) -> int:
    mapping = getattr(raw, _GWEX_CHANNEL_DT_GPS_NS_ATTR, None)
    if not isinstance(mapping, Mapping) or name not in mapping:
        raise ValueError(
            "cannot preserve exact GPS metadata through an MNE sample offset "
            "without an integral source sample interval"
        )
    return _strict_exact_ns(mapping[name])


def _raw_channel_epoch(raw: Any, name: str) -> int | None:
    epochs = _raw_exact_epochs(raw)
    if name not in epochs:
        return None
    return epochs[name] + index(raw.first_samp) * _raw_exact_dt(raw, name)


def _raw_legacy_effective_epoch(
    raw: Any, exact_epochs: Mapping[str, int]
) -> float | None:
    """Return the shared official epoch when ``raw`` has a legacy channel."""
    if not any(name not in exact_epochs for name in raw.ch_names):
        return None
    meas_date = raw.info.get("meas_date")
    t0 = float(datetime_utc_to_gps(meas_date)) if meas_date else 0.0
    return t0 + index(raw.first_samp) / raw.info["sfreq"]


def _raw_exact_intervals(raw: Any, epochs: Mapping[str, int]) -> dict[str, int]:
    """Validate and return exact per-channel sample intervals for ``raw``."""
    mapping = getattr(raw, _GWEX_CHANNEL_DT_GPS_NS_ATTR, None)
    if epochs:
        if not isinstance(mapping, Mapping) or set(mapping) != set(epochs):
            raise ValueError(
                "exact GPS metadata must provide one sample interval per exact channel"
            )
        return {name: _strict_exact_ns(mapping[name]) for name in epochs}
    if mapping is not None and (not isinstance(mapping, Mapping) or mapping):
        raise ValueError("sample interval metadata requires exact GPS metadata")
    return {}


def _validate_exact_metadata_structure(raw: Any, epochs: Mapping[str, int]) -> None:
    """Reject exact metadata which cannot describe this Raw's channels."""
    unknown = set(epochs).difference(raw.ch_names)
    if unknown:
        raise ValueError("exact GPS metadata contains unknown channel names")


def _preflight_add_channels(raw: Any, add_list: Any) -> dict[str, Any]:
    """Validate and compute all metadata changes before MNE mutates ``raw``."""
    current = _raw_exact_epochs(raw)
    _validate_exact_metadata_structure(raw, current)
    current_dt = _raw_exact_intervals(raw, current)
    current_effective = {
        name: current[name] + index(raw.first_samp) * current_dt[name]
        for name in current
    }
    additions: dict[str, int] = {}
    additions_dt: dict[str, int] = {}
    additions_effective: dict[str, int] = {}
    receiver_legacy_epoch = _raw_legacy_effective_epoch(raw, current)
    legacy_effective_epochs: list[float] = []
    if receiver_legacy_epoch is not None:
        legacy_effective_epochs.append(receiver_legacy_epoch)

    names = set(raw.ch_names)
    for other in add_list:
        other_names = list(other.ch_names)
        if len(other_names) != len(set(other_names)) or names.intersection(other_names):
            raise ValueError("cannot add channels with duplicate channel names")
        names.update(other_names)

        other_epochs = _raw_exact_epochs(other)
        _validate_exact_metadata_structure(other, other_epochs)
        other_dt = _raw_exact_intervals(other, other_epochs)
        additions.update(other_epochs)
        additions_dt.update(other_dt)
        other_first_samp = index(other.first_samp)
        additions_effective.update(
            {
                name: other_epochs[name] + other_first_samp * other_dt[name]
                for name in other_epochs
            }
        )
        legacy_epoch = _raw_legacy_effective_epoch(other, other_epochs)
        if legacy_epoch is not None:
            legacy_effective_epochs.append(legacy_epoch)

    exact_values = set(current_effective.values()) | set(additions_effective.values())
    if len(exact_values) > 1:
        raise ValueError(
            "cannot add channels with mismatched exact GPS epochs "
            "(effective epoch mismatch)"
        )
    exact_intervals = set(current_dt.values()) | set(additions_dt.values())
    if len(exact_intervals) > 1:
        raise ValueError(
            "cannot add channels with mismatched exact GPS sample intervals"
        )
    if legacy_effective_epochs and any(
        abs(value - legacy_effective_epochs[0]) > _MEAS_DATE_TOLERANCE_S
        for value in legacy_effective_epochs[1:]
    ):
        raise ValueError("cannot add channels with mismatched effective legacy epochs")

    meas_date = None
    if legacy_effective_epochs and receiver_legacy_epoch is None:
        base_epoch = (
            legacy_effective_epochs[0] - index(raw.first_samp) / raw.info["sfreq"]
        )
        # Convert and validate with MNE before mutation.  In particular, this
        # may reject a UTC leap second, which Python's datetime cannot encode.
        meas_date = gps_to_datetime_utc(base_epoch)
        raw.info.copy().set_meas_date(meas_date)

    merged: dict[str, int] | None = None
    merged_dt: dict[str, int] | None = None
    global_epoch: int | None = None
    if current or additions:
        receiver_first_samp = index(raw.first_samp)
        normalized_additions = {
            name: epoch - receiver_first_samp * additions_dt[name]
            for name, epoch in additions_effective.items()
        }
        merged = {**current, **normalized_additions}
        merged_dt = {**current_dt, **additions_dt}
        # A mixed Raw deliberately has metadata only for its exact channels.
        # Every such channel must have exactly one normalized base and interval.
        if not set(merged).issubset(names) or set(merged_dt) != set(merged):
            raise ValueError("invalid prospective exact GPS metadata")
        if len(merged) == len(names) and len(set(merged.values())) == 1:
            global_epoch = next(iter(merged.values()))

    return {
        "epochs": merged,
        "intervals": merged_dt,
        "global_epoch": global_epoch,
        "meas_date": meas_date,
    }


def _install_prevalidated_add_channels_metadata(
    raw: Any, plan: Mapping[str, Any]
) -> None:
    """Install only values fully validated by :func:`_preflight_add_channels`."""
    meas_date = plan["meas_date"]
    if meas_date is not None:
        # ``Info.set_meas_date`` was run on a copy during preflight.  Avoid
        # another validating conversion after MNE has changed ``raw``: the
        # remaining update is assignment-only (and mirrors Raw.set_meas_date
        # for a non-None datetime).
        with raw.info._unlock():
            raw.info["meas_date"] = meas_date
        if hasattr(raw, "annotations"):
            raw.annotations._orig_time = meas_date
    epochs = plan["epochs"]
    if epochs is None:
        return
    setattr(raw, _GWEX_CHANNEL_T0_GPS_NS_ATTR, dict(epochs))
    setattr(raw, _GWEX_CHANNEL_DT_GPS_NS_ATTR, dict(plan["intervals"]))
    global_epoch = plan["global_epoch"]
    if global_epoch is None:
        raw.__dict__.pop(_GWEX_T0_GPS_NS_ATTR, None)
    else:
        setattr(raw, _GWEX_T0_GPS_NS_ATTR, global_epoch)
    setattr(raw, _GWEX_MEAS_DATE_ATTR, raw.info.get("meas_date"))


@dataclass
class _MemmapAddChannelsJournal:
    """The file state MNE 1.12's Raw.add_channels may resize in place.

    MNE's memmap implementation is private and platform-specific: it avoids
    resize on Darwin, resizes in place elsewhere, and can collide with duplicate
    mappings on Windows.  The journal therefore records enough to reopen only
    when file truncation cannot retain the original mapping identity.
    """

    data: np.memmap
    filename: str
    contents: bytes
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    offset: int
    writable: bool


@dataclass
class _RawAddChannelsJournal:
    """References and small mutable values written by MNE Raw.add_channels.

    MNE replaces ``_data``, ``info``, ``_cals``, and ``_read_picks``.  It also
    updates ``_orig_units`` in place.  Capturing only these fields preserves
    aliases and leaves arbitrary user attributes alone.  A memmap is the one
    exception: MNE resizes its backing file, so its contents are journaled.
    """

    data: Any
    info: Any
    cals: Any
    read_picks: Any
    orig_units: Any
    orig_units_contents: dict[Any, Any]
    annotations: Any
    annotation_orig_time: Any
    exact_metadata: dict[str, tuple[bool, Any]]
    memmap: _MemmapAddChannelsJournal | None


_EXACT_RAW_METADATA_ATTRS = (
    _GWEX_T0_GPS_NS_ATTR,
    _GWEX_CHANNEL_T0_GPS_NS_ATTR,
    _GWEX_CHANNEL_DT_GPS_NS_ATTR,
    _GWEX_MEAS_DATE_ATTR,
)


def _snapshot_raw_add_channels_state(raw: Any) -> _RawAddChannelsJournal:
    """Journal precisely the state MNE and the exact-metadata install mutate."""
    data = raw._data
    memmap = None
    if isinstance(data, np.memmap):
        data.flush()
        if data.filename is None:
            raise ValueError("cannot transactionally add channels to an unnamed memmap")
        filename = os.fspath(data.filename)
        with open(filename, "rb") as backing:
            contents = backing.read()
        memmap = _MemmapAddChannelsJournal(
            data=data,
            filename=filename,
            contents=contents,
            shape=tuple(data.shape),
            dtype=data.dtype,
            offset=data.offset,
            writable=bool(data.flags.writeable),
        )
    annotations = raw._annotations
    return _RawAddChannelsJournal(
        data=data,
        info=raw.info,
        cals=raw._cals,
        read_picks=raw._read_picks,
        orig_units=raw._orig_units,
        orig_units_contents=dict(raw._orig_units),
        annotations=annotations,
        annotation_orig_time=annotations._orig_time,
        exact_metadata={
            name: (name in raw.__dict__, raw.__dict__.get(name))
            for name in _EXACT_RAW_METADATA_ATTRS
        },
        memmap=memmap,
    )


def _read_memmap_backing(journal: _MemmapAddChannelsJournal) -> bytes:
    with open(journal.filename, "rb") as backing:
        return backing.read()


def _write_memmap_backing(journal: _MemmapAddChannelsJournal) -> None:
    with open(journal.filename, "r+b") as backing:
        backing.write(journal.contents)
        backing.truncate(len(journal.contents))


def _detach_memmap_mapping(data: np.memmap) -> None:
    """Close an MNE-created replacement mapping before truncating its file."""
    # ``_mmap`` is a NumPy runtime implementation detail not described by its
    # stubs; keep the type escape confined to this platform-bound helper.
    mapping = cast(Any, data)._mmap
    if mapping is not None and not mapping.closed:
        mapping.close()


_MEMMAP_MAPPING_WINERRORS = frozenset({32, 33, 1224})


def _mapping_blocks_memmap_restore(error: BaseException) -> bool:
    """Whether ``error`` specifically means an active mapping blocked restore.

    This is intentionally capability/error based rather than a platform-name
    check.  ``EBUSY`` covers POSIX active-map truncate failures; the Windows
    values are the documented sharing/lock violations.  NumPy can also expose
    a failed mmap capability as ``BufferError`` or a mapping-specific
    ``SystemError``.  Permission, I/O, and unrelated runtime errors must
    propagate into the transaction's ``BaseExceptionGroup`` untouched.
    """
    if isinstance(error, BufferError):
        return True
    if isinstance(error, OSError):
        return error.errno == errno.EBUSY or getattr(error, "winerror", None) in (
            _MEMMAP_MAPPING_WINERRORS
        )
    if isinstance(error, SystemError):
        detail = str(error).lower()
        return "mmap" in detail and any(
            marker in detail for marker in ("mapping", "resize", "truncate")
        )
    return False


def _reopen_memmap_backing(journal: _MemmapAddChannelsJournal) -> np.memmap:
    mode = "r+" if journal.writable else "r"
    # NumPy's overloads omit the dtype/offset/shape combination accepted by
    # the runtime constructor used for reopening an existing mapping.
    return cast(Any, np.memmap)(
        journal.filename,
        dtype=journal.dtype,
        mode=mode,
        offset=journal.offset,
        shape=journal.shape,
    )


def _restore_memmap_add_channels_state(
    raw: Any, journal: _MemmapAddChannelsJournal
) -> None:
    """Restore a changed memmap without relying on ``mmap.resize``.

    The replacement mapping is detached before any write/truncate.  Linux can
    generally truncate beneath the original mapping and retain aliases.  If a
    any active mapping blocks that operation, the original mapping is closed
    and reopened only as a last resort.  This follows MNE 1.12's private
    memmap behavior by capability/error rather than assuming a host OS.
    """
    current_data = raw.__dict__.get("_data")
    replacement = (
        current_data
        if isinstance(current_data, np.memmap) and current_data is not journal.data
        else None
    )
    if replacement is not None:
        _detach_memmap_mapping(replacement)
    if _read_memmap_backing(journal) == journal.contents:
        return
    try:
        _write_memmap_backing(journal)
    except (OSError, SystemError, BufferError) as error:
        if not _mapping_blocks_memmap_restore(error):
            raise
        _detach_memmap_mapping(journal.data)
        _write_memmap_backing(journal)
        journal.data = _reopen_memmap_backing(journal)


def _restore_raw_add_channels_state(raw: Any, state: _RawAddChannelsJournal) -> None:
    """Restore a Raw journal while continuing after individual restore failures.

    Direct instance-dictionary updates intentionally bypass ``Info._unlock``
    and ``Annotations.__setattr__``: either may be the operation that failed.
    The narrow ``__dict__`` dependency is preferable to MNE's private mutable
    update helpers and keeps the receiver object and its aliases intact.
    """
    errors: list[BaseException] = []

    def restore(action: Any) -> None:
        try:
            action()
        except BaseException as error:
            errors.append(error)

    if state.memmap is not None:
        restore(lambda: _restore_memmap_add_channels_state(raw, state.memmap))
    # A Windows-style duplicate mapping can force the memmap helper to close
    # the original mapping and reopen it.  Keep the original object whenever
    # possible, but install that only safe replacement when the platform made
    # preserving its identity impossible.
    restored_data = state.memmap.data if state.memmap is not None else state.data
    restore(lambda: raw.__dict__.__setitem__("_data", restored_data))
    restore(lambda: raw.__dict__.__setitem__("info", state.info))
    restore(lambda: raw.__dict__.__setitem__("_cals", state.cals))
    restore(lambda: raw.__dict__.__setitem__("_read_picks", state.read_picks))
    restore(lambda: raw.__dict__.__setitem__("_orig_units", state.orig_units))
    restore(lambda: state.orig_units.clear())
    restore(lambda: state.orig_units.update(state.orig_units_contents))
    restore(lambda: raw.__dict__.__setitem__("_annotations", state.annotations))
    restore(
        lambda: state.annotations.__dict__.__setitem__(
            "_orig_time", state.annotation_orig_time
        )
    )
    for name, (present, value) in state.exact_metadata.items():
        if present:
            restore(
                lambda name=name, value=value: raw.__dict__.__setitem__(name, value)
            )
        else:
            restore(lambda name=name: raw.__dict__.pop(name, None))
    restore(lambda: raw.__dict__.pop("add_channels", None))
    restore(lambda: raw.__dict__.pop("_gwex_exact_add_channels_guard", None))
    restore(lambda: _install_add_channels_guard(raw))
    if errors:
        raise BaseExceptionGroup("Raw.add_channels rollback failed", errors)


def _install_add_channels_guard(raw: Any) -> None:
    """Reject conflicting exact epochs before MNE mutates an in-memory Raw."""
    if getattr(raw, "_gwex_exact_add_channels_guard", False):
        return

    def guarded(self: Any, add_list: Any, *args: Any, **kwargs: Any) -> Any:
        plan = _preflight_add_channels(self, add_list)
        state = _snapshot_raw_add_channels_state(self)

        # Look up the class method at call time.  Copy/deepcopy can duplicate
        # this instance-bound guard, so closing over ``raw.add_channels`` would
        # instead mutate the original Raw object.
        try:
            result = type(self).add_channels(self, add_list, *args, **kwargs)
            _install_prevalidated_add_channels_metadata(self, plan)
        except BaseException as operation_error:
            try:
                _restore_raw_add_channels_state(self, state)
            except BaseException as restore_error:
                raise BaseExceptionGroup(
                    "Raw.add_channels failed and transaction rollback failed",
                    [operation_error, restore_error],
                ) from restore_error
            raise
        return result

    raw.add_channels = MethodType(guarded, raw)
    raw._gwex_exact_add_channels_guard = True


def _attach_exact_metadata(
    raw: Any, epochs: dict[str, int], dt_ns: dict[str, int]
) -> None:
    """Attach exact in-memory channel metadata and protect ``add_channels``."""
    if not epochs:
        return
    setattr(raw, _GWEX_CHANNEL_T0_GPS_NS_ATTR, dict(epochs))
    setattr(raw, _GWEX_CHANNEL_DT_GPS_NS_ATTR, dict(dt_ns))
    if len(epochs) == len(raw.ch_names) and len(set(epochs.values())) == 1:
        setattr(raw, _GWEX_T0_GPS_NS_ATTR, next(iter(epochs.values())))
    setattr(raw, _GWEX_MEAS_DATE_ATTR, raw.info.get("meas_date"))
    _install_add_channels_guard(raw)


def _apply_meas_date_contract(info: Any, t0_seconds: float) -> Any:
    """Reconcile an input epoch (GPS seconds) with ``info["meas_date"]``.

    Contract (#493): ``t0`` is authoritative when ``info`` has no
    ``meas_date`` yet (``t0 == 0`` leaves it unset, preserving the legacy
    default). Once ``info["meas_date"]`` is set, ``t0`` -- including ``0``,
    which is not treated as a special case -- is always compared against it;
    a match within ``_MEAS_DATE_TOLERANCE_S`` keeps the existing value, and a
    mismatch raises ``ValueError`` rather than silently overwriting it.

    Returns the ``info`` to use (a copy if a new ``meas_date`` was set, so
    the caller's original ``info`` object is never mutated).
    """
    existing = info.get("meas_date")
    if existing is None:
        if t0_seconds != 0.0:
            info = info.copy()
            info.set_meas_date(gps_to_datetime_utc(t0_seconds))
        return info

    existing_gps = float(datetime_utc_to_gps(existing))
    if abs(existing_gps - t0_seconds) > _MEAS_DATE_TOLERANCE_S:
        raise ValueError(
            f"info['meas_date'] ({existing!r}, GPS {existing_gps}) does not "
            f"match the input epoch (GPS {t0_seconds}); pass an info whose "
            "meas_date agrees with the data's t0, or omit meas_date"
        )
    return info


def _select_items(
    items: list[tuple[Any, Any]], picks: Any | None
) -> list[tuple[Any, Any]]:
    # ... existing implementation ...
    if picks is None:
        return items
    if isinstance(picks, (str, int)):
        picks = [picks]

    if not isinstance(picks, Sequence):
        raise TypeError("picks must be a sequence of channel names or indices")

    if all(isinstance(p, str) for p in picks):
        pick_set = set(picks)
        return [(k, v) for (k, v) in items if str(k) in pick_set]

    indices = [int(p) for p in picks]
    return [items[i] for i in indices]


def to_mne_rawarray(tsd, info=None, picks=None):
    """Convert a TimeSeries-like object to an ``mne.io.RawArray``.

    Parameters
    ----------
    tsd
        ``TimeSeriesDict``-like mapping (multi-channel) or a single ``TimeSeries``.
    info
        Optional MNE ``Info``. If omitted, a minimal ``Info`` is created.
    picks
        Optional channel selection (names or indices). Only supported for mapping inputs.

    Returns
    -------
    mne.io.RawArray
        The converted MNE Raw object.

    Raises
    ------
    TypeError
        If ``picks`` is given for a single-channel input.
    ValueError
        If ``info``'s channel count does not match the input; if mapping
        channels have mismatched sampling frequency, length, or (for
        same-length channels) epoch; or if ``t0`` conflicts with an
        existing ``info["meas_date"]``.
    LeapSecondConversionError
        If ``t0`` falls on a leap second.

    Notes
    -----
    The input epoch (``t0``) is reconciled with ``info["meas_date"]``: if
    ``info`` has no ``meas_date`` yet, it is set from ``t0`` (unless
    ``t0 == 0``, which leaves it unset); if ``info`` already has a
    ``meas_date``, ``t0`` -- including ``0`` -- is always compared against
    it, and a mismatch beyond ~1us raises ``ValueError`` instead of silently
    overwriting or ignoring it. A ``t0`` that falls on a leap second raises
    ``LeapSecondConversionError``.

    For a mapping input, all channels must share the same sampling
    frequency; a mismatch always raises ``ValueError`` (previously stacked
    silently), even when channel lengths differ. Same-length channels are
    then stacked without resampling/alignment and must also share an
    *exactly* matching epoch; a mismatch raises ``ValueError`` (previously
    stacked silently). Only channels of *differing length* (with matching
    sampling frequency) are automatically aligned, via ``to_matrix()`` on a
    ``TimeSeriesDict`` input -- sampling-frequency or epoch mismatches are
    never auto-aligned and must be resolved by the caller beforehand.

    A mixed exact/legacy mapping uses MNE's one shared official time axis for
    the legacy channels, while GWexpy retains exact epochs as private
    per-channel in-memory metadata. Exact channels must still agree exactly on
    epoch and sample interval; exact/exact conflicts are rejected.

    """
    mne = require_optional("mne")

    # Single-channel input
    if not isinstance(tsd, Mapping):
        if picks is not None:
            raise TypeError("picks is only supported for mapping inputs")

        from .base import to_plain_array

        data_1d = to_plain_array(tsd)
        if data_1d.ndim != 1:
            raise ValueError("Single-channel input must be 1D")

        ch_name = _default_ch_name(tsd, fallback="ch0")
        sfreq = _infer_sfreq_hz(tsd)

        if info is None:
            info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types=["misc"])
        elif int(info["nchan"]) != 1:
            raise ValueError(f"info expects nchan=1, got {info['nchan']}")

        exact_t0_ns = _t0_ns(tsd)
        info = _apply_meas_date_contract(info, exact_t0_ns / 1e9)

        raw = mne.io.RawArray(data_1d[None, :], info)
        source_exact_ns = _exact_authority(tsd)
        source_dt_ns = _exact_dt_authority(tsd)
        if source_exact_ns is not None and source_dt_ns is None:
            raise ValueError(
                "cannot preserve exact GPS metadata through an MNE sample offset "
                "without an integral source sample interval"
            )
        if source_exact_ns is not None:
            assert source_dt_ns is not None
            _attach_exact_metadata(
                raw, {ch_name: source_exact_ns}, {ch_name: source_dt_ns}
            )
        else:
            _install_add_channels_guard(raw)
        return raw

    # Multi-channel mapping input
    items = _select_items(list(tsd.items()), picks)
    if not items:
        raise ValueError("No channels selected")

    ch_names = [str(k) for (k, _) in items]
    series = [v for (_, v) in items]

    sfreq = _infer_sfreq_hz(series[0])
    for ts in series[1:]:
        if not np.isclose(_infer_sfreq_hz(ts), sfreq):
            raise ValueError("All channels must share the same sampling frequency")

    lengths = {len(ts) for ts in series}
    exact_epochs = {
        name: exact
        for name, ts in zip(ch_names, series, strict=True)
        if (exact := _exact_authority(ts)) is not None
    }
    exact_dt_ns = {
        name: exact
        for name, ts in zip(ch_names, series, strict=True)
        if (exact := _exact_dt_authority(ts)) is not None
    }
    if exact_epochs and set(exact_dt_ns) != set(exact_epochs):
        raise ValueError(
            "cannot preserve exact GPS metadata through an MNE sample offset "
            "without an integral source sample interval"
        )
    if len(set(exact_dt_ns.values())) > 1:
        raise ValueError(
            "All exact channels must share matching exact sample intervals"
        )

    legacy_series = [ts for ts in series if _exact_authority(ts) is None]
    if len(lengths) == 1:
        data = np.stack([np.asarray(ts.value) for ts in series], axis=0)
        # Same-length channels are stacked as-is (no alignment), so their
        # epochs must match exactly -- an exact ns comparison (not a
        # dt-scaled tolerance) so genuinely different acquisition times are
        # never silently stacked together (#493).
        t0_ns_values = {_t0_ns(ts) for ts in series if _exact_authority(ts) is not None}
        if len(t0_ns_values) > 1:
            raise ValueError(
                "All channels must share the same epoch (t0); found mismatched "
                "epochs across channels and no alignment was requested (use a "
                "TimeSeriesDict with to_matrix() for alignment instead)"
            )
        legacy_t0_ns_values = {_t0_ns(ts) for ts in legacy_series}
        if len(legacy_t0_ns_values) > 1:
            raise ValueError(
                "All legacy channels have a mismatched epoch (t0) and cannot "
                "share an MNE Raw time axis"
            )
        common_t0_ns = _t0_ns(legacy_series[0]) if legacy_series else _t0_ns(series[0])
    elif hasattr(tsd, "to_matrix"):
        try:
            if len(set(exact_epochs.values())) > 1:
                raise ValueError("exact channel epochs differ before alignment")
            tsd_sel = cast("_TimeSeriesDictLike", tsd.__class__())
            for k, ts in items:
                tsd_sel[k] = ts
            from .base import to_plain_array

            mat = tsd_sel.to_matrix()
            data = to_plain_array(mat)
            if data.ndim == 3:
                data = data[:, 0, :]
            if data.shape[0] != len(ch_names):
                raise ValueError("Unexpected channel dimension after alignment")
            ch_names = list(getattr(mat, "channel_names", ch_names))
            common_t0_ns = (
                _t0_ns(legacy_series[0])
                if legacy_series
                else next(iter(exact_epochs.values()))
                if exact_epochs
                else _t0_ns(mat)
            )
        except (ValueError, TypeError, AttributeError, IndexError, KeyError) as e:
            raise ValueError(
                "Channels have mismatched lengths and could not be aligned via to_matrix()"
            ) from e
    else:
        raise ValueError(
            "All channels must have the same length (or provide a TimeSeriesDict with to_matrix() for alignment)"
        )

    if info is None:
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=["misc"] * len(ch_names)
        )
    elif int(info["nchan"]) != len(ch_names):
        raise ValueError(f"info expects nchan={len(ch_names)}, got {info['nchan']}")

    info = _apply_meas_date_contract(info, common_t0_ns / 1e9)

    raw = mne.io.RawArray(data, info)
    if exact_epochs:
        _attach_exact_metadata(raw, exact_epochs, exact_dt_ns)
    else:
        _install_add_channels_guard(raw)
    return raw


def from_mne_raw(cls, raw, unit_map=None):
    """Create a `TimeSeriesDict` from `mne.io.Raw`.

    Parameters
    ----------
    cls
        The `TimeSeriesDict`-like class to construct and populate.
    raw : mne.io.Raw
        The MNE `Raw` object to convert.
    unit_map : dict, optional
        Optional mapping from channel name to unit, applied to the
        resulting `TimeSeries` entries. Channels absent from the mapping
        (or when ``unit_map`` is omitted) get ``unit=None``.

    Returns
    -------
    TimeSeriesDict
        A `cls` instance populated with one `TimeSeries` per channel.

    Notes
    -----
    The GPS epoch is reconstructed as
    ``datetime_utc_to_gps(raw.info["meas_date"]) + raw.first_samp / sfreq``
    (or just the ``first_samp`` offset if ``meas_date`` is unset), so
    cropped or resumed `Raw` objects (``first_samp > 0``) get the correct
    absolute epoch instead of always starting at ``meas_date`` (or ``0``).

    """
    data, times = raw.get_data(return_times=True)
    # data: (n_ch, n_times)
    # times: (n_times,) relative to the *returned* data, always starting at 0
    # -- it does NOT include raw.first_samp (verified against mne 1.12), so
    # the first_samp offset below is not double-counted.

    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]
    dt = 1.0 / sfreq

    t0 = 0.0
    if raw.info["meas_date"]:
        # meas_date is an aware-UTC datetime.
        t0 = float(datetime_utc_to_gps(raw.info["meas_date"]))
    t0 = t0 + index(raw.first_samp) / sfreq

    tsd = cls()
    for i, name in enumerate(ch_names):
        unit = unit_map.get(name) if unit_map else None
        exact_t0_ns = _raw_channel_epoch(raw, name)
        epoch_kwargs: dict[str, Any]
        if exact_t0_ns is not None:
            # GWpy's TimeSeriesDict.EntryClass is its base TimeSeries, which
            # cannot represent the exact authority.
            from gwexpy.timeseries import TimeSeries

            entry_class = TimeSeries
            epoch_kwargs = {"t0_ns": exact_t0_ns}
            entry_dt = _raw_exact_dt(raw, name) * u.ns
        else:
            entry_class = tsd.EntryClass
            epoch_kwargs = {"t0": t0}
            entry_dt = dt
        tsd[name] = entry_class(
            data[i], dt=entry_dt, name=name, unit=unit, **epoch_kwargs
        )

    return tsd


def to_mne(data, info=None, **kwargs):
    """Convert a gwexpy object to an MNE object.

    Parameters
    ----------
    data : FrequencySeries, Spectrogram, or TimeSeries (or dicts)
        The data object to convert.
    info : mne.Info, optional
        Measurement info to use. If None, one is created.
    **kwargs
        Additional arguments passed to MNE constructors.

    Returns
    -------
    mne_object
        The converted MNE object (e.g. RawArray, SpectrumArray, EpochsTFRArray).

    """
    require_optional("mne")

    # Check for Spectrogram (or dict) first because it has both time and freq
    is_spec = False
    if hasattr(data, "frequencies") and hasattr(data, "times"):
        is_spec = True
    elif isinstance(data, Mapping) and len(data) > 0:
        first = next(iter(data.values()))
        if hasattr(first, "frequencies") and hasattr(first, "times"):
            is_spec = True

    if is_spec:
        return _spec_to_mne_tfr(data, info, **kwargs)

    # Check for FrequencySeries (or dict)
    is_fs = False
    if hasattr(data, "frequencies"):  # Single FrequencySeries
        is_fs = True
    elif isinstance(data, Mapping) and len(data) > 0:
        first = next(iter(data.values()))
        if hasattr(first, "frequencies"):
            is_fs = True

    if is_fs:
        return _fs_to_mne_spectrum(data, info, **kwargs)

    # Default to RawArray (TimeSeries)
    return to_mne_rawarray(data, info, **kwargs)


def from_mne(cls, data, **kwargs):
    """Convert an MNE object to a gwexpy object.

    Parameters
    ----------
    cls : type
        The target class (e.g. FrequencySeries, Spectrogram, TimeSeries).
    data : mne object
        The MNE object to convert.
    **kwargs
        Additional arguments passed to from_mne_* helpers.

    Returns
    -------
    gwexpy object

    """
    require_optional("mne")

    # Spectrum -> FrequencySeries
    # Check if data is Spectrum (using string check to avoid direct import or try/except)
    if "Spectrum" in type(data).__name__:
        return _mne_spectrum_to_fs(cls, data, **kwargs)

    # TFR -> Spectrogram
    if "TFR" in type(data).__name__:
        return _mne_tfr_to_spec(cls, data, **kwargs)

    # Raw -> TimeSeries
    if "Raw" in type(data).__name__:
        return from_mne_raw(cls, data, **kwargs)

    raise TypeError(f"Unsupported MNE object type: {type(data)}")


def _fs_to_mne_spectrum(fsd, info=None, **kwargs):
    """Convert `FrequencySeries` data to `mne.time_frequency.SpectrumArray`."""
    mne = require_optional("mne")

    # Normalize input to list of (name, series)
    if isinstance(fsd, Mapping):
        items = list(fsd.items())
    elif hasattr(fsd, "name"):  # Single series
        name = _default_ch_name(fsd, fallback="ch0")
        items = [(name, fsd)]
    else:
        # Fallback using provided name or ch0 if everything fails
        items = [(_default_ch_name(fsd, fallback="ch0"), fsd)]

    if not items:
        raise ValueError("No data provided")

    # Extract data and frequencies
    # MNE SpectrumArray expects data shape (n_epochs, n_channels, n_freqs)
    # We treat single series as 1 epoch.

    first = items[0][1]
    freqs = first.frequencies.value

    data_list = []
    ch_names = []

    for name, fs in items:
        # Consistency check
        if not np.allclose(fs.frequencies.value, freqs):
            raise ValueError("All channels must have same frequencies")
        data_list.append(fs.value)
        ch_names.append(str(name))

    # Stack channels: (n_channels, n_freqs)
    data_2d = np.stack(data_list, axis=0)

    sfreq = _infer_sfreq_hz(first)

    if info is None:
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=["mag"] * len(ch_names)
        )

    # MNE >= 1.2 required for SpectrumArray
    # SpectrumArray in MNE is for static spectra (averaged or single trial), so (n_ch, n_freqs)
    # EpochsSpectrumArray would be (n_epochs, n_ch, n_freqs) but we stick to SpectrumArray for now.
    if not hasattr(mne.time_frequency, "SpectrumArray"):
        raise ImportError("mne.time_frequency.SpectrumArray requires MNE >= 1.2")

    return mne.time_frequency.SpectrumArray(data_2d, info, freqs, **kwargs)


def _mne_spectrum_to_fs(cls, spectrum, **kwargs):
    """Convert `mne.time_frequency.Spectrum` to `FrequencySeries` data."""
    data = spectrum.get_data()
    freqs = spectrum.freqs
    ch_names = spectrum.ch_names

    # Handle data shape (might be 2D or 3D)
    if data.ndim == 3:  # (n_epochs, n_channels, n_freqs)
        n_epochs, n_ch, n_freqs_dim = data.shape
        if n_epochs > 1:
            data = data.mean(axis=0)  # (n_ch, n_freqs)
        else:
            data = data[0]
    elif data.ndim == 2:  # (n_channels, n_freqs)
        n_ch, n_freqs_dim = data.shape
    else:
        raise ValueError(f"Unexpected spectrum data shape: {data.shape}")

    if n_ch == 1:
        # data[0] is (n_freqs,) array.
        val = data[0] if data.ndim == 2 else data
        return cls(val, frequencies=freqs, name=ch_names[0], **kwargs)

    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")

    fsd = FrequencySeriesDict()
    for i, name in enumerate(ch_names):
        fsd[name] = cls(data[i], frequencies=freqs, name=name, **kwargs)
    return fsd


def _spec_to_mne_tfr(specd, info=None, **kwargs):
    """Convert spectrogram data to `mne.time_frequency.EpochsTFRArray`."""
    mne = require_optional("mne")

    if isinstance(specd, Mapping):
        items = list(specd.items())
    elif hasattr(specd, "name"):
        name = _default_ch_name(specd, fallback="ch0")
        items = [(name, specd)]
    else:
        items = [(_default_ch_name(specd, fallback="ch0"), specd)]

    if not items:
        raise ValueError("No data provided")

    first = items[0][1]
    freqs = first.frequencies.value
    times = first.times.value  # relative time usually? Or GPS?
    # MNE times are usually relative to trigger.
    # If Spectrogram times are GPS, we might want to shift them or put t0 in info['meas_date']?
    # For TFRArray, tmin is optional arg (default times[0]).

    data_list = []
    ch_names = []

    for name, spec in items:
        # spec.value shape: (n_times, n_freqs) usually in gwexpy?
        # Wait, gwexpy Spectrogram is (times, frequencies) usually?
        # Check Spectrogram: it inherits from SeriesMatrix.
        # usually (n_times, n_freqs) or (n_freqs, n_times)?
        # Let's check docs or assume standard (times, freqs).
        # MNE expects (n_epochs, n_channels, n_freqs, n_times).

        # Spectrogram.value is likely (times, freqs) based on typical matrix orientation?
        # Wait, if `fs` is from `FrequencySeries`, it's 1D.
        # `Spectrogram` is 2D.
        # Let's verify shape.
        # Usually Spectrogram[time, freq].

        val = spec.value
        # If (times, freqs), we transpose to (freqs, times) for MNE.
        if val.shape == (len(times), len(freqs)):
            val = val.T

        data_list.append(val)
        ch_names.append(str(name))

    # Stack channels: (n_channels, n_freqs, n_times)
    data_3d = np.stack(data_list, axis=0)
    # Add epoch: (1, n_ch, n_fr, n_ti)
    data_4d = data_3d[None, :, :, :]

    sfreq = _infer_sfreq_hz(first)

    if info is None:
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=["misc"] * len(ch_names)
        )

    # MNE >= 1.3 required for EpochsTFRArray
    if not hasattr(mne.time_frequency, "EpochsTFRArray"):
        # Fallback to EpochsTFR if available (it might take different args)
        # Or error.
        # Actually EpochsTFR usually takes precomputed data in constructor in some versions?
        # But EpochsTFRArray is the consistent way for computed arrays.
        raise ImportError("mne.time_frequency.EpochsTFRArray requires MNE >= 1.3")

    return mne.time_frequency.EpochsTFRArray(info, data_4d, times, freqs, **kwargs)


def _mne_tfr_to_spec(cls, tfr, **kwargs):
    """Convert MNE TFR objects to spectrogram data."""
    data = tfr.data
    # Shape:
    # EpochsTFR: (n_epochs, n_channels, n_freqs, n_times)
    # AverageTFR: (n_channels, n_freqs, n_times)

    times = tfr.times
    freqs = tfr.freqs
    ch_names = tfr.ch_names

    # Handle epochs
    if data.ndim == 4:
        # Average over epochs
        data = data.mean(axis=0)

    # Now (n_ch, n_fr, n_ti)

    # Convert to gwexpy: (n_ti, n_fr) usually?
    SpectrogramDict = ConverterRegistry.get_constructor("SpectrogramDict")

    sd = SpectrogramDict()

    for i, name in enumerate(ch_names):
        # Transpose back to (times, freqs)
        val = data[i].T
        sd[name] = cls(val, times=times, frequencies=freqs, name=name, **kwargs)

    if len(ch_names) == 1:
        return sd[ch_names[0]]

    return sd
