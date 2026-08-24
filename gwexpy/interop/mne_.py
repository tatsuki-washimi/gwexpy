from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _set_legacy_effective_epoch(raw: Any, epoch: float) -> None:
    """Set ``raw``'s official base time for an effective legacy epoch."""
    base_epoch = epoch - index(raw.first_samp) / raw.info["sfreq"]
    raw.set_meas_date(gps_to_datetime_utc(base_epoch))


def _install_add_channels_guard(raw: Any) -> None:
    """Reject conflicting exact epochs before MNE mutates an in-memory Raw."""
    if getattr(raw, "_gwex_exact_add_channels_guard", False):
        return

    def guarded(self: Any, add_list: Any, *args: Any, **kwargs: Any) -> Any:
        current = _raw_exact_epochs(self)
        current_dt = {name: _raw_exact_dt(self, name) for name in current}
        current_effective: dict[str, int] = {}
        for name in current:
            epoch = _raw_channel_epoch(self, name)
            assert epoch is not None
            current_effective[name] = epoch
        additions: dict[str, int] = {}
        additions_dt: dict[str, int] = {}
        additions_effective: dict[str, int] = {}
        receiver_legacy_epoch = _raw_legacy_effective_epoch(self, current)
        legacy_effective_epochs: list[float] = []
        if receiver_legacy_epoch is not None:
            legacy_effective_epochs.append(receiver_legacy_epoch)
        for other in add_list:
            other_epochs = _raw_exact_epochs(other)
            additions.update(other_epochs)
            additions_dt.update(
                {name: _raw_exact_dt(other, name) for name in other_epochs}
            )
            for name in other_epochs:
                epoch = _raw_channel_epoch(other, name)
                assert epoch is not None
                additions_effective[name] = epoch
            legacy_epoch = _raw_legacy_effective_epoch(other, other_epochs)
            if legacy_epoch is not None:
                legacy_effective_epochs.append(legacy_epoch)
        exact_values = set(current_effective.values()) | set(
            additions_effective.values()
        )
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
            raise ValueError(
                "cannot add channels with mismatched effective legacy epochs"
            )

        # Look up the class method at call time.  Copy/deepcopy can duplicate
        # this instance-bound guard, so closing over ``raw.add_channels`` would
        # instead mutate the original Raw object.
        result = type(self).add_channels(self, add_list, *args, **kwargs)
        if legacy_effective_epochs and receiver_legacy_epoch is None:
            _set_legacy_effective_epoch(self, legacy_effective_epochs[0])
        if current or additions:
            receiver_first_samp = index(self.first_samp)
            normalized_additions = {
                name: epoch - receiver_first_samp * additions_dt[name]
                for name, epoch in additions_effective.items()
            }
            merged = {**current, **normalized_additions}
            merged_dt = {**current_dt, **additions_dt}
            setattr(self, _GWEX_CHANNEL_T0_GPS_NS_ATTR, merged)
            setattr(self, _GWEX_CHANNEL_DT_GPS_NS_ATTR, merged_dt)
            if len(merged) == len(self.ch_names) and len(set(merged.values())) == 1:
                setattr(self, _GWEX_T0_GPS_NS_ATTR, next(iter(merged.values())))
            else:
                self.__dict__.pop(_GWEX_T0_GPS_NS_ATTR, None)
            setattr(self, _GWEX_MEAS_DATE_ATTR, self.info.get("meas_date"))
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
