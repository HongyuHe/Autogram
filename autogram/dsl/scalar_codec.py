"""Lossless JSON/text codec for DSL condition and category scalars."""

from __future__ import annotations

import json
import math
import numbers
import os
import re
from datetime import timedelta, timezone as datetime_timezone
from zoneinfo import TZPATH, ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from dateutil import tz as dateutil_tz

try:
    import pytz
except ImportError:  # pragma: no cover - optional timezone implementation
    pytz = None


_TEMPORAL_TAG = "__autogram_temporal__"
_NUMBER_TOKEN = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z"
)
_NONFINITE_TOKEN = re.compile(
    r"[+-]?(?:nan|inf(?:inity)?)\Z",
    re.IGNORECASE,
)
_INT_TOKEN = re.compile(r"-?(?:0|[1-9]\d*)\Z")
_TIMEZONE_KEY_PART = re.compile(r"[A-Za-z0-9._+-]+\Z")
_MAX_UTC_OFFSET_MICROSECONDS = 86_400 * 1_000_000


def _offset_microseconds(offset: timedelta | None) -> int | None:
    if offset is None:
        return None
    return (
        (
            offset.days * 86_400
            + offset.seconds
        ) * 1_000_000
        + offset.microseconds
    )


def _valid_timezone_key(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    parts = value.split("/")
    return all(
        part not in ("", ".", "..")
        and _TIMEZONE_KEY_PART.fullmatch(part) is not None
        for part in parts
    )


def _dateutil_tzfile_key(value: object) -> str | None:
    filename = getattr(value, "_filename", None)
    if not isinstance(filename, str) or not filename:
        return None
    roots = tuple(dict.fromkeys((
        *TZPATH,
        *getattr(dateutil_tz, "TZPATHS", ()),
    )))
    filename = os.path.abspath(filename)
    for root in roots:
        try:
            relative = os.path.relpath(filename, os.path.abspath(root))
        except (TypeError, ValueError):
            continue
        key = relative.replace(os.sep, "/")
        if not _valid_timezone_key(key):
            continue
        candidate = dateutil_tz.gettz(key)
        if (
            type(candidate) is dateutil_tz.tzfile
            and candidate == value
        ):
            return key
    return None


def _unsupported_timezone(label: str, value: object) -> ValueError:
    timezone_type = (
        f"{type(value).__module__}.{type(value).__qualname__}"
    )
    return ValueError(
        f"{label} uses unsupported timezone object {timezone_type}; "
        "supported timezones are datetime.timezone, zoneinfo, pytz, "
        "and reconstructable dateutil zones"
    )


def _timezone_payload(value: object, label: str) -> dict:
    if type(value) is ZoneInfo:
        if not _valid_timezone_key(value.key):
            raise _unsupported_timezone(label, value)
        return {
            "kind": "zoneinfo",
            "key": value.key,
        }
    if (
        pytz is not None
        and type(value) is type(pytz.FixedOffset(1))
    ):
        offset_microseconds = _offset_microseconds(
            value.utcoffset(None)
        )
        if (
            offset_microseconds is None
            or offset_microseconds % 60_000_000
            or not -_MAX_UTC_OFFSET_MICROSECONDS
            < offset_microseconds
            < _MAX_UTC_OFFSET_MICROSECONDS
        ):
            raise _unsupported_timezone(label, value)
        return {
            "kind": "pytz.FixedOffset",
            "offset_microseconds": str(offset_microseconds),
        }
    if (
        pytz is not None
        and isinstance(value, pytz.tzinfo.BaseTzInfo)
        and type(value).__module__.startswith("pytz")
    ):
        key = getattr(value, "zone", None)
        if not _valid_timezone_key(key):
            raise _unsupported_timezone(label, value)
        try:
            pytz.timezone(key)
        except (KeyError, ValueError):
            raise _unsupported_timezone(label, value) from None
        return {
            "kind": "pytz",
            "key": key,
        }
    if type(value) is dateutil_tz.tzfile:
        key = _dateutil_tzfile_key(value)
        if key is None:
            raise _unsupported_timezone(label, value)
        return {
            "kind": "dateutil.tzfile",
            "key": key,
        }
    if type(value) is dateutil_tz.tzutc:
        return {"kind": "dateutil.tzutc"}
    if type(value) is dateutil_tz.tzoffset:
        offset_microseconds = _offset_microseconds(
            value.utcoffset(None)
        )
        name = value.tzname(None)
        if (
            offset_microseconds is None
            or not -_MAX_UTC_OFFSET_MICROSECONDS
            < offset_microseconds
            < _MAX_UTC_OFFSET_MICROSECONDS
            or (name is not None and not isinstance(name, str))
        ):
            raise _unsupported_timezone(label, value)
        return {
            "kind": "dateutil.tzoffset",
            "name": name,
            "offset_microseconds": str(offset_microseconds),
        }
    if type(value) is datetime_timezone:
        if value is datetime_timezone.utc:
            return {"kind": "datetime.timezone.utc"}
        offset_microseconds = _offset_microseconds(
            value.utcoffset(None)
        )
        name = value.tzname(None)
        if (
            offset_microseconds is None
            or not -_MAX_UTC_OFFSET_MICROSECONDS
            < offset_microseconds
            < _MAX_UTC_OFFSET_MICROSECONDS
            or not isinstance(name, str)
        ):
            raise _unsupported_timezone(label, value)
        return {
            "kind": "datetime.timezone",
            "name": name,
            "offset_microseconds": str(offset_microseconds),
        }
    raise _unsupported_timezone(label, value)


def _timezone_offset(
    payload: dict,
    *,
    expected_fields: set[str],
    label: str,
) -> timedelta:
    if set(payload) != expected_fields:
        raise ValueError(f"{label} has an invalid timezone descriptor")
    offset_text = payload.get("offset_microseconds")
    if (
        not isinstance(offset_text, str)
        or _INT_TOKEN.fullmatch(offset_text) is None
    ):
        raise ValueError(f"{label} has an invalid timezone descriptor")
    offset_microseconds = int(offset_text)
    if not (
        -_MAX_UTC_OFFSET_MICROSECONDS
        < offset_microseconds
        < _MAX_UTC_OFFSET_MICROSECONDS
    ):
        raise ValueError(f"{label} has an invalid timezone descriptor")
    return timedelta(microseconds=offset_microseconds)


def _timezone_from_json(payload: object, label: str):
    if not isinstance(payload, dict):
        raise ValueError(f"{label} has an invalid timezone descriptor")
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise ValueError(f"{label} has an invalid timezone descriptor")
    if kind in {"zoneinfo", "pytz", "dateutil.tzfile"}:
        if set(payload) != {"kind", "key"}:
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        key = payload.get("key")
        if not _valid_timezone_key(key):
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        if kind == "zoneinfo":
            try:
                return ZoneInfo(key)
            except (
                KeyError,
                ValueError,
                ZoneInfoNotFoundError,
            ) as error:
                raise ValueError(
                    f"{label} has an invalid timezone descriptor"
                ) from error
        if kind == "pytz":
            if pytz is None:
                raise ValueError(
                    f"{label} requires pytz to reconstruct its timezone"
                )
            try:
                return pytz.timezone(key)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"{label} has an invalid timezone descriptor"
                ) from error
        result = dateutil_tz.gettz(key)
        if type(result) is not dateutil_tz.tzfile:
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        return result
    if kind == "dateutil.tzutc":
        if set(payload) != {"kind"}:
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        return dateutil_tz.UTC
    if kind == "pytz.FixedOffset":
        if pytz is None:
            raise ValueError(
                f"{label} requires pytz to reconstruct its timezone"
            )
        offset = _timezone_offset(
            payload,
            expected_fields={
                "kind",
                "offset_microseconds",
            },
            label=label,
        )
        offset_microseconds = _offset_microseconds(offset)
        if (
            offset_microseconds is None
            or offset_microseconds % 60_000_000
        ):
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        return pytz.FixedOffset(offset_microseconds // 60_000_000)
    if kind == "dateutil.tzoffset":
        offset = _timezone_offset(
            payload,
            expected_fields={
                "kind",
                "name",
                "offset_microseconds",
            },
            label=label,
        )
        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        return dateutil_tz.tzoffset(name, offset)
    if kind == "datetime.timezone.utc":
        if set(payload) != {"kind"}:
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        return datetime_timezone.utc
    if kind == "datetime.timezone":
        offset = _timezone_offset(
            payload,
            expected_fields={
                "kind",
                "name",
                "offset_microseconds",
            },
            label=label,
        )
        name = payload.get("name")
        if not isinstance(name, str):
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            )
        try:
            return datetime_timezone(offset, name)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} has an invalid timezone descriptor"
            ) from error
    raise ValueError(
        f"{label} has an unsupported timezone descriptor {kind!r}"
    )


def _temporal_payload(value: object, label: str) -> dict | None:
    if value is pd.NaT:
        return {_TEMPORAL_TAG: "pandas.NaT"}
    if isinstance(value, pd.Timestamp):
        offset_microseconds = _offset_microseconds(value.utcoffset())
        offset_microseconds = (
            None
            if offset_microseconds is None
            else str(offset_microseconds)
        )
        return {
            _TEMPORAL_TAG: "pandas.Timestamp",
            "dtype": str(value.asm8.dtype),
            "value": str(int(value.asm8.view("i8"))),
            "timezone": (
                None
                if value.tzinfo is None
                else _timezone_payload(value.tz, label)
            ),
            "utc_offset_microseconds": offset_microseconds,
        }
    if isinstance(value, pd.Timedelta):
        return {
            _TEMPORAL_TAG: "pandas.Timedelta",
            "dtype": str(value.asm8.dtype),
            "value": str(int(value.asm8.view("i8"))),
        }
    if isinstance(value, np.datetime64):
        return {
            _TEMPORAL_TAG: "numpy.datetime64",
            "dtype": str(value.dtype),
            "value": str(int(value.view("i8"))),
        }
    if isinstance(value, np.timedelta64):
        return {
            _TEMPORAL_TAG: "numpy.timedelta64",
            "dtype": str(value.dtype),
            "value": str(int(value.view("i8"))),
        }
    return None


def scalar_to_json(value: object, label: str) -> object:
    """Encode one runtime DSL scalar as a JSON-compatible value."""
    temporal = _temporal_payload(value, label)
    if temporal is not None:
        return temporal
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return number
    raise TypeError(
        f"{label} must be a string, boolean, number, null, "
        f"or supported temporal scalar; got {value!r}"
    )


def _tagged_dtype(
    payload: dict,
    *,
    expected_kind: str,
    expected_fields: set[str],
    label: str,
):
    if set(payload) != expected_fields:
        raise ValueError(f"{label} has an invalid tagged temporal payload")
    dtype_text = payload.get("dtype")
    raw_text = payload.get("value")
    if (
        not isinstance(dtype_text, str)
        or not isinstance(raw_text, str)
        or _INT_TOKEN.fullmatch(raw_text) is None
    ):
        raise ValueError(f"{label} has an invalid tagged temporal payload")
    try:
        dtype = np.dtype(dtype_text)
        raw = int(raw_text)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{label} has an invalid tagged temporal payload"
        ) from error
    if dtype.kind != expected_kind or not (
        np.iinfo(np.int64).min <= raw <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"{label} has an invalid tagged temporal payload")
    return np.asarray(raw, dtype=np.int64).view(dtype)[()]


def _temporal_from_json(payload: dict, label: str):
    kind = payload.get(_TEMPORAL_TAG)
    if not isinstance(kind, str):
        raise ValueError(f"{label} must be a JSON scalar")
    if kind == "pandas.NaT":
        if set(payload) != {_TEMPORAL_TAG}:
            raise ValueError(f"{label} has an invalid tagged temporal payload")
        return pd.NaT
    if kind == "pandas.Timestamp":
        scalar = _tagged_dtype(
            payload,
            expected_kind="M",
            expected_fields={
                _TEMPORAL_TAG,
                "dtype",
                "value",
                "timezone",
                "utc_offset_microseconds",
            },
            label=label,
        )
        timezone_payload = payload.get("timezone")
        offset_text = payload.get("utc_offset_microseconds")
        if (
            timezone_payload is not None
            and not isinstance(timezone_payload, dict)
        ) or (
            offset_text is not None
            and (
                not isinstance(offset_text, str)
                or _INT_TOKEN.fullmatch(offset_text) is None
            )
        ) or (
            (timezone_payload is None) != (offset_text is None)
        ):
            raise ValueError(f"{label} has an invalid tagged temporal payload")
        result = pd.Timestamp(scalar)
        if timezone_payload is not None:
            expected_offset = int(offset_text)
            try:
                timezone = _timezone_from_json(
                    timezone_payload,
                    label,
                )
                result = (
                    result.tz_localize(datetime_timezone.utc)
                    .tz_convert(timezone)
                )
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{label} has an invalid tagged temporal payload"
                ) from error
            if (
                _offset_microseconds(result.utcoffset())
                != expected_offset
                or _timezone_payload(result.tz, label)
                != timezone_payload
            ):
                raise ValueError(
                    f"{label} has an invalid tagged temporal payload"
                )
        return result
    if kind == "pandas.Timedelta":
        return pd.Timedelta(_tagged_dtype(
            payload,
            expected_kind="m",
            expected_fields={
                _TEMPORAL_TAG,
                "dtype",
                "value",
            },
            label=label,
        ))
    if kind == "numpy.datetime64":
        return _tagged_dtype(
            payload,
            expected_kind="M",
            expected_fields={
                _TEMPORAL_TAG,
                "dtype",
                "value",
            },
            label=label,
        )
    if kind == "numpy.timedelta64":
        return _tagged_dtype(
            payload,
            expected_kind="m",
            expected_fields={
                _TEMPORAL_TAG,
                "dtype",
                "value",
            },
            label=label,
        )
    raise ValueError(f"{label} has an unknown tagged temporal type")


def scalar_from_json(value: object, label: str):
    """Decode one JSON-compatible DSL scalar, validating finiteness."""
    if isinstance(value, dict):
        return _temporal_from_json(value, label)
    temporal = _temporal_payload(value, label)
    if temporal is not None:
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return number
    raise ValueError(f"{label} must be a JSON scalar")


def scalar_to_text(value: object, label: str = "DSL scalar") -> str:
    """Encode one scalar as deterministic compact JSON surface text."""
    return json.dumps(
        scalar_to_json(value, label),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def scalar_from_text(text: str, label: str = "DSL scalar"):
    """Decode canonical scalar text plus the legacy bare scalar spellings."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    else:
        return scalar_from_json(value, label)
    if text == "True":
        return True
    if text == "False":
        return False
    if text == "None":
        return None
    if _NONFINITE_TOKEN.fullmatch(text):
        raise ValueError(f"{label} must be finite")
    if _NUMBER_TOKEN.fullmatch(text):
        try:
            number = float(text)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} must be finite") from error
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return int(number) if number.is_integer() else number
    return text
