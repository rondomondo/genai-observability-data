"""Shared slowapi Limiter instance and per-request helper used by all routers."""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)


def get_limit(attr: str) -> str:
    """Return the rate-limit string for attr from live settings.

    Evaluated lazily so the settings singleton is read at request time,
    not at import time -- which would prevent overriding in tests.

    Args:
        attr: The Settings field name, e.g. "rate_limit_ingest".

    Returns:
        A limits-syntax string such as "120/minute".
    """
    from app.config import get_settings

    return getattr(get_settings(), attr)


def limit_for(attr: str):
    """Return a slowapi decorator that reads its limit string from settings.

    Args:
        attr: The Settings field name to read the limit from.
    """

    def _dynamic_limit() -> str:
        return get_limit(attr)

    return limiter.limit(_dynamic_limit)
