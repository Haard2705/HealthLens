"""The connection URL and the engine. Supabase-shaped URLs are normalised here, once.

Supabase hands out ``postgresql://postgres:<password>@<host>:5432/postgres``. Two things have
to happen before SQLAlchemy 2.x can use it:

* the driver must be named, or SQLAlchemy picks psycopg2 (which we do not install), so
  ``postgresql://`` becomes ``postgresql+psycopg://``;
* Supabase requires TLS, so ``sslmode=require`` is added unless the URL already says
  otherwise. A local database is left alone - it usually has no TLS at all.

Nothing here reads the environment: the URL arrives as an argument, from Settings.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DRIVER = "postgresql+psycopg"
SSL_PARAM = "sslmode"
SSL_REQUIRE = "require"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "")
# Supabase: 5432 = direct connection, 6543 = the pooler (see docs/assignments/H17-database.md).
POOLER_PORT = 6543


class DatabaseUrlError(ValueError):
    """``DATABASE_URL`` is not a Postgres URL this app can use."""


def normalise_url(raw: str) -> str:
    """A ``DATABASE_URL`` turned into the exact URL SQLAlchemy should be given."""
    url = (raw or "").strip()
    if not url:
        raise DatabaseUrlError("DATABASE_URL is empty")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme in ("postgres", "postgresql"):
        scheme = DRIVER
    elif not scheme.startswith("postgresql+"):
        raise DatabaseUrlError(f"DATABASE_URL must be a postgresql:// URL, got {parts.scheme!r}://")
    if not parts.netloc:
        raise DatabaseUrlError("DATABASE_URL has no host")

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    if SSL_PARAM not in query and not _is_local(parts.hostname):
        query[SSL_PARAM] = SSL_REQUIRE
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _is_local(hostname: str | None) -> bool:
    return (hostname or "") in LOCAL_HOSTS


def is_pooled(raw: str) -> bool:
    """True for Supabase's pooled port, where prepared statements must be disabled."""
    try:
        return urlsplit(raw).port == POOLER_PORT
    except ValueError:
        return False


def create_engine(raw: str) -> Any:
    """A pooled SQLAlchemy engine for ``raw``. Imports SQLAlchemy lazily, on purpose."""
    from sqlalchemy import create_engine as sa_create_engine

    url = normalise_url(raw)
    # pool_pre_ping: Supabase closes idle connections; a dead one is replaced, not raised.
    options: dict[str, Any] = {"pool_pre_ping": True, "pool_size": 5, "max_overflow": 5}
    if is_pooled(raw):
        # Transaction pooling cannot keep server-side prepared statements between statements.
        options["connect_args"] = {"prepare_threshold": None}
    return sa_create_engine(url, **options)
