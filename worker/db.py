"""Thin psycopg (v3) helpers. No ORM.

The connection is opened with ``autocommit=True`` because the worker issues
small, independent status writes; there is no multi-statement transaction to
manage. Rows come back as dicts so callers can read Prisma's camelCase columns
by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg import Connection
from psycopg.rows import DictRow, dict_row

# Positional query parameters (%s placeholders).
Params = Sequence[Any] | None


def connect(database_url: str) -> Connection[DictRow]:
    """Open an autocommit connection that yields dict rows."""

    return psycopg.connect(database_url, autocommit=True, row_factory=dict_row)


def fetch_one(
    conn: Connection[DictRow], sql: str, params: Params = None
) -> DictRow | None:
    """Run a query and return the first row (or None)."""

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(conn: Connection[DictRow], sql: str, params: Params = None) -> int:
    """Run a statement and return the number of affected rows."""

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount
