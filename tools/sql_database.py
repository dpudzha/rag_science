"""SQLite database wrapper for tabular data querying."""
import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

from config import SQL_DATABASE_PATH, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

_UNSAFE_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|EXEC|EXECUTE"
    r"|ATTACH|DETACH|PRAGMA|LOAD_EXTENSION|SAVEPOINT|RELEASE|REINDEX|VACUUM)\b",
    re.IGNORECASE,
)

DEFAULT_DB_PATH = (
    Path(SQL_DATABASE_PATH).expanduser()
    if SQL_DATABASE_PATH.strip()
    else Path(VECTORSTORE_DIR) / "tables.db"
)


class SQLDatabase:
    """SQLite wrapper for tabular data. Only allows SELECT queries."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self, readonly: bool = True) -> sqlite3.Connection:
        if readonly and Path(self._db_path).exists():
            uri = f"file:{self._db_path}?mode=ro"
            return sqlite3.connect(uri, uri=True)
        return sqlite3.connect(self._db_path)

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Create a table from a pandas DataFrame, replacing if exists."""
        conn = self._connect(readonly=False)
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info("Created table '%s' with %d rows", table_name, len(df))
        finally:
            conn.close()

    def execute_query(self, sql: str) -> list[dict]:
        """Execute a SELECT-only query and return results as list of dicts."""
        sql = sql.strip().rstrip(";")

        if _UNSAFE_PATTERN.search(sql):
            raise ValueError("Unsafe SQL operation detected. Only SELECT queries are allowed.")

        if not sql.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")

        conn = self._connect()
        try:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()

    def get_schema(self) -> str:
        """Return schema information for all tables."""
        tables = self.get_table_names()
        if not tables:
            return "No tables found."
        conn = self._connect()
        try:
            schema_parts = []
            for table_name in tables:
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    schema_parts.append(row[0])
            return "\n\n".join(schema_parts)
        finally:
            conn.close()

    def get_table_names(self) -> list[str]:
        """Return list of table names."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_sample_rows(self, table_name: str, limit: int = 3) -> list[dict]:
        """Return sample rows from a table."""
        safe_name = table_name.replace('"', '""')
        return self.execute_query(f'SELECT * FROM "{safe_name}" LIMIT {limit}')
