"""SQLite database wrapper for tabular data querying."""
import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

from config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)

_UNSAFE_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

DEFAULT_DB_PATH = Path(VECTORSTORE_DIR) / "tables.db"


class SQLDatabase:
    """SQLite wrapper for tabular data. Only allows SELECT queries."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame):
        """Create a table from a pandas DataFrame, replacing if exists."""
        conn = self._connect()
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info("Created table '%s' with %d rows", table_name, len(df))
        finally:
            conn.close()

    def execute_query(self, sql: str) -> list[dict]:
        """Execute a SELECT-only query and return results as list of dicts."""
        sql = sql.strip().rstrip(";")

        if _UNSAFE_PATTERN.search(sql):
            raise ValueError(f"Unsafe SQL operation detected. Only SELECT queries are allowed.")

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
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            schema_parts = []
            for table in tables:
                cursor = conn.execute(f"PRAGMA table_info('{table}')")
                columns = cursor.fetchall()
                col_defs = [f"  {col[1]} {col[2]}" for col in columns]

                # Get row count
                count_cursor = conn.execute(f"SELECT COUNT(*) FROM '{table}'")
                row_count = count_cursor.fetchone()[0]

                schema_parts.append(
                    f"Table: {table} ({row_count} rows)\n" + "\n".join(col_defs)
                )

            return "\n\n".join(schema_parts) if schema_parts else "No tables found."
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
        return self.execute_query(f"SELECT * FROM '{table_name}' LIMIT {limit}")
