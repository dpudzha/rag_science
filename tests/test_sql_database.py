"""Tests for sql_database.py."""
import importlib
import sqlite3
import pytest
import pandas as pd

from tools.sql_database import SQLDatabase


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    return SQLDatabase(db_path=db_path)


@pytest.fixture
def db_with_table(db):
    df = pd.DataFrame({
        "method": ["CNN", "RNN", "Transformer"],
        "accuracy": [95.2, 92.1, 97.3],
        "year": [2019, 2018, 2020],
    })
    db.create_table_from_dataframe("results", df)
    return db


class TestSQLDatabase:
    def test_create_table(self, db):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        db.create_table_from_dataframe("test", df)
        assert "test" in db.get_table_names()

    def test_select_query(self, db_with_table):
        results = db_with_table.execute_query("SELECT * FROM results")
        assert len(results) == 3
        assert results[0]["method"] == "CNN"

    def test_select_with_where(self, db_with_table):
        results = db_with_table.execute_query("SELECT * FROM results WHERE accuracy > 93")
        assert len(results) == 2

    def test_rejects_drop(self, db_with_table):
        with pytest.raises(ValueError, match="Unsafe SQL"):
            db_with_table.execute_query("DROP TABLE results")

    def test_rejects_delete(self, db_with_table):
        with pytest.raises(ValueError, match="Unsafe SQL"):
            db_with_table.execute_query("DELETE FROM results")

    def test_rejects_update(self, db_with_table):
        with pytest.raises(ValueError, match="Unsafe SQL"):
            db_with_table.execute_query("UPDATE results SET accuracy = 100")

    def test_rejects_insert(self, db_with_table):
        with pytest.raises(ValueError, match="Unsafe SQL"):
            db_with_table.execute_query("INSERT INTO results VALUES ('X', 1, 2)")

    def test_rejects_non_select(self, db_with_table):
        with pytest.raises(ValueError, match="Only SELECT"):
            db_with_table.execute_query("PRAGMA table_info(results)")

    def test_get_schema(self, db_with_table):
        schema = db_with_table.get_schema()
        assert "results" in schema
        assert "method" in schema
        assert "TEXT" in schema or "REAL" in schema

    def test_get_sample_rows(self, db_with_table):
        rows = db_with_table.get_sample_rows("results", limit=2)
        assert len(rows) == 2

    def test_empty_db_schema(self, db):
        assert "No tables found" in db.get_schema()

    def test_empty_db_table_names(self, db):
        assert db.get_table_names() == []


class TestSQLDatabaseDefaultPath:
    def test_default_path_falls_back_to_vectorstore_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VECTORSTORE_DIR", str(tmp_path / "vs"))
        monkeypatch.setenv("SQL_DATABASE_PATH", "")

        import config
        import tools.sql_database as sql_database

        importlib.reload(config)
        importlib.reload(sql_database)

        db = sql_database.SQLDatabase()
        assert db._db_path == str(tmp_path / "vs" / "tables.db")

    def test_default_path_uses_sql_database_path_when_set(self, tmp_path, monkeypatch):
        explicit_db = tmp_path / "custom" / "tables.sqlite"
        monkeypatch.setenv("VECTORSTORE_DIR", str(tmp_path / "vs"))
        monkeypatch.setenv("SQL_DATABASE_PATH", str(explicit_db))

        import config
        import tools.sql_database as sql_database

        importlib.reload(config)
        importlib.reload(sql_database)

        db = sql_database.SQLDatabase()
        assert db._db_path == str(explicit_db)


class TestSQLBlocklist:
    """Test that dangerous SQL operations beyond basic DML are blocked."""

    def test_rejects_attach(self, db_with_table):
        with pytest.raises(ValueError, match="Unsafe SQL"):
            db_with_table.execute_query("ATTACH DATABASE ':memory:' AS other")

    def test_rejects_pragma(self, db_with_table):
        with pytest.raises(ValueError, match="Unsafe SQL|Only SELECT"):
            db_with_table.execute_query("PRAGMA table_info(results)")

    def test_rejects_load_extension(self, db_with_table):
        with pytest.raises(ValueError, match="Unsafe SQL"):
            db_with_table.execute_query("SELECT LOAD_EXTENSION('evil.so')")


class TestReadOnlyConnection:
    """Test that read-only connections reject writes."""

    def test_readonly_connection_rejects_writes(self, db_with_table):
        conn = db_with_table._connect(readonly=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO results VALUES ('X', 1, 2)")
        finally:
            conn.close()

    def test_readonly_connection_allows_reads(self, db_with_table):
        conn = db_with_table._connect(readonly=True)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM results")
            count = cursor.fetchone()[0]
            assert count == 3
        finally:
            conn.close()
