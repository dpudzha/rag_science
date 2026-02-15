"""Tests for agent.py and tools."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestRAGTool:
    def test_returns_formatted_results(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Neural networks are powerful.",
                     metadata={"source": "paper1.pdf", "page": 1}),
        ]
        from tools.rag_tool import RAGTool
        tool = RAGTool(retriever=mock_retriever)
        result = tool._run("test query")
        assert "paper1.pdf" in result
        assert "Neural networks" in result

    def test_returns_message_when_no_docs(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        from tools.rag_tool import RAGTool
        tool = RAGTool(retriever=mock_retriever)
        result = tool._run("test query")
        assert "No relevant documents" in result


class TestSQLTool:
    def test_returns_results(self, tmp_path):
        import pandas as pd
        from sql_database import SQLDatabase
        from tools.sql_tool import SQLTool
        from langchain_core.messages import AIMessage

        db = SQLDatabase(db_path=tmp_path / "test.db")
        df = pd.DataFrame({"method": ["CNN"], "accuracy": [95.2]})
        db.create_table_from_dataframe("results", df)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="SELECT * FROM results")

        tool = SQLTool(db=db, llm=mock_llm)
        result = tool._run("What methods were used?")
        assert "CNN" in result
        assert "95.2" in result

    def test_rejects_unsafe_sql(self, tmp_path):
        import pandas as pd
        from sql_database import SQLDatabase
        from tools.sql_tool import SQLTool
        from langchain_core.messages import AIMessage

        db = SQLDatabase(db_path=tmp_path / "test.db")
        df = pd.DataFrame({"a": [1]})
        db.create_table_from_dataframe("test", df)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="DROP TABLE test")

        tool = SQLTool(db=db, llm=mock_llm)
        result = tool._run("Drop the table")
        assert "safety check failed" in result.lower() or "Only SELECT" in result

    def test_handles_no_tables(self, tmp_path):
        from sql_database import SQLDatabase
        from tools.sql_tool import SQLTool

        db = SQLDatabase(db_path=tmp_path / "empty.db")
        tool = SQLTool(db=db)
        result = tool._run("What data is available?")
        assert "No tabular data" in result
