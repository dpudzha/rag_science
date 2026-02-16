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


class TestRAGAgent:
    def test_invoke_builds_fresh_executor_per_call(self):
        from agent import RAGAgent

        retriever = MagicMock()
        agent = RAGAgent(retriever=retriever, llm=MagicMock())

        executor_a = MagicMock()
        executor_a.invoke.return_value = {
            "output": "Answer A",
            "intermediate_steps": [],
            "source_documents": [],
        }
        executor_b = MagicMock()
        executor_b.invoke.return_value = {
            "output": "Answer B",
            "intermediate_steps": [],
            "source_documents": [],
        }

        with patch.object(agent, "_build_executor", side_effect=[executor_a, executor_b]) as build_exec:
            result_a = agent.invoke("Question A")
            result_b = agent.invoke("Question B")

        assert build_exec.call_count == 2
        assert result_a["answer"] == "Answer A"
        assert result_b["answer"] == "Answer B"

    def test_fallback_supports_legacy_retriever_interface(self):
        from agent import RAGAgent

        class LegacyRetriever:
            def get_relevant_documents(self, query):
                return [
                    Document(
                        page_content=f"Legacy match for {query}",
                        metadata={"source": "legacy.pdf", "page": 3},
                    )
                ]

        agent = RAGAgent(retriever=LegacyRetriever(), llm=MagicMock())
        broken_executor = MagicMock()
        broken_executor.invoke.side_effect = RuntimeError("agent failed")

        with patch.object(agent, "_build_executor", return_value=broken_executor):
            result = agent.invoke("legacy question")

        assert result["tool_used"] == "fallback_rag"
        assert len(result["source_documents"]) == 1
        assert result["source_documents"][0].metadata["source"] == "legacy.pdf"
        assert "Legacy match for legacy question" in result["answer"]

    def test_invoke_returns_deduplicated_source_documents(self):
        from agent import RAGAgent

        retriever = MagicMock()
        agent = RAGAgent(retriever=retriever, llm=MagicMock())

        tool_action = MagicMock()
        tool_action.tool = "search_papers"

        doc_a = Document(page_content="Same content", metadata={"source": "a.pdf", "page": 1})
        doc_a_dup = Document(page_content="Same content", metadata={"source": "a.pdf", "page": 1})
        doc_b = Document(page_content="Other content", metadata={"source": "b.pdf", "page": 2})

        executor = MagicMock()
        executor.invoke.return_value = {
            "output": "Combined answer",
            "intermediate_steps": [(tool_action, "observation")],
            "source_documents": [doc_a, doc_a_dup, doc_b],
        }

        with patch.object(agent, "_build_executor", return_value=executor):
            result = agent.invoke("question")

        assert result["answer"] == "Combined answer"
        assert result["tool_used"] == "search_papers"
        assert "intermediate_steps" in result
        assert len(result["source_documents"]) == 2
