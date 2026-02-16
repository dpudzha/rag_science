"""Text-to-SQL tool for querying tabular data via natural language."""
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from tools.sql_database import SQLDatabase

logger = logging.getLogger(__name__)

_SQL_GENERATION_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a valid SQLite SELECT query.

Database schema:
{schema}

Sample data:
{samples}

Rules:
- Generate ONLY a SELECT query, nothing else
- Do NOT use DROP, DELETE, UPDATE, INSERT, or any destructive operations
- Use single quotes for string literals
- Output ONLY the SQL query, no explanation

The user's question follows. Respond with only the SQL query:"""


class SQLTool(BaseTool):
    """Query tabular data using natural language converted to SQL."""

    name: str = "query_tables"
    description: str = (
        "Query tabular data stored in the database. Use this tool when the question "
        "is about specific data points, statistics, or numerical information from "
        "tables or spreadsheets. Input should be a natural language question about the data."
    )
    db: SQLDatabase
    llm: Any = None  # ChatOllama | None

    model_config = {"arbitrary_types_allowed": True}

    def _get_llm(self):
        if self.llm is None:
            from utils import get_default_llm
            self.llm = get_default_llm()
        return self.llm

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        schema = self.db.get_schema()
        if "No tables found" in schema:
            return "No tabular data available in the database."

        # Get sample rows for context
        samples = ""
        for table_name in self.db.get_table_names()[:5]:
            rows = self.db.get_sample_rows(table_name, limit=2)
            if rows:
                samples += f"\n{table_name}: {rows}"

        system_prompt = (_SQL_GENERATION_PROMPT
                         .replace("{schema}", schema)
                         .replace("{samples}", samples))

        llm = self._get_llm()
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ])
            sql = response.content.strip()
            # Clean up: remove markdown code blocks if present
            if sql.startswith("```"):
                sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
                sql = sql.rsplit("```", 1)[0]
            sql = sql.strip()

            logger.info("Generated SQL: %s", sql)
            results = self.db.execute_query(sql)

            if not results:
                return "Query returned no results."

            # Format results
            output = f"SQL: {sql}\n\nResults ({len(results)} rows):\n"
            for row in results[:20]:  # Limit output
                output += str(row) + "\n"
            if len(results) > 20:
                output += f"... ({len(results) - 20} more rows)"

            return output
        except ValueError as e:
            return f"SQL safety check failed: {e}"
        except Exception as e:
            logger.warning("SQL tool failed: %s", e)
            return f"Failed to query database: {e}"
