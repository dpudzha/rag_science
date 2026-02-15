"""RAG Agent with tool selection between vector search and text-to-SQL."""
import logging
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from config import OLLAMA_BASE_URL, LLM_MODEL
from sql_database import SQLDatabase
from tools.rag_tool import RAGTool
from tools.sql_tool import SQLTool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system.txt"
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text()


class RAGAgent:
    """Agent that selects between RAG search and SQL query tools."""

    def __init__(self, retriever, max_iterations: int = 5, llm: ChatOllama | None = None):
        self._llm = llm or ChatOllama(
            model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0
        )
        self._retriever = retriever
        self._max_iterations = max_iterations
        self._executor = None
        self._tools = None

    def _build_tools(self) -> list:
        tools = [RAGTool(retriever=self._retriever)]

        # Add SQL tool if tables exist
        db = SQLDatabase()
        if db.get_table_names():
            tools.append(SQLTool(db=db))
            logger.info("SQL tool enabled with tables: %s", db.get_table_names())
        else:
            logger.info("No SQL tables found, SQL tool disabled")

        return tools

    def _build_executor(self) -> AgentExecutor:
        self._tools = self._build_tools()

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self._llm, self._tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            max_iterations=self._max_iterations,
            handle_parsing_errors=True,
            verbose=False,
        )

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = self._build_executor()
        return self._executor

    def invoke(self, question: str, chat_history: list | None = None) -> dict:
        """Run the agent on a question. Returns dict with 'output' and 'tool_used'."""
        try:
            result = self.executor.invoke({
                "input": question,
                "chat_history": chat_history or [],
            })
            # Determine which tool was used
            tool_used = "unknown"
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                tool_used = intermediate_steps[0][0].tool if intermediate_steps[0] else "unknown"

            return {
                "answer": result.get("output", ""),
                "tool_used": tool_used,
                "intermediate_steps": intermediate_steps,
            }
        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            # Fall back to direct RAG retrieval
            logger.info("Falling back to direct RAG retrieval")
            docs = self._retriever.invoke(question)
            if docs:
                context = "\n\n".join(doc.page_content[:500] for doc in docs)
                return {
                    "answer": f"Based on the retrieved documents:\n\n{context}",
                    "tool_used": "fallback_rag",
                    "intermediate_steps": [],
                }
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "tool_used": "none",
                "intermediate_steps": [],
            }
