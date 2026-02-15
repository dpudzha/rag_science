"""RAGAS-based evaluation metrics for RAG pipeline quality assessment."""
import logging
from dataclasses import dataclass

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """Container for RAGAS evaluation results."""
    context_precision: float
    context_recall: float
    answer_relevancy: float
    faithfulness: float

    def to_dict(self) -> dict:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_relevancy": self.answer_relevancy,
            "faithfulness": self.faithfulness,
        }


def _build_ollama_llm(model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
    """Create a LangChain ChatOllama instance for RAGAS judge."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def _build_ollama_embeddings(model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
    """Create Ollama embeddings for RAGAS."""
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model, base_url=base_url)


def _build_openai_llm(model: str = "gpt-4o"):
    """Create an OpenAI LLM for RAGAS judge."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0)


def _build_openai_embeddings(model: str = "text-embedding-3-small"):
    """Create OpenAI embeddings for RAGAS."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model)


def build_ragas_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> Dataset:
    """Build a HuggingFace Dataset formatted for RAGAS evaluation.

    Args:
        questions: List of input questions.
        answers: List of generated answers.
        contexts: List of lists of retrieved context strings.
        ground_truths: List of ideal/reference answers.

    Returns:
        HuggingFace Dataset ready for RAGAS evaluate().
    """
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_ragas_eval(
    dataset: Dataset,
    judge: str = "local",
    ollama_model: str = "llama3.1:8b",
    ollama_base_url: str = "http://localhost:11434",
    openai_model: str = "gpt-4o",
) -> RAGASResult:
    """Run RAGAS evaluation on a dataset.

    Args:
        dataset: HuggingFace Dataset with question, answer, contexts, ground_truth columns.
        judge: "local" for Ollama or "openai" for OpenAI as the judge LLM.
        ollama_model: Ollama model name when judge="local".
        ollama_base_url: Ollama API URL when judge="local".
        openai_model: OpenAI model name when judge="openai".

    Returns:
        RAGASResult with all four metric scores.
    """
    metrics = [context_precision, context_recall, answer_relevancy, faithfulness]

    if judge == "local":
        logger.info("Using Ollama (%s) as RAGAS judge", ollama_model)
        llm = _build_ollama_llm(ollama_model, ollama_base_url)
        embeddings = _build_ollama_embeddings(base_url=ollama_base_url)
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    elif judge == "openai":
        logger.info("Using OpenAI (%s) as RAGAS judge", openai_model)
        llm = _build_openai_llm(openai_model)
        embeddings = _build_openai_embeddings()
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    else:
        raise ValueError(f"Unknown judge type: {judge!r}. Use 'local' or 'openai'.")

    return RAGASResult(
        context_precision=result["context_precision"],
        context_recall=result["context_recall"],
        answer_relevancy=result["answer_relevancy"],
        faithfulness=result["faithfulness"],
    )
