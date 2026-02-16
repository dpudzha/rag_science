"""Experiment orchestration for RAG pipeline parameter sweeps."""
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict]:
    """Load golden Q&A dataset from JSON file.

    Each row typically includes:
    - question
    - ideal_answer (for answer-level metrics)
    - expected_sources (source-level retrieval labels)
    - expected_chunk_ids (optional chunk-level retrieval labels)
    """
    return json.loads(path.read_text())


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    name: str
    config: dict
    metrics: dict
    per_question: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_series(self) -> pd.Series:
        """Convert to a pandas Series for easy DataFrame construction."""
        data = {"experiment": self.name, "duration_s": self.duration_seconds}
        data.update(self.config)
        data.update(self.metrics)
        return pd.Series(data)


def _apply_config_overrides(overrides: dict) -> dict[str, Any]:
    """Monkey-patch config module with overrides. Returns original values for restoration."""
    import config

    originals = {}
    for key, value in overrides.items():
        if hasattr(config, key):
            originals[key] = getattr(config, key)
            setattr(config, key, value)
            logger.debug("Config override: %s = %r (was %r)", key, value, originals[key])
        else:
            logger.warning("Unknown config key: %s", key)
    return originals


def _restore_config(originals: dict):
    """Restore original config values after an experiment."""
    import config

    for key, value in originals.items():
        setattr(config, key, value)


def _needs_reingestion(overrides: dict) -> bool:
    """Check if config overrides require re-ingesting the vectorstore."""
    reingest_keys = {"CHUNK_SIZE", "CHUNK_OVERLAP", "EMBEDDING_MODEL", "CHILD_CHUNK_SIZE", "CHILD_CHUNK_OVERLAP"}
    return bool(reingest_keys & set(overrides))


def run_experiment(
    name: str,
    config_overrides: dict,
    golden_dataset: list[dict] | None = None,
    reingest: bool | None = None,
    compute_ragas: bool = False,
    ragas_judge: str = "local",
) -> ExperimentResult:
    """Run a single experiment with given config overrides.

    Args:
        name: Human-readable experiment name.
        config_overrides: Dict of config.py attribute names to override values.
        golden_dataset: List of Q&A dicts. Loaded from file if None.
        reingest: Whether to re-ingest PDFs. Auto-detected from overrides if None.
        compute_ragas: Whether to compute RAGAS metrics (slower).
        ragas_judge: "local" or "openai" for RAGAS judge LLM.

    Returns:
        ExperimentResult with metrics and per-question details.
    """
    import importlib

    if golden_dataset is None:
        golden_dataset = load_golden_dataset()

    # Apply config overrides
    originals = _apply_config_overrides(config_overrides)

    try:
        # Re-import modules that cache config values at import time
        import config
        import ingest as ingest_mod
        import query as query_mod
        importlib.reload(ingest_mod)
        importlib.reload(query_mod)

        if reingest is None:
            reingest = _needs_reingestion(config_overrides)

        start = time.time()

        # Re-ingest if needed
        if reingest:
            logger.info("[%s] Re-ingesting with overrides: %s", name, config_overrides)
            ingest_mod.ingest()

        # Load retriever and QA chain
        vs = query_mod.load_vectorstore()
        retriever = query_mod.build_retriever(vs)

        # Override retriever params that don't require reingestion
        if "TOP_K" in config_overrides:
            retriever.k = config_overrides["TOP_K"]
        if "TOP_K_CANDIDATES" in config_overrides:
            retriever.k_candidates = config_overrides["TOP_K_CANDIDATES"]
        if "BM25_WEIGHT" in config_overrides:
            retriever.bm25_weight = config_overrides["BM25_WEIGHT"]
        if "DENSE_WEIGHT" in config_overrides:
            retriever.dense_weight = config_overrides["DENSE_WEIGHT"]

        qa = query_mod.build_qa_chain(retriever)

        # Run evaluation
        from eval.evaluate import (
            reciprocal_rank,
            recall_at_k,
            reciprocal_rank_chunks,
            precision_at_k_chunks,
            recall_at_k_chunks,
            extract_retrieved_chunk_ids,
        )

        per_question = []
        mrr_sum = 0.0
        recall_sum = 0.0
        chunk_mrr_sum = 0.0
        chunk_precision_sum = 0.0
        chunk_recall_sum = 0.0
        chunk_labeled_n = 0
        questions_list = []
        answers_list = []
        contexts_list = []
        ground_truths_list = []

        for item in golden_dataset:
            question = item["question"]
            expected = item.get("expected_sources", [])
            expected_chunk_ids = item.get("expected_chunk_ids", [])
            ideal = item.get("ideal_answer", "")

            result = qa.invoke({"question": question, "chat_history": []})
            answer = result["answer"]
            source_docs = result["source_documents"]

            mrr = reciprocal_rank(expected, source_docs)
            recall = recall_at_k(expected, source_docs, retriever.k)
            retrieved_chunk_ids = extract_retrieved_chunk_ids(source_docs)

            mrr_sum += mrr
            recall_sum += recall

            per_q = {
                "question": question,
                "answer": answer,
                "mrr": mrr,
                f"recall@{retriever.k}": recall,
                "sources": [d.metadata.get("source", "?") for d in source_docs],
                "retrieved_chunk_ids": retrieved_chunk_ids,
            }

            if expected_chunk_ids:
                chunk_labeled_n += 1
                chunk_mrr = reciprocal_rank_chunks(expected_chunk_ids, source_docs)
                chunk_precision = precision_at_k_chunks(expected_chunk_ids, source_docs, retriever.k)
                chunk_recall = recall_at_k_chunks(expected_chunk_ids, source_docs, retriever.k)

                chunk_mrr_sum += chunk_mrr
                chunk_precision_sum += chunk_precision
                chunk_recall_sum += chunk_recall

                per_q["chunk_mrr"] = chunk_mrr
                per_q[f"chunk_precision@{retriever.k}"] = chunk_precision
                per_q[f"chunk_recall@{retriever.k}"] = chunk_recall

            per_question.append(per_q)

            # Collect data for RAGAS
            questions_list.append(question)
            answers_list.append(answer)
            contexts_list.append([d.page_content for d in source_docs])
            ground_truths_list.append(ideal)

        n = len(golden_dataset)
        metrics = {
            "mrr": mrr_sum / n if n else 0,
            f"recall@{retriever.k}": recall_sum / n if n else 0,
        }
        if chunk_labeled_n:
            metrics["chunk_mrr"] = chunk_mrr_sum / chunk_labeled_n
            metrics[f"chunk_precision@{retriever.k}"] = chunk_precision_sum / chunk_labeled_n
            metrics[f"chunk_recall@{retriever.k}"] = chunk_recall_sum / chunk_labeled_n
            metrics["chunk_labeled_queries"] = chunk_labeled_n

        # Optionally compute RAGAS metrics
        if compute_ragas:
            from eval.ragas_metrics import build_ragas_dataset, run_ragas_eval

            ragas_ds = build_ragas_dataset(questions_list, answers_list, contexts_list, ground_truths_list)
            ragas_result = run_ragas_eval(ragas_ds, judge=ragas_judge)
            metrics.update(ragas_result.to_dict())

        duration = time.time() - start

        logger.info("[%s] Done in %.1fs — MRR=%.3f Recall@%d=%.3f",
                     name, duration, metrics["mrr"], retriever.k, metrics[f"recall@{retriever.k}"])

        return ExperimentResult(
            name=name,
            config=config_overrides,
            metrics=metrics,
            per_question=per_question,
            duration_seconds=duration,
        )
    finally:
        _restore_config(originals)


def compare_experiments(results: list[ExperimentResult]) -> pd.DataFrame:
    """Create a comparison DataFrame from multiple experiment results."""
    rows = [r.to_series() for r in results]
    df = pd.DataFrame(rows)
    if "experiment" in df.columns:
        df = df.set_index("experiment")
    return df


def plot_comparison(df: pd.DataFrame, metrics: list[str] | None = None, title: str = "Experiment Comparison"):
    """Create bar charts comparing experiments across metrics.

    Args:
        df: DataFrame from compare_experiments().
        metrics: List of metric column names to plot. Auto-detected if None.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = [c for c in df.columns if c in {
            "mrr", "recall@2", "recall@4", "recall@6", "recall@8",
            "context_precision", "context_recall", "answer_relevancy", "faithfulness",
        }]

    if not metrics:
        logger.warning("No metrics columns found to plot")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in df.columns:
            df[metric].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=45)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_radar(df: pd.DataFrame, metrics: list[str] | None = None, title: str = "Multi-Metric Comparison"):
    """Create a radar chart comparing experiments across multiple metrics.

    Args:
        df: DataFrame from compare_experiments().
        metrics: Metric columns to include. Auto-detected if None.
        title: Plot title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics is None:
        metrics = [c for c in df.columns if c in {
            "mrr", "recall@4", "context_precision", "context_recall",
            "answer_relevancy", "faithfulness",
        }]

    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        logger.warning("Need at least 3 metrics for radar chart, got %d", len(available))
        return

    n_metrics = len(available)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for exp_name in df.index:
        values = df.loc[exp_name, available].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=exp_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in available])
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig
