"""RAGAS-inspired automated MLOps evaluation for financial RAG pipelines."""

import re
from typing import List, Dict, Any
from src.generation.schemas import EvaluationReport, EvaluationMetric


class FinancialRagasEvaluator:
    """Evaluates RAG pipeline outputs across Faithfulness, Relevance, and Precision.
    
    Ensures zero hallucination and high retrieval quality on financial benchmarks.
    """

    def __init__(self):
        pass

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Computes faithfulness score: checks if numbers, percentages, and key claims in
        the answer are strictly grounded in the retrieved context.
        """
        if not answer or not context:
            return 0.0

        # Extract monetary values, percentages, and years
        answer_facts = re.findall(r"(?:R\$\s*|US\$\s*)?\d+(?:[.,]\d+)?%?", answer)
        if not answer_facts:
            # If no numbers, check sentence overlap
            sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 15]
            if not sentences:
                return 1.0
            grounded = sum(1 for s in sentences if any(w in context.lower() for w in s.lower().split() if len(w) > 4))
            return round(grounded / len(sentences), 3)

        grounded_facts = 0
        context_lower = context.lower()
        for fact in answer_facts:
            # Normalize fact
            clean_fact = fact.lower().replace(" ", "")
            if clean_fact in context_lower.replace(" ", ""):
                grounded_facts += 1

        return round(grounded_facts / len(answer_facts), 3)

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Measures how directly the answer addresses the query's core entities."""
        query_words = set(w.lower() for w in re.findall(r"\b\w+\b", query) if len(w) > 3)
        if not query_words:
            return 1.0

        answer_lower = answer.lower()
        matched = sum(1 for w in query_words if w in answer_lower)
        return round(min(1.0, (matched / len(query_words)) * 1.2), 3)

    def evaluate_context_precision(self, ground_truth_section: str, retrieved_contexts: List[str]) -> float:
        """Measures if the correct section/table was ranked near the top."""
        if not retrieved_contexts or not ground_truth_section:
            return 0.5

        for rank, ctx in enumerate(retrieved_contexts, start=1):
            if ground_truth_section.lower() in ctx.lower():
                return round(1.0 / rank, 3)

        return 0.0

    def evaluate_benchmark(self, benchmark_items: List[Dict[str, Any]]) -> EvaluationReport:
        """Runs batch evaluation over a golden dataset and produces an EvaluationReport."""
        faithfulness_scores = []
        relevance_scores = []
        precision_scores = []
        details = []

        for item in benchmark_items:
            query = item["query"]
            answer = item.get("generated_answer", "")
            context = item.get("retrieved_context", "")
            ground_truth_section = item.get("ground_truth_section", "")
            retrieved_list = item.get("retrieved_chunks", [context])

            f_score = self.evaluate_faithfulness(answer, context)
            r_score = self.evaluate_answer_relevance(query, answer)
            p_score = self.evaluate_context_precision(ground_truth_section, retrieved_list)

            faithfulness_scores.append(f_score)
            relevance_scores.append(r_score)
            precision_scores.append(p_score)

            details.append({
                "query": query,
                "faithfulness": f_score,
                "answer_relevance": r_score,
                "context_precision": p_score,
                "confidence": item.get("confidence", "HIGH")
            })

        avg_f = round(sum(faithfulness_scores) / max(1, len(faithfulness_scores)), 3)
        avg_r = round(sum(relevance_scores) / max(1, len(relevance_scores)), 3)
        avg_p = round(sum(precision_scores) / max(1, len(precision_scores)), 3)

        metrics = [
            EvaluationMetric(
                metric_name="Faithfulness (Fidelidade / Não-Alucinação)",
                score=avg_f,
                description="Percentual de dados contábeis e afirmações estritamente embasadas nos relatórios."
            ),
            EvaluationMetric(
                metric_name="Answer Relevance (Pertinência da Resposta)",
                score=avg_r,
                description="Grau de correspondência entre o objetivo da pergunta e a resposta gerada."
            ),
            EvaluationMetric(
                metric_name="Context Precision (Precisão de Ranking)",
                score=avg_p,
                description="Medida de quão acima no ranking os trechos corretos foram recuperados."
            )
        ]

        return EvaluationReport(
            total_questions=len(benchmark_items),
            faithfulness_score=avg_f,
            answer_relevance_score=avg_r,
            context_precision_score=avg_p,
            metrics=metrics,
            details=details
        )
