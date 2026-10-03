"""Re-ranker for fine-grained ranking of candidate financial chunks."""

import re
from typing import List, Tuple
from src.generation.schemas import DocumentChunk


class FinancialReranker:
    """Specialized re-ranker that evaluates candidates by combining query-chunk term density,
    numeric entity alignment (e.g. years, percentages, note numbers), and table priority.
    """

    def __init__(self):
        pass

    def rerank(
        self, 
        query: str, 
        candidates: List[Tuple[DocumentChunk, float]], 
        top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Re-ranks retrieved candidates using financial heuristic scoring and contextual density."""
        if not candidates:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w+\b", query_lower))
        query_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", query_lower))

        scored_candidates: List[Tuple[DocumentChunk, float]] = []

        for chunk, initial_score in candidates:
            content_lower = chunk.content.lower()
            content_words = set(re.findall(r"\b\w+\b", content_lower))
            content_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", content_lower))

            # 1. Base score from previous retrieval (RRF or hybrid)
            score = initial_score

            # 2. Financial numbers and dates alignment bonus
            if query_numbers:
                matched_numbers = query_numbers.intersection(content_numbers)
                score += 0.25 * (len(matched_numbers) / len(query_numbers))

            # 3. Exact phrase match bonus
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 0.05

            # 4. Table bonus if the query asks for tabular / metric data
            metric_keywords = ["tabela", "ebitda", "balanço", "dre", "receita", "lucro", "passivo", "ativo", "custo", "margem", "quanto", "valor"]
            if any(kw in query_lower for kw in metric_keywords) and chunk.chunk_type == "table":
                score += 0.20

            # 5. Section match bonus
            if any(w in chunk.section.lower() for w in query_words if len(w) > 3):
                score += 0.15

            scored_candidates.append((chunk, round(score, 4)))

        # Sort descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]
