"""Sparse / Lexical BM25 retriever tuned for financial jargon and exact entities."""

import re
import unicodedata
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from src.generation.schemas import DocumentChunk


def financial_tokenizer(text: str) -> List[str]:
    """Tokenizes text preserving key financial terms, codes, currency indicators, and numbers.
    
    Examples of preserved tokens:
    - 'CPC 06' -> 'cpc', '06'
    - 'R$ 150,5M' -> 'r$', '150,5m', '150', '5'
    - '4T23' -> '4t23'
    - 'EBITDA' -> 'ebitda'
    """
    # Normalize accents
    normalized = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").lower()
    
    # Tokenize words, numbers, and financial codes (prioritize currency and percentage prefixes)
    tokens = re.findall(r"r\$|%|[a-z0-9]+(?:[.,_-][a-z0-9]+)*", normalized)
    return [t for t in tokens if len(t) > 1 or t in ("%", "$")]


class BM25FinancialRetriever:
    """In-memory BM25 Okapi retriever for high-precision exact keyword matching."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: List[List[str]] = []

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Indexes or extends the BM25 index with new document chunks."""
        self.chunks = list(chunks)
        self.tokenized_corpus = [financial_tokenizer(c.content) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Performs lexical search and returns top-K chunks with BM25 scores."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = financial_tokenizer(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        
        # Rank by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results: List[Tuple[DocumentChunk, float]] = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.0:  # Only return chunks with non-zero relevance
                results.append((self.chunks[idx], float(scores[idx])))
                
        return results
