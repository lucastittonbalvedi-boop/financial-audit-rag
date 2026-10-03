"""Hybrid Retriever combining BM25 (Lexical) and ChromaDB (Dense) via RRF (Reciprocal Rank Fusion)."""

from typing import List, Tuple, Optional, Dict
from src.config import settings
from src.generation.schemas import DocumentChunk
from src.retrieval.bm25_retriever import BM25FinancialRetriever
from src.retrieval.vector_store import VectorStoreRetriever
from src.retrieval.reranker import FinancialReranker


class HybridFinancialRetriever:
    """Enterprise-grade Hybrid Retriever implementing Reciprocal Rank Fusion (RRF) and Re-ranking."""

    def __init__(
        self,
        bm25: Optional[BM25FinancialRetriever] = None,
        vector_store: Optional[VectorStoreRetriever] = None,
        reranker: Optional[FinancialReranker] = None,
        rrf_k: int = 60
    ):
        self.bm25 = bm25 or BM25FinancialRetriever()
        self.vector_store = vector_store or VectorStoreRetriever()
        self.reranker = reranker or FinancialReranker()
        self.rrf_k = rrf_k or settings.rrf_k

    def index_document_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Indexes chunks in both lexical and dense vector indices simultaneously."""
        self.bm25.index_chunks(chunks)
        self.vector_store.index_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[str] = None,
        dense_limit: int = 15,
        sparse_limit: int = 15,
        enable_reranker: bool = True
    ) -> List[Tuple[DocumentChunk, float]]:
        """Executes Hybrid Search with Reciprocal Rank Fusion and optional Re-ranking."""
        # 1. Retrieve dense candidates
        dense_results = self.vector_store.search(
            query=query,
            top_k=dense_limit,
            document_filter=document_filter
        )

        # 2. Retrieve sparse BM25 candidates
        sparse_results = self.bm25.search(
            query=query,
            top_k=sparse_limit
        )
        if document_filter:
            sparse_results = [r for r in sparse_results if r[0].document_name == document_filter]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF score: sum(1 / (k + rank))
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Add dense ranks
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Add sparse ranks
        for rank, (chunk, _) in enumerate(sparse_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort candidate list by fused RRF score
        fused_candidates = [
            (chunk_map[c_id], score)
            for c_id, score in rrf_scores.items()
        ]
        fused_candidates.sort(key=lambda x: x[1], reverse=True)

        # 4. Optional fine-grained re-ranking
        if enable_reranker and fused_candidates:
            final_results = self.reranker.rerank(
                query=query,
                candidates=fused_candidates[:dense_limit],
                top_k=top_k
            )
            return final_results

        return fused_candidates[:top_k]
