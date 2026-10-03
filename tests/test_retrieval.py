"""Unit tests for the BM25, VectorStore, and Hybrid RRF retrievers."""

import pytest
from src.generation.schemas import DocumentChunk
from src.retrieval.bm25_retriever import BM25FinancialRetriever, financial_tokenizer
from src.retrieval.vector_store import VectorStoreRetriever
from src.retrieval.reranker import FinancialReranker
from src.retrieval.hybrid import HybridFinancialRetriever


def test_financial_tokenizer():
    tokens = financial_tokenizer("A provisão da Nota 14 (CPC 25) totalizou R$ 63.140M (+10,7%).")
    assert "nota" in tokens
    assert "14" in tokens
    assert "cpc" in tokens
    assert "25" in tokens
    assert "r$" in tokens
    assert "%" in tokens


def test_bm25_exact_entity_retrieval():
    retriever = BM25FinancialRetriever()
    
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_name="dfp.pdf",
            page_number=1,
            section="DRE",
            content="A Receita Líquida de Vendas atingiu R$ 511.994 milhões no exercício de 2023."
        ),
        DocumentChunk(
            chunk_id="c2",
            document_name="dfp.pdf",
            page_number=2,
            section="Nota Explicativa 14",
            content="A Nota Explicativa 14 trata de Provisões para litígios tributários e cíveis conforme CPC 25."
        ),
        DocumentChunk(
            chunk_id="c3",
            document_name="dfp.pdf",
            page_number=3,
            section="Endividamento",
            content="O prazo médio de vencimento da dívida consolidada encerrou em 11,4 anos."
        )
    ]
    retriever.index_chunks(chunks)

    # Search exact accounting entity
    results = retriever.search("Nota Explicativa 14 CPC 25", top_k=2)
    assert len(results) > 0
    assert results[0][0].chunk_id == "c2"

    # Search exact number
    num_results = retriever.search("511.994 Receita Líquida", top_k=2)
    assert len(num_results) > 0
    assert num_results[0][0].chunk_id == "c1"


def test_hybrid_rrf_retrieval():
    hybrid = HybridFinancialRetriever()
    
    chunks = [
        DocumentChunk(
            chunk_id="h1",
            document_name="dfp.pdf",
            page_number=1,
            section="DRE",
            chunk_type="table",
            content="| Receita Líquida | 511.994 | 641.256 |"
        ),
        DocumentChunk(
            chunk_id="h2",
            document_name="dfp.pdf",
            page_number=2,
            section="Riscos",
            chunk_type="narrative",
            content="A Companhia mantém política rigorosa de proteção cambial e mitigação de volatilidade."
        )
    ]
    hybrid.index_document_chunks(chunks)

    results = hybrid.search(query="receita líquida 511.994", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "h1"
    assert "511.994" in top_chunk.content
