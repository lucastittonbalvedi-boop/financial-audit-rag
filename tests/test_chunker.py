"""Unit tests for the financial-aware chunker."""

import pytest
from src.ingestion.chunker import FinancialChunker
from src.generation.schemas import DocumentChunk


def test_chunker_keeps_small_tables_atomic():
    chunker = FinancialChunker(chunk_size=500, chunk_overlap=50)
    
    table_content = (
        "| Conta | 2023 | 2022 |\n"
        "| :--- | ---: | ---: |\n"
        "| Receita | 100 | 90 |\n"
        "| Lucro | 20 | 18 |"
    )
    raw_chunk = DocumentChunk(
        chunk_id="doc1_tab1",
        document_name="relatorio.pdf",
        page_number=1,
        section="DRE",
        chunk_type="table",
        content=table_content
    )

    optimized = chunker.chunk_document([raw_chunk])
    assert len(optimized) == 1
    assert optimized[0].content == table_content
    assert optimized[0].chunk_type == "table"


def test_chunker_splits_large_table_with_repeated_headers():
    chunker = FinancialChunker(chunk_size=120, chunk_overlap=20)
    
    large_table = (
        "| Ano | Linha 1 | Linha 2 |\n"
        "| :--- | ---: | ---: |\n"
        "| 2020 | Valor A1 | Valor B1 |\n"
        "| 2021 | Valor A2 | Valor B2 |\n"
        "| 2022 | Valor A3 | Valor B3 |\n"
        "| 2023 | Valor A4 | Valor B4 |\n"
        "| 2024 | Valor A5 | Valor B5 |"
    )
    raw_chunk = DocumentChunk(
        chunk_id="doc1_tab_large",
        document_name="relatorio.pdf",
        page_number=2,
        section="Balanço",
        chunk_type="table",
        content=large_table
    )

    slices = chunker.chunk_document([raw_chunk])
    assert len(slices) > 1
    for s in slices:
        # Every slice must contain the table header
        assert "| Ano | Linha 1 | Linha 2 |" in s.content
        assert s.chunk_type == "table"


def test_chunker_narrative_text_splits():
    chunker = FinancialChunker(chunk_size=150, chunk_overlap=30)
    
    long_text = (
        "A política de gerenciamento de risco da Companhia visa proteger os fluxos de caixa "
        "contra a volatilidade de mercado do petróleo e variações cambiais. "
        "A Companhia utiliza instrumentos derivativos exclusivamente com fins de hedge econômico, "
        "sendo vedada a realização de operações com caráter especulativo conforme aprovado pelo Conselho."
    )
    raw_chunk = DocumentChunk(
        chunk_id="doc1_nar",
        document_name="relatorio.pdf",
        page_number=3,
        section="Gestão de Riscos",
        chunk_type="narrative",
        content=long_text
    )

    chunks = chunker.chunk_document([raw_chunk])
    assert len(chunks) >= 2
    assert all(c.section == "Gestão de Riscos" for c in chunks)
    assert all(c.page_number == 3 for c in chunks)
