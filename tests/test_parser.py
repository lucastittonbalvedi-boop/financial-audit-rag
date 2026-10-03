"""Unit tests for the financial PDF and table parser."""

import pytest
from pathlib import Path

from src.ingestion.pdf_parser import FinancialPDFParser, format_table_to_markdown
from src.config import settings


def test_format_table_to_markdown():
    raw_table = [
        ["Conta", "2023", "2022"],
        ["Receita Líquida", "100.000", "80.000"],
        ["Lucro Líquido", "25.000", "20.000"],
    ]
    md = format_table_to_markdown(raw_table)
    assert "| Conta | 2023 | 2022 |" in md
    assert "| Receita Líquida | 100.000 | 80.000 |" in md
    assert md.count("\n") == 3  # Header + separator + 2 data rows


def test_parse_sample_pdf():
    pdf_path = settings.sample_reports_dir / "petrobras_dfp_2023_sample.pdf"
    assert pdf_path.exists(), "Sample PDF must exist for this test."

    parser = FinancialPDFParser()
    doc = parser.parse_pdf(pdf_path)

    assert doc.document_name == "petrobras_dfp_2023_sample.pdf"
    assert doc.total_pages == 3
    assert doc.tables_found >= 2
    assert len(doc.chunks) > 0

    # Verify table chunk content
    table_chunks = [c for c in doc.chunks if c.chunk_type == "table"]
    assert len(table_chunks) >= 2
    
    # Check that financial numbers are preserved inside markdown tables
    combined_tables = " ".join(t.content for t in table_chunks)
    assert "511.994" in combined_tables
    assert "63.140" in combined_tables


def test_parse_sample_markdown():
    md_path = settings.sample_reports_dir / "petrobras_dfp_2023_sample.md"
    assert md_path.exists(), "Sample MD must exist for this test."

    parser = FinancialPDFParser()
    doc = parser.parse_text_or_markdown(md_path)

    assert doc.total_pages == 5
    assert doc.tables_found >= 3
    assert any("Nota Explicativa 14" in c.section for c in doc.chunks)
