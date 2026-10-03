"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.config import settings

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["hybrid_retrieval_active"] is True


def test_ingest_markdown_file():
    sample_file = settings.sample_reports_dir / "petrobras_dfp_2023_sample.md"
    assert sample_file.exists()

    with open(sample_file, "rb") as f:
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("petrobras_dfp_2023_sample.md", f, "text/markdown")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["total_pages"] == 5
    assert data["tables_extracted"] >= 3
    assert data["chunks_created"] > 0


def test_audit_query_endpoint():
    query_payload = {
        "query": "Qual foi a receita líquida e o lucro líquido em 2023?",
        "top_k": 3,
        "enable_reranker": True
    }
    response = client.post("/api/v1/audit/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "confidence" in data
    assert data["confidence"] in ["HIGH", "MEDIUM", "LOW"]
    assert "citations" in data
    assert len(data["citations"]) > 0
    
    first_cit = data["citations"][0]
    assert "document_name" in first_cit
    assert "page_number" in first_cit
    assert "verbatim_quote" in first_cit


def test_list_documents_endpoint():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] >= 1


def test_evaluate_benchmark_endpoint():
    response = client.post("/api/v1/evaluate")
    assert response.status_code == 200
    data = response.json()
    assert "faithfulness_score" in data
    assert "answer_relevance_score" in data
    assert "context_precision_score" in data
    assert data["total_questions"] >= 1
