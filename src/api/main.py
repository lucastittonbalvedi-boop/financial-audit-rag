"""FastAPI enterprise server for Financial Audit Hybrid RAG."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.generation.schemas import (
    AuditResponse,
    AuditQueryRequest,
    IngestionResponse,
    EvaluationReport,
    DocumentChunk,
)
from src.ingestion.pdf_parser import FinancialPDFParser
from src.ingestion.chunker import FinancialChunker
from src.retrieval.hybrid import HybridFinancialRetriever
from src.generation.generator import FinancialAuditGenerator
from src.evaluation.ragas_evaluator import FinancialRagasEvaluator

app = FastAPI(
    title="Financial Compliance & Audit Hybrid RAG API",
    description="Enterprise-grade Hybrid RAG for corporate financial statements (CVM/SEC), table preservation and auditable citations.",
    version=settings.app_version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services singleton
parser = FinancialPDFParser()
chunker = FinancialChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
retriever = HybridFinancialRetriever()
generator = FinancialAuditGenerator()
evaluator = FinancialRagasEvaluator()

# In-memory document registry
INGESTED_DOCS: Dict[str, Dict[str, Any]] = {}
ALL_CHUNKS: List[DocumentChunk] = []


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint confirming API status and model availability."""
    has_api_key = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "chroma_chunks_count": retriever.vector_store.collection.count(),
        "total_documents_indexed": len(INGESTED_DOCS),
        "llm_provider": "Google Gemini (Active)" if has_api_key else "Deterministic Local Fallback",
        "hybrid_retrieval_active": True
    }


@app.post("/api/v1/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """Uploads, parses with table preservation, chunks, and indexes a financial PDF or text report."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome de arquivo inválido.")

    file_path = settings.sample_reports_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse based on extension
        if file.filename.lower().endswith(".pdf"):
            parsed_doc = parser.parse_pdf(file_path)
        else:
            parsed_doc = parser.parse_text_or_markdown(file_path)

        # Chunk preserving tables
        optimized_chunks = chunker.chunk_document(parsed_doc.chunks)

        # Index in both lexical (BM25) and dense (ChromaDB) indices
        retriever.index_document_chunks(optimized_chunks)

        ALL_CHUNKS.extend(optimized_chunks)
        INGESTED_DOCS[file.filename] = {
            "document_name": file.filename,
            "total_pages": parsed_doc.total_pages,
            "tables_found": parsed_doc.tables_found,
            "chunks_count": len(optimized_chunks),
            "file_path": str(file_path)
        }

        return IngestionResponse(
            status="SUCCESS",
            document_name=file.filename,
            total_pages=parsed_doc.total_pages,
            chunks_created=len(optimized_chunks),
            tables_extracted=parsed_doc.tables_found,
            message=f"Documento '{file.filename}' processado com sucesso. {parsed_doc.tables_found} tabelas e {len(optimized_chunks)} chunks indexados."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar documento: {str(e)}")


@app.post("/api/v1/audit/query", response_model=AuditResponse, tags=["Audit Query"])
def query_audit_copilot(request: AuditQueryRequest):
    """Executes hybrid retrieval (BM25 + Dense + RRF + Re-ranker) and generates an auditable compliance answer."""
    try:
        # Retrieve candidates via Hybrid search
        candidates = retriever.search(
            query=request.query,
            top_k=request.top_k,
            document_filter=request.document_filter,
            enable_reranker=request.enable_reranker
        )

        # Generate response with strict citation requirements
        audit_response = generator.generate_audit_response(
            query=request.query,
            retrieved_chunks=candidates
        )

        return audit_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a auditoria: {str(e)}")


@app.get("/api/v1/documents", tags=["Documents"])
def list_documents():
    """Lists all financial documents currently indexed in the hybrid store."""
    return {
        "total_documents": len(INGESTED_DOCS),
        "documents": list(INGESTED_DOCS.values())
    }


@app.post("/api/v1/evaluate", response_model=EvaluationReport, tags=["MLOps"])
def run_evaluation_benchmark():
    """Runs automated RAGAS MLOps evaluation over golden benchmark questions."""
    import json
    golden_path = settings.evaluation_dir / "golden_dataset.json"
    if not golden_path.exists():
        raise HTTPException(status_code=404, detail="Dataset de benchmark (golden_dataset.json) não encontrado.")

    with open(golden_path, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    # Run each item through the pipeline
    evaluation_payload = []
    for item in benchmark_items:
        query = item["query"]
        candidates = retriever.search(query=query, top_k=3)
        resp = generator.generate_audit_response(query=query, retrieved_chunks=candidates)
        
        context_str = " ".join([c[0].content for c in candidates])
        evaluation_payload.append({
            "query": query,
            "generated_answer": resp.answer,
            "retrieved_context": context_str,
            "ground_truth_section": item.get("expected_section", ""),
            "retrieved_chunks": [c[0].content for c in candidates],
            "confidence": resp.confidence
        })

    report = evaluator.evaluate_benchmark(evaluation_payload)
    return report
