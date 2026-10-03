"""Data schemas and Pydantic models for the Financial Audit RAG system."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Auditable citation pointing to the exact source in the financial report."""
    document_name: str = Field(description="Nome do documento/relatório contábil.")
    page_number: int = Field(description="Número exato da página onde o trecho se encontra.")
    section: str = Field(description="Título da seção ou Nota Explicativa correspondente.")
    verbatim_quote: str = Field(description="Trecho exato do documento citado como evidência.")


class AuditResponse(BaseModel):
    """Structured response from the Financial Auditor Copilot."""
    query: str = Field(description="Pergunta realizada pelo auditor/analista.")
    answer: str = Field(description="Resposta sintetizada, clara e técnica com embasamento contábil.")
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Nível de confiança baseado na evidência encontrada no contexto."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Lista de evidências com citação de documento, página e trecho literal."
    )
    financial_metrics_detected: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métricas financeiras extraídas (ex: EBITDA, Lucro Líquido, Margem, Variação %)."
    )
    retrieval_strategy_used: str = Field(
        default="hybrid_rrf_rerank",
        description="Estratégia de recuperação utilizada."
    )


class DocumentChunk(BaseModel):
    """A granular chunk of financial text or table with rich metadata."""
    chunk_id: str
    document_name: str
    page_number: int
    section: str = "Geral"
    chunk_type: Literal["narrative", "table", "mixed"] = "narrative"
    content: str
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Structured representation of an ingested financial report."""
    document_name: str
    total_pages: int
    tables_found: int
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditQueryRequest(BaseModel):
    """API query request payload."""
    query: str
    document_filter: Optional[str] = None
    top_k: int = 5
    enable_reranker: bool = True


class IngestionResponse(BaseModel):
    """Response payload after ingesting a financial report."""
    status: str
    document_name: str
    total_pages: int
    chunks_created: int
    tables_extracted: int
    message: str


class EvaluationMetric(BaseModel):
    """Individual metric calculated during MLOps evaluation."""
    metric_name: str
    score: float
    description: str


class EvaluationReport(BaseModel):
    """Consolidated RAGAS / MLOps benchmark report."""
    total_questions: int
    faithfulness_score: float
    answer_relevance_score: float
    context_precision_score: float
    metrics: List[EvaluationMetric]
    details: List[Dict[str, Any]]
