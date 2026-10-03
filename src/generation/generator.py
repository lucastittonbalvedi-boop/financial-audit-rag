"""Generation module: orchestrates LLM inference with strict citation extraction."""

import os
import json
import re
from typing import List, Tuple, Optional, Dict, Any

from src.config import settings
from src.generation.schemas import AuditResponse, Citation, DocumentChunk
from src.generation.prompts import AUDIT_SYSTEM_PROMPT, AUDIT_USER_PROMPT_TEMPLATE


class FinancialAuditGenerator:
    """Generates auditable, citation-backed compliance answers from retrieved chunks."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.default_llm_model
        self._gemini_client = None

    def _get_gemini_client(self):
        """Initializes Google GenAI client if API key is configured."""
        if self._gemini_client is None:
            api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                from google import genai
                self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def _format_context(self, chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Formats candidate chunks into an audit evidence block."""
        context_parts = []
        for rank, (chunk, score) in enumerate(chunks, start=1):
            part = (
                f"[EVIDÊNCIA #{rank}] Documento: {chunk.document_name} | "
                f"Página: {chunk.page_number} | "
                f"Seção: {chunk.section} | "
                f"Tipo: {chunk.chunk_type} | Relevância: {score:.3f}\n"
                f"{chunk.content}\n"
            )
            context_parts.append(part)
        return "\n----------------------------------------\n".join(context_parts)

    def _generate_offline_fallback(
        self, 
        query: str, 
        chunks: List[Tuple[DocumentChunk, float]]
    ) -> AuditResponse:
        """Rule-based deterministic synthesis when no LLM API key is present.
        
        Extracts key evidence, numbers, citations and builds a valid AuditResponse.
        """
        if not chunks:
            return AuditResponse(
                query=query,
                answer="Nenhuma evidência ou relatório financeiro relevante foi localizado para responder à consulta.",
                confidence="LOW",
                citations=[],
                financial_metrics_detected={},
                retrieval_strategy_used="hybrid_rrf_rerank"
            )

        top_chunk, top_score = chunks[0]
        citations: List[Citation] = []
        metrics: Dict[str, Any] = {}

        # Build citations from top chunks
        for chunk, score in chunks[:3]:
            # Extract first sentence or table row as verbatim quote
            lines = [l.strip() for l in chunk.content.split("\n") if l.strip() and not l.startswith("#")]
            verbatim = lines[0] if lines else chunk.content[:100]
            if len(verbatim) > 200:
                verbatim = verbatim[:197] + "..."

            citations.append(Citation(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                section=chunk.section,
                verbatim_quote=verbatim
            ))

        # Extract basic metrics found in the top chunks (e.g. R$ 123,45 or 15%)
        found_values = re.findall(r"(?:R\$\s*|US\$\s*)?(\d+(?:[.,]\d+)?\s*(?:milhões|bilhões|mil|%))", top_chunk.content)
        for i, val in enumerate(found_values[:4]):
            metrics[f"metrica_{i+1}"] = val

        answer = (
            f"Com base na análise das evidências em '{top_chunk.document_name}' "
            f"(Página {top_chunk.page_number}, {top_chunk.section}), verificou-se que os dados "
            f"apontam: {top_chunk.content[:250].strip()}... "
            f"Todas as citações de página e linha foram anexadas para fins de auditoria."
        )

        return AuditResponse(
            query=query,
            answer=answer,
            confidence="HIGH" if top_score > 0.5 else "MEDIUM",
            citations=citations,
            financial_metrics_detected=metrics,
            retrieval_strategy_used="hybrid_rrf_rerank"
        )

    def generate_audit_response(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]]
    ) -> AuditResponse:
        """Synthesizes the financial audit response with mandatory citations."""
        if not retrieved_chunks:
            return AuditResponse(
                query=query,
                answer="Nenhuma informação ou documento correspondente foi localizado no acervo contábil.",
                confidence="LOW",
                citations=[],
                financial_metrics_detected={},
                retrieval_strategy_used="hybrid_rrf_rerank"
            )

        client = self._get_gemini_client()
        if client:
            context = self._format_context(retrieved_chunks)
            user_prompt = AUDIT_USER_PROMPT_TEMPLATE.format(context=context, query=query)
            try:
                from google.genai import types
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=AUDIT_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                if response.text:
                    parsed_json = json.loads(response.text)
                    return AuditResponse(**parsed_json)
            except Exception as e:
                print(f"[Generator] Erro ao chamar LLM via API ({e}). Alternando para sintetizador offline.")

        # Offline fallback synthesis
        return self._generate_offline_fallback(query, retrieved_chunks)
