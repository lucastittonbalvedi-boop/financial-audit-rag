"""Dense vector retriever using ChromaDB with Gemini Embeddings and local fallback."""

import os
import hashlib
from typing import List, Tuple, Optional
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.generation.schemas import DocumentChunk


class VectorStoreRetriever:
    """Manages ChromaDB vector collection and semantic embedding search."""

    def __init__(self, collection_name: str = "financial_reports"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Financial reports chunk embeddings"}
        )
        self._gemini_client = None

    def _get_gemini_client(self):
        """Lazy initialization of the official Google GenAI client."""
        if self._gemini_client is None:
            api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                from google import genai
                self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates dense embeddings via Gemini API, or falls back to local deterministic vectors."""
        gemini_client = self._get_gemini_client()
        if gemini_client:
            try:
                # Use Gemini text-embedding-004
                response = gemini_client.models.embed_content(
                    model=settings.default_embedding_model,
                    contents=texts
                )
                if hasattr(response, "embeddings"):
                    return [list(e.values) for e in response.embeddings]
            except Exception as e:
                # Log and fallback gracefully
                print(f"[VectorStore] Erro ao chamar Gemini Embeddings ({e}). Usando fallback semântico local.")

        # Fallback deterministic dense vector representation (384 dimensions)
        return [self._local_semantic_embedding(t) for t in texts]

    def _local_semantic_embedding(self, text: str, dim: int = 384) -> List[float]:
        """High-entropy local pseudo-semantic vector representation for offline/testing mode.
        
        Uses n-gram hash projection with cosine normalization.
        """
        words = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        for i, word in enumerate(words):
            # Compute hash bucket
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            weight = 1.0 / (1.0 + 0.1 * i)
            vec[idx] += sign * weight

        # Normalize to unit sphere (cosine metric)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embeds and indexes document chunks into ChromaDB."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [
            {
                "document_name": c.document_name,
                "page_number": c.page_number,
                "section": c.section,
                "chunk_type": c.chunk_type
            }
            for c in chunks
        ]

        embeddings = self._generate_embeddings(texts)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(
        self, 
        query: str, 
        top_k: int = 10,
        document_filter: Optional[str] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Searches ChromaDB using cosine distance and returns DocumentChunks with similarity scores."""
        query_embeddings = self._generate_embeddings([query])
        
        where_filter = {"document_name": document_filter} if document_filter else None

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=min(top_k, max(1, self.collection.count())),
            where=where_filter
        )

        ranked: List[Tuple[DocumentChunk, float]] = []
        if results and results["ids"] and results["ids"][0]:
            chunk_ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(chunk_ids)

            for c_id, doc, meta, dist in zip(chunk_ids, documents, metadatas, distances):
                # Convert cosine distance to similarity score in [0, 1]
                similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 0.5
                chunk = DocumentChunk(
                    chunk_id=c_id,
                    document_name=meta.get("document_name", "Desconhecido"),
                    page_number=int(meta.get("page_number", 1)),
                    section=meta.get("section", "Geral"),
                    chunk_type=meta.get("chunk_type", "narrative"),
                    content=doc,
                    metadata=meta
                )
                ranked.append((chunk, similarity))

        return ranked

    def clear(self) -> None:
        """Clears all records in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Financial reports chunk embeddings"}
        )
