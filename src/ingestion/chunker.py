"""Financial-aware chunker that preserves table integrity and section boundaries."""

import re
from typing import List
from src.generation.schemas import DocumentChunk


class FinancialChunker:
    """Specialized chunker for financial and audit documents.
    
    Ensures markdown tables remain atomic (or split cleanly by rows with repeated headers)
    and segments narrative text at natural paragraph/sentence boundaries.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_narrative_text(self, text: str) -> List[str]:
        """Splits narrative text into chunks respecting paragraphs and sentences."""
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_len + para_len <= self.chunk_size:
                current_chunk.append(para)
                current_len += para_len + 2
            else:
                # If current chunk has content, finalize it
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # Keep some overlap if possible
                    overlap_para = current_chunk[-1] if len(current_chunk[-1]) <= self.chunk_overlap else ""
                    current_chunk = [overlap_para, para] if overlap_para else [para]
                    current_len = sum(len(p) for p in current_chunk) + 2
                else:
                    # Paragraph is longer than chunk_size, split by sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    s_chunk: List[str] = []
                    s_len = 0
                    for s in sentences:
                        if s_len + len(s) <= self.chunk_size:
                            s_chunk.append(s)
                            s_len += len(s) + 1
                        else:
                            if s_chunk:
                                chunks.append(" ".join(s_chunk))
                            s_chunk = [s]
                            s_len = len(s)
                    if s_chunk:
                        current_chunk = s_chunk
                        current_len = s_len

        if current_chunk:
            final_text = "\n\n".join(current_chunk).strip()
            if final_text and (not chunks or final_text != chunks[-1]):
                chunks.append(final_text)

        return chunks if chunks else [text]

    def _split_oversized_table(self, table_content: str) -> List[str]:
        """Splits large markdown tables by rows, retaining the header row in every slice."""
        lines = table_content.strip().split("\n")
        if len(lines) <= 4:
            return [table_content]

        # Extract title (if any) and header rows (header + divider)
        title_lines = [l for l in lines if l.startswith("#")]
        table_lines = [l for l in lines if not l.startswith("#") and l.strip()]

        if len(table_lines) < 2:
            return [table_content]

        header = table_lines[0]
        divider = table_lines[1]
        data_rows = table_lines[2:]

        title_prefix = ("\n".join(title_lines) + "\n\n") if title_lines else ""
        header_block = f"{title_prefix}{header}\n{divider}\n"

        slices: List[str] = []
        current_rows: List[str] = []
        current_size = len(header_block)

        for row in data_rows:
            row_size = len(row) + 1
            if current_size + row_size > self.chunk_size and current_rows:
                slices.append(header_block + "\n".join(current_rows))
                current_rows = [row]
                current_size = len(header_block) + row_size
            else:
                current_rows.append(row)
                current_size += row_size

        if current_rows:
            slices.append(header_block + "\n".join(current_rows))

        return slices if slices else [table_content]

    def chunk_document(self, raw_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Processes raw chunks from the parser into optimal sized retrieval chunks."""
        optimized_chunks: List[DocumentChunk] = []

        for chunk in raw_chunks:
            if chunk.chunk_type == "table":
                # For tables: Keep atomic if under threshold, else split by rows keeping header
                if len(chunk.content) > self.chunk_size * 1.5:
                    table_slices = self._split_oversized_table(chunk.content)
                    for idx, slice_content in enumerate(table_slices):
                        sub_chunk = DocumentChunk(
                            chunk_id=f"{chunk.chunk_id}_part{idx+1}",
                            document_name=chunk.document_name,
                            page_number=chunk.page_number,
                            section=chunk.section,
                            chunk_type="table",
                            content=slice_content,
                            metadata={**chunk.metadata, "is_table_slice": True, "part": idx + 1}
                        )
                        optimized_chunks.append(sub_chunk)
                else:
                    optimized_chunks.append(chunk)

            else:
                # Narrative text: split carefully
                if len(chunk.content) > self.chunk_size:
                    text_parts = self._split_narrative_text(chunk.content)
                    for idx, part in enumerate(text_parts):
                        sub_chunk = DocumentChunk(
                            chunk_id=f"{chunk.chunk_id}_part{idx+1}",
                            document_name=chunk.document_name,
                            page_number=chunk.page_number,
                            section=chunk.section,
                            chunk_type="narrative",
                            content=part,
                            metadata={**chunk.metadata, "part": idx + 1}
                        )
                        optimized_chunks.append(sub_chunk)
                else:
                    optimized_chunks.append(chunk)

        return optimized_chunks
