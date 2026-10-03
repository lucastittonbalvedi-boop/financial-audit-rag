"""Table-aware financial document parser using pdfplumber."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber

from src.generation.schemas import ParsedDocument, DocumentChunk


def format_table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """Converts a raw list of table rows into a clean, aligned Markdown table.
    
    Preserves financial numbers, header alignments and handles None/empty cells.
    """
    if not table or not any(table):
        return ""
    
    # Clean cell values: remove excessive internal linebreaks and strip whitespace
    cleaned_rows: List[List[str]] = []
    for row in table:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("-")
            else:
                text = str(cell).replace("\n", " ").strip()
                cleaned_row.append(text if text else "-")
        # Only keep row if it has at least one non-empty value
        if any(c != "-" for c in cleaned_row):
            cleaned_rows.append(cleaned_row)
            
    if not cleaned_rows:
        return ""

    # Normalize row lengths
    max_cols = max(len(r) for r in cleaned_rows)
    for r in cleaned_rows:
        while len(r) < max_cols:
            r.append("-")
            
    header = cleaned_rows[0]
    separator = [":---" if i == 0 else "---:" for i in range(max_cols)]
    
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |"
    ]
    
    for row in cleaned_rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join(md_lines)


class FinancialPDFParser:
    """Parser specifically designed for financial statements (DFP, ITR, 10-K, Releases).
    
    Extracts text alongside formatted markdown tables, ensuring numbers and column structures
    are never mangled or dropped.
    """

    def __init__(self):
        # Common section indicators in Brazilian and International financial reports
        self.section_patterns = [
            re.compile(r"(Nota\s+Explicativa\s+\d+[\s\w–—-]*)", re.IGNORECASE),
            re.compile(r"(Balanço\s+Patrimonial[\s\w–—-]*)", re.IGNORECASE),
            re.compile(r"(Demonstração\s+do\s+Resultado[\s\w–—-]*)", re.IGNORECASE),
            re.compile(r"(Demonstração\s+dos\s+Fluxos\s+de\s+Caixa[\s\w–—-]*)", re.IGNORECASE),
            re.compile(r"(Relatório\s+da\s+Administração[\s\w–—-]*)", re.IGNORECASE),
            re.compile(r"(Parecer\s+dos\s+Auditores[\s\w–—-]*)", re.IGNORECASE),
            re.compile(r"(Comentário\s+do\s+Desempenho[\s\w–—-]*)", re.IGNORECASE),
        ]

    def _detect_section(self, text: str, current_section: str = "Geral") -> str:
        """Detects if page content introduces a new financial section or note."""
        for pattern in self.section_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return current_section

    def parse_pdf(self, file_path: Path | str) -> ParsedDocument:
        """Parses a financial PDF, extracting tables and narrative text page by page."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        document_name = path.name
        chunks: List[DocumentChunk] = []
        total_tables = 0
        current_section = "Relatório Contábil Geral"

        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                current_section = self._detect_section(raw_text, current_section)
                
                # 1. Extract tables
                tables = page.extract_tables()
                page_table_chunks: List[DocumentChunk] = []
                
                if tables:
                    for t_idx, table in enumerate(tables):
                        md_table = format_table_to_markdown(table)
                        if md_table:
                            total_tables += 1
                            table_chunk = DocumentChunk(
                                chunk_id=f"{document_name}_p{page_idx}_tab{t_idx+1}",
                                document_name=document_name,
                                page_number=page_idx,
                                section=current_section,
                                chunk_type="table",
                                content=f"### Tabela Contábil (Página {page_idx} - {current_section})\n\n{md_table}",
                                metadata={
                                    "table_index": t_idx + 1,
                                    "rows": len(table),
                                    "cols": len(table[0]) if table else 0
                                }
                            )
                            page_table_chunks.append(table_chunk)

                # 2. Extract narrative text
                # We clean up excessive spaces
                cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
                if cleaned_text:
                    narrative_chunk = DocumentChunk(
                        chunk_id=f"{document_name}_p{page_idx}_nar",
                        document_name=document_name,
                        page_number=page_idx,
                        section=current_section,
                        chunk_type="narrative",
                        content=cleaned_text,
                        metadata={"has_tables": bool(tables)}
                    )
                    chunks.append(narrative_chunk)

                # Append table chunks right after narrative so context is ordered
                chunks.extend(page_table_chunks)

        return ParsedDocument(
            document_name=document_name,
            total_pages=total_pages,
            tables_found=total_tables,
            chunks=chunks,
            metadata={"file_size_bytes": path.stat().st_size}
        )

    def parse_text_or_markdown(self, file_path: Path | str) -> ParsedDocument:
        """Parses a structured text or markdown financial report (useful for exports/APIs)."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        
        document_name = path.name
        # Split by page delimiter if present (e.g. <!-- PAGE 1 --> or form feeds \f)
        if "<!-- PAGE" in content:
            raw_pages = re.split(r"<!--\s*PAGE\s*(\d+)\s*-->", content)[1:]
            pages = []
            for i in range(0, len(raw_pages), 2):
                p_num = int(raw_pages[i])
                p_text = raw_pages[i+1].strip()
                pages.append((p_num, p_text))
        else:
            pages = [(1, content)]

        chunks: List[DocumentChunk] = []
        current_section = "Relatório Contábil Geral"
        total_tables = 0

        for p_num, p_text in pages:
            current_section = self._detect_section(p_text, current_section)
            
            # Check for markdown tables inside text
            table_matches = re.findall(r"(\|.+?\|\n\|[-:| ]+\|\n(?:\|.+?\|\n?)+)", p_text)
            for t_idx, t_match in enumerate(table_matches):
                total_tables += 1
                chunks.append(DocumentChunk(
                    chunk_id=f"{document_name}_p{p_num}_tab{t_idx+1}",
                    document_name=document_name,
                    page_number=p_num,
                    section=current_section,
                    chunk_type="table",
                    content=f"### Tabela Contábil (Página {p_num} - {current_section})\n\n{t_match.strip()}",
                    metadata={"table_index": t_idx + 1}
                ))

            # Narrative chunk
            chunks.append(DocumentChunk(
                chunk_id=f"{document_name}_p{p_num}_nar",
                document_name=document_name,
                page_number=p_num,
                section=current_section,
                chunk_type="narrative",
                content=p_text.strip()
            ))

        return ParsedDocument(
            document_name=document_name,
            total_pages=len(pages),
            tables_found=total_tables,
            chunks=chunks,
            metadata={"file_size_bytes": path.stat().st_size}
        )
