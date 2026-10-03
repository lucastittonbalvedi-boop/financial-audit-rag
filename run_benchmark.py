"""CLI Benchmark Runner for Financial Audit Hybrid RAG & MLOps Evaluation."""

import sys
import json
from pathlib import Path
from tabulate import tabulate

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import settings
from src.ingestion.pdf_parser import FinancialPDFParser
from src.ingestion.chunker import FinancialChunker
from src.retrieval.hybrid import HybridFinancialRetriever
from src.generation.generator import FinancialAuditGenerator
from src.evaluation.ragas_evaluator import FinancialRagasEvaluator


def main():
    print("=" * 70)
    print("🚀 BENCHMARK MLOps & RAGAS - FINANCIAL AUDIT HYBRID RAG")
    print("=" * 70)

    parser = FinancialPDFParser()
    chunker = FinancialChunker()
    retriever = HybridFinancialRetriever()
    generator = FinancialAuditGenerator()
    evaluator = FinancialRagasEvaluator()

    # 1. Ingest sample report
    sample_file = settings.sample_reports_dir / "petrobras_dfp_2023_sample.pdf"
    if not sample_file.exists():
        sample_file = settings.sample_reports_dir / "petrobras_dfp_2023_sample.md"

    print(f"\n📂 1. Indexando documento de auditoria: {sample_file.name}...")
    if sample_file.suffix.lower() == ".pdf":
        parsed = parser.parse_pdf(sample_file)
    else:
        parsed = parser.parse_text_or_markdown(sample_file)

    chunks = chunker.chunk_document(parsed.chunks)
    retriever.index_document_chunks(chunks)
    print(f"   ✓ {parsed.total_pages} páginas processadas.")
    print(f"   ✓ {parsed.tables_found} tabelas contábeis extraídas.")
    print(f"   ✓ {len(chunks)} chunks indexados nos índices BM25 e ChromaDB.")

    # 2. Load Golden Dataset
    golden_path = settings.evaluation_dir / "golden_dataset.json"
    print(f"\n🧪 2. Carregando Golden Dataset de Testes ({golden_path.name})...")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_items = json.load(f)

    # 3. Run Pipeline and Evaluate
    print(f"🔍 3. Executando {len(golden_items)} consultas de auditoria e avaliando fidelidade...\n")
    eval_payload = []
    summary_rows = []

    for item in golden_items:
        query = item["query"]
        candidates = retriever.search(query=query, top_k=3)
        response = generator.generate_audit_response(query=query, retrieved_chunks=candidates)
        context_str = " ".join([c[0].content for c in candidates])

        eval_payload.append({
            "query": query,
            "generated_answer": response.answer,
            "retrieved_context": context_str,
            "ground_truth_section": item.get("expected_section", ""),
            "retrieved_chunks": [c[0].content for c in candidates],
            "confidence": response.confidence
        })

    report = evaluator.evaluate_benchmark(eval_payload)

    for d in report.details:
        q_short = (d["query"][:45] + "...") if len(d["query"]) > 45 else d["query"]
        summary_rows.append([
            q_short,
            f"{d['faithfulness'] * 100:.1f}%",
            f"{d['answer_relevance'] * 100:.1f}%",
            f"{d['context_precision'] * 100:.1f}%",
            d["confidence"]
        ])

    print(tabulate(
        summary_rows,
        headers=["Consulta de Auditoria", "Faithfulness", "Relevance", "Precision", "Confiança"],
        tablefmt="github"
    ))

    print("\n" + "=" * 70)
    print("📊 RESULTADOS CONSOLIDADOS DO BENCHMARK (RAGAS)")
    print("=" * 70)
    for m in report.metrics:
        print(f"• {m.metric_name}: {m.score * 100:.1f}% ({m.description})")

    print("\n✅ Benchmark concluído com sucesso!")


if __name__ == "__main__":
    main()
