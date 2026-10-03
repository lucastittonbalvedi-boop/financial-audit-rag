"""Streamlit interactive UI for Financial Compliance & Audit Hybrid RAG."""

import os
import json
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported when running from any folder
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Local imports
from src.config import settings
from src.ingestion.pdf_parser import FinancialPDFParser
from src.ingestion.chunker import FinancialChunker
from src.retrieval.hybrid import HybridFinancialRetriever
from src.generation.generator import FinancialAuditGenerator
from src.evaluation.ragas_evaluator import FinancialRagasEvaluator
from src.generation.schemas import AuditQueryRequest

# Page config
st.set_page_config(
    page_title="Copiloto de Auditoria Financeira | Hybrid RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .citation-card {
        border-left: 4px solid #3B82F6;
        background-color: #F9FAFB;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Initializes backend singletons."""
    parser = FinancialPDFParser()
    chunker = FinancialChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    retriever = HybridFinancialRetriever()
    generator = FinancialAuditGenerator()
    evaluator = FinancialRagasEvaluator()
    return parser, chunker, retriever, generator, evaluator


parser, chunker, retriever, generator, evaluator = get_services()

# Initialize session state
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:
    st.title("⚙️ Configurações & Acervo")
    st.markdown("---")
    
    # LLM status
    has_api_key = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
    if has_api_key:
        st.success("🟢 Provedor: Google Gemini API (Ativo)")
    else:
        st.info("ℹ️ Provedor: Sintetizador Heurístico Local (Offline)")

    st.markdown("### Parâmetros de Recuperação")
    top_k = st.slider("Top Chunks Recuperados", min_value=1, max_value=10, value=5)
    enable_reranker = st.checkbox("Habilitar Re-ranker Contábil", value=True)
    
    st.markdown("---")
    st.markdown("### Ingestão de Relatórios")
    uploaded_files = st.file_uploader(
        "Upload de DFP / ITR / 10-K (PDF ou Texto)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in [d["name"] for d in st.session_state.indexed_docs]:
                with st.spinner(f"Processando {file.name}..."):
                    save_path = settings.sample_reports_dir / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    if file.name.lower().endswith(".pdf"):
                        parsed = parser.parse_pdf(save_path)
                    else:
                        parsed = parser.parse_text_or_markdown(save_path)
                        
                    chunks = chunker.chunk_document(parsed.chunks)
                    retriever.index_document_chunks(chunks)
                    
                    st.session_state.indexed_docs.append({
                        "name": file.name,
                        "pages": parsed.total_pages,
                        "tables": parsed.tables_found,
                        "chunks": len(chunks)
                    })
                st.success(f"✓ {file.name} indexado!")

    # Pre-loaded sample files button
    if st.button("Carregar Relatório Exemplo (Petrobras B3 2023)"):
        sample_path = settings.sample_reports_dir / "petrobras_dfp_2023_sample.md"
        if sample_path.exists() and "petrobras_dfp_2023_sample.md" not in [d["name"] for d in st.session_state.indexed_docs]:
            parsed = parser.parse_text_or_markdown(sample_path)
            chunks = chunker.chunk_document(parsed.chunks)
            retriever.index_document_chunks(chunks)
            st.session_state.indexed_docs.append({
                "name": "petrobras_dfp_2023_sample.md",
                "pages": parsed.total_pages,
                "tables": parsed.tables_found,
                "chunks": len(chunks)
            })
            st.success("Relatório de exemplo carregado com sucesso!")
            st.rerun()

    st.markdown("### Documentos Indexados")
    if st.session_state.indexed_docs:
        for doc in st.session_state.indexed_docs:
            st.write(f"📄 **{doc['name']}** ({doc['pages']} págs, {doc['tables']} tabs, {doc['chunks']} chunks)")
    else:
        st.caption("Nenhum documento carregado ainda.")


# Main Dashboard Tabs
st.markdown('<div class="main-header">Copiloto de Auditoria & Demonstrações Financeiras</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid RAG com Extração Tabular, RRF, Re-ranking e Citações Auditáveis de Página/Linha</div>', unsafe_allow_html=True)

tab_chat, tab_mlops, tab_architecture = st.tabs([
    "🔍 Auditoria Interativa", 
    "📈 MLOps & Avaliação RAGAS", 
    "🏛️ Arquitetura & Diferenciais"
])

with tab_chat:
    # Example questions
    st.markdown("**Consultas sugeridas para auditoria:**")
    col1, col2, col3 = st.columns(3)
    if col1.button("Receita Líquida e Margem EBITDA em 2023"):
        st.session_state.user_input = "Qual foi a receita líquida e o EBITDA ajustado reportado em 2023 comparado a 2022?"
    if col2.button("Contingências Tributárias e Trabalhistas (Nota 14)"):
        st.session_state.user_input = "Qual o saldo total provisionado para contingências e litígios na Nota Explicativa 14?"
    if col3.button("Política de Investimentos & Dívida Líquida"):
        st.session_state.user_input = "Qual era a dívida líquida consolidada ao final do exercício e qual o prazo médio de vencimento?"

    user_query = st.chat_input("Digite sua pergunta de auditoria ou compliance contábil...")
    if "user_input" in st.session_state and st.session_state.user_input:
        user_query = st.session_state.user_input
        st.session_state.user_input = ""

    if user_query:
        # Run pipeline
        with st.spinner("Executando recuperação híbrida e síntese auditável..."):
            candidates = retriever.search(
                query=user_query,
                top_k=top_k,
                enable_reranker=enable_reranker
            )
            response = generator.generate_audit_response(
                query=user_query,
                retrieved_chunks=candidates
            )
            
            st.session_state.chat_history.append({
                "query": user_query,
                "response": response,
                "candidates": candidates
            })

    # Render Chat History
    for item in reversed(st.session_state.chat_history):
        st.markdown(f"### 🧑‍💼 Consulta: *\"{item['query']}\"*")
        resp = item["response"]
        
        # Badge de Confiança
        conf = resp.confidence
        badge_class = f"badge-{conf.lower()}"
        st.markdown(f'<span class="{badge_class}">Grau de Confiança: {conf}</span>', unsafe_allow_html=True)
        
        # Resposta
        st.markdown(f"\n{resp.answer}\n")
        
        # Métricas extraídas
        if resp.financial_metrics_detected:
            st.markdown("**Métricas Contábeis Identificadas:**")
            m_cols = st.columns(min(4, len(resp.financial_metrics_detected)))
            for idx, (k, v) in enumerate(resp.financial_metrics_detected.items()):
                m_cols[idx % 4].metric(label=k.replace("_", " ").title(), value=str(v))

        # Citações Auditáveis
        if resp.citations:
            with st.expander(f"📚 Ver Evidências e Citações Auditáveis ({len(resp.citations)} fontes)", expanded=True):
                for cit in resp.citations:
                    st.markdown(f"""
                    <div class="citation-card">
                        <b>Documento:</b> <code>{cit.document_name}</code> | 
                        <b>Página:</b> {cit.page_number} | 
                        <b>Seção:</b> <i>{cit.section}</i><br>
                        <b>Citação Literal:</b> <blockquote>"{cit.verbatim_quote}"</blockquote>
                    </div>
                    """, unsafe_allow_html=True)

        # Chunks brutos recuperados
        with st.expander("🛠️ Inspecionar Chunks Brutos Recuperados (Hybrid RRF)"):
            for rank, (chunk, score) in enumerate(item["candidates"], start=1):
                st.markdown(f"**Chunk #{rank}** (Score: `{score:.4f}` | Tipo: `{chunk.chunk_type}` | Pág: `{chunk.page_number}`)")
                st.code(chunk.content, language="markdown")
                st.markdown("---")

with tab_mlops:
    st.subheader("Avaliação Contínua de MLOps com RAGAS")
    st.markdown("""
    Medição sistemática de fidelidade contábil e mitigação de alucinações. Um portfólio profissional de IA
    não apenas gera respostas, mas **mensura a probabilidade de falha e alucinação matemática**.
    """)
    
    if st.button("Executar Benchmark de Avaliação"):
        with st.spinner("Calculando métricas RAGAS sobre o Golden Dataset..."):
            golden_path = settings.evaluation_dir / "golden_dataset.json"
            if golden_path.exists():
                with open(golden_path, "r", encoding="utf-8") as f:
                    golden_items = json.load(f)
                    
                eval_payload = []
                for item in golden_items:
                    q = item["query"]
                    cand = retriever.search(query=q, top_k=3)
                    r = generator.generate_audit_response(query=q, retrieved_chunks=cand)
                    ctx = " ".join([c[0].content for c in cand])
                    eval_payload.append({
                        "query": q,
                        "generated_answer": r.answer,
                        "retrieved_context": ctx,
                        "ground_truth_section": item.get("expected_section", ""),
                        "retrieved_chunks": [c[0].content for c in cand],
                        "confidence": r.confidence
                    })
                
                report = evaluator.evaluate_benchmark(eval_payload)
                
                # Metrics cards
                c1, c2, c3 = st.columns(3)
                c1.metric("Faithfulness (Fidelidade)", f"{report.faithfulness_score * 100:.1f}%", help="Zero alucinação em valores contábeis")
                c2.metric("Answer Relevance", f"{report.answer_relevance_score * 100:.1f}%", help="Aderência ao propósito da pergunta")
                c3.metric("Context Precision", f"{report.context_precision_score * 100:.1f}%", help="Capacidade do ranking posicionar o trecho correto no topo")
                
                st.markdown("### Detalhamento por Caso de Teste")
                df_details = pd.DataFrame(report.details)
                st.dataframe(df_details, use_container_width=True)
            else:
                st.warning("Arquivo golden_dataset.json não encontrado.")

with tab_architecture:
    st.subheader("Por que o Naive RAG falha em Finanças?")
    st.markdown("""
    A maioria dos projetos amadores de RAG divide documentos cegamente por contagem de caracteres (`RecursiveCharacterTextSplitter`).
    Em balanços contábeis, isso divide linhas de tabelas ao meio, separando cabeçalhos de valores numéricos e provocando alucinações severas.
    
    ### Diferenciais deste Projeto:
    1. **Parsing Tabular com `pdfplumber`**: Converte tabelas contábeis para Markdown Tables alinhadas, preservando a semântica relacional de linhas e colunas.
    2. **Busca Híbrida (Lexical BM25 + Dense ChromaDB)**:
       - O BM25 captura termos exatos como *"Nota Explicativa 14"* e *"CPC 06"*.
       - Embeddings capturam conceitos como *"risco de solvência"* e *"litígio tributário"*.
    3. **Fusão RRF (Reciprocal Rank Fusion)**: Combina rankings de métodos distintos sem distorção de escala de distâncias vetoriais.
    4. **Re-ranker Contábil**: Avalia a densidade de anos fiscais e entidades numéricas presentes na pergunta.
    5. **Saída Estruturada com Citações**: Respostas validadas por schemas Pydantic exigindo página e citação literal exata.
    """)
