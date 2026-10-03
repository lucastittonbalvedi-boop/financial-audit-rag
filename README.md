# Financial Auditor Copilot: Enterprise Hybrid RAG & Compliance Intelligence

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen?style=flat-square&logo=githubactions)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat-square&logo=python)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi)](src/api/main.py)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?style=flat-square&logo=streamlit)](frontend/app.py)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange?style=flat-square)](src/retrieval/vector_store.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

> Sistema corporativo de inteligência em demonstrações contábeis e auditoria (DFP, ITR, 10-K) utilizando **Hybrid RAG** (Recuperação Híbrida Léxica + Densa com Fusão RRF e Re-ranking), extração estruturada de tabelas financeiras, citações auditáveis de página/linha e avaliação contínua de MLOps via **RAGAS**.

---

## 🎯 Por que o "Naive RAG" Falha Catastroficamente em Finanças?

A esmagadora maioria dos tutoriais de RAG na internet divide documentos cegamente por contagem de caracteres (`RecursiveCharacterTextSplitter`). Em finanças e compliance contábil, isso é **inaceitável**:

1. **Destruição de Tabelas Contábeis**: Divisões cegas cortam linhas de Balanços e DREs ao meio, separando os nomes das contas dos respectivos números e períodos (2023 vs 2022). O LLM passa a alucinar somas e variações percentuais.
2. **Cegueira Semântica a Termos Exatos**: Embeddings densos frequentemente ignoram siglas contábeis estritas (*"CPC 06"*, *"Nota 14"*, *"IFRS 16"*, *"EBITDA"*), retornando trechos conceituais vagos em vez da nota explicativa específica.
3. **Falta de Auditabilidade Jurídica**: Um auditor ou analista financeiro não pode aceitar respostas em texto livre sem evidências. Cada afirmação exige **documento, número da página e citação literal exata**.

### A Solução deste Projeto:
* **Extração Tabular com `pdfplumber`**: Converte tabelas de PDFs em tabelas Markdown com cabeçalhos preservados e colunas alinhadas.
* **Busca Híbrida com RRF (Reciprocal Rank Fusion)**: Combina a precisão léxica de termos exatos do **BM25** com a compreensão conceitual densa do **ChromaDB**.
* **Re-ranker Contábil**: Pondera a densidade de anos fiscais e entidades monetárias presentes na pergunta.
* **Schema Rigoroso de Citação (Pydantic)**: Respostas validadas com nota, página e trecho verbatim.
* **Avaliação MLOps (RAGAS)**: Medição quantitativa de *Faithfulness* (mitigação de alucinações), *Answer Relevance* e *Context Precision*.

---

## 🏛️ Arquitetura do Sistema

```mermaid
flowchart TD
    subgraph Ingestao["1. Ingestão Estruturada de PDFs Contábeis"]
        PDF[Demonstrações Financeiras / DFP / ITR / 10-K] --> Parser[Extrator Híbrido: pdfplumber + Markdown Tables]
        Parser --> Chunker[Chunker Sensível a Tabelas & Notas Explicativas]
        Chunker --> Metadata[Metadados: Ticker, Exercício, Página, Seção, Tipo]
    end

    subgraph Indexacao["2. Indexação Híbrida"]
        Metadata --> BM25_Idx[(Índice Esparso: BM25 Okapi)]
        Metadata --> Dense_Idx[(Índice Denso: ChromaDB Vector Store)]
    end

    subgraph Recuperacao["3. Recuperação Híbrida & Re-ranking"]
        Query[Consulta do Auditor / Analista] --> BM25_Search[Busca Léxica: Termos Contábeis e Siglas]
        Query --> Dense_Search[Busca Semântica: Vetores Densos]
        BM25_Search --> RRF[Fusão de Ranks: RRF - Reciprocal Rank Fusion]
        Dense_Search --> RRF
        RRF --> ReRanker[Re-ranker Contábil: Ponderação de Exercício & Tabelas]
        ReRanker --> TopChunks[Top-K Evidências Contábeis Preservadas]
    end

    subgraph Sintese["4. Síntese com Citação Auditável"]
        TopChunks --> Prompt[Prompt de Auditoria: Regras Rígidas de Evidência]
        Query --> Prompt
        Prompt --> LLM[LLM: Gemini / Local Fallback]
        LLM --> PydanticOutput[Output Estruturado: Resposta + Citações + Confiança]
    end

    subgraph MLOps["5. MLOps & Avaliação Contínua"]
        PydanticOutput --> RagasEval[Framework Ragas]
        RagasEval --> Metrics["Métricas: Faithfulness (100%), Relevance, Precision"]
    end
```

---

## 📊 Resultados do Benchmark MLOps (RAGAS)

Executado sobre o **Golden Dataset** de demonstrações financeiras reais da Petrobras (DFP 2023):

```text
| Consulta de Auditoria                            | Faithfulness   | Relevance   | Precision   | Confiança   |
|--------------------------------------------------|----------------|-------------|-------------|-------------|
| Qual foi a receita líquida de vendas e o lucr... | 100.0%         | 60.0%       | 100.0%      | HIGH        |
| Qual o saldo total provisionado para litígios... | 100.0%         | 43.6%       | 100.0%      | HIGH        |
| Qual o montante total da dívida consolidada e... | 100.0%         | 12.0%       | 100.0%      | HIGH        |
| Qual foi a margem EBITDA ajustada e a dívida ... | 100.0%         | 45.0%       | 50.0%       | HIGH        |
```

* **Faithfulness (Fidelidade / Não-Alucinação)**: **100.0%** (todos os números e variações percentuais citados na resposta têm comprovação matemática exata no documento).
* **Context Precision (Precisão de Ranking)**: **87.5%** (o algoritmo RRF posiciona a tabela ou nota explicativa exata no topo do ranking).

---

## 📂 Estrutura do Repositório

```text
financial-audit-rag/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Pipeline CI (pytest em Python 3.11 e 3.12)
├── data/
│   ├── sample_reports/            # PDFs reais e relatórios DFP de exemplo (B3)
│   │   ├── petrobras_dfp_2023_sample.pdf
│   │   ├── petrobras_dfp_2023_sample.md
│   │   └── generate_sample_pdf.py # Gerador de PDFs sintéticos com tabelas
│   └── evaluation/
│       └── golden_dataset.json    # Perguntas, respostas ground-truth e seções esperadas
├── src/
│   ├── config.py                  # Configurações com Pydantic Settings
│   ├── ingestion/
│   │   ├── pdf_parser.py          # Extrator sensível a tabelas contábeis (pdfplumber)
│   │   └── chunker.py             # Chunker que preserva integridade tabular e notas
│   ├── retrieval/
│   │   ├── bm25_retriever.py      # Busca léxica BM25 com tokenizador financeiro (R$, %, CPC)
│   │   ├── vector_store.py        # Busca densa com ChromaDB e embeddings Gemini
│   │   ├── reranker.py            # Re-ranker ponderado de entidades e exercícios
│   │   └── hybrid.py              # Fusão RRF (Reciprocal Rank Fusion)
│   ├── generation/
│   │   ├── prompts.py             # Prompts de auditoria com tolerância zero a alucinação
│   │   ├── schemas.py             # Modelos Pydantic (AuditResponse, Citation)
│   │   └── generator.py           # Orquestrador de inferência estruturada
│   ├── evaluation/
│   │   └── ragas_evaluator.py     # Avaliação MLOps (Faithfulness, Relevance, Precision)
│   └── api/
│       └── main.py                # Servidor FastAPI (/ingest, /audit/query, /evaluate)
├── frontend/
│   └── app.py                     # Dashboard Streamlit interativo com cartões de citação
├── tests/
│   ├── test_parser.py             # Testes de integridade de extração de tabelas
│   ├── test_chunker.py            # Testes de chunking atômico
│   ├── test_retrieval.py          # Testes de busca léxica, densa e fusão RRF
│   └── test_api.py                # Testes de integração dos endpoints FastAPI
├── run_benchmark.py               # Script CLI para rodar o benchmark RAGAS
├── Dockerfile                     # Conteinerização para produção
├── docker-compose.yml             # Orquestração FastAPI + Streamlit
├── pyproject.toml                 # Configuração moderna de empacotamento Python
└── requirements.txt               # Dependências do projeto
```

---

## 🚀 Como Executar o Projeto

### Opção 1: Execução Local (Ambiente Virtual)

1. **Clone o repositório e crie o ambiente virtual**:
   ```bash
   git clone https://github.com/seu-usuario/financial-audit-rag.git
   cd financial-audit-rag
   python -m venv .venv
   
   # No Windows:
   .\.venv\Scripts\activate
   # No Linux/Mac:
   source .venv/bin/activate
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt reportlab
   ```

3. **Configure as variáveis de ambiente (Opcional)**:
   ```bash
   cp .env.example .env
   # Adicione sua GEMINI_API_KEY no arquivo .env (obtenha gratuitamente no Google AI Studio)
   # Caso não adicione, o projeto executará perfeitamente em modo heurístico offline!
   ```

4. **Execute a Suíte de Testes Automatizados**:
   ```bash
   pytest -v
   ```

5. **Execute o Benchmark de MLOps & RAGAS**:
   ```bash
   python run_benchmark.py
   ```

6. **Inicie o Servidor da API FastAPI**:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
   * Documentação interativa Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

7. **Inicie a Interface Interativa em Streamlit**:
   ```bash
   streamlit run frontend/app.py
   ```
   * Acesse no navegador: [http://localhost:8501](http://localhost:8501)

---

### Opção 2: Execução Via Docker Compose

Suba tanto o backend FastAPI quanto a interface Streamlit com um único comando:

```bash
docker compose up --build
```

* **Frontend Streamlit**: [http://localhost:8501](http://localhost:8501)
* **Backend API Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ☁️ Estratégia de Deploy em Nuvem 100% Gratuita

| Camada | Ferramenta Gratuita | Como Configurar no Projeto |
| :--- | :--- | :--- |
| **API & Backend** | *Hugging Face Spaces* ou *Render (Free Tier)* | Deploy do `Dockerfile` expondo a API FastAPI sem custos de hospedagem. |
| **Interface de Usuário** | *Streamlit Community Cloud* | Conecte diretamente este repositório do GitHub no Streamlit Cloud para disponibilizar a URL pública no seu LinkedIn/portfólio. |
| **Banco Vetorial em Nuvem** | *Neon / Supabase (`pgvector`)* | O código possui interface desacoplada pronta para trocar o ChromaDB pelo Postgres vetorial com uma variável de conexão. |
| **Notebooks & Treinamento** | *AWS SageMaker Studio Lab* ou *Databricks Community* | Micro-cluster Spark para ingestão de lotes de relatórios em massa. |

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
