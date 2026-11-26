# ChromCopilot

**A Grounded Multi-Scale Agentic Copilot for Synchronized 1D, 2D, and 3D Chromatin Exploration**

ChromCopilot couples an interactive web workspace (1D genomic tracks + TAD insulation, 2D Hi-C contact heatmaps, and 3D chromatin polymer conformations) with a verified **LangGraph** bioinformatics copilot. Rather than letting an LLM generate brittle, unverified analysis scripts, ChromCopilot enforces three architectural invariants:

1. **Bidirectional Visual Grounding (`ViewportState v1.0.0`):** Supports both **Query-to-View** (atomic workspace patches, highlights, 3D camera flights, and one-click bookmarks) and **Point-to-Query** (clicking any 3D chromatin bead or 2D Hi-C bin injects exact coordinates into the agent context).
2. **Pre-Execution Plan & Sparsity Verification Gate:** Deterministically validates every planned DAG against a strict **8-tool allowlist**, genome build (`hg19`), chromosome bounds, biological resolution guardrails, sample-size rules, and live contact sparsity (~11 ms overhead) before execution.
3. **Zero-Hallucination Provenance Binder:** Every numeric claim binds programmatically to an auditable `tool_call_id` persisted in the Provenance Ledger and inspectable via the UI **ProvenanceDrawer**.

---

## Architecture

- **Backend (`backend/`):** FastAPI, Pydantic v2 contracts, `cooler` / `cooltools` / `scipy` deterministic engine, LangGraph state machines, Google Gemini API (`google-genai`), and persistent Provenance Ledger.
- **Frontend (`frontend/`):** React + TypeScript + Vite + Tailwind CSS, Zustand centralized `ViewportState` store, Canvas/Plotly synchronized 1D+2D genomics viewer, React Three Fiber 3D chromatin viewer, Plan Verification Card, and ProvenanceDrawer.
- **Evaluation (`eval/`):** 35-task multi-scale benchmark (`tier1_single_scale`, `tier2_cross_scale`, `tier3_traps`), live LLM ablation & unconstrained code-agent comparison, latency/scalability profiler, and paired NASA-TLX HMI pilot study protocol.

---

## Quick Start

### 1. Environment & Dependencies

```bash
conda create -n chrom-co python=3.11 -y
conda activate chrom-co
pip install -r backend/requirements.txt
pip install google-genai
cd frontend && npm install && cd ..
```

Configure .env at the repo root (optional for deterministic mode; required for live Gemini LLM planning):

```bash
DATA_ROOT=./data/processed
DEFAULT_DATASET_ID=rao2014_gm12878_k562
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 2. Run Backend & Frontend

```bash
# Terminal 1 — FastAPI Backend (Swagger UI at (http://127.0.0.1:8000/docs)
cd backend
uvicorn app.main:app --reload

# Terminal 2 — Synchronized 4-Panel React Workspace (http://localhost:5173)
cd frontend
npm run dev
```

### 3. Run Tests, Benchmark & Publication Figures

```bash
# Unit, Schema, Tool & Verifier Trap Tests (22 tests)
cd backend && PYTHONPATH=. pytest -v && cd ..

# Build 35 Ground-Truth Benchmark Tasks & Run Evaluation
python eval/build_benchmark.py
python eval/run_benchmark.py            # Offline deterministic + ablation
python eval/run_benchmark.py --use-llm  # Live Gemini + Ablation + Unconstrained Code Agent

# Profile Deterministic Tool Latency & Single-Cell Scalability (100–2,000 cells)
python eval/profile_latency.py

# Generate 300-DPI Publication Figures in paper/figures/
python eval/generate_figures.py
```
