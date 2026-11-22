from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ChromCopilot API",
    version="0.1.0",
    description="Grounded Multi-Scale Agentic Copilot for 3D + 1D Chromatin Exploration",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "chrom-copilot", "schema_version": "1.0.0"}