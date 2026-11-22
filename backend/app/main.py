from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes_data import router as data_router
from app.api.routes_tools import router as tools_router

app = FastAPI(
    title="ChromCopilot API",
    version="0.2.0",
    description="Grounded Multi-Scale Agentic Copilot for 3D + 1D Chromatin Exploration",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tools_router)
app.include_router(data_router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "chrom-copilot", "schema_version": "1.0.0"}