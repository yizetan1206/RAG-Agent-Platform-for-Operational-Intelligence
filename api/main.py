from fastapi import FastAPI
from api.health import router as health_router

app = FastAPI(
    title="AI Knowledge Assistant",
    version="0.1.0"
)

app.include_router(health_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "ai-knowledge-assistant"}
