"""Evaluation metrics and benchmark endpoints."""

import os
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.core.config import BASE_DIR
from app.core.logging import logger

router = APIRouter(prefix="/eval", tags=["Evaluation & Benchmarks"])

REPORT_PATH = BASE_DIR / "eval" / "ragas_report.json"


@router.get("/metrics")
async def get_evaluation_metrics():
    """Retrieve the latest RAGAS evaluation scores, latencies, and sample test traces."""
    if not REPORT_PATH.exists():
        return {
            "status": "no_report_found",
            "message": "No evaluation has been executed yet. Run POST /api/v1/eval/run or 'python eval/evaluate_ragas.py'.",
            "metrics": {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "average_latency_sec": 0.0,
            },
        }

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "success",
            "metrics": data.get("metrics", {}),
            "total_samples": len(data.get("samples", [])),
            "samples": data.get("samples", []),
        }
    except Exception as e:
        logger.error(f"Failed to read evaluation report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
