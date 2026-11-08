"""
Production FastAPI Serving Layer for Autonomous Spacecraft Descent Policy.
Features sub-millisecond inference, Pydantic v2 schemas, safety overrides,
and production telemetry probes (/healthz, /livez, /readyz, /metrics).
"""

from contextlib import asynccontextmanager
import time
from typing import Dict, List, Optional
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import numpy as np

from lunar_lander.policy import PolicyEngine, ACTION_NAMES
from lunar_lander.config import SafetyConfig, ServingConfig

# Global application state
app_state: Dict = {
    "engine": None,
    "start_time": time.time(),
    "total_requests": 0,
    "total_overrides": 0,
    "latencies_ms": [],
}


class TelemetryInput(BaseModel):
    """Telemetry state vector for LunarLander-v3."""
    x: float = Field(..., ge=-5.0, le=5.0, description="Horizontal coordinate (-5 to +5)")
    y: float = Field(..., ge=-1.0, le=5.0, description="Vertical altitude (-1 to +5)")
    vx: float = Field(..., ge=-15.0, le=15.0, description="Horizontal linear velocity")
    vy: float = Field(..., ge=-15.0, le=15.0, description="Vertical linear velocity")
    angle: float = Field(..., ge=-3.1416, le=3.1416, description="Lander orientation angle (radians)")
    angular_velocity: float = Field(..., ge=-10.0, le=10.0, description="Angular velocity (rad/s)")
    leg1_contact: float = Field(..., ge=0.0, le=1.0, description="Left leg ground contact (0.0 or 1.0)")
    leg2_contact: float = Field(..., ge=0.0, le=1.0, description="Right leg ground contact (0.0 or 1.0)")

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.x, self.y, self.vx, self.vy,
            self.angle, self.angular_velocity,
            self.leg1_contact, self.leg2_contact
        ], dtype=np.float32)


class StateVectorInput(BaseModel):
    """Raw array input alternative for high-throughput clients."""
    state: List[float] = Field(..., min_length=8, max_length=8, description="8-dimensional state vector")

    def to_numpy(self) -> np.ndarray:
        return np.array(self.state, dtype=np.float32)


class BatchTelemetryInput(BaseModel):
    """Batch input containing multiple state vectors."""
    telemetries: List[TelemetryInput] = Field(..., min_length=1, max_length=1000)


class ActionPredictionResponse(BaseModel):
    """Production decision output from policy and safety guardrail."""
    action: int = Field(..., description="Action ID (0: Coast, 1: Left, 2: Main, 3: Right)")
    action_name: str = Field(..., description="Human-readable action description")
    raw_policy_action: int = Field(..., description="Action originally emitted by the neural network")
    confidence: float = Field(..., description="Softmax confidence score for chosen action")
    critic_value: float = Field(..., description="Expected remaining return estimated by critic")
    probabilities: Dict[str, float] = Field(..., description="Discrete action probability distribution")
    safety: Dict = Field(..., description="Safety filter audit telemetry")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds")


class BatchActionPredictionResponse(BaseModel):
    predictions: List[ActionPredictionResponse]
    batch_size: int
    total_latency_ms: float


class HealthStatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    total_overrides: int
    avg_latency_ms: float
    model_path: str
    vecnorm_path: Optional[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing the policy engine."""
    model_path = os.getenv("MODEL_PATH", "models/best_model.zip")
    vecnorm_path = os.getenv("VECNORM_PATH", "models/lander_vecnorm.pkl")
    device = os.getenv("DEVICE", "cpu")

    app_state["engine"] = PolicyEngine(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        device=device,
        warmup=True,
    )
    app_state["start_time"] = time.time()
    yield
    app_state["engine"] = None


app = FastAPI(
    title="Lunar Lander Autonomous Flight Guidance API",
    description="Sub-millisecond inference service for PPO spacecraft descent with defensive safety gating",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=Dict[str, str], tags=["Health"])
def healthz():
    """Liveness probe."""
    return {"status": "HEALTHY"}


@app.get("/livez", response_model=Dict[str, str], tags=["Health"])
def livez():
    """Kubernetes liveness probe."""
    return {"status": "LIVE"}


@app.get("/readyz", response_model=Dict[str, str], tags=["Health"])
def readyz():
    """Kubernetes readiness probe."""
    if app_state["engine"] is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="PolicyEngine not ready")
    return {"status": "READY"}


@app.get("/metrics", response_model=HealthStatusResponse, tags=["Observability"])
def get_metrics():
    """Telemetry and performance metrics."""
    engine = app_state["engine"]
    latencies = app_state["latencies_ms"]
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    return HealthStatusResponse(
        status="OPERATIONAL" if engine is not None else "DEGRADED",
        uptime_seconds=round(time.time() - app_state["start_time"], 2),
        total_requests=app_state["total_requests"],
        total_overrides=app_state["total_overrides"],
        avg_latency_ms=round(avg_lat, 3),
        model_path=engine.model_path if engine else "NONE",
        vecnorm_path=engine.vecnorm_path if engine else None,
    )


@app.post("/v1/predict", response_model=ActionPredictionResponse, tags=["Inference"])
def predict(input_data: TelemetryInput):
    """
    Sub-millisecond single-step guidance inference with defensive safety boundary check.
    """
    engine: PolicyEngine = app_state["engine"]
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine offline")

    obs = input_data.to_numpy()
    result = engine.predict(obs, deterministic=True, apply_safety=True)

    app_state["total_requests"] += 1
    if result["safety"]["overridden"]:
        app_state["total_overrides"] += 1

    app_state["latencies_ms"].append(result["latency_ms"])
    if len(app_state["latencies_ms"]) > 1000:
        app_state["latencies_ms"].pop(0)

    return ActionPredictionResponse(**result)


@app.post("/v1/predict_vector", response_model=ActionPredictionResponse, tags=["Inference"])
def predict_vector(input_data: StateVectorInput):
    """Vectorized state array endpoint for fast batch/micro-agent controllers."""
    engine: PolicyEngine = app_state["engine"]
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine offline")

    obs = input_data.to_numpy()
    result = engine.predict(obs, deterministic=True, apply_safety=True)

    app_state["total_requests"] += 1
    if result["safety"]["overridden"]:
        app_state["total_overrides"] += 1

    return ActionPredictionResponse(**result)


@app.post("/v1/batch_predict", response_model=BatchActionPredictionResponse, tags=["Inference"])
def batch_predict(batch_input: BatchTelemetryInput):
    """High-throughput batched state inference."""
    engine: PolicyEngine = app_state["engine"]
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine offline")

    t0 = time.perf_counter()
    obs_batch = np.array([t.to_numpy() for t in batch_input.telemetries], dtype=np.float32)
    batch_results = engine.predict_batch(obs_batch, deterministic=True, apply_safety=True)
    total_lat_ms = (time.perf_counter() - t0) * 1000.0

    app_state["total_requests"] += len(batch_results)
    for res in batch_results:
        res["latency_ms"] = round(total_lat_ms / len(batch_results), 3)
        if res["safety"]["overridden"]:
            app_state["total_overrides"] += 1

    return BatchActionPredictionResponse(
        predictions=[ActionPredictionResponse(**r) for r in batch_results],
        batch_size=len(batch_results),
        total_latency_ms=round(total_lat_ms, 3),
    )
