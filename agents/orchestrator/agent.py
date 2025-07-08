from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool

from agents.base import BaseAgent
from agents.contracts import OpSpec, OrchestratorIn, OrchestratorOut


def _plan_kernel(op_spec: OrchestratorIn) -> OrchestratorOut:
    """Very small heuristic plan: we always pick Triton backend and use cache if caller says so."""
    # In real implementation you might inspect op_spec.level or history.
    op_hash = _hash_spec(op_spec.model_dump())
    return OrchestratorOut(op_hash=op_hash, backend="triton", use_cache=op_spec.cache_hit)


# Helper moved here to avoid import cycle with services.db
import hashlib, json

def _hash_spec(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


action_tool = FunctionTool(_plan_kernel)


class OrchestratorAgent(BaseAgent):
    """Agent responsible for planning the overall pipeline and deciding cache usage."""

    def __init__(self):
        super().__init__(
            name="orchestrator",
            description="Plans kernel compilation pipeline and cache lookup strategy.",
            tools=[action_tool]
        )

    async def plan_kernel(self, payload: OrchestratorIn) -> OrchestratorOut:
        # Direct call to underlying function for synchronous path
        return _plan_kernel(payload)
