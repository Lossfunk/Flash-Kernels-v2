from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool

from agents.base import BaseAgent
from agents.contracts import FallbackIn, FallbackOut


def _fallback_strategy(payload: FallbackIn) -> FallbackOut:
    """Return compiled kernel if correct & fast; otherwise default to PyTorch baseline."""
    if payload.correct and payload.speedup >= 1.2:
        final_kernel = "compiled"  # Indicate to orchestration to use compiled PTX
        return FallbackOut(final_kernel=final_kernel, speedup=payload.speedup)
    else:
        # Baseline fallback call string; orchestrator can interpret
        final_kernel = "torch_baseline"
        return FallbackOut(final_kernel=final_kernel, speedup=1.0)

fallback_tool = FunctionTool(_fallback_strategy)


class FallbackAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="fallback",
            description="Handles fallback logic when compilation fails or speedup is insufficient.",
            tools=[fallback_tool]
        )

    async def fallback(self, payload: FallbackIn) -> FallbackOut:
        return _fallback_strategy(payload)
