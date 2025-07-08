from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool

from agents.base import BaseAgent
from agents.contracts import MemoryQueryIn, MemoryQueryOut, MemoryPutIn, MemoryAgentIn, MemoryAgentOut
from services import db as db_service

from utils.logging_utils import get_logger

logger = get_logger("MemoryAgent")


def _memory_query(payload: MemoryQueryIn) -> MemoryQueryOut:
    logger.debug("Memory query for op_hash=%s", payload.op_hash)
    res = db_service.get_kernel_by_hash(payload.op_hash)
    if res is None:
        logger.debug("Cache miss for %s", payload.op_hash)
        return MemoryQueryOut(found=False)
    logger.debug("Cache hit for %s", payload.op_hash)
    return MemoryQueryOut(found=True, kernel=res["kernel_src"], speedup=res["speedup"])


def _memory_put(payload: MemoryPutIn) -> MemoryQueryOut:
    logger.debug("Putting kernel into cache | op_hash=%s speedup=%.2f", payload.op_hash, payload.speedup)
    db_service.put_kernel_raw(payload.op_hash, payload.kernel, payload.latency_ms, payload.speedup, payload.ptx_path, None, None)
    return MemoryQueryOut(found=True, kernel=payload.kernel, speedup=payload.speedup)

query_tool = FunctionTool(_memory_query)
put_tool = FunctionTool(_memory_put)


class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="memory",
            description="Provides persistent cache for compiled Triton kernels. Can get or put kernel data.",
            tools=[query_tool, put_tool]
        )

    async def memory_get(self, q: MemoryQueryIn) -> MemoryQueryOut:
        return _memory_query(q)

    async def memory_put(self, p: MemoryPutIn) -> MemoryQueryOut:
        return _memory_put(p)
