from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List

from google.adk.tools.function_tool import FunctionTool

from agents.base import BaseAgent
from agents.pipeline_orchestrator import KernelPipelineAgent
from utils.kernelbench import list_problems, build_op_spec
from utils.logging_utils import get_logger
from agents.contracts import EvaluatorIn, EvaluatorOut


logger = get_logger("EvaluatorAgent")


async def _evaluate_kernelbench(payload: EvaluatorIn) -> EvaluatorOut:
    """Run pipeline agent on a subset of KernelBench problems and return JSON results string."""
    logger.info("Starting KernelBench evaluation | level=%s limit=%s", payload.level, payload.limit)
    problems: List[Path] = list_problems(payload.level)
    if payload.limit is not None:
        problems = problems[:payload.limit]

    pipeline = KernelPipelineAgent()

    async def _run_single(path: Path):
        logger.debug("Running single problem %s", path.name)
        problem_id = int(path.name.split("_")[0])
        op_spec = build_op_spec(path, problem_id, payload.level)
        start = time.perf_counter()
        result = await pipeline.run(op_spec.model_dump())
        latency = time.perf_counter() - start
        result_dict = {
            "file": path.name,
            "result": result,
            "latency_s": latency,
        }
        logger.debug("Finished problem %s | latency=%.3fs", path.name, latency)
        return result_dict

    tasks = [_run_single(p) for p in problems]
    results = await asyncio.gather(*tasks)
    logger.info("Completed evaluation of %d problems", len(results))
    return EvaluatorOut(results_json=json.dumps(results, indent=2))


eval_tool = FunctionTool(_evaluate_kernelbench)


class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="kernelbench_evaluator",
            description="Evaluates the pipeline agent on KernelBench suites.",
            tools=[eval_tool]
        )

    async def evaluate(self, payload: EvaluatorIn) -> EvaluatorOut:
        return await _evaluate_kernelbench(payload) 