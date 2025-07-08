from __future__ import annotations

import dotenv
dotenv.load_dotenv()

from google.adk.tools.function_tool import FunctionTool
from utils.genai_client import chat

from agents.base import BaseAgent
from agents.contracts import ReasonerIn, ReasonerOut
from agents.reasoner.prompts import SYSTEM_PROMPT

from utils.logging_utils import get_logger

logger = get_logger("ReasonerAgent")


def _reason_compile_failure(payload: ReasonerIn) -> ReasonerOut:
    logger.info("Analyzing compilation failure, generating fix hint")
    
    # Build the user message with optional research context
    user_content = f"Compilation log:\n{payload.compile_log}\n\nFull Source Code that Failed:\n```python\n{payload.kernel_src_to_analyze}\n```"
    
    # Add research context if available
    if payload.research_context:
        user_content += f"\n\nAdditional Research Context:\n{payload.research_context}"
        logger.info("Including research context in reasoner analysis (length: %d chars)", len(payload.research_context))
    else:
        logger.info("No research context provided to reasoner")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.debug("Sending messages to LLM: %s", messages)
    hint = chat(messages, temperature=0.0)
    logger.info("Received fix hint")
    return ReasonerOut(fix_hint=hint)


reason_tool = FunctionTool(_reason_compile_failure)


class ReasonerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="reasoner",
            description="Analyzes compilation failures and suggests fixes.",
            tools=[reason_tool]
        )

    async def reason(self, payload: ReasonerIn) -> ReasonerOut:
        return _reason_compile_failure(payload)
