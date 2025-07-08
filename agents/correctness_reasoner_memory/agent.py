from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import Field

from agents.base import BaseAgent # Assuming BaseAgent is in agents.base
from agents.contracts import (
    CorrectnessReasonerMemoryIn,
    CorrectnessReasonerMemoryOut,
    CorrectnessReasonerMemoryAddIn,
    CorrectnessReasonerMemoryGetIn,
    CorrectnessReasonerMemoryClearIn,
    CorrectnessReasonerMemoryGetOut,
    CorrectnessReasonerMemoryWriteOut,
    ReasoningAttemptDetail # Make sure this is imported from contracts
)
from utils.logging_utils import get_logger # Assuming logging utility

logger = get_logger("CorrectnessReasonerMemoryAgent")

class CorrectnessReasonerMemoryAgent(BaseAgent):
    history_store: Dict[str, List[ReasoningAttemptDetail]] = Field(default_factory=dict)

    def __init__(self):
        super().__init__(
            name="correctness_reasoner_memory",
            description="Stores and retrieves the history of reasoning attempts for the CorrectnessReasonerAgent.",
            # This agent might not have callable tools if its operations are handled internally
            # or via specific methods rather than a generic 'tool' interface.
            tools=[] 
        )
        logger.info("CorrectnessReasonerMemoryAgent initialized.")

    def _add_attempt(self, payload: CorrectnessReasonerMemoryAddIn) -> CorrectnessReasonerMemoryWriteOut:
        key = payload.key.kernel_source_path
        if key not in self.history_store:
            self.history_store[key] = []
        self.history_store[key].append(payload.attempt)
        logger.debug(f"Added attempt for key '{key}'. History size: {len(self.history_store[key])}")
        return CorrectnessReasonerMemoryWriteOut(success=True, message="Attempt added successfully.")

    def _get_history(self, payload: CorrectnessReasonerMemoryGetIn) -> CorrectnessReasonerMemoryGetOut:
        key = payload.key.kernel_source_path
        history = self.history_store.get(key, [])
        logger.debug(f"Retrieved history for key '{key}'. Found {len(history)} attempts.")
        return CorrectnessReasonerMemoryGetOut(history=history, success=True)

    def _clear_history(self, payload: CorrectnessReasonerMemoryClearIn) -> CorrectnessReasonerMemoryWriteOut:
        key = payload.key.kernel_source_path
        if key in self.history_store:
            del self.history_store[key]
            logger.debug(f"Cleared history for key '{key}'.")
            return CorrectnessReasonerMemoryWriteOut(success=True, message=f"History for '{key}' cleared.")
        logger.debug(f"No history found for key '{key}' to clear.")
        return CorrectnessReasonerMemoryWriteOut(success=False, message=f"No history found for '{key}' to clear.")

    async def process(self, request: CorrectnessReasonerMemoryIn) -> CorrectnessReasonerMemoryOut:
        logger.info(f"Received operation: {request.operation}")
        write_result: Optional[CorrectnessReasonerMemoryWriteOut] = None
        get_result: Optional[CorrectnessReasonerMemoryGetOut] = None

        if request.operation == "add_attempt" and request.add_payload:
            write_result = self._add_attempt(request.add_payload)
        elif request.operation == "get_history" and request.get_payload:
            get_result = self._get_history(request.get_payload)
        elif request.operation == "clear_history" and request.clear_payload:
            write_result = self._clear_history(request.clear_payload)
        else:
            logger.warning(f"Invalid operation or missing payload: {request.operation}")
            # In a real scenario, might return an error status in the output
            # For now, returning empty successful write to align with structure
            write_result = CorrectnessReasonerMemoryWriteOut(success=False, message="Invalid operation or payload")

        return CorrectnessReasonerMemoryOut(
            get_history_result=get_result,
            write_result=write_result
        )

# Example usage (for testing, not part of the agent usually)
# async def main():
#     agent = CorrectnessReasonerMemoryAgent()
#     key_details = {"kernel_source_path": "/path/to/kernel.c"}
#     reasoning_attempt = ReasoningAttemptDetail(
#         suggested_grid="(128,)", 
#         suggested_args=["out", "in1"], 
#         error_received="Some error"
#     )

#     # Add an attempt
#     add_payload_in = CorrectnessReasonerMemoryAddIn(key=key_details, attempt=reasoning_attempt)
#     add_request = CorrectnessReasonerMemoryIn(operation="add_attempt", add_payload=add_payload_in)
#     result = await agent.process(add_request)
#     print(f"Add result: {result.write_result}")

#     # Get history
#     get_payload_in = CorrectnessReasonerMemoryGetIn(key=key_details)
#     get_request = CorrectnessReasonerMemoryIn(operation="get_history", get_payload=get_payload_in)
#     result = await agent.process(get_request)
#     if result.get_history_result and result.get_history_result.history:
#         print(f"Get history result: {result.get_history_result.history}")
    
#     # Clear history
#     clear_payload_in = CorrectnessReasonerMemoryClearIn(key=key_details)
#     clear_request = CorrectnessReasonerMemoryIn(operation="clear_history", clear_payload=clear_payload_in)
#     result = await agent.process(clear_request)
#     print(f"Clear result: {result.write_result}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main()) 