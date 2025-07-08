"""Shared base classes and helpers for all agents."""
from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool


class BaseAgent(Agent):
    """Common configuration for our Triton agents."""

    def __init__(self, *, name: str, description: str, tools: list[FunctionTool],
                 input_schema=None, output_schema=None, output_key: str | None = None):
        super().__init__(name=name, description=description, tools=tools,
                         input_schema=input_schema, output_schema=output_schema,
                         output_key=output_key) 