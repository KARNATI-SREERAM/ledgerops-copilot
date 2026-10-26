# Pydantic shapes for what the model produces. The model's text is untrusted: it can be invalid
# JSON, name a tool that doesn't exist, or skip fields. Validating it here turns "whatever the
# model wrote" into either a well-formed object or a clear error we can handle.

from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """The planner's decision for the next step."""
    action: Literal["retrieve", "tool", "answer"]
    query: str | None = None          # for retrieve: what to search for (defaults to the question)
    tool: str | None = None           # for tool: one of tools.TOOLS
    args: dict = Field(default_factory=dict)
