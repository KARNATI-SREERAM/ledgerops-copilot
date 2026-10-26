# Pydantic shapes for what the model produces and what the copilot returns. The model's text is
# untrusted: it can be invalid JSON, name a tool that doesn't exist, or skip fields. Validating it
# here turns "whatever the model wrote" into either a well-formed object or a clear error.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copilot.tools import TOOLS

REFUSAL = "I don't know from the docs."


class Plan(BaseModel):
    """The planner's decision for the next step."""
    action: Literal["retrieve", "tool", "answer"]
    query: str | None = None          # for retrieve: what to search for (defaults to the question)
    tool: str | None = None           # for tool: one of tools.TOOLS
    args: dict = Field(default_factory=dict)


class CopilotAnswer(BaseModel):
    """Guardrail (d): every response has exactly this shape, and these rules always hold."""
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=4000)
    citations: list[str]              # "file § section" of each cited chunk
    tools_used: list[str]             # names of the ledger tools called
    refused: bool
    reason: str | None = None         # why it refused, e.g. "write_blocked", "low_retrieval_score"

    @model_validator(mode="after")
    def consistent(self):
        if self.refused and not self.reason:
            raise ValueError("a refusal must say why")
        if not self.refused and not (self.citations or self.tools_used):
            raise ValueError("an answer must be grounded: cite a doc chunk or use a ledger tool")
        if unknown := set(self.tools_used) - set(TOOLS):
            raise ValueError(f"unknown tools in tools_used: {sorted(unknown)}")
        return self
