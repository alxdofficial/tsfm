"""
Pydantic schemas for structured outputs from Gemini.

These enforce the correct tool use syntax and response format.
"""

from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional
from enum import Enum


# ============================================================================
# Query Generation
# ============================================================================

class GeneratedQuery(BaseModel):
    """A generated user query."""
    query: str = Field(description="The user's question about the session")


# ============================================================================
# Next Step Decision
# ============================================================================

class ToolUseDecision(BaseModel):
    """Decision to use a tool."""
    reasoning: str = Field(description="Why this tool is needed")
    action: Literal["use_tool"] = Field(default="use_tool")
    tool_name: Literal["show_channel_stats", "select_channels"] = Field(
        description="Which tool to call"
    )
    parameters: Dict[str, Any] = Field(description="Tool parameters as JSON")


class ClassificationDecision(BaseModel):
    """Decision to provide final classification."""
    reasoning: str = Field(description="Analytical thought process")
    action: Literal["classify"] = Field(default="classify")
    classification: str = Field(description="The predicted activity name")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in prediction"
    )
    explanation: str = Field(
        description="Detailed explanation of why this classification"
    )


# For Gemini to choose between the two
class NextStepAction(str, Enum):
    """What action to take next."""
    USE_TOOL = "use_tool"
    CLASSIFY = "classify"


class NextStepDecision(BaseModel):
    """
    Decision for what to do next in the analysis.

    Either use a tool to get more information, or provide final classification.
    """
    action: NextStepAction = Field(
        description="Whether to use a tool or provide classification"
    )
    reasoning: str = Field(description="Thought process for this decision")

    # If action == USE_TOOL
    tool_name: Optional[Literal["show_channel_stats", "select_channels"]] = Field(
        default=None,
        description="Tool to call (required if action is use_tool)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool parameters (required if action is use_tool)"
    )

    # If action == CLASSIFY
    classification: Optional[str] = Field(
        default=None,
        description="Activity name (required if action is classify)"
    )
    confidence: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="Confidence level (required if action is classify)"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Detailed explanation (required if action is classify)"
    )

    def validate_completeness(self) -> bool:
        """Check that required fields are present based on action."""
        if self.action == NextStepAction.USE_TOOL:
            return self.tool_name is not None and self.parameters is not None
        elif self.action == NextStepAction.CLASSIFY:
            return (
                self.classification is not None
                and self.confidence is not None
                and self.explanation is not None
            )
        return False
