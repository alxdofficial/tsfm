"""
Pydantic schemas for structured outputs from Gemini.

These enforce the correct tool use syntax and response format.
"""

from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional
from enum import Enum


# ============================================================================
# Thread Initialization
# ============================================================================

class GeneratedQuery(BaseModel):
    """A generated user query to start a new thread."""
    query: str = Field(description="The user's question about the session")


# ============================================================================
# Next Step Decision
# ============================================================================

class NextStepAction(str, Enum):
    """What action to take next."""
    RESPOND = "respond"  # Provide intermediate response without tool
    USE_TOOL = "use_tool"  # Call a tool to get more information
    FINISH_CONVERSATION = "finish_conversation"  # Finish with final answer using e-tokens


class NextStepDecision(BaseModel):
    """
    Decision for what to do next in the analysis.

    Three possible actions:
    - respond: Give an intermediate answer/observation without calling a tool
    - use_tool: Call a tool to gather more information
    - finish_conversation: Finish the conversation with final answer (after model returns e-tokens)
    """
    action: NextStepAction = Field(
        description="What action to take: respond, use_tool, or finish_conversation"
    )
    reasoning: str = Field(description="Thought process for this decision")

    # If action == RESPOND
    response: Optional[str] = Field(
        default=None,
        description="Intermediate response to user (required if action is respond)"
    )

    # If action == USE_TOOL
    tool_name: Optional[Literal[
        "show_channel_stats",
        "select_channels",
        "human_activity_recognition_model",
        "motion_capture_model"
    ]] = Field(
        default=None,
        description="Tool to call (required if action is use_tool)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool parameters (required if action is use_tool)"
    )

    # If action == FINISH_CONVERSATION
    # No additional fields needed - the system will call generate_final_answer()

    def validate_completeness(self) -> bool:
        """Check that required fields are present based on action."""
        if self.action == NextStepAction.RESPOND:
            return self.response is not None
        elif self.action == NextStepAction.USE_TOOL:
            return self.tool_name is not None and self.parameters is not None
        elif self.action == NextStepAction.FINISH_CONVERSATION:
            return True  # No additional fields required
        return False


# ============================================================================
# Final Answer Generation
# ============================================================================

class FinalAnswer(BaseModel):
    """
    Final answer to the user's query.

    This is generated separately after the conversation reaches a conclusion.
    """
    reasoning: str = Field(description="Step-by-step reasoning leading to the answer")
    final_answer: str = Field(description="The final classification or answer")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level")
    explanation: str = Field(description="Detailed explanation of why this is the answer")
