"""
Training data generation functions using Gemini.

Two main functions:
1. generate_query: Generate a plausible user query
2. generate_next_step: Generate the next step in a conversation
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from google import genai
from google.genai.types import GenerateContentConfig, Part

from datascripts.generate_tool_chains.schemas import GeneratedQuery, NextStepDecision


# Paths to prompts (in same folder as this file)
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_INSTRUCTIONS = PROMPTS_DIR / "system_instructions.txt"
QUERY_GENERATION_PROMPT = PROMPTS_DIR / "query_generation_examples.txt"
NEXT_STEP_PROMPT = PROMPTS_DIR / "next_step_examples.txt"


def load_system_instructions() -> str:
    """Load the system instructions for Gemini."""
    with open(SYSTEM_INSTRUCTIONS, 'r', encoding='utf-8') as f:
        return f.read()


def generate_query(
    client: genai.Client,
    manifest: Dict[str, Any],
    model: str = "gemini-2.5-flash",
    temperature: float = 0.9
) -> str:
    """
    Generate a realistic user query for activity classification.

    Args:
        client: Gemini client instance
        manifest: Dataset manifest dictionary
        model: Which Gemini model to use
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        Generated query string
    """
    # Load prompts
    system_instructions = load_system_instructions()

    with open(QUERY_GENERATION_PROMPT, 'r', encoding='utf-8') as f:
        query_template = f.read()

    # Fill in manifest
    prompt = query_template.replace("{{MANIFEST}}", json.dumps(manifest, indent=2))

    # Combine system instructions and prompt
    full_prompt = f"{system_instructions}\n\n---\n\n{prompt}"

    # Configure with structured output
    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=1000,  # Gemini 2.5 Flash uses ~500 thinking tokens + output
        response_schema=GeneratedQuery.model_json_schema(),
        response_mime_type="application/json",
    )

    # Call Gemini
    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [Part(text=full_prompt)]}],
        config=config,
    )

    if response.parsed is None:
        # Debug: Show what we got instead
        error_msg = "Failed to parse query response from Gemini\n"
        error_msg += f"Response text: {response.text if hasattr(response, 'text') else 'N/A'}\n"
        error_msg += f"Response object: {response}\n"
        raise ValueError(error_msg)

    parsed_response = GeneratedQuery.model_validate(response.parsed)
    return parsed_response.query


def generate_next_step(
    client: genai.Client,
    dataset_name: str,
    session_id: str,
    user_query: str,
    conversation_history: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7
) -> NextStepDecision:
    """
    Generate the next step in the analysis conversation.

    Args:
        client: Gemini client instance
        dataset_name: Name of the dataset
        session_id: ID of the session being analyzed
        user_query: The user's question
        conversation_history: List of previous turns (tool calls + results)
        model: Which Gemini model to use
        temperature: Sampling temperature

    Returns:
        NextStepDecision with action and parameters
    """
    # Load prompts
    system_instructions = load_system_instructions()

    with open(NEXT_STEP_PROMPT, 'r', encoding='utf-8') as f:
        next_step_template = f.read()

    # Format conversation history
    if not conversation_history:
        history_text = "No previous steps taken."
    else:
        history_parts = []
        for i, turn in enumerate(conversation_history, 1):
            history_parts.append(f"**Step {i}:**")
            history_parts.append(f"Reasoning: {turn['reasoning']}")
            history_parts.append(f"Tool: {turn['tool_call']['tool_name']}")
            history_parts.append(f"Parameters: {json.dumps(turn['tool_call']['parameters'])}")
            history_parts.append(f"Result: {json.dumps(turn['tool_result'], indent=2)}")
            history_parts.append("")
        history_text = '\n'.join(history_parts)

    # Fill in template
    prompt = next_step_template.replace("{{DATASET_NAME}}", dataset_name)
    prompt = prompt.replace("{{SESSION_ID}}", session_id)
    prompt = prompt.replace("{{USER_QUERY}}", user_query)
    prompt = prompt.replace("{{CONVERSATION_HISTORY}}", history_text)

    # Combine system instructions and prompt
    full_prompt = f"{system_instructions}\n\n---\n\n{prompt}"

    # Configure with structured output
    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=2048,  # Increased for Gemini 2.5 Flash thinking tokens (~100) + reasoning
        response_schema=NextStepDecision.model_json_schema(),
        response_mime_type="application/json",
    )

    # Call Gemini
    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [Part(text=full_prompt)]}],
        config=config,
    )

    if response.parsed is None:
        # Debug: Show what we got instead
        error_msg = "Failed to parse next step response from Gemini\n"
        error_msg += f"Response text: {response.text if hasattr(response, 'text') else 'N/A'}\n"
        error_msg += f"Response object: {response}\n"
        raise ValueError(error_msg)

    decision = NextStepDecision.model_validate(response.parsed)

    # Validate completeness
    if not decision.validate_completeness():
        raise ValueError(
            f"Incomplete decision: action={decision.action} but required fields missing"
        )

    return decision


def format_next_step_for_storage(decision: NextStepDecision) -> Dict[str, Any]:
    """
    Format a NextStepDecision into storage format.

    Args:
        decision: The decision from Gemini

    Returns:
        Dictionary suitable for JSON storage
    """
    if decision.action.value == "use_tool":
        return {
            "reasoning": decision.reasoning,
            "action": "use_tool",
            "tool_call": {
                "tool_name": decision.tool_name,
                "parameters": decision.parameters
            }
        }
    else:  # classify
        return {
            "reasoning": decision.reasoning,
            "action": "classify",
            "classification": decision.classification,
            "confidence": decision.confidence,
            "explanation": decision.explanation
        }
