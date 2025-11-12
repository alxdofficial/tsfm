"""
Training data generation functions using Gemini.

Main functions:
- start_new_thread: Initialize a new conversation with a user query
- generate_next_step: Generate the next step in an ongoing conversation
- format_next_step_for_storage: Format decisions for JSON storage
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from google import genai
from google.genai.types import GenerateContentConfig, Part

from datascripts.generate_tool_chains.schemas import GeneratedQuery, NextStepDecision, FinalAnswer


# Paths to prompts (in same folder as this file)
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_INSTRUCTIONS = PROMPTS_DIR / "system_instructions.txt"
QUERY_GENERATION_PROMPT = PROMPTS_DIR / "query_generation_examples.txt"
NEXT_STEP_PROMPT = PROMPTS_DIR / "next_step_examples.txt"
FINAL_ANSWER_PROMPT = PROMPTS_DIR / "final_answer_prompt.txt"


def load_system_instructions() -> str:
    """Load the system instructions for Gemini."""
    with open(SYSTEM_INSTRUCTIONS, 'r', encoding='utf-8') as f:
        return f.read()


def start_new_thread(
    client: genai.Client,
    manifest: Dict[str, Any],
    model: str = "gemini-2.5-flash",
    temperature: float = 0.9
) -> str:
    """
    Generate a realistic user query to initialize a new conversation thread.

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
    artifact_id: str = None,
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
        artifact_id: Current artifact ID to use in tools (optional)
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

            # Check if this is a tool call or a response
            if turn.get('action') == 'use_tool' and 'tool_call' in turn:
                history_parts.append(f"Tool: {turn['tool_call']['tool_name']}")
                history_parts.append(f"Parameters: {json.dumps(turn['tool_call']['parameters'])}")
                history_parts.append(f"Result: {json.dumps(turn['tool_result'], indent=2)}")

                # Highlight new artifact IDs if created
                if isinstance(turn['tool_result'], dict) and 'artifact_id' in turn['tool_result']:
                    new_artifact_id = turn['tool_result']['artifact_id']
                    history_parts.append(f"→ Created new artifact: {new_artifact_id}")

            elif turn.get('action') == 'respond' and 'response' in turn:
                history_parts.append(f"Action: respond")
                history_parts.append(f"Response: {turn['response']}")

            else:
                # Fallback for unknown turn type
                history_parts.append(f"Action: {turn.get('action', 'unknown')}")

            history_parts.append("")
        history_text = '\n'.join(history_parts)

    # Fill in template
    prompt = next_step_template.replace("{{DATASET_NAME}}", dataset_name)
    prompt = prompt.replace("{{SESSION_ID}}", session_id)
    prompt = prompt.replace("{{USER_QUERY}}", user_query)
    prompt = prompt.replace("{{CONVERSATION_HISTORY}}", history_text)

    # Replace {{ARTIFACT_ID}} placeholder with actual artifact ID from thread
    if artifact_id:
        prompt = prompt.replace("{{ARTIFACT_ID}}", artifact_id)
    else:
        # Fallback if no artifact ID provided
        prompt = prompt.replace("{{ARTIFACT_ID}}", "<no_artifact_available>")

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
    if decision.action.value == "respond":
        return {
            "reasoning": decision.reasoning,
            "action": "respond",
            "response": decision.response
        }
    else:  # use_tool
        return {
            "reasoning": decision.reasoning,
            "action": "use_tool",
            "tool_call": {
                "tool_name": decision.tool_name,
                "parameters": decision.parameters
            }
        }


def generate_final_answer(
    client: genai.Client,
    dataset_name: str,
    session_id: str,
    user_query: str,
    ground_truth_label: str,
    conversation_history: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7
) -> FinalAnswer:
    """
    Generate the final answer using ground truth label as proxy for e-tokens.

    This function simulates what will happen when e-tokens are available:
    the model will receive classification results from a pretrained classifier
    and use them to generate a natural response to the user.

    Args:
        client: Gemini client instance
        dataset_name: Name of the dataset
        session_id: ID of the session being analyzed
        user_query: The user's original question
        ground_truth_label: Ground truth activity label (proxy for e-tokens)
        conversation_history: List of previous turns (tool calls + results)
        model: Which Gemini model to use
        temperature: Sampling temperature

    Returns:
        FinalAnswer with reasoning, final_answer, confidence, and explanation
    """
    # Load prompts
    system_instructions = load_system_instructions()

    with open(FINAL_ANSWER_PROMPT, 'r', encoding='utf-8') as f:
        final_answer_template = f.read()

    # Format conversation history (same as in generate_next_step)
    if not conversation_history:
        history_text = "No preprocessing steps taken."
    else:
        history_parts = []
        for i, turn in enumerate(conversation_history, 1):
            history_parts.append(f"**Step {i}:**")
            history_parts.append(f"Reasoning: {turn['reasoning']}")

            # Check if this is a tool call or a response
            if turn.get('action') == 'use_tool' and 'tool_call' in turn:
                history_parts.append(f"Tool: {turn['tool_call']['tool_name']}")
                history_parts.append(f"Parameters: {json.dumps(turn['tool_call']['parameters'])}")
                history_parts.append(f"Result: {json.dumps(turn['tool_result'], indent=2)}")

                # Highlight new artifact IDs if created
                if isinstance(turn['tool_result'], dict) and 'artifact_id' in turn['tool_result']:
                    new_artifact_id = turn['tool_result']['artifact_id']
                    history_parts.append(f"→ Created new artifact: {new_artifact_id}")

            elif turn.get('action') == 'respond' and 'response' in turn:
                history_parts.append(f"Action: respond")
                history_parts.append(f"Response: {turn['response']}")

            else:
                # Fallback for unknown turn type
                history_parts.append(f"Action: {turn.get('action', 'unknown')}")

            history_parts.append("")
        history_text = '\n'.join(history_parts)

    # Fill in template
    prompt = final_answer_template.replace("{{DATASET_NAME}}", dataset_name)
    prompt = prompt.replace("{{SESSION_ID}}", session_id)
    prompt = prompt.replace("{{USER_QUERY}}", user_query)
    prompt = prompt.replace("{{GROUND_TRUTH_LABEL}}", ground_truth_label)
    prompt = prompt.replace("{{CONVERSATION_HISTORY}}", history_text)

    # Combine system instructions and prompt
    full_prompt = f"{system_instructions}\n\n---\n\n{prompt}"

    # Configure with structured output
    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=2048,
        response_schema=FinalAnswer.model_json_schema(),
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
        error_msg = "Failed to parse final answer response from Gemini\n"
        error_msg += f"Response text: {response.text if hasattr(response, 'text') else 'N/A'}\n"
        error_msg += f"Response object: {response}\n"
        raise ValueError(error_msg)

    final_answer = FinalAnswer.model_validate(response.parsed)

    # Validate that final_answer matches ground truth
    if final_answer.final_answer != ground_truth_label:
        # Log warning but don't fail - the model might format it slightly differently
        print(f"⚠ Warning: Final answer '{final_answer.final_answer}' doesn't match ground truth '{ground_truth_label}'")

    return final_answer


def format_final_answer_for_storage(final_answer: FinalAnswer) -> Dict[str, Any]:
    """
    Format a FinalAnswer into storage format.

    Args:
        final_answer: The final answer from Gemini

    Returns:
        Dictionary suitable for JSON storage
    """
    return {
        "reasoning": final_answer.reasoning,
        "action": "final_answer",
        "final_answer": final_answer.final_answer,
        "confidence": final_answer.confidence,
        "explanation": final_answer.explanation
    }
