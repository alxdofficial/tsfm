"""
Classification Chain Generation V2

Uses the two-function approach:
1. generate_query: Create diverse user queries
2. generate_next_step: Generate next step with structured outputs

Usage:
    python datascripts/generate_tool_chains/generate_chains_v2.py

Edit hardcoded config at top of file.
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from tools.tool_executor import (
    execute_tool, load_manifest, load_labels, get_session_paths
)
from datascripts.generate_tool_chains.generation import generate_query, generate_next_step, format_next_step_for_storage
from datascripts.generate_tool_chains.schemas import NextStepDecision


# ============================================================================
# HARDCODED CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# GCP Settings
PROJECT_ID = "research-459618"  # ← CHANGE THIS
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

# Dataset Settings
DATASETS = ["uci_har", "pamap2", "mhealth", "wisdm"]  # Randomly choose from these
NUM_SAMPLES = 5

# Temperature Settings
QUERY_TEMP = 0.9  # Higher = more diverse queries
STEP_TEMP = 0.7   # Balanced creativity + consistency

# Behavior
MAX_TURNS = 5
VERBOSE = True
INTERACTIVE = False  # Set to True for manual approval

# Output
OUTPUT_DIR = Path("data/tool_chain_of_thoughts")
DATA_ROOT = Path("data")

# ============================================================================


def generate_classification_chain(
    client: genai.Client,
    project_id: str,
    location: str,
    dataset_name: str,
    session_id: str,
    ground_truth: str,
    model: str = "gemini-2.5-flash",
    query_temperature: float = 0.9,
    step_temperature: float = 0.7,
    max_turns: int = 5,
    verbose: bool = True,
    interactive: bool = False
) -> Dict[str, Any]:
    """
    Generate a complete classification chain using the two-function approach.

    Args:
        client: Gemini client
        project_id: GCP project
        location: GCP region
        dataset_name: Dataset to use
        session_id: Session to classify
        ground_truth: Actual label (for evaluation)
        model: Gemini model name
        query_temperature: Temperature for query generation (higher = more diverse)
        step_temperature: Temperature for step generation
        max_turns: Maximum conversation turns
        verbose: Print progress
        interactive: Enable manual approval at each step (query, tool use, classification, save)

    Returns:
        Complete conversation chain dictionary, or None if rejected/incomplete
    """
    # Load manifest
    manifest = load_manifest(dataset_name)

    # Step 1: Generate user query
    if verbose:
        print("\n" + "="*80)
        print("STEP 1: Generating user query...")
        print("="*80)

    # Query generation with optional regeneration
    while True:
        user_query = generate_query(
            client=client,
            manifest=manifest,
            model=model,
            temperature=query_temperature
        )

        if verbose:
            print(f"\nGenerated query: \"{user_query}\"")

        if interactive:
            response = input("\nAccept this query? [Y/n/r to regenerate]: ").strip().lower()
            if response == 'n':
                if verbose:
                    print("Query rejected. Skipping this chain.")
                return None
            elif response == 'r':
                if verbose:
                    print("Regenerating query...")
                continue

        # Query accepted
        break

    # Initialize conversation
    conv_id = f"{dataset_name}_{session_id}_v2_{random.randint(1, 999):03d}"

    conversation = {
        "conversation_id": conv_id,
        "dataset_name": dataset_name,
        "session_id": session_id,
        "ground_truth_label": ground_truth,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "2.0",
        "generation_metadata": {
            "gemini_model": model,
            "query_temperature": query_temperature,
            "step_temperature": step_temperature,
            "phase": "phase1_classification_v2",
            "tools_available": ["show_channel_stats", "select_channels"],
            "vertex_ai_project": project_id,
            "vertex_ai_location": location,
            "generation_method": "two_function_structured_output"
        },
        "initial_context": {
            "user_query": user_query,
            "manifest": manifest,
            "session_id": session_id
        },
        "conversation": [],
        "final_classification": None,
        "confidence": None,
        "explanation": None,
        "is_correct": None,
        "execution_verified": True
    }

    # Step 2: Iterative next-step generation
    if verbose:
        print("\n" + "="*80)
        print("STEP 2: Generating conversation steps...")
        print("="*80)

    turn = 1
    while turn <= max_turns:
        if verbose:
            print(f"\n--- Turn {turn} ---")

        # Generate next step with optional regeneration
        while True:
            decision = generate_next_step(
                client=client,
                dataset_name=dataset_name,
                session_id=session_id,
                user_query=user_query,
                conversation_history=conversation["conversation"],
                model=model,
                temperature=step_temperature
            )

            if verbose:
                print(f"\nAction: {decision.action.value}")
                print(f"Reasoning: {decision.reasoning}")

            if decision.action.value == "classify":
                if verbose:
                    print(f"\n✓ Classification: {decision.classification}")
                    print(f"  Confidence: {decision.confidence}")
                    print(f"  Explanation: {decision.explanation}")
                    print(f"  Ground truth: {ground_truth}")

                if interactive:
                    response = input("\nAccept this classification? [Y/n/r to regenerate]: ").strip().lower()
                    if response == 'n':
                        if verbose:
                            print("Classification rejected. Skipping this chain.")
                        return None
                    elif response == 'r':
                        if verbose:
                            print("Regenerating step...")
                        continue
                # Classification accepted
                break
            else:
                # Tool use
                if verbose:
                    print(f"  Tool: {decision.tool_name}")
                    print(f"  Parameters: {json.dumps(decision.parameters, indent=2)}")

                if interactive:
                    response = input("\nExecute this tool? [Y/n/s to skip turn/r to regenerate]: ").strip().lower()
                    if response == 'n':
                        if verbose:
                            print("Tool use rejected. Skipping this chain.")
                        return None
                    elif response == 's':
                        if verbose:
                            print("Skipping this turn.")
                        turn += 1
                        break
                    elif response == 'r':
                        if verbose:
                            print("Regenerating step...")
                        continue
                # Tool use accepted
                break

        if decision.action.value == "classify":
            # Final classification reached - store results
            conversation["final_classification"] = decision.classification
            conversation["confidence"] = decision.confidence
            conversation["explanation"] = decision.explanation
            conversation["is_correct"] = (
                decision.classification.lower() == ground_truth.lower()
            )
            break

        # Execute tool

        try:
            tool_result = execute_tool(decision.tool_name, decision.parameters)
            if verbose:
                print(f"✓ Tool executed successfully")
        except Exception as e:
            if verbose:
                print(f"✗ Tool execution failed: {e}")
            break

        # Record turn
        conversation["conversation"].append({
            "turn": turn,
            "reasoning": decision.reasoning,
            "tool_call": {
                "tool_name": decision.tool_name,
                "parameters": decision.parameters
            },
            "tool_result": tool_result
        })

        turn += 1

    # Check if classification was reached
    if conversation["final_classification"] is None:
        if verbose:
            print("\n⚠ No final classification reached")
        return None

    if verbose:
        print(f"\n✓ Chain complete!")
        print(f"  Conversation ID: {conv_id}")
        print(f"  Turns: {len(conversation['conversation'])}")
        print(f"  Correct: {conversation['is_correct']}")

    # Final save confirmation in interactive mode
    if interactive:
        print("\n" + "="*80)
        print("CHAIN SUMMARY:")
        print(f"  Query: \"{user_query}\"")
        print(f"  Turns: {len(conversation['conversation'])}")
        print(f"  Classification: {conversation['final_classification']}")
        print(f"  Confidence: {conversation['confidence']}")
        print(f"  Correct: {conversation['is_correct']}")
        print("="*80)
        response = input("\nSave this chain? [Y/n]: ").strip().lower()
        if response == 'n':
            if verbose:
                print("Chain discarded.")
            return None

    return conversation


def main():
    """Main entry point."""

    print("="*80)
    print("Classification Chain Generator V2")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Location: {LOCATION}")
    print(f"  Model: {MODEL}")
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Query temp: {QUERY_TEMP}")
    print(f"  Step temp: {STEP_TEMP}")
    print(f"  Interactive: {INTERACTIVE}")
    print()

    # Validate all datasets
    for dataset in DATASETS:
        if not (DATA_ROOT / dataset / "manifest.json").exists():
            print(f"✗ Error: Dataset not found: {dataset}")
            print(f"  Expected: {DATA_ROOT / dataset / 'manifest.json'}")
            return 1
    print(f"✓ All {len(DATASETS)} datasets validated")

    # Initialize client
    try:
        client = genai.Client(
            vertexai=True,
            location=LOCATION,
            project=PROJECT_ID
        )
        print("✓ Gemini client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Gemini client: {e}")
        return 1

    # Generate chains
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    successful = 0

    for i in range(NUM_SAMPLES):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{NUM_SAMPLES}")
        print(f"{'='*80}")

        # Randomly pick a dataset
        dataset = random.choice(DATASETS)
        print(f"Dataset: {dataset}")

        # Load labels for this dataset
        labels_dict = load_labels(dataset)

        # Pick random session from this dataset
        session_id = random.choice(list(labels_dict.keys()))
        ground_truth = labels_dict[session_id][0]

        try:
            chain = generate_classification_chain(
                client=client,
                project_id=PROJECT_ID,
                location=LOCATION,
                dataset_name=dataset,
                session_id=session_id,
                ground_truth=ground_truth,
                model=MODEL,
                query_temperature=QUERY_TEMP,
                step_temperature=STEP_TEMP,
                max_turns=MAX_TURNS,
                verbose=VERBOSE,
                interactive=INTERACTIVE
            )

            if chain is None:
                print("Skipped")
                continue

            # Save
            output_path = OUTPUT_DIR / f"{chain['conversation_id']}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chain, f, indent=2)

            print(f"\n✓ Saved: {output_path}")
            successful += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            if VERBOSE:
                import traceback
                traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"Generated {successful}/{NUM_SAMPLES} chains")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
