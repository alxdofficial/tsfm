"""
Simple Test Script for V2 Generation

Tests the two core functions and one full chain.

Usage:
    python datascripts/generate_tool_chains/test_simple.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from datascripts.generate_tool_chains.generation import generate_query, generate_next_step
from tools.tool_executor import load_manifest, load_labels


# ============================================================================
# HARDCODED CONFIGURATION - EDIT THESE VALUES
# ============================================================================

PROJECT_ID = "research-459618"  # ← CHANGE THIS
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"
DATASET = "uci_har"

# ============================================================================


def test_query_generation(client):
    """Test query generation function."""
    print("\n" + "="*80)
    print("TEST 1: Query Generation")
    print("="*80)

    manifest = load_manifest(DATASET)

    print("\nGenerating 3 diverse queries...")
    for i in range(3):
        query = generate_query(
            client=client,
            manifest=manifest,
            model=MODEL,
            temperature=0.9
        )
        print(f"\nQuery {i+1}: \"{query}\"")

    print("\n✓ Query generation working!")


def test_next_step_generation(client):
    """Test next step generation function."""
    print("\n" + "="*80)
    print("TEST 2: Next Step Generation")
    print("="*80)

    # Test with no conversation history
    labels_dict = load_labels(DATASET)
    session_id = list(labels_dict.keys())[0]

    print(f"\nGenerating first step for session: {session_id}")
    print("User query: \"What activity is happening in this session?\"")

    decision = generate_next_step(
        client=client,
        dataset_name=DATASET,
        session_id=session_id,
        user_query="What activity is happening in this session?",
        conversation_history=[],
        model=MODEL,
        temperature=0.7
    )

    print(f"\nAction: {decision.action.value}")
    print(f"Reasoning: {decision.reasoning[:200]}...")

    if decision.action.value == "use_tool":
        print(f"Tool: {decision.tool_name}")
        print(f"Parameters: {decision.parameters}")
    else:
        print(f"Classification: {decision.classification}")
        print(f"Confidence: {decision.confidence}")

    print("\n✓ Next step generation working!")


def test_full_chain(client):
    """Test generating a complete chain."""
    print("\n" + "="*80)
    print("TEST 3: Full Chain Generation")
    print("="*80)

    from generate_simple import generate_one_chain

    labels_dict = load_labels(DATASET)
    session_id = list(labels_dict.keys())[0]
    ground_truth = labels_dict[session_id][0]

    print(f"\nGenerating chain for session: {session_id}")
    print(f"Ground truth: {ground_truth}")

    # Temporarily disable interactive mode for testing
    import generate_simple
    original_interactive = generate_simple.INTERACTIVE
    generate_simple.INTERACTIVE = False

    try:
        chain = generate_one_chain(
            client=client,
            dataset_name=DATASET,
            session_id=session_id,
            ground_truth=ground_truth
        )

        if chain:
            print(f"\n✓ Classification: {chain['final_classification']}")
            print(f"  Confidence: {chain['confidence']}")
            print(f"  Turns: {len(chain['conversation'])}")
            print(f"  Correct: {chain['is_correct']}")
            print("\n✓ Full chain generation working!")
        else:
            print("\n✗ Chain generation returned None")

    finally:
        generate_simple.INTERACTIVE = original_interactive


def main():
    """Run all tests."""
    print("="*80)
    print("V2 Generation System Tests")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Location: {LOCATION}")
    print(f"  Model: {MODEL}")
    print(f"  Dataset: {DATASET}")

    # Initialize client
    try:
        client = genai.Client(
            vertexai=True,
            location=LOCATION,
            project=PROJECT_ID
        )
        print("\n✓ Gemini client initialized")
    except Exception as e:
        print(f"\n✗ Failed to initialize client: {e}")
        return 1

    # Run tests
    try:
        test_query_generation(client)
        test_next_step_generation(client)
        test_full_chain(client)

        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
