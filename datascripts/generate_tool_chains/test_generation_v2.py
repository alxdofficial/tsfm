"""
Test the V2 generation functions.

Tests:
1. Query generation with structured output
2. Next step generation with structured output
3. Full chain generation

Usage:
    python datascripts/generate_tool_chains/test_generation_v2.py --project your-project-id
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from tools.tool_executor import load_manifest
from datascripts.generate_tool_chains.generation import generate_query, generate_next_step


def test_query_generation(client: genai.Client, dataset_name: str):
    """Test query generation."""
    print("\n" + "="*80)
    print("TEST 1: Query Generation")
    print("="*80)

    manifest = load_manifest(dataset_name)

    print(f"Dataset: {dataset_name}")
    print("Generating 3 diverse queries...")

    for i in range(3):
        query = generate_query(
            client=client,
            manifest=manifest,
            temperature=0.9  # High temp for diversity
        )
        print(f"\nQuery {i+1}: \"{query}\"")

    print("\n✓ Query generation test passed!")


def test_next_step_generation(client: genai.Client, dataset_name: str):
    """Test next step generation."""
    print("\n" + "="*80)
    print("TEST 2: Next Step Generation")
    print("="*80)

    # Test scenario: User just asked, no tools used yet
    decision = generate_next_step(
        client=client,
        dataset_name=dataset_name,
        session_id="session_001",
        user_query="What activity is happening in this session?",
        conversation_history=[],
        temperature=0.7
    )

    print(f"\nAction: {decision.action.value}")
    print(f"Reasoning: {decision.reasoning[:200]}...")

    if decision.action.value == "use_tool":
        print(f"Tool: {decision.tool_name}")
        print(f"Parameters: {json.dumps(decision.parameters, indent=2)}")
    else:
        print(f"Classification: {decision.classification}")
        print(f"Confidence: {decision.confidence}")

    print(f"\nValidation: {decision.validate_completeness()}")
    print("\n✓ Next step generation test passed!")


def test_full_chain(client: genai.Client, project_id: str, location: str, dataset_name: str):
    """Test full chain generation."""
    print("\n" + "="*80)
    print("TEST 3: Full Chain Generation")
    print("="*80)

    from tools.tool_executor import load_labels
    import random

    labels = load_labels(dataset_name)
    session_id = random.choice(list(labels.keys()))
    ground_truth = labels[session_id][0]

    from datascripts.generate_chains_v2 import generate_classification_chain

    chain = generate_classification_chain(
        client=client,
        project_id=project_id,
        location=location,
        dataset_name=dataset_name,
        session_id=session_id,
        ground_truth=ground_truth,
        verbose=True
    )

    if chain:
        print("\n✓ Full chain generation test passed!")
        print(f"  Generated: {chain['conversation_id']}")
        print(f"  Turns: {len(chain['conversation'])}")
        print(f"  Classification: {chain['final_classification']}")
        print(f"  Correct: {chain['is_correct']}")
    else:
        print("\n✗ Full chain generation failed")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test V2 generation functions")
    parser.add_argument("--project", type=str, required=True, help="GCP project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--dataset", type=str, default="uci_har", help="Dataset to test")

    args = parser.parse_args()

    # Initialize client
    client = genai.Client(
        vertexai=True,
        location=args.location,
        project=args.project
    )

    print("Testing V2 generation functions...")
    print(f"Project: {args.project}")
    print(f"Dataset: {args.dataset}")

    try:
        # Test 1: Query generation
        test_query_generation(client, args.dataset)

        # Test 2: Next step generation
        test_next_step_generation(client, args.dataset)

        # Test 3: Full chain
        test_full_chain(client, args.project, args.location, args.dataset)

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
