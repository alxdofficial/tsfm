"""
Thread-Based Chain Generator

Two modes:
1. Start new thread: Initialize conversation with user query
2. Continue thread: Load existing active threads and continue generation

Each step saves immediately to disk (crash-resistant).

Usage:
    python datascripts/generate_tool_chains/generate_simple.py
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from tools.tool_executor import execute_tool, load_manifest, load_labels, load_session_as_artifact
from datascripts.generate_tool_chains.generation import start_new_thread, generate_next_step
from datascripts.generate_tool_chains.schemas import NextStepDecision
from artifacts import get_artifact_metadata


# ============================================================================
# HARDCODED CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# GCP Settings
PROJECT_ID = "research-459618"  # ← CHANGE THIS
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

# Dataset Settings
DATASETS = ["uci_har", "pamap2", "mhealth", "wisdm", "actionsense"]  # Randomly choose from these

# Temperature Settings
QUERY_TEMP = 0.9  # Higher = more diverse queries
STEP_TEMP = 0.7   # Balanced creativity + consistency

# Behavior
MAX_TURNS = 10  # Maximum turns per thread
VERBOSE = True

# Output
THREADS_DIR = Path("data/tool_chain_of_thoughts/threads")
DATA_ROOT = Path("data")

# ============================================================================


def calculate_thread_metadata(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate metadata from messages array."""
    # Count tool calls (assistant messages with tool_calls)
    num_tool_calls = sum(
        1 for m in messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )

    # Count responds (assistant messages without tool_calls, excluding final answer)
    num_responds = sum(
        1 for m in messages
        if m.get("role") == "assistant" and not m.get("tool_calls")
    ) - 1  # Subtract 1 for final answer
    num_responds = max(0, num_responds)  # Don't go negative

    # Count turns (all assistant + tool messages after the initial user query)
    num_turns = len([
        m for m in messages
        if m["role"] in ["assistant", "tool"]
    ])

    return {
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "num_responds": num_responds
    }


def save_thread(thread: Dict[str, Any]) -> None:
    """Save thread to disk immediately in OpenAI format."""
    THREADS_DIR.mkdir(parents=True, exist_ok=True)

    # Update last modified timestamp
    thread["metadata"]["last_modified"] = datetime.now(timezone.utc).isoformat()

    # Update calculated metadata from messages
    calculated = calculate_thread_metadata(thread["messages"])
    thread["metadata"].update(calculated)

    output_path = THREADS_DIR / f"{thread['id']}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(thread, f, indent=2)

    if VERBOSE:
        print(f"💾 Saved: {output_path}")


def load_all_threads() -> List[Dict[str, Any]]:
    """Load all thread files from disk."""
    if not THREADS_DIR.exists():
        return []

    threads = []
    for thread_file in THREADS_DIR.glob("*.json"):
        with open(thread_file, 'r', encoding='utf-8') as f:
            threads.append(json.load(f))

    # Sort by last_modified (most recent first)
    threads.sort(key=lambda t: t.get("metadata", {}).get("last_modified", ""), reverse=True)
    return threads


def create_new_thread(
    dataset_name: str,
    session_id: str,
    ground_truth: str,
    user_query: str,
    manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a new thread structure in OpenAI format with initial timeseries artifact."""
    thread_id = f"{dataset_name}_{session_id}_v4_{random.randint(1, 9999):04d}"

    # Load session as artifact
    artifact_id = load_session_as_artifact(
        dataset_name=dataset_name,
        session_id=session_id,
        created_at_turn=0,
        created_by="user"
    )

    # Get artifact metadata for display
    artifact_meta = get_artifact_metadata(artifact_id)

    return {
        "id": thread_id,
        "metadata": {
            "dataset": dataset_name,
            "session": session_id,
            "ground_truth": ground_truth,
            "status": "active",
            "num_turns": 0,
            "num_tool_calls": 0,
            "num_responds": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_modified": datetime.now(timezone.utc).isoformat(),
            "schema_version": "openai_v1",
            "gemini_model": MODEL,
            "query_temperature": QUERY_TEMP,
            "step_temperature": STEP_TEMP,
            "user_query": user_query,  # Store plain query for easy access
            "initial_artifact_id": artifact_id  # Track initial artifact
        },
        "messages": [
            # Message 1: Dataset manifest
            {
                "role": "user",
                "content": f"Dataset Information:\n{json.dumps(manifest, indent=2)}"
            },
            # Message 2: Timeseries artifact
            {
                "role": "user",
                "content": {
                    "type": "timeseries",
                    "artifact_id": artifact_id
                }
            },
            # Message 3: User query
            {
                "role": "user",
                "content": f"User Query: {user_query}"
            }
        ]
    }


def messages_to_conversation_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI messages format to conversation_history format for generate_next_step.

    Args:
        messages: List of OpenAI format messages

    Returns:
        List of conversation turns in the format expected by generate_next_step
    """
    conversation_history = []
    turn = 1
    i = 0

    while i < len(messages):
        msg = messages[i]

        # Skip initial 3 user messages (manifest, artifact, query)
        if i < 3 and msg["role"] == "user":
            i += 1
            continue

        # Handle assistant message with tool call
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            tool_call = msg["tool_calls"][0]  # Assume single tool call per message

            # Next message should be the tool result
            if i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                tool_result_msg = messages[i + 1]

                # Parse tool result - handle both dict and string
                tool_result_content = tool_result_msg["content"]
                if isinstance(tool_result_content, str):
                    tool_result = json.loads(tool_result_content)
                else:
                    tool_result = tool_result_content

                conversation_history.append({
                    "turn": turn,
                    "action": "use_tool",
                    "reasoning": msg["content"],
                    "tool_call": {
                        "tool_name": tool_call["function"]["name"],
                        "parameters": json.loads(tool_call["function"]["arguments"])
                    },
                    "tool_result": tool_result
                })

                turn += 1
                i += 2  # Skip both assistant and tool messages
            else:
                i += 1

        # Handle assistant message without tool call (respond)
        elif msg["role"] == "assistant" and not msg.get("tool_calls"):
            conversation_history.append({
                "turn": turn,
                "action": "respond",
                "reasoning": msg["content"],
                "response": msg["content"]
            })
            turn += 1
            i += 1

        else:
            i += 1

    return conversation_history


def start_new_thread_mode(client: genai.Client) -> Optional[Dict[str, Any]]:
    """Mode 1: Start a new thread with a generated query."""

    print("\n" + "="*80)
    print("MODE: START NEW THREAD")
    print("="*80)

    # Randomly pick dataset and session
    dataset = random.choice(DATASETS)
    labels_dict = load_labels(dataset)
    session_id = random.choice(list(labels_dict.keys()))
    ground_truth = labels_dict[session_id][0]

    print(f"\nDataset: {dataset}")
    print(f"Session: {session_id}")
    print(f"Ground truth: {ground_truth}")

    # Load manifest
    manifest = load_manifest(dataset)

    # Step 1: Generate initial query
    print("\n" + "-"*80)
    print("STEP: Generating initial user query...")
    print("-"*80)

    while True:
        user_query = start_new_thread(
            client=client,
            manifest=manifest,
            model=MODEL,
            temperature=QUERY_TEMP
        )

        print(f"\n💬 Query: \"{user_query}\"")
        print("\n[y] Accept and save  [n] Retry  [x] Cancel")
        response = input("→ ").strip().lower()

        if response == 'x':
            print("❌ Cancelled")
            return None
        elif response == 'n':
            print("🔄 Regenerating query...")
            continue
        elif response == 'y':
            # Create thread and save immediately
            thread = create_new_thread(dataset, session_id, ground_truth, user_query, manifest)
            save_thread(thread)
            print(f"✅ Thread created: {thread['id']}")
            return thread
        else:
            print("Invalid input. Use y/n/x")


def generate_step_interactive(
    client: genai.Client,
    thread: Dict[str, Any]
) -> bool:
    """
    Generate one step interactively.

    Returns:
        True if should continue, False if finished or user exits
    """
    # Calculate current turn from messages
    num_turns = len([m for m in thread["messages"] if m["role"] in ["assistant", "tool"]])
    turn = num_turns + 1

    print(f"\n--- Turn {turn} ---")

    # Extract user query from metadata
    user_query = thread["metadata"]["user_query"]

    # Extract most recent artifact ID from messages
    # Look for the most recent artifact in user or tool messages
    artifact_id = thread["metadata"].get("initial_artifact_id")  # Fallback to initial

    for msg in reversed(thread["messages"]):
        # Check tool results for artifact returns
        if msg.get("role") == "tool":
            content = msg.get("content")
            if isinstance(content, dict) and "artifact_id" in content:
                artifact_id = content["artifact_id"]
                break
        # Check user messages for artifacts
        elif msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, dict) and "artifact_id" in content:
                artifact_id = content["artifact_id"]
                break

    # Convert messages to conversation_history for generate_next_step
    conversation_history = messages_to_conversation_history(thread["messages"])

    while True:
        decision = generate_next_step(
            client=client,
            dataset_name=thread["metadata"]["dataset"],
            session_id=thread["metadata"]["session"],
            user_query=user_query,
            conversation_history=conversation_history,
            artifact_id=artifact_id,
            model=MODEL,
            temperature=STEP_TEMP
        )

        print(f"\n🎯 Action: {decision.action.value}")
        print(f"💭 Reasoning: {decision.reasoning}")

        # Display action-specific content
        if decision.action.value == "respond":
            print(f"\n📝 Response: {decision.response}")

        elif decision.action.value == "use_tool":
            print(f"\n🔧 Tool: {decision.tool_name}")
            print(f"📋 Parameters: {json.dumps(decision.parameters, indent=2)}")

        print("\n[y] Accept and save  [n] Retry  [x] Exit")
        response = input("→ ").strip().lower()

        if response == 'x':
            # Exit without saving (already saved on each 'y')
            return False

        elif response == 'n':
            print("🔄 Regenerating step...")
            continue

        elif response == 'y':
            # Accept and save
            if decision.action.value == "respond":
                # Add assistant response message
                thread["messages"].append({
                    "role": "assistant",
                    "content": decision.response,
                    "tool_calls": None
                })
                save_thread(thread)
                return True

            elif decision.action.value == "use_tool":
                # Execute tool
                try:
                    # Validate artifact_id if present in parameters
                    if "artifact_id" in decision.parameters:
                        from artifacts import get_artifact
                        requested_id = decision.parameters["artifact_id"]
                        try:
                            get_artifact(requested_id)  # Will raise KeyError if not found
                        except KeyError:
                            print(f"❌ Invalid artifact_id: '{requested_id}'")
                            if artifact_id:
                                print(f"   Current artifact (use this): {artifact_id}")
                            print("\n⚠️  The LLM made up an artifact ID instead of using the real one.")
                            print("    Retrying will regenerate with correct ID in prompt.")
                            continue

                    # Pass turn number to execute_tool for artifact creation
                    tool_result = execute_tool(
                        decision.tool_name,
                        decision.parameters,
                        turn_number=turn
                    )
                    print("✅ Tool executed successfully")
                    print(f"\n📊 Tool Result:")
                    print(json.dumps(tool_result, indent=2))

                    # If tool returned an artifact, show it
                    if isinstance(tool_result, dict) and "type" in tool_result and "artifact_id" in tool_result:
                        print(f"📦 Created new artifact: {tool_result['artifact_id']}")

                    # Generate unique tool_call_id
                    tool_call_id = f"call_{thread['id']}_turn_{turn}"

                    # Add assistant message with tool call
                    thread["messages"].append({
                        "role": "assistant",
                        "content": decision.reasoning,
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": decision.tool_name,
                                "arguments": json.dumps(decision.parameters)
                            }
                        }]
                    })

                    # Add tool result message - store as dict, not JSON string
                    thread["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": decision.tool_name,
                        "content": tool_result  # Store as dict
                    })

                    save_thread(thread)
                    return True

                except Exception as e:
                    print(f"❌ Tool execution failed: {e}")
                    print("Retry with different parameters?")
                    continue

        else:
            print("Invalid input. Use y/n/x")


def display_conversation_histogram(all_threads: List[Dict[str, Any]]) -> None:
    """Display a histogram of conversation lengths."""
    if not all_threads:
        return

    # Count threads by number of turns
    turn_counts = {}
    for thread in all_threads:
        num_turns = thread["metadata"]["num_turns"]
        turn_counts[num_turns] = turn_counts.get(num_turns, 0) + 1

    # Display histogram
    print("\n📊 Conversation Length Distribution:")
    max_count = max(turn_counts.values())
    max_bar_width = 40

    for turns in sorted(turn_counts.keys()):
        count = turn_counts[turns]
        bar_width = int((count / max_count) * max_bar_width)
        bar = "█" * bar_width
        print(f"  {turns:2d} turns: {bar} {count}")


def continue_thread_mode(client: genai.Client) -> Optional[Dict[str, Any]]:
    """Mode 2: Continue an existing active thread."""

    print("\n" + "="*80)
    print("MODE: CONTINUE EXISTING THREAD")
    print("="*80)

    # Load all threads
    all_threads = load_all_threads()
    active_threads = [t for t in all_threads if t["metadata"]["status"] == "active"]

    if not active_threads:
        print("\n⚠️  No active threads found")
        return None

    # Ask for filter
    max_turns_input = input("\nFilter by max turns (press Enter for all): ").strip()
    max_turns_filter = None
    if max_turns_input:
        try:
            max_turns_filter = int(max_turns_input)
            active_threads = [
                t for t in active_threads
                if t["metadata"]["num_turns"] <= max_turns_filter
            ]
            if not active_threads:
                print(f"\n⚠️  No active threads with ≤{max_turns_filter} turns")
                return None
            print(f"✓ Showing threads with ≤{max_turns_filter} turns")
        except ValueError:
            print("⚠️  Invalid number, showing all threads")

    # Display threads
    print(f"\n📂 Found {len(active_threads)} active threads:\n")
    for i, thread in enumerate(active_threads, 1):
        meta = thread["metadata"]
        # Extract user query from metadata
        user_query = meta.get("user_query", "N/A")

        print(f"{i}. {thread['id']}")
        print(f"   Dataset: {meta['dataset']} | Session: {meta['session']}")
        print(f"   Query: \"{user_query}\"")
        print(f"   Stats: {meta['num_turns']} turns, {meta['num_tool_calls']} tools, {meta['num_responds']} responds")
        print(f"   Modified: {meta['last_modified']}")
        print()

    # User selects thread
    while True:
        try:
            choice = input(f"Select thread (1-{len(active_threads)}) or [x] to cancel: ").strip().lower()
            if choice == 'x':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(active_threads):
                return active_threads[idx]
            else:
                print(f"Please enter 1-{len(active_threads)}")
        except ValueError:
            print("Invalid input")


def run_thread_generation(client: genai.Client, thread: Dict[str, Any]) -> None:
    """Run generation loop for a thread."""

    meta = thread["metadata"]
    # Extract user query from metadata
    user_query = meta.get("user_query", "N/A")

    print(f"\n{'='*80}")
    print(f"THREAD: {thread['id']}")
    print(f"{'='*80}")
    print(f"Query: \"{user_query}\"")
    print(f"Dataset: {meta['dataset']} | Session: {meta['session']}")
    print(f"Current turns: {meta['num_turns']}")

    # Generate steps - count only assistant+tool messages in messages array
    while meta["num_turns"] < MAX_TURNS:
        should_continue = generate_step_interactive(client, thread)
        if not should_continue:
            break
        # Refresh metadata after each step
        meta = thread["metadata"]

    if meta["num_turns"] >= MAX_TURNS and meta["status"] == "active":
        print(f"\n⚠️  Reached maximum turns ({MAX_TURNS})")
        thread["metadata"]["status"] = "max_turns_reached"
        save_thread(thread)


def export_mode() -> None:
    """Mode 3: Export threads to JSONL training format."""

    print("\n" + "="*80)
    print("MODE: EXPORT TO JSONL TRAINING FORMAT")
    print("="*80)
    print("\nℹ️  Threads are already stored in OpenAI format!")
    print("   This will copy them to a JSONL file for training.")

    # Ask for filter
    print("\nFilter:")
    print("  [1] Finished threads only (recommended)")
    print("  [2] All threads (including unfinished)")
    print("  [x] Cancel")

    filter_choice = input("\nSelect: ").strip().lower()

    if filter_choice == 'x':
        return

    filter_finished = filter_choice != '2'

    # Load and filter threads
    print("\n🔄 Loading threads...")
    all_threads = load_all_threads()

    if filter_finished:
        threads_to_export = [t for t in all_threads if t["metadata"]["status"] == "finished"]
    else:
        threads_to_export = all_threads

    if not threads_to_export:
        print("\n⚠️  No threads found matching the filter")
        input("\nPress Enter to continue...")
        return

    # Create output directory and file
    output_dir = Path("data/training_exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"training_data_openai_{timestamp}.jsonl"

    # Write to JSONL
    print(f"\n📝 Writing {len(threads_to_export)} threads to JSONL...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for thread in threads_to_export:
            f.write(json.dumps(thread) + '\n')

    # Show summary
    print("\n" + "="*80)
    print("Export Summary")
    print("="*80)
    print(f"Total threads: {len(all_threads)}")
    print(f"Exported: {len(threads_to_export)}")
    print(f"Skipped: {len(all_threads) - len(threads_to_export)}")
    print(f"\n✅ Output file: {output_file}")

    # Show preview
    print("\nPreview (first 2 examples):")
    for i, thread in enumerate(threads_to_export[:2], 1):
        meta = thread.get("metadata", {})
        print(f"\n{i}. {thread['id']}")
        print(f"   Messages: {len(thread.get('messages', []))}")
        print(f"   Dataset: {meta.get('dataset')}")
        print(f"   Status: {meta.get('status')}")
        print(f"   Turns: {meta.get('num_turns')} | Tools: {meta.get('num_tool_calls')}")
        if meta.get('is_correct') is not None:
            print(f"   Correct: {meta.get('is_correct')}")

    print("\n" + "="*80)
    print("ℹ️  This file is ready for Llama fine-tuning with:")
    print("   - Axolotl (set type: sharegpt, conversation: openai)")
    print("   - LLaMA-Factory (set formatting: sharegpt)")
    print("   - TRL (use datasets.load_dataset)")
    print("="*80)

    input("\nPress Enter to continue...")


def main():
    """Main entry point."""

    print("="*80)
    print("Thread-Based Chain Generator")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Location: {LOCATION}")
    print(f"  Model: {MODEL}")
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Max turns: {MAX_TURNS}")
    print(f"  Threads dir: {THREADS_DIR}")
    print()

    # Validate datasets
    for dataset in DATASETS:
        if not (DATA_ROOT / dataset / "manifest.json").exists():
            print(f"✗ Error: Dataset not found: {dataset}")
            return 1
    print(f"✅ All {len(DATASETS)} datasets validated")

    # Initialize client
    try:
        client = genai.Client(
            vertexai=True,
            location=LOCATION,
            project=PROJECT_ID
        )
        print("✅ Gemini client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Gemini client: {e}")
        return 1

    # Main loop
    while True:
        # Load and display thread statistics
        all_threads = load_all_threads()
        active_threads = [t for t in all_threads if t["metadata"]["status"] == "active"]
        finished_threads = [t for t in all_threads if t["metadata"]["status"] == "finished"]

        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print(f"📈 Total threads: {len(all_threads)} ({len(active_threads)} active, {len(finished_threads)} finished)")

        # Show histogram if there are threads
        if all_threads:
            display_conversation_histogram(all_threads)

        print("\n[1] Start new thread")
        print("[2] Continue existing thread")
        print("[3] Export to training format")
        print("[q] Quit")

        choice = input("\n→ ").strip().lower()

        if choice == 'q':
            print("\n👋 Goodbye!")
            break

        elif choice == '1':
            thread = start_new_thread_mode(client)
            if thread:
                run_thread_generation(client, thread)

        elif choice == '2':
            thread = continue_thread_mode(client)
            if thread:
                run_thread_generation(client, thread)

        elif choice == '3':
            export_mode()
        else:
            print("Invalid choice. Use 1/2/3/q")

    return 0


if __name__ == "__main__":
    sys.exit(main())
