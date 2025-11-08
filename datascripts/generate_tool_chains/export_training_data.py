"""
Export thread data to OpenAI format for fine-tuning Llama models.

OpenAI format is the standard for tool-use training data and is compatible with:
- Axolotl
- LLaMA-Factory
- TRL
- Most fine-tuning frameworks

Usage:
    python datascripts/generate_tool_chains/export_training_data.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def convert_thread_to_openai_format(thread: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert our thread format to OpenAI messages format.

    Args:
        thread: Thread dictionary in our format

    Returns:
        Dictionary in OpenAI messages format
    """
    messages = []

    # System message
    system_content = (
        "You are an AI assistant that analyzes time-series sensor data to classify human activities. "
        "You have access to tools for examining sensor channel statistics and selecting specific sensors. "
        "Use the tools to gather information, then provide your classification with confidence level and explanation."
    )
    messages.append({
        "role": "system",
        "content": system_content
    })

    # User query
    messages.append({
        "role": "user",
        "content": thread["initial_context"]["user_query"]
    })

    # Process conversation turns
    for turn in thread["conversation"]:
        if turn["action"] == "respond":
            # Intermediate response without tool
            messages.append({
                "role": "assistant",
                "content": turn["response"],
                "tool_calls": None
            })

        elif turn["action"] == "use_tool":
            # Assistant decides to use tool
            # Combine reasoning with tool call
            messages.append({
                "role": "assistant",
                "content": turn["reasoning"],
                "tool_calls": [{
                    "id": f"call_{thread['thread_id']}_turn_{turn['turn']}",
                    "type": "function",
                    "function": {
                        "name": turn["tool_call"]["tool_name"],
                        "arguments": json.dumps(turn["tool_call"]["parameters"])
                    }
                }]
            })

            # Tool execution result
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{thread['thread_id']}_turn_{turn['turn']}",
                "name": turn["tool_call"]["tool_name"],
                "content": json.dumps(turn["tool_result"])
            })

    # Final answer (if conversation finished)
    if thread["status"] == "finished" and thread.get("final_answer"):
        # Combine reasoning, answer, and explanation
        final_content = f"{thread.get('explanation', '')}\n\nFinal Answer: {thread['final_answer']}"
        messages.append({
            "role": "assistant",
            "content": final_content.strip(),
            "tool_calls": None
        })

    # Build output
    output = {
        "id": thread["thread_id"],
        "metadata": {
            "dataset": thread["dataset_name"],
            "session": thread["session_id"],
            "ground_truth": thread["ground_truth_label"],
            "status": thread["status"],
            "num_turns": thread["metadata"]["num_turns"],
            "num_tool_calls": thread["metadata"]["num_tool_calls"],
            "created_at": thread["created_at"],
            "schema_version": "openai_v1"
        },
        "messages": messages
    }

    # Add correctness if available
    if thread.get("is_correct") is not None:
        output["metadata"]["is_correct"] = thread["is_correct"]
        output["metadata"]["confidence"] = thread.get("confidence")

    return output


def convert_thread_to_sharegpt_format(thread: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert to ShareGPT format (alternative format, simpler).

    Args:
        thread: Thread dictionary in our format

    Returns:
        Dictionary in ShareGPT format
    """
    conversations = []

    # System message
    conversations.append({
        "from": "system",
        "value": "You are an AI assistant that analyzes sensor data to classify human activities."
    })

    # User query
    conversations.append({
        "from": "human",
        "value": thread["initial_context"]["user_query"]
    })

    # Process turns
    for turn in thread["conversation"]:
        if turn["action"] == "respond":
            conversations.append({
                "from": "gpt",
                "value": turn["response"]
            })

        elif turn["action"] == "use_tool":
            # Combine reasoning + tool call in one message
            tool_text = f"{turn['reasoning']}\n\nTool: {turn['tool_call']['tool_name']}\nParameters: {json.dumps(turn['tool_call']['parameters'], indent=2)}"
            conversations.append({
                "from": "gpt",
                "value": tool_text
            })

            # Tool result as system message
            conversations.append({
                "from": "system",
                "value": f"Tool Result:\n{json.dumps(turn['tool_result'], indent=2)}"
            })

    # Final answer
    if thread["status"] == "finished" and thread.get("final_answer"):
        conversations.append({
            "from": "gpt",
            "value": thread["final_answer"]
        })

    return {
        "id": thread["thread_id"],
        "conversations": conversations
    }


def validate_openai_format(data: Dict[str, Any]) -> List[str]:
    """
    Validate OpenAI format data.

    Args:
        data: Data in OpenAI format

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if "messages" not in data:
        errors.append("Missing 'messages' field")
        return errors

    messages = data["messages"]

    # Check message structure
    for i, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"Message {i}: missing 'role'")
        if "content" not in msg and "tool_calls" not in msg:
            errors.append(f"Message {i}: missing 'content' or 'tool_calls'")

        # Validate roles
        valid_roles = {"system", "user", "assistant", "tool"}
        if msg.get("role") not in valid_roles:
            errors.append(f"Message {i}: invalid role '{msg.get('role')}'")

        # Validate tool messages
        if msg.get("role") == "tool":
            if "tool_call_id" not in msg:
                errors.append(f"Message {i}: tool message missing 'tool_call_id'")
            if "name" not in msg:
                errors.append(f"Message {i}: tool message missing 'name'")

        # Validate assistant tool calls
        if msg.get("tool_calls"):
            for j, tc in enumerate(msg["tool_calls"]):
                if "id" not in tc:
                    errors.append(f"Message {i}, tool_call {j}: missing 'id'")
                if "function" not in tc:
                    errors.append(f"Message {i}, tool_call {j}: missing 'function'")
                elif "name" not in tc["function"] or "arguments" not in tc["function"]:
                    errors.append(f"Message {i}, tool_call {j}: invalid function structure")

    return errors


def export_threads(
    threads_dir: Path,
    output_dir: Path,
    format: str = "openai",
    filter_finished_only: bool = True
) -> Dict[str, Any]:
    """
    Batch export all threads to training format.

    Args:
        threads_dir: Directory containing thread JSON files
        output_dir: Directory to save exported files
        format: 'openai' or 'sharegpt'
        filter_finished_only: Only export finished threads

    Returns:
        Dictionary with export statistics
    """
    if not threads_dir.exists():
        return {
            "success": False,
            "error": f"Threads directory not found: {threads_dir}"
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all threads
    thread_files = list(threads_dir.glob("*.json"))

    exported = 0
    skipped = 0
    errors = 0
    validation_errors = []

    all_exports = []

    for thread_file in thread_files:
        try:
            with open(thread_file, 'r', encoding='utf-8') as f:
                thread = json.load(f)

            # Filter if needed
            if filter_finished_only and thread.get("status") != "finished":
                skipped += 1
                continue

            # Convert based on format
            if format == "openai":
                converted = convert_thread_to_openai_format(thread)

                # Validate
                errors_found = validate_openai_format(converted)
                if errors_found:
                    validation_errors.append({
                        "thread_id": thread["thread_id"],
                        "errors": errors_found
                    })
                    errors += 1
                    continue

            elif format == "sharegpt":
                converted = convert_thread_to_sharegpt_format(thread)

            else:
                raise ValueError(f"Unknown format: {format}")

            all_exports.append(converted)
            exported += 1

        except Exception as e:
            print(f"Error processing {thread_file.name}: {e}")
            errors += 1

    # Save all exports
    if all_exports:
        output_file = output_dir / f"training_data_{format}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for export in all_exports:
                f.write(json.dumps(export) + '\n')

        print(f"\n✅ Exported {exported} threads to: {output_file}")

    # Summary
    stats = {
        "success": True,
        "total_files": len(thread_files),
        "exported": exported,
        "skipped": skipped,
        "errors": errors,
        "output_file": str(output_file) if all_exports else None
    }

    if validation_errors:
        stats["validation_errors"] = validation_errors

    return stats


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Export threads to training format")
    parser.add_argument(
        "--threads-dir",
        type=Path,
        default=Path("data/tool_chain_of_thoughts/threads"),
        help="Directory containing thread JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training_exports"),
        help="Directory to save exported files"
    )
    parser.add_argument(
        "--format",
        choices=["openai", "sharegpt"],
        default="openai",
        help="Output format (default: openai)"
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Include unfinished threads"
    )

    args = parser.parse_args()

    print("="*80)
    print("Thread Export Utility")
    print("="*80)
    print(f"Threads dir: {args.threads_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Format: {args.format}")
    print(f"Filter: {'All threads' if args.include_unfinished else 'Finished only'}")
    print()

    stats = export_threads(
        threads_dir=args.threads_dir,
        output_dir=args.output_dir,
        format=args.format,
        filter_finished_only=not args.include_unfinished
    )

    if not stats["success"]:
        print(f"✗ Error: {stats.get('error')}")
        return 1

    print("\n" + "="*80)
    print("Export Summary")
    print("="*80)
    print(f"Total files: {stats['total_files']}")
    print(f"Exported: {stats['exported']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")

    if stats.get("validation_errors"):
        print(f"\n⚠️  Validation errors found in {len(stats['validation_errors'])} threads:")
        for ve in stats["validation_errors"][:5]:  # Show first 5
            print(f"  - {ve['thread_id']}: {ve['errors'][0]}")
        if len(stats["validation_errors"]) > 5:
            print(f"  ... and {len(stats['validation_errors']) - 5} more")

    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
