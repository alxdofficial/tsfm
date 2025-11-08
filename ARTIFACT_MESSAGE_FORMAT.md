# Artifact Message Format

## Overview

Artifacts are now properly represented as dict objects in message content, not JSON strings. The thread structure no longer maintains a separate `artifacts` field.

## Changes Made

### 1. Initial User Messages (3 messages)

**Before:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Dataset Information:\n{manifest}\n\nArtifacts:\n- ts_abc...\n\nUser Query: ..."
    }
  ]
}
```

**After:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Dataset Information:\n{manifest}"
    },
    {
      "role": "user",
      "content": {
        "type": "timeseries",
        "artifact_id": "ts_abc123"
      }
    },
    {
      "role": "user",
      "content": "User Query: What activity is this?"
    }
  ]
}
```

### 2. Tool Result Messages

**Before:**
```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "name": "select_channels",
  "content": "{\"type\": \"timeseries\", \"artifact_id\": \"ts_xyz\"}"  // JSON string
}
```

**After:**
```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "name": "select_channels",
  "content": {
    "type": "timeseries",
    "artifact_id": "ts_xyz"
  }  // Dict object
}
```

### 3. Thread Metadata

**Before:**
```json
{
  "metadata": {...},
  "artifacts": {
    "ts_abc123": {
      "type": "timeseries",
      "num_samples": 128,
      "num_channels": 9,
      ...
    }
  },
  "messages": [...]
}
```

**After:**
```json
{
  "metadata": {
    ...,
    "initial_artifact_id": "ts_abc123"
  },
  "messages": [...]
}
```

The `artifacts` field is **removed**. Artifact metadata is stored in the global registry, not in the thread.

## Artifact ID Extraction

**Before:**
```python
# From thread["artifacts"] dict
artifact_id = list(thread["artifacts"].keys())[0]
```

**After:**
```python
# From messages, walking backwards
artifact_id = thread["metadata"].get("initial_artifact_id")

for msg in reversed(thread["messages"]):
    if msg.get("role") == "tool":
        content = msg.get("content")
        if isinstance(content, dict) and "artifact_id" in content:
            artifact_id = content["artifact_id"]
            break
```

Extracts the most recent artifact from tool results or user messages.

## Conversation History Parsing

**Before:**
```python
# Assumed all turns had tool_call
tool_result = json.loads(tool_result_msg["content"])  # String parse
```

**After:**
```python
# Handle both dict and string (backwards compatibility)
tool_result_content = tool_result_msg["content"]
if isinstance(tool_result_content, str):
    tool_result = json.loads(tool_result_content)
else:
    tool_result = tool_result_content
```

## Benefits

1. **Type Safety**: Content is typed as dict, not opaque string
2. **Cleaner Storage**: No duplicate artifact metadata in thread JSON
3. **Better Semantics**: Artifact is a first-class message content type
4. **Simpler Logic**: No need to maintain thread["artifacts"] separately
5. **Standard Format**: Follows OpenAI message conventions better

## Example Thread JSON

```json
{
  "id": "uci_har_train_01_0000_v4_1234",
  "metadata": {
    "dataset": "uci_har",
    "session": "train_01_0000",
    "ground_truth": "walking",
    "initial_artifact_id": "ts_abc123def456",
    "user_query": "What activity is this?",
    "num_turns": 2,
    "num_tool_calls": 2,
    "num_responds": 0
  },
  "messages": [
    {
      "role": "user",
      "content": "Dataset Information:\n{...}"
    },
    {
      "role": "user",
      "content": {
        "type": "timeseries",
        "artifact_id": "ts_abc123def456"
      }
    },
    {
      "role": "user",
      "content": "User Query: What activity is this?"
    },
    {
      "role": "assistant",
      "content": "I need to check the available channels...",
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "show_channel_stats",
          "arguments": "{\"dataset_name\": \"uci_har\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "show_channel_stats",
      "content": {
        "channels": [...]
      }
    },
    {
      "role": "assistant",
      "content": "Now I'll select only IMU channels...",
      "tool_calls": [{
        "id": "call_2",
        "type": "function",
        "function": {
          "name": "select_channels",
          "arguments": "{\"artifact_id\": \"ts_abc123def456\", \"channel_names\": [...]}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_2",
      "name": "select_channels",
      "content": {
        "type": "timeseries",
        "artifact_id": "ts_xyz789ghi012"
      }
    }
  ]
}
```

## Files Modified

- `datascripts/generate_tool_chains/generate_simple.py`:
  - `create_new_thread()` - 3 separate user messages
  - `messages_to_conversation_history()` - Handle dict content
  - `generate_step_interactive()` - Store dict, extract artifact from messages
  - Removed `thread["artifacts"]` maintenance

- `datascripts/generate_tool_chains/generation.py`:
  - `generate_next_step()` - Handle both tool and respond turns in history

## Migration

Old thread files with JSON string content will still work due to backwards compatibility in `messages_to_conversation_history()`. New threads use the dict format.
