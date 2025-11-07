# Tool Use Chain Data Format

This document describes the JSON format for storing tool use reasoning chains. These chains are used to train LLMs to orchestrate specialized tools for time series analysis.

## Overview

Each conversation chain is a self-contained JSON file representing:
- A user query about a specific dataset
- The full dataset manifest (for context)
- A series of reasoning steps and tool calls
- The actual tool execution results
- A final answer to the user's query

## Schema Version: 1.0

## Top-Level Structure

```json
{
  "conversation_id": "string",           // Unique identifier for this chain
  "dataset_name": "string",              // Dataset this chain operates on
  "created_at": "ISO8601 timestamp",     // When this chain was generated
  "schema_version": "1.0",               // Schema version for compatibility

  "generation_metadata": {
    "gemini_model": "string",            // Model used to generate (e.g., "gemini-1.5-pro")
    "temperature": 0.7,                  // Generation temperature
    "phase": "string",                   // "phase1_eda_only" or "phase2_domain_tools"
    "tools_available": ["string"],       // List of tool names available during generation
    "human_verified": false              // Whether a human reviewed this chain
  },

  "initial_context": {
    "user_query": "string",              // The original user question
    "manifest": {...},                   // Full manifest.json for the dataset
    "labels_summary": {                  // Summary of labels.json
      "total_sessions": 0,
      "unique_labels": ["string"],
      "label_counts": {...}
    }
  },

  "conversation": [
    {
      "turn": 1,                         // Sequential turn number
      "reasoning": "string",             // Model's thought process for this step
      "tool_call": {
        "tool_name": "string",           // Name of the tool to call
        "parameters": {...}              // Tool-specific parameters (JSON object)
      },
      "tool_result": {...}               // Full raw output from tool execution
    }
    // ... more turns
  ],

  "final_answer": "string",              // Model's final response to user query
  "execution_verified": false            // Whether tools were actually executed vs mocked
}
```

## Field Descriptions

### conversation_id
- Format: `{dataset}_{query_slug}_{version}`
- Examples: `uci_har_walking_analysis_001`, `pamap2_compare_running_cycling_002`
- Unique across all chains

### generation_metadata.phase
- **phase1_eda_only**: Only EDA tools available (show_session_stats, show_channel_stats, select_channels, filter_by_time)
- **phase2_domain_tools**: Domain-specific tools available (motion_tokenizer, motion_classifier, motion_forecaster)
- **phase3_full**: All tools including advanced analysis

### initial_context.manifest
- Complete manifest.json contents from the dataset
- Allows chain to be self-contained
- Model can reference channels, descriptions, sampling rates

### conversation[].reasoning
- Natural language explanation of what the model is thinking
- Should justify why this tool is needed
- Should reference information from previous turns

### conversation[].tool_result
- **Always store full raw output** from tool execution
- Do not summarize or abbreviate
- Allows retracing and verification
- For large outputs (e.g., selected data), store handles/references

### execution_verified
- `true`: All tools were actually executed, results are real
- `false`: Chain generated synthetically or tools mocked

## Phase 1 Example: EDA Tools Only

```json
{
  "conversation_id": "uci_har_walking_analysis_001",
  "dataset_name": "uci_har",
  "created_at": "2025-01-07T10:30:00Z",
  "schema_version": "1.0",

  "generation_metadata": {
    "gemini_model": "gemini-1.5-pro",
    "temperature": 0.7,
    "phase": "phase1_eda_only",
    "tools_available": [
      "show_session_stats",
      "show_channel_stats",
      "select_channels",
      "filter_by_time"
    ],
    "human_verified": true
  },

  "initial_context": {
    "user_query": "Analyze walking patterns in the UCI HAR dataset. What are the key characteristics?",
    "manifest": {
      "dataset_name": "UCI_HAR",
      "description": "Human activity recognition using smartphone accelerometer and gyroscope. 30 subjects performing 6 activities.",
      "channels": [
        {
          "name": "body_acc_x",
          "description": "Body acceleration X-axis (gravity removed)",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "body_acc_y",
          "description": "Body acceleration Y-axis (gravity removed)",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "body_acc_z",
          "description": "Body acceleration Z-axis (gravity removed)",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "body_gyro_x",
          "description": "Angular velocity X-axis",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "body_gyro_y",
          "description": "Angular velocity Y-axis",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "body_gyro_z",
          "description": "Angular velocity Z-axis",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "total_acc_x",
          "description": "Total acceleration X-axis (includes gravity)",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "total_acc_y",
          "description": "Total acceleration Y-axis (includes gravity)",
          "sampling_rate_hz": 50.0
        },
        {
          "name": "total_acc_z",
          "description": "Total acceleration Z-axis (includes gravity)",
          "sampling_rate_hz": 50.0
        }
      ]
    },
    "labels_summary": {
      "total_sessions": 10299,
      "unique_labels": ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"],
      "label_counts": {
        "walking": 1722,
        "walking_upstairs": 1544,
        "walking_downstairs": 1406,
        "sitting": 1777,
        "standing": 1906,
        "laying": 1944
      }
    }
  },

  "conversation": [
    {
      "turn": 1,
      "reasoning": "I need to understand the dataset structure first. Let me check the session statistics to see how the data is distributed across activities and understand the session characteristics.",
      "tool_call": {
        "tool_name": "show_session_stats",
        "parameters": {
          "dataset_name": "uci_har"
        }
      },
      "tool_result": {
        "total_sessions": 10299,
        "label_distribution": {
          "walking": 1722,
          "walking_upstairs": 1544,
          "walking_downstairs": 1406,
          "sitting": 1777,
          "standing": 1906,
          "laying": 1944
        },
        "duration_stats": {
          "min": 2.54,
          "max": 2.54,
          "mean": 2.54,
          "median": 2.54,
          "unit": "seconds"
        },
        "subjects": 30,
        "sessions_per_subject": {
          "min": 343,
          "max": 344,
          "mean": 343.3
        }
      }
    },
    {
      "turn": 2,
      "reasoning": "Good. I can see there are 1,722 walking sessions, each exactly 2.54 seconds long. Now I need to understand which sensor channels would be most informative for analyzing walking patterns. Let me check the available channels and their characteristics.",
      "tool_call": {
        "tool_name": "show_channel_stats",
        "parameters": {
          "dataset_name": "uci_har"
        }
      },
      "tool_result": {
        "channels": [
          {
            "name": "body_acc_x",
            "sampling_rate_hz": 50.0,
            "description": "Body acceleration X-axis (gravity removed)",
            "samples_per_session": 128
          },
          {
            "name": "body_acc_y",
            "sampling_rate_hz": 50.0,
            "description": "Body acceleration Y-axis (gravity removed)",
            "samples_per_session": 128
          },
          {
            "name": "body_acc_z",
            "sampling_rate_hz": 50.0,
            "description": "Body acceleration Z-axis (gravity removed)",
            "samples_per_session": 128
          },
          {
            "name": "body_gyro_x",
            "sampling_rate_hz": 50.0,
            "description": "Angular velocity X-axis",
            "samples_per_session": 128
          },
          {
            "name": "body_gyro_y",
            "sampling_rate_hz": 50.0,
            "description": "Angular velocity Y-axis",
            "samples_per_session": 128
          },
          {
            "name": "body_gyro_z",
            "sampling_rate_hz": 50.0,
            "description": "Angular velocity Z-axis",
            "samples_per_session": 128
          },
          {
            "name": "total_acc_x",
            "sampling_rate_hz": 50.0,
            "description": "Total acceleration X-axis (includes gravity)",
            "samples_per_session": 128
          },
          {
            "name": "total_acc_y",
            "sampling_rate_hz": 50.0,
            "description": "Total acceleration Y-axis (includes gravity)",
            "samples_per_session": 128
          },
          {
            "name": "total_acc_z",
            "sampling_rate_hz": 50.0,
            "description": "Total acceleration Z-axis (includes gravity)",
            "samples_per_session": 128
          }
        ]
      }
    },
    {
      "turn": 3,
      "reasoning": "For walking pattern analysis, the body acceleration (gravity-removed) and gyroscope channels are most relevant. The body_acc channels capture the dynamic motion patterns, while the gyro channels capture rotational movements. I'll select these 6 IMU channels and focus only on walking sessions.",
      "tool_call": {
        "tool_name": "select_channels",
        "parameters": {
          "dataset_name": "uci_har",
          "channel_names": ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"],
          "label_filter": ["walking"]
        }
      },
      "tool_result": {
        "selected_dataset_handle": "ds_7a3f9e2b",
        "num_sessions": 1722,
        "selected_channels": ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"],
        "filtered_labels": ["walking"],
        "total_samples": 220416,
        "memory_usage_mb": 10.6
      }
    }
  ],

  "final_answer": "Based on my analysis of the UCI HAR dataset, here are the key characteristics of walking patterns:\n\n**Dataset Overview:**\n- 1,722 walking sessions from 30 subjects\n- Each session is 2.54 seconds (128 samples at 50 Hz)\n- Data collected from smartphone IMU sensors (accelerometer + gyroscope)\n\n**Key Channels for Walking Analysis:**\nI selected 6 channels most relevant for walking patterns:\n- Body acceleration (X, Y, Z): Captures dynamic motion with gravity removed\n- Angular velocity (X, Y, Z): Captures rotational movements during gait\n\n**Next Steps for Pattern Analysis:**\nWith this curated dataset (handle: ds_7a3f9e2b), you could:\n1. Apply motion tokenization to extract gait cycle features\n2. Analyze frequency components to identify stride patterns\n3. Compare walking with walking_upstairs and walking_downstairs to understand terrain effects\n4. Use classification models to detect individual gait signatures\n\nThe 50 Hz sampling rate is well-suited for capturing human walking (typical stride frequency: 1-2 Hz), and the 2.56-second windows provide enough context for multiple gait cycles.",

  "execution_verified": true
}
```

## Phase 2 Example: With Domain Tools

```json
{
  "conversation_id": "uci_har_walking_tokenize_001",
  "dataset_name": "uci_har",
  "created_at": "2025-01-07T14:30:00Z",
  "schema_version": "1.0",

  "generation_metadata": {
    "gemini_model": "gemini-1.5-pro",
    "temperature": 0.7,
    "phase": "phase2_domain_tools",
    "tools_available": [
      "show_session_stats",
      "show_channel_stats",
      "select_channels",
      "filter_by_time",
      "motion_tokenizer",
      "motion_classifier"
    ],
    "human_verified": true,
    "extended_from": "uci_har_walking_analysis_001"
  },

  "initial_context": {
    "user_query": "Extract motion tokens from walking data for downstream classification",
    "manifest": {
      // ... same as phase1
    },
    "labels_summary": {
      // ... same as phase1
    }
  },

  "conversation": [
    {
      "turn": 1,
      "reasoning": "I need to first select the appropriate IMU channels for walking sessions.",
      "tool_call": {
        "tool_name": "select_channels",
        "parameters": {
          "dataset_name": "uci_har",
          "channel_names": ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"],
          "label_filter": ["walking"]
        }
      },
      "tool_result": {
        "selected_dataset_handle": "ds_7a3f9e2b",
        "num_sessions": 1722,
        "selected_channels": ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
      }
    },
    {
      "turn": 2,
      "reasoning": "Now I'll use the motion tokenizer to extract compact representations. Given that each session is 2.54 seconds and walking has a stride frequency around 1-2 Hz, I'll use 1-second patches with 25% overlap to capture gait cycle features.",
      "tool_call": {
        "tool_name": "motion_tokenizer",
        "parameters": {
          "data_handle": "ds_7a3f9e2b",
          "patch_size_sec": 1.0,
          "overlap": 0.25
        }
      },
      "tool_result": {
        "token_handle": "tok_9f2e1a3c",
        "num_sessions": 1722,
        "tokens_per_session": 3,
        "total_tokens": 5166,
        "token_dim": 256,
        "memory_usage_mb": 5.1
      }
    }
  ],

  "final_answer": "Successfully extracted motion tokens from 1,722 walking sessions. Each session produced 3 tokens (1-second patches with 25% overlap), resulting in 5,166 total tokens. These 256-dimensional embeddings are now ready for downstream tasks like classification or clustering. Token handle: tok_9f2e1a3c",

  "execution_verified": true
}
```

## Guidelines for Chain Generation

### Query Diversity
Generate chains covering different analysis types:
- **Exploratory**: "What activities are in this dataset?"
- **Comparative**: "Compare running vs walking patterns"
- **Filtering**: "Extract only upper body movements"
- **Temporal**: "Analyze the first 30 seconds of each session"
- **Channel-specific**: "Focus on gyroscope data only"

### Reasoning Quality
Good reasoning should:
- Reference specific information from previous tool results
- Justify tool choice based on the current goal
- Show understanding of data characteristics (sampling rate, duration, etc.)
- Build logically on previous steps

### Tool Call Sequencing
Typical patterns:
1. **Exploration first**: show_session_stats, show_channel_stats
2. **Selection**: select_channels (often with label_filter)
3. **Analysis**: motion_tokenizer, motion_classifier, etc.
4. **Final answer**: Synthesize results

### Realism
- Tool parameters should be realistic (e.g., patch_size_sec appropriate for activity type)
- Reasoning should acknowledge data constraints
- Final answers should be specific and actionable

## Storage Location

All generated chains are stored in:

```
data/tool_chain_of_thoughts/
├── README.md
├── {dataset}_{session}_v2_{id}.json
├── {dataset}_{session}_v2_{id}.json
└── ...
```

Example filenames:
- `uci_har_session_001_v2_042.json`
- `pamap2_session_045_v2_123.json`
- `mhealth_session_002_v2_087.json`

## Version History

- **1.0** (2025-01-07): Initial schema
  - Support for EDA tools (Phase 1)
  - Support for domain tools (Phase 2)
  - Full manifest inline
  - Full raw tool results
