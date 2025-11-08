# Tokenizer and Classifier Tools

## Overview

Added 4 new mock tools for tokenizing and classifying timeseries data:

1. **human_activity_motion_tokenizer** - Tokenizes IMU sensor data (HAR domain)
2. **human_activity_motion_classifier** - Classifies HAR z-tokens into semantic labels
3. **human_activity_motion_capture_tokenizer** - Tokenizes joint rotation data (MoCap domain)
4. **human_activity_motion_capture_classifier** - Classifies MoCap z-tokens into semantic labels

## Tool Descriptions

### human_activity_motion_tokenizer
- **Domain**: Human Activity Recognition (HAR)
- **Input**: Timeseries artifact with IMU sensors
- **Best for**: Accelerometer, gyroscope, magnetometer data at 50-100 Hz
- **Output**: Z-tokens artifact (discrete tokens in latent space)
- **Vocabulary**: 512 tokens (mock)
- **Typical channels**: body_acc_x/y/z, body_gyro_x/y/z, ankle_acc_x/y/z

### human_activity_motion_classifier
- **Domain**: Human Activity Recognition (HAR)
- **Input**: Z-tokens artifact from HAR tokenizer
- **Output**: E-tokens artifact with semantic labels
- **Labels**: walking, running, sitting, standing, laying, etc. (12 classes)
- **Purpose**: Convert domain-specific tokens to language-aligned semantic tokens

### human_activity_motion_capture_tokenizer
- **Domain**: Motion Capture (MoCap) - Full body kinematics
- **Input**: Timeseries artifact with joint rotation data
- **Best for**: Joint angles/rotations at 60-120 Hz
- **Output**: Z-tokens artifact
- **Vocabulary**: 1024 tokens (mock, larger for complex joint space)
- **Typical channels**: hip_rot_x/y/z, knee_rot_x/y/z, shoulder_rot_x/y/z
- **NOT FOR**: Raw IMU sensors

### human_activity_motion_capture_classifier
- **Domain**: Motion Capture (MoCap)
- **Input**: Z-tokens artifact from MoCap tokenizer
- **Output**: E-tokens with semantic labels for full-body movements
- **Labels**: jumping, squatting, lunging, reaching, etc. (15 classes)

## Workflow

The LLM should follow this sequence:

```
1. show_channel_stats (understand available channels)
   ↓
2. select_channels (narrow to sensor type, exclude incompatible sensors)
   ↓
3. [CHOOSE TOKENIZER based on data domain]
   ↓
4. select_channels (optional - refine for tokenizer if needed)
   ↓
5. Apply tokenizer (creates z-tokens)
   ↓
6. Apply matching classifier (creates e-tokens)
```

### Example: HAR Workflow

```json
// Step 1: Check channels
{"tool_name": "show_channel_stats", "parameters": {"dataset_name": "mhealth"}}

// Step 2: Select IMU only (exclude ECG)
{"tool_name": "select_channels", "parameters": {
  "artifact_id": "ts_abc",
  "channel_names": ["chest_acc_x/y/z", "ankle_acc_x/y/z", "ankle_gyro_x/y/z"]
}}
// Returns: {"type": "timeseries", "artifact_id": "ts_def"}

// Step 3: Tokenize
{"tool_name": "human_activity_motion_tokenizer", "parameters": {
  "artifact_id": "ts_def"
}}
// Returns: {"type": "z_tokens", "artifact_id": "zt_ghi"}

// Step 4: Classify
{"tool_name": "human_activity_motion_classifier", "parameters": {
  "artifact_id": "zt_ghi"
}}
// Returns: {"type": "e_tokens", "artifact_id": "et_jkl"}
```

## Key Constraints

### Sensor Compatibility
- **DO NOT** mix IMU + ECG
- **DO NOT** mix IMU + EMG
- **DO NOT** mix motion sensors + physiological sensors
- **DO** keep sensors from the same modality together

### Tokenizer Selection
- **IMU sensors** (acc/gyro/mag) → `human_activity_motion_tokenizer`
- **Joint rotations** (hip_rot/knee_rot) → `human_activity_motion_capture_tokenizer`
- **Wrong tokenizer** = poor quality tokens

### Classifier Requirements
- **MUST** use matching classifier for tokenizer:
  - HAR tokenizer → HAR classifier
  - MoCap tokenizer → MoCap classifier
- Classifier expects z-tokens from its matching tokenizer

## Artifact Flow

```
timeseries (raw data)
  ↓ select_channels
timeseries (filtered)
  ↓ tokenizer
z_tokens (discrete latent space)
  ↓ classifier
e_tokens (semantic labels aligned with language)
```

Each step creates a new artifact with parent tracking:
- Initial: `ts_abc` (user provides)
- After select: `ts_def` (parent: ts_abc)
- After tokenize: `zt_ghi` (parent: ts_def)
- After classify: `et_jkl` (parent: zt_ghi)

## Implementation Details

### Mock Behavior
- **Tokenizers**: Generate random token IDs, downsample 10x
- **Classifiers**: Generate random semantic token IDs
- **Codebook info**: Stored in artifact metadata
- **Semantic labels**: Predefined list for each domain

### Files Modified
- `tools/tool_executor.py` - Added 4 new tools
- `tools/__init__.py` - Exported new tools
- `datascripts/generate_tool_chains/schemas.py` - Added to allowed tool names
- `datascripts/generate_tool_chains/prompts/next_step_examples.txt` - Added descriptions and examples

### Testing
Run `python3 test_tokenizers.py` to verify:
- Session loading
- Channel selection
- HAR tokenization
- HAR classification
- Artifact chain creation

## Prompt Guidance

The LLM prompt now includes:
- **Tool descriptions** with domain specifications
- **Workflow examples** showing complete pipelines
- **Bad examples** showing what NOT to do
- **Sensor compatibility warnings**
- **Tokenizer selection rules**

Key prompt sections:
1. Tool descriptions (lines 36-69)
2. Complete workflow example (Example 1)
3. Refining channels example (Example 2)
4. Wrong tokenizer example (Example 3)
