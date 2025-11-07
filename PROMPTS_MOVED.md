# Prompts Folder Moved

## What Changed

The `prompts/` folder has been moved into the chain generation folder where it belongs.

### Before:
```
prompts/                           ← Root level (confusing)
├── system_instructions.txt
├── query_generation_examples.txt
└── next_step_examples.txt
```

### After:
```
datascripts/generate_tool_chains/
├── prompts/                       ← Inside generation folder (logical)
│   ├── system_instructions.txt
│   ├── query_generation_examples.txt
│   └── next_step_examples.txt
└── generation.py                  ← Uses prompts
```

## Why This Makes Sense

The prompts are:
- ✅ **Only used** by chain generation scripts
- ✅ **Not shared** with any other part of the system
- ✅ **Tightly coupled** to generation.py

Therefore they should be **in the same folder** as the generation code.

## What Was Updated

### Code:
- ✅ `datascripts/generate_tool_chains/generation.py`
  - Updated to load prompts from same folder using `Path(__file__).parent`
  - More robust (works regardless of working directory)

```python
# Before:
SYSTEM_INSTRUCTIONS = Path("prompts/system_instructions.txt")

# After:
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_INSTRUCTIONS = PROMPTS_DIR / "system_instructions.txt"
```

### Documentation:
- ✅ `datascripts/generate_tool_chains/README.md`
- ✅ `START_HERE.md`
- ✅ `CLEANUP_SUMMARY.md`

### Directory Move:
- ✅ Moved `prompts/` → `datascripts/generate_tool_chains/prompts/`

## Final Structure

```
datascripts/generate_tool_chains/     ← 100% SELF-CONTAINED
├── Scripts:
│   ├── generate_simple.py
│   ├── test_simple.py
│   ├── generate_chains_v2.py
│   ├── test_generation_v2.py
│   └── test_gemini_setup.py
│
├── Core code:
│   ├── generation.py              ← Core functions
│   └── schemas.py                 ← Pydantic models
│
├── Prompts:
│   └── prompts/
│       ├── system_instructions.txt
│       ├── query_generation_examples.txt
│       └── next_step_examples.txt
│
└── README.md
```

## Benefits

1. ✅ **Self-contained** - Everything for chain generation in one folder
2. ✅ **Clearer organization** - Related files together
3. ✅ **No confusion** - Prompts are clearly part of chain generation
4. ✅ **Easier to understand** - Folder contains everything it needs
5. ✅ **More maintainable** - Changes to prompts stay in context

## Works Exactly the Same

No behavior changes - scripts work identically:

```bash
# Test
python datascripts/generate_tool_chains/test_simple.py

# Generate
python datascripts/generate_tool_chains/generate_simple.py
```

The prompts are just loaded from a better location!

## Root Directory Cleanup

Root directory is now cleaner:

**Before:**
```
tsfm/
├── prompts/           ← What is this for?
├── tools/
├── datascripts/
└── ...
```

**After:**
```
tsfm/
├── tools/             ← Tool implementations
├── datascripts/       ← Scripts (includes prompts inside)
├── data/              ← Data
└── ...
```

Much clearer what each folder does!

## Summary

**Before:** Prompts in root, used only by datascripts/generate_tool_chains

**After:** Prompts inside datascripts/generate_tool_chains where they're used

Everything for chain generation is now in one place! 🎯
