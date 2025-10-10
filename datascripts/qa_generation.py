#!/usr/bin/env python3
"""
Generate structured Q/A pairs from ActionSense per-activity videos using Gemini.

- Reads data/actionsenseqa/data/manifest.csv (from your slicing pipeline).
- For each video clip, sends the clip inline + a dataset-generation prompt to Gemini.
- Uses Pydantic schema + Gemini structured output to get up to 4 Q/A pairs.
- Writes a single CSV: data/actionsenseqa/data/qa_pairs.csv

Adds caching: if qa_pairs.csv already exists, previously processed videos
are skipped so you don’t generate duplicates.

Adds live feedback:
- tqdm progress bar
- Every N videos, prints out the Q/A pairs just generated
"""

from __future__ import annotations

import os
import json
import mimetypes
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

# --- Google Gemini (Vertex) ---
from google import genai
from google.genai.types import (
    Part,
    Blob,
    FileData,
    GenerateContentConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)


# ===========================
# Hard-coded parameters
# ===========================
DATA_ROOT = "data/actionsenseqa/data"
MANIFEST_PATH = f"{DATA_ROOT}/manifest.csv"
OUTPUT_CSV = f"{DATA_ROOT}/qa_pairs.csv"

# Gemini (Vertex) settings
MODEL_NAME = "gemini-2.5-flash"
LOCATION = "us-central1"
PROJECT = "alex-oix"  # <-- change if needed

# Max Q/A pairs per video
MAX_QA_PER_VIDEO = 4

# Print sample every N videos
PRINT_EVERY = 10


# ===========================
# Pydantic Schemas (structured output)
# ===========================
class QAPair(BaseModel):
    question: str = Field(..., description="A single question about the activity in the video, grounded in observable motion/signals.")
    answer: str = Field(..., description="A concise ground-truth answer that could be inferred using only IMU/EMG/gaze/skeleton signals.")

class QAOutput(BaseModel):
    qa_pairs: List[QAPair] = Field(..., max_items=MAX_QA_PER_VIDEO, description="Up to 4 Q/A pairs.")


# ===========================
# Minimal Gemini wrapper
# ===========================
class GeminiClient:
    def __init__(self, model: str = MODEL_NAME, location: str = LOCATION, project: str = PROJECT):
        self.model = model
        self.client = genai.Client(vertexai=True, location=location, project=project)

    @staticmethod
    def _to_part(item) -> Part:
        if isinstance(item, Path):
            if not item.exists():
                raise FileNotFoundError(f"File not found: {item}")
            mime_type, _ = mimetypes.guess_type(item)
            if mime_type is None:
                if item.suffix.lower() == ".mp4":
                    mime_type = "video/mp4"
                else:
                    raise ValueError(f"Could not guess MIME type for: {item}")
            return Part(inline_data=Blob(data=item.read_bytes(), mime_type=mime_type))
        elif isinstance(item, str) and (item.startswith("gs://") or item.startswith("https://")):
            return Part(file_data=FileData(file_uri=item))
        else:
            return Part(text=str(item))

    def _default_safety(self) -> list[SafetySetting]:
        return [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]

    def generate(self, contents: list, schema: Optional[type[BaseModel]] = None):
        parts = [self._to_part(x) for x in contents]

        cfg_kwargs = {"safety_settings": self._default_safety()}
        if schema is not None:
            cfg_kwargs.update(
                {
                    "response_schema": schema.model_json_schema(),
                    "response_mime_type": "application/json",
                }
            )

        cfg = GenerateContentConfig(**cfg_kwargs)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": parts}],
            config=cfg,
        )

        if schema is None:
            text = resp.text if resp.text else "".join(candidate.text or "" for candidate in resp.candidates or [])
            if not text.strip():
                raise ValueError("Gemini returned empty text response.")
            return text.strip()

        if resp.parsed is None:
            if not resp.text:
                raise ValueError("Gemini returned no parsed data and empty text.")
            try:
                obj = json.loads(resp.text)
            except Exception as e:
                raise ValueError(f"Could not parse JSON from response: {e}\nText was: {resp.text[:4000]}")
        else:
            obj = resp.parsed

        return schema.model_validate(obj)


# ===========================
# Prompts
# ===========================
REASONING_PROMPT = (
    "You are a question answering dataset generation expert for a time series sensor data project. "
    "You will be given a first-person video of someone doing household chores. "
    "During collection, the person wore sensor gloves and body trackers (IMU) and EMG bands. "
    "Your purpose is to generate questions that can be answered by a model using only the sensor data (IMU/EMG/gaze/skeleton)—no video.\n\n"
    "you may use the format in the examples below, but you are encouraged to be creative and generate questions not in the examples. use the "
    "fpv video to decide what questions are appropriate. Change up the sentence structure to ensure a heterogenous dataset.\n\n"
    "Some examples:\n"
    "- If the video shows grabbing items from fridge/cabinets, maybe ask how many items, whether the handle/grab-point was high/low relative to body, "
    "  whether items were high/low/left/right, or whether the person turned left/right after grabbing. Feel free to ask how many items were retrieved.\n"
    "- If cutting vegetables, maybe ask whether cutting was quick or hesitant, whether peeling vs cutting, "
    "  whether the vegetable was round/stubby or skinny/long, or what kind of vegetable/food (only if inferable from motion patterns).\n"
    "- If cleaning/doing dishes, maybe ask which hand did the cleaning and which hand held the item.\n"
    "- you may ask questions like is the person peeling the cucumber or cutting it? feel free to mention items or vegetables by name. \n"
    "- It is OK to ask about walking vs being stationary (e.g., approximate amount of motion or whether the person walked between steps).\n\n"
    "CRITICAL RULES:\n"
    "- Only ask questions that could be inferred from IMU/EMG/gaze/skeleton signals alone (no reliance on appearance/color/brand/text/faces).\n"
    "- Be specific and measurable where possible (left/right, high/low, quick/slow, turning direction, stationary vs walking).\n"
    "- Feel free to also ask general questions like what is the person doing? this is good to test activity classification ability (e.g., cutting cucumbers).\n"
    "- If something cannot be determined from time-series motion/EMG/gaze alone, DO NOT ask about it.\n"
    "- avoid asking about exact counts if they are high (ie >5), such as slices of cucumbers, as it is hard to be certain\n"
    "- Both question and answer should be full sentences.\n"
    "Return a plain-text reasoning response (no JSON)."
)

STRUCTURED_PROMPT_TEMPLATE = (
    "Convert the following analysis into JSON that follows the schema qa_pairs: [{{question, answer}}]. "
    "Use only the information provided, do not add new facts or reasoning, and ensure each question and answer is a full sentence.\n\n"
    "Analysis:\n{analysis}\n"
)


# ===========================
# Runner
# ===========================
def generate_qa_from_manifest(
    manifest_path: str = MANIFEST_PATH,
    output_csv: str = OUTPUT_CSV,
    data_root: str = DATA_ROOT,
    model_name: str = MODEL_NAME,
    location: str = LOCATION,
    project: str = PROJECT,
):
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    man = pd.read_csv(manifest_path)
    if man.empty:
        print("[INFO] Manifest is empty; nothing to process.")
        return

    # --- Load existing QA pairs (cache) ---
    processed = set()
    if Path(output_csv).exists():
        prev = pd.read_csv(output_csv)
        processed = set(prev["activity_index"].astype(str))
        print(f"[INFO] Found existing {len(processed)} processed activities")

    client = GeminiClient(model=model_name, location=location, project=project)

    print(f"[INFO] Loaded manifest with {len(man)} rows")

    # Ensure output CSV has header if it doesn’t exist
    cols = ["subject", "split", "activity_index", "activity_name", "video_path", "question", "answer"]
    if not Path(output_csv).exists():
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)

    for i, (idx, row) in enumerate(tqdm(man.iterrows(), total=len(man), desc="Videos"), start=1):
        subject = row.get("subject", "")
        split = row.get("split", "")
        activity_index = str(row.get("activity_index", ""))
        activity_name = row.get("activity_name", "")
        rel_video = row.get("video_path", "")
        video_path = Path(data_root) / rel_video

        if activity_index in processed:
            continue
        if not video_path.exists():
            continue

        try:
            reasoning_text = client.generate([REASONING_PROMPT, video_path])

            structured_prompt = STRUCTURED_PROMPT_TEMPLATE.format(analysis=reasoning_text)
            qa_struct: QAOutput = client.generate([structured_prompt], schema=QAOutput)
            qa_list = qa_struct.qa_pairs[:MAX_QA_PER_VIDEO] if qa_struct.qa_pairs else []

            for qa in qa_list:
                row_dict = {
                    "subject": subject,
                    "split": split,
                    "activity_index": activity_index,
                    "activity_name": activity_name,
                    "video_path": str(video_path),
                    "question": qa.question.strip(),
                    "answer": qa.answer.strip(),
                }

                # Append immediately to CSV
                pd.DataFrame([row_dict])[cols].to_csv(
                    output_csv, mode="a", header=False, index=False
                )

                # Print immediately
                print(f"\n[New QA] ({activity_name}, {video_path.name})")
                print(f"Q: {qa.question.strip()}")
                print(f"A: {qa.answer.strip()}")
                print("-" * 40)

        except Exception as e:
            print(f"[ERROR] Gemini failed for {video_path.name}: {e}")
            traceback.print_exc()

    print(f"[DONE] All available Q/A pairs written to {output_csv}")


if __name__ == "__main__":
    generate_qa_from_manifest()
