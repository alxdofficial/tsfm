"""
LanHAR baseline assessment.

Trains a LanHAR-style model (Language-guided HAR) on our 10 training datasets
and runs the same 3-metric framework as LiMU-BERT and MOMENT:
  1. Zero-shot open-set (all 87 training labels, group-based matching)
  2. Closed-set (test dataset labels only, exact match)
  3. 1% supervised (train on 1% of test data)

LanHAR uses a CLIP-style approach:
  - SciBERT text encoder for activity descriptions
  - TimeSeriesTransformer sensor encoder for IMU data
  - Contrastive learning to align sensor and text embeddings

After training, 768-dim sensor embeddings are extracted for scoring.

Two training stages:
  Stage 1: Fine-tune SciBERT with supervised contrastive learning on text prototypes
  Stage 2: Train sensor encoder with CLIP loss on sensor-text pairs

Usage:
    python val_scripts/human_activity_recognition/evaluate_lanhar.py
"""

import json
import sys
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import GradScaler
from torch import amp
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imu_pretraining_dataset.label_groups import (
    LABEL_GROUPS,
    get_label_to_group_mapping,
)

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = LIMUBERT_DATA_DIR / "global_label_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

# Model settings
BERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
DATA_SEQ_LEN = 120
DATA_CHANNELS = 6
SENSOR_D_MODEL = 768  # Must match SciBERT hidden size
EMB_DIM = 768

# Training hyperparameters
STAGE1_EPOCHS = 10
STAGE1_BATCH_SIZE = 10  # Matches original LanHAR paper
STAGE1_LR = 1e-5

STAGE2_EPOCHS = 50   # Matches original LanHAR paper (was 30)
STAGE2_BATCH_SIZE = 128  # Original paper uses 256, reduced to fit shared GPU
STAGE2_LR = 4e-5

# Classifier hyperparameters (same as MOMENT/LiMU-BERT)
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_LR = 1e-3
CLASSIFIER_SEED = 3431

# Data split parameters
TRAINING_RATE = 0.8
VALI_RATE = 0.1
SUPERVISED_LABEL_RATE = 0.01

SEED = 3431

# Load configs
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

with open(GLOBAL_LABEL_PATH) as f:
    GLOBAL_LABELS = json.load(f)["labels"]

TRAIN_DATASETS = DATASET_CONFIG["train_datasets"]
TEST_DATASETS = DATASET_CONFIG["zero_shot_datasets"]


# =============================================================================
# Text Prototypes for 87 Activity Classes
# =============================================================================

# Activity -> motion category mapping
ACTIVITY_CATEGORY = {
    "adductor_machine": "exercise_machine",
    "arm_curl": "exercise_machine",
    "ascending_stairs": "stairs_up",
    "bench_press": "exercise_machine",
    "brushing_teeth": "fine_motor",
    "clapping": "fine_motor",
    "climbing_stairs": "stairs_up",
    "cycling": "cycling",
    "cycling_horizontal": "cycling",
    "cycling_vertical": "cycling",
    "descending_stairs": "stairs_down",
    "dribbling": "sports",
    "drinking": "fine_motor",
    "eating_chips": "fine_motor",
    "eating_pasta": "fine_motor",
    "eating_sandwich": "fine_motor",
    "eating_soup": "fine_motor",
    "exercising_cross_trainer": "exercise_cardio",
    "exercising_stepper": "exercise_cardio",
    "falling_backward": "fall",
    "falling_backward_sitting": "fall",
    "falling_forward": "fall",
    "falling_hitting_obstacle": "fall",
    "falling_left": "fall",
    "falling_right": "fall",
    "falling_with_protection": "fall",
    "folding_clothes": "household",
    "frontal_elevation_arms": "exercise_upper",
    "going_down_stairs": "stairs_down",
    "going_up_stairs": "stairs_up",
    "ironing": "household",
    "jogging": "run",
    "jump_front_back": "jump",
    "jumping": "jump",
    "kicking": "sports",
    "knees_bending": "exercise_lower",
    "laying": "lying",
    "leg_curl": "exercise_machine",
    "leg_press": "exercise_machine",
    "lie_to_sit": "transition",
    "lie_to_stand": "transition",
    "lying": "lying",
    "lying_back": "lying",
    "lying_down_from_standing": "transition",
    "lying_side": "lying",
    "moving_elevator": "elevator",
    "nordic_walking": "walk",
    "picking_up": "transition",
    "playing_basketball": "sports",
    "playing_catch": "sports",
    "playing_sports": "sports",
    "push_up": "exercise_upper",
    "rope_jumping": "jump",
    "rope_skipping": "jump",
    "rowing": "exercise_cardio",
    "running": "run",
    "running_treadmill": "run",
    "sit_to_lie": "transition",
    "sit_to_stand": "transition",
    "sit_up": "exercise_upper",
    "sitting": "static_sit",
    "sitting_down": "transition",
    "squat": "exercise_lower",
    "stairclimber": "stairs_up",
    "stairs": "stairs_general",
    "stairs_down": "stairs_down",
    "stairs_up": "stairs_up",
    "stand_to_lie": "transition",
    "stand_to_sit": "transition",
    "standing": "static_stand",
    "standing_elevator": "elevator",
    "standing_up_from_laying": "transition",
    "standing_up_from_sitting": "transition",
    "syncope": "fall",
    "talking_sitting": "static_sit",
    "talking_standing": "static_stand",
    "typing": "fine_motor",
    "vacuum_cleaning": "household",
    "waist_bends_forward": "exercise_lower",
    "walking": "walk",
    "walking_backwards": "walk",
    "walking_downstairs": "stairs_down",
    "walking_parking": "walk",
    "walking_treadmill_flat": "walk",
    "walking_treadmill_incline": "walk",
    "walking_upstairs": "stairs_up",
    "writing": "fine_motor",
}

# Sensor pattern descriptions by motion category
CATEGORY_SENSORS = {
    "walk": [
        "periodic vertical oscillation around 1 to 2 Hz from footsteps",
        "repetitive step cycle with regular heel strikes and toe-offs",
        "moderate rhythmic acceleration in the vertical axis",
        "consistent gait pattern with symmetric stride intervals",
        "small gyroscope sway from arm and body motion during walking",
    ],
    "run": [
        "high frequency periodic acceleration around 2 to 3 Hz",
        "strong vertical impact at each landing with brief flight phases",
        "rapid cyclic motion with higher amplitude than walking",
        "elevated acceleration magnitude with pronounced vertical peaks",
        "high-energy rhythmic pattern with strong ground reaction forces",
    ],
    "stairs_up": [
        "walk-like rhythm with stronger upward thrust on each step",
        "pronounced vertical impulse during body lifting phase",
        "slower cadence than flat walking with heavier footfalls",
        "periodic pattern with elevated vertical acceleration component",
        "positive pitch rotation as the body leans upward on each step",
    ],
    "stairs_down": [
        "sharp downward acceleration followed by strong landing impact",
        "higher jerk magnitude from abrupt foot strikes on each step",
        "walk-like cadence with pronounced downward landing spikes",
        "descent pattern with rich high-frequency impact components",
        "negative pitch rotation as the body leans forward descending",
    ],
    "stairs_general": [
        "rhythmic stair climbing or descending pattern",
        "periodic vertical motion with step-like impacts on stairs",
        "gait-like pattern with enhanced vertical acceleration from steps",
        "step cycle with elevation changes between consecutive steps",
    ],
    "cycling": [
        "smooth rotational leg motion from pedaling cycle",
        "periodic low-frequency acceleration from cycling cadence",
        "reduced vertical impact compared to walking or running",
        "rhythmic circular motion pattern from lower body pedaling",
        "minimal upper body motion with periodic lower limb activity",
    ],
    "jump": [
        "explosive high vertical acceleration during takeoff phase",
        "strong landing impact with brief airborne period between",
        "high-amplitude transient acceleration peaks from jumps",
        "rapid vertical motion with abrupt direction changes at peak",
        "intermittent high-energy bursts separated by brief pauses",
    ],
    "static_sit": [
        "acceleration nearly constant close to gravity value",
        "very low motion energy with only small random noise",
        "gyroscope readings essentially flat with minimal drift",
        "absence of step-like impacts or periodic motion patterns",
        "near-static signal with negligible body movement",
    ],
    "static_stand": [
        "acceleration close to gravity with slight postural sway",
        "minimal motion energy with low-frequency balance adjustments",
        "near-static signal with negligible periodic components",
        "gravity-dominated signal with very small perturbations",
        "upright orientation maintained with subtle balance corrections",
    ],
    "lying": [
        "gravity vector orientation shifted from upright position",
        "extremely low motion energy in all sensor axes",
        "accelerometer shows non-vertical gravity orientation",
        "minimal gyroscope activity with stable recumbent body position",
        "flat and low-variance signal across all measurement axes",
    ],
    "transition": [
        "brief transient motion during posture change",
        "non-periodic acceleration burst during body repositioning",
        "orientation change reflected in gravity vector shift",
        "short-duration motion event between two stable positions",
        "rapid change in body orientation followed by stabilization",
    ],
    "exercise_machine": [
        "repetitive controlled motion from machine-guided exercise",
        "periodic acceleration from resistance training movement",
        "structured motion cycle with consistent range of motion",
        "rhythmic exercise pattern with machine-constrained trajectory",
        "regular repetition of controlled force application pattern",
    ],
    "exercise_upper": [
        "repetitive upper body motion with moderate acceleration",
        "periodic arm or torso movement pattern during exercise",
        "rhythmic acceleration from upper body workout movement",
        "controlled repetitive motion centered on upper limbs",
        "periodic force generation from arm and shoulder muscles",
    ],
    "exercise_lower": [
        "repetitive lower body motion with vertical displacement",
        "periodic leg movement pattern during exercise",
        "rhythmic vertical acceleration from lower body workout",
        "controlled squat or bending motion cycle with regularity",
        "up-down motion pattern from leg-dominant exercise",
    ],
    "exercise_cardio": [
        "sustained rhythmic full-body motion at moderate intensity",
        "continuous periodic acceleration from cardio exercise machine",
        "rhythmic motion pattern maintained over extended duration",
        "moderate-to-high intensity repetitive full-body motion",
        "steady cardiovascular exercise with consistent rhythm",
    ],
    "sports": [
        "irregular high-intensity motion with sudden acceleration bursts",
        "non-periodic acceleration spikes from athletic movements",
        "variable motion pattern with rapid direction changes",
        "high-energy activity with unpredictable acceleration patterns",
        "mixed motion intensities with intermittent high-force events",
    ],
    "fine_motor": [
        "small-amplitude wrist and hand movements",
        "low overall acceleration magnitude with fine motor activity",
        "minimal body motion with localized hand or arm movements",
        "subtle periodic or irregular patterns from hand activity",
        "low-energy signal dominated by small precise movements",
    ],
    "household": [
        "moderate arm motion from household task performance",
        "variable acceleration pattern from manual domestic activity",
        "semi-repetitive motion from household chore execution",
        "moderate-intensity arm and body movements during chores",
        "mixed motion pattern with varied force and speed",
    ],
    "fall": [
        "sudden high-magnitude acceleration spike from body impact",
        "rapid uncontrolled orientation change during fall",
        "brief high-jerk event followed by static post-fall signal",
        "abrupt transition from motion to stillness after impact",
        "extreme acceleration transient lasting under one second",
    ],
    "elevator": [
        "subtle vertical acceleration from elevator movement",
        "low-magnitude acceleration change with stable body position",
        "brief acceleration onset followed by constant velocity phase",
        "minimal body motion during passive vertical transport",
        "gentle vertical acceleration change at start and stop",
    ],
}

# Activity name variants (additional synonyms beyond the label itself)
ACTIVITY_NAMES = {
    "adductor_machine": ["adductor machine exercise", "inner thigh machine", "hip adduction machine"],
    "arm_curl": ["arm curl", "bicep curl", "dumbbell curl"],
    "ascending_stairs": ["ascending stairs", "walking up stairs", "going upstairs"],
    "bench_press": ["bench press", "chest press", "barbell bench press"],
    "brushing_teeth": ["brushing teeth", "toothbrushing", "dental hygiene motion"],
    "clapping": ["clapping", "hand clapping", "applauding with hands"],
    "climbing_stairs": ["climbing stairs", "stair climbing", "going up steps"],
    "cycling": ["cycling", "bike riding", "pedaling on bicycle"],
    "cycling_horizontal": ["horizontal cycling", "recumbent cycling", "flat bike pedaling"],
    "cycling_vertical": ["vertical cycling", "upright stationary cycling", "upright bike"],
    "descending_stairs": ["descending stairs", "walking down stairs", "going downstairs"],
    "dribbling": ["dribbling a ball", "ball dribbling", "bouncing ball repeatedly"],
    "drinking": ["drinking a beverage", "sipping a drink", "taking a drink"],
    "eating_chips": ["eating chips", "snacking on chips", "eating crisps"],
    "eating_pasta": ["eating pasta", "eating noodles with fork", "having pasta"],
    "eating_sandwich": ["eating a sandwich", "having a sandwich", "biting into sandwich"],
    "eating_soup": ["eating soup", "having soup with spoon", "drinking soup"],
    "exercising_cross_trainer": ["cross trainer exercise", "elliptical machine", "elliptical workout"],
    "exercising_stepper": ["stepper exercise", "step machine workout", "stair stepper"],
    "falling_backward": ["falling backward", "backward fall", "toppling backwards"],
    "falling_backward_sitting": ["falling backward into sitting", "backward collapse to seat"],
    "falling_forward": ["falling forward", "forward fall", "tripping forward"],
    "falling_hitting_obstacle": ["falling and hitting obstacle", "collision fall", "fall with impact"],
    "falling_left": ["falling to the left", "leftward fall", "left side fall"],
    "falling_right": ["falling to the right", "rightward fall", "right side fall"],
    "falling_with_protection": ["protected fall", "controlled fall with protection", "braced fall"],
    "folding_clothes": ["folding clothes", "folding laundry", "clothing folding"],
    "frontal_elevation_arms": ["frontal arm elevation", "front arm raise", "raising arms forward"],
    "going_down_stairs": ["going down stairs", "walking downstairs", "descending steps"],
    "going_up_stairs": ["going up stairs", "walking upstairs", "ascending steps"],
    "ironing": ["ironing clothes", "pressing garments", "ironing fabric"],
    "jogging": ["jogging", "light running", "slow steady run"],
    "jump_front_back": ["jumping front and back", "forward backward jump", "front-back hop"],
    "jumping": ["jumping", "jump", "hopping in place"],
    "kicking": ["kicking", "leg kick motion", "kicking a ball"],
    "knees_bending": ["knee bending", "knee flexion exercise", "bending knees repeatedly"],
    "laying": ["laying down", "lying in reclined position", "resting horizontally"],
    "leg_curl": ["leg curl", "hamstring curl", "leg flexion machine"],
    "leg_press": ["leg press", "leg press machine", "pushing with legs"],
    "lie_to_sit": ["lying to sitting", "getting up from lying to sit", "rising to seated"],
    "lie_to_stand": ["lying to standing", "getting up from lying", "rising from horizontal"],
    "lying": ["lying down", "recumbent position", "horizontal rest"],
    "lying_back": ["lying on back", "supine position", "back-lying posture"],
    "lying_down_from_standing": ["lying down from standing", "lowering to ground", "going to lie down"],
    "lying_side": ["lying on side", "side-lying position", "lateral recumbent"],
    "moving_elevator": ["riding moving elevator", "in elevator moving", "elevator transport"],
    "nordic_walking": ["nordic walking", "pole walking", "walking with trekking poles"],
    "picking_up": ["picking up object", "bending to pick up", "retrieving item from ground"],
    "playing_basketball": ["playing basketball", "basketball game", "basketball activity"],
    "playing_catch": ["playing catch", "throwing and catching ball", "catch game"],
    "playing_sports": ["playing sports", "athletic activity", "sports game"],
    "push_up": ["push up exercise", "press up", "pushup repetitions"],
    "rope_jumping": ["rope jumping", "jump rope exercise", "skipping with rope"],
    "rope_skipping": ["rope skipping", "skip rope", "jumping rope exercise"],
    "rowing": ["rowing exercise", "rowing machine", "pulling rowing motion"],
    "running": ["running", "fast running", "sprinting locomotion"],
    "running_treadmill": ["running on treadmill", "treadmill running", "indoor treadmill run"],
    "sit_to_lie": ["sitting to lying", "reclining from seated", "transitioning sit to lie"],
    "sit_to_stand": ["sitting to standing", "standing up from chair", "rising from seat"],
    "sit_up": ["sit up exercise", "abdominal crunch", "core sit-up"],
    "sitting": ["sitting still", "seated stationary", "sitting in place"],
    "sitting_down": ["sitting down", "taking a seat", "lowering body to sit"],
    "squat": ["squat exercise", "deep knee bend", "squatting down and up"],
    "stairclimber": ["stair climber machine", "stairmaster exercise", "step climbing machine"],
    "stairs": ["using stairs", "stair activity", "going on stairs"],
    "stairs_down": ["descending stairs", "going down stairs", "walking down steps"],
    "stairs_up": ["ascending stairs", "going up stairs", "walking up steps"],
    "stand_to_lie": ["standing to lying", "lowering to horizontal", "going to lie from standing"],
    "stand_to_sit": ["standing to sitting", "sitting down from standing", "lowering to seat"],
    "standing": ["standing still", "upright stationary", "standing in place"],
    "standing_elevator": ["standing in stationary elevator", "waiting in elevator", "idle in elevator"],
    "standing_up_from_laying": ["standing up from laying", "getting up from floor", "rising from ground"],
    "standing_up_from_sitting": ["standing up from sitting", "rising from chair", "getting up from seat"],
    "syncope": ["syncope event", "fainting", "loss of consciousness collapse"],
    "talking_sitting": ["talking while sitting", "seated conversation", "chatting while seated"],
    "talking_standing": ["talking while standing", "standing conversation", "chatting while standing"],
    "typing": ["typing on keyboard", "keyboard typing", "computer key pressing"],
    "vacuum_cleaning": ["vacuum cleaning", "vacuuming floor", "using vacuum cleaner"],
    "waist_bends_forward": ["waist bending forward", "forward trunk bend", "bending at waist"],
    "walking": ["walking", "walk", "on foot at normal pace"],
    "walking_backwards": ["walking backwards", "reverse walking", "backward locomotion"],
    "walking_downstairs": ["walking downstairs", "descending stairs on foot", "going down steps walking"],
    "walking_parking": ["walking in parking area", "outdoor walking on pavement", "walking on flat ground"],
    "walking_treadmill_flat": ["walking on flat treadmill", "flat treadmill walking", "indoor level walking"],
    "walking_treadmill_incline": ["walking on inclined treadmill", "uphill treadmill walk", "incline walking"],
    "walking_upstairs": ["walking upstairs", "ascending stairs on foot", "going up steps walking"],
    "writing": ["writing by hand", "handwriting", "pen and paper writing"],
}


def build_text_protos(labels: List[str]) -> Dict[str, List[str]]:
    """Build text prototypes for all activity labels.

    Each label gets name variants + category-based sensor descriptions.
    """
    protos = {}
    for label in labels:
        names = ACTIVITY_NAMES.get(label, [label.replace("_", " ")])
        cat = ACTIVITY_CATEGORY.get(label, "fine_motor")
        sensors = CATEGORY_SENSORS.get(cat, CATEGORY_SENSORS["fine_motor"])
        protos[label] = list(names) + list(sensors)
    return protos


def wrap_template(label: str, description: str) -> str:
    """Wrap a description in the LanHAR template format."""
    return f"Activity={label.replace('_', ' ')}. Sensor pattern: {description}"


# =============================================================================
# Model Architecture (inlined from LanHAR)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

    def forward(self, src, src_key_padding_mask=None):
        x = self.input_linear(src)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x


class LanHARModel(nn.Module):
    """LanHAR model: SciBERT text encoder + TimeSeriesTransformer sensor encoder."""

    def __init__(self, bert_model=BERT_MODEL_NAME, max_len=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size  # 768
        self.max_len = max_len

        self.sensor_encoder = TimeSeriesTransformer(
            input_dim=DATA_CHANNELS,
            d_model=self.hidden_size,
            nhead=2,
            num_encoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.sen_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_text(self, input_ids, attention_mask):
        """Encode text through BERT with mean pooling."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom  # (B, H)

    def encode_sensor(self, time_series):
        """Encode sensor data through transformer, sum-pool over time."""
        h = self.sensor_encoder(time_series)  # (B, T, H)
        return h.sum(dim=1)  # (B, H)

    def get_text_embedding(self, input_ids, attention_mask):
        """Get projected and normalized text embedding."""
        e = self.encode_text(input_ids, attention_mask)
        return F.normalize(self.txt_proj(e), dim=-1)

    def get_sensor_embedding(self, time_series):
        """Get projected and normalized sensor embedding."""
        h = self.encode_sensor(time_series)
        return F.normalize(self.sen_proj(h), dim=-1)


# =============================================================================
# Loss Functions
# =============================================================================

def clip_loss(sensor_vec, text_vec, logit_scale):
    """Standard CLIP contrastive loss."""
    scale = logit_scale.exp().clamp(max=100.0)
    logits_st = (sensor_vec @ text_vec.T) * scale
    logits_ts = (text_vec @ sensor_vec.T) * scale
    labels = torch.arange(sensor_vec.size(0), device=sensor_vec.device)
    loss_st = F.cross_entropy(logits_st, labels)
    loss_ts = F.cross_entropy(logits_ts, labels)
    return 0.5 * (loss_st + loss_ts)


def build_label_prototypes(model, tokenizer, text_protos, labels, device,
                           topk=12, temperature=0.07):
    """Build label prototypes using margin-based top-k selection.

    For each class, encodes all text descriptions, picks top-k by margin
    (similarity to own class center minus max similarity to other centers),
    and computes weighted centroids using temperature-scaled softmax.

    Matches original LanHAR label_generation.py build_class_centers().

    Returns: (num_classes, hidden_size) tensor of normalized label prototypes
    """
    model_training = model.training
    model.train(False)

    # Encode all texts
    all_texts = []
    all_class_indices = []
    for i, label in enumerate(labels):
        for desc in text_protos.get(label, [label.replace("_", " ")]):
            wrapped = wrap_template(label, desc)
            all_texts.append(wrapped)
            all_class_indices.append(i)

    all_class_indices = np.array(all_class_indices)

    embeddings_list = []
    with torch.no_grad():
        for start in range(0, len(all_texts), 32):
            batch_texts = all_texts[start:start + 32]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=model.max_len, return_tensors="pt",
            )
            e = model.encode_text(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )
            e = F.normalize(e, dim=-1)
            embeddings_list.append(e.cpu().numpy())

    X = np.concatenate(embeddings_list, axis=0)  # (N, H)
    num_classes = len(labels)

    # Compute initial class centers (mean of all texts per class)
    centers = np.zeros((num_classes, X.shape[1]), dtype=np.float32)
    for i in range(num_classes):
        mask = all_class_indices == i
        if mask.sum() > 0:
            centers[i] = X[mask].mean(axis=0)
            centers[i] /= np.linalg.norm(centers[i]) + 1e-9

    # Compute similarities to all centers
    sims = X @ centers.T  # (N, num_classes)

    # Compute margin: own_sim - max_other_sim
    own_sim = sims[np.arange(len(X)), all_class_indices]
    other_sims = sims.copy()
    other_sims[np.arange(len(X)), all_class_indices] = -1e9
    max_other = other_sims.max(axis=1)
    margin = own_sim - max_other

    # Select top-k per class and compute weighted prototypes
    selected = np.zeros(len(X), dtype=bool)
    weights = np.zeros(len(X), dtype=np.float32)

    for i in range(num_classes):
        idx = np.where(all_class_indices == i)[0]
        if len(idx) == 0:
            continue
        k = min(topk, len(idx))
        top = idx[np.argsort(-margin[idx])[:k]]
        selected[top] = True

        scores = margin[top] / temperature
        w = np.exp(scores - scores.max())
        w = w / (w.sum() + 1e-9)
        weights[top] = w

    # Compute weighted prototypes
    prototypes = np.zeros((num_classes, X.shape[1]), dtype=np.float32)
    for i in range(num_classes):
        idx = np.where((all_class_indices == i) & selected)[0]
        if len(idx) == 0:
            idx = np.where(all_class_indices == i)[0]
            if len(idx) == 0:
                continue
            w = np.ones(len(idx), dtype=np.float32) / len(idx)
        else:
            w = weights[idx]
        prototypes[i] = (X[idx] * w[:, None]).sum(axis=0)
        prototypes[i] /= np.linalg.norm(prototypes[i]) + 1e-9

    if model_training:
        model.train()

    return torch.from_numpy(prototypes).float().to(device)


def clip_loss_multipos(z_a, z_b, labels, temperature=0.1, eps=1e-12):
    """Multi-positive CLIP loss for supervised contrastive learning."""
    dev = z_a.device
    scale = torch.tensor(1.0 / max(float(temperature), 1e-6), device=dev)

    logits_ab = (z_a @ z_b.T) * scale
    logits_ba = (z_b @ z_a.T) * scale

    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    target_ab = same.float()
    target_ba = same.t().float()

    target_ab = target_ab / (target_ab.sum(dim=1, keepdim=True) + eps)
    target_ba = target_ba / (target_ba.sum(dim=1, keepdim=True) + eps)

    log_q_ab = logits_ab - torch.logsumexp(logits_ab, dim=1, keepdim=True)
    log_q_ba = logits_ba - torch.logsumexp(logits_ba, dim=1, keepdim=True)

    loss_ab = -(target_ab * log_q_ab).sum(dim=1).mean()
    loss_ba = -(target_ba * log_q_ba).sum(dim=1).mean()
    return 0.5 * (loss_ab + loss_ba)


# =============================================================================
# Stage 1: Text Encoder Fine-tuning
# =============================================================================

def train_stage1(model, tokenizer, text_protos, labels, device,
                 epochs=STAGE1_EPOCHS, lr=STAGE1_LR, batch_size=STAGE1_BATCH_SIZE):
    """Fine-tune SciBERT with supervised contrastive + cross-entropy on text prototypes.

    Matches original LanHAR Stage 1 losses:
      - Multi-positive CLIP loss on text embeddings
      - Cross-entropy with label prototypes (cls_scale=30.0, lam=0.3)
    Label prototypes are recomputed each epoch using margin-based top-k selection.
    """
    print(f"\n  Stage 1: Fine-tuning SciBERT ({epochs} epochs, lr={lr})...")

    # Build all text entries with labels
    all_texts = []
    all_label_indices = []
    for i, label in enumerate(labels):
        for desc in text_protos.get(label, [label.replace("_", " ")]):
            wrapped = wrap_template(label, desc)
            all_texts.append(wrapped)
            all_label_indices.append(i)

    all_label_indices = torch.tensor(all_label_indices, dtype=torch.long)
    print(f"    Text entries: {len(all_texts)} across {len(labels)} classes")

    optimizer = torch.optim.AdamW(model.bert.parameters(), lr=lr)
    model.train()

    cls_scale = 30.0
    lam = 0.3

    for epoch in range(epochs):
        # Rebuild label prototypes each epoch (detached, no_grad)
        label_protos = build_label_prototypes(
            model, tokenizer, text_protos, labels, device,
            topk=12, temperature=0.07,
        )  # (num_classes, H), normalized, detached

        model.train()
        perm = torch.randperm(len(all_texts))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(all_texts), batch_size):
            batch_idx = perm[i : i + batch_size]
            if len(batch_idx) < 2:
                continue

            batch_texts = [all_texts[j] for j in batch_idx]
            batch_labels = all_label_indices[batch_idx].to(device)

            enc = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=model.max_len, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            text_emb = F.normalize(model.encode_text(input_ids, attention_mask), dim=-1)

            # Get prototypes for batch labels
            batch_protos = F.normalize(label_protos[batch_labels], dim=-1)

            # Loss 1: Multi-positive CLIP loss
            loss_clip = clip_loss_multipos(text_emb, batch_protos, batch_labels, temperature=0.1)

            # Loss 2: Cross-entropy with label prototypes (matching original LanHAR)
            logits_cls = (text_emb @ label_protos.T) * cls_scale
            loss_ce = F.cross_entropy(logits_cls, batch_labels)

            loss = loss_clip + lam * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# =============================================================================
# Stage 2: Sensor Encoder Training
# =============================================================================

class SensorTextDataset(Dataset):
    """Dataset pairing sensor windows with text descriptions for CLIP training."""

    def __init__(self, sensor_data, labels, text_protos, label_list):
        self.sensor_data = sensor_data
        self.labels = labels
        self.text_protos = text_protos
        self.label_list = label_list

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, idx):
        ts = self.sensor_data[idx]
        label_idx = int(self.labels[idx])
        label_name = self.label_list[label_idx]

        desc = random.choice(self.text_protos[label_name])
        text = wrap_template(label_name, desc)

        return torch.from_numpy(ts).float(), text, label_idx


def collate_stage2(batch, tokenizer, max_len=512):
    """Collate sensor-text pairs for Stage 2 training."""
    ts_list, texts, labels = zip(*batch)
    ts = torch.stack(ts_list)
    labels = torch.tensor(labels, dtype=torch.long)

    enc = tokenizer(
        list(texts), padding=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    return ts, enc["input_ids"], enc["attention_mask"], labels


def train_stage2(model, tokenizer, train_data, train_labels, text_protos, label_list,
                 device, epochs=STAGE2_EPOCHS, lr=STAGE2_LR, batch_size=STAGE2_BATCH_SIZE):
    """Train sensor encoder with CLIP loss on sensor-text pairs.

    Uses validation-based model selection (saves best model by val loss).
    """
    print(f"\n  Stage 2: Training sensor encoder ({epochs} epochs, lr={lr}, "
          f"batch_size={batch_size})...")
    print(f"    Training samples: {len(train_data)}")

    # Freeze BERT
    for p in model.bert.parameters():
        p.requires_grad = False

    # Optimize sensor encoder + projections + logit_scale
    optim_params = (
        list(model.sensor_encoder.parameters())
        + list(model.txt_proj.parameters())
        + list(model.sen_proj.parameters())
        + [model.logit_scale]
    )
    optimizer = torch.optim.AdamW(optim_params, lr=lr)
    scaler = GradScaler()

    # Split into train/val (90/10)
    rng = np.random.RandomState(SEED)
    n_total = len(train_data)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    val_n = int(n_total * 0.1)
    val_indices = indices[:val_n]
    train_indices = indices[val_n:]

    train_dataset = SensorTextDataset(
        train_data[train_indices], train_labels[train_indices],
        text_protos, label_list,
    )
    val_dataset = SensorTextDataset(
        train_data[val_indices], train_labels[val_indices],
        text_protos, label_list,
    )

    collate_fn = partial(collate_stage2, tokenizer=tokenizer, max_len=model.max_len)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    best_val_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0

        for ts, input_ids, attention_mask, labels_batch in train_loader:
            ts = ts.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast("cuda"):
                text_vec = model.get_text_embedding(input_ids, attention_mask)
                sensor_vec = model.get_sensor_embedding(ts)
                loss = clip_loss(sensor_vec, text_vec, model.logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)

        # Validation
        model.train(False)
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for ts, input_ids, attention_mask, labels_batch in val_loader:
                ts = ts.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                with amp.autocast("cuda"):
                    text_vec = model.get_text_embedding(input_ids, attention_mask)
                    sensor_vec = model.get_sensor_embedding(ts)
                    loss = clip_loss(sensor_vec, text_vec, model.logit_scale)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    Restored best model (val_loss={best_val_loss:.4f})")

    # Unfreeze BERT for potential further use
    for p in model.bert.parameters():
        p.requires_grad = True


# =============================================================================
# Embedding Extraction
# =============================================================================

def extract_lanhar_embeddings(
    model, raw_data: np.ndarray, device: torch.device, batch_size: int = 256,
) -> np.ndarray:
    """Extract sensor embeddings from trained LanHAR model.

    Returns (N, 768) normalized embeddings.
    """
    model.eval()
    N = raw_data.shape[0]
    all_embeddings = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(raw_data[start:end]).float().to(device)

        with torch.no_grad():
            emb = model.get_sensor_embedding(batch)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw sensor data and labels for a dataset."""
    ds_dir = LIMUBERT_DATA_DIR / dataset_name
    data = np.load(str(ds_dir / "data_20_120.npy")).astype(np.float32)
    labels = np.load(str(ds_dir / "label_20_120.npy")).astype(np.float32)
    return data, labels


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Extract per-window activity labels (majority vote)."""
    act_labels = labels_raw[:, :, label_index]
    t = int(np.min(act_labels))
    act_labels = act_labels - t
    window_labels = np.array(
        [np.bincount(row.astype(int)).argmax() for row in act_labels], dtype=np.int64
    )
    return window_labels


def get_dataset_labels(dataset_name: str) -> List[str]:
    """Get sorted activity labels for a dataset."""
    return sorted(DATASET_CONFIG["datasets"][dataset_name]["activities"])


# =============================================================================
# Data Splitting
# =============================================================================

def balanced_subsample(data, labels, rate, rng):
    """Balanced subsampling: equal samples per class."""
    unique_labels = np.unique(labels)
    n_total = max(1, int(len(data) * rate))
    n_per_class = max(1, n_total // len(unique_labels))

    selected_idx = []
    for lbl in unique_labels:
        class_idx = np.where(labels == lbl)[0]
        rng.shuffle(class_idx)
        selected_idx.extend(class_idx[:n_per_class])

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)
    return data[selected_idx], labels[selected_idx]


def prepare_train_test_split(data, labels, training_rate=0.8, vali_rate=0.1,
                              label_rate=1.0, seed=CLASSIFIER_SEED, balance=True):
    """Split data into train/val/test."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    train_n = int(len(data) * training_rate)
    vali_n = int(len(data) * vali_rate)

    train_data = data[:train_n]
    train_labels = labels[:train_n]
    vali_data = data[train_n : train_n + vali_n]
    vali_labels = labels[train_n : train_n + vali_n]
    test_data = data[train_n + vali_n :]
    test_labels = labels[train_n + vali_n :]

    if label_rate < 1.0:
        if balance:
            train_data, train_labels = balanced_subsample(
                train_data, train_labels, label_rate, rng
            )
        else:
            n_labeled = max(1, int(len(train_data) * label_rate))
            train_data = train_data[:n_labeled]
            train_labels = train_labels[:n_labeled]

    return train_data, train_labels, vali_data, vali_labels, test_data, test_labels


# =============================================================================
# Classifier
# =============================================================================

class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x, training=False):
        if training:
            x = F.dropout(x, p=0.3, training=True)
        return self.linear(x)


def train_linear_classifier(
    train_data, train_labels, val_data, val_labels,
    num_classes, input_dim=EMB_DIM,
    epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
    lr=CLASSIFIER_LR, device=None, verbose=False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LinearClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_data).float(),
        torch.from_numpy(train_labels).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_data).float(),
        torch.from_numpy(val_labels).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_data, training=True)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_gt = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data, training=False)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_gt.extend(batch_labels.numpy())

        val_acc = accuracy_score(val_gt, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 20 == 0:
            val_f1 = f1_score(val_gt, val_preds, average="macro", zero_division=0)
            print(f"    Epoch {epoch+1}/{epochs}: val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_classifier(model, data, batch_size=CLASSIFIER_BATCH_SIZE, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    ds = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data, training=False)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


# =============================================================================
# Scoring Functions
# =============================================================================

def score_open_set(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 1: Zero-shot open-set scoring."""
    print("  [Open-set] Preparing training data with all labels...")
    label_to_group = get_label_to_group_mapping()
    test_activities = get_dataset_labels(test_dataset)
    global_label_to_idx = {label: i for i, label in enumerate(GLOBAL_LABELS)}
    num_global_classes = len(GLOBAL_LABELS)

    all_train_data = []
    all_train_labels = []

    for ds_name in TRAIN_DATASETS:
        if ds_name not in train_embeddings:
            continue
        emb, labels = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        for i in range(len(labels)):
            local_idx = labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                global_idx = global_label_to_idx.get(activity_name, -1)
                if global_idx >= 0:
                    all_train_data.append(emb[i])
                    all_train_labels.append(global_idx)

    all_train_data = np.array(all_train_data)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)

    print(f"  [Open-set] Training data: {len(all_train_data)} samples, "
          f"{num_global_classes} classes")

    rng = np.random.RandomState(CLASSIFIER_SEED)
    idx = np.arange(len(all_train_data))
    rng.shuffle(idx)
    all_train_data = all_train_data[idx]
    all_train_labels = all_train_labels[idx]

    val_n = int(len(all_train_data) * 0.1)
    train_data = all_train_data[val_n:]
    train_labels_arr = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels_arr = all_train_labels[:val_n]

    print(f"  [Open-set] Training linear classifier ({num_global_classes} classes)...")
    clf = train_linear_classifier(
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_global_classes, device=device, verbose=True,
    )

    pred_global_indices = predict_classifier(clf, test_embeddings, device=device)

    pred_groups = []
    gt_groups = []
    for i in range(len(test_labels)):
        local_idx = test_labels[i]
        if local_idx < len(test_activities):
            gt_name = test_activities[local_idx]
        else:
            continue
        gt_group = label_to_group.get(gt_name, gt_name)

        pred_idx = pred_global_indices[i]
        pred_name = GLOBAL_LABELS[pred_idx] if pred_idx < len(GLOBAL_LABELS) else "unknown"
        pred_group = label_to_group.get(pred_name, pred_name)

        gt_groups.append(gt_group)
        pred_groups.append(pred_group)

    acc = accuracy_score(gt_groups, pred_groups) * 100
    f1 = f1_score(gt_groups, pred_groups, average="macro", zero_division=0) * 100

    return {"accuracy": acc, "f1_macro": f1, "n_samples": len(gt_groups),
            "n_train_samples": len(train_data), "n_classes_train": num_global_classes}


def score_closed_set(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 2: Closed-set scoring."""
    label_to_group = get_label_to_group_mapping()
    test_activities = get_dataset_labels(test_dataset)
    test_label_groups = {label_to_group.get(a, a) for a in test_activities}

    group_to_test_label = {}
    for act in test_activities:
        group = label_to_group.get(act, act)
        group_to_test_label[group] = act

    test_label_to_idx = {a: i for i, a in enumerate(test_activities)}
    num_test_classes = len(test_activities)

    print(f"  [Closed-set] Collecting training data for {num_test_classes} test classes...")

    all_train_data = []
    all_train_labels = []

    for ds_name in TRAIN_DATASETS:
        if ds_name not in train_embeddings:
            continue
        emb, labels = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        for i in range(len(labels)):
            local_idx = labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                group = label_to_group.get(activity_name, activity_name)
                if group in test_label_groups:
                    test_label = group_to_test_label.get(group)
                    if test_label is not None:
                        test_idx = test_label_to_idx[test_label]
                        all_train_data.append(emb[i])
                        all_train_labels.append(test_idx)

    all_train_data = np.array(all_train_data)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)

    covered_classes = len(np.unique(all_train_labels))
    print(f"  [Closed-set] Training data: {len(all_train_data)} samples, "
          f"{covered_classes}/{num_test_classes} classes covered")

    if len(all_train_data) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "n_samples": 0,
                "n_classes": num_test_classes}

    rng = np.random.RandomState(CLASSIFIER_SEED)
    idx = np.arange(len(all_train_data))
    rng.shuffle(idx)
    all_train_data = all_train_data[idx]
    all_train_labels = all_train_labels[idx]

    val_n = int(len(all_train_data) * 0.1)
    train_data = all_train_data[val_n:]
    train_labels_arr = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels_arr = all_train_labels[:val_n]

    print(f"  [Closed-set] Training linear classifier ({num_test_classes} classes)...")
    clf = train_linear_classifier(
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_test_classes, device=device, verbose=True,
    )

    pred_indices = predict_classifier(clf, test_embeddings, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(test_labels)):
        local_idx = test_labels[i]
        if local_idx < len(test_activities):
            gt_names.append(test_activities[local_idx])
        else:
            continue
        pred_idx = pred_indices[i]
        pred_names.append(
            test_activities[pred_idx] if pred_idx < len(test_activities) else "unknown"
        )

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(gt_names, pred_names, average="macro", zero_division=0) * 100

    return {"accuracy": acc, "f1_macro": f1, "n_samples": len(gt_names),
            "n_train_samples": len(train_data), "n_classes": num_test_classes,
            "covered_classes": covered_classes}


def score_1pct_supervised(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 3: 1% supervised scoring."""
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [1% supervised] Total samples: {len(test_embeddings)}, "
          f"{num_test_classes} classes")

    train_data, train_labels_arr, val_data, val_labels_arr, test_data, test_labels_arr = \
        prepare_train_test_split(
            test_embeddings, test_labels,
            training_rate=TRAINING_RATE, vali_rate=VALI_RATE,
            label_rate=SUPERVISED_LABEL_RATE, seed=CLASSIFIER_SEED, balance=True,
        )

    print(f"  [1% supervised] Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    if len(train_data) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "n_samples": 0,
                "n_classes": num_test_classes}

    print(f"  [1% supervised] Training linear classifier ({num_test_classes} classes)...")
    clf = train_linear_classifier(
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_test_classes, device=device, verbose=True,
    )

    pred_indices = predict_classifier(clf, test_data, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(test_labels_arr)):
        local_idx = test_labels_arr[i]
        if local_idx < len(test_activities):
            gt_names.append(test_activities[local_idx])
        else:
            continue
        pred_idx = pred_indices[i]
        pred_names.append(
            test_activities[pred_idx] if pred_idx < len(test_activities) else "unknown"
        )

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(gt_names, pred_names, average="macro", zero_division=0) * 100

    return {"accuracy": acc, "f1_macro": f1, "n_samples": len(gt_names),
            "n_train_samples": len(train_data), "n_classes": num_test_classes}


# =============================================================================
# Main
# =============================================================================

def print_results_table(all_results):
    print()
    print("=" * 94)
    print("LanHAR BASELINE RESULTS")
    print("=" * 94)

    header = (f"{'Dataset':<16}"
              f"{'Open-Set Acc':>13}{'Open-Set F1':>13}"
              f"{'Closed Acc':>13}{'Closed F1':>13}"
              f"{'1% Sup Acc':>13}{'1% Sup F1':>13}")
    print(header)
    print("-" * 94)

    for ds in TEST_DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        os_acc = r.get("open_set", {}).get("accuracy", 0.0)
        os_f1 = r.get("open_set", {}).get("f1_macro", 0.0)
        cs_acc = r.get("closed_set", {}).get("accuracy", 0.0)
        cs_f1 = r.get("closed_set", {}).get("f1_macro", 0.0)
        sup_acc = r.get("1pct_supervised", {}).get("accuracy", 0.0)
        sup_f1 = r.get("1pct_supervised", {}).get("f1_macro", 0.0)
        print(
            f"{ds:<16}"
            f"{os_acc:>12.1f}%{os_f1:>12.1f}%"
            f"{cs_acc:>12.1f}%{cs_f1:>12.1f}%"
            f"{sup_acc:>12.1f}%{sup_f1:>12.1f}%"
        )

    print("=" * 94)
    print()
    print("Details:")
    print(f"  Model: LanHAR (SciBERT + TimeSeriesTransformer)")
    print(f"  Text encoder: {BERT_MODEL_NAME}")
    print(f"  Stage 1: {STAGE1_EPOCHS} epochs text contrastive+CE, lr={STAGE1_LR}, "
          f"batch_size={STAGE1_BATCH_SIZE}")
    print(f"  Stage 2: {STAGE2_EPOCHS} epochs sensor-text CLIP, lr={STAGE2_LR}, "
          f"batch_size={STAGE2_BATCH_SIZE}, val-based selection")
    print(f"  Sensor embedding dim: {EMB_DIM}")
    print(f"  Classifier: Linear, {CLASSIFIER_EPOCHS} epochs, lr={CLASSIFIER_LR}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build text prototypes for all 87 classes
    text_protos = build_text_protos(GLOBAL_LABELS)
    total_descs = sum(len(v) for v in text_protos.values())
    print(f"\nText prototypes: {len(text_protos)} classes, {total_descs} total descriptions")

    # Initialize model and tokenizer
    print(f"\nLoading LanHAR model (SciBERT: {BERT_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = LanHARModel(BERT_MODEL_NAME).float().to(device)
    print("Model loaded successfully")

    # Load all training data with global labels
    print("\nLoading training data...")
    global_label_to_idx = {label: i for i, label in enumerate(GLOBAL_LABELS)}
    all_train_sensor = []
    all_train_labels = []

    for ds in TRAIN_DATASETS:
        raw_data, raw_labels = load_raw_data(ds)
        labels = get_window_labels(raw_labels)
        ds_activities = get_dataset_labels(ds)

        ds_count = 0
        for i in range(len(labels)):
            local_idx = labels[i]
            if local_idx < len(ds_activities):
                activity = ds_activities[local_idx]
                global_idx = global_label_to_idx.get(activity, -1)
                if global_idx >= 0:
                    all_train_sensor.append(raw_data[i])
                    all_train_labels.append(global_idx)
                    ds_count += 1
        print(f"  {ds}: {ds_count} samples")

    all_train_sensor = np.array(all_train_sensor, dtype=np.float32)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)
    print(f"Total training data: {len(all_train_sensor)} samples, "
          f"{len(np.unique(all_train_labels))} classes")

    # Stage 1: Fine-tune text encoder
    train_stage1(model, tokenizer, text_protos, GLOBAL_LABELS, device)

    # Stage 2: Train sensor encoder
    train_stage2(
        model, tokenizer, all_train_sensor, all_train_labels,
        text_protos, GLOBAL_LABELS, device,
    )

    # Extract embeddings for all training datasets
    print("\nExtracting training embeddings...")
    train_embeddings = {}
    for ds in TRAIN_DATASETS:
        raw_data, raw_labels = load_raw_data(ds)
        emb = extract_lanhar_embeddings(model, raw_data, device)
        labels = get_window_labels(raw_labels)
        train_embeddings[ds] = (emb, labels)
        print(f"  {ds}: {emb.shape[0]} windows -> embeddings {emb.shape}")

    print(f"\nExtracted embeddings for {len(train_embeddings)}/{len(TRAIN_DATASETS)} "
          f"training datasets")

    # Run scoring on each test dataset
    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Testing LanHAR on {test_ds}")
        print(f"{'='*60}")

        raw_data, raw_labels = load_raw_data(test_ds)
        test_emb = extract_lanhar_embeddings(model, raw_data, device)
        test_labels = get_window_labels(raw_labels)
        test_activities = get_dataset_labels(test_ds)

        print(f"  Test data: {test_emb.shape[0]} windows -> embeddings {test_emb.shape}, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        # Metric 1: Open-set
        print(f"\n  --- Metric 1: Zero-shot Open-Set ---")
        ds_results["open_set"] = score_open_set(
            train_embeddings, test_emb, test_labels, test_ds, device
        )
        print(f"  Open-set: Acc={ds_results['open_set']['accuracy']:.1f}%, "
              f"F1={ds_results['open_set']['f1_macro']:.1f}%")

        # Metric 2: Closed-set
        print(f"\n  --- Metric 2: Closed-Set ---")
        ds_results["closed_set"] = score_closed_set(
            train_embeddings, test_emb, test_labels, test_ds, device
        )
        print(f"  Closed-set: Acc={ds_results['closed_set']['accuracy']:.1f}%, "
              f"F1={ds_results['closed_set']['f1_macro']:.1f}%")

        # Metric 3: 1% supervised
        print(f"\n  --- Metric 3: 1% Supervised ---")
        ds_results["1pct_supervised"] = score_1pct_supervised(
            test_emb, test_labels, test_ds, device
        )
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        all_results[test_ds] = ds_results

    # Print summary table
    print_results_table(all_results)

    # Save results
    results_path = OUTPUT_DIR / "lanhar_results.json"
    save_data = {}
    for ds, metrics in all_results.items():
        save_data[ds] = {}
        for metric_name, metric_vals in metrics.items():
            save_data[ds][metric_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metric_vals.items()
            }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
