"""
Dataset-specific label augmentation for IMU activity recognition.

Each dataset gets custom synonyms and templates tailored to its activities.
This provides rich, natural language variations for contrastive learning.
"""

import random
from typing import List, Dict

# ============================================================================
# UCI-HAR: Basic 6 activities (lab-controlled)
# ============================================================================

UCI_HAR_SYNONYMS = {
    "walking": ["walking", "strolling", "striding", "ambulating", "pacing"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "ascending stairs", "going upstairs", "stair climbing"],
    "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descending"],
    "sitting": ["sitting", "seated", "sitting down", "in a seated position"],
    "standing": ["standing", "standing up", "upright", "in a standing position", "on feet"],
    "laying": ["laying", "lying down", "reclining", "horizontal", "lying", "supine"],
}

UCI_HAR_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "subject {}",
    "user {}",
    "{} activity",
    "{} posture",
    "body {}",
]

# ============================================================================
# MHEALTH: Exercise and daily activities (medical monitoring)
# ============================================================================

MHEALTH_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating", "taking a walk"],
    "jogging": ["jogging", "light running", "slow running", "jog"],
    "running": ["running", "sprinting", "fast running", "run"],
    "cycling": ["cycling", "riding a bike", "pedaling", "biking"],
    "climbing_stairs": ["climbing stairs", "stair climbing", "ascending stairs", "going up stairs"],
    "sitting": ["sitting", "seated", "sitting down"],
    "standing": ["standing", "upright", "standing still"],
    "lying": ["lying", "laying down", "horizontal", "supine", "reclining"],
    "frontal_elevation_arms": ["frontal arm elevation", "raising arms forward", "arm lifting", "frontal arm raise"],
    "knees_bending": ["knee bending", "squatting", "knee flexion", "bending knees"],
    "waist_bends_forward": ["forward waist bend", "bending forward", "torso flexion", "waist bending"],
    "jump_front_back": ["jumping front and back", "forward-backward jumping", "jump exercise"],
}

MHEALTH_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "patient {}",
    "subject performing {}",
    "{} motion",
    "{} exercise",
    "{} movement",
    "health monitoring during {}",
]

# ============================================================================
# PAMAP2: Daily and sports activities (physical activity monitoring)
# ============================================================================

PAMAP2_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating", "casual walking"],
    "nordic_walking": ["nordic walking", "pole walking", "nordic walk", "walking with poles"],
    "running": ["running", "jogging", "fast running"],
    "cycling": ["cycling", "biking", "riding bicycle", "pedaling"],
    "ascending_stairs": ["ascending stairs", "climbing stairs", "going upstairs", "stair ascent"],
    "descending_stairs": ["descending stairs", "going downstairs", "stair descent", "walking downstairs"],
    "rope_jumping": ["rope jumping", "jump rope", "skipping rope", "jumping rope"],
    "sitting": ["sitting", "seated", "sitting down"],
    "standing": ["standing", "upright", "standing still"],
    "lying": ["lying", "laying down", "horizontal", "reclining"],
    "ironing": ["ironing", "pressing clothes", "ironing clothes"],
    "vacuum_cleaning": ["vacuum cleaning", "vacuuming", "cleaning with vacuum", "using vacuum cleaner"],
}

PAMAP2_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "user performing {}",
    "{} activity",
    "physical activity: {}",
    "daily activity: {}",
    "{} behavior",
]

# ============================================================================
# WISDM: Diverse daily activities with hand gestures
# ============================================================================

WISDM_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating"],
    "jogging": ["jogging", "light jogging", "slow running"],
    "stairs": ["using stairs", "stair activity", "stair movement", "on stairs"],
    "sitting": ["sitting", "seated", "sitting down"],
    "standing": ["standing", "upright", "standing still"],

    # Eating activities
    "eating_pasta": ["eating pasta", "consuming pasta", "having pasta"],
    "eating_chips": ["eating chips", "snacking on chips", "consuming chips", "eating crisps"],
    "eating_sandwich": ["eating sandwich", "having a sandwich", "consuming sandwich"],
    "eating_soup": ["eating soup", "having soup", "consuming soup", "spooning soup"],
    "drinking": ["drinking", "having a drink", "consuming beverage"],

    # Hand activities
    "brushing_teeth": ["brushing teeth", "tooth brushing", "dental hygiene", "oral care"],
    "typing": ["typing", "keyboard typing", "texting", "using keyboard"],
    "writing": ["writing", "handwriting", "writing by hand", "penmanship"],
    "clapping": ["clapping", "hand clapping", "applauding", "clapping hands"],
    "folding_clothes": ["folding clothes", "clothing folding", "folding laundry", "organizing clothes"],

    # Sports activities
    "playing_catch": ["playing catch", "throwing and catching", "tossing ball", "catch game"],
    "dribbling": ["dribbling", "ball dribbling", "dribbling basketball"],
    "kicking": ["kicking", "kicking ball", "foot kicking"],
}

WISDM_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "user {}",
    "individual {}",
    "{} activity",
    "{} gesture",
    "{} action",
    "human {}",
    "smartphone user {}",
]

# ============================================================================
# MotionSense: Phone pocket activities (similar to UCI-HAR)
# ============================================================================

MOTIONSENSE_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating", "taking steps"],
    "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descent", "walking down stairs"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "going upstairs", "ascending stairs", "walking up stairs"],
    "jogging": ["jogging", "light running", "slow running", "jog", "light jog"],
    "sitting": ["sitting", "seated", "sitting down", "in a seated position"],
    "standing": ["standing", "upright", "standing still", "on feet", "standing up"],
}

MOTIONSENSE_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "smartphone user {}",
    "{} activity",
    "mobile phone sensing {}",
    "phone in pocket {}",
]

# ============================================================================
# UniMiB SHAR: ADL and postural transitions
# ============================================================================

UNIMIB_SHAR_SYNONYMS = {
    "standing_up_from_sitting": ["standing up from sitting", "rising from chair", "getting up from seated", "sit-to-stand"],
    "standing_up_from_laying": ["standing up from laying", "getting out of bed", "rising from horizontal", "lay-to-stand"],
    "walking": ["walking", "strolling", "ambulating", "taking steps"],
    "running": ["running", "jogging", "sprinting", "fast movement"],
    "going_up_stairs": ["going up stairs", "climbing stairs", "ascending stairs", "stair ascent"],
    "jumping": ["jumping", "hopping", "leaping", "vertical jump"],
    "going_down_stairs": ["going down stairs", "descending stairs", "walking downstairs", "stair descent"],
    "lying_down_from_standing": ["lying down from standing", "going to bed", "laying down", "stand-to-lay"],
    "sitting_down_from_standing": ["sitting down from standing", "taking a seat", "sitting down", "stand-to-sit"],
}

UNIMIB_SHAR_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "{} transition",
    "postural transition: {}",
    "activity: {}",
    "daily movement: {}",
]

# ============================================================================
# HHAR: Heterogeneity Activity Recognition (multi-device)
# ============================================================================

HHAR_SYNONYMS = {
    "standing": ["standing", "upright", "standing still", "on feet"],
    "sitting": ["sitting", "seated", "sitting down", "in a seated position"],
    "walking": ["walking", "strolling", "ambulating", "taking steps"],
    "cycling": ["cycling", "biking", "riding bicycle", "pedaling"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "ascending stairs", "going upstairs"],
    "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descent"],
}

HHAR_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "smartphone user {}",
    "{} activity",
    "heterogeneous device {}",
    "mobile sensing {}",
]

# ============================================================================
# Main augmentation function
# ============================================================================

# Map dataset names to their augmentation configs
DATASET_CONFIGS = {
    "uci_har": {
        "synonyms": UCI_HAR_SYNONYMS,
        "templates": UCI_HAR_TEMPLATES,
    },
    "mhealth": {
        "synonyms": MHEALTH_SYNONYMS,
        "templates": MHEALTH_TEMPLATES,
    },
    "pamap2": {
        "synonyms": PAMAP2_SYNONYMS,
        "templates": PAMAP2_TEMPLATES,
    },
    "wisdm": {
        "synonyms": WISDM_SYNONYMS,
        "templates": WISDM_TEMPLATES,
    },
    "unimib_shar": {
        "synonyms": UNIMIB_SHAR_SYNONYMS,
        "templates": UNIMIB_SHAR_TEMPLATES,
    },
    "hhar": {
        "synonyms": HHAR_SYNONYMS,
        "templates": HHAR_TEMPLATES,
    },
    "motionsense": {
        "synonyms": MOTIONSENSE_SYNONYMS,
        "templates": MOTIONSENSE_TEMPLATES,
    },
    # New datasets - minimal configs for label retrieval
    # Training datasets
    "dsads": {
        "synonyms": {
            "sitting": ["sitting"],
            "standing": ["standing"],
            "lying_back": ["lying_back", "lying on back"],
            "lying_side": ["lying_side", "lying on side"],
            "stairs_up": ["stairs_up", "ascending stairs"],
            "stairs_down": ["stairs_down", "descending stairs"],
            "walking_parking": ["walking_parking", "walking in parking lot"],
            "walking_treadmill_flat": ["walking_treadmill_flat", "walking on treadmill"],
            "walking_treadmill_incline": ["walking_treadmill_incline", "walking on inclined treadmill"],
            "running_treadmill": ["running_treadmill", "running on treadmill"],
            "exercising_stepper": ["exercising_stepper", "using stepper machine"],
            "exercising_cross_trainer": ["exercising_cross_trainer", "using elliptical"],
            "cycling_horizontal": ["cycling_horizontal", "cycling recumbent"],
            "cycling_vertical": ["cycling_vertical", "cycling upright"],
            "rowing": ["rowing", "rowing machine"],
            "jumping": ["jumping"],
            "playing_basketball": ["playing_basketball", "basketball"],
            "moving_elevator": ["moving_elevator", "in elevator"],
            "standing_elevator": ["standing_elevator", "standing in elevator"],
        },
        "templates": ["{}"],
    },
    "mobiact": {
        "synonyms": {
            "standing": ["standing"],
            "walking": ["walking"],
            "jogging": ["jogging"],
            "jumping": ["jumping"],
            "stairs_up": ["stairs_up", "ascending stairs"],
            "stairs_down": ["stairs_down", "descending stairs"],
            "sitting_chair": ["sitting_chair", "sitting on chair"],
            "car_step_in": ["car_step_in", "getting into car"],
            "car_step_out": ["car_step_out", "getting out of car"],
            "fall_forward": ["fall_forward", "falling forward"],
            "fall_backward_knees": ["fall_backward_knees", "falling backward onto knees"],
            "fall_backward_sitting": ["fall_backward_sitting", "falling backward into sitting"],
            "fall_sideways": ["fall_sideways", "falling sideways"],
        },
        "templates": ["{}"],
    },
    "hapt": {
        "synonyms": {
            # Basic activities (same as UCI HAR)
            "walking": ["walking"],
            "walking_upstairs": ["walking_upstairs", "ascending stairs"],
            "walking_downstairs": ["walking_downstairs", "descending stairs"],
            "sitting": ["sitting"],
            "standing": ["standing"],
            "lying": ["lying"],
            # Postural transitions
            "stand_to_sit": ["stand_to_sit", "transitioning from standing to sitting"],
            "sit_to_stand": ["sit_to_stand", "transitioning from sitting to standing"],
            "sit_to_lie": ["sit_to_lie", "transitioning from sitting to lying"],
            "lie_to_sit": ["lie_to_sit", "transitioning from lying to sitting"],
            "stand_to_lie": ["stand_to_lie", "transitioning from standing to lying"],
            "lie_to_stand": ["lie_to_stand", "transitioning from lying to standing"],
        },
        "templates": ["{}"],
    },
    "kuhar": {
        "synonyms": {
            # Basic postures
            "standing": ["standing"],
            "sitting": ["sitting"],
            "lying": ["lying"],
            # Locomotion
            "walking": ["walking"],
            "walking_backwards": ["walking_backwards", "walking backward"],
            "walking_upstairs": ["walking_upstairs", "ascending stairs"],
            "walking_downstairs": ["walking_downstairs", "descending stairs"],
            "running": ["running"],
            "jumping": ["jumping"],
            # Transitions
            "standing_up_from_sitting": ["standing_up_from_sitting", "standing up from chair"],
            "standing_up_from_laying": ["standing_up_from_laying", "getting up from lying"],
            # Activities
            "picking_up": ["picking_up", "picking up object"],
            "push_up": ["push_up", "doing push-ups"],
            "sit_up": ["sit_up", "doing sit-ups"],
            "talking_sitting": ["talking_sitting", "talking while sitting"],
            "talking_standing": ["talking_standing", "talking while standing"],
            "playing_sports": ["playing_sports", "playing table tennis"],
        },
        "templates": ["{}"],
    },
    # Zero-shot datasets
    "realworld": {
        "synonyms": {
            "walking": ["walking"],
            "running": ["running"],
            "sitting": ["sitting"],
            "standing": ["standing"],
            "lying": ["lying"],
            "stairs_up": ["stairs_up", "climbing stairs"],
            "stairs_down": ["stairs_down", "descending stairs"],
            "jumping": ["jumping"],
        },
        "templates": ["{}"],
    },
    "vtt_coniot": {
        "synonyms": {
            "walking_straight": ["walking_straight", "walking straight"],
            "walking_winding": ["walking_winding", "walking winding path"],
            "sitting": ["sitting"],
            "laying_back": ["laying_back", "lying on back"],
            "stairs": ["stairs", "using stairs"],
            "jumping_down": ["jumping_down", "jumping down"],
            "standing_work": ["standing_work", "standing while working"],
            "kneeling_work": ["kneeling_work", "kneeling while working"],
            "roll_painting": ["roll_painting", "painting with roller"],
            "spraying_paint": ["spraying_paint", "spray painting"],
            "leveling_paint": ["leveling_paint", "leveling paint"],
            "raising_hands": ["raising_hands", "raising hands"],
            "climbing_ladder": ["climbing_ladder", "climbing a ladder"],
            "carrying": ["carrying", "carrying object"],
            "lifting": ["lifting", "lifting object"],
            "pushing_cart": ["pushing_cart", "pushing a cart"],
        },
        "templates": ["{}"],
    },
    "recgym": {
        "synonyms": {
            "walking": ["walking"],
            "running": ["running"],
            "cycling": ["cycling"],
            "stairclimber": ["stairclimber", "stair climber machine"],
            "rope_skipping": ["rope_skipping", "jumping rope"],
            "squat": ["squat", "squatting"],
            "bench_press": ["bench_press", "bench pressing"],
            "arm_curl": ["arm_curl", "bicep curl"],
            "leg_curl": ["leg_curl", "leg curling"],
            "leg_press": ["leg_press", "leg pressing"],
            "adductor_machine": ["adductor_machine", "adductor exercise"],
        },
        "templates": ["{}"],
    },
}


def augment_label(
    label: str,
    dataset_name: str,
    augmentation_rate: float = 0.8,
    use_synonyms: bool = True,
    use_templates: bool = True,
) -> str:
    """
    Augment a single activity label with dataset-specific synonyms and templates.

    Args:
        label: Original activity label (e.g., "walking")
        dataset_name: Name of dataset (e.g., "uci_har", "mhealth", "pamap2", "wisdm")
        augmentation_rate: Probability of augmenting (0.0 to 1.0)
        use_synonyms: Whether to apply synonym replacement
        use_templates: Whether to apply template wrapping

    Returns:
        Augmented label text

    Examples:
        >>> augment_label("walking", "uci_har")
        "person strolling"  # synonym + template

        >>> augment_label("eating_pasta", "wisdm")
        "user consuming pasta"  # synonym + template

        >>> augment_label("cycling", "pamap2")
        "physical activity: biking"  # synonym + template
    """
    # No augmentation during validation or with probability (1 - augmentation_rate)
    if random.random() > augmentation_rate:
        return label

    # Get dataset-specific config
    if dataset_name not in DATASET_CONFIGS:
        # Fallback to generic template if dataset unknown
        if use_templates and random.random() < 0.5:
            return random.choice(["person {}", "{} activity", "human {}"]).format(label)
        return label

    config = DATASET_CONFIGS[dataset_name]
    synonyms = config["synonyms"]
    templates = config["templates"]

    # Step 1: Apply synonym (if available and enabled)
    if use_synonyms and label in synonyms:
        label = random.choice(synonyms[label])

    # Step 2: Apply template (if enabled)
    if use_templates:
        template = random.choice(templates)
        label = template.format(label)

    return label


def batch_augment_labels(
    labels: List[str],
    dataset_names: List[str],
    augmentation_rate: float = 0.8,
    use_synonyms: bool = True,
    use_templates: bool = True,
) -> List[str]:
    """
    Augment a batch of labels with dataset-specific augmentation.

    Args:
        labels: List of activity labels
        dataset_names: List of dataset names (parallel to labels)
        augmentation_rate: Probability of augmenting each label
        use_synonyms: Whether to apply synonyms
        use_templates: Whether to apply templates

    Returns:
        List of augmented labels
    """
    return [
        augment_label(label, dataset_name, augmentation_rate, use_synonyms, use_templates)
        for label, dataset_name in zip(labels, dataset_names)
    ]


def get_augmentation_stats(dataset_name: str) -> Dict[str, int]:
    """
    Get statistics about augmentation diversity for a dataset.

    Returns dict with:
        - num_activities: Number of unique base activities
        - num_synonyms: Total number of synonyms
        - num_templates: Number of templates
        - max_variations: Maximum possible variations per activity
    """
    if dataset_name not in DATASET_CONFIGS:
        return {"num_activities": 0, "num_synonyms": 0, "num_templates": 0, "max_variations": 0}

    config = DATASET_CONFIGS[dataset_name]
    synonyms = config["synonyms"]
    templates = config["templates"]

    num_activities = len(synonyms)
    num_synonyms = sum(len(syns) for syns in synonyms.values())
    num_templates = len(templates)
    max_variations = max(len(syns) for syns in synonyms.values()) * num_templates

    return {
        "num_activities": num_activities,
        "num_synonyms": num_synonyms,
        "num_templates": num_templates,
        "max_variations": max_variations,
    }


if __name__ == "__main__":
    # Test augmentation
    print("=" * 70)
    print("Dataset-Specific Label Augmentation Test")
    print("=" * 70)

    test_cases = [
        ("walking", "uci_har"),
        ("eating_pasta", "wisdm"),
        ("cycling", "pamap2"),
        ("jogging", "mhealth"),
        ("standing_up_from_sitting", "unimib_shar"),
        ("jogging", "motionsense"),
    ]

    for label, dataset in test_cases:
        print(f"\n{dataset.upper()}: '{label}'")
        print("Variations:")
        for i in range(5):
            augmented = augment_label(label, dataset, augmentation_rate=1.0)
            print(f"  {i+1}. {augmented}")

    print("\n" + "=" * 70)
    print("Augmentation Statistics per Dataset:")
    print("=" * 70)

    all_datasets = ["uci_har", "mhealth", "pamap2", "wisdm", "unimib_shar", "hhar", "motionsense"]
    for dataset_name in all_datasets:
        stats = get_augmentation_stats(dataset_name)
        print(f"\n{dataset_name.upper()}:")
        print(f"  Activities:      {stats['num_activities']}")
        print(f"  Total Synonyms:  {stats['num_synonyms']}")
        print(f"  Templates:       {stats['num_templates']}")
        print(f"  Max Variations:  {stats['max_variations']} per activity")
