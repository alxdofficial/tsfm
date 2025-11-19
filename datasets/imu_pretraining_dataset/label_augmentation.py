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

    for dataset_name in ["uci_har", "mhealth", "pamap2", "wisdm"]:
        stats = get_augmentation_stats(dataset_name)
        print(f"\n{dataset_name.upper()}:")
        print(f"  Activities:      {stats['num_activities']}")
        print(f"  Total Synonyms:  {stats['num_synonyms']}")
        print(f"  Templates:       {stats['num_templates']}")
        print(f"  Max Variations:  {stats['max_variations']} per activity")
