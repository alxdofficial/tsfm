"""
Label groups for synonym handling in semantic alignment.

Shared between training (for class balancing) and evaluation (for group-aware metrics).
"""

from typing import Dict, List


# =============================================================================
# Label Groups - Synonyms that should be treated as equivalent
# =============================================================================

# LABEL_GROUPS: Fine-grained grouping (~25 effective groups)
# Groups together only true synonyms (same activity, different naming)
# Keeps semantically different activities separate (e.g., ascending vs descending stairs)
LABEL_GROUPS = {
    # Walking variants (same gait pattern)
    'walking': ['walking', 'nordic_walking'],

    # Stairs - ascending (upward motion pattern)
    'ascending_stairs': ['ascending_stairs', 'climbing_stairs', 'going_up_stairs', 'walking_upstairs'],

    # Stairs - descending (downward motion pattern)
    'descending_stairs': ['descending_stairs', 'going_down_stairs', 'walking_downstairs'],

    # Generic stairs (ambiguous direction - WISDM doesn't specify)
    'stairs': ['stairs'],

    # Running/jogging (fast locomotion)
    'running': ['running', 'jogging'],

    # Lying/laying (horizontal posture) - includes transition TO lying
    'lying': ['lying', 'laying', 'lying_down_from_standing'],

    # Sitting (seated posture) - includes transition TO sitting
    'sitting': ['sitting', 'sitting_down', 'sitting_down_from_standing'],

    # Standing (upright posture) - includes transitions TO standing
    'standing': ['standing', 'standing_up_from_laying', 'standing_up_from_sitting'],

    # Falling variants (loss of balance - all share sudden acceleration patterns)
    'falling': ['falling_backward', 'falling_backward_sitting', 'falling_forward',
                'falling_hitting_obstacle', 'falling_left', 'falling_right',
                'falling_with_protection', 'syncope'],

    # Jumping variants (vertical explosive motion)
    'jumping': ['jumping', 'jump_front_back', 'rope_jumping'],

    # Eating variants (hand-to-mouth repetitive motion)
    'eating': ['eating_chips', 'eating_pasta', 'eating_sandwich', 'eating_soup'],
}

# LABEL_GROUPS_SIMPLE: Simplified grouping (~10 effective groups)
# For coarser evaluation - groups semantically similar activities
# Use when fine-grained distinctions aren't important or data is limited
LABEL_GROUPS_SIMPLE = {
    # Locomotion - walking pace
    'walking': ['walking', 'nordic_walking'],

    # Locomotion - running pace
    'running': ['running', 'jogging'],

    # Locomotion - stairs (any direction, model may not reliably distinguish)
    'stairs': ['stairs', 'ascending_stairs', 'climbing_stairs', 'going_up_stairs', 'walking_upstairs',
               'descending_stairs', 'going_down_stairs', 'walking_downstairs'],

    # Locomotion - cycling
    'cycling': ['cycling'],

    # Stationary - sitting (any seated position or transition to it)
    'sitting': ['sitting', 'sitting_down', 'sitting_down_from_standing'],

    # Stationary - standing (any upright position or transition to it)
    'standing': ['standing', 'standing_up_from_laying', 'standing_up_from_sitting'],

    # Stationary - lying (any horizontal position or transition to it)
    'lying': ['lying', 'laying', 'lying_down_from_standing'],

    # Dynamic exercise (jumping, body-weight exercises)
    'exercise': ['jumping', 'jump_front_back', 'rope_jumping',
                 'knees_bending', 'waist_bends_forward', 'frontal_elevation_arms'],

    # Hand/arm activities (fine motor, repetitive arm movements)
    'hand_activity': ['eating_chips', 'eating_pasta', 'eating_sandwich', 'eating_soup',
                      'drinking', 'brushing_teeth', 'typing', 'writing', 'folding_clothes', 'clapping'],

    # Household chores (varied whole-body movements)
    'household': ['ironing', 'vacuum_cleaning'],

    # Ball sports (throwing, catching, kicking patterns)
    'sports': ['playing_catch', 'dribbling', 'kicking'],

    # Falling (loss of balance - grouped for safety applications)
    'falling': ['falling_backward', 'falling_backward_sitting', 'falling_forward',
                'falling_hitting_obstacle', 'falling_left', 'falling_right',
                'falling_with_protection', 'syncope'],
}

# Active label groups (change this to switch between groupings)
ACTIVE_LABEL_GROUPS = LABEL_GROUPS  # Use LABEL_GROUPS_SIMPLE for coarser evaluation


def get_label_to_group_mapping(use_simple: bool = False) -> Dict[str, str]:
    """
    Create reverse mapping from individual labels to their group name.
    Labels not in any group map to themselves.

    Args:
        use_simple: If True, use LABEL_GROUPS_SIMPLE instead of LABEL_GROUPS
    """
    label_to_group = {}
    groups = LABEL_GROUPS_SIMPLE if use_simple else LABEL_GROUPS

    # Map grouped labels to their group name
    for group_name, labels in groups.items():
        for label in labels:
            # Note: some labels like 'stairs' may belong to multiple groups
            # We take the first assignment (ascending_stairs comes before descending_stairs)
            if label not in label_to_group:
                label_to_group[label] = group_name

    return label_to_group


def get_group_for_label(label: str, use_simple: bool = False) -> str:
    """
    Get the group name for a given label.
    Returns the label itself if not in any group (singleton group).

    Args:
        label: The activity label
        use_simple: If True, use LABEL_GROUPS_SIMPLE

    Returns:
        Group name or the label itself
    """
    mapping = get_label_to_group_mapping(use_simple)
    return mapping.get(label, label)


def get_group_members(label: str, use_simple: bool = False) -> List[str]:
    """
    Get all labels that are synonyms of the given label.
    Returns list including the label itself.

    Args:
        label: The label to find synonyms for
        use_simple: If True, use LABEL_GROUPS_SIMPLE instead of LABEL_GROUPS
    """
    mapping = get_label_to_group_mapping(use_simple)
    group = mapping.get(label, label)
    groups = LABEL_GROUPS_SIMPLE if use_simple else LABEL_GROUPS

    if group in groups:
        return groups[group]
    else:
        return [label]  # Ungrouped label - only matches itself
