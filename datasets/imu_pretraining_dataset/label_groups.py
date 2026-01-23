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
    # Note: nordic_walking, walking_straight, walking_winding are VTT-ConIoT (zero-shot)
    # but walking exists in training, so these can be evaluated as synonyms
    # KU-HAR: walking_backwards (backwards locomotion, still walking pattern)
    'walking': ['walking', 'nordic_walking', 'walking_parking', 'walking_treadmill_flat',
                'walking_treadmill_incline', 'walking_straight', 'walking_winding',
                'walking_backwards'],

    # Stairs - ascending (upward motion pattern)
    'ascending_stairs': ['ascending_stairs', 'climbing_stairs', 'going_up_stairs',
                         'walking_upstairs', 'stairs_up'],

    # Stairs - descending (downward motion pattern)
    'descending_stairs': ['descending_stairs', 'going_down_stairs', 'walking_downstairs',
                          'stairs_down'],

    # Generic stairs (ambiguous direction - WISDM doesn't specify)
    'stairs': ['stairs'],

    # Running/jogging (fast locomotion)
    'running': ['running', 'jogging', 'running_treadmill'],

    # Lying/laying (horizontal posture) - includes transitions TO lying
    # HAPT: sit_to_lie, stand_to_lie
    'lying': ['lying', 'laying', 'lying_down_from_standing', 'lying_back',
              'lying_side', 'laying_back', 'sit_to_lie', 'stand_to_lie'],

    # Sitting (seated posture) - includes transitions TO sitting
    # HAPT: stand_to_sit
    # KU-HAR: talking_sitting (sitting while talking)
    'sitting': ['sitting', 'sitting_down', 'sitting_down_from_standing', 'sitting_chair',
                'stand_to_sit', 'talking_sitting'],

    # Standing (upright posture) - includes transitions TO standing
    # HAPT: sit_to_stand, lie_to_stand, lie_to_sit (getting up from lying is standing action)
    # KU-HAR: talking_standing (standing while talking)
    'standing': ['standing', 'standing_up_from_laying', 'standing_up_from_sitting',
                 'standing_elevator', 'standing_work', 'sit_to_stand', 'lie_to_stand',
                 'lie_to_sit', 'talking_standing'],

    # Falling variants (loss of balance - all share sudden acceleration patterns)
    # MobiFall (training) provides fall types
    'falling': ['falling_backward', 'falling_backward_sitting', 'falling_forward',
                'falling_hitting_obstacle', 'falling_left', 'falling_right',
                'falling_with_protection', 'syncope',
                # MobiFall fall types
                'fall_forward', 'fall_backward_knees', 'fall_backward_sitting', 'fall_sideways'],

    # Jumping variants (vertical explosive motion)
    # Note: rope_jumping/rope_skipping are RecGym (zero-shot) but jumping is in training
    'jumping': ['jumping', 'jump_front_back', 'rope_jumping', 'rope_skipping', 'jumping_down'],

    # Eating/drinking (hand-to-mouth repetitive motion)
    'eating': ['eating_chips', 'eating_pasta', 'eating_sandwich', 'eating_soup', 'drinking'],

    # Cycling variants (pedaling motion)
    # DSADS has cycling_horizontal, cycling_vertical
    'cycling': ['cycling', 'cycling_horizontal', 'cycling_vertical'],

    # Cardio exercise machines (rhythmic full-body motion)
    'cardio_machine': ['exercising_stepper', 'exercising_cross_trainer', 'rowing', 'stairclimber'],

    # Vehicle entry/exit (stepping motion with support)
    'vehicle_entry': ['car_step_in', 'car_step_out'],

    # Elevator (stationary or subtle motion)
    'elevator': ['moving_elevator'],

    # Ball sports (throwing, catching, dribbling patterns)
    'sports': ['playing_basketball', 'playing_catch', 'dribbling', 'kicking'],

    # Hand/arm activities - fine motor movements (WISDM, MHEALTH activities)
    'typing': ['typing', 'writing'],  # Fine motor, similar wrist patterns
    'grooming': ['brushing_teeth'],
    'clapping': ['clapping'],

    # Household chores (PAMAP2, WISDM activities)
    'ironing': ['ironing'],
    'vacuum_cleaning': ['vacuum_cleaning'],
    'folding_clothes': ['folding_clothes'],

    # Exercise/stretching movements (MHEALTH activities)
    'arm_exercise': ['frontal_elevation_arms', 'raising_hands'],
    'leg_exercise': ['knees_bending'],
    'torso_exercise': ['waist_bends_forward'],

    # KU-HAR exercises (floor/body-weight exercises)
    'push_up': ['push_up'],
    'sit_up': ['sit_up'],
    'picking_up': ['picking_up'],  # Bending to pick something up

    # Ball sports / general sports (KU-HAR: table-tennis maps to playing_sports)
    'playing_sports': ['playing_sports'],

    # Construction/manual work (VTT-ConIoT activities)
    'carrying': ['carrying', 'lifting', 'pushing_cart'],
    'climbing': ['climbing_ladder'],
    'kneeling': ['kneeling_work'],
    'painting': ['roll_painting', 'spraying_paint', 'leveling_paint'],

    # Gym weight exercises (RecGym activities)
    'gym_weights': ['squat', 'bench_press', 'leg_press', 'leg_curl', 'arm_curl', 'adductor_machine'],
}

# LABEL_GROUPS_SIMPLE: Simplified grouping (~10 effective groups)
# For coarser evaluation - groups semantically similar activities
# Use when fine-grained distinctions aren't important or data is limited
LABEL_GROUPS_SIMPLE = {
    # Locomotion - walking pace (includes backwards walking)
    'walking': ['walking', 'nordic_walking', 'walking_parking', 'walking_treadmill_flat',
                'walking_treadmill_incline', 'walking_straight', 'walking_winding',
                'walking_backwards'],

    # Locomotion - running pace
    'running': ['running', 'jogging', 'running_treadmill'],

    # Locomotion - stairs (any direction, model may not reliably distinguish)
    # Note: climbing_ladder and stairclimber removed (zero-shot only)
    'stairs': ['stairs', 'ascending_stairs', 'climbing_stairs', 'going_up_stairs', 'walking_upstairs',
               'descending_stairs', 'going_down_stairs', 'walking_downstairs',
               'stairs_up', 'stairs_down'],

    # Locomotion - cycling
    'cycling': ['cycling', 'cycling_horizontal', 'cycling_vertical'],

    # Stationary - sitting (any seated position or transition to it)
    # HAPT: stand_to_sit; KU-HAR: talking_sitting
    'sitting': ['sitting', 'sitting_down', 'sitting_down_from_standing', 'sitting_chair',
                'stand_to_sit', 'talking_sitting'],

    # Stationary - standing (any upright position or transition to it)
    # HAPT: sit_to_stand, lie_to_stand, lie_to_sit; KU-HAR: talking_standing
    'standing': ['standing', 'standing_up_from_laying', 'standing_up_from_sitting',
                 'standing_elevator', 'moving_elevator', 'standing_work',
                 'sit_to_stand', 'lie_to_stand', 'lie_to_sit', 'talking_standing'],

    # Stationary - lying (any horizontal position or transition to it)
    # HAPT: sit_to_lie, stand_to_lie
    'lying': ['lying', 'laying', 'lying_down_from_standing', 'lying_back', 'lying_side', 'laying_back',
              'sit_to_lie', 'stand_to_lie'],

    # Dynamic exercise (jumping, body-weight exercises, cardio machines)
    # KU-HAR: push_up, sit_up, picking_up
    'exercise': ['jumping', 'jump_front_back', 'rope_jumping', 'rope_skipping', 'jumping_down',
                 'knees_bending', 'waist_bends_forward', 'frontal_elevation_arms', 'raising_hands',
                 'exercising_stepper', 'exercising_cross_trainer', 'rowing', 'stairclimber', 'climbing_ladder',
                 'squat', 'bench_press', 'leg_press', 'leg_curl', 'arm_curl', 'adductor_machine',
                 'push_up', 'sit_up', 'picking_up'],

    # Hand/arm activities (fine motor, repetitive arm movements)
    'hand_activity': ['eating_chips', 'eating_pasta', 'eating_sandwich', 'eating_soup',
                      'drinking', 'brushing_teeth', 'typing', 'writing', 'folding_clothes', 'clapping'],

    # Manual work / construction (VTT-ConIoT activities)
    'manual_work': ['carrying', 'lifting', 'pushing_cart', 'kneeling_work',
                    'roll_painting', 'spraying_paint', 'leveling_paint'],

    # Vehicle interaction
    'vehicle': ['car_step_in', 'car_step_out'],

    # Household chores (varied whole-body movements)
    'household': ['ironing', 'vacuum_cleaning'],

    # Ball sports (throwing, catching, kicking patterns)
    # KU-HAR: playing_sports (table tennis)
    'sports': ['playing_catch', 'dribbling', 'kicking', 'playing_basketball', 'playing_sports'],

    # Falling (loss of balance - grouped for safety applications)
    'falling': ['falling_backward', 'falling_backward_sitting', 'falling_forward',
                'falling_hitting_obstacle', 'falling_left', 'falling_right',
                'falling_with_protection', 'syncope',
                'fall_forward', 'fall_backward_knees', 'fall_backward_sitting', 'fall_sideways'],
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
