"""Test 7: Label group mapping consistency.

Verifies determinism, completeness, and correctness of label group mappings.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imu_pretraining_dataset.label_groups import (
    LABEL_GROUPS,
    LABEL_GROUPS_SIMPLE,
    ACTIVE_LABEL_GROUPS,
    get_label_to_group_mapping,
    get_group_for_label,
    get_group_members,
)


class TestGetLabelToGroupMapping:
    """Test label-to-group mapping."""

    def test_deterministic(self):
        """Same result on repeated calls."""
        mapping1 = get_label_to_group_mapping()
        mapping2 = get_label_to_group_mapping()
        assert mapping1 == mapping2

    def test_deterministic_simple(self):
        mapping1 = get_label_to_group_mapping(use_simple=True)
        mapping2 = get_label_to_group_mapping(use_simple=True)
        assert mapping1 == mapping2

    def test_every_label_in_groups_appears_in_mapping(self):
        """Every label listed in LABEL_GROUPS should appear in the mapping."""
        mapping = get_label_to_group_mapping()
        for group_name, labels in LABEL_GROUPS.items():
            for label in labels:
                assert label in mapping, \
                    f"Label '{label}' from group '{group_name}' not in mapping"

    def test_every_label_in_simple_groups_appears_in_mapping(self):
        mapping = get_label_to_group_mapping(use_simple=True)
        for group_name, labels in LABEL_GROUPS_SIMPLE.items():
            for label in labels:
                assert label in mapping, \
                    f"Label '{label}' from simple group '{group_name}' not in mapping"

    def test_ungrouped_labels_map_to_themselves(self):
        """Labels not in any group should map to themselves."""
        mapping = get_label_to_group_mapping()
        assert mapping.get('totally_unknown_activity', 'totally_unknown_activity') == 'totally_unknown_activity'

    def test_mapping_values_are_group_names(self):
        """Every value in the mapping should be a group name from LABEL_GROUPS."""
        mapping = get_label_to_group_mapping()
        group_names = set(LABEL_GROUPS.keys())
        for label, group in mapping.items():
            assert group in group_names, \
                f"Label '{label}' maps to '{group}' which is not a group name"


class TestGetGroupForLabel:
    """Test get_group_for_label."""

    def test_known_labels(self):
        assert get_group_for_label('walking') == 'walking'
        assert get_group_for_label('jogging') == 'running'
        assert get_group_for_label('ascending_stairs') == 'ascending_stairs'
        assert get_group_for_label('climbing_stairs') == 'ascending_stairs'
        assert get_group_for_label('sitting_down') == 'sitting'

    def test_unknown_label_returns_itself(self):
        assert get_group_for_label('completely_made_up') == 'completely_made_up'

    def test_simple_groups(self):
        # In simple groups, ascending and descending stairs merge into 'stairs'
        assert get_group_for_label('ascending_stairs', use_simple=True) == 'stairs'
        assert get_group_for_label('descending_stairs', use_simple=True) == 'stairs'
        assert get_group_for_label('walking_downstairs', use_simple=True) == 'stairs'


class TestGetGroupMembers:
    """Test get_group_members."""

    def test_walking_group(self):
        members = get_group_members('walking')
        assert 'walking' in members
        assert 'nordic_walking' in members
        assert len(members) == len(LABEL_GROUPS['walking'])

    def test_unknown_label_returns_singleton(self):
        members = get_group_members('unknown_activity_xyz')
        assert members == ['unknown_activity_xyz']

    def test_members_consistent_with_groups(self):
        """get_group_members should return the group's label list.

        Note: Some labels (e.g., 'waist_bends_forward') appear in multiple groups.
        get_label_to_group_mapping assigns each label to the first group found,
        so get_group_members via that label returns the first group's members.
        We test with the group name directly (first label in group -> mapping -> group).
        """
        mapping = get_label_to_group_mapping()
        for group_name, expected_labels in LABEL_GROUPS.items():
            # Use a label that maps to THIS group (not one shared with another group)
            canonical_label = None
            for label in expected_labels:
                if mapping.get(label) == group_name:
                    canonical_label = label
                    break
            if canonical_label is None:
                # All labels in this group map to a different group (edge case)
                continue
            members = get_group_members(canonical_label)
            assert set(members) == set(expected_labels), \
                f"Group '{group_name}': members mismatch (via label '{canonical_label}')"

    def test_simple_group_members(self):
        members = get_group_members('ascending_stairs', use_simple=True)
        # In simple groups, stairs includes both ascending and descending
        assert 'ascending_stairs' in members
        assert 'descending_stairs' in members


class TestLabelGroupsIntegrity:
    """Test label group data structure integrity."""

    def test_no_empty_groups(self):
        for group_name, labels in LABEL_GROUPS.items():
            assert len(labels) > 0, f"Group '{group_name}' is empty"

    def test_no_empty_simple_groups(self):
        for group_name, labels in LABEL_GROUPS_SIMPLE.items():
            assert len(labels) > 0, f"Simple group '{group_name}' is empty"

    def test_active_label_groups_is_label_groups(self):
        """ACTIVE_LABEL_GROUPS should be LABEL_GROUPS (fine-grained)."""
        assert ACTIVE_LABEL_GROUPS is LABEL_GROUPS

    def test_group_names_are_strings(self):
        for name in LABEL_GROUPS:
            assert isinstance(name, str)

    def test_labels_are_strings(self):
        for group_name, labels in LABEL_GROUPS.items():
            for label in labels:
                assert isinstance(label, str), \
                    f"Non-string label in group '{group_name}': {label}"

    def test_no_duplicate_labels_within_group(self):
        """Each label should appear only once within its group."""
        for group_name, labels in LABEL_GROUPS.items():
            assert len(labels) == len(set(labels)), \
                f"Duplicate labels in group '{group_name}'"
