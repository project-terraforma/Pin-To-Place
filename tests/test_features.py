from src.features import (
    categorize_place,
    get_tier,
    pin_ambiguity,
    place_complexity,
    should_move_rule,
)


def test_get_tier_standard_commercial():
    tier, label = get_tier("restaurant")
    assert tier == 1
    assert label == "standard_commercial"


def test_get_tier_no_building():
    tier, label = get_tier("lawyer")
    assert tier == 4
    assert label == "no_building"


def test_categorize_place_food():
    assert categorize_place("pizza_restaurant") == "food"


def test_place_complexity_complex_name():
    row = {
        "name": "Ocean View Resort",
        "category_primary": "hotel",
        "full_address": "1 Beach Road",
    }
    assert place_complexity(row) == "complex"


def test_place_complexity_multi_tenant_address():
    row = {
        "name": "Small Office",
        "category_primary": "professional_services",
        "full_address": "123 Main St Suite 200",
    }
    assert place_complexity(row) == "multi_tenant"


def test_pin_ambiguity_no_building_high():
    row = {
        "tier_label": "no_building",
        "category_primary": "lawyer",
        "name": "Law Office",
        "full_address": "123 Main St",
    }
    assert pin_ambiguity(row) == "high"


def test_should_move_rule_protects_no_building():
    row = {
        "tier_label": "no_building",
        "gt_confidence": 1.0,
        "offset_haversine_m": 50,
    }
    assert should_move_rule(row) is False


def test_should_move_rule_moves_high_confidence_offset():
    row = {
        "tier_label": "standard_commercial",
        "gt_confidence": 0.9,
        "offset_haversine_m": 30,
    }
    assert should_move_rule(row) is True
