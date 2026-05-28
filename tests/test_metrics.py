import pandas as pd

from src.metrics import (
    arrival_cost_score,
    haversine_meters,
    task_aware_report,
)


def test_haversine_same_point_is_zero():
    assert haversine_meters(40.0, -70.0, 40.0, -70.0) == 0


def test_haversine_known_small_distance():
    dist = haversine_meters(40.0, -70.0, 40.001, -70.0)
    assert 110 <= dist <= 112


def test_arrival_cost_score_adds_penalties():
    score = arrival_cost_score(
        10,
        sidewalk_visible=False,
        parking_lot_crossing=True,
        barrier_detected=True,
    )
    assert score == 85


def test_task_aware_report():
    df = pd.DataFrame({"offset_haversine_m": [0, 10, 20, 30, 40]})
    report = task_aware_report(df)

    assert report["count"] == 5
    assert report["median_m"] == 20
    assert report["pct_exact_no_move"] == 20.0
    assert report["pct_over_25m"] == 40.0
