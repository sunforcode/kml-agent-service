from datetime import datetime, timedelta

import pytest

from app.agents.segmentation_agent import SegmentationAgent
from app.core.track_processor import TrackPoint


def fragmented_days(fragment_counts):
    """Two points per recording fragment; gaps between fragments are not walking."""
    points = []
    segment_index = 0
    distance_km = 0.0
    for day, fragment_count in enumerate(fragment_counts):
        for fragment in range(fragment_count):
            start = datetime(2026, 7, 1, 8) + timedelta(days=day, hours=fragment)
            # A source boundary deliberately has a large elevation discontinuity.
            elevation = 100 + segment_index * 100
            for offset in range(2):
                if offset:
                    distance_km += 0.1
                points.append(TrackPoint(
                    latitude=30 + segment_index * 0.01,
                    longitude=120 + offset * 0.001,
                    elevation=elevation + offset * 10,
                    timestamp=start + timedelta(minutes=offset * 20),
                    distance_from_start=distance_km,
                    segment_index=segment_index,
                ))
            segment_index += 1
    return points


def test_fifteen_recording_fragments_form_six_days():
    # Reproduces the recording pattern of the reported task without private coordinates.
    counts = [3, 2, 2, 4, 3, 1]
    points = fragmented_days(counts)

    days = SegmentationAgent()._build_day_segments(points)

    assert len(days) == 6
    assert [day["name"] for day in days] == [f"第{i}天" for i in range(1, 7)]
    assert [(day["track_start_index"], day["track_end_index"]) for day in days] == [
        (0, 5), (6, 9), (10, 13), (14, 21), (22, 27), (28, 29),
    ]
    assert [day["distance"] for day in days] == pytest.approx([n * 0.1 for n in counts])
    assert [day["elevation_gain"] for day in days] == pytest.approx([n * 10 for n in counts])
    assert [day["elevation_loss"] for day in days] == [0] * 6


def test_recording_restarts_within_one_day_do_not_create_more_days():
    assert SegmentationAgent()._build_day_segments(fragmented_days([3])) == []


def test_day_statistics_exclude_distance_and_elevation_jumps_at_recording_boundaries():
    points = fragmented_days([2, 1])
    # Defend against a cumulative distance that includes the disconnected edge too.
    for point in points[2:]:
        point.distance_from_start += 50
    points[2].elevation = 50
    points[3].elevation = 40

    days = SegmentationAgent()._build_day_segments(points)

    assert len(days) == 2
    assert days[0]["distance"] == pytest.approx(0.2)
    assert days[0]["elevation_gain"] == pytest.approx(10)
    assert days[0]["elevation_loss"] == pytest.approx(10)


@pytest.mark.parametrize("gap_hours, expected_days", [(6, 0), (7, 2)])
def test_time_gap_still_splits_a_continuous_source(gap_hours, expected_days):
    points = fragmented_days([1, 1])
    points[2].timestamp = points[1].timestamp + timedelta(hours=gap_hours)
    points[3].timestamp = points[2].timestamp + timedelta(minutes=20)
    for point in points:
        point.segment_index = 0

    assert len(SegmentationAgent()._build_day_segments(points)) == expected_days


def test_recording_fragments_without_timestamps_do_not_imply_days():
    points = fragmented_days([3, 2])
    for point in points:
        point.timestamp = None

    assert SegmentationAgent()._build_day_segments(points) == []
