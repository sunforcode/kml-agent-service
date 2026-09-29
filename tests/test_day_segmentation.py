from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta

import pytest

from app.agents.segmentation_agent import SegmentationAgent
from app.core.track_processor import TrackPoint, TrackProcessor
from app.models.response import SegmentSchemeOutput


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


@pytest.mark.parametrize("config", [None, {
    "smooth_window": 5, "smooth_polyorder": 2, "elevation_threshold_m": 10,
}])
def test_daily_elevation_reuses_the_shared_filter_and_does_not_mutate_points(config):
    points = []
    for day in range(2):
        for offset in range(61):
            points.append(TrackPoint(
                latitude=30 + day * 0.01,
                longitude=120 + offset * 0.0001,
                elevation=100 + day * 500 + offset * 0.5 + (2 if offset % 2 else -2),
                timestamp=datetime(2026, 7, 1 + day, 8) + timedelta(minutes=offset),
                distance_from_start=day * 0.6 + offset * 0.01,
                segment_index=day,
            ))
    original = deepcopy(points)
    agent = SegmentationAgent(TrackProcessor(config)) if config else SegmentationAgent()

    days = agent._build_day_segments(points)

    for day in days:
        selected = points[day["track_start_index"]:day["track_end_index"] + 1]
        expected = agent.processor.calculate_elevation_stats(selected)
        assert day["elevation_gain"] == expected.total_gain_m
        assert day["elevation_loss"] == expected.total_loss_m
        assert day["elevation_gain"] < 40
    whole = agent.processor.calculate_elevation_stats(points)
    assert sum(day["elevation_gain"] for day in days) == pytest.approx(whole.total_gain_m, abs=0.1)
    assert sum(day["elevation_loss"] for day in days) == pytest.approx(whole.total_loss_m, abs=0.1)
    assert points == original


def test_daily_estimated_time_sums_existing_continuous_segment_estimates():
    points = fragmented_days([2, 1])
    agent = SegmentationAgent()
    segments = agent._build_slope_segments(points, agent.default_strategy)
    estimates = [agent._enrich_segment(segment, points, i, len(segments))["estimated_time"]
                 for i, segment in enumerate(segments)]

    days = agent._build_day_segments(points)

    assert estimates == [2, 2, 2]
    assert [day["estimated_time"] for day in days] == [sum(estimates[:2]), estimates[2]]
    assert [day["difficulty"] for day in days] == [1, 1]


def test_estimated_time_does_not_use_recording_elapsed_time():
    points = fragmented_days([1, 1])
    with_breaks = deepcopy(points)
    with_breaks[1].timestamp = with_breaks[0].timestamp + timedelta(hours=5)
    with_breaks[3].timestamp = with_breaks[2].timestamp + timedelta(hours=4)
    agent = SegmentationAgent()

    original = agent._build_day_segments(points)
    extended = agent._build_day_segments(with_breaks)

    assert [day["estimated_time"] for day in original] == [2, 2]
    assert [day["estimated_time"] for day in extended] == [2, 2]


@pytest.mark.asyncio
async def test_daily_estimate_clips_a_slope_segment_spanning_two_days():
    points = fragmented_days([1, 1])
    for point in points:
        point.segment_index = 0
    points[2].distance_from_start = 1.1
    points[3].distance_from_start = 1.2
    points[2].elevation = 400
    points[3].elevation = 410
    serialized = [dict(asdict(p), distance_from_start=p.distance_from_start * 1000) for p in points]
    original = deepcopy(serialized)
    agent = SegmentationAgent()
    slope = agent._build_slope_segments(points, agent.default_strategy)
    assert len(slope) == 1
    expected_slope = [agent._enrich_segment(slope[0], points, 0, 1)]

    result = await agent.execute({"request": {"segmentation_strategy": "slope"}, "track_points": serialized})

    schemes = result["segment_schemes"]
    days = next(s["segments"] for s in schemes if s["scheme_type"] == "day")
    assert [(d["track_start_index"], d["track_end_index"]) for d in days] == [(0, 1), (2, 3)]
    assert [d["estimated_time"] for d in days] == [2, 2]
    assert [d["elevation_gain"] for d in days] == [10, 10]
    assert schemes[0]["segments"] == expected_slope
    assert serialized == original
    for scheme in schemes:
        SegmentSchemeOutput.model_validate(scheme)


@pytest.mark.parametrize("elevations", [(None, None), (None, 110), (100, None)])
def test_daily_estimate_with_missing_elevation_uses_only_distance(elevations):
    points = fragmented_days([1, 1])
    for index, point in enumerate(points):
        point.elevation = elevations[index % 2]

    days = SegmentationAgent()._build_day_segments(points)

    assert [day["estimated_time"] for day in days] == [1, 1]
    assert [day["elevation_gain"] for day in days] == [0, 0]
    assert [day["elevation_loss"] for day in days] == [0, 0]


def test_stationary_day_does_not_invent_one_minute_of_walking():
    points = fragmented_days([1, 1])
    points[1].distance_from_start = points[0].distance_from_start
    points[1].latitude = points[0].latitude
    points[1].longitude = points[0].longitude

    days = SegmentationAgent()._build_day_segments(points)

    assert [day["estimated_time"] for day in days] == [0, 2]


def test_day_with_only_isolated_recordings_does_not_count_the_gap_as_walking():
    source = fragmented_days([2, 1])
    points = [source[0], source[2], *source[4:]]
    points[1].distance_from_start = 50
    for point in points[2:]:
        point.distance_from_start += 50

    days = SegmentationAgent()._build_day_segments(points)

    assert days[0]["distance"] == 0
    assert days[0]["elevation_gain"] == 0
    assert [day["estimated_time"] for day in days] == [0, 2]


@pytest.mark.asyncio
async def test_cross_day_estimate_with_missing_slice_endpoint_keeps_existing_slope_output():
    points = fragmented_days([1, 1])
    for index, point in enumerate(points):
        point.segment_index = 0
        point.distance_from_start = index * 0.1
    points[1].elevation = None
    agent = SegmentationAgent()
    slope = agent._build_slope_segments(points, agent.default_strategy)
    expected_slope = [agent._enrich_segment(slope[0], points, 0, 1)]
    serialized = [dict(asdict(p), distance_from_start=p.distance_from_start * 1000) for p in points]

    result = await agent.execute({"request": {}, "track_points": serialized})

    assert result["segment_schemes"][0]["segments"] == expected_slope
    days = next(s["segments"] for s in result["segment_schemes"] if s["scheme_type"] == "day")
    assert [day["estimated_time"] for day in days] == [1, 2]


@pytest.mark.asyncio
async def test_short_day_not_covered_by_auto_slope_segments_still_has_an_estimate():
    points = []
    for index in range(12):
        day = 0 if index < 2 else 1
        offset = index if day == 0 else index - 2
        points.append(TrackPoint(
            latitude=30,
            longitude=120 + index * 0.001,
            elevation=100 + index * 10,
            timestamp=datetime(2026, 7, 1 + day, 8) + timedelta(minutes=offset * 20),
            distance_from_start=index * 0.1,
            segment_index=0,
        ))
    agent = SegmentationAgent()
    slope = agent._build_slope_segments(points, agent.default_strategy)
    assert [(s.start_index, s.end_index) for s in slope] == [(2, 11)]
    expected_slope = [agent._enrich_segment(slope[0], points, 0, 1)]
    serialized = [dict(asdict(p), distance_from_start=p.distance_from_start * 1000) for p in points]

    result = await agent.execute({"request": {}, "track_points": serialized})

    days = next(s["segments"] for s in result["segment_schemes"] if s["scheme_type"] == "day")
    assert [day["estimated_time"] for day in days] == [2, expected_slope[0]["estimated_time"]]
    assert result["segment_schemes"][0]["segments"] == expected_slope


def test_partial_slope_coverage_estimates_uncovered_edges_once():
    points = [TrackPoint(
        latitude=30, longitude=120 + index * 0.001, elevation=100,
        timestamp=datetime(2026, 7, 1, 8) + timedelta(minutes=index * 20),
        distance_from_start=index * 0.1, segment_index=0,
    ) for index in range(5)]
    points.extend([TrackPoint(
        latitude=31, longitude=121 + index * 0.001, elevation=200,
        timestamp=datetime(2026, 7, 2, 8) + timedelta(minutes=index * 20),
        distance_from_start=0.4 + index * 0.1, segment_index=1,
    ) for index in range(2)])
    agent = SegmentationAgent()
    middle = agent.processor._create_segment(points[:5], 1, 3)
    second_day = agent.processor._create_segment(points[5:], 0, 1)
    second_day.start_index += 5
    second_day.end_index += 5

    days = agent._build_day_segments(points, [middle, second_day])

    # 0->1 and 3->4 are uncovered; 1->3 must not be counted twice.
    assert [day["estimated_time"] for day in days] == [4, 1]
