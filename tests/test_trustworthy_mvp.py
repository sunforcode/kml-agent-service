import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.agents.analysis_workflow import AnalysisWorkflow
from app.agents.content_agent import ContentAgent
from app.agents.quality_agent import QualityAgent
from app.agents.segment_merge_agent import SegmentMergeAgent
from app.agents.segmentation_agent import SegmentationAgent
from app.agents.trajectory_agent import TrajectoryAgent
from app.models.response import EnhancedRouteOutput
from app.core.track_processor import TrackProcessor
from app.models.state import AgentState
from app.services.callback_service import CallbackService
from geopy.distance import geodesic
from langgraph.graph import END, StateGraph


def valid_segment():
    return {
        "id": "slope_seg_001",
        "name": "路段1",
        "sequence_number": 1,
        "color": "#FF5722",
        "distance": 1.2,
        "elevation_gain": 100.0,
        "elevation_loss": 20.0,
        "estimated_time": 30,
        "difficulty": 2,
        "scheme_type": "slope",
    }


def valid_result():
    return {
        "source_kml_url": "inline.kml",
        "analysis_timestamp": "2026-01-01T00:00:00",
        "quality_score": 80,
        "total_distance_km": 1.2,
        "total_elevation_gain_m": 100,
        "total_elevation_loss_m": 20,
        "max_elevation": 300,
        "min_elevation": 200,
        "is_loop": False,
        "estimated_difficulty": 2,
        "segment_schemes": [
            {
                "scheme_type": "slope",
                "label": "按坡度",
                "is_default": True,
                "segments": [valid_segment()],
            }
        ],
        "poi_points": [
            {
                "category": "start",
                "name": "起点",
                "latitude": 30.0,
                "longitude": 120.0,
                "source": "kml_marker",
                "confidence": 1.0,
            }
        ],
    }


def base_state():
    return {
        "request": {"kml_source": "inline.kml", "enable_content_generation": True},
        "track_points": [
            {"latitude": 30.0, "longitude": 120.0, "elevation": 100, "timestamp": "2026-01-01T00:00:00"},
            {"latitude": 30.01, "longitude": 120.01, "elevation": 120, "timestamp": "2026-01-01T00:10:00"},
        ],
        "basic_stats": {
            "total_distance_km": 1.5,
            "total_gain_m": 20,
            "total_loss_m": 0,
            "max_elevation": 120,
            "min_elevation": 100,
            "is_loop": False,
            "valid_elevation_points": 2,
            "valid_timestamp_points": 2,
        },
        "segment_schemes": [
            {"scheme_type": "slope", "label": "按坡度", "is_default": True, "segments": [valid_segment()]}
        ],
        "poi_points": [],
        "errors": [],
        "warnings": [],
    }


def test_final_result_uses_new_segment_and_poi_contract():
    model = EnhancedRouteOutput.model_validate(valid_result())
    dumped = model.model_dump(mode="json")
    assert dumped["segment_schemes"][0]["segments"][0]["scheme_type"] == "slope"
    assert dumped["poi_points"][0]["source"] == "kml_marker"
    assert "segments" not in dumped
    assert "water_sources" not in dumped


def test_final_result_rejects_invalid_poi_confidence():
    result = valid_result()
    result["poi_points"][0]["confidence"] = 1.5
    with pytest.raises(ValidationError):
        EnhancedRouteOutput.model_validate(result)


@pytest.mark.asyncio
async def test_invalid_trajectory_is_fatal_and_workflow_without_result_raises(monkeypatch):
    agent = TrajectoryAgent(processor=SimpleNamespace(parse_kml_or_gpx=lambda *_args, **_kwargs: []))
    update = await agent.execute({"request": {"kml_content": "<kml></kml>"}})
    assert update["errors"]
    assert "FATAL" in update["errors"][0]

    class FailedCompiled:
        async def astream(self, _state, stream_mode="values"):
            assert stream_mode == "values"
            yield update

    workflow = AnalysisWorkflow()
    workflow._compiled = FailedCompiled()
    with pytest.raises(RuntimeError, match="FATAL"):
        await workflow.execute_workflow({"kml_content": "<kml></kml>", "kml_source": "inline.kml"})


@pytest.mark.asyncio
async def test_content_agent_parses_structured_llm_json(monkeypatch):
    response = SimpleNamespace(content=json.dumps({
        "description": "基于轨迹统计生成的描述",
        "highlights": ["累计爬升 20 米"],
        "difficulties": [],
        "safety_notes": ["请结合现场情况判断"],
        "equipment_recommendations": [],
    }, ensure_ascii=False))
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
    monkeypatch.setattr("app.agents.content_agent.settings.openai_api_key", "test-key")
    monkeypatch.setattr("app.agents.content_agent.get_llm", lambda: llm)

    update = await ContentAgent().execute(base_state())

    assert update["generated_content"]["description"] == "基于轨迹统计生成的描述"
    assert update["generated_content"]["generation_mode"] == "llm"
    assert "warnings" not in update


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_key", "invalid_json", "call_error"])
async def test_content_agent_falls_back_without_invented_facts(monkeypatch, failure):
    if failure == "missing_key":
        monkeypatch.setattr("app.agents.content_agent.settings.openai_api_key", None)
    else:
        monkeypatch.setattr("app.agents.content_agent.settings.openai_api_key", "test-key")
        response = SimpleNamespace(content="not-json")
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        if failure == "call_error":
            llm.ainvoke = AsyncMock(side_effect=RuntimeError("offline"))
        monkeypatch.setattr("app.agents.content_agent.get_llm", lambda: llm)

    update = await ContentAgent().execute(base_state())
    serialized = json.dumps(update, ensure_ascii=False)

    assert update["generated_content"]["generation_mode"] == "fallback"
    assert "1.5" in update["generated_content"]["description"]
    assert update["warnings"]
    assert "五台山" not in serialized
    assert "prompt_tokens" not in serialized
    assert "external_api_calls" not in serialized


@pytest.mark.asyncio
async def test_segment_merge_execute_uses_llm_result(monkeypatch):
    state = base_state()
    state["segment_schemes"][0]["segments"] = [valid_segment(), {**valid_segment(), "id": "slope_seg_002", "sequence_number": 2}]
    merged = [{**valid_segment(), "distance": 2.4}]
    agent = SegmentMergeAgent()
    call = AsyncMock(return_value=merged)
    monkeypatch.setattr(agent, "_llm_merge", call)

    update = await agent.execute(state)

    call.assert_awaited_once()
    assert update["segment_schemes"][0]["segments"] == merged


@pytest.mark.asyncio
async def test_segment_merge_failure_preserves_segments_and_warns(monkeypatch):
    state = base_state()
    original = [valid_segment(), {**valid_segment(), "id": "slope_seg_002", "sequence_number": 2}]
    state["segment_schemes"][0]["segments"] = original
    agent = SegmentMergeAgent()
    monkeypatch.setattr(agent, "_llm_merge", AsyncMock(side_effect=RuntimeError("offline")))

    update = await agent.execute(state)

    assert update["segment_schemes"][0]["segments"] == original
    assert update["warnings"]


@pytest.mark.asyncio
async def test_segment_merge_missing_key_preserves_segments_and_warns(monkeypatch):
    state = base_state()
    original = [valid_segment(), {**valid_segment(), "id": "slope_seg_002", "sequence_number": 2}]
    state["segment_schemes"][0]["segments"] = original
    monkeypatch.setattr("app.agents.segment_merge_agent.settings.openai_api_key", None)

    update = await SegmentMergeAgent().execute(state)

    assert update["segment_schemes"][0]["segments"] == original
    assert update["warnings"]


@pytest.mark.asyncio
async def test_quality_is_deterministic_and_reports_real_missing_data():
    state = base_state()
    state["generated_content"] = {"description": "模板", "generation_mode": "fallback"}
    agent = QualityAgent()

    first = (await agent.execute(state))["quality_assessment"]
    second = (await agent.execute(state))["quality_assessment"]

    assert first == second
    assert "poi_points" in first["missing_data"]
    assert "llm_generated_content" in first["missing_data"]
    assert first["needs_human_review"] is True
    assert "external_api_calls" not in json.dumps(first)


@pytest.mark.asyncio
async def test_workflow_reports_monotonic_progress_to_sync_callback():
    final = valid_result()

    class FakeCompiled:
        async def astream(self, _state, stream_mode="values"):
            assert stream_mode == "values"
            yield {"current_step": "init", "overall_progress": 5}
            yield {"current_step": "trajectory_analysis", "overall_progress": 15}
            yield {"current_step": "aggregate_result", "overall_progress": 100, "final_result": final}

    events = []
    workflow = AnalysisWorkflow()
    workflow._compiled = FakeCompiled()

    result = await workflow.execute_workflow({}, progress_callback=lambda step, progress: events.append((step, progress)))

    assert result["segment_schemes"]
    assert events == [("init", 5), ("trajectory_analysis", 15), ("aggregate_result", 100)]


def test_callback_payload_uses_new_contract_only():
    payload = CallbackService()._build_callback_payload("task-1", "route-1", valid_result(), "completed")
    assert payload["status"] == "completed"
    assert payload["segment_schemes"]
    assert payload["segment_schemes"][0]["segments"][0]["scheme_type"] == "slope"
    assert payload["poi_points"]
    assert "segments" not in payload
    assert "water_sources" not in payload


def test_completed_callback_rejects_invalid_result():
    invalid = valid_result()
    invalid["poi_points"][0]["confidence"] = 2
    with pytest.raises(ValidationError):
        CallbackService()._build_callback_payload("task-1", "route-1", invalid, "completed")


@pytest.mark.asyncio
async def test_workflow_reports_monotonic_progress_to_async_callback():
    final = valid_result()

    class FakeCompiled:
        async def astream(self, _state, stream_mode="values"):
            yield {"current_step": "trajectory_analysis", "overall_progress": 15}
            yield {"current_step": "stale", "overall_progress": 10}
            yield {"current_step": "aggregate_result", "overall_progress": 100, "final_result": final}

    events = []

    async def callback(step, progress):
        events.append((step, progress))

    workflow = AnalysisWorkflow()
    workflow._compiled = FakeCompiled()
    await workflow.execute_workflow({}, progress_callback=callback)

    assert events == [("trajectory_analysis", 15), ("aggregate_result", 100)]


@pytest.mark.asyncio
async def test_main_syncs_workflow_progress_and_completed_callback(monkeypatch):
    from app import main

    task_id = main.task_manager.create_task({"kml_source": "inline.kml"})

    class SuccessfulWorkflow:
        async def execute_workflow(self, _request, progress_callback):
            await progress_callback("trajectory_analysis", 15)
            await progress_callback("quality_assessment", 85)
            return valid_result()

    send_callback = AsyncMock(return_value=True)
    monkeypatch.setattr(main, "workflow", SuccessfulWorkflow())
    monkeypatch.setattr(main.callback_service, "send_callback", send_callback)

    await main.execute_analysis_async(task_id, {"kml_source": "inline.kml", "route_id": "route-1"})

    task = main.task_manager.get_task(task_id)
    assert task["status"].value == "completed"
    assert task["progress"] == 100
    send_callback.assert_awaited_once()
    assert send_callback.await_args.kwargs["status"] == "completed"


@pytest.mark.asyncio
async def test_main_marks_workflow_failure_and_sends_failed_callback(monkeypatch):
    from app import main

    task_id = main.task_manager.create_task({"kml_source": "inline.kml"})

    class FailedWorkflow:
        async def execute_workflow(self, _request, progress_callback):
            await progress_callback("trajectory_analysis_failed", 15)
            raise RuntimeError("FATAL: invalid trajectory")

    send_callback = AsyncMock(return_value=True)
    monkeypatch.setattr(main, "workflow", FailedWorkflow())
    monkeypatch.setattr(main.callback_service, "send_callback", send_callback)

    await main.execute_analysis_async(task_id, {"kml_source": "inline.kml", "route_id": "route-1"})

    task = main.task_manager.get_task(task_id)
    assert task["status"].value == "failed"
    assert "FATAL" in task["error"]
    send_callback.assert_awaited_once()
    assert send_callback.await_args.kwargs["status"] == "failed"


@pytest.mark.parametrize(
    ("mutate", "error_fragment"),
    [
        (lambda result: result.update({"unexpected": True}), "unexpected"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"unexpected": True}), "unexpected"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"id": " "}), "id"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"name": ""}), "name"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"sequence_number": 0}), "sequence_number"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"distance": -0.1}), "distance"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"elevation_gain": -1}), "elevation_gain"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"elevation_loss": -1}), "elevation_loss"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"estimated_time": -1}), "estimated_time"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"difficulty": 6}), "difficulty"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"color": "red"}), "color"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"confidence": -0.1}), "confidence"),
        (lambda result: result["segment_schemes"][0].update({"label": " "}), "label"),
        (lambda result: result["segment_schemes"][0].update({"scheme_type": ""}), "scheme_type"),
        (lambda result: result["segment_schemes"][0]["segments"][0].update({"scheme_type": "day"}), "scheme_type"),
        (lambda result: result["segment_schemes"][0].update({"segments": []}), "segments"),
        (lambda result: result["poi_points"][0].update({"name": ""}), "name"),
        (lambda result: result["poi_points"][0].update({"category": "restaurant"}), "category"),
        (lambda result: result["poi_points"][0].update({"source": "llm"}), "source"),
        (lambda result: result["poi_points"][0].pop("confidence"), "confidence"),
    ],
)
def test_final_result_rejects_untrusted_structure(mutate, error_fragment):
    result = valid_result()
    mutate(result)

    with pytest.raises(ValidationError) as exc_info:
        EnhancedRouteOutput.model_validate(result)

    assert error_fragment in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_state_accumulates_warnings_from_consecutive_nodes():
    async def first_warning(_state):
        return {"warnings": [{"level": "warning", "message": "first"}]}

    async def second_warning(_state):
        return {"warnings": [{"level": "warning", "message": "second"}]}

    graph = StateGraph(AgentState)
    graph.add_node("first", first_warning)
    graph.add_node("second", second_warning)
    graph.set_entry_point("first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    final_state = await graph.compile().ainvoke({"warnings": [], "errors": []})

    assert [warning["message"] for warning in final_state["warnings"]] == ["first", "second"]


@pytest.mark.asyncio
async def test_workflow_does_not_report_initial_zero_progress_snapshot():
    final = valid_result()

    class FakeCompiled:
        async def astream(self, _state, stream_mode="values"):
            yield {"current_step": "init", "overall_progress": 0}
            yield {"current_step": "init", "overall_progress": 5}
            yield {"current_step": "aggregate_result", "overall_progress": 100, "final_result": final}

    events = []
    workflow = AnalysisWorkflow()
    workflow._compiled = FakeCompiled()
    await workflow.execute_workflow({}, progress_callback=lambda step, progress: events.append((step, progress)))

    assert events == [("init", 5), ("aggregate_result", 100)]


def test_callback_segment_conversion_preserves_validated_identity_and_order_fields():
    segment = {
        **valid_segment(),
        "id": "slope_seg_007",
        "name": "保留名称",
        "sequence_number": 7,
        "color": "#123ABC",
    }

    converted = CallbackService()._convert_segments([segment])[0]

    assert converted["id"] == "slope_seg_007"
    assert converted["name"] == "保留名称"
    assert converted["sequence_number"] == 7
    assert converted["color"] == "#123ABC"


@pytest.mark.asyncio
async def test_short_valid_track_uses_agent_state_meters_and_covers_full_track():
    points = [
        {"latitude": 30.0, "longitude": 120.0, "elevation": 100, "distance_from_start": 0.0},
        {"latitude": 30.00027, "longitude": 120.0, "elevation": 100, "distance_from_start": 30.0},
    ]

    update = await SegmentationAgent().execute({"request": {}, "track_points": points})
    slope_segments = update["segment_schemes"][0]["segments"]

    assert len(slope_segments) == 1
    assert slope_segments[0]["id"] == "slope_seg_001"
    assert slope_segments[0]["distance"] == pytest.approx(0.03)
    assert slope_segments[0]["estimated_time"] == 1
    assert slope_segments[0]["track_start_index"] == 0
    assert slope_segments[0]["track_end_index"] == 1


@pytest.mark.asyncio
async def test_main_keeps_task_processing_until_completed_callback_returns(monkeypatch):
    from app import main

    task_id = main.task_manager.create_task({"kml_source": "inline.kml"})
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()

    class SuccessfulWorkflow:
        async def execute_workflow(self, _request, progress_callback):
            return valid_result()

    async def blocking_callback(**_kwargs):
        callback_started.set()
        await release_callback.wait()
        return True

    monkeypatch.setattr(main, "workflow", SuccessfulWorkflow())
    monkeypatch.setattr(main.callback_service, "send_callback", blocking_callback)

    execution = asyncio.create_task(
        main.execute_analysis_async(
            task_id,
            {"kml_source": "inline.kml", "route_id": "route-1"},
        )
    )
    await callback_started.wait()

    task_during_callback = main.task_manager.get_task(task_id)
    assert task_during_callback["status"].value == "processing"
    assert task_during_callback["result"] is None

    release_callback.set()
    await execution

    task_after_callback = main.task_manager.get_task(task_id)
    assert task_after_callback["status"].value == "completed"
    assert task_after_callback["result"] == valid_result()


@pytest.mark.asyncio
async def test_main_marks_completed_analysis_failed_when_callback_is_not_delivered(monkeypatch):
    from app import main

    task_id = main.task_manager.create_task({"kml_source": "inline.kml"})

    class SuccessfulWorkflow:
        async def execute_workflow(self, _request, progress_callback):
            return valid_result()

    monkeypatch.setattr(main, "workflow", SuccessfulWorkflow())
    monkeypatch.setattr(main.callback_service, "send_callback", AsyncMock(return_value=False))

    await main.execute_analysis_async(task_id, {"kml_source": "inline.kml", "route_id": "route-1"})

    task = main.task_manager.get_task(task_id)
    assert task["status"].value == "failed"
    assert task["result"] is None
    assert "分析完成" in task["error"]
    assert "未送达" in task["error"]


def test_kml_markers_only_include_point_placemarks():
    kml = """\
    <kml xmlns="http://www.opengis.net/kml/2.2"><Document>
      <Placemark><name>路线</name><LineString><coordinates>
        120.0,30.0,100 120.001,30.001,110
      </coordinates></LineString></Placemark>
      <Placemark><name>轨迹</name><gx:Track xmlns:gx="http://www.google.com/kml/ext/2.2">
        <gx:coord>120.0 30.0 100</gx:coord>
      </gx:Track></Placemark>
      <Placemark><name>补给点</name><Point><coordinates>120.002,30.002,120</coordinates></Point></Placemark>
    </Document></kml>
    """

    markers = TrajectoryAgent()._extract_markers(kml, "kml")

    assert [marker["name"] for marker in markers] == ["补给点"]
    assert markers[0]["longitude"] == pytest.approx(120.002)
    assert markers[0]["latitude"] == pytest.approx(30.002)


@pytest.mark.asyncio
async def test_quality_agent_exception_is_fatal_and_prevents_final_result(monkeypatch):
    workflow = AnalysisWorkflow()
    monkeypatch.setattr(
        workflow.trajectory_agent,
        "execute",
        AsyncMock(
            return_value={
                "track_points": [
                    {"latitude": 30.0, "longitude": 120.0, "elevation": 100, "distance_from_start": 0.0},
                    {"latitude": 30.001, "longitude": 120.001, "elevation": 110, "distance_from_start": 150.0},
                ],
                "basic_stats": base_state()["basic_stats"],
                "kml_markers": [],
                "current_step": "trajectory_analysis",
                "overall_progress": 15,
            }
        ),
    )
    monkeypatch.setattr(
        workflow.segmentation_agent,
        "execute",
        AsyncMock(return_value={"segment_schemes": base_state()["segment_schemes"], "overall_progress": 30}),
    )
    monkeypatch.setattr(workflow.segment_merge_agent, "execute", AsyncMock(return_value={"overall_progress": 40}))
    monkeypatch.setattr(
        workflow.poi_agent,
        "execute",
        AsyncMock(return_value={"poi_points": [], "overall_progress": 50}),
    )
    monkeypatch.setattr(
        workflow.quality_agent,
        "execute",
        AsyncMock(side_effect=RuntimeError("quality crashed")),
    )

    with pytest.raises(RuntimeError, match="QUALITY_ASSESSMENT_ERROR"):
        await workflow.execute_workflow(
            {"kml_source": "inline.kml", "enable_content_generation": False}
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_final_result_rejects_non_finite_numbers_but_accepts_numeric_strings(non_finite):
    result = valid_result()
    result["total_distance_km"] = non_finite
    with pytest.raises(ValidationError):
        EnhancedRouteOutput.model_validate(result)

    numeric_string_result = valid_result()
    numeric_string_result["total_distance_km"] = "1.2"
    assert EnhancedRouteOutput.model_validate(numeric_string_result).total_distance_km == 1.2


def test_multi_segment_gpx_distance_excludes_gap_between_segments():
    gpx = """\
    <gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
      <trk><name>two segments</name>
        <trkseg>
          <trkpt lat="30.0" lon="120.0"><ele>100</ele></trkpt>
          <trkpt lat="30.0" lon="120.01"><ele>110</ele></trkpt>
        </trkseg>
        <trkseg>
          <trkpt lat="40.0" lon="130.0"><ele>200</ele></trkpt>
          <trkpt lat="40.0" lon="130.01"><ele>210</ele></trkpt>
        </trkseg>
      </trk>
    </gpx>
    """

    points = TrackProcessor().parse_kml_or_gpx(gpx, "gpx")
    expected_km = geodesic((30.0, 120.0), (30.0, 120.01)).km + geodesic(
        (40.0, 130.0), (40.0, 130.01)
    ).km

    assert [point.distance_from_start for point in points] == sorted(
        point.distance_from_start for point in points
    )
    assert points[2].distance_from_start == pytest.approx(points[1].distance_from_start)
    assert points[-1].distance_from_start == pytest.approx(expected_km)


def test_multiple_kml_linestring_distance_excludes_gap_between_lines():
    kml = """\
    <kml xmlns="http://www.opengis.net/kml/2.2"><Document>
      <Placemark><LineString><coordinates>
        120.0,30.0,100 120.01,30.0,110
      </coordinates></LineString></Placemark>
      <Placemark><LineString><coordinates>
        130.0,40.0,200 130.01,40.0,210
      </coordinates></LineString></Placemark>
    </Document></kml>
    """

    points = TrackProcessor().parse_kml_or_gpx(kml, "kml")
    expected_km = geodesic((30.0, 120.0), (30.0, 120.01)).km + geodesic(
        (40.0, 130.0), (40.0, 130.01)
    ).km

    assert points[2].distance_from_start == pytest.approx(points[1].distance_from_start)
    assert points[-1].distance_from_start == pytest.approx(expected_km)


def test_final_difficulty_uses_distance_weighted_default_slope_segments():
    state = base_state()
    state["request"].pop("estimated_difficulty", None)
    state["basic_stats"].update({"total_distance_km": 10.0, "total_gain_m": 1000.0})
    state["segment_schemes"][0]["segments"] = [
        {**valid_segment(), "distance": 10.0, "difficulty": 5}
    ]

    result = AnalysisWorkflow()._build_final_result(state)

    assert result["estimated_difficulty"] == 5


def test_final_difficulty_prefers_explicit_request_value():
    state = base_state()
    state["request"]["estimated_difficulty"] = 2
    state["segment_schemes"][0]["segments"] = [
        {**valid_segment(), "distance": 10.0, "difficulty": 5}
    ]

    result = AnalysisWorkflow()._build_final_result(state)

    assert result["estimated_difficulty"] == 2


def test_final_difficulty_falls_back_to_basic_stats_deterministically():
    state = base_state()
    state["request"].pop("estimated_difficulty", None)
    state["segment_schemes"] = []
    state["basic_stats"] = {"distance_km": 10.0, "elevation_gain": 1000.0}

    result = AnalysisWorkflow()._build_final_result(state)

    assert result["estimated_difficulty"] == 5


@pytest.mark.parametrize(
    "raw",
    [
        '{"merge_groups": [[1, 2], [2, 3]]}',
        '{"merge_groups": [[1, 2], [3]]}',
    ],
)
def test_segment_merge_rejects_entire_response_when_any_group_is_invalid(raw):
    assert SegmentMergeAgent()._parse_response(raw, total_segs=3) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '{"merge_groups": [[1, 2], [2, 3]]}',
        '{"merge_groups": [[1, 2], [3]]}',
    ],
)
async def test_segment_merge_invalid_plan_preserves_original_segments_and_warns(monkeypatch, raw):
    state = base_state()
    original = [
        valid_segment(),
        {**valid_segment(), "id": "slope_seg_002", "sequence_number": 2},
        {**valid_segment(), "id": "slope_seg_003", "sequence_number": 3},
    ]
    state["segment_schemes"][0]["segments"] = original
    agent = SegmentMergeAgent()
    monkeypatch.setattr(
        agent,
        "_get_llm",
        lambda: SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=raw))),
    )

    update = await agent.execute(state)

    assert update["segment_schemes"] is state["segment_schemes"]
    assert update["segment_schemes"][0]["segments"] is original
    assert update["warnings"]
    assert "保持原样" in update["warnings"][0]["message"]


def test_multiple_kml_gx_tracks_exclude_gap_and_keep_cumulative_distance():
    kml = """\
    <kml xmlns="http://www.opengis.net/kml/2.2"
         xmlns:gx="http://www.google.com/kml/ext/2.2"><Document>
      <Placemark><gx:Track>
        <gx:coord>120.0 30.0 100</gx:coord>
        <gx:coord>120.01 30.0 110</gx:coord>
      </gx:Track></Placemark>
      <Placemark><gx:Track>
        <gx:coord>130.0 40.0 200</gx:coord>
        <gx:coord>130.01 40.0 210</gx:coord>
      </gx:Track></Placemark>
    </Document></kml>
    """

    points = TrackProcessor().parse_kml_or_gpx(kml, "kml")
    expected_km = geodesic((30.0, 120.0), (30.0, 120.01)).km + geodesic(
        (40.0, 130.0), (40.0, 130.01)
    ).km

    assert points[2].distance_from_start == pytest.approx(points[1].distance_from_start)
    assert points[-1].distance_from_start == pytest.approx(expected_km)
    assert [point.segment_index for point in points] == [0, 0, 1, 1]


def test_multiple_gpx_segments_elevation_gain_excludes_boundary_jump():
    gpx = """\
    <gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
      <trk>
        <trkseg>
          <trkpt lat="30.0" lon="120.0"><ele>100</ele></trkpt>
          <trkpt lat="30.0" lon="120.01"><ele>110</ele></trkpt>
        </trkseg>
        <trkseg>
          <trkpt lat="40.0" lon="130.0"><ele>200</ele></trkpt>
          <trkpt lat="40.0" lon="130.01"><ele>210</ele></trkpt>
        </trkseg>
      </trk>
    </gpx>
    """
    processor = TrackProcessor()

    points = processor.parse_kml_or_gpx(gpx, "gpx")
    stats = processor.calculate_elevation_stats(points)

    assert [point.segment_index for point in points] == [0, 0, 1, 1]
    assert stats.total_gain_m == pytest.approx(20.0)
    assert stats.total_loss_m == pytest.approx(0.0)


def test_multiple_kml_linestrings_elevation_stats_exclude_segment_boundary():
    kml = """\
    <kml xmlns="http://www.opengis.net/kml/2.2"><Document>
      <Placemark><LineString><coordinates>
        120.0,30.0,100 120.01,30.0,110
      </coordinates></LineString></Placemark>
      <Placemark><LineString><coordinates>
        130.0,40.0,200 130.01,40.0,190
      </coordinates></LineString></Placemark>
    </Document></kml>
    """
    processor = TrackProcessor()

    points = processor.parse_kml_or_gpx(kml, "kml")
    stats = processor.calculate_elevation_stats(points)

    assert [point.segment_index for point in points] == [0, 0, 1, 1]
    assert stats.total_gain_m == pytest.approx(10.0)
    assert stats.total_loss_m == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_track_segment_metadata_survives_state_conversion_without_entering_final_contract():
    kml = """\
    <kml xmlns="http://www.opengis.net/kml/2.2"><Document>
      <Placemark><LineString><coordinates>
        120.0,30.0,100 120.01,30.0,110
      </coordinates></LineString></Placemark>
      <Placemark><LineString><coordinates>
        130.0,40.0,200 130.01,40.0,210
      </coordinates></LineString></Placemark>
    </Document></kml>
    """

    update = await TrajectoryAgent(processor=TrackProcessor()).execute(
        {"request": {"kml_content": kml}}
    )
    restored = SegmentationAgent()._dicts_to_track_points(update["track_points"])

    assert [point["segment_index"] for point in update["track_points"]] == [0, 0, 1, 1]
    assert [point.segment_index for point in restored] == [0, 0, 1, 1]
    assert "segment_index" not in EnhancedRouteOutput.model_validate(valid_result()).model_dump(mode="json")


@pytest.mark.asyncio
async def test_disconnected_singleton_gpx_segments_are_fatal_in_full_workflow():
    gpx = """\
    <gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
      <trk>
        <trkseg><trkpt lat="30.0" lon="120.0"><ele>100</ele></trkpt></trkseg>
        <trkseg><trkpt lat="40.0" lon="130.0"><ele>1000</ele></trkpt></trkseg>
      </trk>
    </gpx>
    """

    with pytest.raises(RuntimeError, match="连续轨迹段|可行走距离"):
        await AnalysisWorkflow().execute_workflow(
            {
                "kml_source": "inline.gpx",
                "kml_content": gpx,
                "enable_content_generation": False,
                "enable_poi_query": False,
            }
        )


@pytest.mark.asyncio
async def test_multi_track_segments_keep_slope_and_day_calculations_within_source_boundaries():
    points = [
        {
            "latitude": 30.0,
            "longitude": 120.0,
            "elevation": 100,
            "distance_from_start": 0.0,
            "timestamp": "2026-01-01T00:00:00",
            "segment_index": 0,
        },
        {
            "latitude": 30.0,
            "longitude": 120.01,
            "elevation": 110,
            "distance_from_start": 960.0,
            "timestamp": "2026-01-01T01:00:00",
            "segment_index": 0,
        },
        {
            "latitude": 40.0,
            "longitude": 130.0,
            "elevation": 1000,
            "distance_from_start": 960.0,
            "timestamp": "2026-01-01T02:00:00",
            "segment_index": 1,
        },
        {
            "latitude": 40.0,
            "longitude": 130.01,
            "elevation": 990,
            "distance_from_start": 1810.0,
            "timestamp": "2026-01-01T03:00:00",
            "segment_index": 1,
        },
    ]

    update = await SegmentationAgent().execute({"request": {}, "track_points": points})
    source_segment_indexes = [point["segment_index"] for point in points]

    for scheme in update["segment_schemes"][:2]:
        for segment in scheme["segments"]:
            start = segment["track_start_index"]
            end = segment["track_end_index"]
            assert source_segment_indexes[start] == source_segment_indexes[end]

    slope_segments = update["segment_schemes"][0]["segments"]
    assert [(segment["track_start_index"], segment["track_end_index"]) for segment in slope_segments] == [
        (0, 1),
        (2, 3),
    ]
    assert sum(segment["elevation_gain"] for segment in slope_segments) == pytest.approx(10.0)
    assert sum(segment["elevation_loss"] for segment in slope_segments) == pytest.approx(10.0)

    day_segments = update["segment_schemes"][1]["segments"]
    assert day_segments
    assert sum(segment["elevation_gain"] for segment in day_segments) == pytest.approx(10.0)
    assert sum(segment["elevation_loss"] for segment in day_segments) == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_segment_merge_does_not_accept_a_cross_source_boundary_merge(monkeypatch):
    state = base_state()
    first = {
        **valid_segment(),
        "track_start_index": 0,
        "track_end_index": 1,
    }
    second = {
        **valid_segment(),
        "id": "slope_seg_002",
        "sequence_number": 2,
        "track_start_index": 2,
        "track_end_index": 3,
    }
    original = [first, second]
    state["track_points"] = [
        {"segment_index": 0},
        {"segment_index": 0},
        {"segment_index": 1},
        {"segment_index": 1},
    ]
    state["segment_schemes"][0]["segments"] = original
    crossing_merge = {
        **valid_segment(),
        "distance": 2.4,
        "track_start_index": 0,
        "track_end_index": 3,
    }
    agent = SegmentMergeAgent()
    monkeypatch.setattr(agent, "_llm_merge", AsyncMock(return_value=[crossing_merge]))

    update = await agent.execute(state)

    assert update["segment_schemes"][0]["segments"] == original
    assert update["warnings"]
    assert "原始轨迹边界" in update["warnings"][0]["message"]
