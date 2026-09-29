import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.agents.analysis_workflow import AnalysisWorkflow
from app.agents.content_agent import ContentAgent
from app.agents.trajectory_agent import TrajectoryAgent
from app.models.response import EnhancedRouteOutput
from app.services.callback_service import CallbackService


def segment(scheme, number, start, end):
    return {
        "id": f"{scheme}_{number}", "name": f"第{number}天" if scheme == "day" else f"路段{number}",
        "sequence_number": number, "color": "#2196F3", "scheme_type": scheme,
        "distance": 1.2, "elevation_gain": 100, "elevation_loss": 30,
        "estimated_time": 40, "difficulty": 1,
        "track_start_index": start, "track_end_index": end,
        "description": None, "notes": None,
    }


def source_marker(name, longitude, timestamp=None):
    marker = {"name": name, "latitude": 30.0, "longitude": longitude, "elevation": 4100.0,
              "description": "", "confidence": 1.0}
    if timestamp:
        marker["timestamp"] = timestamp
    return marker


def source_state():
    return {
        "request": {"kml_source": "inline.kml", "enable_content_generation": True},
        "track_points": [
            {"latitude": 30.0, "longitude": lon, "elevation": 4000 + index * 10,
             "timestamp": f"2026-07-{20 if index < 3 else 21}T{8 + index % 3:02}:00:00",
             "segment_index": 0 if index < 3 else 1, "distance_from_start": index * 100}
            for index, lon in enumerate([120.0, 120.001, 120.002, 120.002, 120.003, 120.004])
        ],
        "source_description": "第一天途经溪谷到营地，原作者约数2km。\n第二天沿山脊返回。",
        "kml_markers": [
            source_marker("过河", 120.001, "2026-07-20T09:00:00+00:00"),
            source_marker("看到甲营地", 120.0002, "2026-07-20T08:05:00+00:00"),
            source_marker("第一天乙营地", 120.002, "2026-07-20T10:40:00+00:00"),
            source_marker("乱石坡横切", 120.003, "2026-07-21T09:00:00+00:00"),
        ],
        "segment_schemes": [
            {"scheme_type": "slope", "label": "按坡度", "is_default": True,
             "segments": [segment("slope", 1, 0, 2), segment("slope", 2, 3, 5)]},
            {"scheme_type": "day", "label": "按天", "is_default": False,
             "segments": [segment("day", 1, 0, 2), segment("day", 2, 3, 5)]},
        ],
        "basic_stats": {"total_distance_km": 2.4, "total_gain_m": 200, "total_loss_m": 60,
                        "max_elevation": 4100, "min_elevation": 4000, "is_loop": False},
        "errors": [], "warnings": [],
    }


async def enriched(state=None):
    state = state or source_state()
    update = await AnalysisWorkflow()._recognize_poi_node(state)
    return {**state, **update}


def days(state):
    return next(s["segments"] for s in state["segment_schemes"] if s["scheme_type"] == "day")


def camps(state):
    return [p for p in state["poi_points"] if p["category"] == "camp"]


def llm_content(**extra):
    return {"description": "路线概述", "highlights": [], "difficulties": [],
            "safety_notes": [], "equipment_recommendations": [], **extra}


@pytest.mark.asyncio
async def test_trajectory_preserves_document_notes_and_explicit_marker_times():
    content = '''<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
    <Document><description><![CDATA[<div>第一天过河</div><div>第二天返回</div><script>ignore_me()</script>]]></description>
    <Placemark id="camp"><name>第一天营地</name><TimeStamp><when>2026-07-20T05:45:43Z</when></TimeStamp>
      <Point><coordinates>120,30,4000</coordinates></Point></Placemark>
    <Placemark id="note"><name>过河</name><ExtendedData><Data name="Time"><value>1784526368000</value></Data></ExtendedData>
      <Point><coordinates>120.001,30,4010</coordinates></Point></Placemark>
    <Placemark><gx:Track><gx:coord>120 30 4000</gx:coord><gx:coord>120.001 30 4010</gx:coord>
      <when>2026-07-20T05:45:43Z</when><when>2026-07-20T05:46:08Z</when></gx:Track></Placemark>
    </Document></kml>'''
    result = await TrajectoryAgent().execute({"request": {"kml_content": content}})
    assert result.get("source_description") == "第一天过河\n第二天返回"
    assert [m.get("timestamp") for m in result["kml_markers"]] == [
        "2026-07-20T05:45:43+00:00", "2026-07-20T05:46:08+00:00"]
    assert result["kml_markers"][0]["source_id"] == "camp"


@pytest.mark.asyncio
async def test_enrichment_keeps_statistics_and_raw_input_and_assigns_notes_to_their_day():
    state = source_state()
    original = deepcopy(state)
    result = await enriched(state)
    first, second = days(result)
    assert "1.2" in (first["description"] or "")
    assert "原作者" in first["description"] and "2km" in first["description"]
    assert "过河" in first["notes"] and "乱石坡横切" not in first["notes"]
    assert "乱石坡横切" in second["notes"] and "过河" not in second["notes"]
    assert "饮用" not in json.dumps(result, ensure_ascii=False)
    assert state == original
    for before, after in zip(state["segment_schemes"], result["segment_schemes"]):
        for left, right in zip(before["segments"], after["segments"]):
            assert {k: v for k, v in left.items() if k not in {"description", "notes"}} == {
                k: v for k, v in right.items() if k not in {"description", "notes"}}
    assert result["track_points"] == state["track_points"]
    assert len(result["poi_points"]) == len(state["kml_markers"])


@pytest.mark.asyncio
async def test_camp_snapshot_distinguishes_seen_camp_from_recorded_overnight():
    result = await enriched()
    seen, overnight = camps(result)
    assert seen["card_data"]["camp_analysis"]["usage"] == "along_route"
    snapshot = overnight["card_data"]["camp_analysis"]
    assert snapshot["usage"] == "recorded_overnight"
    assert snapshot["status"] == "pending_verification"
    assert snapshot["source"] == "source_marker"
    assert snapshot["evidence"] == "第一天乙营地"
    assert [(a["scheme_type"], a.get("day_number")) for a in snapshot["associations"]] == [
        ("day", 1), ("slope", None)]
    assert "原记录当晚营地" in days(result)[0]["description"]
    assert "当晚营地" not in days(result)[1]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,lon,timestamp", [
    ("乙营地", 120.002, None),
    ("第一天乙营地", 130, "2026-07-20T10:40:00+00:00"),
    ("第一天乙营地", 120.002, "2026-07-21T08:00:00+00:00"),
])
async def test_ambiguous_distant_or_conflicting_camp_remains_unassigned(name, lon, timestamp):
    state = source_state()
    state["kml_markers"] = [source_marker(name, lon, timestamp)]
    result = await enriched(state)
    snapshot = camps(result)[0]["card_data"]["camp_analysis"]
    assert snapshot["usage"] == "unknown"
    assert snapshot["associations"] == []
    assert snapshot["unresolved_reason"]
    assert result["warnings"]
    assert all("原记录当晚营地" not in d["description"] for d in days(result))


@pytest.mark.asyncio
async def test_timestamp_disambiguates_overlapping_days_without_inventing_overnight_use():
    state = source_state()
    state["kml_markers"] = [source_marker("路过乙营地", 120.002, "2026-07-21T08:00:00Z")]
    result = await enriched(state)
    snapshot = camps(result)[0]["card_data"]["camp_analysis"]
    assert snapshot["usage"] == "along_route"
    assert [a["day_number"] for a in snapshot["associations"] if a["scheme_type"] == "day"] == [2]
    assert "路过乙营地" not in days(result)[0]["description"]


@pytest.mark.asyncio
async def test_no_markers_keeps_notes_empty_but_produces_statistical_descriptions():
    state = source_state()
    state["kml_markers"] = []
    state["source_description"] = ""
    result = await enriched(state)
    assert result["poi_points"] == []
    assert all(segment["description"] and segment["notes"] is None
               for scheme in result["segment_schemes"] for segment in scheme["segments"])


@pytest.mark.asyncio
async def test_llm_only_reorders_existing_segment_units(monkeypatch):
    state = await enriched()
    facts = state.get("segment_content_facts") or []
    assert facts
    choices = [{"scheme_type": f["scheme_type"], "segment_id": f["segment_id"],
                "unit_order": [u["id"] for u in reversed(f["units"])]} for f in facts]
    response = SimpleNamespace(content=json.dumps(llm_content(segment_content=choices), ensure_ascii=False))
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
    monkeypatch.setattr("app.agents.content_agent.settings.openai_api_key", "test-key")
    monkeypatch.setattr("app.agents.content_agent.get_llm", lambda: model)
    original = deepcopy(state)
    result = await ContentAgent().execute(state)
    for scheme in result["segment_schemes"]:
        for s in scheme["segments"]:
            f = next(f for f in facts if (f["scheme_type"], f["segment_id"]) == (scheme["scheme_type"], s["id"]))
            assert s["description"] == "\n".join(u["text"] for u in reversed(f["units"]))
    assert state == original
    assert result["generated_content"]["generation_mode"] == "llm"
    assert not result.get("degraded")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, [], "free text", [{"scheme_type": "day", "segment_id": "day_1", "unit_order": ["invented"]}]])
async def test_invalid_llm_content_order_preserves_grounded_descriptions(monkeypatch, invalid):
    state = await enriched()
    response = SimpleNamespace(content=json.dumps(llm_content(segment_content=invalid), ensure_ascii=False))
    monkeypatch.setattr("app.agents.content_agent.settings.openai_api_key", "test-key")
    monkeypatch.setattr("app.agents.content_agent.get_llm", lambda: SimpleNamespace(ainvoke=AsyncMock(return_value=response)))
    result = await ContentAgent().execute(state)
    assert result.get("segment_schemes") == state["segment_schemes"]
    assert result["degraded"] is True
    assert result["warnings"]


@pytest.mark.asyncio
async def test_grounded_output_survives_model_failure_and_callback_serialization(monkeypatch):
    state = await enriched()
    monkeypatch.setattr("app.agents.content_agent.settings.openai_api_key", None)
    state.update(await ContentAgent().execute(state))
    final = EnhancedRouteOutput.model_validate(AnalysisWorkflow()._build_final_result(state)).model_dump(mode="json")
    callback = CallbackService()._build_callback_payload("task-1", "route-1", final, "completed")
    assert all(s["description"] for scheme in callback["segment_schemes"] for s in scheme["segments"])
    assert callback["poi_points"][2]["card_data"]["camp_analysis"]["usage"] == "recorded_overnight"
    assert callback["generation_mode"] == "fallback"


@pytest.mark.asyncio
async def test_final_model_rejects_camp_snapshot_with_nonexistent_range():
    state = await enriched()
    final = AnalysisWorkflow()._build_final_result(state)
    camp = next(p for p in final["poi_points"] if p["category"] == "camp")
    camp["card_data"]["camp_analysis"]["associations"][0]["track_end_index"] = 999
    with pytest.raises(ValidationError):
        EnhancedRouteOutput.model_validate(final)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["day_number", "duplicate", "foreign_field", "verified", "not_camp", "null_snapshot"])
async def test_final_camp_contract_rejects_inconsistent_or_invented_snapshot(case):
    state = await enriched()
    final = AnalysisWorkflow()._build_final_result(state)
    camp = next(p for p in final["poi_points"] if p["category"] == "camp")
    snapshot = camp["card_data"]["camp_analysis"]
    if case == "day_number":
        snapshot["associations"][0]["day_number"] = 2
    elif case == "duplicate":
        snapshot["associations"].append(deepcopy(snapshot["associations"][0]))
    elif case == "foreign_field":
        snapshot["facilities"] = ["hot shower"]
    elif case == "verified":
        snapshot["status"] = "verified"
    elif case == "not_camp":
        camp["category"] = "photo"
    else:
        camp["card_data"]["camp_analysis"] = None
    with pytest.raises(ValidationError):
        EnhancedRouteOutput.model_validate(final)


@pytest.mark.asyncio
async def test_one_invalid_llm_range_falls_back_without_discarding_other_valid_orders(monkeypatch):
    state = await enriched()
    facts = state["segment_content_facts"]
    orders = [{"scheme_type": f["scheme_type"], "segment_id": f["segment_id"],
               "unit_order": [u["id"] for u in reversed(f["units"])]} for f in facts]
    orders[0]["description"] = "模型虚构的温泉营地"
    response = SimpleNamespace(content=json.dumps(llm_content(segment_content=orders), ensure_ascii=False))
    monkeypatch.setattr("app.agents.content_agent.settings.openai_api_key", "test-key")
    monkeypatch.setattr("app.agents.content_agent.get_llm", lambda: SimpleNamespace(ainvoke=AsyncMock(return_value=response)))

    update = await ContentAgent().execute(state)

    assert update["segment_schemes"][0]["segments"][0] == state["segment_schemes"][0]["segments"][0]
    assert "模型虚构的温泉营地" not in json.dumps(update, ensure_ascii=False)
    expected = "\n".join(u["text"] for u in reversed(facts[-1]["units"]))
    assert days(update)[-1]["description"] == expected
    assert update["degraded"]


@pytest.mark.asyncio
async def test_disabled_content_generation_preserves_grounded_workflow_results(monkeypatch):
    content = '''<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
    <Document><description>第一天前往营地。第二天返回。</description>
    <Placemark><name>第一天营地</name><TimeStamp><when>2026-07-20T09:00:00Z</when></TimeStamp>
      <Point><coordinates>120.001,30,4100</coordinates></Point></Placemark>
    <Placemark><gx:Track><gx:coord>120 30 4000</gx:coord><gx:coord>120.001 30 4100</gx:coord>
      <when>2026-07-20T08:00:00Z</when><when>2026-07-20T09:00:00Z</when></gx:Track></Placemark>
    <Placemark><gx:Track><gx:coord>120.001 30 4100</gx:coord><gx:coord>120.002 30 4050</gx:coord>
      <when>2026-07-21T08:00:00Z</when><when>2026-07-21T09:00:00Z</when></gx:Track></Placemark>
    </Document></kml>'''
    workflow = AnalysisWorkflow()
    # Merge policy is independent of content_generation; isolate only its external call.
    monkeypatch.setattr(workflow.segment_merge_agent, "execute", AsyncMock(return_value={
        "current_step": "merge_segments", "overall_progress": 35,
    }))
    model = AsyncMock(side_effect=AssertionError("Content generation was disabled"))
    monkeypatch.setattr("app.agents.content_agent.get_llm", model)

    result = await workflow.execute_workflow({"kml_content": content, "enable_content_generation": False})

    model.assert_not_called()
    assert result["generation_mode"] == "disabled"
    assert result["generated_description"] is None
    assert len(result["track_path"]) == 4
    assert len(days(result)) == 2
    assert all(s["description"] for scheme in result["segment_schemes"] for s in scheme["segments"])
    snapshot = camps(result)[0]["card_data"]["camp_analysis"]
    assert snapshot["usage"] == "recorded_overnight"
    assert [a["day_number"] for a in snapshot["associations"] if a["scheme_type"] == "day"] == [1]
    assert "第一天前往营地" in days(result)[0]["description"]


@pytest.mark.asyncio
async def test_content_enrichment_failure_cannot_discard_successfully_extracted_pois(monkeypatch):
    def fail(_state):
        raise ValueError("invalid source context")
    monkeypatch.setattr("app.agents.analysis_workflow.enrich_segment_content", fail)

    result = await AnalysisWorkflow()._recognize_poi_node(source_state())

    assert len(result["poi_points"]) == 4
    assert result["poi_points"][0]["name"] == "过河"
    assert result["warnings"]
    assert result["degraded"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["甲营地，禁止扎营", "甲营地，未在此过夜", "第一天甲营地午餐休息", "第一天甲营地（备选）"])
async def test_negative_daytime_or_alternative_camp_does_not_assert_overnight(name):
    state = source_state()
    state["kml_markers"] = [source_marker(name, 120.001, "2026-07-20T09:00:00Z")]

    result = await enriched(state)

    snapshot = camps(result)[0]["card_data"]["camp_analysis"]
    assert snapshot["usage"] != "recorded_overnight"
    assert snapshot["evidence"] == name
    assert snapshot["associations"]
    assert all("原记录当晚营地" not in d["description"] for d in days(result))
    assert "禁止扎营" not in name or "禁止扎营" in days(result)[0]["notes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp,expected_slope", [("2026-07-20T08:50:00Z", "slope_2"), (None, None)])
async def test_repeated_visit_uses_time_within_the_same_day_and_recording(timestamp, expected_slope):
    state = source_state()
    state["track_points"] = [
        {"latitude": 30.0, "longitude": lon, "elevation": 4000,
         "timestamp": f"2026-07-20T{time}", "segment_index": 0, "distance_from_start": index * 1000}
        for index, (lon, time) in enumerate([
            (120, "08:00:00"), (120.01, "08:25:00"), (120.0001, "08:50:00"), (120.001, "09:00:00")])
    ]
    state["segment_schemes"][0]["segments"] = [segment("slope", 1, 0, 1), segment("slope", 2, 1, 3)]
    state["segment_schemes"][1]["segments"] = [segment("day", 1, 0, 3)]
    state["kml_markers"] = [source_marker("路过甲营地", 120, timestamp)]

    result = await enriched(state)

    snapshot = camps(result)[0]["card_data"]["camp_analysis"]
    if expected_slope is None:
        assert snapshot["associations"] == []
        assert snapshot["unresolved_reason"]
    else:
        expected = next(s for s in state["segment_schemes"][0]["segments"] if s["id"] == expected_slope)
        slope = next(a for a in snapshot["associations"] if a["scheme_type"] == "slope")
        assert (slope["track_start_index"], slope["track_end_index"]) == (
            expected["track_start_index"], expected["track_end_index"])
        assert "路过甲营地" not in result["segment_schemes"][0]["segments"][0]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["甲营地，不要在此过夜", "甲营地，严禁扎营", "未经过甲营地", "计划路过甲营地"])
async def test_negation_or_plan_blocks_all_positive_camp_usage(name):
    state = source_state()
    state["kml_markers"] = [source_marker(name, 120.001, "2026-07-20T09:00:00Z")]

    result = await enriched(state)

    snapshot = camps(result)[0]["card_data"]["camp_analysis"]
    assert snapshot["usage"] == "unknown"
    assert snapshot["evidence"] == name
    assert snapshot["associations"]
    assert "原记录当晚营地" not in days(result)[0]["description"]
    assert "沿途营地标注" not in days(result)[0]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp", ["2026-07-20T08:04:00Z", None])
async def test_short_revisit_across_slope_boundary_is_not_assigned_by_nearest_point_only(timestamp):
    state = source_state()
    state["track_points"] = [
        {"latitude": 30.0, "longitude": lon, "elevation": 4000,
         "timestamp": f"2026-07-20T{time}", "segment_index": 0, "distance_from_start": distance}
        for lon, time, distance in [
            (119.999, "07:58:00", 0), (120, "08:00:00", 100),
            (120.0009, "08:02:00", 185), (120.0001, "08:04:00", 264), (120.001, "08:06:00", 350)]
    ]
    state["segment_schemes"][0]["segments"] = [segment("slope", 1, 0, 2), segment("slope", 2, 2, 4)]
    state["segment_schemes"][1]["segments"] = [segment("day", 1, 0, 4)]
    state["kml_markers"] = [source_marker("路过甲营地", 120, timestamp)]

    result = await enriched(state)

    snapshot = camps(result)[0]["card_data"]["camp_analysis"]
    if timestamp is None:
        assert snapshot["associations"] == []
        assert snapshot["unresolved_reason"]
    else:
        slope = next(a for a in snapshot["associations"] if a["scheme_type"] == "slope")
        assert (slope["track_start_index"], slope["track_end_index"]) == (2, 4)
