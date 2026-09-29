"""Ground range descriptions in source evidence, without changing route facts."""

from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any

import numpy as np

# Matching tolerances are not assertions of real-world accuracy or availability.
_MATCH_METERS = 100.0
_AMBIGUITY_METERS = 20.0
_TIME_TOLERANCE_SECONDS = 3600
_DAY_PATTERN = re.compile(r"(?:第\s*([一二三四五六七八九十两\d]+)\s*[天日]|\bD(?:ay)?\s*(\d+)\b)", re.I)
_NOTICE_WORDS = ("过河", "涉水", "小心", "横切", "路迹", "注意", "电瓶车", "接驳", "信号", "不要", "沿箭头", "乱石", "禁", "不允许")
# Conservative veto: a qualified/negative statement must never become a positive use claim.
_UNCERTAIN_CAMP_USAGE = re.compile(r"[不未无没禁否]|备选|备用|计划|建议|预计|可能|拟|打算|准备")


class _SourceText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"div", "p", "br", "li"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"div", "p", "li"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def source_text(value: str | None) -> str:
    parser = _SourceText()
    parser.feed(value or "")
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())


def source_time(value: str | None, *, track: bool = False) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            if not track:
                return None
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _day_number(text: str) -> int | None:
    match = _DAY_PATTERN.search(text)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    if value.isdigit():
        return int(value) or None
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if "十" in value:
        left, right = value.split("十", 1)
        return digits.get(left, 1) * 10 + digits.get(right, 0)
    return digits.get(value)


def _day_notes(description: str) -> dict[int, list[str]]:
    result = defaultdict(list)
    matches = list(_DAY_PATTERN.finditer(description))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(description)
        number = _day_number(match.group())
        if number:
            result[number].append(description[match.start():end].strip())
    return result


def _poi_key(value: dict) -> tuple:
    return value.get("name") or "未命名标记点", value.get("latitude"), value.get("longitude")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def enrich_segment_content(state: dict[str, Any]) -> dict[str, Any]:
    """Return isolated state updates; all coordinates, metrics and POIs stay intact."""
    points = state.get("track_points") or []
    schemes = deepcopy(state.get("segment_schemes") or [])
    pois = deepcopy(state.get("poi_points") or [])
    ranges = []
    for scheme in schemes:
        if scheme.get("scheme_type") not in {"day", "slope"}:
            continue
        for segment in scheme.get("segments", []):
            start, end = segment.get("track_start_index"), segment.get("track_end_index")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(points):
                ranges.append((scheme["scheme_type"], segment))
    if not ranges:
        return {"segment_schemes": schemes, "poi_points": pois, "segment_content_facts": []}

    coordinates = np.radians([[p["latitude"], p["longitude"]] for p in points])
    times = np.array([
        stamp.timestamp() if (stamp := source_time(p.get("timestamp"), track=True)) else np.nan
        for p in points
    ])
    day_ranges = [s for kind, s in ranges if kind == "day"]
    poi_indices = defaultdict(deque)
    for index, poi in enumerate(pois):
        poi_indices[_poi_key(poi)].append(index)
    assigned = defaultdict(list)
    warnings = []

    def match_marker(marker, text):
        try:
            lat, lon = np.radians([float(marker["latitude"]), float(marker["longitude"])])
        except (KeyError, TypeError, ValueError):
            return None, None, None, "标注坐标不可用"
        h = np.sin((coordinates[:, 0] - lat) / 2) ** 2
        h += np.cos(lat) * np.cos(coordinates[:, 0]) * np.sin((coordinates[:, 1] - lon) / 2) ** 2
        distances = 12742000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        candidates = np.flatnonzero(distances <= _MATCH_METERS)
        if not len(candidates):
            return None, None, None, "标注位置不在可关联轨迹范围内"
        explicit_day = _day_number(text)
        timestamp = source_time(marker.get("timestamp"))
        basis = "position"
        if explicit_day is not None:
            day = next((d for d in day_ranges if d["sequence_number"] == explicit_day), None)
            if day is None:
                return None, None, None, "原文日序不存在于本次分日结果"
            start, end = day["track_start_index"], day["track_end_index"]
            candidates = candidates[(candidates >= start) & (candidates <= end)]
            if not len(candidates):
                return None, None, None, "原文日序与标注位置冲突"
            valid_times = times[start:end + 1][np.isfinite(times[start:end + 1])]
            if timestamp and len(valid_times):
                if not valid_times.min() - _TIME_TOLERANCE_SECONDS <= timestamp.timestamp() <= valid_times.max() + _TIME_TOLERANCE_SECONDS:
                    return None, None, None, "原文日序与标注时间冲突"
            basis = "explicit_day"
        elif timestamp:
            candidates = candidates[np.isfinite(times[candidates]) & (np.abs(times[candidates] - timestamp.timestamp()) <= _TIME_TOLERANCE_SECONDS)]
            if not len(candidates):
                return None, None, None, "标注时间与附近轨迹记录不一致"
            basis = "time_and_position"

        closest_distance = float(distances[candidates].min())
        nearby = candidates[distances[candidates] <= closest_distance + _AMBIGUITY_METERS]
        contexts = defaultdict(list)
        visit = 0
        previous_index = None
        for index in nearby:
            # Leaving and revisiting the spatial neighborhood is ambiguous at any distance.
            if previous_index is not None and index - previous_index > 1:
                visit += 1
            day_number = next((d["sequence_number"] for d in day_ranges
                               if d["track_start_index"] <= index <= d["track_end_index"]), None)
            contexts[(day_number, points[index].get("segment_index", 0), visit)].append(int(index))
            previous_index = index
        if len(contexts) > 1:
            if not timestamp:
                return None, None, None, "位置被多日或多次经过，缺少消歧依据"
            ranked = sorted((min(abs(times[i] - timestamp.timestamp()) for i in group), group)
                            for group in contexts.values())
            if not np.isfinite(ranked[0][0]) or ranked[1][0] - ranked[0][0] < 60:
                return None, None, None, "标注时间位置存在多个近似归属"
            candidates = np.array(ranked[0][1])
        chosen = min((int(i) for i in candidates), key=lambda i: (float(distances[i]), abs(times[i] - timestamp.timestamp()) if timestamp and np.isfinite(times[i]) else 0))
        return chosen, round(float(distances[chosen]), 1), basis, None

    for marker in state.get("kml_markers") or []:
        name = source_text(marker.get("name"))
        detail = source_text(marker.get("description"))
        if not name and not detail:
            continue
        text = "；".join(_unique([name, detail]))
        index, distance, basis, reason = match_marker(marker, text)
        matches = [(kind, seg) for kind, seg in ranges
                   if index is not None and seg["track_start_index"] <= index <= seg["track_end_index"]]
        matching_pois = poi_indices[_poi_key(marker)]
        poi = pois[matching_pois.popleft()] if matching_pois else None
        is_camp = poi is not None and poi.get("category") == "camp"
        usage = "unknown"
        if is_camp:
            if matches and not _UNCERTAIN_CAMP_USAGE.search(text):
                if any(term in text for term in ("看到", "路过", "途经", "经过", "远望", "午餐", "午休", "休息")):
                    usage = "along_route"
                elif any(kind == "day" for kind, _ in matches) and (
                    _day_number(text) is not None or any(term in text for term in ("当晚", "过夜", "扎营", "宿营"))
                ):
                    usage = "recorded_overnight"
            associations = []
            for kind, seg in sorted(matches, key=lambda item: (item[0] != "day", item[1]["sequence_number"])):
                association = {"scheme_type": kind, "track_start_index": seg["track_start_index"],
                               "track_end_index": seg["track_end_index"], "basis": basis, "distance_meters": distance}
                if kind == "day":
                    association["day_number"] = seg["sequence_number"]
                associations.append(association)
            snapshot = {"status": "pending_verification", "source": "source_marker", "usage": usage,
                        "evidence": text, "associations": associations}
            if not associations:
                snapshot["unresolved_reason"] = reason or "未能关联本次分析范围"
            poi["card_data"] = {**(poi.get("card_data") or {}), "camp_analysis": snapshot}
        if reason:
            if name:
                warnings.append({"level": "warning", "message": "原标注归属待确认", "detail": f"{name}：{reason}"})
            continue
        for kind, seg in matches:
            assigned[(kind, seg["id"])].append({"name": name, "text": text, "index": index,
                                              "camp": is_camp, "usage": usage})

    author_notes = _day_notes(source_text(state.get("source_description")))
    facts = []
    for kind, seg in ranges:
        items = sorted(assigned[(kind, seg["id"])], key=lambda item: item["index"])
        units = [{"id": "statistics", "text": (
            f"{seg['name']}：按轨迹计算约{seg['distance']:g}公里，爬升{seg['elevation_gain']:g}米，"
            f"下降{seg['elevation_loss']:g}米，预计行进{seg['estimated_time']}分钟（算法估计，不含休息）。"
        )}]
        marker_names = _unique([item["name"] for item in items if item["name"] and not item["camp"]])
        if marker_names:
            units.append({"id": "markers", "text": "沿途原始标注（待核验，按轨迹顺序）：" + "、".join(f"“{name}”" for name in marker_names) + "。"})
        if kind == "day" and author_notes.get(seg["sequence_number"]):
            units.append({"id": "source_description", "text": "原作者当日说明（历史记录，待核验）：" + "；".join(author_notes[seg["sequence_number"]])})
        camp_items = [item for item in items if item["camp"]]
        if camp_items:
            labels = {"recorded_overnight": "原记录当晚营地", "along_route": "沿途营地标注", "unknown": "用途待确认的营地标注"}
            units.append({"id": "camps", "text": "；".join(f"{labels[item['usage']]}（待核验）：{item['name']}" for item in camp_items) + "。"})
        notices = _unique([item["text"] for item in items if any(word in item["text"] for word in _NOTICE_WORDS)])
        if kind == "day":
            notices.extend(line for line in author_notes.get(seg["sequence_number"], []) if any(word in line for word in _NOTICE_WORDS))
        seg["description"] = "\n".join(unit["text"] for unit in units)
        seg["notes"] = "\n".join(f"原轨迹提示（待核验）：{text}" for text in _unique(notices)) or None
        facts.append({"scheme_type": kind, "segment_id": seg["id"], "units": units})
    return {"segment_schemes": schemes, "poi_points": pois, "segment_content_facts": facts, "warnings": warnings}
