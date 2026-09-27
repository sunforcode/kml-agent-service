"""Submission identity tests: direct endpoints, no workflow, HTTP or model calls."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from app import main
from app.models.request import KmlAnalysisRequest
from app.services.task_service import TaskManager, TaskStatus


@pytest.fixture
def manager(monkeypatch):
    tasks = TaskManager()
    monkeypatch.setattr(main, "task_manager", tasks)
    monkeypatch.setattr(main, "workflow", object())
    return tasks


def request(**overrides):
    return KmlAnalysisRequest.model_validate({
        "task_id": "task-preallocated",
        "route_id": "route-1",
        "kml_source": "/static/kml-upload/input.kml",
        "kml_content": "<kml></kml>",
        **overrides,
    })


def submit(model):
    background = BackgroundTasks()
    response = asyncio.run(main.submit_analysis(model, background))
    # Deliberately do not execute BackgroundTasks: submission must not contact external services.
    return response, background.tasks


def test_preallocated_identity_reaches_task_and_runner(manager):
    response, scheduled = submit(request())
    assert response.task_id == "task-preallocated"
    assert manager.get_task(response.task_id)["request"]["route_id"] == "route-1"
    assert len(scheduled) == 1
    assert scheduled[0].args[0] == response.task_id


def test_identical_pending_request_reuses_task_without_second_schedule(manager):
    first, first_jobs = submit(request())
    second, second_jobs = submit(request(enable_content_generation=True))
    assert second.task_id == first.task_id
    assert len(first_jobs) == 1
    assert second_jobs == []
    assert len(manager.get_all_tasks()) == 1


@pytest.mark.parametrize("status", [TaskStatus.PROCESSING, TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_replay_preserves_existing_status_and_does_not_restart(manager, status):
    first, _ = submit(request())
    manager.update_task_status(first.task_id, status, progress=45)
    if status == TaskStatus.COMPLETED:
        manager.set_task_result(first.task_id, {"result": "preserved"})
    if status == TaskStatus.FAILED:
        manager.set_task_error(first.task_id, "callback undelivered")
    snapshot = dict(manager.get_task(first.task_id))

    replay, jobs = submit(request())

    assert replay.task_id == first.task_id
    assert replay.status == status.value
    assert jobs == []
    assert manager.get_task(first.task_id) == snapshot


@pytest.mark.parametrize("change", [
    {"route_id": "route-other"},
    {"kml_content": "<kml><Document/></kml>"},
    {"kml_source": "https://example.test/other.kml"},
    {"enable_content_generation": False},
    {"poi_search_radius": 1000},
])
def test_same_identity_different_request_is_conflict_without_mutation(manager, change):
    first, _ = submit(request())
    snapshot = dict(manager.get_task(first.task_id))
    background = BackgroundTasks()

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.submit_analysis(request(**change), background))

    assert error.value.status_code == 409
    assert background.tasks == []
    assert len(manager.get_all_tasks()) == 1
    assert manager.get_task(first.task_id) == snapshot


def test_legacy_request_without_identity_still_creates_independent_tasks(manager):
    first, first_jobs = submit(request(task_id=None))
    second, second_jobs = submit(request(task_id=None))
    assert first.task_id != second.task_id
    assert len(first_jobs) == len(second_jobs) == 1


@pytest.mark.parametrize("identity", ["", "   ", "a" * 65])
def test_invalid_explicit_identity_is_rejected(identity):
    with pytest.raises(ValidationError):
        request(task_id=identity)


def test_two_event_loops_cannot_schedule_same_identity_twice(manager):
    barrier = Barrier(2)

    def concurrent_submit():
        barrier.wait(timeout=5)
        return submit(request())

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: concurrent_submit(), range(2)))

    assert {response.task_id for response, _ in results} == {"task-preallocated"}
    assert sum(len(jobs) for _, jobs in results) == 1
    assert len(manager.get_all_tasks()) == 1


def test_cleanup_cannot_reopen_preallocated_identity_in_same_process(manager):
    first, _ = submit(request())
    manager.set_task_error(first.task_id, "closed")
    manager.get_task(first.task_id)["created_at"] = "2000-01-01T00:00:00"
    manager.cleanup_old_tasks()

    replay, jobs = submit(request())

    assert replay.task_id == first.task_id
    assert replay.status == "failed"
    assert jobs == []
