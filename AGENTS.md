# KML Agent Service AI Instructions

This file guides AI assistants working in `kml-agent-service/`.

## Role

`kml-agent-service` is the asynchronous KML/GPX analysis worker. It accepts analysis tasks, runs a LangGraph workflow, exposes task status, and calls back `walkbg` with structured results.

It does not own main business data and must not write directly to the backend database.

## Read Before Editing

1. `../AGENTS.md`
2. `../DEVELOPMENT_PARADIGM.md`
3. `../AI_DEVELOPMENT.md`
4. `README.md` if present
5. `app/main.py`
6. `app/models/request.py`
7. `app/models/response.py`
8. `app/models/state.py`
9. `app/agents/analysis_workflow.py`
10. `app/services/callback_service.py`

If the change modifies callback payloads, task status, progress semantics, or analysis output shape, read or create the relevant root OpenSpec change first.

## Architecture

```text
FastAPI endpoint
  -> task_service
  -> AnalysisWorkflow
  -> agents/*
  -> callback_service
```

## Rules

- Request and response models must use Pydantic.
- Workflow state must flow through `AgentState`.
- Each agent should have clear input, output, progress, and failure behavior.
- Callback payloads must conform to `agent-data-contract`.
- Failures must update task status and be queryable through `GET /api/v1/tasks/{task_id}`.
- LLM-generated content must have deterministic fallback behavior.
- Do not hide failures in logs only.
- Do not introduce network calls or model providers without configuration and failure handling.

## Verification

Use:

```bash
pytest
```

For endpoint changes, add or update FastAPI tests where practical. For workflow changes, add focused tests around request/response schema and state transitions.

