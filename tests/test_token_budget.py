"""The token budget caps the remote-teacher loop and shows up in live status."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from crucible.server.app import create_app


def test_token_budget_stops_teacher_loop(tmp_path: Path) -> None:
    client = TestClient(create_app(run_dir=tmp_path / "ui"))
    resp = client.post(
        "/api/run",
        json={
            "backend": "simulator",
            "time_budget_minutes": 1,
            "token_budget_m": 0.5,
            "continuous_mode": True,
            "async_mode": False,
        },
    )
    assert resp.status_code == 200, resp.text
    assert "token_budget_reached" in resp.text
    budget = client.get("/api/live-status").json()["live_budget"]
    assert budget["token_budget_m"] == 0.5
    assert budget["spent_tokens"] >= 0.5e6
    assert budget["teacher_round"] == 0
