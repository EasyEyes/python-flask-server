import importlib
import os

import pytest


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("EASYEYES_ALLOWED_ORIGINS", "https://speaker.easyeyes.app")
    monkeypatch.delenv("EASYEYES_DIAGNOSTICS_TOKEN", raising=False)

    import app as app_module

    app_module = importlib.reload(app_module)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def test_diagnostics_are_disabled_without_a_server_token(client):
    memory_response = client.post("/memory")
    snapshot_response = client.get("/snapshot")

    assert memory_response.status_code == 404
    assert snapshot_response.status_code == 404


def test_diagnostics_require_the_configured_bearer_token(client, monkeypatch):
    monkeypatch.setenv("EASYEYES_DIAGNOSTICS_TOKEN", "test-diagnostics-token")

    unauthorized_response = client.post("/memory")
    authorized_response = client.post(
        "/memory",
        headers={"Authorization": "Bearer test-diagnostics-token"},
    )

    assert unauthorized_response.status_code == 401
    assert authorized_response.status_code == 200
    assert "memory" in authorized_response.json


def test_cors_allows_configured_origin_but_not_arbitrary_origin(client):
    allowed_response = client.options(
        "/task/mls",
        headers={
            "Origin": "https://speaker.easyeyes.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied_response = client.options(
        "/task/mls",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert (
        allowed_response.headers["Access-Control-Allow-Origin"]
        == "https://speaker.easyeyes.app"
    )
    assert "Access-Control-Allow-Origin" not in denied_response.headers


def test_request_body_limit_is_enforced(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 128

    response = client.post(
        "/task/mls",
        data=b"{" + b'"payload":"' + (b"x" * 256) + b'"}',
        content_type="application/json",
    )

    assert response.status_code == 413


def test_unknown_tasks_and_unsupported_content_types_have_error_statuses(client):
    unknown_task_response = client.post(
        "/task/not-a-task", json={"payload": []}
    )
    unsupported_type_response = client.post(
        "/task/mls", data="payload", content_type="text/plain"
    )

    assert unknown_task_response.status_code == 404
    assert unsupported_type_response.status_code == 415


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        '{"length": NaN, "amplitude": 1}',
    ],
)
def test_task_payloads_require_an_object_with_finite_numbers(client, body):
    response = client.post(
        "/task/mls",
        data=body,
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json == {"error": "Invalid JSON task payload"}
