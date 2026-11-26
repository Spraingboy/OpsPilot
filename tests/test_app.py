
import app
from fastapi.testclient import TestClient
import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    # 每条测试使用独立数据库，避免污染正式演示数据
    monkeypatch.setattr(
        app, "DB_PATH", str(tmp_path / "test.db")
    )
    app.init_db()


client = TestClient(app.api)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unauthorized():
    response = client.post(
        "/v1/agent/run",
        json={
            "message": "VPN 认证失败",
            "operation": "answer"
        }
    )
    assert response.status_code == 401


def test_employee_vpn_search(monkeypatch):
    # 禁用外部模型调用，保证测试可重复
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post(
        "/v1/agent/run",
        headers={"X-Demo-Token": "demo-employee-a"},
        json={
            "message": "VPN 认证失败",
            "operation": "answer"
        }
    )

    assert response.status_code == 200

    data = response.json()
    source_ids = [s["id"] for s in data["sources"]]

    assert "A-VPN-01" in source_ids
    assert "A-SEC-01" not in source_ids
    assert "B-VPN-01" not in source_ids


def test_ticket_requires_approval():
    response = client.post(
        "/v1/agent/run",
        headers={"X-Demo-Token": "demo-employee-a"},
        json={
            "message": "我的 VPN 无法连接，请创建工单",
            "operation": "draft_ticket"
        }
    )

    assert response.status_code == 200
    ticket_id = response.json()["ticket_id"]
    assert ticket_id is not None

    denied = client.post(
        f"/v1/tickets/{ticket_id}/decision",
        headers={"X-Demo-Token": "demo-employee-a"},
        json={"decision": "approve"}
    )
    assert denied.status_code == 403

    approved = client.post(
        f"/v1/tickets/{ticket_id}/decision",
        headers={"X-Demo-Token": "demo-it-a"},
        json={"decision": "approve"}
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    assert approved.json()["executed"] is False