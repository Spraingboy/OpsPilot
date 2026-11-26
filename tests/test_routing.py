
from types import SimpleNamespace

import pytest
import app


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app, "DB_PATH", str(tmp_path / "routing.db")
    )
    monkeypatch.delenv(
        "OPENAI_API_KEY", raising=False
    )
    app.init_db()


def make_state(operation="answer"):
    return {
        "message": "VPN 已经多次失败，请 IT 帮忙",
        "operation": operation,
        "user": {
            "id": "alice",
            "tenant": "a",
            "groups": ["employee"],
            "role": "employee",
        }
    }


def test_explicit_ticket_route():
    state = {
        "operation": "draft_ticket",
        "proposed_action": "answer",
    }
    assert app.route_operation(state) == "ticket"


def test_model_cannot_execute_ticket():
    state = {
        "operation": "answer",
        "proposed_action": "draft_ticket",
    }
    assert app.route_operation(state) == "suggest_ticket"


def test_invalid_model_action_rejected():
    state = {
        "operation": "answer",
        "proposed_action": "delete_users",
    }
    with pytest.raises(ValueError):
        app.route_operation(state)


def test_normal_answer_path():
    state = {
        "operation": "answer",
        "proposed_action": "answer",
    }
    assert app.route_operation(state) == "answer"


def test_llm_suggestion_has_no_side_effect(
    monkeypatch,
):
    # 模拟配置了真实 LLM
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")

    # 用 Mock 替代真实网络调用
    def fake_classify(message):
        return SimpleNamespace(
            action="draft_ticket",
            reason="用户需要人工 IT 支持"
        )

    monkeypatch.setattr(
        app, "classify_intent", fake_classify
    )

    events = list(
        app.agent_graph.stream(
            make_state(),
            stream_mode="updates"
        )
    )

    node_names = [
        node
        for event in events
        for node in event
    ]

    assert node_names == [
        "retrieve",
        "classify",
        "suggest_ticket",
    ]

    # 最关键的安全断言：数据库没有新增工单
    with app.db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tickets"
        ).fetchone()[0]

    assert count == 0


def test_explicit_ticket_creation():
    result = app.agent_graph.invoke(
        make_state(operation="draft_ticket")
    )

    assert result["ticket_id"] is not None

    with app.db() as conn:
        ticket = conn.execute(
            "SELECT status FROM tickets WHERE id = ?",
            (result["ticket_id"],)
        ).fetchone()

    assert ticket is not None
    assert ticket["status"] == "PENDING_APPROVAL"