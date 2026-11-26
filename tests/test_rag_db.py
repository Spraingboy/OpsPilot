"""Opt-in integration tests against an isolated disposable PostgreSQL database.

Set RAG_TEST_DATABASE_URL to a DIFFERENT empty database from your dev database.
"""
import os
import uuid
import pytest

if "RAG_TEST_DATABASE_URL" not in os.environ:
    pytest.skip("Set RAG_TEST_DATABASE_URL for isolated database tests", allow_module_level=True)

import rag

@pytest.fixture
def isolated_tenant(monkeypatch):
    monkeypatch.setattr(rag, "DSN", os.environ["RAG_TEST_DATABASE_URL"])
    rag.init_schema()
    tenant_a = "test-a-" + uuid.uuid4().hex[:12]
    tenant_b = "test-b-" + uuid.uuid4().hex[:12]
    yield tenant_a, tenant_b
    with rag.connect() as conn:
        conn.execute(
            "DELETE FROM kb_documents WHERE tenant=ANY(%s::text[])",
            ([tenant_a, tenant_b],),
        )

def fake_embed(texts):
    # Deterministic, same-length vectors for permission tests.
    return [[1.0] + [0.0] * (rag.DIM - 1) for _ in texts]

def test_tenant_group_and_revision(isolated_tenant):
    a, b = isolated_tenant
    rag.upsert_document(
        tenant=a, doc_id="public", title="VPN",
        allowed_group="employee", content="VPN 认证失败处理",
        embedder=fake_embed,
    )
    rag.upsert_document(
        tenant=a, doc_id="secret", title="admin",
        allowed_group="it", content="管理员权限规则",
        embedder=fake_embed,
    )
    rag.upsert_document(
        tenant=b, doc_id="other", title="B企业",
        allowed_group="employee", content="B 企业 VPN",
        embedder=fake_embed,
    )
    results = rag.search_chunks(
        "VPN", a, ["employee"], embedder=fake_embed
    )
    assert {r["id"] for r in results} == {"public"}
    assert {r["id"] for r in rag.search_chunks(
        "管理员", a, ["it"], embedder=fake_embed
    )} == {"secret"}
    assert all(r["id"] != "other" for r in results)
    assert rag.search_chunks("VPN", a, [], embedder=fake_embed) == []

    # Upsert replaces a revision, not accumulates old chunks.
    rag.upsert_document(
        tenant=a, doc_id="public", title="VPN新版",
        allowed_group="employee", content="全新内容 XYZ",
        embedder=fake_embed,
    )
    updated = rag.search_chunks(
        "XYZ", a, ["employee"], embedder=fake_embed
    )
    assert len(updated) == 1
    assert updated[0]["body"] == "全新内容 XYZ"
    rag.delete_document(a, "public")
    assert all(r["id"] != "public" for r in rag.search_chunks(
        "XYZ", a, ["employee"], embedder=fake_embed
    ))
