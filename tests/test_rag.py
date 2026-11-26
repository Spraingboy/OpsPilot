"""Pure tests run without Postgres/API key. DB test is opt-in with dedicated DB."""
from rag import chunk_text, lexical_tokens, rrf_fuse

def test_chunks_overlap():
    chunks = chunk_text("a" * 1000, size=500, overlap=80)
    assert len(chunks) == 3
    assert chunks[0][-80:] == chunks[1][:80]

def test_chinese_and_english_tokens():
    tokens = lexical_tokens("VPN认证失败 E1047")
    assert "vpn" in tokens
    assert "认证" in tokens
    assert "证失" in tokens
    assert "e1047" in tokens

def test_rrf():
    a = {"chunk_id": "a", "id": "doc-a"}
    b = {"chunk_id": "b", "id": "doc-b"}
    result = rrf_fuse([[a, b], [b, a]], limit=2)
    assert {x["chunk_id"] for x in result} == {"a", "b"}

def test_empty_text():
    assert chunk_text("  ") == []
