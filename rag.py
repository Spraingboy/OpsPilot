"""Lesson 4: document ingestion, embeddings, tenant-safe hybrid retrieval.

Demo scope: a single allowed_group per document; exact vector KNN;
simple Chinese character bigram/English word lexical retrieval.
"""
import hashlib
import os
import re
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from pgvector import Vector
from pgvector.psycopg import register_vector
from openai import OpenAI

DIM = 1536
MODEL = "text-embedding-v4"

BAILIAN_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

DSN = os.getenv(
    "RAG_DATABASE_URL",
    "postgresql://opspilot:local_only_change_me@localhost:5433/opspilot",
)


def connect():
    conn = psycopg.connect(DSN, row_factory=dict_row)
    register_vector(conn)
    return conn


def init_schema():
    # Install extension before registering pgvector's Python adapter.
    with psycopg.connect(DSN) as bootstrap:
        bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                tenant TEXT NOT NULL,
                id TEXT NOT NULL,
                title TEXT NOT NULL,
                allowed_group TEXT NOT NULL,
                checksum TEXT NOT NULL,
                PRIMARY KEY (tenant, id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                chunk_id TEXT PRIMARY KEY,
                tenant TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                body TEXT NOT NULL,
                tokens TEXT[] NOT NULL,
                embedding vector(1536) NOT NULL,
                embedding_model TEXT NOT NULL,
                UNIQUE (tenant, doc_id, chunk_index),
                FOREIGN KEY (tenant, doc_id)
                   REFERENCES kb_documents (tenant, id)
                   ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS kb_chunks_lex_gin
            ON kb_chunks USING gin (tokens)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS kb_chunks_tenant_doc
            ON kb_chunks (tenant, doc_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS kb_docs_acl
            ON kb_documents (tenant, allowed_group)
        """)


def load_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(
            page.extract_text() or "" for page in PdfReader(str(path)).pages
        )
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    raise ValueError("Only .txt, .md, .pdf and .docx are supported")


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def lexical_tokens(text: str) -> list[str]:
    """English words and Chinese character bigrams (baseline, not jieba)."""
    items = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower())
    out = set()
    for item in items:
        if re.fullmatch(r"[\u4e00-\u9fff]+", item):
            if len(item) == 1:
                out.add(item)
            else:
                out.update(item[i:i + 2] for i in range(len(item) - 1))
        else:
            out.add(item)
    return sorted(out)



def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    使用阿里云百炼 text-embedding-v4 生成文本向量
    """

    if not texts:
        return []

    # 获取阿里云百炼 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "未配置 DASHSCOPE_API_KEY，请检查环境变量"
        )

    # 创建阿里云百炼客户端
    client = OpenAI(
        api_key=api_key,
        base_url=BAILIAN_BASE_URL,
        timeout=60,
        max_retries=2,
    )

    vectors = []

    # 每批处理10条文本
    batch_size = 10

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        result = client.embeddings.create(
            model=MODEL,
            input=batch,
            dimensions=DIM,
            encoding_format="float",
        )

        # 按输入顺序提取向量
        batch_vectors = [
            item.embedding
            for item in sorted(
                result.data,
                key=lambda x: x.index
            )
        ]

        # 检查返回数量
        if len(batch_vectors) != len(batch):
            raise ValueError(
                "Embedding 返回数量不匹配"
            )

        # 检查向量维度
        if any(len(v) != DIM for v in batch_vectors):
            raise ValueError(
                "Embedding dimension mismatch"
            )

        vectors.extend(batch_vectors)

    return vectors



def upsert_document(
    *,
    tenant: str,
    doc_id: str,
    title: str,
    allowed_group: str,
    content: str,
    embedder=embed_texts,
) -> int:
    """Embeds before a transaction; changes old revision atomically."""
    if not all(x.strip() for x in (tenant, doc_id, title, allowed_group)):
        raise ValueError("tenant, doc_id, title, allowed_group required")
    chunks = chunk_text(content)
    if not chunks:
        raise ValueError("Cannot index empty document")
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    embeddings = embedder([f"{title}\n{part}" for part in chunks])
    if len(embeddings) != len(chunks) or any(len(v) != DIM for v in embeddings):
        raise ValueError("Embedding count or dimension mismatch")

    with connect() as conn:
        conn.execute("""
            INSERT INTO kb_documents (tenant, id, title, allowed_group, checksum)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (tenant, id) DO UPDATE SET
                title = EXCLUDED.title,
                allowed_group = EXCLUDED.allowed_group,
                checksum = EXCLUDED.checksum
        """, (tenant, doc_id, title, allowed_group, checksum))
        conn.execute(
            "DELETE FROM kb_chunks WHERE tenant=%s AND doc_id=%s",
            (tenant, doc_id),
        )
        for index, (part, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{tenant}:{doc_id}:c{index + 1}"
            conn.execute("""
                INSERT INTO kb_chunks
                  (chunk_id, tenant, doc_id, chunk_index, body,
                   tokens, embedding, embedding_model)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                chunk_id, tenant, doc_id, index, part,
                lexical_tokens(title + " " + part), Vector(vector), MODEL,
            ))
    return len(chunks)


def delete_document(tenant: str, doc_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM kb_documents WHERE tenant=%s AND id=%s",
            (tenant, doc_id),
        )


def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60, limit: int = 4):
    scores, details = {}, {}
    for result_list in ranked_lists:
        for rank, doc in enumerate(result_list, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)
            details[cid] = doc
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [
        {**details[cid], "rrf_score": scores[cid]}
        for cid in ordered[:limit]
    ]


def search_chunks(
    question: str,
    tenant: str,
    groups: list[str],
    top_k: int = 4,
    embedder=embed_texts,
) -> list[dict]:
    if not tenant or not groups or not question.strip():
        return []
    query_vector = embedder([question])[0]
    if len(query_vector) != DIM:
        raise ValueError("Wrong embedding dimension")
    query_tokens = lexical_tokens(question)
    candidate_k = max(10, top_k * 5)

    # CRITICAL: tenant and ACL predicates are inside BOTH SQL searches.
    # No unauthorized chunks should ever leave the database.
    with connect() as conn:
        vector_rows = conn.execute("""
            SELECT c.chunk_id, c.doc_id AS id, d.title, c.body,
                   (1 - (c.embedding <=> %s)) AS similarity
            FROM kb_chunks c
            JOIN kb_documents d
              ON d.tenant=c.tenant AND d.id=c.doc_id
            WHERE c.tenant=%s
              AND d.allowed_group = ANY(%s::text[])
              AND c.embedding_model=%s
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """, (
            Vector(query_vector), tenant, groups, MODEL,
            Vector(query_vector), candidate_k,
        )).fetchall()

        lexical_rows = []
        if query_tokens:
            lexical_rows = conn.execute("""
                SELECT c.chunk_id, c.doc_id AS id, d.title, c.body,
                  (
                    SELECT COUNT(*)
                    FROM unnest(c.tokens) tok
                    WHERE tok = ANY(%s::text[])
                  ) AS matches
                FROM kb_chunks c
                JOIN kb_documents d
                  ON d.tenant=c.tenant AND d.id=c.doc_id
                WHERE c.tenant=%s
                  AND d.allowed_group = ANY(%s::text[])
                  AND c.tokens && %s::text[]
                ORDER BY matches DESC, c.chunk_id
                LIMIT %s
            """, (
                query_tokens, tenant, groups, query_tokens, candidate_k,
            )).fetchall()

    # Heuristic safeguard; tune on the evaluation set. Not a guarantee of relevance.
    min_similarity = 0.35
    vector_rows = [
        r for r in vector_rows if float(r["similarity"]) >= min_similarity
    ]
    fused = rrf_fuse([vector_rows, lexical_rows], limit=top_k)
    return [
        {"id": r["id"], "title": r["title"],
         "body": r["body"], "chunk_id": r["chunk_id"]}
        for r in fused
    ]
