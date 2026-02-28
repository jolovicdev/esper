import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from blackgeorge import Desk, Job, Worker
from blackgeorge.tools import tool
from dotenv import load_dotenv
from litellm import embedding
import sqlite_vec

from esper import Cardinality, ESPERDatabase, EvidenceEvent, Polarity, PredicateDefinition, SourceType

CHAT_MODEL = "openrouter/google/gemini-3-flash-preview"
EMBEDDING_MODEL = "openrouter/google/gemini-embedding-001"
ESPER_BELIEF_WEIGHT = 0.6
SEPARATOR = "─" * 64
FLAT_VECTOR_DB_PATH = ".blackgeorge/poc_flat_vec.db"
ESPER_VECTOR_DB_PATH = ".blackgeorge/poc_esper_vec.db"


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = embedding(model=EMBEDDING_MODEL, input=texts)
    return [item["embedding"] for item in response.data]


class LiveFlatIndex:
    def __init__(self, db_path: str = ":memory:", reset_on_init: bool = False) -> None:
        if db_path != ":memory:":
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            if reset_on_init and db_file.exists():
                db_file.unlink()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        sqlite_vec.load(self.conn)
        self._embedding_dim: int | None = None
        self._setup_tables()

    def _setup_tables(self) -> None:
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS docs (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL
                )
            """)

    def _ensure_vec_table(self, embedding_dim: int) -> None:
        with self._lock:
            if self._embedding_dim is None:
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0(embedding float[{embedding_dim}])"
                )
                self._embedding_dim = embedding_dim
                return
            if self._embedding_dim != embedding_dim:
                raise ValueError(
                    f"Embedding dimension changed from {self._embedding_dim} to {embedding_dim}"
                )

    def add_documents(self, docs: list[dict[str, str]]) -> None:
        if not docs:
            return
        vectors = embed_texts([doc["text"] for doc in docs])
        self._ensure_vec_table(len(vectors[0]))

        with self._lock:
            self.conn.execute("DELETE FROM docs")
            self.conn.execute("DELETE FROM vec_docs")

            for doc, vector in zip(docs, vectors):
                cur = self.conn.execute(
                    "INSERT INTO docs (doc_id, text) VALUES (?, ?)",
                    (doc["doc_id"], doc["text"]),
                )
                rowid = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO vec_docs (rowid, embedding) VALUES (?, ?)",
                    (rowid, json.dumps(vector)),
                )

    def search(self, query: str, top_k: int = 3, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        if self._embedding_dim is None:
            return []
        query_vec = embed_texts([query])[0]
        if len(query_vec) != self._embedding_dim:
            raise ValueError(
                f"Query embedding dim {len(query_vec)} does not match index dim {self._embedding_dim}"
            )

        with self._lock:
            if allowed_ids is not None:
                if not allowed_ids:
                    return []
                doc_id_placeholders = ",".join("?" for _ in allowed_ids)
                total_docs = int(self.conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"])
                if total_docs == 0:
                    return []
                base_sql = f"""
                    SELECT d.doc_id, d.text, v.distance
                    FROM (
                        SELECT rowid, distance
                        FROM vec_docs
                        WHERE embedding MATCH ? AND k = ?
                    ) v
                    JOIN docs d ON d.rowid = v.rowid
                    WHERE d.doc_id IN ({doc_id_placeholders})
                    ORDER BY v.distance ASC
                    LIMIT ?
                """
                params: list[Any] = [json.dumps(query_vec), total_docs, *sorted(allowed_ids), top_k]
            else:
                base_sql = """
                    SELECT d.doc_id, d.text, v.distance
                    FROM vec_docs v
                    JOIN docs d ON d.rowid = v.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance ASC
                    LIMIT ?
                """
                params = [json.dumps(query_vec), max(top_k, 1), top_k]

            rows = self.conn.execute(base_sql, params).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "text": row["text"],
                "similarity": 1.0 / (1.0 + float(row["distance"])),
                "belief": 0.0,
                "score": 1.0 / (1.0 + float(row["distance"])),
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self.conn.close()


class LiveEsperIndex(LiveFlatIndex):
    USER_ID = "demo"
    SUBJECT = "kb"
    PREDICATE = "FACT_RELEVANCE"

    def __init__(
        self,
        cardinality: Cardinality,
        belief_weight: float = ESPER_BELIEF_WEIGHT,
        db_path: str = ":memory:",
        reset_on_init: bool = False,
    ) -> None:
        super().__init__(db_path=db_path, reset_on_init=reset_on_init)
        if not (0.0 <= belief_weight <= 1.0):
            raise ValueError("belief_weight must be in [0, 1]")
        self.belief_weight = belief_weight
        self.similarity_weight = 1.0 - belief_weight
        self.db = ESPERDatabase(
            predicate_definitions=[PredicateDefinition(self.PREDICATE, cardinality)]
        )

    def ingest(self, doc_id: str, conf_cal: float, episode_id: str, source_type: SourceType) -> None:
        self.db.insert_evidence(
            EvidenceEvent(
                user_id=self.USER_ID,
                subject=self.SUBJECT,
                predicate=self.PREDICATE,
                object=doc_id,
                polarity=Polarity.POSITIVE,
                conf_cal=conf_cal,
                episode_id=episode_id,
                evidence_ts=datetime.now(timezone.utc).isoformat(),
                source_type=source_type,
            )
        )

    def belief(self, doc_id: str) -> float:
        rows = self.db.get_belief(self.USER_ID, self.SUBJECT, self.PREDICATE, object_filter=doc_id)
        if not rows:
            return 0.0
        return float(rows[0]["last_net_estimate"])

    def active_doc_ids(self) -> set[str]:
        rows = self.db.get_belief(self.USER_ID, self.SUBJECT, self.PREDICATE)
        if not rows:
            with self._lock:
                all_rows = self.conn.execute("SELECT doc_id FROM docs").fetchall()
            return {row["doc_id"] for row in all_rows}
        top = rows[0]
        if int(top["is_ambiguous"]) == 1:
            top_score = float(top["last_net_estimate"])
            margin = self.db.ambiguity_margin
            return {
                row["object"]
                for row in rows
                if float(row["last_net_estimate"]) >= (top_score - margin)
            }
        return {top["object"]}

    def search(self, query: str, top_k: int = 3, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        chosen_ids = allowed_ids if allowed_ids is not None else self.active_doc_ids()
        base = super().search(query=query, top_k=max(top_k, len(chosen_ids)), allowed_ids=chosen_ids)
        enriched: list[dict[str, Any]] = []
        for row in base:
            belief = self.belief(row["doc_id"])
            score = (self.similarity_weight * row["similarity"]) + (self.belief_weight * belief)
            enriched.append(
                {
                    **row,
                    "belief": belief,
                    "score": score,
                }
            )
        enriched.sort(key=lambda item: item["score"], reverse=True)
        return enriched[:top_k]

    def close(self) -> None:
        self.db.close()
        super().close()


def run_retrieval_agent(label: str, index: LiveFlatIndex, question: str, top_k: int = 2) -> dict[str, Any]:
    retrieval_trace: list[dict[str, Any]] = []

    @tool(description="Retrieve memory items for the user query.")
    def recall_memory(query: str, top_k: int = 2) -> str:
        rows = index.search(query=query, top_k=top_k)
        retrieval_trace.extend(rows)
        return json.dumps(rows)

    worker = Worker(
        name=f"{label}-agent",
        tools=[recall_memory],
        instructions="Call recall_memory exactly once, then answer in one sentence based only on tool output.",
    )
    desk = Desk(model=CHAT_MODEL, temperature=0.0, max_tokens=350, max_iterations=5, max_tool_calls=3)
    report = desk.run(
        worker,
        Job(
            input=(
                f"Question: {question}\n"
                f"First call recall_memory with JSON arguments "
                f'{{"query": "{question}", "top_k": {top_k}}}.\n'
                "Then answer using only retrieved memory."
            )
        ),
    )
    content = (report.content or "").strip()
    if not retrieval_trace:
        retrieval_trace.extend(index.search(query=question, top_k=top_k))
    if not content:
        content = (
            f"Top retrieved memory is {retrieval_trace[0]['doc_id']}."
            if retrieval_trace
            else "No memory found."
        )
    return {
        "label": label,
        "answer": content,
        "retrieval_trace": retrieval_trace,
    }


def run_metrics_agent(metrics: dict[str, float | int]) -> str:
    @tool(description="Return reinforcement metrics as JSON.")
    def reinforcement_metrics(request: str) -> str:
        _ = request
        return json.dumps(metrics)

    worker = Worker(
        name="metrics-agent",
        tools=[reinforcement_metrics],
        instructions="Call reinforcement_metrics exactly once, then provide a one-sentence conclusion.",
    )
    desk = Desk(model=CHAT_MODEL, temperature=0.0, max_tokens=220, max_iterations=4, max_tool_calls=2)
    report = desk.run(
        worker,
        Job(
            input=(
                'First call reinforcement_metrics with JSON arguments {"request": "fetch"}.\n'
                "Then state whether independent evidence is stronger than correlated evidence."
            )
        ),
    )
    content = (report.content or "").strip()
    if content:
        return content
    if metrics["independent_estimate"] > metrics["correlated_estimate"]:
        return "Independent evidence is stronger than correlated evidence."
    return "Correlated evidence is not weaker than independent evidence."


def print_retrieval_result(result: dict[str, Any]) -> None:
    print(f"\n  [{result['label']}]")
    for i, row in enumerate(result["retrieval_trace"]):
        marker = "  ← TOP" if i == 0 else ""
        print(
            f"    #{i+1} {row['doc_id']:24s} sim={row['similarity']:.3f} "
            f"belief={row['belief']:.3f} score={row['score']:.3f}{marker}"
        )
    print(f"    answer: {result['answer']}")
    if result["retrieval_trace"]:
        print(f"    top_doc: {result['retrieval_trace'][0]['doc_id']}")


def scenario_a_knowledge_update() -> None:
    print(f"\n{SEPARATOR}")
    print("SCENARIO A: Knowledge Update (ONE cardinality, live embeddings + live tool-calling agent)")
    print(SEPARATOR)

    docs = [
        {
            "doc_id": "codename_borealis",
            "text": (
                "Current production codename is Borealis. All systems use Borealis. "
                "Deploy with Borealis. Official codename: Borealis."
            ),
        },
        {
            "doc_id": "codename_atlas",
            "text": "Codename updated to Atlas per memo dated last week.",
        },
        {
            "doc_id": "ops_runbook",
            "text": "Deploy every Friday with health checks and rollback policy.",
        },
    ]
    question = "What is the current production codename?"

    flat = LiveFlatIndex(db_path=FLAT_VECTOR_DB_PATH, reset_on_init=True)
    esper = LiveEsperIndex(
        cardinality=Cardinality.ONE,
        db_path=ESPER_VECTOR_DB_PATH,
        reset_on_init=True,
    )
    try:
        flat.add_documents(docs)
        esper.add_documents(docs)
        esper.ingest("codename_borealis", conf_cal=0.90, episode_id=str(uuid.uuid4()), source_type=SourceType.EXPLICIT)
        esper.ingest("codename_atlas", conf_cal=0.92, episode_id=str(uuid.uuid4()), source_type=SourceType.EXPLICIT)

        flat_result = run_retrieval_agent("Flat", flat, question)
        esper_result = run_retrieval_agent("ESPER", esper, question)

        print_retrieval_result(flat_result)
        print_retrieval_result(esper_result)
        print(
            f"\n  ESPER score blend: "
            f"{esper.similarity_weight:.1f}*similarity + {esper.belief_weight:.1f}*belief "
            "(applied over active contenders)"
        )
        print(f"\n  belief(codename_borealis)={esper.belief('codename_borealis'):.3f}")
        print(f"  belief(codename_atlas)={esper.belief('codename_atlas'):.3f}")
    finally:
        esper.close()
        flat.close()


def scenario_b_correlated_reinforcement() -> None:
    print(f"\n{SEPARATOR}")
    print("SCENARIO B: Correlated Reinforcement (MANY cardinality, live tool-calling summary)")
    print(SEPARATOR)

    predicate = "LANGUAGE_KNOWN"
    defs = [PredicateDefinition(predicate, Cardinality.MANY)]
    now = datetime.now(timezone.utc).isoformat()

    def run_case(episode_ids: list[str]) -> tuple[int, float]:
        db = ESPERDatabase(predicate_definitions=defs)
        try:
            for conf, episode_id in zip([0.85, 0.88, 0.90], episode_ids):
                db.insert_evidence(
                    EvidenceEvent(
                        user_id="demo",
                        subject="user",
                        predicate=predicate,
                        object="Python",
                        polarity=Polarity.POSITIVE,
                        conf_cal=conf,
                        episode_id=episode_id,
                        evidence_ts=now,
                        source_type=SourceType.EXTRACTOR,
                    )
                )
            row = db.get_belief("demo", "user", predicate, "Python")[0]
            return int(row["reinforcement_count"]), float(row["last_net_estimate"])
        finally:
            db.close()

    shared_episode = str(uuid.uuid4())
    correlated_count, correlated_estimate = run_case([shared_episode] * 3)
    independent_count, independent_estimate = run_case([str(uuid.uuid4()) for _ in range(3)])

    metrics = {
        "correlated_reinforcement_count": correlated_count,
        "correlated_estimate": correlated_estimate,
        "independent_reinforcement_count": independent_count,
        "independent_estimate": independent_estimate,
    }
    summary = run_metrics_agent(metrics)

    print(f"  correlated_count={metrics['correlated_reinforcement_count']} estimate={metrics['correlated_estimate']:.3f}")
    print(f"  independent_count={metrics['independent_reinforcement_count']} estimate={metrics['independent_estimate']:.3f}")
    print(
        "  dedup rule: 3 events with one episode_id are max-pooled into 1 independent signal; "
        "3 distinct episode_ids remain 3 independent signals."
    )
    print(f"  metrics_agent answer: {summary}")


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    print("Agentic RAG Compare 2 (live)")
    print(f"chat_model: {CHAT_MODEL}")
    print(f"embedding_model: {EMBEDDING_MODEL}")
    print(f"flat_vector_db: {FLAT_VECTOR_DB_PATH}")
    print(f"esper_vector_db: {ESPER_VECTOR_DB_PATH}")
    scenario_a_knowledge_update()
    scenario_b_correlated_reinforcement()


if __name__ == "__main__":
    main()
