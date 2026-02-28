import sqlite3
import uuid
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

EPSILON = 1e-9
DEFAULT_SOFT_OVERWRITE_KAPPA = 2.0
DEFAULT_SOFT_OVERWRITE_TAU_SECONDS = 3 * 24 * 3600
DEFAULT_AMBIGUITY_MARGIN = 0.05


class SourceType(Enum):
    TOOL = "tool"
    EXPLICIT = "explicit"
    EXTRACTOR = "extractor"
    RULE = "rule"
    CLASSIFIER = "classifier"


class Polarity(Enum):
    POSITIVE = 1
    NEGATIVE = -1


class Cardinality(Enum):
    ONE = "ONE"
    MANY = "MANY"


SOURCE_TYPE_RELIABILITY_WEIGHTS = {
    SourceType.TOOL: 1.0,
    SourceType.EXPLICIT: 1.0,
    SourceType.RULE: 0.8,
    SourceType.CLASSIFIER: 0.6,
    SourceType.EXTRACTOR: 0.5,
}


def resolve_reliability_weight(source_type: SourceType) -> float:
    return SOURCE_TYPE_RELIABILITY_WEIGHTS[source_type]


@dataclass
class EvidenceEvent:
    user_id: str
    subject: str
    predicate: str
    object: str
    polarity: Polarity
    conf_cal: float
    episode_id: str
    evidence_ts: str
    source_type: SourceType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class PredicateDefinition:
    predicate: str
    cardinality: Cardinality


class ESPERDatabase:
    def __init__(
        self,
        db_path: str = ":memory:",
        predicate_definitions: Optional[list[PredicateDefinition]] = None,
        soft_overwrite_kappa: float = DEFAULT_SOFT_OVERWRITE_KAPPA,
        soft_overwrite_tau_seconds: float = DEFAULT_SOFT_OVERWRITE_TAU_SECONDS,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    ):
        if soft_overwrite_kappa <= 0:
            raise ValueError("soft_overwrite_kappa must be > 0")
        if soft_overwrite_tau_seconds <= 0:
            raise ValueError("soft_overwrite_tau_seconds must be > 0")
        if not (0.0 <= ambiguity_margin <= 1.0):
            raise ValueError("ambiguity_margin must be in [0, 1]")
        self.conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.predicate_registry = {p.predicate: p for p in (predicate_definitions or [])}
        self.soft_overwrite_kappa = soft_overwrite_kappa
        self.soft_overwrite_tau_seconds = soft_overwrite_tau_seconds
        self.ambiguity_margin = ambiguity_margin
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS evidence_ledger (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                subject      TEXT NOT NULL,
                predicate    TEXT NOT NULL,
                object       TEXT NOT NULL,
                polarity     INTEGER NOT NULL CHECK (polarity IN (1, -1)),
                conf_cal     REAL NOT NULL CHECK (conf_cal >= 0 AND conf_cal <= 1),
                episode_id   TEXT NOT NULL,
                evidence_ts  TEXT NOT NULL,
                source_type  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ledger_key
                ON evidence_ledger(user_id, subject, predicate);

            CREATE TABLE IF NOT EXISTS fact_agg (
                user_id             TEXT NOT NULL,
                subject             TEXT NOT NULL,
                predicate           TEXT NOT NULL,
                object              TEXT NOT NULL,
                polarity            INTEGER NOT NULL,
                reinforcement_count INTEGER NOT NULL DEFAULT 0,
                last_net_estimate   REAL NOT NULL DEFAULT 0.0,
                last_update_ts      TEXT NOT NULL,
                log_accumulator     REAL NOT NULL DEFAULT 0.0,
                negative_log_accumulator REAL NOT NULL DEFAULT 0.0,
                is_ambiguous        INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, subject, predicate, object)
            );

            CREATE TABLE IF NOT EXISTS episode_contributions (
                user_id    TEXT NOT NULL,
                subject    TEXT NOT NULL,
                predicate  TEXT NOT NULL,
                object     TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                p_e        REAL NOT NULL,
                w_e        REAL NOT NULL,
                PRIMARY KEY (user_id, subject, predicate, object, episode_id)
            );

            CREATE TABLE IF NOT EXISTS episode_contributions_neg (
                user_id    TEXT NOT NULL,
                subject    TEXT NOT NULL,
                predicate  TEXT NOT NULL,
                object     TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                n_e        REAL NOT NULL,
                w_e        REAL NOT NULL,
                PRIMARY KEY (user_id, subject, predicate, object, episode_id)
            );
        """)
        self._ensure_fact_agg_column("negative_log_accumulator", "REAL NOT NULL DEFAULT 0.0")

    def _ensure_fact_agg_column(self, column_name: str, column_definition: str):
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(fact_agg)").fetchall()
        }
        if column_name not in columns:
            self.conn.execute(
                f"ALTER TABLE fact_agg ADD COLUMN {column_name} {column_definition}"
            )

    def _resolve_cardinality(self, predicate: str) -> Cardinality:
        definition = self.predicate_registry.get(predicate)
        return definition.cardinality if definition else Cardinality.MANY

    def _clamp_score(self, score: float) -> float:
        return min(1.0 - EPSILON, max(0.0, score))

    def _parse_event_ts(self, ts: str) -> datetime:
        parsed = datetime.fromisoformat(ts)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _current_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _apply_decay(self, conf_cal: float, evidence_ts: str, decay_lambda: float) -> float:
        if decay_lambda < 0:
            raise ValueError("decay_lambda must be >= 0")
        if decay_lambda == 0:
            return self._clamp_score(conf_cal)
        event_dt = self._parse_event_ts(evidence_ts)
        age_seconds = max(0.0, (self._current_utc() - event_dt).total_seconds())
        return self._clamp_score(conf_cal * math.exp(-decay_lambda * age_seconds))

    def insert_evidence(self, event: EvidenceEvent, decay_lambda: float = 0.0):
        if not (0.0 <= event.conf_cal <= 1.0):
            raise ValueError(f"conf_cal must be in [0, 1], got {event.conf_cal}")

        w_e = resolve_reliability_weight(event.source_type)
        cardinality = self._resolve_cardinality(event.predicate)
        s_i = self._apply_decay(event.conf_cal, event.evidence_ts, decay_lambda)

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("""
                INSERT INTO evidence_ledger
                    (id, user_id, subject, predicate, object, polarity,
                     conf_cal, episode_id, evidence_ts, source_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id, event.user_id, event.subject, event.predicate,
                event.object, event.polarity.value, event.conf_cal,
                event.episode_id, event.evidence_ts, event.source_type.value
            ))

            if cardinality == Cardinality.MANY:
                self._project_many_cardinality(event, s_i, w_e)
            else:
                self._project_one_cardinality(event, s_i)

            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _project_many_cardinality(self, event: EvidenceEvent, s_i: float, w_e: float):
        composite_key = (event.user_id, event.subject, event.predicate, event.object)

        current_fact_row = self.conn.execute("""
            SELECT log_accumulator, negative_log_accumulator, reinforcement_count FROM fact_agg
            WHERE user_id=? AND subject=? AND predicate=? AND object=?
        """, composite_key).fetchone()

        current_log_accumulator = current_fact_row["log_accumulator"] if current_fact_row else 0.0
        current_negative_log_accumulator = (
            current_fact_row["negative_log_accumulator"] if current_fact_row else 0.0
        )
        current_reinforcement_count = current_fact_row["reinforcement_count"] if current_fact_row else 0

        if event.polarity == Polarity.POSITIVE:
            updated_log_accumulator, reinforcement_delta = self._apply_positive_episode_update(
                composite_key=composite_key,
                episode_id=event.episode_id,
                score=s_i,
                reliability_weight=w_e,
                current_log_accumulator=current_log_accumulator,
            )
            updated_negative_log_accumulator = current_negative_log_accumulator
        else:
            updated_negative_log_accumulator, reinforcement_delta = self._apply_negative_episode_update(
                composite_key=composite_key,
                episode_id=event.episode_id,
                score=s_i,
                reliability_weight=w_e,
                current_log_accumulator=current_negative_log_accumulator,
            )
            updated_log_accumulator = current_log_accumulator

        updated_reinforcement_count = current_reinforcement_count + reinforcement_delta
        positive_estimate = 1.0 - math.exp(updated_log_accumulator)
        negative_estimate = 1.0 - math.exp(updated_negative_log_accumulator)
        updated_net_estimate = self._clamp_score(max(0.0, positive_estimate - negative_estimate))

        self.conn.execute("""
            INSERT INTO fact_agg
                (user_id, subject, predicate, object, polarity,
                 reinforcement_count, last_net_estimate, last_update_ts, log_accumulator, negative_log_accumulator)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, subject, predicate, object) DO UPDATE SET
                polarity             = excluded.polarity,
                reinforcement_count = excluded.reinforcement_count,
                last_net_estimate   = excluded.last_net_estimate,
                last_update_ts      = excluded.last_update_ts,
                log_accumulator     = excluded.log_accumulator,
                negative_log_accumulator = excluded.negative_log_accumulator
        """, (
            event.user_id, event.subject, event.predicate, event.object,
            event.polarity.value, updated_reinforcement_count, updated_net_estimate,
            self._current_utc().isoformat(), updated_log_accumulator, updated_negative_log_accumulator
        ))

    def _apply_positive_episode_update(
        self,
        composite_key: tuple[str, str, str, str],
        episode_id: str,
        score: float,
        reliability_weight: float,
        current_log_accumulator: float,
    ) -> tuple[float, int]:
        existing_episode = self.conn.execute(
            """
            SELECT p_e, w_e FROM episode_contributions
            WHERE user_id=? AND subject=? AND predicate=? AND object=? AND episode_id=?
            """,
            (*composite_key, episode_id),
        ).fetchone()

        if existing_episode is not None:
            stored_score = existing_episode["p_e"]
            stored_weight = existing_episode["w_e"]
            updated_score = max(stored_score, score)
            if updated_score > stored_score:
                current_log_accumulator = (
                    current_log_accumulator
                    - (stored_weight * math.log(1.0 - stored_score))
                    + (reliability_weight * math.log(1.0 - updated_score))
                )
                self.conn.execute(
                    """
                    UPDATE episode_contributions SET p_e=?, w_e=?
                    WHERE user_id=? AND subject=? AND predicate=? AND object=? AND episode_id=?
                    """,
                    (updated_score, reliability_weight, *composite_key, episode_id),
                )
            return current_log_accumulator, 0

        current_log_accumulator += reliability_weight * math.log(1.0 - score)
        self.conn.execute(
            """
            INSERT INTO episode_contributions
                (user_id, subject, predicate, object, episode_id, p_e, w_e)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (*composite_key, episode_id, score, reliability_weight),
        )
        return current_log_accumulator, 1

    def _apply_negative_episode_update(
        self,
        composite_key: tuple[str, str, str, str],
        episode_id: str,
        score: float,
        reliability_weight: float,
        current_log_accumulator: float,
    ) -> tuple[float, int]:
        existing_episode = self.conn.execute(
            """
            SELECT n_e, w_e FROM episode_contributions_neg
            WHERE user_id=? AND subject=? AND predicate=? AND object=? AND episode_id=?
            """,
            (*composite_key, episode_id),
        ).fetchone()

        if existing_episode is not None:
            stored_score = existing_episode["n_e"]
            stored_weight = existing_episode["w_e"]
            updated_score = max(stored_score, score)
            if updated_score > stored_score:
                current_log_accumulator = (
                    current_log_accumulator
                    - (stored_weight * math.log(1.0 - stored_score))
                    + (reliability_weight * math.log(1.0 - updated_score))
                )
                self.conn.execute(
                    """
                    UPDATE episode_contributions_neg SET n_e=?, w_e=?
                    WHERE user_id=? AND subject=? AND predicate=? AND object=? AND episode_id=?
                    """,
                    (updated_score, reliability_weight, *composite_key, episode_id),
                )
            return current_log_accumulator, 0

        current_log_accumulator += reliability_weight * math.log(1.0 - score)
        self.conn.execute(
            """
            INSERT INTO episode_contributions_neg
                (user_id, subject, predicate, object, episode_id, n_e, w_e)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (*composite_key, episode_id, score, reliability_weight),
        )
        return current_log_accumulator, 1

    def _project_one_cardinality(self, event: EvidenceEvent, s_new: float):
        predicate_key = (event.user_id, event.subject, event.predicate)
        event_datetime = self._parse_event_ts(event.evidence_ts)
        event_ts = event_datetime.isoformat()

        if event.polarity == Polarity.NEGATIVE:
            current_row = self.conn.execute("""
                SELECT last_net_estimate, reinforcement_count FROM fact_agg
                WHERE user_id=? AND subject=? AND predicate=? AND object=?
            """, (*predicate_key, event.object)).fetchone()
            current_estimate = current_row["last_net_estimate"] if current_row else 0.0
            current_count = current_row["reinforcement_count"] if current_row else 0
            suppressed_estimate = self._clamp_score(current_estimate * (1.0 - s_new))
            self.conn.execute("""
                INSERT INTO fact_agg
                    (user_id, subject, predicate, object, polarity,
                     reinforcement_count, last_net_estimate, last_update_ts, log_accumulator, negative_log_accumulator)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0)
                ON CONFLICT(user_id, subject, predicate, object) DO UPDATE SET
                    polarity             = excluded.polarity,
                    reinforcement_count = excluded.reinforcement_count,
                    last_net_estimate   = excluded.last_net_estimate,
                    last_update_ts      = excluded.last_update_ts
            """, (
                event.user_id,
                event.subject,
                event.predicate,
                event.object,
                event.polarity.value,
                current_count + 1,
                suppressed_estimate,
                event_ts,
            ))
            self._update_ambiguity(predicate_key)
            return

        conflicting_incumbents = self.conn.execute("""
            SELECT object, last_net_estimate, last_update_ts FROM fact_agg
            WHERE user_id=? AND subject=? AND predicate=? AND object != ?
            ORDER BY last_net_estimate DESC
        """, (*predicate_key, event.object)).fetchall()

        for incumbent_row in conflicting_incumbents:
            s_old = incumbent_row["last_net_estimate"]
            incumbent_datetime = self._parse_event_ts(incumbent_row["last_update_ts"])
            if event_datetime <= incumbent_datetime:
                recency_factor = 0.0
            else:
                delta_seconds = (event_datetime - incumbent_datetime).total_seconds()
                recency_factor = math.exp(-delta_seconds / self.soft_overwrite_tau_seconds)
            penalized_score = s_old * math.exp(-self.soft_overwrite_kappa * s_new * recency_factor)
            penalized_score = self._clamp_score(penalized_score)

            if penalized_score < s_old:
                self.conn.execute("""
                    UPDATE fact_agg SET last_net_estimate=?, last_update_ts=?, is_ambiguous=0
                    WHERE user_id=? AND subject=? AND predicate=? AND object=?
                """, (penalized_score, event_ts, *predicate_key, incumbent_row["object"]))

        self.conn.execute("""
            INSERT INTO fact_agg
                (user_id, subject, predicate, object, polarity,
                 reinforcement_count, last_net_estimate, last_update_ts, log_accumulator, negative_log_accumulator)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, 0.0, 0.0)
            ON CONFLICT(user_id, subject, predicate, object) DO UPDATE SET
                polarity             = excluded.polarity,
                last_net_estimate   = excluded.last_net_estimate,
                last_update_ts      = excluded.last_update_ts,
                reinforcement_count = reinforcement_count + 1
        """, (
            event.user_id, event.subject, event.predicate, event.object,
            event.polarity.value, s_new, event_ts
        ))
        self._update_ambiguity(predicate_key)

    def _update_ambiguity(self, predicate_key: tuple[str, str, str]):
        self.conn.execute("""
            UPDATE fact_agg SET is_ambiguous=0
            WHERE user_id=? AND subject=? AND predicate=?
        """, predicate_key)
        top_two_rows = self.conn.execute("""
            SELECT object, last_net_estimate FROM fact_agg
            WHERE user_id=? AND subject=? AND predicate=?
            ORDER BY last_net_estimate DESC LIMIT 2
        """, predicate_key).fetchall()
        if len(top_two_rows) >= 2:
            gap = top_two_rows[0]["last_net_estimate"] - top_two_rows[1]["last_net_estimate"]
            is_ambiguous = 1 if gap < self.ambiguity_margin else 0
            if is_ambiguous == 1:
                self.conn.execute("""
                    UPDATE fact_agg SET is_ambiguous=1
                    WHERE user_id=? AND subject=? AND predicate=? AND object IN (?, ?)
                """, (*predicate_key, top_two_rows[0]["object"], top_two_rows[1]["object"]))
            else:
                self.conn.execute("""
                    UPDATE fact_agg SET is_ambiguous=0
                    WHERE user_id=? AND subject=? AND predicate=? AND object IN (?, ?)
                """, (*predicate_key, top_two_rows[0]["object"], top_two_rows[1]["object"]))

    def get_belief(self, user_id: str, subject: str, predicate: str, object_filter: Optional[str] = None):
        if object_filter is None:
            rows = self.conn.execute("""
                SELECT * FROM fact_agg
                WHERE user_id=? AND subject=? AND predicate=?
                ORDER BY last_net_estimate DESC
            """, (user_id, subject, predicate)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT * FROM fact_agg
                WHERE user_id=? AND subject=? AND predicate=? AND object=?
            """, (user_id, subject, predicate, object_filter)).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
