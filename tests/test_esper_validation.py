import uuid
from datetime import datetime, timedelta, timezone

from esper import Cardinality, ESPERDatabase, EvidenceEvent, Polarity, PredicateDefinition, SourceType


def make_event(
    *,
    predicate: str,
    object_value: str,
    conf: float,
    episode_id: str,
    evidence_ts: str,
    source_type: SourceType = SourceType.EXPLICIT,
    polarity: Polarity = Polarity.POSITIVE,
) -> EvidenceEvent:
    return EvidenceEvent(
        user_id="user-1",
        subject="profile",
        predicate=predicate,
        object=object_value,
        polarity=polarity,
        conf_cal=conf,
        episode_id=episode_id,
        evidence_ts=evidence_ts,
        source_type=source_type,
    )


def test_update_flip_prefers_latest_value_and_penalizes_previous():
    db = ESPERDatabase(
        predicate_definitions=[PredicateDefinition("CURRENT_CODENAME", Cardinality.ONE)]
    )
    now = datetime.now(timezone.utc)
    ep1 = str(uuid.uuid4())
    ep2 = str(uuid.uuid4())

    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Borealis",
            conf=0.95,
            episode_id=ep1,
            evidence_ts=(now - timedelta(hours=2)).isoformat(),
        )
    )
    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Atlas",
            conf=0.95,
            episode_id=ep2,
            evidence_ts=now.isoformat(),
        )
    )

    beliefs = db.get_belief("user-1", "profile", "CURRENT_CODENAME")
    top = beliefs[0]
    borealis = next(item for item in beliefs if item["object"] == "Borealis")

    assert top["object"] == "Atlas"
    assert top["last_net_estimate"] > 0.9
    assert borealis["last_net_estimate"] < 0.2
    assert top["is_ambiguous"] == 0
    db.close()


def test_correlated_reinforcement_dedupes_same_episode_and_rewards_new_episode():
    db = ESPERDatabase(
        predicate_definitions=[PredicateDefinition("KNOWN_SKILL", Cardinality.MANY)]
    )
    now = datetime.now(timezone.utc).isoformat()
    shared_episode = str(uuid.uuid4())

    db.insert_evidence(
        make_event(
            predicate="KNOWN_SKILL",
            object_value="Python",
            conf=0.7,
            episode_id=shared_episode,
            evidence_ts=now,
            source_type=SourceType.EXTRACTOR,
        )
    )
    db.insert_evidence(
        make_event(
            predicate="KNOWN_SKILL",
            object_value="Python",
            conf=0.9,
            episode_id=shared_episode,
            evidence_ts=now,
            source_type=SourceType.EXPLICIT,
        )
    )

    first = db.get_belief("user-1", "profile", "KNOWN_SKILL", object_filter="Python")[0]
    assert first["reinforcement_count"] == 1
    assert first["last_net_estimate"] > 0.85

    db.insert_evidence(
        make_event(
            predicate="KNOWN_SKILL",
            object_value="Python",
            conf=0.8,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
            source_type=SourceType.EXPLICIT,
        )
    )

    second = db.get_belief("user-1", "profile", "KNOWN_SKILL", object_filter="Python")[0]
    assert second["reinforcement_count"] == 2
    assert second["last_net_estimate"] > first["last_net_estimate"]
    db.close()


def test_schema_collision_isolated_by_predicate_namespace():
    db = ESPERDatabase(
        predicate_definitions=[
            PredicateDefinition("CURRENT_CODENAME", Cardinality.ONE),
            PredicateDefinition("DEFAULT_THEME", Cardinality.ONE),
        ]
    )
    now = datetime.now(timezone.utc).isoformat()

    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Borealis",
            conf=0.95,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
        )
    )
    db.insert_evidence(
        make_event(
            predicate="DEFAULT_THEME",
            object_value="Borealis",
            conf=0.9,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
        )
    )
    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Atlas",
            conf=0.95,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
        )
    )

    codename_beliefs = db.get_belief("user-1", "profile", "CURRENT_CODENAME")
    theme_beliefs = db.get_belief("user-1", "profile", "DEFAULT_THEME")

    assert codename_beliefs[0]["object"] == "Atlas"
    assert any(item["object"] == "Borealis" for item in codename_beliefs)
    assert len(theme_beliefs) == 1
    assert theme_beliefs[0]["object"] == "Borealis"
    assert theme_beliefs[0]["last_net_estimate"] > 0.85
    db.close()


def test_out_of_order_arrival_preserves_current_fact_priority():
    db = ESPERDatabase(
        predicate_definitions=[PredicateDefinition("CURRENT_CODENAME", Cardinality.ONE)]
    )
    now = datetime.now(timezone.utc)

    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Atlas",
            conf=0.95,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now.isoformat(),
        )
    )
    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Borealis",
            conf=0.85,
            episode_id=str(uuid.uuid4()),
            evidence_ts=(now - timedelta(days=30)).isoformat(),
        )
    )

    beliefs = db.get_belief("user-1", "profile", "CURRENT_CODENAME")
    atlas = next(item for item in beliefs if item["object"] == "Atlas")
    top = beliefs[0]

    assert top["object"] == "Atlas"
    assert atlas["last_net_estimate"] > 0.9
    db.close()


def rerank_score(similarity: float, belief: float, belief_weight: float) -> float:
    similarity_weight = 1.0 - belief_weight
    return similarity_weight * similarity + belief_weight * belief


def test_weight_ablation_shows_flip_is_not_single_weight_artifact():
    sim_borealis = 0.80
    sim_atlas = 0.77
    belief_borealis = 0.14
    belief_atlas = 0.95

    pure_similarity_scores = {
        "Borealis": rerank_score(sim_borealis, belief_borealis, belief_weight=0.0),
        "Atlas": rerank_score(sim_atlas, belief_atlas, belief_weight=0.0),
    }
    pure_similarity_winner = max(pure_similarity_scores, key=pure_similarity_scores.get)
    assert pure_similarity_winner == "Borealis"

    for weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        scores = {
            "Borealis": rerank_score(sim_borealis, belief_borealis, belief_weight=weight),
            "Atlas": rerank_score(sim_atlas, belief_atlas, belief_weight=weight),
        }
        assert max(scores, key=scores.get) == "Atlas"


def test_negative_polarity_reduces_many_cardinality_estimate():
    db = ESPERDatabase(
        predicate_definitions=[PredicateDefinition("KNOWN_SKILL", Cardinality.MANY)]
    )
    now = datetime.now(timezone.utc).isoformat()

    db.insert_evidence(
        make_event(
            predicate="KNOWN_SKILL",
            object_value="Python",
            conf=0.9,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
            source_type=SourceType.EXPLICIT,
            polarity=Polarity.POSITIVE,
        )
    )
    before = db.get_belief("user-1", "profile", "KNOWN_SKILL", object_filter="Python")[0]

    db.insert_evidence(
        make_event(
            predicate="KNOWN_SKILL",
            object_value="Python",
            conf=0.85,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
            source_type=SourceType.EXPLICIT,
            polarity=Polarity.NEGATIVE,
        )
    )
    after = db.get_belief("user-1", "profile", "KNOWN_SKILL", object_filter="Python")[0]

    assert after["last_net_estimate"] < before["last_net_estimate"]
    db.close()


def test_decay_lambda_penalizes_older_evidence():
    db = ESPERDatabase(
        predicate_definitions=[PredicateDefinition("FAVORITE_TOOL", Cardinality.MANY)]
    )
    now = datetime.now(timezone.utc)

    db.insert_evidence(
        make_event(
            predicate="FAVORITE_TOOL",
            object_value="legacy_item",
            conf=0.9,
            episode_id=str(uuid.uuid4()),
            evidence_ts=(now - timedelta(days=30)).isoformat(),
            source_type=SourceType.EXPLICIT,
        ),
        decay_lambda=1e-6,
    )
    db.insert_evidence(
        make_event(
            predicate="FAVORITE_TOOL",
            object_value="fresh_item",
            conf=0.9,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now.isoformat(),
            source_type=SourceType.EXPLICIT,
        ),
        decay_lambda=1e-6,
    )

    stale = db.get_belief("user-1", "profile", "FAVORITE_TOOL", object_filter="legacy_item")[0]
    fresh = db.get_belief("user-1", "profile", "FAVORITE_TOOL", object_filter="fresh_item")[0]

    assert stale["last_net_estimate"] < fresh["last_net_estimate"]
    assert stale["last_net_estimate"] < 0.2
    assert fresh["last_net_estimate"] > 0.8
    db.close()


def test_one_cardinality_replay_uses_event_order_not_wall_clock_age():
    db = ESPERDatabase(
        predicate_definitions=[PredicateDefinition("CURRENT_CODENAME", Cardinality.ONE)]
    )
    base = datetime.now(timezone.utc) - timedelta(days=365)

    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Borealis",
            conf=0.95,
            episode_id=str(uuid.uuid4()),
            evidence_ts=base.isoformat(),
        )
    )
    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Atlas",
            conf=0.95,
            episode_id=str(uuid.uuid4()),
            evidence_ts=(base + timedelta(hours=1)).isoformat(),
        )
    )

    beliefs = db.get_belief("user-1", "profile", "CURRENT_CODENAME")
    atlas = next(item for item in beliefs if item["object"] == "Atlas")
    borealis = next(item for item in beliefs if item["object"] == "Borealis")

    assert atlas["last_net_estimate"] > 0.9
    assert borealis["last_net_estimate"] < 0.2
    db.close()


def test_context_manager_closes_database():
    with ESPERDatabase(
        predicate_definitions=[PredicateDefinition("KNOWN_SKILL", Cardinality.MANY)]
    ) as db:
        now = datetime.now(timezone.utc).isoformat()
        db.insert_evidence(
            make_event(
                predicate="KNOWN_SKILL",
                object_value="Python",
                conf=0.9,
                episode_id=str(uuid.uuid4()),
                evidence_ts=now,
            )
        )

    try:
        db.conn.execute("SELECT 1")
        assert False
    except Exception as exc:
        assert "closed" in str(exc).lower()


def test_ambiguity_flags_both_top_candidates_when_gap_within_margin():
    db = ESPERDatabase(
        predicate_definitions=[PredicateDefinition("CURRENT_CODENAME", Cardinality.ONE)],
        soft_overwrite_kappa=0.01,
        ambiguity_margin=0.05,
    )
    now = datetime.now(timezone.utc).isoformat()

    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Borealis",
            conf=0.95,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
        )
    )
    db.insert_evidence(
        make_event(
            predicate="CURRENT_CODENAME",
            object_value="Atlas",
            conf=0.95,
            episode_id=str(uuid.uuid4()),
            evidence_ts=now,
        )
    )

    beliefs = db.get_belief("user-1", "profile", "CURRENT_CODENAME")
    assert len(beliefs) >= 2
    assert beliefs[0]["is_ambiguous"] == 1
    assert beliefs[1]["is_ambiguous"] == 1
    db.close()
