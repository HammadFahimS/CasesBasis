from __future__ import annotations

import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

import falkordb

if TYPE_CHECKING:
    from schemas.clinical_case import ClinicalCase
    from schemas.investigation import InvestigationResult
    from schemas.physical_exam import ExamFinding

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """
Extract clinical entities from the conversation turn.
Respond ONLY with compact JSON — no markdown, no explanation.
Format:
{"entities":[{"name":"...","type":"...","description":"...","negated":false}],"relations":[{"source":"...","relation":"...","target":"..."}]}

Entity types: Symptom, Medication, RiskFactor, Finding, Temporal, Location, Action, BodyPart, Measurement
Rules:
- Keep entity names short (1-4 words).
- "description": a short clinical phrase capturing severity/quality/timing (e.g. "8/10, substernal, radiating to left arm"). Empty string if nothing to add.
- "negated": true only when explicitly denied (e.g. "no fever", "denies cough").
- Include relations only when clearly stated.
""".strip()


_llm_client: Any = None
_llm_model: str = ""


def _get_llm_client() -> tuple[Any, str]:
    global _llm_client, _llm_model
    if _llm_client is None:
        from agents.base import build_llm_client, get_llm_provider, resolve_model
        provider = get_llm_provider()
        _llm_client = build_llm_client(provider)
        _llm_model = resolve_model(provider, None)
    return _llm_client, _llm_model


def _extract_entities(text: str) -> dict[str, Any]:
    """Extract clinical entities using the configured LLM provider."""
    try:
        client, model = _get_llm_client()
        raw = client.text_response(
            model=model,
            system_prompt=_EXTRACT_SYSTEM,
            user_prompt=text[:1000],
        )
        # Strip markdown fences if model wraps output
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as exc:
        logger.warning("entity extraction failed: %s", exc)
        return {}


class ClinicalKnowledgeGraph:
    """
    Per-session knowledge graph backed by FalkorDB.

    Entity types and relationships are extracted by the LLM from each conversation
    turn, so the graph schema expands freely as the simulation progresses.

    All writes happen in a background thread; the web handlers are never blocked.
    Falls back to no-op gracefully if FalkorDB is unreachable.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._db: falkordb.FalkorDB | None = None
        self._lock = threading.Lock()
        self._connect()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            host = os.getenv("FALKORDB_HOST", "localhost")
            port = int(os.getenv("FALKORDB_PORT", "6379"))
            self._db = falkordb.FalkorDB(host=host, port=port)
            self._db.connection.ping()
            self._enabled = True
            logger.info("ClinicalKnowledgeGraph connected to FalkorDB at %s:%s", host, port)
        except Exception as exc:
            logger.warning("FalkorDB unavailable — knowledge graph disabled: %s", exc)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _graph(self, session_id: str) -> falkordb.Graph:
        if self._db is None:
            raise RuntimeError("FalkorDB is not connected")
        return self._db.select_graph(f"session:{session_id}")

    # ── Internal write (runs in background thread) ─────────────────────────────

    def _write_entities(self, session_id: str, extraction: dict[str, Any]) -> None:
        """Persist extracted entities and relations into FalkorDB."""
        if not self._enabled or not extraction:
            return
        try:
            g = self._graph(session_id)
            entities = extraction.get("entities") or []
            relations = extraction.get("relations") or []

            for ent in entities:
                name = str(ent.get("name", "")).strip()
                etype = str(ent.get("type", "Entity")).strip()
                description = str(ent.get("description", "")).strip()
                negated = bool(ent.get("negated", False))
                if not name:
                    continue
                with self._lock:
                    g.query(
                        f"MERGE (n:{etype} {{name: $name, session_id: $sid}}) "
                        f"ON CREATE SET n.description = $desc, n.negated = $neg "
                        f"ON MATCH SET n.description = $desc, n.negated = $neg",
                        {"name": name, "sid": session_id, "desc": description, "neg": negated},
                    )

            for rel in relations:
                src = str(rel.get("source", "")).strip()
                tgt = str(rel.get("target", "")).strip()
                rel_type = str(rel.get("relation", "RELATED_TO")).strip().upper().replace(" ", "_")
                if not src or not tgt:
                    continue
                with self._lock:
                    g.query(
                        f"MATCH (a {{name: $src, session_id: $sid}}), (b {{name: $tgt, session_id: $sid}}) "
                        f"MERGE (a)-[:{rel_type}]->(b)",
                        {"src": src, "tgt": tgt, "sid": session_id},
                    )
        except Exception as exc:
            logger.warning("_write_entities failed: %s", exc)

    def _bg(self, fn, *args) -> None:
        """Run fn(*args) in a daemon background thread."""
        threading.Thread(target=fn, args=args, daemon=True).start()

    # ── Public API (sync, non-blocking) ───────────────────────────────────────

    def init_session(self, session_id: str, case: ClinicalCase) -> None:
        if not self._enabled:
            return
        self._bg(self._init_session, session_id, case)

    def _init_session(self, session_id: str, case: ClinicalCase) -> None:
        # Seed the graph with the case's structured fields (no LLM needed)
        try:
            g = self._graph(session_id)
            with self._lock:
                for symptom in case.symptoms:
                    g.query(
                        "MERGE (n:Symptom {name: $name, session_id: $sid})",
                        {"name": symptom, "sid": session_id},
                    )
                for rf in case.risk_factors:
                    g.query(
                        "MERGE (n:RiskFactor {name: $name, session_id: $sid})",
                        {"name": rf, "sid": session_id},
                    )
        except Exception as exc:
            logger.warning("_init_session failed: %s", exc)

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        if not self._enabled:
            return
        self._bg(self._add_turn, session_id, question, answer)

    def _add_turn(self, session_id: str, question: str, answer: str) -> None:
        text = f"Student: {question}\nPatient: {answer}"
        extraction = _extract_entities(text)
        self._write_entities(session_id, extraction)

    def add_investigation(self, session_id: str, result: InvestigationResult) -> None:
        if not self._enabled:
            return
        self._bg(self._add_investigation, session_id, result)

    def _add_investigation(self, session_id: str, result: InvestigationResult) -> None:
        text = f"Investigation: {result.investigation}. Result: {result.summary or ''}"
        extraction = _extract_entities(text)
        self._write_entities(session_id, extraction)

    def add_examination(self, session_id: str, finding: ExamFinding) -> None:
        if not self._enabled:
            return
        self._bg(self._add_examination, session_id, finding)

    def _add_examination(self, session_id: str, finding: ExamFinding) -> None:
        findings_text = ". ".join(finding.findings)
        text = f"Physical examination — {finding.system}: {findings_text}. Summary: {finding.summary}"
        extraction = _extract_entities(text)
        self._write_entities(session_id, extraction)

    # ── Context extraction ────────────────────────────────────────────────────

    def get_context(self, session_id: str) -> str:
        """Return a formatted clinical summary of all entities elicited so far.

        Groups entities by type into a compact paragraph the LLM can use as
        context. Returns an empty string when the graph is unavailable or empty.
        """
        if not self._enabled:
            return ""
        try:
            g = self._graph(session_id)
            result = g.query(
                "MATCH (n) WHERE n.session_id = $sid "
                "RETURN labels(n), n.name, n.description, n.negated",
                {"sid": session_id},
            )
            rows = result.result_set or []
            if not rows:
                return ""

            buckets: dict[str, list[str]] = {}
            negated: list[str] = []

            for row in rows:
                labels, name, description, is_negated = row[0], row[1], row[2], row[3]
                if not name:
                    continue
                label = labels[0] if labels else "Entity"
                phrase = name if not description else f"{name} ({description})"
                if is_negated:
                    negated.append(name)
                else:
                    buckets.setdefault(label, []).append(phrase)

            lines: list[str] = []
            order = ["Symptom", "Finding", "RiskFactor", "Medication", "Measurement", "Temporal"]
            seen = set()
            for lbl in order:
                if lbl in buckets:
                    lines.append(f"{lbl}s: {', '.join(buckets[lbl])}")
                    seen.add(lbl)
            for lbl, phrases in buckets.items():
                if lbl not in seen:
                    lines.append(f"{lbl}: {', '.join(phrases)}")
            if negated:
                lines.append(f"Denied: {', '.join(negated)}")

            return "\n".join(lines)
        except Exception as exc:
            logger.warning("get_context failed: %s", exc)
            return ""

    def get_entity_names(self, session_id: str) -> list[str]:
        """Return a flat list of all non-negated entity names for this session."""
        if not self._enabled:
            return []
        try:
            g = self._graph(session_id)
            result = g.query(
                "MATCH (n) WHERE n.session_id = $sid AND (n.negated = false OR n.negated IS NULL) "
                "RETURN n.name",
                {"sid": session_id},
            )
            return [row[0] for row in (result.result_set or []) if row[0]]
        except Exception as exc:
            logger.warning("get_entity_names failed: %s", exc)
            return []

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def delete_session(self, session_id: str) -> None:
        if not self._enabled:
            return
        try:
            self._db.delete_graph(f"session:{session_id}")  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("delete_session failed: %s", exc)
