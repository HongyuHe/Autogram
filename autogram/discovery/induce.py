"""LLM-style schema induction from observable column names.

v2 exposes exactly two induction backends: ``subagent`` and ``openai``.  Both return the same
bounded :class:`SchemaSpec` data contract.  The subagent backend is mandatory and concrete: it
invokes a real long-context agentic-CLI subagent by default (Copilot, Codex, or Claude -- selected
by harness, Copilot by default), or a caller-supplied responder that does the same.  If no real
responder/transport is available, induction raises a hard error.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from ..loader.names import NameModel
from ..schema.adapter import SchemaAdapter
from ..schema.compiler import compile_spec
from ..schema.spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    PRED_SLOTS,
    RefTemplate,
    RoleOntology,
    SchemaSpec,
)
from .subagent import AutogramSubagentRunner

_BACKENDS = ("subagent", "openai")


class SchemaCompletenessError(ValueError):
    """Raised when a model-produced spec compiles but cannot ground the live data."""


def available_inducer_backends() -> tuple[str, str]:
    return _BACKENDS


class SchemaInducer:
    backend = ""

    def induce(self, columns: Sequence[str], sample_rows: Optional[Sequence[dict]] = None) -> SchemaSpec:  # pragma: no cover
        raise NotImplementedError


class SubagentSchemaInducer(SchemaInducer):
    """Schema induction backend for a long-context subagent.

    ``responder`` is an injectable concrete transport.  If omitted, Autogram invokes a real
    agentic-CLI subagent (``harness`` selects Copilot/Codex/Claude; Copilot by default) with a
    long-context model.  Passing ``None`` explicitly means there is no available transport and
    therefore raises instead of falling back to local parsing.
    """

    backend = "subagent"
    _DEFAULT_RESPONDER = object()

    def __init__(self, responder=_DEFAULT_RESPONDER, max_attempts: int | None = None,
                 harness: str | None = None):
        if responder is self._DEFAULT_RESPONDER:
            self.responder = AutogramSubagentRunner(harness=harness)
        else:
            self.responder = responder
        self.harness = harness
        self.max_attempts = max(1, int(max_attempts or os.environ.get("AUTOGRAM_SUBAGENT_MAX_ATTEMPTS", "3")))

    def induce(self, columns, sample_rows=None) -> SchemaSpec:
        if self.responder is None:
            raise RuntimeError(
                "Subagent schema induction requires a real subagent responder/transport; "
                "no offline morphology fallback is available."
            )
        prompt = _schema_prompt(columns, sample_rows)
        payload = self.responder(prompt)
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                raw = _load_json_object(payload)
                spec = _spec_from_json(raw)
                _validate_schema_completeness(spec, columns)
                _log_schema_event("schema_complete", {"attempt": attempt + 1, "backend": self.backend})
                return spec
            except SchemaCompletenessError as exc:
                last_error = exc
                repaired, repair_notes = _repair_schema_completeness(raw, columns)
                if repair_notes:
                    try:
                        spec = _spec_from_json(repaired)
                        _validate_schema_completeness(spec, columns)
                        _log_schema_event(
                            "schema_repaired",
                            {
                                "attempt": attempt + 1,
                                "backend": self.backend,
                                "repairs": repair_notes,
                                "reason": str(exc)[:1000],
                            },
                        )
                        return spec
                    except Exception as repair_exc:
                        last_error = repair_exc
                        raw = repaired
                if attempt + 1 < self.max_attempts:
                    _log_schema_event(
                        "schema_incomplete_retry",
                        {"attempt": attempt + 1, "backend": self.backend, "reason": str(last_error)[:1000]},
                    )
                    payload = self.responder(_completeness_repair_prompt(prompt, raw, last_error))
                    continue
                break
            except Exception as exc:  # model returned malformed JSON or an invalid SchemaSpec
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    payload = self.responder(_repair_prompt(prompt, payload, exc))
                    continue
                break
        raise RuntimeError("Subagent returned invalid or incomplete SchemaSpec JSON; no offline fallback is available") from last_error

    def to_json_spec(self, columns, sample_rows=None) -> str:
        return json.dumps(_spec_to_json(self.induce(columns, sample_rows)))


class OpenAISchemaInducer(SchemaInducer):
    """OpenAI SDK backend; tests may inject a real-model responder transport."""

    backend = "openai"

    def __init__(self, responder=None, model: str = "gpt-5.5"):
        self.responder = responder
        self.model = model

    def induce(self, columns, sample_rows=None) -> SchemaSpec:
        prompt = _schema_prompt(columns, sample_rows)
        if self.responder is not None:
            payload = self.responder(prompt)
        else:  # pragma: no cover - exercised only with credentials
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set; inject a real responder transport")
            from openai import OpenAI
            client = OpenAI()
            msg = client.responses.create(model=self.model, input=prompt)
            payload = msg.output_text
        raw = _load_json_object(payload)
        spec = _spec_from_json(raw)
        try:
            _validate_schema_completeness(spec, columns)
            return spec
        except SchemaCompletenessError as exc:
            repaired, repair_notes = _repair_schema_completeness(raw, columns)
            if repair_notes:
                spec = _spec_from_json(repaired)
                _validate_schema_completeness(spec, columns)
                return spec
            raise RuntimeError("OpenAI returned incomplete SchemaSpec JSON") from exc


def make_inducer(backend: str = "subagent", **kwargs) -> SchemaInducer:
    if backend == "subagent":
        return SubagentSchemaInducer(**kwargs)
    if backend == "openai":
        return OpenAISchemaInducer(**kwargs)
    raise ValueError(f"unknown schema inducer backend {backend!r}; expected one of {_BACKENDS}")


def induce_spec(columns: Sequence[str], inducer: Optional[SchemaInducer] = None, sample_rows=None) -> SchemaSpec:
    return (inducer or make_inducer("subagent")).induce(columns, sample_rows)


def induce_adapter(columns: Sequence[str], inducer: Optional[SchemaInducer] = None, sample_rows=None) -> SchemaAdapter:
    return compile_spec(induce_spec(columns, inducer, sample_rows))


def _schema_prompt(columns, sample_rows) -> str:
    head = "\n".join(list(columns)[:500])
    return (
        "Return ONLY valid JSON for an Autogram SchemaSpec. This is NOT a SQL/database schema. "
        "Use ONLY the exact column names between EXACT_COLUMNS_BEGIN and EXACT_COLUMNS_END; "
        "ignore all other context and never invent id/title/status columns.\n\n"
        "EXACT_COLUMNS_BEGIN\n" + head + "\nEXACT_COLUMNS_END\n\n"
        "Top-level keys MUST be exactly: name, patterns, ontology, ref_templates, "
        "family_selectors, binder_enumerate, cell_codec, noisy_kind, demand_kind, "
        "link_marker_direction, notes. The object must contain key 'ontology'.\n\n"
        "ColumnPattern fields: name, matcher, kind, direction, regex, node_groups, source_group, "
        "destination_group, peer_group, token_groups, prefix, sep, split_slots. Use matcher='regex' and "
        "anchored regexes. Ontology fields: binders, ref_roles, fam_roles, ops, agg_kinds, "
        "ref_glyphs, fam_glyphs. IMPORTANT: ref_roles and fam_roles must be JSON objects/maps "
        "from binder name to an array of role strings, not arrays of objects. RefTemplate fields: "
        "binder, role, template. FamilySelector fields: binder, family_role, match_kind, "
        "match_direction, predicates. binder_enumerate must be a JSON object whose values are "
        "plain strategy strings, never objects or column lists.\n\n"
        "Use these Autogram role conventions:\n"
        "1. Always include binder 'cell' with ref role 'self', fam roles [], strategy "
        "per_measured_col, and ref template '{col}'.\n"
        "2. Always include binder 'node' for per-entity columns and binder 'network' with strategy "
        "singleton. Single-entity measured directions "
        "become node ref roles named measurement_<direction>, with templates '<measured_kind>_{X}_<direction>'.\n"
        "3. If there is a pair-matrix layer, call its ref role demand_self and family roles "
        "demand_row/demand_col. Its direction string is 'demand'. Demand entity tokens may "
        "contain dots, hyphens, or other punctuation; do NOT split entities on punctuation. "
        "Discover the complete entity-token alternation from the observed columns, escape those "
        "tokens in the regex, and make the pair pattern match the full "
        "'<demand_kind>_<X>_<Y>' string with X and Y as whole entity tokens. Add templates "
        "'<demand_kind>_{X}_{X}' and selectors demand_row source==X destination!=X, demand_col destination==X source!=X.\n"
        "4. Directed measured link directions are the full middle token between two entities "
        "(preserve embedded underscores, e.g. egress_to, ingress_from). Add binder 'link' with "
        "strategy per_directed_link. Create one separate ColumnPattern for EACH directed "
        "measured link direction; do NOT use a regex group named direction and do NOT use a "
        "generic direction value like 'directed'. MANDATORY directed-link contract: the regex "
        "for '<measured_kind>_<X>_<direction>_<Y>' must expose BOTH endpoints as source_group and "
        "peer_group. The second entity is the peer/Y endpoint; never leave peer_group empty and "
        "never label the second entity only as destination_group, because per_directed_link binds Y from "
        "sem.peer. For link directions in sorted order, add ref roles o0/o0_rev, o1/o1_rev, ... "
        "with templates '<measured_kind>_{X}_<direction>_{Y}' and '<measured_kind>_{Y}_<direction>_{X}'. "
        "Also add link demand/demand_rev templates when a pair-matrix layer exists. Set "
        "link_marker_direction to the first directed measured direction.\n"
        "5. For each directed measured link direction, add node family role fam_<direction> "
        "selecting measured columns with source==X. For each single-entity measured direction, add "
        "network family role all_measurement_<direction>. Add all_demand for the demand matrix with source!=@destination.\n"
        "6. Ignore metadata columns that are not numeric observation families, such as timestamp, "
        "*_perturbed_type, and true_*.\n\n"
        "Entity tokens may contain dots/punctuation but are separated from kind/direction tokens "
        "by underscores. For pair-matrix demand columns, prefer a full escaped entity alternation "
        "over naive dot or punctuation splits; X/Y must be the complete entity tokens. "
        "If entity tokens do not contain underscores, generic [^_]+ groups are OK for measured "
        "columns, but demand columns with punctuation still need whole-token matching. "
        "Preserve observed kind and direction spellings exactly in regexes and templates. "
        "Use cell_codec {'kind':'dict_gt_hidden','primary':'ground_truth','clean':'hidden_ground_truth'}, "
        "ops ['~=','==','!=','<=','>=','<|>'], and agg_kinds ['SUM','MIN','MAX','AVG']. "
        "The binder_enumerate object should look like {'cell':'per_measured_col','node':'per_node',"
        "'network':'singleton','link':'per_directed_link'} when link exists."
    )


def _repair_prompt(original_prompt: str, bad_payload: str, error: Exception) -> str:
    return (
        "Your previous Autogram SchemaSpec response was invalid. Return ONLY corrected valid "
        "JSON for the same task. Use double-quoted JSON keys/strings; escape regex backslashes "
        "as JSON strings; include the top-level key 'ontology'. Do not add markdown.\n\n"
        f"Validation error: {type(error).__name__}: {error}\n\n"
        "Original task:\n" + original_prompt + "\n\n"
        "Previous invalid response (truncated):\n" + str(bad_payload)[:4000]
    )


def _completeness_repair_prompt(original_prompt: str, bad_payload, error: Exception) -> str:
    return (
        "Your previous Autogram SchemaSpec was valid JSON but incomplete on the real columns. "
        "Return ONLY corrected JSON for the same task. Preserve the real model-induced schema, "
        "but fix directed measured link patterns so '<kind>_<X>_<connector>_<Y>' sets "
        "peer_group to the second entity group, and every declared binder grounds at least one "
        "real non-degenerate binding. Also fix demand matrix patterns so '<demand_kind>_<X>_<Y>' "
        "matches dotted/punctuated entity tokens as whole tokens and the inferred node universe "
        "covers every entity present in demand columns.\n\n"
        f"Completeness error: {type(error).__name__}: {error}\n\n"
        "Original task:\n" + original_prompt + "\n\n"
        "Previous incomplete response (truncated):\n" + json.dumps(bad_payload)[:4000]
    )


def _load_json_object(payload: str | dict) -> dict:
    if isinstance(payload, dict):
        return payload
    text = payload.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(text[start:end + 1])


def _spec_to_json(spec: SchemaSpec) -> dict:
    return {
        "name": spec.name,
        "patterns": [p.__dict__ for p in spec.patterns],
        "ontology": {
            "binders": list(spec.ontology.binders),
            "ref_roles": {k: list(v) for k, v in spec.ontology.ref_roles.items()},
            "fam_roles": {k: list(v) for k, v in spec.ontology.fam_roles.items()},
            "ops": list(spec.ontology.ops),
            "agg_kinds": list(spec.ontology.agg_kinds),
            "ref_glyphs": dict(spec.ontology.ref_glyphs),
            "fam_glyphs": dict(spec.ontology.fam_glyphs),
        },
        "ref_templates": [t.__dict__ for t in spec.ref_templates],
        "family_selectors": [s.__dict__ for s in spec.family_selectors],
        "binder_enumerate": dict(spec.binder_enumerate),
        "cell_codec": spec.cell_codec.__dict__,
        "noisy_kind": spec.noisy_kind,
        "demand_kind": spec.demand_kind,
        "link_marker_direction": spec.link_marker_direction,
        "notes": spec.notes,
    }


def _payload_node_groups(p: dict) -> tuple[str, ...]:
    raw = p.get("node_groups") or ()
    if isinstance(raw, dict):
        return tuple(str(k) for k in raw)
    return tuple(str(x) for x in raw)


def _repair_directed_link_peer_groups(payload: dict) -> tuple[dict, list[str]]:
    """Map a model's second directed endpoint to peer_group without inducing a schema offline."""

    repaired = json.loads(json.dumps(payload))
    noisy_kind = repaired.get("noisy_kind", "measurement")
    notes: list[str] = []
    for p in repaired.get("patterns", []):
        if p.get("kind") != noisy_kind:
            continue
        nodes = _payload_node_groups(p)
        if len(nodes) < 2:
            continue
        if p.get("peer_group"):
            continue
        second = p.get("destination_group") or nodes[1]
        if not second:
            continue
        p["peer_group"] = second
        if p.get("matcher") == "split":
            slots = list(p.get("split_slots") or ("source", "destination"))
            if len(slots) >= 2 and slots[1] == "destination":
                slots[1] = "peer"
                p["split_slots"] = slots[:2]
        notes.append(f"{p.get('name') or p.get('direction')}: peer_group={second}")
    return repaired, notes


def _repair_schema_completeness(payload: dict, columns: Sequence[str]) -> tuple[dict, list[str]]:
    repaired, notes = _repair_directed_link_peer_groups(payload)
    repaired, demand_notes = _repair_demand_matrix_pattern(repaired, columns)
    demand_kind = str(repaired.get("demand_kind", "demand"))
    demand_entities = _demand_entities_from_pairs(_derive_demand_pairs(columns, demand_kind))
    repaired, measured_notes = _repair_measured_pair_patterns_from_entities(repaired, demand_entities)
    return repaired, notes + demand_notes + measured_notes


def _demand_column_bodies(columns: Sequence[str], demand_kind: str) -> list[tuple[str, str]]:
    prefix = f"{demand_kind}_"
    return [
        (str(c), str(c)[len(prefix):])
        for c in columns
        if str(c).startswith(prefix) and "_" in str(c)[len(prefix):]
    ]


def _split_body_with_entities(body: str, entities: set[str]) -> tuple[str, str] | None:
    for source in sorted(entities, key=len, reverse=True):
        marker = source + "_"
        if body.startswith(marker):
            destination = body[len(marker):]
            if destination in entities:
                return source, destination
    return None


def _split_candidates(body: str) -> list[tuple[str, str]]:
    return [
        (body[:idx], body[idx + 1:])
        for idx, ch in enumerate(body)
        if ch == "_" and idx > 0 and idx < len(body) - 1
    ]


def _derive_demand_pairs(
    columns: Sequence[str],
    demand_kind: str,
    known_entities: Sequence[str] = (),
) -> list[tuple[str, str, str]]:
    bodies = _demand_column_bodies(columns, demand_kind)
    if not bodies:
        return []

    known = {str(x) for x in known_entities if str(x)}
    if known:
        parsed = []
        for col, body in bodies:
            pair = _split_body_with_entities(body, known)
            if pair is None:
                parsed = []
                break
            parsed.append((col, pair[0], pair[1]))
        if parsed:
            return parsed

    if all(body.count("_") == 1 for _, body in bodies):
        return [
            (col, body.split("_", 1)[0], body.split("_", 1)[1])
            for col, body in bodies
            if body.split("_", 1)[0] and body.split("_", 1)[1]
        ]

    options = {body: _split_candidates(body) for _, body in bodies}
    if any(not choices for choices in options.values()):
        return []
    left_seen: dict[str, int] = {}
    right_seen: dict[str, int] = {}
    for choices in options.values():
        for left, right in choices:
            left_seen[left] = left_seen.get(left, 0) + 1
            right_seen[right] = right_seen.get(right, 0) + 1
    parsed = []
    for col, body in bodies:
        choices = options[body]
        left, right = max(
            choices,
            key=lambda pair: (
                left_seen.get(pair[0], 0)
                + right_seen.get(pair[1], 0)
                + (2 if pair[0] in right_seen else 0)
                + (2 if pair[1] in left_seen else 0),
                len(pair[0]) + len(pair[1]),
            ),
        )
        parsed.append((col, left, right))
    return parsed


def _demand_entities_from_pairs(pairs: Sequence[tuple[str, str, str]]) -> set[str]:
    out: set[str] = set()
    for _, source, destination in pairs:
        if source:
            out.add(source)
        if destination:
            out.add(destination)
    return out


def _repair_demand_matrix_pattern(payload: dict, columns: Sequence[str]) -> tuple[dict, list[str]]:
    demand_kind = str(payload.get("demand_kind", "demand"))
    pairs = _derive_demand_pairs(columns, demand_kind)
    entities = _demand_entities_from_pairs(pairs)
    if not pairs or not entities:
        return payload, []

    repaired = json.loads(json.dumps(payload))
    demand_patterns = [
        p for p in repaired.get("patterns", [])
        if p.get("kind") == demand_kind or p.get("direction") == "demand"
    ]
    if not demand_patterns:
        demand_patterns = [{
            "name": f"{demand_kind}_demand",
            "matcher": "regex",
            "kind": demand_kind,
            "direction": "demand",
            "regex": "",
            "node_groups": [],
            "source_group": "",
            "destination_group": "",
            "peer_group": "",
            "token_groups": [],
            "prefix": "",
            "sep": "_",
            "split_slots": ["source", "destination"],
        }]
        repaired.setdefault("patterns", []).append(demand_patterns[0])

    alt = "|".join(re.escape(entity) for entity in sorted(entities, key=lambda x: (-len(x), x)))
    regex = rf"^{re.escape(demand_kind)}_(?P<source>(?:{alt}))_(?P<destination>(?:{alt}))$"
    notes = []
    for p in demand_patterns:
        changed = (
            p.get("matcher") != "regex"
            or p.get("regex") != regex
            or tuple(p.get("node_groups") or ()) != ("source", "destination")
            or p.get("source_group") != "source"
            or p.get("destination_group") != "destination"
            or tuple(p.get("token_groups") or ()) != ("source", "destination")
        )
        p.update({
            "matcher": "regex",
            "kind": demand_kind,
            "direction": "demand",
            "regex": regex,
            "node_groups": ["source", "destination"],
            "source_group": "source",
            "destination_group": "destination",
            "peer_group": "",
            "token_groups": ["source", "destination"],
            "prefix": "",
            "sep": "_",
            "split_slots": ["source", "destination"],
        })
        if changed:
            notes.append(
                f"{p.get('name') or 'demand'}: demand_regex_entities={len(entities)} "
                f"demand_columns={len(pairs)}"
            )
    return repaired, notes


def _repair_measured_pair_patterns_from_entities(payload: dict, entities: set[str]) -> tuple[dict, list[str]]:
    if not entities:
        return payload, []
    repaired = json.loads(json.dumps(payload))
    noisy_kind = str(repaired.get("noisy_kind", "measurement"))
    alt = "|".join(re.escape(entity) for entity in sorted(entities, key=lambda x: (-len(x), x)))
    notes: list[str] = []
    for p in repaired.get("patterns", []):
        direction = str(p.get("direction") or "")
        if not direction or direction in ("demand", "directed", "link"):
            continue
        nodes = _payload_node_groups(p)
        if len(nodes) < 2 and not p.get("peer_group"):
            continue
        regex = rf"^{re.escape(noisy_kind)}_(?P<source>(?:{alt}))_{re.escape(direction)}_(?P<peer>(?:{alt}))$"
        changed = (
            p.get("kind") != noisy_kind
            or p.get("direction") != direction
            or p.get("matcher") != "regex"
            or p.get("regex") != regex
            or tuple(p.get("node_groups") or ()) != ("source", "peer")
            or p.get("source_group") != "source"
            or p.get("peer_group") != "peer"
            or tuple(p.get("token_groups") or ()) != ("source", "peer")
        )
        p.update({
            "matcher": "regex",
            "kind": noisy_kind,
            "direction": direction,
            "regex": regex,
            "node_groups": ["source", "peer"],
            "source_group": "source",
            "destination_group": "",
            "peer_group": "peer",
            "token_groups": ["source", "peer"],
            "prefix": "",
            "sep": "_",
            "split_slots": ["source", "peer"],
        })
        if changed:
            notes.append(
                f"{p.get('name') or direction}: measured_pair_regex_entities={len(entities)}"
            )
    return repaired, notes


def _nondegenerate_bindings(bindings: Sequence[dict]) -> list[dict]:
    return [
        b for b in bindings
        if all(value is not None and value != "" for value in b.values())
    ]


def _validate_schema_completeness(spec: SchemaSpec, columns: Sequence[str]) -> None:
    adapter = compile_spec(spec)
    nm = NameModel.from_columns_with_adapter(columns, adapter)
    errors: list[str] = []
    demand_pairs = _derive_demand_pairs(columns, spec.demand_kind, nm.node_list())
    raw_demand_count = len(_demand_column_bodies(columns, spec.demand_kind))
    demand_entities = _demand_entities_from_pairs(demand_pairs)

    for p in spec.patterns:
        if p.kind == spec.noisy_kind and len(p.node_groups) >= 2 and not p.peer_group:
            errors.append(
                f"directed measured pattern {p.name!r}/{p.direction!r} has no peer_group"
            )
        if p.kind == spec.noisy_kind and len(p.node_groups) >= 2 and p.direction in ("directed", "link"):
            errors.append(
                f"directed measured pattern {p.name!r} uses generic direction {p.direction!r}"
            )

    if raw_demand_count:
        demand_sems = [
            sem for sem in nm.by_name.values()
            if sem.kind == adapter.demand_kind and sem.direction == "demand"
        ]
        grounded = [
            sem for sem in demand_sems
            if sem.source and sem.destination and sem.name in {col for col, _ in _demand_column_bodies(columns, spec.demand_kind)}
        ]
        if not grounded:
            errors.append(
                f"demand matrix pattern grounded 0 demand columns for kind {spec.demand_kind!r} "
                f"({raw_demand_count} real demand columns)"
            )
        elif len(grounded) < raw_demand_count:
            errors.append(
                f"demand matrix pattern grounded only {len(grounded)}/{raw_demand_count} demand columns"
            )
        parsed_entities = {
            value
            for sem in grounded
            for value in (sem.source, sem.destination, *sem.nodes)
            if value
        }
        missing_grounded = demand_entities - parsed_entities
        if missing_grounded:
            errors.append(
                "demand matrix pattern truncated entity tokens; missing "
                + ", ".join(sorted(missing_grounded)[:8])
            )
        missing_nodes = demand_entities - set(nm.node_list())
        if missing_nodes:
            errors.append(
                "demand matrix node universe missing entities "
                + ", ".join(sorted(missing_nodes)[:8])
            )

    for sem in nm.by_name.values():
        if sem.kind == adapter.noisy_kind and len(sem.nodes) >= 2:
            if not sem.source or not sem.peer:
                errors.append(
                    f"directed measured column {sem.name!r} parsed without both source and peer endpoints"
                )
            if demand_entities and (
                (sem.source and sem.source not in demand_entities)
                or (sem.peer and sem.peer not in demand_entities)
            ):
                errors.append(
                    f"directed measured column {sem.name!r} parsed endpoints outside demand entity set"
                )

    for binder in adapter.binders:
        bindings = adapter.enumerate_bindings(binder, nm)
        nondegenerate = _nondegenerate_bindings(bindings)
        if binder == "network" and bindings == [{}]:
            nondegenerate = [{}]
        if not nondegenerate:
            errors.append(
                f"binder {binder!r} yielded 0 non-degenerate bindings on real columns"
            )
            continue

    if errors:
        shown = "; ".join(errors[:8])
        extra = "" if len(errors) <= 8 else f"; ... ({len(errors) - 8} more)"
        raise SchemaCompletenessError(shown + extra)


def _log_schema_event(event: str, payload: dict) -> None:
    path = Path(os.environ.get("AUTOGRAM_SUBAGENT_LOG", r"artifacts\subagent_schema_induction.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def _spec_from_json(payload) -> SchemaSpec:
    onto = payload["ontology"]
    noisy_kind = payload.get("noisy_kind", "measurement")
    demand_kind = payload.get("demand_kind", "demand")
    patterns = tuple(_column_pattern(p) for p in payload["patterns"])
    link_directions = _link_directions(patterns, noisy_kind)
    single_directions = _single_node_directions(patterns, noisy_kind)
    has_demand = any(p.kind == demand_kind and p.direction == "demand" for p in patterns)
    ref_templates = _augment_ref_templates(
        tuple(_ref_template(t, noisy_kind, demand_kind, link_directions) for t in payload["ref_templates"]),
        noisy_kind, demand_kind, single_directions, link_directions, has_demand,
    )
    family_selectors = _augment_family_selectors(tuple(
        _family_selector(s)
        for s in payload["family_selectors"]
    ), noisy_kind, demand_kind, single_directions, link_directions, has_demand)
    binders = _augment_binders(tuple(_binder_names(onto["binders"])), bool(link_directions))
    ref_roles = _canonical_roles(_role_map(onto["ref_roles"], "ref_roles"), binders,
                                 ((t.binder, t.role) for t in ref_templates))
    fam_roles = _canonical_roles(_role_map(onto["fam_roles"], "fam_roles"), binders,
                                 ((s.binder, s.family_role) for s in family_selectors))
    return SchemaSpec(
        name=payload.get("name", "induced"),
        patterns=patterns,
        ontology=RoleOntology(
            binders=binders,
            ref_roles=ref_roles,
            fam_roles=fam_roles,
            ops=tuple(onto.get("ops", ("~=", "==", "!=", "<=", ">=", "<|>"))),
            agg_kinds=tuple(onto.get("agg_kinds", ("SUM", "MIN", "MAX", "AVG"))),
            ref_glyphs=dict(onto.get("ref_glyphs", {})),
            fam_glyphs=dict(onto.get("fam_glyphs", {})),
        ),
        ref_templates=ref_templates,
        family_selectors=family_selectors,
        binder_enumerate=_augment_binder_enumerate(dict(payload["binder_enumerate"]), bool(link_directions)),
        cell_codec=CellCodec(**payload.get("cell_codec", {"kind": "dict_gt_hidden"})),
        noisy_kind=noisy_kind,
        demand_kind=demand_kind,
        link_marker_direction=(link_directions[0] if link_directions else payload.get("link_marker_direction") or _first_link_direction(ref_templates) or "demand"),
        notes=payload.get("notes", ""),
    )


def _canonical_roles(existing: dict[str, tuple[str, ...]], binders: tuple[str, ...], emitted) -> dict[str, tuple[str, ...]]:
    by_binder = {b: [] for b in binders}
    emitted_any = False
    for binder, role in emitted:
        emitted_any = True
        by_binder.setdefault(binder, [])
        if role not in by_binder[binder]:
            by_binder[binder].append(role)
    if emitted_any:
        return {b: tuple(by_binder.get(b, ())) for b in binders}
    return {b: tuple(existing.get(b, ())) for b in binders}


def _augment_binders(binders: tuple[str, ...], has_link: bool) -> tuple[str, ...]:
    out = list(binders)
    for binder in ("cell", "node", "network"):
        if binder not in out:
            out.append(binder)
    if has_link and "link" not in out:
        out.append("link")
    return tuple(out)


def _augment_binder_enumerate(value: dict, has_link: bool) -> dict:
    out = dict(value)
    out.setdefault("cell", "per_measured_col")
    out.setdefault("node", "per_node")
    out.setdefault("network", "singleton")
    if has_link:
        out.setdefault("link", "per_directed_link")
    return out


def _augment_ref_templates(
    templates: tuple[RefTemplate, ...],
    noisy_kind: str,
    demand_kind: str,
    single_directions: tuple[str, ...],
    link_directions: tuple[str, ...],
    has_demand: bool,
) -> tuple[RefTemplate, ...]:
    out = list(templates)

    def add(t: RefTemplate) -> None:
        if (t.binder, t.role) not in {(x.binder, x.role) for x in out}:
            out.append(t)

    add(RefTemplate("cell", "self", "{col}"))
    for direction in single_directions:
        add(RefTemplate("node", f"measurement_{direction}", f"{noisy_kind}_{{X}}_{direction}"))
    if has_demand:
        add(RefTemplate("node", "demand_self", f"{demand_kind}_{{X}}_{{X}}"))
    for idx, direction in enumerate(link_directions):
        add(RefTemplate("link", f"o{idx}", f"{noisy_kind}_{{X}}_{direction}_{{Y}}"))
        add(RefTemplate("link", f"o{idx}_rev", f"{noisy_kind}_{{Y}}_{direction}_{{X}}"))
    if has_demand and link_directions:
        add(RefTemplate("link", "demand", f"{demand_kind}_{{X}}_{{Y}}"))
        add(RefTemplate("link", "demand_rev", f"{demand_kind}_{{Y}}_{{X}}"))
    return tuple(out)


def _augment_family_selectors(
    selectors: tuple[FamilySelector, ...],
    noisy_kind: str,
    demand_kind: str,
    single_directions: tuple[str, ...],
    link_directions: tuple[str, ...],
    has_demand: bool,
) -> tuple[FamilySelector, ...]:
    out = list(selectors)

    def add(s: FamilySelector) -> None:
        if (s.binder, s.family_role) not in {(x.binder, x.family_role) for x in out}:
            out.append(s)

    if has_demand:
        add(FamilySelector("node", "demand_row", demand_kind, "demand", (("source", "==", "X"), ("destination", "!=", "X"))))
        add(FamilySelector("node", "demand_col", demand_kind, "demand", (("destination", "==", "X"), ("source", "!=", "X"))))
        add(FamilySelector("network", "all_demand", demand_kind, "demand", (("source", "!=", "@destination"),)))
    for direction in link_directions:
        add(FamilySelector("node", f"fam_{direction}", noisy_kind, direction, (("source", "==", "X"),)))
    for direction in single_directions:
        add(FamilySelector("network", f"all_measurement_{direction}", noisy_kind, direction, ()))
    return tuple(out)


def _binder_names(value) -> tuple[str, ...]:
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(item.get("binder") or item.get("name"))
    return tuple(x for x in out if x)


def _role_map(value, field_name: str) -> dict[str, tuple[str, ...]]:
    if isinstance(value, dict):
        return {k: tuple(v) for k, v in value.items()}
    out = {}
    for item in value:
        if isinstance(item, dict):
            binder = item.get("binder") or item.get("name")
            roles = (
                item.get("roles")
                or item.get(field_name)
                or item.get("ref_roles")
                or item.get("fam_roles")
                or item.get("values")
                or []
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            binder, roles = item
        else:
            continue
        if binder:
            out[str(binder)] = tuple(roles)
    return out


def _predicates(value) -> tuple[tuple[str, str, str], ...]:
    out = []
    for pred in value or ():
        if isinstance(pred, str):
            if "!=" in pred:
                left, right = pred.split("!=", 1)
                out.append((left.strip(), "!=", right.strip()))
            elif "==" in pred:
                left, right = pred.split("==", 1)
                out.append((left.strip(), "==", right.strip()))
        elif isinstance(pred, dict):
            out.append((str(pred.get("slot", pred.get("lhs"))), str(pred["op"]), str(pred["rhs"])))
        else:
            out.append(tuple(str(x) for x in pred))
    return tuple(out)


def _ref_template(payload: dict, noisy_kind: str, demand_kind: str, link_directions: tuple[str, ...]) -> RefTemplate:
    t = RefTemplate(**payload)
    if t.binder == "node" and t.role.startswith("measurement_"):
        return RefTemplate(t.binder, t.role, f"{noisy_kind}_{{X}}_{t.role[len('measurement_'):]}")
    if t.binder == "node" and t.role == "demand_self":
        return RefTemplate(t.binder, t.role, f"{demand_kind}_{{X}}_{{X}}")
    if t.binder == "link" and t.role == "demand":
        return RefTemplate(t.binder, t.role, f"{demand_kind}_{{X}}_{{Y}}")
    if t.binder == "link" and t.role == "demand_rev":
        return RefTemplate(t.binder, t.role, f"{demand_kind}_{{Y}}_{{X}}")
    m = re.fullmatch(r"o(\d+)(_rev)?", t.role)
    if t.binder == "link" and m:
        idx = int(m.group(1))
        if idx < len(link_directions):
            direction = link_directions[idx]
            if m.group(2):
                return RefTemplate(t.binder, t.role, f"{noisy_kind}_{{Y}}_{direction}_{{X}}")
            return RefTemplate(t.binder, t.role, f"{noisy_kind}_{{X}}_{direction}_{{Y}}")
    return t


def _family_selector(payload: dict) -> FamilySelector:
    role = payload["family_role"]
    binder = payload["binder"]
    predicates = _predicates(payload.get("predicates", ()))
    if role == "demand_row":
        predicates = (("source", "==", "X"), ("destination", "!=", "X"))
    elif role == "demand_col":
        predicates = (("destination", "==", "X"), ("source", "!=", "X"))
    elif role == "all_demand":
        predicates = (("source", "!=", "@destination"),)
    elif binder == "node" and role.startswith("fam_"):
        predicates = (("source", "==", "X"),)
    elif binder == "network" and role.startswith("all_measurement_"):
        predicates = ()
    else:
        predicates = tuple(p for p in predicates if len(p) == 3 and p[0] in PRED_SLOTS)
    return FamilySelector(
        binder=binder,
        family_role=role,
        match_kind=payload["match_kind"],
        match_direction=payload.get("match_direction"),
        predicates=predicates,
    )


def _first_link_direction(ref_templates: tuple[RefTemplate, ...]) -> str:
    for t in ref_templates:
        if t.binder == "link" and t.role == "o0":
            m = re.search(r"_\{X\}_(.+)_\{Y\}$", t.template)
            if m:
                return m.group(1)
    return ""


def _link_directions(patterns: tuple[ColumnPattern, ...], noisy_kind: str) -> tuple[str, ...]:
    directions = {
        p.direction
        for p in patterns
        if p.kind == noisy_kind and p.peer_group and p.direction not in ("directed", "link")
    }
    return tuple(sorted(directions))


def _single_node_directions(patterns: tuple[ColumnPattern, ...], noisy_kind: str) -> tuple[str, ...]:
    directions = {
        p.direction
        for p in patterns
        if p.kind == noisy_kind and not p.peer_group and len(p.node_groups) == 1
    }
    return tuple(sorted(directions))


def _column_pattern(payload: dict) -> ColumnPattern:
    p = dict(payload)
    for key in ("source_group", "destination_group", "peer_group"):
        if p.get(key) is None:
            p[key] = ""

    index_to_name: dict[int, str] = {}
    token_groups = p.get("token_groups", ())
    if isinstance(token_groups, dict):
        for name, index in token_groups.items():
            if isinstance(index, int):
                index_to_name[index] = str(name)
        p["token_groups"] = tuple(str(k) for k in token_groups)
    else:
        tokens = tuple(str(x) for x in token_groups or ())
        p["token_groups"] = tokens
        for idx, name in enumerate(tokens, start=1):
            index_to_name.setdefault(idx, name)

    for key, fallback in (("source_group", "source"), ("destination_group", "destination"), ("peer_group", "peer")):
        value = p.get(key)
        if isinstance(value, int):
            index_to_name.setdefault(value, fallback)

    node_groups = p.get("node_groups", ())
    if isinstance(node_groups, dict):
        for name, index in node_groups.items():
            if isinstance(index, int):
                index_to_name[index] = str(name)
        p["node_groups"] = tuple(str(k) for k in node_groups)
    else:
        raw_nodes = tuple(node_groups or ())
        if raw_nodes and all(not isinstance(value, int) for value in raw_nodes):
            for idx, name in enumerate(raw_nodes, start=1):
                index_to_name.setdefault(idx, str(name))
        normalized_nodes = []
        for value in raw_nodes:
            if isinstance(value, int):
                normalized_nodes.append(index_to_name.setdefault(value, f"g{value}"))
            else:
                normalized_nodes.append(str(value))
        p["node_groups"] = tuple(normalized_nodes)

    node_group_set = set(p["node_groups"])
    p["token_groups"] = tuple(g for g in p["token_groups"] if g in node_group_set)

    for key, fallback in (("source_group", "source"), ("destination_group", "destination"), ("peer_group", "peer")):
        value = p.get(key)
        if isinstance(value, int):
            p[key] = index_to_name.setdefault(value, fallback)
        elif value is None:
            p[key] = ""
        else:
            p[key] = str(value) if value else ""

    if not index_to_name:
        ordered_groups = [p.get("source_group"), p.get("destination_group"), p.get("peer_group")]
        for idx, name in enumerate((g for g in ordered_groups if g), start=1):
            index_to_name[idx] = name

    if p.get("matcher") == "regex":
        p["regex"] = re.sub(r"\(\?<([A-Za-z_][A-Za-z0-9_]*)>", r"(?P<\1>", p.get("regex", ""))
        p["regex"] = _name_unnamed_regex_groups(p.get("regex", ""), index_to_name)
        declared = set(re.compile(p["regex"]).groupindex)
        if not p["node_groups"] and p.get("source_group") in declared and (p.get("destination_group") in declared or p.get("peer_group") in declared):
            p["node_groups"] = tuple(g for g in (p.get("source_group"), p.get("destination_group") or p.get("peer_group")) if g in declared)
        if any(g not in declared for g in p["node_groups"]):
            candidate_nodes = tuple(
                g for g in (p.get("source_group"), p.get("destination_group"), p.get("peer_group")) if g in declared
            )
            if candidate_nodes:
                p["node_groups"] = candidate_nodes
        p["token_groups"] = tuple(g for g in p["token_groups"] if g in declared)
        if not p["token_groups"]:
            p["token_groups"] = tuple(g for g in p["node_groups"] if g in declared)

    p["split_slots"] = tuple(p.get("split_slots", ("source", "destination"))[:2])
    return ColumnPattern(**p)


def _name_unnamed_regex_groups(regex: str, index_to_name: dict[int, str]) -> str:
    if "?P<" in regex or not index_to_name:
        return regex
    out = []
    group_index = 0
    i = 0
    while i < len(regex):
        ch = regex[i]
        if ch == "\\":
            out.append(regex[i:i + 2])
            i += 2
            continue
        if ch == "(" and not regex.startswith("(?", i):
            group_index += 1
            name = index_to_name.get(group_index, f"g{group_index}")
            out.append(f"(?P<{name}>")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)
