"""Bug real reproducible en Exp03/Exp04: ``final_claim_audit.csv``
mostraba ``evidence_pair_count=0``/``evidence_present=False`` para
TODOS los claims, aunque 07 sí verificó con evidencia real.

Causa raíz confirmada (con el código real, no supuesta):
``build_agent08_input_from_committed_agent07`` (``src/adapters/
evaluation_upstream.py``) leía ``bundle["claim_evidence_traceability_
rows"]`` -- el artefacto REAL de 07 que contiene los pares claim ->
evidencia -> source_filename -> chunk_id (``ClaimEvidenceTraceability
Row``, ``src/tools/verification/traceability.py``) -- únicamente para
VALIDAR su presencia (``isinstance(evidence_rows, list)``), pero JAMÁS
usaba su contenido al construir ``compatibility_rows``: cada claim
producía UNA fila con solo metadatos de veredicto/riesgo, sin
``source_filename``/``chunk_id`` en ninguna parte. ``build_claim_audit_
rows`` (``src/tools/evaluation/claim_citation_audit.py``) solo computa
``evidence_pairs`` cuando esas DOS columnas existen en el conjunto de
columnas del ``traceability_rows`` recibido
(``REQUIRED_TRACEABILITY_COLUMNS = {"claim_id", "verdict"}`` -- ninguna
de las dos era obligatoria) -- al no existir NUNCA, ``evidence_pairs``
quedaba vacío siempre, para todos los claims, sin ningún error visible.

El join correcto es por ``claim_id`` -- seguro DENTRO de un mismo
bundle/ejecución de 07 (ambas colecciones las produce la MISMA corrida
sobre los MISMOS claims; la inestabilidad de ``claim_id`` es un
problema de comparar ENTRE rondas, nunca dentro de un artefacto). Solo
se incluye evidencia con ``used_in_original_verification=True`` (la
rechazada no cuenta como soporte) y con ``source_filename``/
``chunk_id`` no vacíos (fail-closed: nunca se inventa un par)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_qualitative_correction_keyerror_and_return as R  # noqa: E402
import test_verification_stagespec_integration as T  # noqa: E402

from src.adapters.evaluation_upstream import build_agent08_input_from_committed_agent07  # noqa: E402
from src.tools.evaluation.claim_citation_audit import build_claim_audit_rows  # noqa: E402

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


FIXTURE = REPO_ROOT / "tests" / "evaluation" / "fixtures" / "agent07_direct"


def _load_fixture_bundle():
    return json.loads((FIXTURE / "provisional_verification_traceability_bundle.json").read_text())


def _write_agent07_directory(tmp_path, bundle):
    target = tmp_path / "agent07"
    shutil.copytree(FIXTURE, target)
    (target / "provisional_verification_traceability_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False)
    )
    return target


def _resolve_and_audit(tmp_path, bundle, *, extra_chunks=()):
    directory = _write_agent07_directory(tmp_path, bundle)
    upstream_input = build_agent08_input_from_committed_agent07(
        agent07_directory=directory,
        draft_json_path=directory / "state_of_art_draft.json",
        draft_markdown_path=directory / "state_of_art_draft.md",
    )
    draft = json.loads((directory / "state_of_art_draft.json").read_text())
    generated_text = " ".join(
        section.get("draft_text", "") for section in draft.get("sections", [])
    )
    chunks = [
        {"source_filename": "paper_s2c5.pdf", "chunk_id": "c1"},
        {"source_filename": "paper_s3c5_a.pdf", "chunk_id": "c1"},
        {"source_filename": "paper_s3c5_b.pdf", "chunk_id": "c1"},
        {"source_filename": "paper_s3c5_c.pdf", "chunk_id": "c1"},
        *extra_chunks,
    ]
    rows = build_claim_audit_rows(
        traceability_rows=[dict(r) for r in upstream_input.traceability_rows],
        generated_content_text=generated_text,
        valid_source_chunk_pairs={(c["source_filename"], c["chunk_id"]) for c in chunks},
    )
    return {row["claim_id"]: row for row in rows}


@scenario("EE01. Claim con evidencia válida en 07 -> evidence_pair_count > 0 en 08 (dato real del fixture, S3_C5 con evidencia usada)")
def test_claim_with_valid_evidence_has_nonzero_pair_count():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _load_fixture_bundle()
        rows_by_claim = _resolve_and_audit(Path(tmp), bundle)
        assert rows_by_claim["S3_C5"]["evidence_pair_count"] > 0
        assert rows_by_claim["S3_C5"]["evidence_present"] is True


@scenario("EE02. Múltiples evidencias válidas para el mismo claim -> conteo correcto, ninguna se pierde")
def test_multiple_valid_evidence_pairs_counted_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _load_fixture_bundle()
        real_count = sum(
            1 for r in bundle["claim_evidence_traceability_rows"]
            if r["claim_id"] == "S3_C5" and r.get("used_in_original_verification")
            and r.get("source_filename") and r.get("chunk_id")
        )
        assert real_count >= 2  # confirma que el fixture real tiene múltiples pares
        rows_by_claim = _resolve_and_audit(Path(tmp), bundle)
        assert rows_by_claim["S3_C5"]["evidence_pair_count"] == real_count


@scenario("EE03. Claim sin evidencia usada -> evidence_pair_count=0, evidence_present=False (nunca se inventa evidencia)")
def test_claim_without_evidence_stays_zero():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _load_fixture_bundle()
        # S2_C1 (NOT_EVALUATED) no tiene evidencia usada en el fixture real.
        assert not any(
            r["claim_id"] == "S2_C1" and r.get("used_in_original_verification")
            for r in bundle["claim_evidence_traceability_rows"]
        )
        rows_by_claim = _resolve_and_audit(Path(tmp), bundle)
        assert rows_by_claim["S2_C1"]["evidence_pair_count"] == 0
        assert rows_by_claim["S2_C1"]["evidence_present"] is False


@scenario("EE04. Par con source/chunk que NO está en los chunks reales -> invalid_evidence_pair_count > 0 (fail-closed, nunca se descarta en silencio)")
def test_invalid_pair_is_flagged_not_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _load_fixture_bundle()
        bundle["claim_evidence_traceability_rows"].append({
            "claim_id": "S2_C1", "section_id": "S2", "evidence_id": "E_FAKE",
            "source_filename": "paper_inexistente.pdf", "chunk_id": "chunk_inexistente",
            "text_fingerprint": "f" * 64, "usage_role": "SUPPORT",
            "authorized_for_section": True, "used_in_original_verification": True,
            "supports_original_claim": "NOT_EVALUATED",
        })
        rows_by_claim = _resolve_and_audit(Path(tmp), bundle)
        assert rows_by_claim["S2_C1"]["evidence_pair_count"] == 1
        assert rows_by_claim["S2_C1"]["invalid_evidence_pair_count"] == 1


@scenario("EE05. Join por claim_id dentro del mismo bundle: identidad estable (claim_uid) presente no rompe el join, y claims con más de un par de evidencia unen correctamente todos")
def test_join_by_claim_id_stable_within_bundle_with_claim_uid_present():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _load_fixture_bundle()
        # Simula un experimento ya migrado a identidad estable: los
        # claim_traceability_rows traen claim_uid, pero el join de
        # evidencia sigue siendo por claim_id (claim_evidence_
        # traceability_rows no versiona claim_uid) -- debe seguir
        # funcionando sin ambigüedad.
        for row in bundle["claim_traceability_rows"]:
            if row["claim_id"] == "S3_C5":
                row["claim_uid"] = "11111111-1111-1111-1111-111111111111"
        rows_by_claim = _resolve_and_audit(Path(tmp), bundle)
        assert rows_by_claim["S3_C5"]["evidence_pair_count"] > 0


@scenario("EE06. Identidad ambigua (evidencia con claim_id vacío) -> fail-closed, nunca se infiere a qué claim pertenece")
def test_ambiguous_claim_id_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _load_fixture_bundle()
        bundle["claim_evidence_traceability_rows"].append({
            "claim_id": "", "section_id": "S2", "evidence_id": "E_AMBIGUOUS",
            "source_filename": "paper_s2c5.pdf", "chunk_id": "c1",
            "text_fingerprint": "f" * 64, "usage_role": "SUPPORT",
            "authorized_for_section": True, "used_in_original_verification": True,
            "supports_original_claim": "NOT_EVALUATED",
        })
        directory = _write_agent07_directory(Path(tmp), bundle)
        try:
            build_agent08_input_from_committed_agent07(
                agent07_directory=directory,
                draft_json_path=directory / "state_of_art_draft.json",
                draft_markdown_path=directory / "state_of_art_draft.md",
            )
        except ValueError as exc:
            assert "AGENT08_CLAIM_ID_MISSING" in str(exc)
        else:
            raise AssertionError("una evidencia sin claim_id identificable debe fallar cerrado, nunca inferirse")


@scenario("EE07. PARTIAL_HALT (runtime_status=PARTIAL, el caso real Exp04) no elimina la evidencia existente -- se preserva igual que en el camino COMPLETED")
def test_partial_halt_preserves_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _load_fixture_bundle()
        assert bundle["aggregation_status"] == "PARTIAL"  # ya es el escenario PARTIAL_HALT real
        rows_by_claim = _resolve_and_audit(Path(tmp), bundle)
        assert rows_by_claim["S3_C5"]["evidence_pair_count"] > 0
        assert rows_by_claim["S2_C5"]["evidence_pair_count"] > 0


@scenario("EE08. Regresión con datos REALES de Exp05 (fixture real, no sintético): claims SUPPORTED con evidencia genuina producen evidence_pair_count>0, claims NOT_EVALUATED correctamente en 0, evidence_coverage se deriva del audit ya reparado")
def test_real_exp05_fixture_end_to_end():
    fixture = REPO_ROOT / "tests" / "evaluation" / "fixtures" / "agent07_exp05_real"
    from src.tools.evaluation.claim_citation_audit import compute_claim_factual_metrics

    upstream_input = build_agent08_input_from_committed_agent07(
        agent07_directory=fixture,
        draft_json_path=fixture / "state_of_art_draft.json",
        draft_markdown_path=fixture / "state_of_art_draft.md",
    )
    draft = json.loads((fixture / "state_of_art_draft.json").read_text())
    generated_text = " ".join(s.get("draft_text", "") for s in draft["sections"])

    bundle = json.loads((fixture / "provisional_verification_traceability_bundle.json").read_text())
    valid_pairs = {
        (r["source_filename"], r["chunk_id"])
        for r in bundle["claim_evidence_traceability_rows"]
        if r.get("used_in_original_verification")
    }
    rows = build_claim_audit_rows(
        traceability_rows=[dict(r) for r in upstream_input.traceability_rows],
        generated_content_text=generated_text,
        valid_source_chunk_pairs=valid_pairs,
    )
    rows_by_claim = {r["claim_id"]: r for r in rows}

    # Claims SUPPORTED reales del Exp05 -- deben tener evidencia real,
    # nunca 0 como en el bug original reportado.
    for claim_id in ("S2_C1", "S2_C2", "S2_C3", "S3_C1"):
        assert rows_by_claim[claim_id]["verdict"] == "supported"
        assert rows_by_claim[claim_id]["evidence_pair_count"] > 0
        assert rows_by_claim[claim_id]["evidence_present"] is True

    # Claims NOT_EVALUATED reales -- correctamente sin evidencia (07
    # nunca llegó a evaluarlos con soporte), nunca se inventa nada.
    for claim_id in ("S2_C5", "S3_C2", "S4_C2", "S5_C3"):
        assert rows_by_claim[claim_id]["verdict"] == "not_evaluated"
        assert rows_by_claim[claim_id]["evidence_pair_count"] == 0
        assert rows_by_claim[claim_id]["evidence_present"] is False

    # evidence_coverage se deriva del final_claim_audit YA reparado --
    # nunca de un cálculo separado que pudiera esconder el problema.
    metrics = compute_claim_factual_metrics(rows)
    expected_coverage = sum(1 for r in rows if r["evidence_present"]) / len(rows)
    assert metrics["evidence_coverage"] == expected_coverage
    assert metrics["evidence_coverage"] > 0.0  # nunca 0.0 artificial como en el bug real
    assert metrics["total_active_claims"] == 14


if __name__ == "__main__":
    for fn in (
        test_claim_with_valid_evidence_has_nonzero_pair_count,
        test_multiple_valid_evidence_pairs_counted_correctly,
        test_claim_without_evidence_stays_zero,
        test_invalid_pair_is_flagged_not_dropped,
        test_join_by_claim_id_stable_within_bundle_with_claim_uid_present,
        test_ambiguous_claim_id_fails_closed,
        test_partial_halt_preserves_evidence,
        test_real_exp05_fixture_end_to_end,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} escenarios OK")
    raise SystemExit(1 if failed else 0)
