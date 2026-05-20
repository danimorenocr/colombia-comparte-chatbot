"""Suite de validación de calidad para /preguntar.

Enfocada en verificar si las respuestas están bien o mal:
- calidad funcional (idioma, lead, session_id, reply útil)
- grounding RAG (fuentes y chunks)
- restricciones del asistente (sin términos prohibidos en respuesta)
- casos negativos (payload inválido)

Uso:
    python test_generacion.py
    python test_generacion.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


Validator = Callable[[int, dict | None], tuple[bool, str]]


@dataclass
class TestCase:
    name: str
    category: str
    payload: dict
    validator: Validator


@dataclass
class TestResult:
    name: str
    category: str
    ok: bool
    detail: str
    status_code: int


def post_json(url: str, payload: dict, timeout: int = 30) -> tuple[int, dict | None]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.getcode(), json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        payload = json.loads(body) if body else None
        return exc.code, payload


def must_have_non_empty_reply(
    expected_lang: str,
    expected_lead: bool,
    require_sources: bool = True,
    min_chunks: int = 1,
    forbidden_terms: tuple[str, ...] = ("contexto", "documento", "sección", "section", "document", "context"),
) -> Validator:
    def _validate(status: int, body: dict | None) -> tuple[bool, str]:
        if status != 200:
            return False, f"status esperado=200 obtenido={status}"
        if not body:
            return False, "body vacío"

        reply = (body.get("reply") or "").strip()
        lang = body.get("idioma_detectado")
        is_lead = bool(body.get("es_lead"))
        session_id = body.get("session_id")
        fuentes = body.get("fuentes") or []
        chunks = int(body.get("chunks_encontrados") or 0)

        contains_forbidden = any(term in reply.lower() for term in forbidden_terms)

        checks = [
            (bool(reply), "reply no vacío"),
            (lang == expected_lang, f"idioma={expected_lang}"),
            (is_lead == expected_lead, f"es_lead={expected_lead}"),
            (bool(session_id), "session_id presente"),
            (not contains_forbidden, "respuesta sin palabras prohibidas del prompt"),
        ]
        if require_sources:
            checks.append((len(fuentes) > 0, "fuentes presentes"))
            checks.append((chunks >= min_chunks, f"chunks_encontrados>={min_chunks}"))

        failed = [label for ok, label in checks if not ok]
        if failed:
            return False, f"falló {', '.join(failed)}"

        return True, f"ok | reply_chars={len(reply)} fuentes={len(fuentes)} chunks={chunks}"

    return _validate


def must_be_initial_greeting() -> Validator:
    def _validate(status: int, body: dict | None) -> tuple[bool, str]:
        if status != 200:
            return False, f"status esperado=200 obtenido={status}"
        if not body:
            return False, "body vacío"
        reply = (body.get("reply") or "").lower()
        ok = ("hola" in reply or "hi" in reply) and bool(body.get("session_id"))
        return (True, "ok | saludo inicial") if ok else (False, "saludo inicial inválido")

    return _validate


def must_fail_with_status(expected_status: int) -> Validator:
    def _validate(status: int, body: dict | None) -> tuple[bool, str]:
        if status != expected_status:
            return False, f"status esperado={expected_status} obtenido={status}"
        detail = None
        if isinstance(body, dict):
            detail = body.get("detail")
        return True, f"ok | status={status} detail={detail}"

    return _validate


def must_return_fallback(expected_lang: str) -> Validator:
    fallback_fragments = {
        "es": "no tengo suficiente información",
        "en": "don't have enough information",
    }

    def _validate(status: int, body: dict | None) -> tuple[bool, str]:
        if status != 200:
            return False, f"status esperado=200 obtenido={status}"
        if not body:
            return False, "body vacío"

        reply = (body.get("reply") or "").lower()
        lang = body.get("idioma_detectado")
        sources = body.get("fuentes") or []
        chunks = int(body.get("chunks_encontrados") or 0)

        expected_fragment = fallback_fragments[expected_lang]
        checks = [
            (lang == expected_lang, f"idioma={expected_lang}"),
            (expected_fragment in reply, "mensaje fallback esperado"),
            (chunks == 0, "chunks_encontrados=0"),
            (len(sources) == 0, "fuentes vacías"),
        ]
        failed = [label for ok, label in checks if not ok]
        if failed:
            return False, f"falló {', '.join(failed)}"
        return True, "ok | fallback correcto"

    return _validate


def run_case(base_url: str, case: TestCase, timeout: int) -> TestResult:
    try:
        status, body = post_json(f"{base_url}/preguntar", case.payload, timeout=timeout)
    except Exception as exc:  # pragma: no cover
        return TestResult(case.name, case.category, False, f"error conexión: {exc}", 0)

    ok, detail = case.validator(status, body)
    return TestResult(case.name, case.category, ok, detail, status)


def build_test_cases() -> list[TestCase]:
    return [
        TestCase(
            name="Saludo inicial",
            category="functional",
            payload={
                "message": "",
                "country": "Colombia",
                "language": "es",
                "action": "initial",
                "history": [],
            },
            validator=must_be_initial_greeting(),
        ),
        TestCase(
            name="Generación ES lead",
            category="functional",
            payload={
                "message": "Tengo un emprendimiento y quiero apoyo para crecer y vender más.",
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            },
            validator=must_have_non_empty_reply("es", True, require_sources=True, min_chunks=1),
        ),
        TestCase(
            name="Generación EN lead",
            category="functional",
            payload={
                "message": "I have a startup and need support to scale in Latin America.",
                "country": "Chile",
                "language": "en",
                "action": "chat",
                "history": [],
            },
            validator=must_have_non_empty_reply("en", True, require_sources=True, min_chunks=1),
        ),
        TestCase(
            name="Generación ES no lead",
            category="functional",
            payload={
                "message": "Hola, qué testimonios han publicado últimamente?",
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            },
            validator=must_have_non_empty_reply("es", False, require_sources=True, min_chunks=1),
        ),
        TestCase(
            name="Fallback sin contexto (ES)",
            category="quality",
            payload={
                "message": "blorptak zyrnq 8844 ???",
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            },
            validator=must_return_fallback("es"),
        ),
        TestCase(
            name="Error mensaje vacío en chat",
            category="negative",
            payload={
                "message": "",
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            },
            validator=must_fail_with_status(400),
        ),
        TestCase(
            name="Error límite 500 chars",
            category="negative",
            payload={
                "message": "x" * 501,
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            },
            validator=must_fail_with_status(400),
        ),
        TestCase(
            name="Error schema sin message",
            category="negative",
            payload={
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            },
            validator=must_fail_with_status(422),
        ),
        TestCase(
            name="Error schema message número",
            category="negative",
            payload={
                "message": 12345,
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": [],
            },
            validator=must_fail_with_status(422),
        ),
        TestCase(
            name="Error schema history inválido",
            category="negative",
            payload={
                "message": "Hola",
                "country": "Colombia",
                "language": "es",
                "action": "chat",
                "history": "esto debería ser lista",
            },
            validator=must_fail_with_status(422),
        ),
    ]


def summarize(results: list[TestResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.ok)

    by_category: dict[str, dict] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"total": 0, "passed": 0}
        by_category[r.category]["total"] += 1
        by_category[r.category]["passed"] += int(r.ok)

    for category, values in by_category.items():
        values["pass_rate"] = round((values["passed"] / values["total"]) * 100, 2)

    status_counter: dict[int, int] = {}
    for r in results:
        status_counter[r.status_code] = status_counter.get(r.status_code, 0) + 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "by_category": by_category,
        "status_counter": status_counter,
    }


def print_results(results: list[TestResult], summary: dict):
    print("\n=== RESULTADOS POR CASO ===")
    for r in results:
        badge = "PASS" if r.ok else "FAIL"
        print(
            f"[{badge}] {r.name} | category={r.category} | "
            f"status={r.status_code} | {r.detail}"
        )

    print("\n=== RESUMEN FUNCIONAL ===")
    print(f"Total: {summary['total']} | Passed: {summary['passed']} | Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']}%")
    print(f"Status codes: {summary['status_counter']}")
    print(f"Por categoría: {summary['by_category']}")


def save_report(path: str, base_url: str, results: list[TestResult], summary: dict):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "summary": summary,
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "ok": r.ok,
                "detail": r.detail,
                "status_code": r.status_code,
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Suite de validación de calidad de respuestas")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="URL base de la API")
    parser.add_argument("--timeout", type=int, default=35, help="Timeout por request en segundos")
    parser.add_argument("--report", default="test_generacion_report.json", help="Ruta del reporte JSON")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Ejecutando suite integral contra: {base_url}")

    cases = build_test_cases()
    results: list[TestResult] = [run_case(base_url, case, args.timeout) for case in cases]
    summary = summarize(results)

    print_results(results, summary)
    save_report(args.report, base_url, results, summary)
    print(f"\nReporte guardado en: {args.report}")

    functional_rate = summary["by_category"].get("functional", {}).get("pass_rate", 0)
    negative_rate = summary["by_category"].get("negative", {}).get("pass_rate", 0)
    quality_rate = summary["by_category"].get("quality", {}).get("pass_rate", 0)

    # Criterio objetivo de aceptación centrado en calidad de respuesta.
    accepted = (
        summary["pass_rate"] >= 90
        and functional_rate >= 90
        and negative_rate >= 90
        and quality_rate >= 90
    )

    if accepted:
        print("\nVEREDICTO: APROBADO ✅")
        return 0

    print("\nVEREDICTO: REQUIERE MEJORAS ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())
