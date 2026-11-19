import json
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

from openai import OpenAIError

from src.answer import generate_answer
from src.retrieve import (
    build_evidence,
    filter_asset_chunks,
    filter_current_chunks,
    load_chunks,
    search_chunks,
)


# Each case contains the station, question, and required sources.
CASES = [
    ("S-140", "Welche Angaben müssen für S-140 dokumentiert sein?", {"[D05 p. 1]"}),
    ("S-104", "Wie heißt die Anlage mit der Betriebsmittel-ID S-104?", {"[D01 p. 1]"}),
    ("S-140", "In welcher Region liegt die Demo Station Nordost?", {"[D02 p. 1]"}),
    ("S-104", "Seit wann gilt Version 2 der Dokumentationsnotiz für S-104?", {"[D04 p. 1]"}),
    ("S-104", "Welches frühere Dokument wird durch die aktuelle Dokumentationsnotiz für S-104 ersetzt?", {"[D04 p. 1]"}),
    ("S-140", "Gelten die gemeinsamen Dokumentationsregeln für S-140 auch für S-104?", {"[D05 p. 1]"}),
    ("S-140", "Was bedeutet „Zeichnungsreferenz“ in den gemeinsamen Dokumentationsregeln?", {"[D05 p. 1]"}),
    ("S-140", "Welche fehlenden Einträge wurden bei der Prüfung von S-140 festgestellt?", {"[D07 p. 1]"}),
    ("S-104", "Ist DEMO-Z-140-B die Zeichnungsreferenz für S-104?", {"[D06 p. 1]"}),
    ("S-104", "Nennen die aktuellen Unterlagen dieselbe Aufbewahrungsdauer für die Prüfprotokolle von S-104?", {"[D04 p. 1]", "[D06 p. 1]"}),
    ("S-140", "Wann findet die nächste Prüfung der Anlagendokumentation für S-140 statt?", set()),
    ("S-104", "Welcher Eintrag im Anlagenverzeichnis muss für S-104 noch ergänzt werden?", {"[D06 p. 1]"}),
]


def main():
    root = Path(__file__).resolve().parent
    chunks = load_chunks(root / "data" / "processed" / "chunks.jsonl")
    manifest_path = root / "data" / "demo" / "manifest.json"
    documents = json.loads(manifest_path.read_text(encoding="utf-8"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = root / f"evaluation-{timestamp}.jsonl"
    source_hits = []
    answer_times = []

    with output_path.open("w", encoding="utf-8") as output:
        for case_id, (asset_id, question, expected_sources) in enumerate(CASES, start=1):
            search_start = perf_counter()
            asset_chunks = filter_asset_chunks(chunks, asset_id)
            current_chunks = filter_current_chunks(asset_chunks, documents)
            top_chunks = search_chunks(question, asset_id, current_chunks)
            search_seconds = perf_counter() - search_start

            evidence, allowed_citations = build_evidence(top_chunks)
            retrieved_sources = [
                f"[{chunk['document_id']} p. {chunk['page_number']}]"
                for _, chunk in top_chunks
            ]

            source_hit = None
            if expected_sources:
                source_hit = expected_sources.issubset(allowed_citations)
                source_hits.append(source_hit)

            answer = None
            answer_seconds = None
            status = "no_excerpts"

            if top_chunks:
                answer_start = perf_counter()
                try:
                    answer = generate_answer(
                        question, asset_id, evidence, allowed_citations
                    )
                except (OpenAIError, ValueError) as error:
                    status = type(error).__name__
                else:
                    answer_seconds = perf_counter() - answer_start
                    answer_times.append(answer_seconds)
                    status = "answered"

            result = {
                "case": case_id,
                "station": asset_id,
                "question": question,
                "expected_sources": sorted(expected_sources),
                "retrieved_sources": retrieved_sources,
                "source_hit": source_hit,
                "search_seconds": search_seconds,
                "answer_seconds": answer_seconds,
                "answer": answer,
                "status": status,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(f"Case {case_id:02d}: {status}", flush=True)

    hits = sum(source_hits)
    total = len(source_hits)
    print(f"\nTop-three source hits: {hits}/{total} ({hits / total:.1%})")
    print(f"Answers generated: {len(answer_times)}/{len(CASES)}")
    if answer_times:
        print(f"Median answer time: {median(answer_times):.2f} s")
        print(f"Slowest answer time: {max(answer_times):.2f} s")
    print(f"Results saved to: {output_path.name}")


if __name__ == "__main__":
    main()