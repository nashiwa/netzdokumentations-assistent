import json
import re
from pathlib import Path

from openai import OpenAIError

if __package__:
    from .answer import generate_answer
else:
    from answer import generate_answer


def normalize_search_text(text):
    """Normalize German spellings for keyword matching."""
    return text.casefold().translate(
        str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue"})
    )


def search_chunks(question, selected_asset_id, current_chunks):
    """Return up to three chunks ranked by keyword matches."""
    stop_words = {"welche", "bei", "ist", "fuer", selected_asset_id.casefold()}
    question_words = set(re.findall(r"[\w-]+", normalize_search_text(question)))
    search_words = question_words - stop_words

    scored_chunks = []
    for chunk in current_chunks:
        chunk_words = set(
            re.findall(r"[\w-]+", normalize_search_text(chunk["text"]))
        )
        matched_words = search_words & chunk_words

        if matched_words:
            scored_chunks.append((len(matched_words), chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return scored_chunks[:3]

def filter_current_chunks(asset_chunks, documents):
    """Keep chunks from documents marked as current in the manifest."""
    current_document_ids = {
        document["document_id"]
        for document in documents
        if document["lifecycle_status"] == "current"
    }

    return [
        chunk
        for chunk in asset_chunks
        if chunk["document_id"] in current_document_ids
    ]

def filter_asset_chunks(chunks, selected_asset_id):
    """Keep chunks linked to the selected asset."""
    return [
        chunk
        for chunk in chunks
        if selected_asset_id in chunk["asset_ids"]
    ]

def load_chunks(chunks_path):
    """Read one chunk from each non-empty line in the JSONL file."""
    chunks = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks

def build_evidence(top_chunks):
    """Build the source excerpts and collect their citation labels."""
    evidence_parts = []
    allowed_citations = set()

    for _, chunk in top_chunks:
        source = f"[{chunk['document_id']} p. {chunk['page_number']}]"
        evidence_parts.append(f"{source}\n{chunk['text']}")
        allowed_citations.add(source)

    return "\n\n".join(evidence_parts), allowed_citations

def main():
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "data" / "processed" / "chunks.jsonl"
    manifest_path = project_root / "data" / "demo" / "manifest.json"

    # Read one chunk from each non-empty line.
    chunks = load_chunks(chunks_path)

    selected_asset_id = input("Enter asset ID (e.g. S-140): ").strip().upper()

    # Keep only chunks linked to the selected asset.
    asset_chunks = filter_asset_chunks(chunks, selected_asset_id)

    if not asset_chunks:
        raise SystemExit(
            f"No document chunks found for asset ID '{selected_asset_id}'."
        )

     # Use the manifest to keep only current documents.
    documents = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_chunks = filter_current_chunks(asset_chunks, documents)

    if not current_chunks:
        raise SystemExit(
            f"No current document chunks found for asset ID '{selected_asset_id}'."
        )

    question = input("Enter your question: ").strip()

    if not question:
        raise SystemExit("Please enter a question.")

    top_chunks = search_chunks(question, selected_asset_id, current_chunks)

    if not top_chunks:
        raise SystemExit("No document excerpts matched your question.")

    for score, chunk in top_chunks:
        print(
            f"Source: {chunk['document_id']}, page {chunk['page_number']} "
            f"| Score: {score}"
        )

    evidence, allowed_citations = build_evidence(top_chunks)

    try:
        answer = generate_answer(
            question, selected_asset_id, evidence, allowed_citations
        )
    except OpenAIError:
        print("\nRetrieved excerpts:\n")
        print(evidence)
        raise SystemExit(
            "\nCould not generate an answer. Check the OpenAI connection or API setup."
        )
    except ValueError as error:
        raise SystemExit(str(error))

    print(f"\n{answer}")


if __name__ == "__main__":
    main()