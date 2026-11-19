import json
from pathlib import Path
from pypdf import PdfReader


REQUIRED_DOCUMENT_FIELDS = [
    "document_id",
    "filename",
    "title",
    "asset_ids",
    "version",
    "valid_from",
    "lifecycle_status",
    "supersedes_document_id",
    "language",
]


def count_ids(records, id_field):
    """Count how many times each ID appears."""
    counts = {}

    for record in records:
        record_id = record.get(id_field)
        if isinstance(record_id, str) and record_id.strip():
            counts[record_id] = counts.get(record_id, 0) + 1

    return counts


def validate_assets(assets, document_id_counts):
    """Check asset IDs, coordinates, and document links."""
    asset_id_counts = count_ids(assets, "asset_id")
    valid_assets = []
    errors = []

    for asset in assets:
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id.strip():
            errors.append("Asset is missing a valid asset_id.")
            continue

        if asset_id_counts[asset_id] > 1:
            errors.append(f"Duplicate asset ID: {asset_id}")
            continue

        latitude = asset.get("latitude")
        longitude = asset.get("longitude")

        if type(latitude) not in (int, float) or not -90 <= latitude <= 90:
            errors.append(f"Asset {asset_id} has an invalid latitude.")
            continue

        if type(longitude) not in (int, float) or not -180 <= longitude <= 180:
            errors.append(f"Asset {asset_id} has an invalid longitude.")
            continue

        document_ids = asset.get("document_ids")
        if not isinstance(document_ids, list):
            errors.append(f"Asset {asset_id} must have a document_ids list.")
            continue

        # Each linked ID should point to exactly one entry in the manifest.
        bad_links = []
        for document_id in document_ids:
            if (
                not isinstance(document_id, str)
                or document_id_counts.get(document_id, 0) != 1
            ):
                bad_links.append(str(document_id))

        if bad_links:
            errors.append(
                f"Asset {asset_id} links to missing or duplicate document IDs: "
                + ", ".join(bad_links)
            )
            continue

        valid_assets.append(asset)

    return valid_assets, errors


def validate_documents(documents, document_id_counts, known_asset_ids, demo_folder):
    """Check required fields, files, asset links, and version links."""
    valid_documents = []
    errors = []

    for document in documents:
        document_id = document.get("document_id", "unknown ID")
        missing_fields = []

        for field in REQUIRED_DOCUMENT_FIELDS:
            if field not in document:
                missing_fields.append(field)

        if missing_fields:
            errors.append(
                f"Document {document_id} is missing required fields: "
                + ", ".join(missing_fields)
            )
            continue

        if not isinstance(document_id, str) or not document_id.strip():
            errors.append("Document has an invalid document_id.")
            continue

        if document_id_counts[document_id] > 1:
            errors.append(f"Duplicate document ID: {document_id}")
            continue

        filename = document["filename"]
        if not isinstance(filename, str) or not filename.strip():
            errors.append(f"Document {document_id} has an invalid filename.")
            continue

        document_path = demo_folder / filename
        if not document_path.is_file():
            errors.append(f"File '{filename}' not found for document {document_id}")
            continue

        asset_ids = document["asset_ids"]
        if not isinstance(asset_ids, list) or not asset_ids:
            errors.append(f"Document {document_id} must link to at least one asset.")
            continue

        # Use the asset IDs that passed validation.
        unknown_asset_ids = []
        for asset_id in asset_ids:
            if not isinstance(asset_id, str) or asset_id not in known_asset_ids:
                unknown_asset_ids.append(str(asset_id))

        if unknown_asset_ids:
            errors.append(
                f"Document {document_id} links to unknown or invalid assets: "
                + ", ".join(unknown_asset_ids)
            )
            continue

        # If this document replaces another one, check the older document's ID.
        replaced_id = document["supersedes_document_id"]
        if replaced_id is not None:
            if (
                not isinstance(replaced_id, str)
                or replaced_id == document_id
                or document_id_counts.get(replaced_id, 0) != 1
            ):
                errors.append(
                    f"Document {document_id} has an invalid version link: "
                    f"{replaced_id}"
                )
                continue

        valid_documents.append(document)

    return valid_documents, errors


def main():
    # Find the demo folder relative to this script.
    project_root = Path(__file__).resolve().parent.parent
    demo_folder = project_root / "data" / "demo"

    manifest_text = (demo_folder / "manifest.json").read_text(encoding="utf-8")
    documents = json.loads(manifest_text)

    assets_text = (demo_folder / "assets.json").read_text(encoding="utf-8")
    assets = json.loads(assets_text)

    # Validate assets first because document validation needs their IDs.
    document_id_counts = count_ids(documents, "document_id")
    valid_assets, asset_errors = validate_assets(assets, document_id_counts)
    known_asset_ids = {asset["asset_id"] for asset in valid_assets}

    valid_documents, document_errors = validate_documents(
        documents, document_id_counts, known_asset_ids, demo_folder
    )

    page_records = []

    for document in valid_documents:
        document_path = demo_folder / document["filename"]
        document_reader = PdfReader(document_path)

        for page_number, page in enumerate(document_reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                print(
                    f"WARNING: No readable text in document "
                    f"{document['document_id']}, page {page_number}"
                )
                continue

            # Keep the source with the text so we can cite it later.
            page_record = {
                "document_id": document["document_id"],
                "page_number": page_number,
                "text": text,
                "asset_ids": document["asset_ids"],
                "version": document["version"],
            }
            page_records.append(page_record)

    # The demo pages are short, so each page becomes one chunk.
    chunks = []

    for record in page_records:
        document_id = record["document_id"]
        page_number = record["page_number"]

        chunk = {
            "chunk_id": f"{document_id}-p{page_number}-c1",
            "document_id": document_id,
            "page_number": page_number,
            "text": record["text"],
            "asset_ids": record["asset_ids"],
            "version": record["version"],
        }
        chunks.append(chunk)

    processed_folder = project_root / "data" / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)

    chunks_path = processed_folder / "chunks.jsonl"

    with chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    for error in asset_errors + document_errors:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()