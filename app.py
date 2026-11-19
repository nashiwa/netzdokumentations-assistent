from time import perf_counter

import json
from pathlib import Path
import streamlit as st
from openai import OpenAIError
from src.answer import generate_answer
from src.retrieve import (
    filter_asset_chunks,
    filter_current_chunks,
    load_chunks,
    search_chunks,
    build_evidence
)
# Reload the search data when either file changes.
@st.cache_data(show_spinner=False)
def load_search_data(chunks_path, manifest_path, chunks_modified, manifest_modified):
    print("Reading search files from disk", flush=True)
    chunks = load_chunks(chunks_path)
    documents = json.loads(manifest_path.read_text(encoding="utf-8"))
    return chunks, documents

st.title("Netzdokumentations-Assistent")

# Load the station IDs from the asset catalog.
project_root = Path(__file__).resolve().parent
assets_path = project_root / "data" / "demo" / "assets.json"
assets = json.loads(assets_path.read_text(encoding="utf-8"))
asset_ids = sorted(asset["asset_id"] for asset in assets)

selected_asset_id = st.selectbox("Station auswählen", asset_ids)

# Pass the demo station coordinates and colors to the browser.
markers = [
    {
        "lat": asset["latitude"],
        "lon": asset["longitude"],
        "color": "#F97316" if asset["asset_id"] == selected_asset_id else "#3B82F6",
    }
    for asset in assets
]

map_html = """
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<div id="map" style="height: 350px;"></div>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
    const markers = __MARKERS__;
    const map = L.map("map");
    const points = markers.map(marker => [marker.lat, marker.lon]);

    map.fitBounds(points, {padding: [25, 25], maxZoom: 13});

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    for (const marker of markers) {
        L.circleMarker([marker.lat, marker.lon], {
            radius: 7,
            color: marker.color,
            fillOpacity: 0.9
        }).addTo(map);
    }
</script>
"""

st.subheader("Karte der Demostationen")
st.iframe(map_html.replace("__MARKERS__", json.dumps(markers)), height=370)

with st.form("question_form"):
    question = st.text_input("Frage eingeben")
    submitted = st.form_submit_button("Frage stellen")

if submitted:
    question = question.strip()

    if not question:
        st.warning("Bitte gib eine Frage ein.")
        st.stop()

    # Find the data files relative to app.py.
    project_root = Path(__file__).resolve().parent
    chunks_path = project_root / "data" / "processed" / "chunks.jsonl"
    manifest_path = project_root / "data" / "demo" / "manifest.json"

    chunks, documents = load_search_data(
        chunks_path,
        manifest_path,
        chunks_path.stat().st_mtime_ns,
        manifest_path.stat().st_mtime_ns,
    )

    # Search only current documents linked to the selected station.
    search_start = perf_counter()
    asset_chunks = filter_asset_chunks(chunks, selected_asset_id)
    current_chunks = filter_current_chunks(asset_chunks, documents)
    top_chunks = search_chunks(question, selected_asset_id, current_chunks)
    search_seconds = perf_counter() - search_start

    if not top_chunks:
        st.info("Keine passenden Dokumentabschnitte gefunden.")
        st.stop()

    evidence, allowed_citations = build_evidence(top_chunks)

    try:
        with st.spinner("Antwort wird erstellt …"):
            answer_start = perf_counter()
            answer = generate_answer(
                question, selected_asset_id, evidence, allowed_citations
            )
            answer_seconds = perf_counter() - answer_start

    except OpenAIError:
        st.error("Die Antwort konnte nicht erstellt werden. Bitte versuche es erneut.")
    except ValueError as error:
        st.error(str(error))
    else:
        st.subheader("Antwort")
        st.write(answer)
        st.caption(f"Suche: {search_seconds:.3f} s · Antwort: {answer_seconds:.2f} s")

    st.subheader("Gefundene Quellen")
    # Look up document titles and versions by document ID.
    documents_by_id = {document["document_id"]: document for document in documents}

    for score, chunk in top_chunks:
        document_id = chunk["document_id"]
        document = documents_by_id[document_id]
        label = f"{document_id} · {document['title']} · Seite {chunk['page_number']}"

        with st.expander(label):
            st.caption(f"Version {document['version']} · Score {score}")
            st.text(chunk["text"])