from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag_app.config import AppConfig
from rag_app.ingestion.chunker import build_chunker, enrich_chunk_metadata


@dataclass(slots=True)
class IngestionResult:
    documents: list[Document]
    errors: list[str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_uploaded_files(uploaded_files: Iterable, data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    uploads = list(uploaded_files)
    incoming_names = {uploaded_file.name for uploaded_file in uploads}

    for existing in data_dir.glob("*.pdf"):
        if existing.name not in incoming_names:
            existing.unlink(missing_ok=True)

    saved_paths: list[Path] = []
    for uploaded_file in uploads:
        destination = data_dir / uploaded_file.name
        payload = uploaded_file.getvalue()

        should_write = True
        if destination.exists():
            current_payload = destination.read_bytes()
            should_write = _sha256_bytes(current_payload) != _sha256_bytes(payload)

        if should_write:
            with destination.open("wb") as handle:
                handle.write(payload)
        saved_paths.append(destination)
    return saved_paths


def ingest_documents(config: AppConfig, source_paths: list[Path] | None = None) -> IngestionResult:
    pdf_paths = source_paths or sorted(config.data_dir.glob("*.pdf"))
    raw_documents: list[Document] = []
    errors: list[str] = []

    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            for document in loader.load():
                raw_documents.append(enrich_chunk_metadata(document, pdf_path))
        except Exception as exc:
            errors.append(f"{pdf_path.name}: {exc}")

    if not raw_documents:
        return IngestionResult(documents=[], errors=errors)

    chunker = build_chunker(config.chunk_size, config.chunk_overlap)
    chunked_documents = chunker.split_documents(raw_documents)

    for index, document in enumerate(chunked_documents):
        document.metadata["chunk_id"] = index

    return IngestionResult(documents=chunked_documents, errors=errors)
