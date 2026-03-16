from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_chunker(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def enrich_chunk_metadata(doc: Document, source_path: Path) -> Document:
    metadata = dict(doc.metadata)
    metadata["source"] = str(source_path)
    metadata["filename"] = source_path.name
    metadata["page"] = metadata.get("page", 0) + 1
    metadata["file_type"] = source_path.suffix.lstrip(".").lower()
    doc.metadata = metadata
    return doc
