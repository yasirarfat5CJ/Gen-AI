from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    embeddings_model_id: str = field(
        default_factory=lambda: os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    )
    chat_model_id: str = field(
        default_factory=lambda: os.getenv("BEDROCK_CHAT_MODEL", "meta.llama3-70b-instruct-v1:0")
    )
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    index_dir: Path = field(default_factory=lambda: BASE_DIR / "faiss_index")
    cache_dir: Path = field(default_factory=lambda: BASE_DIR / ".cache")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 18
    final_k: int = 8
    max_context_chars: int = 18000
    min_similarity_score: float = 0.2
    retriever_weights: tuple[float, float] = (0.45, 0.55)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    config = AppConfig()
    config.ensure_dirs()
    return config
