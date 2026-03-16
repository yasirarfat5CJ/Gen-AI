from __future__ import annotations

import random
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_aws import BedrockEmbeddings, ChatBedrock

from rag_app.config import AppConfig


THROTTLING_ERROR_CODES = {
    "Throttling",
    "ThrottlingException",
    "TooManyRequestsException",
    "RequestLimitExceeded",
}


def is_bedrock_throttling_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in THROTTLING_ERROR_CODES
    message = str(exc).lower()
    return "throttl" in message or "too many requests" in message


def invoke_with_backoff(operation, *, max_attempts: int = 6, base_delay: float = 1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if not is_bedrock_throttling_error(exc) or attempt == max_attempts:
                raise
            sleep_seconds = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(sleep_seconds)


def get_bedrock_client(config: AppConfig):
    return boto3.client(
        "bedrock-runtime",
        region_name=config.aws_region,
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=20,
        ),
    )


def get_embeddings(config: AppConfig) -> BedrockEmbeddings:
    return BedrockEmbeddings(
        model_id=config.embeddings_model_id,
        client=get_bedrock_client(config),
    )


def get_chat_model(config: AppConfig, streaming: bool = False) -> ChatBedrock:
    return ChatBedrock(
        model_id=config.chat_model_id,
        client=get_bedrock_client(config),
        model_kwargs={"max_gen_len": 768, "temperature": 0.1},
        streaming=streaming,
    )
