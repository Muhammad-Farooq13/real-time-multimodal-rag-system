from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.core.config import settings
from app.services.embeddings import Embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync local index JSON into Pinecone")
    parser.add_argument("--input", type=Path, default=Path("data/processed/index.json"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--namespace", type=str, default=settings.pinecone_namespace)
    return parser.parse_args()


def _list_index_names(client) -> set[str]:
    indexes = client.list_indexes()
    if hasattr(indexes, "names"):
        return set(indexes.names())
    names: set[str] = set()
    try:
        for item in indexes:
            if isinstance(item, dict) and "name" in item:
                names.add(str(item["name"]))
    except TypeError:
        return set()
    return names


def _create_index_if_missing(client, dimension: int) -> None:
    from pinecone import ServerlessSpec  # type: ignore

    names = _list_index_names(client)
    if settings.pinecone_index_name in names:
        return
    client.create_index(
        name=settings.pinecone_index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )


def chunks(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    args = parse_args()
    if not settings.pinecone_api_key:
        raise SystemExit("PINECONE_API_KEY is required in .env")
    if not args.input.exists():
        raise SystemExit(f"Input index file not found: {args.input}")

    from pinecone import Pinecone  # type: ignore

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("Input index is empty")

    embedder = Embedder()
    first_vec = embedder.embed(str(rows[0].get("text", "")))

    client = Pinecone(api_key=settings.pinecone_api_key)
    _create_index_if_missing(client, dimension=int(first_vec.shape[0]))
    index = client.Index(settings.pinecone_index_name)

    batch_size = max(1, args.batch_size)
    total = 0
    for batch in chunks(rows, batch_size):
        vectors = []
        for row in batch:
            text = str(row.get("text", ""))
            if not text:
                continue
            vec = embedder.embed(text)
            vectors.append(
                {
                    "id": str(row.get("doc_id", f"doc-{total}")),
                    "values": vec.tolist(),
                    "metadata": {
                        "text": text,
                        "modality": str(row.get("modality", "text")),
                    },
                }
            )

        if not vectors:
            continue

        last_exc: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                index.upsert(vectors=vectors, namespace=args.namespace)
                total += len(vectors)
                print(f"Upserted batch size={len(vectors)} total={total}")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                wait_s = 2**attempt
                print(f"Retry {attempt}/{args.retries} after error: {exc}")
                time.sleep(wait_s)

        if last_exc is not None:
            raise SystemExit(f"Upsert failed after retries: {last_exc}")

    print(f"Sync complete. Upserted {total} vectors into index={settings.pinecone_index_name}")


if __name__ == "__main__":
    main()
