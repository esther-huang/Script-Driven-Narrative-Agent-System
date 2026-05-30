from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Iterable, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class ModelEmbedding:
    _model = None

    def __init__(self):
        self.model = None

    def _get_model(self):
        if ModelEmbedding._model is None:
            ModelEmbedding._model = SentenceTransformer("intfloat/multilingual-e5-base")
        self.model = ModelEmbedding._model
        return self.model

    def _embed(self, texts: List[str], prefix: str) -> List[List[float]]:
        processed = [f"{prefix}: {t.strip()}" for t in texts]
        embeddings = self._get_model().encode(processed, normalize_embeddings=True)
        return embeddings.tolist()

    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        return self.embed_documents(list(input))

    def name(self) -> str:
        return "multilingual_e5_base"

    def embed_query(self, input: str | List[str]):
        if isinstance(input, list):
            return self._embed(input, prefix="query")
        return self._embed([input], prefix="query")[0]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._embed(input, prefix="passage")


class ChromaStore:
    def __init__(self, path: str = '.chroma') -> None:
        self.embedding_fn = ModelEmbedding()
        self.client = self._init_client(path)
        self.collection = self.client.get_or_create_collection(
            name='narrative_knowledge',
            metadata={'hnsw:space': 'cosine'},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection('narrative_knowledge')
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name='narrative_knowledge',
            metadata={'hnsw:space': 'cosine'},
        )

    def _init_client(self, path: str) -> chromadb.PersistentClient:
        try:
            return chromadb.PersistentClient(path=path)
        except BaseException:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            try:
                return chromadb.PersistentClient(path=path)
            except BaseException:
                settings = Settings(is_persistent=True, persist_directory=path, allow_reset=True)
                client = chromadb.Client(settings=settings)
                try:
                    client.reset()
                except Exception:
                    pass
                return client

    def add_from_scenes(self, scenes: list[dict], knowledge: list[dict] | None = None) -> None:
        docs: list[dict] = []

        for item in knowledge or []:
            knowledge_type = str(item.get('knowledge_type', 'other')).strip().lower() or 'other'
            mapped_type = self._map_knowledge_type_to_doc_type(knowledge_type)
            title = str(item.get('title', '')).strip() or str(item.get('knowledge_id', 'knowledge')).strip()
            content = str(item.get('content', '')).strip() or title
            if content.startswith(title):
                description = content
            else:
                description = f"{title}\n{content}"

            docs.append(
                {
                    'type': mapped_type,
                    'name': title[:80],
                    'description': description,
                    'metadata': {
                        'knowledge_id': item.get('knowledge_id', ''),
                        'knowledge_type': knowledge_type,
                    },
                }
            )

        if not docs:
            return

        ids = [self._make_doc_id(d, i) for i, d in enumerate(docs)]
        embeddings = self.embedding_fn.embed_documents([d['description'] for d in docs])
        self.collection.add(
            ids=ids,
            documents=[d['description'] for d in docs],
            metadatas=[d['metadata'] | {'type': d['type'], 'name': d['name']} for d in docs],
            embeddings=embeddings,
        )

    def _make_doc_id(self, doc: dict, idx: int) -> str:
        seed = json.dumps(
            {
                'type': doc.get('type', ''),
                'name': doc.get('name', ''),
                'description': doc.get('description', ''),
                'metadata': doc.get('metadata', {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]
        return f"{doc.get('type', 'doc')}::{digest}::{idx}"

    def _map_knowledge_type_to_doc_type(self, knowledge_type: str) -> str:
        if knowledge_type == 'npc':
            return 'npc'
        if knowledge_type == 'clue':
            return 'clue'
        if knowledge_type == 'setting':
            return 'setting'
        return 'other'

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.embedding_fn.embed_query(query)
        result = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        out = []
        for doc, meta, dist in zip(
            result.get('documents', [[]])[0],
            result.get('metadatas', [[]])[0],
            result.get('distances', [[]])[0],
            strict=False,
        ):
            out.append({'content': doc, 'metadata': meta, 'distance': dist})
        return out
