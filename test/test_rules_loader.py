import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rules_loader import load_game_rules_knowledge
from app.vector_store import ChromaStore


class FakeRulesEmbedding:
    KEYWORDS = ['sanity', 'magic', 'clue', 'combat', 'dice', 'keeper', 'scenario', 'skill']

    def _embed_one(self, text: str) -> list[float]:
        lowered = text.lower()
        counts = [float(lowered.count(keyword)) for keyword in self.KEYWORDS]
        counts.append(float(len(lowered.split()) % 17) / 17.0)
        return counts

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in input]

    def embed_query(self, input: str) -> list[float]:
        return self._embed_one(input)


def main() -> int:
    try:
        print('[test_rules_loader] input: load lightweight d100 rules via rules loader')
        chunks = load_game_rules_knowledge()
        print(f'[test_rules_loader] output: loaded chunks -> {len(chunks)}')

        assert chunks, 'rules loader should produce at least one chunk'
        assert any(item.get('knowledge_type') == 'rule' for item in chunks), 'chunks should be tagged as rule knowledge'
        assert any(item.get('title') for item in chunks), 'chunks should preserve section titles'

        store_path = ROOT / 'test' / '.chroma_rules_test'
        print(f'[test_rules_loader] input: initialize ChromaStore(path={store_path})')
        store = ChromaStore(path=str(store_path))
        store.embedding_fn = FakeRulesEmbedding()
        store.reset()

        print('[test_rules_loader] input: insert rule chunks into vector store')
        store.add_from_scenes([], knowledge=chunks)

        query = 'sanity points and magic points'
        print('[test_rules_loader] input: query ->', query)
        result = store.search(query, k=5)
        print('[test_rules_loader] output: search result ->', result)

        assert result, 'search should return indexed rule chunks'
        assert any(r.get('metadata', {}).get('type') == 'rule' for r in result), 'retrieval should include rule documents'
        assert any(
            'sanity' in r.get('content', '').lower() or 'magic points' in r.get('content', '').lower()
            for r in result
        ), 'retrieval should surface relevant d100 investigation rules content'

        print('[test_rules_loader] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_rules_loader] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
