import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest


APP_SCRIPT = f"""
import os
import sys

sys.path.insert(0, {str(ROOT)!r})
os.environ['PUBLIC_DEMO_MODE'] = 'true'
os.environ['PUBLIC_DEMO_DAILY_TURN_BUDGET'] = '0'
os.environ['PUBLIC_DEMO_STRICT_CONFIG'] = 'false'

import app.ui as ui


class FakeVectorStore:
    def __init__(self, path='.chroma'):
        self.path = path
        self.docs = []

    def reset(self):
        self.docs = []

    def add_from_scenes(self, scenes, knowledge=None):
        self.docs = list(knowledge or [])

    def search(self, query, k=5):
        return [
            {{
                'content': 'The lighthouse lens hums when the harbor signal returns.',
                'metadata': {{'type': 'clue', 'name': 'Lens Signal'}},
                'distance': 0.1,
            }}
        ][:k]


class FakeAgent:
    def __init__(self, db, vector):
        self.db = db
        self.vector = vector
        self.debug_mode = False

    def set_debug_mode(self, value):
        self.debug_mode = value

    def generate_initial_response(self):
        return {{
            'response': 'The keeper opens the case file as the harbor light turns once across the glass.',
            'retrieved_docs': [],
            'dice_result': None,
            'skill_check_result': None,
            'debug_prompts': [],
        }}

    def run_turn(self, user_msg):
        return {{
            'response': f'The keeper considers your action: {{user_msg}}',
            'retrieved_docs': [],
            'dice_result': None,
            'skill_check_result': None,
            'debug_prompts': [],
        }}


def fake_demo_bundle():
    return {{
        'script_summary': 'A compact lighthouse mystery.',
        'source_metadata': {{'source_file_name': 'FakeDemo.md'}},
        'scenes': [
            {{
                'scene_id': 'scene_1',
                'scene_name': 'The Harbor Light',
                'scene_goal': 'Investigate the lighthouse signal.',
                'scene_description': 'A lens signal cuts across the harbor.',
                'status': 'pending',
                'plots': [
                    {{
                        'plot_id': 'scene_1_plot_1',
                        'plot_name': 'Enter the lantern room',
                        'plot_goal': 'Inspect the lens and keeper notes.',
                        'raw_text': 'The investigator enters the lantern room.',
                        'status': 'pending',
                    }}
                ],
            }}
        ],
        'knowledge': [
            {{
                'knowledge_id': 'knowledge_1',
                'knowledge_type': 'clue',
                'title': 'Lens Signal',
                'content': 'The lighthouse lens hums when the harbor signal returns.',
            }}
        ],
    }}


ui.ChromaStore = FakeVectorStore
ui.NarrativeAgent = FakeAgent
ui._load_demo_script_bundle = fake_demo_bundle
ui.run_app()
"""


def click_button(at: AppTest, *labels: str) -> AppTest:
    expected = set(labels)
    for button in at.button:
        if getattr(button, 'label', '') in expected:
            return button.click().run(timeout=20)
    available = [getattr(button, 'label', '') for button in at.button]
    raise AssertionError(f'Buttons {sorted(expected)!r} not found. Available: {available}')


def main() -> int:
    try:
        at = AppTest.from_string(APP_SCRIPT)
        at.run(timeout=20)
        assert not at.exception, at.exception

        at = click_button(at, 'Begin Demo', 'Begin Investigation')
        assert not at.exception, at.exception

        at = click_button(at, 'Create Character', 'Create Investigator')
        assert not at.exception, at.exception

        at = click_button(at, 'Enter Story', 'Enter Graymouth Harbor')
        assert not at.exception, at.exception

        page_text = str(at)
        assert 'Keeper' in page_text or 'keeper' in page_text.lower(), 'session should render keeper dialogue'
        assert 'What do you do next?' in page_text, 'chat input should be available in session'

        print('[test_public_demo_smoke] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_public_demo_smoke] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
