from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import shutil
import time
import uuid
from html import escape
from pathlib import Path

import streamlit as st
import requests

from app.agent_graph import KP_OPENING_MARKER, NarrativeAgent
from app.database import Database
from app.llm_client import call_llm
from app.parser import detect_source_type, parse_script_bundle, read_uploaded_document
from app.vector_store import ChromaStore, ModelEmbedding

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


@st.cache_data(show_spinner=False)
def _image_data_uri(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        return ''
    encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
    return f"data:image/png;base64,{encoded}"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT_PATH = PROJECT_ROOT / 'database' / 'DemoScript.md'
DEMO_PARSED_PATH = PROJECT_ROOT / 'database' / 'DemoScript.parsed.json'
DEMO_MANIFEST_PATH = PROJECT_ROOT / 'database' / 'demo_manifest.json'
PUBLIC_DEMO_MODE = os.getenv('PUBLIC_DEMO_MODE', 'true').strip().lower() not in {'0', 'false', 'no', 'off'}
PUBLIC_DEMO_MAX_TURNS = _env_int('PUBLIC_DEMO_MAX_TURNS', 18)
PUBLIC_DEMO_MAX_INPUT_CHARS = _env_int('PUBLIC_DEMO_MAX_INPUT_CHARS', 1200)
PUBLIC_DEMO_MAX_UPLOAD_BYTES = _env_int('PUBLIC_DEMO_MAX_UPLOAD_BYTES', 200_000)
PUBLIC_DEMO_MIN_TURN_SECONDS = _env_float('PUBLIC_DEMO_MIN_TURN_SECONDS', 6.0)
PUBLIC_DEMO_DAILY_TURN_BUDGET = _env_int('PUBLIC_DEMO_DAILY_TURN_BUDGET', 250)
PUBLIC_DEMO_SESSION_TTL_HOURS = _env_float('PUBLIC_DEMO_SESSION_TTL_HOURS', 24.0)
PUBLIC_DEMO_STRICT_CONFIG = _env_bool('PUBLIC_DEMO_STRICT_CONFIG', False)
PUBLIC_DEMO_ADMIN_DIAGNOSTICS = _env_bool('PUBLIC_DEMO_ADMIN_DIAGNOSTICS', False)
RUNTIME_ROOT = PROJECT_ROOT / '.runtime' / 'sessions'
PUBLIC_DEMO_USAGE_PATH = PROJECT_ROOT / '.runtime' / 'public_demo_usage.json'
UPSTASH_REDIS_REST_URL = (os.getenv('UPSTASH_REDIS_REST_URL') or '').strip().rstrip('/')
UPSTASH_REDIS_REST_TOKEN = (os.getenv('UPSTASH_REDIS_REST_TOKEN') or '').strip()
DEFAULT_DEMO_MANIFEST: dict[str, object] = {
    'title': 'Official Demo',
    'tagline': 'A compact playable showcase for the narrative engine.',
    'background_image': 'assets/harbor-light-bg.png',
    'case_kicker': 'Public demo',
    'ready_kicker': 'Demo Prepared',
    'ready_copy': 'Create your character, then step into the story.',
    'case_note': 'A private session is prepared for this browser. Name your character and answer in free-form actions when the Keeper asks what you do next.',
    'route_label': 'Story route',
    'entry_button': 'Begin Demo',
    'continue_button': 'Create Character',
    'save_character_button': 'Enter Story',
    'chat_placeholder': 'What do you do next?',
    'chips': ['Case 01', '{turn_limit} turn cap', 'Solo investigation'],
    'stages': [
        {'key': 'upload', 'label': 'Open case'},
        {'key': 'parse', 'label': 'Set the scene'},
        {'key': 'character', 'label': 'Investigator'},
    ],
    'beats': [],
    'character_presets': [],
    'ending_title': 'Demo Session Complete',
    'ending_copy': 'This public preview has reached its turn limit. Start a new run to investigate from the beginning.',
    'new_demo_button': 'Start New Demo',
    'fallback_opening': 'The case file is ready, but the Keeper needs a moment to connect. Try sending your first action in a few seconds.',
    'fallback_turn': 'The Keeper lost the thread for a moment. Please try that action again.',
    'story_source_label': 'Choose a story',
    'official_demo_label': 'Demo Story',
    'upload_story_label': 'Upload Markdown',
    'upload_story_note': 'Bring your own Markdown scenario. Headings become scenes and beats.',
    'upload_story_button': 'Prepare Uploaded Story',
    'advanced_sheet_label': 'Customize character sheet',
    'upload_title': 'Choose Your Own Story',
    'upload_tagline': 'Upload a Markdown scenario and turn it into a private playable table.',
    'upload_case_kicker': 'Custom Story / Markdown',
    'upload_case_note': 'Use Markdown headings for scenes and beats. You can still choose a ready-made investigator or customize the sheet before play.',
    'upload_chips': ['Markdown upload', '{turn_limit} turn cap', 'Custom table'],
    'upload_beats': [
        {'title': 'Bring a Scenario', 'copy': 'Upload a Markdown file with scenes, clues, NPCs, or encounter notes.'},
        {'title': 'Parse Into Play', 'copy': 'The system prepares scenes and story beats from the document.'},
        {'title': 'Play Your Way', 'copy': 'Use a preset investigator or customize the character sheet before entering.'},
    ],
}


def _demo_manifest() -> dict[str, object]:
    if not DEMO_MANIFEST_PATH.exists():
        return dict(DEFAULT_DEMO_MANIFEST)
    try:
        loaded = json.loads(DEMO_MANIFEST_PATH.read_text(encoding='utf-8'))
    except Exception:
        return dict(DEFAULT_DEMO_MANIFEST)
    if not isinstance(loaded, dict):
        return dict(DEFAULT_DEMO_MANIFEST)
    merged = dict(DEFAULT_DEMO_MANIFEST)
    merged.update(loaded)
    return merged


def _demo_text(key: str, default: str = '') -> str:
    value = _demo_manifest().get(key, default)
    return str(value if value is not None else default)


def _demo_items(key: str) -> list[object]:
    value = _demo_manifest().get(key, [])
    return value if isinstance(value, list) else []


def _demo_background_path() -> Path:
    configured = _demo_text('background_image', 'assets/harbor-light-bg.png').strip()
    if not configured:
        configured = 'assets/harbor-light-bg.png'
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _has_public_demo_llm_credentials() -> bool:
    if (os.getenv('OPENAI_API_KEY') or '').strip():
        return True
    if (os.getenv('NVIDIA_API_KEY') or '').strip():
        return True
    for key_file in ('openai_api_key.txt', 'api_key.txt'):
        path = PROJECT_ROOT / key_file
        if path.exists():
            try:
                text = path.read_text(encoding='utf-8').strip()
            except Exception:
                continue
            if text and text != 'PASTE_YOUR_API_KEY_HERE':
                return True
    return False


def _public_demo_config_issues() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not _has_public_demo_llm_credentials():
        errors.append('No LLM API key is configured.')
    if PUBLIC_DEMO_DAILY_TURN_BUDGET > 0 and not (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN):
        warnings.append('Upstash Redis is not configured; daily budget falls back to local instance storage.')
    if PUBLIC_DEMO_MAX_TURNS <= 0:
        errors.append('PUBLIC_DEMO_MAX_TURNS must be positive.')
    if PUBLIC_DEMO_MAX_INPUT_CHARS < 100:
        warnings.append('PUBLIC_DEMO_MAX_INPUT_CHARS is very low and may block normal play.')
    if PUBLIC_DEMO_MAX_UPLOAD_BYTES < 10_000:
        warnings.append('PUBLIC_DEMO_MAX_UPLOAD_BYTES is very low and may block Markdown uploads.')
    return errors, warnings


def _enforce_public_demo_config() -> None:
    if not PUBLIC_DEMO_MODE or st.session_state.get('public_demo_config_checked'):
        return
    st.session_state.public_demo_config_checked = True
    errors, warnings = _public_demo_config_issues()
    for item in warnings:
        LOGGER.warning('Public demo config warning: %s', item)
    if not errors:
        return
    for item in errors:
        LOGGER.error('Public demo config error: %s', item)
    if PUBLIC_DEMO_STRICT_CONFIG:
        st.error('This public demo is not fully configured yet. Please check back later.')
        st.stop()


def _runtime_root_is_writable() -> bool:
    try:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        probe = RUNTIME_ROOT / '.write_check'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink(missing_ok=True)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Runtime root write check failed: %s', exc)
        return False


def _render_public_demo_diagnostics(db: Database) -> None:
    if not PUBLIC_DEMO_ADMIN_DIAGNOSTICS:
        return
    errors, warnings = _public_demo_config_issues()
    usage = _read_public_demo_usage()
    provider = (os.getenv('LLM_PROVIDER') or 'backend file/default').strip()
    model = (
        os.getenv('OPENAI_MODEL')
        or os.getenv('NVIDIA_MODEL')
        or 'backend file/default'
    )
    budget_used = int(usage.get('turns', 0)) if isinstance(usage, dict) else 0
    rows = [
        ('Public mode', 'on' if PUBLIC_DEMO_MODE else 'off'),
        ('Strict config', 'on' if PUBLIC_DEMO_STRICT_CONFIG else 'off'),
        ('LLM credentials', 'configured' if _has_public_demo_llm_credentials() else 'missing'),
        ('Provider', provider),
        ('Model', model),
        ('Upstash budget', 'configured' if (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN) else 'local fallback'),
        ('Local budget used', f'{budget_used}/{PUBLIC_DEMO_DAILY_TURN_BUDGET}'),
        ('Runtime writable', 'yes' if _runtime_root_is_writable() else 'no'),
        ('Embedding loaded', 'yes' if ModelEmbedding._model is not None else 'no'),
        ('Story stage', str(db.get_system_state().get('stage', 'unknown'))),
    ]
    with st.sidebar.expander('Admin Diagnostics', expanded=False):
        st.caption('No secret values are shown here.')
        for label, value in rows:
            st.write(f'**{label}:** {value}')
        if errors:
            st.error(' / '.join(errors))
        if warnings:
            st.warning(' / '.join(warnings))


def _public_demo_today() -> str:
    return time.strftime('%Y-%m-%d', time.gmtime())


def _read_public_demo_usage() -> dict[str, object]:
    today = _public_demo_today()
    if not PUBLIC_DEMO_USAGE_PATH.exists():
        return {'date': today, 'turns': 0}
    try:
        usage = json.loads(PUBLIC_DEMO_USAGE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {'date': today, 'turns': 0}
    if not isinstance(usage, dict) or usage.get('date') != today:
        return {'date': today, 'turns': 0}
    try:
        turns = int(usage.get('turns', 0))
    except (TypeError, ValueError):
        turns = 0
    return {'date': today, 'turns': max(0, turns)}


def _write_public_demo_usage(usage: dict[str, object]) -> None:
    PUBLIC_DEMO_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PUBLIC_DEMO_USAGE_PATH.with_suffix('.tmp')
    tmp_path.write_text(json.dumps(usage, ensure_ascii=False), encoding='utf-8')
    tmp_path.replace(PUBLIC_DEMO_USAGE_PATH)


def _reserve_public_demo_turn_upstash() -> tuple[bool, int | None]:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return False, None

    key = f"public_demo:turns:{_public_demo_today()}"
    seconds_until_tomorrow = max(60, 86400 - int(time.time() % 86400))
    try:
        response = requests.post(
            f'{UPSTASH_REDIS_REST_URL}/multi-exec',
            headers={'Authorization': f'Bearer {UPSTASH_REDIS_REST_TOKEN}'},
            json=[
                ['INCR', key],
                ['EXPIRE', key, seconds_until_tomorrow],
            ],
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Upstash public demo budget unavailable; falling back to local budget: %s', exc)
        return False, None

    if not isinstance(payload, list) or not payload:
        LOGGER.warning('Unexpected Upstash public demo budget response: %s', payload)
        return False, None
    first = payload[0] if isinstance(payload[0], dict) else {}
    if 'error' in first:
        LOGGER.warning('Upstash public demo budget error: %s', first.get('error'))
        return False, None
    try:
        count = int(first.get('result', 0))
    except (TypeError, ValueError):
        LOGGER.warning('Unexpected Upstash INCR result: %s', first.get('result'))
        return False, None
    return True, count


def _reserve_public_demo_turn() -> tuple[bool, str]:
    if not PUBLIC_DEMO_MODE or PUBLIC_DEMO_DAILY_TURN_BUDGET <= 0:
        return True, ''
    used_shared_budget, shared_turns = _reserve_public_demo_turn_upstash()
    if used_shared_budget and shared_turns is not None:
        if shared_turns > PUBLIC_DEMO_DAILY_TURN_BUDGET:
            return False, 'The public demo table is full for today. Please try again tomorrow.'
        return True, ''

    usage = _read_public_demo_usage()
    turns = int(usage.get('turns', 0))
    if turns >= PUBLIC_DEMO_DAILY_TURN_BUDGET:
        return False, 'The public demo table is full for today. Please try again tomorrow.'
    usage['turns'] = turns + 1
    try:
        _write_public_demo_usage(usage)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Failed to write public demo usage budget: %s', exc)
    return True, ''


def _cleanup_public_demo_runtime_once() -> None:
    if not PUBLIC_DEMO_MODE or st.session_state.get('public_demo_runtime_cleanup_done'):
        return
    st.session_state.public_demo_runtime_cleanup_done = True
    if PUBLIC_DEMO_SESSION_TTL_HOURS <= 0 or not RUNTIME_ROOT.exists():
        return
    cutoff = time.time() - (PUBLIC_DEMO_SESSION_TTL_HOURS * 3600)
    for session_dir in RUNTIME_ROOT.iterdir():
        try:
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning('Failed to clean public demo runtime %s: %s', session_dir, exc)


COC_CORE_KEYS = ['STR', 'CON', 'SIZ', 'DEX', 'APP', 'INT', 'POW', 'EDU']
COC_3D6_KEYS = ['STR', 'CON', 'DEX', 'APP', 'POW']
COC_2D6_KEYS = ['SIZ', 'INT', 'EDU']
COC_ARCHETYPES = [
    {
        'name': 'Detective',
        'weights': {'INT': 3.0, 'DEX': 2.0, 'POW': 2.0, 'APP': 1.0, 'EDU': 2.0},
        'occupation_skills': [('Spot Hidden', 22), ('Psychology', 16), ('Law', 14), ('Listen', 14), ('Stealth', 12), ('Firearms', 12), ('Persuade', 10)],
        'interest_skills': [('Library Use', 30), ('Occult', 25), ('Locksmith', 25), ('First Aid', 20)],
    },
    {
        'name': 'Scholar',
        'weights': {'EDU': 3.0, 'INT': 3.0, 'POW': 1.5, 'APP': 1.0, 'DEX': 0.5},
        'occupation_skills': [('Library Use', 24), ('History', 18), ('Archaeology', 14), ('Language (Other)', 14), ('Anthropology', 12), ('Occult', 10), ('Persuade', 8)],
        'interest_skills': [('Spot Hidden', 30), ('Psychology', 25), ('Credit Rating', 25), ('Charm', 20)],
    },
    {
        'name': 'Journalist',
        'weights': {'INT': 2.5, 'APP': 2.0, 'POW': 2.0, 'DEX': 1.5, 'EDU': 1.5},
        'occupation_skills': [('Persuade', 22), ('Fast Talk', 18), ('Library Use', 16), ('Psychology', 14), ('Photography', 12), ('Spot Hidden', 10), ('Stealth', 8)],
        'interest_skills': [('Credit Rating', 30), ('Occult', 25), ('Listen', 25), ('Drive Auto', 20)],
    },
    {
        'name': 'Soldier',
        'weights': {'STR': 2.5, 'CON': 2.5, 'DEX': 2.0, 'POW': 1.5, 'SIZ': 1.5},
        'occupation_skills': [('Firearms', 24), ('Fighting (Brawl)', 20), ('Dodge', 14), ('Survival', 12), ('First Aid', 12), ('Navigate', 10), ('Intimidate', 8)],
        'interest_skills': [('Spot Hidden', 30), ('Listen', 25), ('Mechanical Repair', 25), ('Psychology', 20)],
    },
    {
        'name': 'Doctor',
        'weights': {'EDU': 3.0, 'INT': 2.5, 'DEX': 1.5, 'POW': 1.5, 'APP': 1.0},
        'occupation_skills': [('Medicine', 26), ('First Aid', 20), ('Science (Biology)', 16), ('Psychology', 12), ('Pharmacy', 10), ('Persuade', 8), ('Listen', 8)],
        'interest_skills': [('Library Use', 30), ('Spot Hidden', 25), ('Occult', 25), ('Drive Auto', 20)],
    },
    {
        'name': 'Explorer',
        'weights': {'CON': 2.0, 'SIZ': 2.0, 'DEX': 2.0, 'POW': 1.5, 'INT': 1.5},
        'occupation_skills': [('Survival', 24), ('Navigate', 18), ('Climb', 14), ('Track', 14), ('Spot Hidden', 12), ('First Aid', 10), ('Natural World', 8)],
        'interest_skills': [('Firearms', 30), ('Mechanic Repair', 25), ('Listen', 25), ('Anthropology', 20)],
    },
    {
        'name': 'Antiquarian',
        'weights': {'EDU': 2.5, 'APP': 2.0, 'INT': 2.0, 'POW': 1.5, 'DEX': 1.0},
        'occupation_skills': [('Appraise', 22), ('History', 18), ('Charm', 14), ('Persuade', 14), ('Library Use', 12), ('Credit Rating', 10), ('Occult', 10)],
        'interest_skills': [('Spot Hidden', 30), ('Psychology', 25), ('Stealth', 25), ('Language (Other)', 20)],
    },
    {
        'name': 'Professor',
        'weights': {'EDU': 3.0, 'INT': 2.5, 'APP': 1.0, 'POW': 1.5, 'CON': 1.0},
        'occupation_skills': [('Library Use', 24), ('Language (Own)', 16), ('Language (Other)', 16), ('Psychology', 12), ('History', 12), ('Persuade', 10), ('Occult', 10)],
        'interest_skills': [('Spot Hidden', 30), ('Credit Rating', 25), ('Charm', 25), ('Listen', 20)],
    },
    {
        'name': 'Private Eye',
        'weights': {'INT': 2.5, 'DEX': 2.0, 'STR': 1.5, 'POW': 2.0, 'APP': 1.0},
        'occupation_skills': [('Spot Hidden', 22), ('Stealth', 16), ('Locksmith', 14), ('Psychology', 14), ('Firearms', 12), ('Dodge', 12), ('Law', 10)],
        'interest_skills': [('Listen', 30), ('Drive Auto', 25), ('Occult', 25), ('Persuade', 20)],
    },
    {
        'name': 'Artist',
        'weights': {'APP': 2.5, 'POW': 2.0, 'DEX': 2.0, 'INT': 1.5, 'EDU': 1.0},
        'occupation_skills': [('Art/Craft', 24), ('Psychology', 16), ('Charm', 16), ('Persuade', 14), ('History', 10), ('Spot Hidden', 10), ('Listen', 10)],
        'interest_skills': [('Occult', 30), ('Library Use', 25), ('Stealth', 25), ('Credit Rating', 20)],
    },
]


def _inject_demo_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --um-blue: #00274c;
            --um-blue-soft: #5a6d82;
            --um-maize: #d7b14a;
            --um-teal: #196b6a;
            --um-plum: #5e3f58;
            --um-ink: #1f2d3a;
            --um-ink-strong: #14212e;
            --um-panel: rgba(255, 253, 248, 0.94);
            --um-panel-soft: rgba(252, 249, 242, 0.92);
            --um-code-bg: #161b26;
            --um-code-ink: #f3f7fb;
            --um-paper: #f5f1e8;
            --um-line: rgba(0, 39, 76, 0.12);
        }

        html, body, [class*="css"] {
            font-family: Georgia, "Times New Roman", serif;
        }

        .stApp {
            background: linear-gradient(180deg, #f7f4ec 0%, #f2efe7 100%);
            color: var(--um-ink);
        }

        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div,
        .stMarkdown,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: var(--um-ink-strong);
        }

        [data-testid="stHeader"] {
            background: rgba(245, 241, 232, 0.9);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(245, 241, 232, 0.98), rgba(239, 234, 222, 0.98));
            color: var(--um-ink-strong);
        }

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] button {
            color: var(--um-ink-strong) !important;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] .stCheckbox label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stToggle label,
        [data-testid="stSidebar"] [role="switch"] + div,
        [data-testid="stSidebar"] [role="switch"] ~ * {
            color: var(--um-ink-strong) !important;
        }

        [data-testid="stSidebar"] [role="switch"] {
            background: #efe4c6 !important;
            border: 1px solid rgba(0, 39, 76, 0.28) !important;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.45);
        }

        [data-testid="stSidebar"] [role="switch"][aria-checked="true"] {
            background: #d7b14a !important;
            border-color: rgba(0, 39, 76, 0.45) !important;
        }

        [data-testid="stSidebar"] [role="switch"] > div,
        [data-testid="stSidebar"] [role="switch"] [data-testid="stThumbValue"] {
            background: #16304d !important;
            color: #16304d !important;
        }

        .block-container {
            max-width: 880px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .gm-section {
            margin-top: 0.4rem;
            margin-bottom: 0.5rem;
        }

        .gm-section-eyebrow {
            color: rgba(31, 45, 58, 0.82);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .gm-section-title {
            font-size: 1.18rem;
            color: var(--um-blue);
            margin: 0 0 0.12rem 0;
            font-weight: 700;
        }

        .gm-section-copy {
            color: var(--um-ink-strong);
            font-size: 0.93rem;
            margin-bottom: 0.6rem;
        }

        .gm-case-hero {
            border: 1px solid rgba(0, 39, 76, 0.14);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(255, 253, 248, 0.96), rgba(241, 247, 245, 0.94)),
                linear-gradient(90deg, rgba(25, 107, 106, 0.1), rgba(94, 63, 88, 0.08));
            padding: 1.2rem 1.25rem 1.05rem 1.25rem;
            margin: 0.25rem 0 0.8rem 0;
        }

        .gm-case-kicker {
            color: var(--um-teal);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .gm-case-title {
            color: var(--um-blue);
            font-size: 2.15rem;
            line-height: 1.05;
            font-weight: 800;
            margin: 0 0 0.45rem 0;
        }

        .gm-case-copy {
            max-width: 42rem;
            color: var(--um-ink-strong);
            font-size: 1.02rem;
            line-height: 1.7;
            margin-bottom: 0.85rem;
        }

        .gm-case-chips,
        .gm-status-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
        }

        .gm-chip {
            display: inline-flex;
            align-items: center;
            min-height: 1.7rem;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(0, 39, 76, 0.13);
            background: rgba(255, 251, 244, 0.78);
            color: rgba(20, 33, 46, 0.92);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .gm-chip-accent {
            border-color: rgba(25, 107, 106, 0.24);
            background: rgba(25, 107, 106, 0.08);
            color: #174f50;
        }

        .gm-feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.8rem 0 1rem 0;
        }

        .gm-feature {
            border-top: 3px solid rgba(25, 107, 106, 0.55);
            background: rgba(255, 253, 248, 0.86);
            padding: 0.72rem 0.75rem;
            min-height: 6.4rem;
        }

        .gm-feature:nth-child(2) {
            border-top-color: rgba(94, 63, 88, 0.5);
        }

        .gm-feature:nth-child(3) {
            border-top-color: rgba(215, 177, 74, 0.78);
        }

        .gm-feature-title {
            color: var(--um-blue);
            font-weight: 800;
            font-size: 0.94rem;
            margin-bottom: 0.25rem;
        }

        .gm-feature-copy {
            color: rgba(20, 33, 46, 0.9);
            line-height: 1.55;
            font-size: 0.86rem;
        }

        .gm-investigator-card,
        .gm-report-card,
        .gm-hint-card,
        .gm-side-panel {
            border: 1px solid rgba(0, 39, 76, 0.13);
            border-radius: 8px;
            background: rgba(255, 253, 248, 0.88);
            padding: 0.85rem 0.95rem;
            margin: 0.55rem 0 0.85rem 0;
        }

        .gm-investigator-title,
        .gm-report-title,
        .gm-hint-title,
        .gm-side-title {
            color: var(--um-blue);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .gm-investigator-copy,
        .gm-report-copy,
        .gm-hint-copy,
        .gm-side-copy {
            color: rgba(20, 33, 46, 0.9);
            line-height: 1.55;
            font-size: 0.9rem;
        }

        .gm-hint-card {
            max-width: 54rem;
            border-left: 3px solid rgba(215, 177, 74, 0.82);
            margin-top: 0;
        }

        .gm-hint-list {
            margin: 0.4rem 0 0 1.1rem;
            padding: 0;
        }

        .gm-hint-list li {
            color: rgba(20, 33, 46, 0.9);
            line-height: 1.55;
            margin: 0.24rem 0;
        }

        .gm-stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.45rem;
            margin-top: 0.75rem;
        }

        .gm-stat {
            border: 1px solid rgba(0, 39, 76, 0.11);
            border-radius: 8px;
            background: rgba(241, 247, 245, 0.72);
            padding: 0.42rem 0.5rem;
            min-height: 3.1rem;
        }

        .gm-stat-label {
            color: rgba(20, 33, 46, 0.62);
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .gm-stat-value {
            color: var(--um-blue);
            font-size: 1.05rem;
            font-weight: 800;
        }

        .gm-case-note {
            border-left: 3px solid rgba(25, 107, 106, 0.55);
            background: rgba(241, 247, 245, 0.72);
            padding: 0.65rem 0.75rem;
            margin: 0.7rem 0 0.35rem 0;
            color: rgba(20, 33, 46, 0.94);
            line-height: 1.6;
            font-size: 0.9rem;
        }

        .gm-session-banner {
            position: sticky;
            top: 0.35rem;
            z-index: 10;
            border: 1px solid rgba(0, 39, 76, 0.12);
            border-radius: 8px;
            background: rgba(247, 244, 236, 0.94);
            backdrop-filter: blur(6px);
            padding: 0.55rem 0.7rem;
            margin: 0.15rem 0 0.9rem 0;
        }

        .gm-session-title {
            color: var(--um-blue);
            font-weight: 800;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }

        .gm-stage-list {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0;
            margin: 0.5rem 0 1.05rem 0;
        }

        .gm-stage {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.35rem;
            color: rgba(20, 33, 46, 0.66);
            font-size: 0.78rem;
            font-weight: 700;
            text-align: center;
            pointer-events: none;
        }

        .gm-stage::before {
            content: "";
            position: absolute;
            top: 0.46rem;
            left: 0;
            right: 0;
            height: 2px;
            background: rgba(0, 39, 76, 0.14);
            z-index: 0;
        }

        .gm-stage:first-child::before {
            left: 50%;
        }

        .gm-stage:last-child::before {
            right: 50%;
        }

        .gm-stage-dot {
            position: relative;
            z-index: 1;
            width: 0.95rem;
            height: 0.95rem;
            border-radius: 999px;
            border: 2px solid rgba(0, 39, 76, 0.25);
            background: rgba(255, 253, 248, 0.96);
        }

        .gm-stage-active {
            color: #174f50;
        }

        .gm-stage-active .gm-stage-dot {
            border-color: rgba(25, 107, 106, 0.9);
            background: #196b6a;
            box-shadow: 0 0 0 4px rgba(25, 107, 106, 0.14);
        }

        .gm-stage-complete {
            color: rgba(20, 33, 46, 0.84);
        }

        .gm-stage-complete .gm-stage-dot {
            border-color: rgba(215, 177, 74, 0.85);
            background: #d7b14a;
        }

        @media (max-width: 760px) {
            .gm-case-title {
                font-size: 1.65rem;
            }

            .gm-feature-grid,
            .gm-stage-list {
                grid-template-columns: 1fr;
            }

            .gm-stat-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        div[data-testid="stMetric"] {
            background: var(--um-panel);
            border: 1px solid var(--um-line);
            border-radius: 10px;
            padding: 0.65rem 0.75rem;
            box-shadow: none;
        }

        div[data-testid="stMetric"] label {
            color: rgba(31, 45, 58, 0.82) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--um-blue) !important;
            font-size: 1.1rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(0, 39, 76, 0.18);
            background: rgba(215, 177, 74, 0.32);
            color: var(--um-ink-strong);
            font-weight: 700;
            padding: 0.45rem 0.85rem;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(0, 39, 76, 0.28);
            background: rgba(215, 177, 74, 0.42);
        }

        .stButton > button[kind="tertiary"] {
            border-radius: 999px;
            border: 1px solid rgba(0, 39, 76, 0.14);
            background: rgba(255, 251, 244, 0.78);
            color: rgba(20, 33, 46, 0.94);
            font-weight: 600;
            font-size: 0.8rem;
            letter-spacing: 0.01em;
            padding: 0.18rem 0.78rem;
            min-height: 2rem;
            box-shadow: 0 1px 0 rgba(0, 39, 76, 0.03);
            backdrop-filter: blur(6px);
        }

        .stButton > button[kind="tertiary"]:hover {
            border-color: rgba(0, 39, 76, 0.22);
            background: rgba(255, 252, 247, 0.96);
            color: var(--um-blue);
        }

        .stButton > button[kind="tertiary"]:disabled {
            opacity: 0.5;
            background: rgba(255, 251, 244, 0.62);
            color: rgba(31, 45, 58, 0.5);
            border-color: rgba(0, 39, 76, 0.1);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stFileUploader section,
        [data-testid="stChatInput"] {
            border-radius: 8px !important;
            border-color: rgba(0, 39, 76, 0.15) !important;
            background: var(--um-panel) !important;
            color: var(--um-ink-strong) !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder,
        .stNumberInput input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {
            color: #5b6673 !important;
        }

        .stTextInput label,
        .stTextArea label,
        .stNumberInput label,
        .stSelectbox label,
        .stFileUploader label,
        .stRadio label,
        .stCheckbox label,
        .stCaption,
        [data-testid="stWidgetLabel"],
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: var(--um-ink-strong) !important;
        }

        .stSelectbox [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] *,
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            color: var(--um-ink-strong) !important;
        }

        [data-baseweb="menu"] *,
        [role="listbox"] *,
        [role="option"] {
            color: var(--um-ink-strong) !important;
            background: #fffdf8 !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus {
            border-color: rgba(215, 177, 74, 0.85) !important;
            box-shadow: 0 0 0 1px rgba(215, 177, 74, 0.45) !important;
        }

        [data-testid="stExpander"] {
            border-radius: 8px;
            border: 1px solid var(--um-line);
            background: var(--um-panel-soft);
            overflow: hidden;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] details > div,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
        [data-testid="stExpander"] summary *,
        [data-testid="stAlert"] *,
        [data-testid="stChatMessage"] * {
            color: var(--um-ink-strong) !important;
        }

        [data-testid="stExpander"] summary {
            background: rgba(247, 241, 229, 0.92) !important;
        }

        [data-testid="stExpander"] details > div {
            background: rgba(255, 251, 244, 0.96) !important;
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--um-line);
            background: rgba(255, 251, 244, 0.96);
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 252, 247, 0.62);
            border: none;
            border-radius: 10px;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.6rem;
            box-shadow: none;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.8;
            font-size: 1rem;
        }

        .gm-dialogue {
            max-width: 54rem;
            margin: 0 0 0.75rem 0;
            padding: 0.78rem 0.9rem;
            border-radius: 8px;
            border: 1px solid rgba(0, 39, 76, 0.12);
            background: rgba(255, 253, 248, 0.88);
        }

        .gm-dialogue-user {
            max-width: min(42rem, 88%);
            margin-left: auto;
            border-color: rgba(25, 107, 106, 0.28);
            background: rgba(241, 247, 245, 0.88);
        }

        .gm-dialogue-keeper {
            border-left: 3px solid rgba(215, 177, 74, 0.82);
        }

        .gm-dialogue-label {
            margin-bottom: 0.34rem;
            color: rgba(31, 45, 58, 0.64);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .gm-dialogue-body {
            color: var(--um-ink-strong);
            font-size: 0.98rem;
            line-height: 1.75;
        }

        .gm-dialogue-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.6rem;
        }

        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, rgba(215, 177, 74, 0.8), rgba(0, 39, 76, 0.45)) !important;
        }

        .stCodeBlock,
        .stCode,
        [data-testid="stCode"],
        [data-testid="stCodeBlock"] {
            background: transparent !important;
        }

        .stCodeBlock pre,
        .stCode pre,
        [data-testid="stCode"] pre,
        [data-testid="stCodeBlock"] pre {
            background: #111827 !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 0.95rem 1rem !important;
            margin: 0.35rem 0 0 0 !important;
            overflow-x: auto !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
            box-shadow: none !important;
        }

        .stCodeBlock pre code,
        .stCode pre code,
        [data-testid="stCode"] pre code,
        [data-testid="stCodeBlock"] pre code {
            display: block !important;
            background: transparent !important;
            color: #f8fafc !important;
            font-family: Consolas, "SFMono-Regular", Menlo, Monaco, monospace !important;
            font-size: 0.92rem !important;
            line-height: 1.58 !important;
            text-shadow: none !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        .stCodeBlock pre code *,
        .stCode pre code *,
        [data-testid="stCode"] pre code *,
        [data-testid="stCodeBlock"] pre code * {
            background: transparent !important;
            color: inherit !important;
            -webkit-text-fill-color: currentColor !important;
            text-shadow: none !important;
            opacity: 1 !important;
            border: none !important;
            box-shadow: none !important;
        }

        .stCaption {
            color: #344557;
        }

        button[kind="secondary"],
        button[kind="secondary"] *,
        [data-baseweb="tab-list"] *,
        [data-baseweb="tab"] *,
        [role="tab"] * {
            color: var(--um-ink-strong) !important;
        }

        [data-baseweb="tab"] {
            background: rgba(248, 243, 233, 0.94) !important;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            background: rgba(215, 177, 74, 0.26) !important;
        }

        .gm-statusline {
            position: sticky;
            top: 0.35rem;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.85rem;
            margin: 0.25rem 0 1rem 0;
            padding: 0.62rem 0.75rem;
            border: 1px solid rgba(0, 39, 76, 0.12);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(247, 244, 236, 0.95), rgba(247, 244, 236, 0.84));
            backdrop-filter: blur(8px);
        }

        .gm-status-kicker {
            color: rgba(31, 45, 58, 0.62);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .gm-status-title {
            color: var(--um-blue);
            font-size: 0.98rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .gm-status-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem;
            justify-content: flex-end;
        }

        .gm-status-pill {
            display: inline-flex;
            align-items: center;
            min-height: 1.65rem;
            padding: 0.12rem 0.52rem;
            border-radius: 999px;
            border: 1px solid rgba(0, 39, 76, 0.12);
            background: rgba(255, 253, 248, 0.75);
            color: rgba(20, 33, 46, 0.86);
            font-size: 0.74rem;
            font-weight: 700;
        }

        .gm-settings {
            margin: 0 0 0.9rem 0;
        }

        .gm-loading-shell {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            gap: 0.7rem;
            min-width: min(25rem, calc(100vw - 2rem));
            padding: 1.35rem 1.5rem 1.2rem;
            border: 1px solid rgba(215, 177, 74, 0.28);
            border-radius: 8px;
            background: rgba(8, 20, 27, 0.82);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
            color: var(--um-ink-strong);
        }

        .gm-loading-shell *,
        .gm-loading-inline * {
            color: #f8f1df !important;
        }

        .gm-parse-overlay {
            position: fixed;
            inset: 0;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            background:
                radial-gradient(circle at 50% 45%, rgba(215, 177, 74, 0.18), transparent 18rem),
                linear-gradient(180deg, rgba(3, 9, 14, 0.94), rgba(5, 18, 24, 0.96));
            text-align: center;
        }

        .gm-loading-inline {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--um-ink-strong);
            padding: 0.2rem 0;
        }

        .gm-loading-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 3.2rem;
            height: 3.2rem;
            border-radius: 999px;
            border: 1px solid rgba(215, 177, 74, 0.5);
            background:
                radial-gradient(circle, rgba(255, 244, 197, 0.96) 0 0.22rem, transparent 0.25rem),
                conic-gradient(from 0deg, transparent 0deg, rgba(215, 177, 74, 0.88) 38deg, transparent 74deg),
                rgba(7, 18, 24, 0.92);
            box-shadow: 0 0 0 0 rgba(215, 177, 74, 0.22);
            animation: gm-sweep 1.9s linear infinite, gm-glow 1.9s ease-in-out infinite;
        }

        .gm-loading-kicker {
            color: rgba(215, 177, 74, 0.92);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .gm-loading-text {
            color: #fff2c6;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.35;
        }

        .gm-loading-subtext {
            max-width: 20rem;
            color: rgba(246, 240, 223, 0.72);
            font-size: 0.84rem;
            line-height: 1.55;
        }

        .gm-loading-dots {
            display: inline-flex;
            align-items: center;
            gap: 0.18rem;
        }

        .gm-loading-dots span {
            width: 0.22rem;
            height: 0.22rem;
            border-radius: 50%;
            background: rgba(215, 177, 74, 0.78);
            animation: gm-pulse 1.4s ease-in-out infinite;
        }

        .gm-loading-dots span:nth-child(2) {
            animation-delay: 0.18s;
        }

        .gm-loading-dots span:nth-child(3) {
            animation-delay: 0.36s;
        }

        @keyframes gm-pulse {
            0%, 80%, 100% {
                opacity: 0.25;
                transform: translateY(0);
            }
            40% {
                opacity: 0.85;
                transform: translateY(-1px);
            }
        }

        @keyframes gm-glow {
            0%, 100% {
                box-shadow: 0 0 0 0 rgba(215, 177, 74, 0.2), 0 0 26px rgba(215, 177, 74, 0.12);
            }
            50% {
                box-shadow: 0 0 0 0.55rem rgba(215, 177, 74, 0.04), 0 0 34px rgba(215, 177, 74, 0.24);
            }
        }

        @keyframes gm-sweep {
            to {
                transform: rotate(360deg);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if PUBLIC_DEMO_MODE:
        bg_uri = _image_data_uri(str(_demo_background_path()))
        bg_layer = f"url('{bg_uri}')" if bg_uri else "linear-gradient(135deg, #071821, #14212e)"
        st.markdown(
            f"""
            <style>
            .stApp {{
                background:
                    radial-gradient(circle at 18% 18%, rgba(215, 177, 74, 0.18), transparent 22rem),
                    linear-gradient(90deg, rgba(3, 9, 14, 0.9) 0%, rgba(5, 18, 24, 0.72) 42%, rgba(5, 14, 20, 0.54) 100%),
                    {bg_layer} center center / cover fixed no-repeat !important;
                color: #f6f0df;
            }}

            [data-testid="stHeader"] {{
                background: linear-gradient(180deg, rgba(3, 9, 14, 0.78), rgba(3, 9, 14, 0.1)) !important;
            }}

            [data-testid="stSidebar"] {{
                background: rgba(7, 18, 24, 0.88) !important;
                border-right: 1px solid rgba(215, 177, 74, 0.18);
            }}

            [data-testid="stSidebar"] *,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] button {{
                color: #f6f0df !important;
            }}

            .block-container {{
                max-width: 1060px;
                padding-top: 1.1rem;
            }}

            .gm-case-hero {{
                min-height: 23rem;
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
                border: 1px solid rgba(215, 177, 74, 0.25);
                background:
                    linear-gradient(90deg, rgba(5, 12, 17, 0.83), rgba(5, 12, 17, 0.36)),
                    radial-gradient(circle at 4% 92%, rgba(215, 177, 74, 0.13), transparent 18rem) !important;
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
            }}

            .gm-case-kicker,
            .gm-section-eyebrow {{
                color: #d8bb67 !important;
            }}

            .gm-case-title {{
                color: #fff6d8 !important;
                font-size: 3.4rem;
                text-shadow: 0 2px 22px rgba(0, 0, 0, 0.62);
            }}

            .gm-case-copy,
            .gm-case-note,
            .gm-feature-copy,
            .gm-section-copy {{
                color: rgba(246, 240, 223, 0.9) !important;
            }}

            .gm-feature-grid {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}

            .gm-feature,
            .gm-investigator-card,
            .gm-report-card,
            .gm-hint-card,
            .gm-side-panel,
            .gm-statusline,
            .gm-session-banner,
            [data-testid="stExpander"],
            [data-testid="stAlert"],
            div[data-testid="stMetric"] {{
                background: rgba(8, 20, 27, 0.76) !important;
                border-color: rgba(215, 177, 74, 0.2) !important;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
                backdrop-filter: blur(10px);
            }}

            .gm-feature-title,
            .gm-investigator-title,
            .gm-report-title,
            .gm-hint-title,
            .gm-side-title,
            .gm-stat-value,
            .gm-section-title,
            .gm-status-title,
            .gm-session-title,
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
                color: #fff2c6 !important;
            }}

            .gm-investigator-copy,
            .gm-report-copy,
            .gm-hint-copy,
            .gm-side-copy,
            .gm-status-kicker,
            .gm-stat-label {{
                color: rgba(246, 240, 223, 0.82) !important;
            }}

            .gm-hint-list li {{
                color: rgba(246, 240, 223, 0.86) !important;
            }}

            .gm-dialogue {{
                background: rgba(7, 18, 24, 0.58) !important;
                border-color: rgba(215, 177, 74, 0.16) !important;
                box-shadow: none !important;
            }}

            .gm-dialogue-user {{
                background: rgba(25, 107, 106, 0.24) !important;
                border-color: rgba(144, 215, 203, 0.24) !important;
            }}

            .gm-dialogue-keeper {{
                border-left-color: rgba(215, 177, 74, 0.82) !important;
            }}

            .gm-dialogue-label {{
                color: rgba(215, 177, 74, 0.88) !important;
            }}

            .gm-dialogue-body {{
                color: rgba(246, 240, 223, 0.94) !important;
            }}

            .gm-status-pill {{
                background: rgba(9, 25, 31, 0.72) !important;
                border-color: rgba(215, 177, 74, 0.22) !important;
                color: #f7efd8 !important;
            }}

            .gm-stat {{
                background: rgba(9, 25, 31, 0.62) !important;
                border-color: rgba(215, 177, 74, 0.18) !important;
            }}

            .gm-chip {{
                background: rgba(9, 25, 31, 0.76);
                border-color: rgba(215, 177, 74, 0.26);
                color: #f7efd8 !important;
            }}

            .gm-chip-accent {{
                background: rgba(25, 107, 106, 0.32) !important;
                border-color: rgba(144, 215, 203, 0.36) !important;
                color: #dcfff8 !important;
            }}

            .gm-stage {{
                color: #f8f1df !important;
            }}

            .gm-stage span {{
                color: #f8f1df !important;
            }}

            .gm-stage::before {{
                background: rgba(246, 240, 223, 0.26) !important;
            }}

            .gm-stage-dot {{
                background: rgba(8, 20, 27, 0.96) !important;
                border-color: rgba(246, 240, 223, 0.32) !important;
            }}

            .gm-stage-active,
            .gm-stage-complete {{
                color: #fff2c6 !important;
            }}

            .gm-stage-active .gm-stage-dot {{
                background: #58b4a9 !important;
                border-color: rgba(204, 255, 246, 0.9) !important;
                box-shadow: 0 0 0 4px rgba(88, 180, 169, 0.2);
            }}

            .gm-stage-complete .gm-stage-dot {{
                background: #d8bb67 !important;
                border-color: rgba(255, 239, 187, 0.9) !important;
            }}

            .gm-case-note {{
                border-left-color: rgba(215, 177, 74, 0.72);
                background: rgba(7, 18, 24, 0.62) !important;
            }}

            .stButton > button,
            .stDownloadButton > button {{
                background: linear-gradient(180deg, rgba(222, 185, 86, 0.96), rgba(154, 119, 46, 0.96)) !important;
                color: #071219 !important;
                border-color: rgba(255, 237, 177, 0.44) !important;
                min-height: 2.85rem;
                letter-spacing: 0.02em;
                white-space: nowrap;
            }}

            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                background: linear-gradient(180deg, rgba(244, 210, 110, 0.98), rgba(179, 139, 54, 0.98)) !important;
                color: #061016 !important;
            }}

            [data-testid="stChatInput"] {{
                max-width: min(62rem, calc(100vw - 2rem));
                margin: 0 auto;
                padding: 0 !important;
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
            }}

            [data-testid="stBottom"],
            [data-testid="stBottomBlockContainer"],
            [data-testid="stChatFloatingInputContainer"] {{
                background: linear-gradient(180deg, rgba(3, 12, 17, 0), rgba(3, 12, 17, 0.84) 38%, rgba(3, 12, 17, 0.96)) !important;
                box-shadow: none !important;
                border: 0 !important;
            }}

            [data-testid="stBottom"] > div,
            [data-testid="stBottomBlockContainer"] > div,
            [data-testid="stChatFloatingInputContainer"] > div {{
                background: transparent !important;
                box-shadow: none !important;
                border: 0 !important;
            }}

            [data-testid="stBottomBlockContainer"] {{
                padding: 1.15rem 0 0.8rem !important;
            }}

            [data-testid="stChatInput"] > div {{
                background: rgba(8, 22, 30, 0.9) !important;
                border: 1px solid rgba(238, 223, 180, 0.58) !important;
                border-radius: 999px !important;
                box-shadow:
                    0 18px 42px rgba(0, 0, 0, 0.42),
                    inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
                backdrop-filter: blur(12px);
            }}

            [data-testid="stChatInput"] form,
            [data-testid="stChatInput"] [data-baseweb="textarea"],
            [data-testid="stChatInput"] [data-baseweb="base-input"],
            [data-testid="stChatInput"] [data-baseweb="textarea"] > div,
            [data-testid="stChatInput"] [data-baseweb="base-input"] > div,
            [data-testid="stChatInput"] div:has(> textarea) {{
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
                outline: 0 !important;
            }}

            [data-testid="stChatInput"] textarea,
            .stTextInput input,
            .stTextArea textarea,
            .stNumberInput input,
            .stSelectbox [data-baseweb="select"] > div {{
                background: rgba(6, 18, 24, 0.78) !important;
                color: #f8f1df !important;
                border-color: rgba(215, 177, 74, 0.22) !important;
            }}

            [data-testid="stChatInput"] textarea {{
                min-height: 2.9rem !important;
                padding: 0.7rem 3.1rem 0.65rem 1.05rem !important;
                background: transparent !important;
                border: 0 !important;
                color: #f8f1df !important;
                box-shadow: none !important;
                outline: 0 !important;
            }}

            [data-testid="stChatInput"] textarea::placeholder {{
                color: rgba(221, 229, 236, 0.52) !important;
                opacity: 1 !important;
            }}

            [data-testid="stChatInput"] button {{
                background: transparent !important;
                border: 0 !important;
                color: #e9edf4 !important;
                box-shadow: none !important;
            }}

            [data-testid="stChatInput"] button svg {{
                color: #e9edf4 !important;
                fill: #e9edf4 !important;
                stroke: #e9edf4 !important;
                opacity: 0.92;
            }}

            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stAlert"] *,
            .stCaption {{
                color: rgba(246, 240, 223, 0.92) !important;
            }}

            .gm-loading-shell,
            .gm-loading-shell *,
            .gm-loading-inline,
            .gm-loading-inline * {{
                color: #f8f1df !important;
            }}

            [data-testid="stExpander"] {{
                background: rgba(245, 241, 232, 0.94) !important;
                border-color: rgba(215, 177, 74, 0.24) !important;
            }}

            [data-testid="stExpander"] *,
            [data-testid="stExpander"] summary *,
            [data-testid="stExpander"] label,
            [data-testid="stExpander"] p,
            [data-testid="stExpander"] span {{
                color: #14212e !important;
            }}

            @media (max-width: 760px) {{
                .gm-case-hero {{
                    min-height: 20rem;
                }}

                .gm-case-title {{
                    font-size: 2.25rem;
                }}

                .gm-feature-grid {{
                    grid-template-columns: 1fr;
                }}

                .gm-statusline {{
                    align-items: flex-start;
                    flex-direction: column;
                }}

                .gm-status-meta {{
                    justify-content: flex-start;
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def _render_section_header(title: str, eyebrow: str, copy: str = '') -> None:
    copy_html = f'<div class="gm-section-copy">{copy}</div>' if copy else ''
    st.markdown(
        f"""
        <div class="gm-section">
            <div class="gm-section-eyebrow">{eyebrow}</div>
            <div class="gm-section-title">{title}</div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_line(state: dict[str, object], db: Database | None = None) -> None:
    if not state.get('current_scene_id') and not state.get('current_plot_id'):
        return

    scene_id = str(state.get('current_scene_id', '') or '')
    plot_id = str(state.get('current_plot_id', '') or '')
    scene = db.get_scene(scene_id) if db and scene_id else None
    plot = db.get_plot(plot_id) if db and plot_id else None
    scene_name = str((scene or {}).get('scene_name', '') or scene_id or 'Story ready')
    plot_name = str((plot or {}).get('plot_name', '') or plot_id or 'Opening beat')
    language = str(state.get('output_language', 'English') or 'English')
    meta = ''.join(
        f"<span class='gm-status-pill'>{escape(item)}</span>"
        for item in [plot_name, language]
        if item
    )
    st.html(
        f"""
        <div class="gm-statusline">
            <div>
                <div class="gm-status-kicker">Current table</div>
                <div class="gm-status-title">{escape(scene_name)}</div>
            </div>
            <div class="gm-status-meta">{meta}</div>
        </div>
        """
    )


def _render_public_dialogue(role: str, text: object, dice: object = None, skill_check: object = None) -> None:
    role_class = 'user' if role == 'user' else 'keeper'
    label = 'Investigator' if role_class == 'user' else 'Keeper'
    body = '<br>'.join(escape(str(text or '')).splitlines()) or '&nbsp;'
    meta_items = []
    if dice:
        meta_items.append(f'Dice: {dice}')
    if skill_check:
        meta_items.append(f'Skill: {skill_check}')
    meta_html = ''.join(f"<span class='gm-chip'>{escape(str(item))}</span>" for item in meta_items)
    meta_block = f"<div class='gm-dialogue-meta'>{meta_html}</div>" if meta_html else ''
    st.html(
        f"""
        <div class="gm-dialogue gm-dialogue-{role_class}">
            <div class="gm-dialogue-label">{label}</div>
            <div class="gm-dialogue-body">{body}</div>
            {meta_block}
        </div>
        """
    )


def _render_stage_tracker(active_stage: str) -> None:
    items = []
    stages = _demo_items('stages') or DEFAULT_DEMO_MANIFEST['stages']
    active_index = 0
    valid_stages = [item for item in stages if isinstance(item, dict)]
    for idx, item in enumerate(valid_stages):
        if isinstance(item, dict) and str(item.get('key', '')).strip() == active_stage:
            active_index = idx
            break
    for item in valid_stages:
        stage_index = len(items)
        key = str(item.get('key', '')).strip()
        label = str(item.get('label', key or 'Step')).strip()
        state_class = ''
        if key == active_stage:
            state_class = ' gm-stage-active'
        elif stage_index < active_index:
            state_class = ' gm-stage-complete'
        items.append(
            f"""
            <div class="gm-stage{state_class}">
                <span class="gm-stage-dot"></span>
                <span>{escape(label)}</span>
            </div>
            """
        )
    st.html(f"<div class='gm-stage-list'>{''.join(items)}</div>")


def _story_text(story_source: str, key: str, default: str = '') -> str:
    if story_source == 'upload':
        upload_key = f'upload_{key}'
        if upload_key in _demo_manifest():
            return _demo_text(upload_key, default)
    return _demo_text(key, default)


def _story_items(story_source: str, key: str) -> list[object]:
    if story_source == 'upload':
        upload_key = f'upload_{key}'
        items = _demo_items(upload_key)
        if items:
            return items
    return _demo_items(key)


def _render_public_demo_entry(story_source: str = 'official') -> None:
    chips = [str(item).replace('{turn_limit}', str(PUBLIC_DEMO_MAX_TURNS)) for item in _story_items(story_source, 'chips')]
    chip_html = ''.join(f"<span class='gm-chip gm-chip-accent'>{escape(chip)}</span>" for chip in chips)
    features: list[tuple[str, str]] = []
    for beat in _story_items(story_source, 'beats'):
        if not isinstance(beat, dict):
            continue
        title = str(beat.get('title', '')).strip()
        copy = str(beat.get('copy', '')).strip()
        if not title and not copy:
            continue
        features.append((title, copy))
    st.markdown(
        f"""
        <div class="gm-case-hero">
            <div class="gm-case-kicker">{escape(_story_text(story_source, 'case_kicker'))}</div>
            <div class="gm-case-title">{escape(_story_text(story_source, 'title'))}</div>
            <div class="gm-case-copy">{escape(_story_text(story_source, 'tagline'))}</div>
            <div class="gm-case-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if features:
        columns = st.columns(min(3, len(features)))
        for index, (title, copy) in enumerate(features):
            with columns[index % len(columns)]:
                st.html(
                    f"""
                    <div class="gm-feature">
                        <div class="gm-feature-title">{escape(title)}</div>
                        <div class="gm-feature-copy">{escape(copy)}</div>
                    </div>
                    """,
                )
    st.markdown(
        f"""
        <div class="gm-case-note">
            {escape(_story_text(story_source, 'case_note'))}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_public_parse_summary(scenes: list[dict[str, object]], plot_count: int, est_minutes: int) -> None:
    first_scene = scenes[0] if scenes else {}
    current_lead = str(first_scene.get('scene_name', '') or 'The opening scene').strip()
    st.markdown(
        f"""
        <div class="gm-case-hero">
            <div class="gm-case-kicker">{escape(_demo_text('ready_kicker'))}</div>
            <div class="gm-case-title">{escape(_demo_text('title'))}</div>
            <div class="gm-case-copy">
                {escape(_demo_text('ready_copy'))}
            </div>
            <div class="gm-case-chips">
                <span class="gm-chip">{len(scenes)} scenes</span>
                <span class="gm-chip">{plot_count} story beats</span>
                <span class="gm-chip">{est_minutes} min estimate</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if current_lead:
        st.markdown(f"**{escape(_demo_text('route_label'))}**")
        st.write(f'{current_lead}. New leads will open as you play.')


def _render_session_banner(state: dict[str, object], player_turns: int) -> None:
    scene = str(state.get('current_scene_id', '') or 'Scene')
    plot = str(state.get('current_plot_id', '') or 'Plot')
    remaining = max(0, PUBLIC_DEMO_MAX_TURNS - player_turns)
    chips = [
        f'{scene}',
        f'{plot}',
        f'{remaining} turns left',
        str(state.get('output_language', 'English') or 'English'),
    ]
    chip_html = ''.join(f"<span class='gm-chip'>{escape(chip)}</span>" for chip in chips)
    st.markdown(
        f"""
        <div class="gm-session-banner">
            <div class="gm-session-title">{escape(_demo_text('title'))}</div>
            <div class="gm-status-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _short_text(value: object, limit: int = 140) -> str:
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + '...'


def _hint_subject(text: str, prefixes: tuple[str, ...]) -> str:
    compact = re.sub(r'\s+', ' ', text or '').strip()
    lower = compact.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            return compact[len(prefix) :].strip(' .:;-') or compact
    return compact


def _fallback_public_hint_lines(db: Database, state: dict[str, object]) -> list[str]:
    scene = db.get_scene(str(state.get('current_scene_id', '') or ''))
    plot = db.get_plot(str(state.get('current_plot_id', '') or ''))
    scene_name = str((scene or {}).get('scene_name', '') or '').strip()
    scene_description = str((scene or {}).get('scene_description', '') or '').strip()
    plot_name = str((plot or {}).get('plot_name', '') or '').strip()
    plot_goal = str((plot or {}).get('plot_goal', '') or '').strip()
    raw_text = str((plot or {}).get('raw_text', '') or '').strip()
    language = str(state.get('output_language', 'English') or 'English').lower()
    chinese = language.startswith('chinese')

    subject = _hint_subject(
        plot_name or plot_goal,
        (
            'speak with ',
            'talk to ',
            'question ',
            'inspect ',
            'examine ',
            'read ',
            'confront ',
            'search ',
            'open ',
            'go to ',
            'enter ',
        ),
    )
    lower_plot = (plot_name or plot_goal).lower()
    lines: list[str] = []

    if 'speak' in lower_plot or 'talk' in lower_plot or 'question' in lower_plot:
        if chinese:
            lines.append(f'先从 {subject or "当前人物"} 入手。可以问对方看见了什么、有什么东西不见了、哪个细节让人不舒服。')
        else:
            lines.append(f'Start with {subject or "the person in front of you"}. Ask what they saw, what is missing, and which detail feels out of place.')
    elif 'inspect' in lower_plot or 'examine' in lower_plot or 'search' in lower_plot:
        if chinese:
            lines.append(f'{subject or "当前地点"} 值得慢一点查。重点看重复的形状、刻痕、摆放得过于刻意的东西，或者通往下一个地方的痕迹。')
        else:
            lines.append(f'{subject or "the current place"} is worth a closer look. Search for repeated shapes, marks, anything arranged too deliberately, or a route to the next lead.')
    elif 'open' in lower_plot or 'locked' in lower_plot or 'gate' in lower_plot:
        if chinese:
            lines.append(f'围绕 {subject or "这道阻碍"} 做一个具体动作：请持钥匙的人帮忙、检查锁和附近痕迹，或尝试安静打开。')
        else:
            lines.append(f'Focus on {subject or "the obstacle"}: ask someone with access, inspect the lock and nearby traces, or try opening it quietly.')
    elif 'read' in lower_plot:
        if chinese:
            lines.append(f'先读 {subject or "这份文字材料"}，再把反复出现的名字、数字、地点和你已经听到的异常现象对照起来。')
        else:
            lines.append(f'Read {subject or "the written clue"} first, then compare repeated names, numbers, places, and odd details you have already heard.')
    elif 'confront' in lower_plot:
        if chinese:
            lines.append(f'不要空手质问 {subject or "对方"}。先拿一个你已经发现的证据或矛盾点去施压。')
        else:
            lines.append(f'Do not confront {subject or "them"} empty-handed. Bring one piece of evidence or one contradiction you have already found.')
    elif 'end' in lower_plot or 'decide' in lower_plot:
        if chinese:
            lines.append('这里更像选择题：先确认你理解了规则，再决定是安静地修正它，还是用更激烈的办法中断它。')
        else:
            lines.append('This is closer to a choice point: make sure you understand the pattern, then decide whether to correct it quietly or interrupt it more forcefully.')
    elif plot_name or plot_goal:
        handle = plot_name or plot_goal
        if chinese:
            lines.append(f'围绕“{handle}”做一个具体动作：询问、观察、搜查，或比较你已经拿到的线索。')
        else:
            lines.append(f'Pick one concrete action around "{handle}": ask, inspect, search, or compare it with a clue you already have.')

    if raw_text and not lines:
        if chinese:
            lines.append(f'先从眼前最具体的东西下手：{_short_text(raw_text, 90)}')
        else:
            lines.append(f'Start with the most concrete thing in front of you: {_short_text(raw_text, 90)}')

    deduped: list[str] = []
    for line in lines:
        clean = line.strip()
        if clean and clean not in deduped:
            deduped.append(clean)
    if deduped:
        return deduped[:1]
    if scene_name:
        if chinese:
            return [f'先选一个场景里的具体对象行动：询问一个人、检查一件物品，或前往一个被提到的地点。']
        return ['Choose one concrete thing in the scene: question a person, inspect an object, or move toward a named place.']
    return ['Try one specific action: ask, inspect, search, or move to a named location.']


def _recent_play_context(messages: list[dict[str, object]], limit: int = 4) -> str:
    recent = messages[-limit:]
    lines: list[str] = []
    for turn in recent:
        user = _short_text(turn.get('user', ''), 220)
        agent = _short_text(turn.get('agent', ''), 320)
        if user:
            lines.append(f'Player: {user}')
        if agent:
            lines.append(f'Keeper: {agent}')
    return '\n'.join(lines)


def _generate_public_hint(db: Database, state: dict[str, object]) -> list[str]:
    scene = db.get_scene(str(state.get('current_scene_id', '') or '')) or {}
    plot = db.get_plot(str(state.get('current_plot_id', '') or '')) or {}
    language = str(state.get('output_language', 'English') or 'English')
    recent_context = _recent_play_context(list(st.session_state.get('messages', [])))
    prompt = f"""You are a tabletop RPG Keeper giving a stuck player one gentle hint.

Goal: help the player choose a next action without spoiling hidden facts or solving the mystery.

Rules:
- Output exactly one short hint sentence.
- Write entirely in {language}.
- Do not mention that you are an AI or that this is a hint.
- Do not reveal hidden clues, answers, culprit identity, final solution, or future plot beats.
- Use the player's recent context first. If they are stuck at an obstacle, suggest a concrete action they can try.
- Keep it under 28 words in English, or under 45 Chinese characters.

Current scene:
Name: {scene.get('scene_name', '')}
Description: {_short_text(scene.get('scene_description', ''), 700)}

Current plot:
Name: {plot.get('plot_name', '')}
Goal: {plot.get('plot_goal', '')}
Keeper-only plot notes: {_short_text(plot.get('raw_text', ''), 900)}

Recent play:
{recent_context or '(No player action yet.)'}
"""
    try:
        text = call_llm(prompt, step_name='generate_player_hint', max_retries=1, timeout=25).strip()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Public hint LLM generation failed; using fallback hint: %s', exc)
        return _fallback_public_hint_lines(db, state)
    text = re.sub(r'\s+', ' ', text).strip().strip('"').strip("'")
    if not text:
        return _fallback_public_hint_lines(db, state)
    return [_short_text(text, 180)]


def _render_public_hint_controls(db: Database, state: dict[str, object]) -> None:
    language = str(state.get('output_language', 'English') or 'English').lower()
    chinese = language.startswith('chinese')
    button_label = 'Get a hint' if not chinese else '获取提示'
    title = 'Hint' if not chinese else '提示'

    hint_col, _ = st.columns([1.1, 4])
    with hint_col:
        if st.button(button_label, key='public_demo_hint_button', use_container_width=True, type='tertiary'):
            if not st.session_state.get('public_demo_hint_lines'):
                budget_ok, budget_message = _reserve_public_demo_turn()
                if budget_ok:
                    st.session_state.public_demo_hint_lines = _generate_public_hint(db, state)
                else:
                    st.session_state.public_demo_hint_lines = [budget_message]

    hint_lines = st.session_state.get('public_demo_hint_lines') or []
    if not hint_lines:
        return
    hint_text = escape(str(hint_lines[0]))
    st.markdown(
        f"""
        <div class="gm-hint-card">
            <div class="gm-hint-title">{escape(title)}</div>
            <div class="gm-hint-copy">{hint_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _demo_character_presets(generated_builds: list[dict[str, object]]) -> list[dict[str, object]]:
    presets: list[dict[str, object]] = []
    for item in _demo_items('character_presets'):
        if not isinstance(item, dict):
            continue
        stats = _parse_stats_line(str(item.get('stats', '') or ''))
        if not stats or not _validate_coc_stats(stats):
            continue
        derived = _calc_derived(stats)
        archetype = str(item.get('archetype', '') or 'Investigator').strip()
        occupation = [str(skill).strip() for skill in item.get('occupation_skills', []) if str(skill).strip()] if isinstance(item.get('occupation_skills'), list) else []
        interest = [str(skill).strip() for skill in item.get('interest_skills', []) if str(skill).strip()] if isinstance(item.get('interest_skills'), list) else []
        presets.append(
            {
                'archetype': archetype,
                'default_name': str(item.get('default_name', '') or archetype).strip(),
                'background': str(item.get('background', '') or '').strip(),
                'stats': stats,
                'line': _stats_to_line(stats),
                'derived': derived,
                'occupation_suggested': _ensure_default_skill_lines(occupation),
                'interest_suggested': interest,
            }
        )

    if presets:
        return presets

    fallback: list[dict[str, object]] = []
    for build in generated_builds[:3]:
        fallback.append(
            {
                'archetype': str(build.get('archetype', '') or 'Investigator'),
                'default_name': str(build.get('archetype', '') or 'Investigator'),
                'background': 'A capable investigator ready to follow the evidence.',
                'stats': build.get('stats', {}),
                'line': str(build.get('line', '')),
                'derived': build.get('derived', {}),
                'occupation_suggested': _ensure_default_skill_lines(list(build.get('occupation_suggested', []))),
                'interest_suggested': list(build.get('interest_suggested', [])),
            }
        )
    return fallback


def _render_investigator_card(preset: dict[str, object]) -> None:
    stats = preset.get('stats', {}) if isinstance(preset.get('stats'), dict) else {}
    derived = preset.get('derived', {}) if isinstance(preset.get('derived'), dict) else {}
    stat_items = [
        ('INT', stats.get('INT')),
        ('DEX', stats.get('DEX')),
        ('POW', stats.get('POW')),
        ('EDU', stats.get('EDU')),
        ('HP', derived.get('HP')),
        ('MP', derived.get('MP')),
        ('SAN', derived.get('SAN')),
        ('Skills', derived.get('occupation_skill_points')),
    ]
    stat_html = ''.join(
        f"""
        <div class="gm-stat">
            <div class="gm-stat-label">{escape(label)}</div>
            <div class="gm-stat-value">{escape(str(value or '-'))}</div>
        </div>
        """
        for label, value in stat_items
    )
    st.markdown(
        f"""
        <div class="gm-investigator-card">
            <div class="gm-investigator-title">{escape(str(preset.get('archetype', 'Investigator')))}</div>
            <div class="gm-investigator-copy">{escape(str(preset.get('background', '') or ''))}</div>
            <div class="gm-stat-grid">{stat_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _profile_from_preset(
    preset: dict[str, object],
    name: str,
    background: str,
    stats_line: str,
    occupation_alloc: str,
    interest_alloc: str,
) -> dict[str, object] | None:
    parsed = _parse_stats_line(stats_line)
    if not parsed or not _validate_coc_stats(parsed):
        return None
    derived = _calc_derived(parsed)
    return {
        'name': name.strip(),
        'background': background.strip(),
        'characteristics': parsed,
        'derived_attributes': {
            'HP': derived['HP'],
            'MP': derived['MP'],
            'SAN': derived['SAN'],
        },
        'skill_points': {
            'occupation': derived['occupation_skill_points'],
            'personal_interest': derived['personal_interest_points'],
        },
        'selected_archetype': str(preset.get('archetype', 'Investigator')),
        'suggested_skill_allocations': {
            'occupation': list(preset.get('occupation_suggested', [])),
            'personal_interest': list(preset.get('interest_suggested', [])),
        },
        'chosen_skill_allocations': {
            'occupation': [line.strip() for line in occupation_alloc.splitlines() if line.strip()],
            'personal_interest': [line.strip() for line in interest_alloc.splitlines() if line.strip()],
        },
    }


def _render_public_investigation_sidebar(db: Database, state: dict[str, object], player_turns: int) -> None:
    scene = db.get_scene(str(state.get('current_scene_id', '') or ''))
    plot = db.get_plot(str(state.get('current_plot_id', '') or ''))
    scene_name = str((scene or {}).get('scene_name', '') or state.get('current_scene_id', '') or 'Scene')
    plot_name = str((plot or {}).get('plot_name', '') or state.get('current_plot_id', '') or 'Beat')
    remaining = max(0, PUBLIC_DEMO_MAX_TURNS - player_turns)

    with st.sidebar:
        st.markdown('**Case Status**')
        st.progress(min(1.0, player_turns / max(1, PUBLIC_DEMO_MAX_TURNS)), text=f'{player_turns}/{PUBLIC_DEMO_MAX_TURNS} turns')
        st.markdown(
            f"""
            <div class="gm-side-panel">
                <div class="gm-side-title">{escape(scene_name)}</div>
                <div class="gm-side-copy">{escape(plot_name)}</div>
                <div class="gm-case-chips" style="margin-top: .55rem;">
                    <span class="gm-chip">{remaining} turns left</span>
                    <span class="gm-chip">{escape(str(state.get('output_language', 'English') or 'English'))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.get('last_retrieved'):
            st.markdown('**Active Leads**')
            for doc in st.session_state.last_retrieved[:3]:
                metadata = doc.get('metadata', {}) if isinstance(doc, dict) else {}
                lead_type = str(metadata.get('type', 'lead') if isinstance(metadata, dict) else 'lead')
                content = _short_text(doc.get('content', '') if isinstance(doc, dict) else '', 90)
                if content:
                    st.caption(f'{lead_type}: {content}')


def _render_demo_end_card(db: Database, vector: ChromaStore, state: dict[str, object], player_turns: int) -> None:
    scene = db.get_scene(str(state.get('current_scene_id', '') or ''))
    scene_name = str((scene or {}).get('scene_name', '') or state.get('current_scene_id', '') or 'Current scene')
    st.markdown(
        f"""
        <div class="gm-report-card">
            <div class="gm-report-title">{escape(_demo_text('ending_title', 'Demo Session Complete'))}</div>
            <div class="gm-report-copy">{escape(_demo_text('ending_copy', 'This public preview has reached its turn limit.'))}</div>
            <div class="gm-case-chips" style="margin-top: .75rem;">
                <span class="gm-chip">{player_turns} turns played</span>
                <span class="gm-chip">{escape(scene_name)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(_demo_text('new_demo_button', 'Start New Demo'), use_container_width=True):
        _reset_to_upload_stage(db, vector)
        st.rerun()


def _ensure_default_skill_lines(skill_lines: list[str]) -> list[str]:
    normalized = []
    seen: set[str] = set()
    for line in skill_lines:
        label = line.split(':', 1)[0].strip().lower()
        normalized.append(line)
        seen.add(label)
    if 'dodge' not in seen:
        # Default Dodge: 40
        normalized.append('Dodge:40')
    if 'fighting' not in seen:
        # Default Fighting: 20
        normalized.append('Fighting:20')
    return normalized


def _render_settings(debug_mode: bool, current_language: str, stage: str, player_profile: dict[str, object] | None = None) -> None:
    if 'show_player_panel' not in st.session_state:
        st.session_state.show_player_panel = False

    if stage == 'session' and player_profile:
        with st.sidebar:
            st.markdown('**Player**')
            name = str(player_profile.get('name', '') or '').strip()
            background = str(player_profile.get('background', '') or '').strip()
            archetype = str(player_profile.get('selected_archetype', '') or '').strip()
            if name:
                st.write(f'Name: {name}')
            if archetype:
                st.write(f'Archetype: {archetype}')
            if background:
                st.write(f'Background: {background}')

            characteristics = player_profile.get('characteristics', {}) or {}
            core_lines = [
                f'{key}:{characteristics.get(key)}'
                for key in COC_CORE_KEYS
                if characteristics.get(key) is not None
            ]
            if core_lines:
                st.markdown('**Core Attributes**')
                st.code('\n'.join(core_lines), language='text')

            chosen_allocations = player_profile.get('chosen_skill_allocations', {}) or {}
            occupation = chosen_allocations.get('occupation', []) if isinstance(chosen_allocations, dict) else []
            personal_interest = chosen_allocations.get('personal_interest', []) if isinstance(chosen_allocations, dict) else []
            if occupation or personal_interest:
                st.markdown('**Skill Allocation**')
            if occupation:
                st.caption('Occupation Skills')
                st.code('\n'.join(str(item) for item in occupation), language='text')
            if personal_interest:
                st.caption('Personal Interest Skills')
                st.code('\n'.join(str(item) for item in personal_interest), language='text')

    st.markdown("<div class='gm-settings'></div>", unsafe_allow_html=True)
    with st.expander('Settings', expanded=False):
        st.selectbox('Output Language', options=['English', 'Chinese'], index=['English', 'Chinese'].index(current_language), key='output_language_select')

def _render_loading_state(target: object, text: str, centered: bool = False) -> None:
    if centered:
        safe_text = escape(text)
        target.markdown(
            f"""
            <div class="gm-parse-overlay">
                <div class="gm-loading-shell">
                    <div class="gm-loading-icon" aria-hidden="true"></div>
                    <div class="gm-loading-kicker">Keeper is preparing the table</div>
                    <div class="gm-loading-text">{safe_text}</div>
                    <div class="gm-loading-subtext">Arranging scenes, clues, and the first playable beat.</div>
                    <div class="gm-loading-dots" aria-hidden="true">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    target.html(
        f"""
        <div class="gm-loading-inline">
            <span class="gm-loading-dots" aria-hidden="true"><span></span><span></span><span></span></span>
            <span>{escape(text)}</span>
        </div>
        """
    )


def _roll_3d6_x5() -> int:
    return sum(random.randint(1, 6) for _ in range(3)) * 5


def _roll_2d6_plus_6_x5() -> int:
    return (sum(random.randint(1, 6) for _ in range(2)) + 6) * 5


def _generate_coc_stats() -> dict[str, int]:
    stats = {k: _roll_3d6_x5() for k in COC_3D6_KEYS}
    stats.update({k: _roll_2d6_plus_6_x5() for k in COC_2D6_KEYS})
    return stats


def _score_archetype(stats: dict[str, int], weights: dict[str, float]) -> float:
    return sum(float(stats.get(k, 0)) * w for k, w in weights.items())


def _calc_derived(stats: dict[str, int]) -> dict[str, int]:
    hp = int((stats['CON'] + stats['SIZ']) / 5)
    mp = int(stats['POW'] / 5)
    san = int(stats['POW'])
    occ = int(stats['EDU'] * 4)
    interest = int(stats['INT'] * 2)
    return {
        'HP': hp,
        'MP': mp,
        'SAN': san,
        'occupation_skill_points': occ,
        'personal_interest_points': interest,
    }


def _alloc_points(total: int, weighted_skills: list[tuple[str, int]]) -> list[str]:
    weight_sum = sum(w for _, w in weighted_skills) or 1
    allocated = []
    used = 0
    for idx, (skill, weight) in enumerate(weighted_skills):
        if idx == len(weighted_skills) - 1:
            points = max(0, total - used)
        else:
            points = int(round(total * (weight / weight_sum)))
            used += points
        allocated.append(f'{skill}:{points}')
    return allocated


def _stats_to_line(stats: dict[str, int]) -> str:
    return ','.join([f'{k}:{int(stats[k])}' for k in COC_CORE_KEYS])


def _parse_stats_line(stats_line: str) -> dict[str, int] | None:
    parts = [p.strip() for p in stats_line.split(',') if p.strip()]
    parsed: dict[str, int] = {}
    for part in parts:
        if ':' not in part:
            return None
        key, raw = part.split(':', 1)
        key = key.strip().upper()
        raw = raw.strip()
        if key not in COC_CORE_KEYS:
            continue
        if not re.fullmatch(r'-?\d+', raw):
            return None
        parsed[key] = int(raw)
    if any(k not in parsed for k in COC_CORE_KEYS):
        return None
    return parsed


def _validate_coc_stats(stats: dict[str, int]) -> bool:
    for key in COC_3D6_KEYS:
        v = stats.get(key, 0)
        if v % 5 != 0 or v < 15 or v > 90:
            return False
    for key in COC_2D6_KEYS:
        v = stats.get(key, 0)
        if v % 5 != 0 or v < 40 or v > 90:
            return False
    return True


def _generate_coc_builds() -> list[dict[str, object]]:
    builds: list[dict[str, object]] = []
    used_lines: set[str] = set()
    for arch in COC_ARCHETYPES:
        best_stats = None
        best_score = float('-inf')
        for _ in range(200):
            candidate = _generate_coc_stats()
            score = _score_archetype(candidate, arch['weights'])  # type: ignore[arg-type]
            if score > best_score:
                best_score = score
                best_stats = candidate
        if best_stats is None:
            best_stats = _generate_coc_stats()
        line = _stats_to_line(best_stats)
        reroll_guard = 0
        while line in used_lines and reroll_guard < 50:
            best_stats = _generate_coc_stats()
            line = _stats_to_line(best_stats)
            reroll_guard += 1
        used_lines.add(line)
        derived = _calc_derived(best_stats)
        builds.append(
            {
                'archetype': arch['name'],
                'stats': best_stats,
                'line': line,
                'derived': derived,
                'occupation_suggested': _alloc_points(int(derived['occupation_skill_points']), arch['occupation_skills']),  # type: ignore[arg-type]
                'interest_suggested': _alloc_points(int(derived['personal_interest_points']), arch['interest_skills']),  # type: ignore[arg-type]
            }
        )
    return builds


def _load_messages_from_db(db: Database) -> list[dict[str, object]]:
    rows = db.conn.execute(
        'SELECT scene_id, plot_id, user, agent FROM memory ORDER BY id ASC'
    ).fetchall()
    messages: list[dict[str, object]] = []
    for row in rows:
        user = row['user'] or ''
        if user == KP_OPENING_MARKER:
            user = ''
        agent = row['agent'] or ''
        messages.append(
            {
                'user': user,
                'agent': agent,
                'dice': None,
                'skill_check': None,
                'debug_prompts': [],
            }
        )
    return messages


STORY_RUNTIME_SESSION_KEYS = (
    'coc_builds',
    'character_stats_line',
    'selected_archetype_name',
    'occupation_alloc_text',
    'interest_alloc_text',
    'character_name_input',
    'character_background_input',
    'build_pick_label',
    'public_demo_hint_lines',
)


def _clear_story_runtime_session_state() -> None:
    st.session_state.messages = []
    st.session_state.last_retrieved = []
    for key in STORY_RUNTIME_SESSION_KEYS:
        st.session_state.pop(key, None)


def _bump_script_upload_nonce() -> None:
    st.session_state.script_upload_nonce = int(st.session_state.get('script_upload_nonce', 0)) + 1


def _set_story_notice(kind: str, text: str) -> None:
    st.session_state.story_notice = {'kind': kind, 'text': text}


def _render_story_notice() -> None:
    notice = st.session_state.pop('story_notice', None)
    if not isinstance(notice, dict):
        return
    text = str(notice.get('text', '')).strip()
    if not text:
        return
    kind = str(notice.get('kind', 'info')).strip().lower()
    if kind == 'success':
        st.success(text)
    elif kind == 'warning':
        st.warning(text)
    elif kind == 'error':
        st.error(text)
    else:
        st.info(text)


def _runtime_paths() -> tuple[str, str]:
    if not PUBLIC_DEMO_MODE:
        return 'narrative.db', '.chroma'

    session_id = st.session_state.get('public_demo_session_id')
    if not session_id:
        session_id = uuid.uuid4().hex
        st.session_state.public_demo_session_id = session_id

    session_dir = RUNTIME_ROOT / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    return str(session_dir / 'narrative.db'), str(session_dir / '.chroma')


@st.cache_data(show_spinner=False)
def _parse_demo_script_cached(script_text: str) -> dict[str, object]:
    document = read_uploaded_document(
        DEMO_SCRIPT_PATH.name,
        script_text.encode('utf-8'),
        mime_type='text/markdown',
    )
    return parse_script_bundle(source_document=document)


def _heading_title(line: str) -> str:
    return re.sub(r'^#+\s*', '', line).strip()


def _clean_heading_title(title: str) -> str:
    return re.sub(r'^(scene|plot)\s+\d+\s*:\s*', '', title, flags=re.IGNORECASE).strip() or title


def _first_sentence(text: str, fallback: str) -> str:
    compact = _short_text(text, 260)
    match = re.search(r'(.+?[.!?])(?:\s|$)', compact)
    return match.group(1).strip() if match else (compact or fallback)


def _knowledge_heading_parts(title: str) -> tuple[str, str]:
    if ':' not in title:
        return 'other', title.strip() or 'Knowledge'
    raw_type, raw_title = title.split(':', 1)
    knowledge_type = re.sub(r'[^a-z0-9_ -]+', '', raw_type.strip().lower()).replace(' ', '_')
    return knowledge_type or 'other', raw_title.strip() or title.strip()


def _parse_markdown_script_locally(script_text: str, source_file_name: str) -> dict[str, object]:
    title = 'Demo Script'
    intro_lines: list[str] = []
    scenes: list[dict[str, object]] = []
    knowledge: list[dict[str, object]] = []
    current_scene: dict[str, object] | None = None
    current_plot: dict[str, object] | None = None
    current_knowledge: dict[str, object] | None = None
    scene_intro: list[str] = []
    plot_lines: list[str] = []
    knowledge_lines: list[str] = []
    in_knowledge = False

    def finish_plot() -> None:
        nonlocal current_plot, plot_lines
        if not current_scene or not current_plot:
            return
        raw_text = '\n'.join(plot_lines).strip()
        current_plot['raw_text'] = raw_text
        current_plot['plot_goal'] = _first_sentence(raw_text, str(current_plot.get('plot_name', 'Advance the scene')))
        plots = current_scene.setdefault('plots', [])
        if isinstance(plots, list):
            plots.append(current_plot)
        current_plot = None
        plot_lines = []

    def finish_scene() -> None:
        nonlocal current_scene, scene_intro
        finish_plot()
        if not current_scene:
            return
        description = '\n'.join(scene_intro).strip()
        current_scene['scene_description'] = description or str(current_scene.get('scene_name', ''))
        current_scene['scene_goal'] = _first_sentence(description, str(current_scene.get('scene_name', 'Advance the story')))
        if not current_scene.get('plots'):
            current_scene['plots'] = [
                {
                    'plot_id': f"{current_scene['scene_id']}_plot_1",
                    'plot_index': 1,
                    'plot_name': str(current_scene.get('scene_name', 'Opening')),
                    'plot_goal': current_scene['scene_goal'],
                    'raw_text': description,
                    'status': 'pending',
                    'progress': 0.0,
                }
            ]
        scenes.append(current_scene)
        current_scene = None
        scene_intro = []

    def finish_knowledge() -> None:
        nonlocal current_knowledge, knowledge_lines
        if not current_knowledge:
            return
        current_knowledge['content'] = '\n'.join(knowledge_lines).strip()
        if current_knowledge['content']:
            knowledge.append(current_knowledge)
        current_knowledge = None
        knowledge_lines = []

    for raw_line in script_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith('# '):
            title = _heading_title(line)
            continue
        if line.startswith('## '):
            heading = _heading_title(line)
            if heading.lower().startswith('knowledge'):
                finish_scene()
                in_knowledge = True
                continue
            finish_knowledge()
            finish_scene()
            in_knowledge = False
            scene_index = len(scenes) + 1
            scene_name = _clean_heading_title(heading)
            current_scene = {
                'scene_id': f'scene_{scene_index}',
                'scene_index': scene_index,
                'scene_name': scene_name,
                'scene_goal': scene_name,
                'scene_description': '',
                'scene_summary': '',
                'status': 'pending',
                'plots': [],
            }
            continue
        if line.startswith('### ') and in_knowledge:
            finish_knowledge()
            knowledge_type, knowledge_title = _knowledge_heading_parts(_heading_title(line))
            current_knowledge = {
                'knowledge_id': f'knowledge_{len(knowledge) + 1}',
                'knowledge_type': knowledge_type,
                'title': knowledge_title,
                'content': '',
            }
            continue
        if line.startswith('### ') and current_scene:
            finish_plot()
            plots = current_scene.get('plots', [])
            plot_index = len(plots) + 1 if isinstance(plots, list) else 1
            plot_name = _clean_heading_title(_heading_title(line))
            current_plot = {
                'plot_id': f"{current_scene['scene_id']}_plot_{plot_index}",
                'plot_index': plot_index,
                'plot_name': plot_name,
                'plot_goal': plot_name,
                'raw_text': '',
                'status': 'pending',
                'progress': 0.0,
            }
            continue

        if in_knowledge and current_knowledge:
            knowledge_lines.append(line)
        elif current_plot:
            plot_lines.append(line)
        elif current_scene:
            scene_intro.append(line)
        elif line.strip():
            intro_lines.append(line.strip())

    finish_knowledge()
    finish_scene()
    intro = ' '.join(intro_lines).strip()
    scene_names = ', '.join(str(scene.get('scene_name', '')) for scene in scenes)
    return {
        'scenes': scenes,
        'knowledge': knowledge,
        'script_summary': intro or f'{title}: {scene_names}',
        'source_metadata': {
            'source_file_name': source_file_name,
            'source_type': 'markdown',
            'parser': 'local_markdown_fallback',
            'line_count': len(script_text.splitlines()),
        },
    }


def _load_preparsed_demo_bundle(script_text: str) -> dict[str, object] | None:
    if not DEMO_PARSED_PATH.exists():
        return None
    try:
        bundle = json.loads(DEMO_PARSED_PATH.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Demo preparse snapshot could not be read; falling back to parser: %s', exc)
        return None
    if not isinstance(bundle, dict):
        LOGGER.warning('Demo preparse snapshot is not a JSON object; falling back to parser.')
        return None
    source_metadata = bundle.get('source_metadata', {})
    if not isinstance(source_metadata, dict):
        source_metadata = {}
        bundle['source_metadata'] = source_metadata
    expected_hash = str(source_metadata.get('source_sha256', '') or '').strip()
    actual_hash = hashlib.sha256(script_text.encode('utf-8')).hexdigest()
    if expected_hash and expected_hash != actual_hash:
        LOGGER.warning('Demo preparse snapshot is stale; falling back to parser.')
        return None
    if not isinstance(bundle.get('scenes'), list) or not isinstance(bundle.get('knowledge'), list):
        LOGGER.warning('Demo preparse snapshot is missing scenes or knowledge; falling back to parser.')
        return None
    source_metadata['loaded_from_preparsed_snapshot'] = True
    return bundle


def _load_demo_script_bundle(*, allow_llm_parse: bool = True) -> dict[str, object]:
    if not DEMO_SCRIPT_PATH.exists():
        raise FileNotFoundError(f'Demo script not found: {DEMO_SCRIPT_PATH}')
    script_text = DEMO_SCRIPT_PATH.read_text(encoding='utf-8')
    preparsed_bundle = _load_preparsed_demo_bundle(script_text)
    if preparsed_bundle is not None:
        return preparsed_bundle
    if not allow_llm_parse:
        LOGGER.info('Demo preparse unavailable; using local markdown fallback without LLM parsing.')
        return _parse_markdown_script_locally(script_text, DEMO_SCRIPT_PATH.name)
    try:
        return _parse_demo_script_cached(script_text)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning('Demo LLM parse failed; using local markdown fallback: %s', exc)
        return _parse_markdown_script_locally(script_text, DEMO_SCRIPT_PATH.name)


def _install_script_bundle(db: Database, vector: ChromaStore, bundle: dict[str, object]) -> tuple[int, int]:
    scenes = bundle.get('scenes', [])
    knowledge = bundle.get('knowledge', [])
    script_summary = str(bundle.get('script_summary', '') or '')
    source_metadata = bundle.get('source_metadata', {})

    if not isinstance(scenes, list):
        scenes = []
    if not isinstance(knowledge, list):
        knowledge = []

    db.reset_story_data()
    vector.reset()
    db.insert_scenes(scenes)
    db.insert_knowledge(knowledge)
    vector.add_from_scenes(scenes, knowledge=knowledge)
    db.save_summary('script', script_summary)
    db.save_summary('parse_source_meta', json.dumps(source_metadata, ensure_ascii=False))

    first_scene, first_plot = _first_playable_position(scenes)
    current_scene_id = str(first_scene.get('scene_id', '')) if first_scene else ''
    current_plot_id = str(first_plot.get('plot_id', '')) if first_plot else ''
    if current_scene_id:
        db.update_scene(current_scene_id, {'status': 'in_progress'})
    db.update_system_state(
        {
            'stage': 'parse',
            'current_scene_id': current_scene_id,
            'current_plot_id': current_plot_id,
            'plot_progress': 0.0,
            'scene_progress': 0.0,
            'current_scene_intro': '',
        }
    )
    db.save_initial_story_snapshot()
    _clear_story_runtime_session_state()
    return len(scenes), len(knowledge)


def _first_playable_position(scenes: list[dict[str, object]]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    for scene in scenes:
        plots = scene.get('plots', [])
        if isinstance(plots, list) and plots:
            first_plot = plots[0]
            if isinstance(first_plot, dict):
                return scene, first_plot
    return None, None


def _player_turn_count(messages: list[dict[str, object]]) -> int:
    return sum(1 for turn in messages if str(turn.get('user', '') or '').strip())


def _reset_to_upload_stage(db: Database, vector: ChromaStore) -> None:
    db.reset_story_data()
    db.delete_initial_story_snapshot()
    vector.reset()
    db.update_system_state(
        {
            'stage': 'upload',
            'current_scene_id': '',
            'current_plot_id': '',
            'plot_progress': 0.0,
            'scene_progress': 0.0,
            'player_profile': {},
            'current_scene_intro': '',
            'navigation_state': {},
            'current_visit_id': 0,
        }
    )
    _clear_story_runtime_session_state()
    _bump_script_upload_nonce()


def _render_story_management_actions(stage: str, db: Database, vector: ChromaStore) -> None:
    if stage == 'upload':
        return

    spacer_col, action_col1, action_col2 = st.columns([4.9, 1.7, 1.25])
    with action_col1:
        restart_clicked = st.button(
            'Restart Demo' if PUBLIC_DEMO_MODE else 'Restart Game',
            key=f'restart_game_{stage}',
            use_container_width=True,
            disabled=not db.has_initial_story_snapshot(),
            type='tertiary',
        )
    with action_col2:
        reparse_clicked = st.button(
            'New Demo' if PUBLIC_DEMO_MODE else 'Parse New Script',
            key=f'parse_new_script_{stage}',
            use_container_width=True,
            type='tertiary',
        )

    if restart_clicked:
        try:
            db.restore_initial_story_snapshot()
            _clear_story_runtime_session_state()
            _set_story_notice('success', 'Game restarted from the parsed snapshot.')
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f'Restart failed: {exc}')

    if reparse_clicked:
        _reset_to_upload_stage(db, vector)
        message = 'Demo data cleared. Start a new demo when ready.' if PUBLIC_DEMO_MODE else 'Story data cleared. Upload a new script to parse.'
        _set_story_notice('success', message)
        st.rerun()


def run_app() -> None:
    st.set_page_config(page_title='Script-Driven Narrative Agent', layout='wide')
    _inject_demo_theme()
    _enforce_public_demo_config()
    _cleanup_public_demo_runtime_once()

    if 'db' not in st.session_state:
        db_path, vector_path = _runtime_paths()
        st.session_state.db = Database(db_path)
        st.session_state.vector = ChromaStore(vector_path)
        st.session_state.agent = NarrativeAgent(st.session_state.db, st.session_state.vector)
        st.session_state.messages = _load_messages_from_db(st.session_state.db)
        st.session_state.last_retrieved = []

    db: Database = st.session_state.db
    vector: ChromaStore = st.session_state.vector
    agent: NarrativeAgent = st.session_state.agent
    _render_public_demo_diagnostics(db)
    if not hasattr(agent, 'set_debug_mode'):
        st.session_state.agent = NarrativeAgent(db, vector)
        agent = st.session_state.agent

    state = db.get_system_state()
    language_options = ['English', 'Chinese']
    current_language = state.get('output_language', 'English')
    if current_language not in language_options:
        current_language = 'English'
    if PUBLIC_DEMO_MODE:
        debug_mode = False
        st.session_state.debug_prompt_toggle = False
    else:
        debug_mode = st.sidebar.toggle('Debug Prompt View', value=bool(st.session_state.get('debug_prompt_toggle', False)), key='debug_prompt_toggle')
    _render_settings(debug_mode, current_language, stage=state.get('stage', 'upload'), player_profile=db.get_player_profile())
    debug_mode = bool(st.session_state.get('debug_prompt_toggle', debug_mode))
    if PUBLIC_DEMO_MODE:
        debug_mode = False
    selected_language = st.session_state.get('output_language_select', current_language)
    if hasattr(agent, 'set_debug_mode'):
        agent.set_debug_mode(debug_mode)
    else:
        setattr(agent, 'debug_mode', bool(debug_mode))
    if selected_language != state.get('output_language', 'English'):
        db.update_system_state({'output_language': selected_language})
        state = db.get_system_state()
    stage = state['stage']
    _render_story_management_actions(stage, db, vector)
    _render_story_notice()
    if stage == 'session' and not st.session_state.messages:
        restored_messages = _load_messages_from_db(db)
        st.session_state.messages = restored_messages
        if not restored_messages and state.get('current_scene_id') and state.get('current_plot_id'):
            budget_ok, budget_message = _reserve_public_demo_turn()
            if budget_ok:
                try:
                    initial_result = agent.generate_initial_response()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception('Initial response generation failed')
                    initial_result = {
                        'response': _demo_text('fallback_opening') if PUBLIC_DEMO_MODE else f'Initial response failed: {exc}',
                        'retrieved_docs': [],
                        'dice_result': None,
                        'skill_check_result': None,
                        'debug_prompts': [],
                    }
            else:
                initial_result = {
                    'response': budget_message,
                    'retrieved_docs': [],
                    'dice_result': None,
                    'skill_check_result': None,
                    'debug_prompts': [],
                }
            st.session_state.last_retrieved = initial_result.get('retrieved_docs', [])
            st.session_state.messages.append(
                {
                    'user': '',
                    'agent': initial_result.get('response', ''),
                    'dice': initial_result.get('dice_result'),
                    'skill_check': initial_result.get('skill_check_result'),
                    'debug_prompts': initial_result.get('debug_prompts', []),
                }
            )
            st.rerun()

    if stage != 'session':
        _render_status_line(state, db)

    if stage == 'upload':
        upload_panel = st.container()
        uploaded = None
        story_source = 'official'
        with upload_panel:
            if PUBLIC_DEMO_MODE:
                _render_stage_tracker('upload')
                story_source_options = [
                    _demo_text('official_demo_label', 'Demo Story'),
                    _demo_text('upload_story_label', 'Upload Markdown'),
                ]
                story_source_label = st.radio(
                    _demo_text('story_source_label', 'Choose a story'),
                    options=story_source_options,
                    horizontal=True,
                    key='public_story_source',
                )
                story_source = 'upload' if story_source_label == story_source_options[1] else 'official'
                _render_public_demo_entry(story_source)
                unsupported_type = False
                upload_too_large = False
                if story_source == 'upload':
                    upload_nonce = int(st.session_state.get('script_upload_nonce', 0))
                    uploaded = st.file_uploader(
                        'Upload script (.md, .markdown)',
                        type=['md', 'markdown'],
                        key=f'public_script_upload_input_{upload_nonce}',
                    )
                    st.caption(_demo_text('upload_story_note', 'Bring your own Markdown scenario.'))
                    if uploaded:
                        uploaded_size = int(getattr(uploaded, 'size', 0) or len(uploaded.getvalue()))
                        if uploaded_size > PUBLIC_DEMO_MAX_UPLOAD_BYTES:
                            upload_too_large = True
                            st.error(
                                f'For the public demo, Markdown uploads must be under '
                                f'{PUBLIC_DEMO_MAX_UPLOAD_BYTES // 1000} KB.'
                            )
                        try:
                            source_type = detect_source_type(uploaded.name, getattr(uploaded, 'type', None))
                            st.caption(f"Detected source type: {source_type}.")
                        except ValueError:
                            unsupported_type = True
                            st.error('Unsupported file type. Please upload a Markdown file.')
                    parse_clicked = bool(
                        uploaded
                        and not upload_too_large
                        and st.button(_demo_text('upload_story_button', 'Prepare Uploaded Story'), use_container_width=True)
                    )
                else:
                    parse_clicked = st.button(_demo_text('entry_button', 'Begin Demo'), use_container_width=True)
            else:
                _render_section_header('Upload Script', 'Step 1')
                upload_nonce = int(st.session_state.get('script_upload_nonce', 0))
                uploaded = st.file_uploader(
                    'Upload script (.md, .markdown)',
                    type=['md', 'markdown'],
                    key=f'script_upload_input_{upload_nonce}',
                )
                unsupported_type = False
                if uploaded:
                    try:
                        source_type = detect_source_type(uploaded.name, getattr(uploaded, 'type', None))
                        st.caption(f"Detected source type: {source_type}.")
                    except ValueError:
                        unsupported_type = True
                        st.error('Unsupported file type. Please upload a Markdown file.')
                parse_clicked = bool(uploaded and st.button('Parse Script'))

        if parse_clicked:
            upload_panel.empty()
            parse_loading = st.empty()
            try:
                _render_loading_state(parse_loading, 'Parsing the script...', centered=True)
                if unsupported_type:
                    parse_loading.empty()
                    st.error('Unsupported file type. Please upload a Markdown file.')
                    st.stop()

                if PUBLIC_DEMO_MODE:
                    if story_source == 'upload':
                        document = read_uploaded_document(
                            uploaded.name,
                            uploaded.getvalue(),
                            mime_type=getattr(uploaded, 'type', None),
                        )
                        if not document.text.strip():
                            parse_loading.empty()
                            st.error('No readable Markdown content found in the uploaded file.')
                            st.stop()
                        _render_loading_state(parse_loading, 'Organizing scenes...', centered=True)
                        budget_ok, budget_message = _reserve_public_demo_turn()
                        if budget_ok:
                            try:
                                bundle = parse_script_bundle(source_document=document)
                            except Exception as exc:  # noqa: BLE001
                                LOGGER.warning('Uploaded script LLM parse failed; using local markdown fallback: %s', exc)
                                bundle = _parse_markdown_script_locally(document.text, document.source_file_name)
                        else:
                            LOGGER.info('Public demo upload parse used local fallback: %s', budget_message)
                            bundle = _parse_markdown_script_locally(document.text, document.source_file_name)
                    else:
                        document = None
                        bundle = _load_demo_script_bundle(allow_llm_parse=False)
                else:
                    document = read_uploaded_document(
                        uploaded.name,
                        uploaded.getvalue(),
                        mime_type=getattr(uploaded, 'type', None),
                    )
                    if not document.text.strip():
                        parse_loading.empty()
                        st.error('No readable Markdown content found in the uploaded file.')
                        st.stop()

                    _render_loading_state(parse_loading, 'Organizing scenes...', centered=True)
                    bundle = parse_script_bundle(source_document=document)

                _render_loading_state(parse_loading, 'Preparing the world...', centered=True)
                scene_count, knowledge_count = _install_script_bundle(db, vector, bundle)
                state = db.get_system_state()

                parse_loading.empty()
                if scene_count and state.get('current_scene_id') and state.get('current_plot_id'):
                    st.success(
                        f"Script parsed and stored. scenes={scene_count}, knowledge={knowledge_count}"
                    )
                elif scene_count:
                    st.warning(
                        f"Script parsed, but no playable plot was extracted. scenes={scene_count}, knowledge={knowledge_count}"
                    )
                else:
                    st.warning(
                        f"Script parsed, but no playable scene was extracted. knowledge={knowledge_count}"
                    )
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                parse_loading.empty()
                st.error(f'Parse failed: {exc}')
                st.stop()

    elif stage == 'parse':
        scenes = db.list_scenes()
        plot_count = sum(len(scene.get('plots', [])) for scene in scenes)
        est_minutes = plot_count * 10
        source_meta_raw = db.get_summary('parse_source_meta')
        script_summary = db.get_summary('script')
        try:
            source_meta = json.loads(source_meta_raw) if source_meta_raw else {}
        except Exception:
            source_meta = {}

        if PUBLIC_DEMO_MODE and not debug_mode:
            _render_stage_tracker('parse')
            _render_public_parse_summary(scenes, plot_count, est_minutes)
        elif debug_mode:
            _render_section_header('Review Parse', 'Step 2')
            if source_meta:
                st.markdown('#### Source Metadata')
                st.write(source_meta)
            if script_summary:
                st.markdown('#### Script Summary')
                st.write(script_summary)

            for scene in scenes:
                st.write(
                    {
                        'scene_id': scene['scene_id'],
                        'scene_name': scene.get('scene_name', ''),
                        'scene_goal': scene['scene_goal'],
                        'scene_description': (scene.get('scene_description', '') or '')[:120],
                        'plots': len(scene.get('plots', [])),
                    }
                )

            knowledge_items = db.list_knowledge()
            if knowledge_items:
                counts: dict[str, int] = {}
                for item in knowledge_items:
                    t = item.get('knowledge_type', 'other')
                    counts[t] = counts.get(t, 0) + 1
                st.markdown('#### Knowledge Overview')
                st.write(counts)
        else:
            _render_section_header('Review Parse', 'Step 2')
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric('Scenes', len(scenes))
            with m2:
                st.metric('Plots', plot_count)
            with m3:
                st.metric('Est. Minutes', est_minutes)
            st.caption(f'estimated play time {est_minutes} minutes')

        continue_label = _demo_text('continue_button', 'Create Character') if PUBLIC_DEMO_MODE else 'Continue'
        if st.button(continue_label, use_container_width=PUBLIC_DEMO_MODE):
            db.update_system_state({'stage': 'character'})
            st.rerun()

    elif stage == 'character':
        if PUBLIC_DEMO_MODE:
            _render_stage_tracker('character')
            _render_section_header(
                _demo_text('continue_button', 'Create Character'),
                'Step 3',
                'Choose a ready-made build, add a name and background, then enter the case.',
            )
        else:
            _render_section_header('Character', 'Step 3')

        if 'coc_builds' not in st.session_state:
            st.session_state.coc_builds = _generate_coc_builds()
        if 'character_stats_line' not in st.session_state:
            st.session_state.character_stats_line = st.session_state.coc_builds[0]['line']
        if 'selected_archetype_name' not in st.session_state:
            st.session_state.selected_archetype_name = st.session_state.coc_builds[0]['archetype']
        if 'occupation_alloc_text' not in st.session_state:
            st.session_state.occupation_alloc_text = ''
        if 'interest_alloc_text' not in st.session_state:
            st.session_state.interest_alloc_text = ''

        archetype_to_build = {str(b['archetype']): b for b in st.session_state.coc_builds}
        archetype_names = list(archetype_to_build.keys())
        selected_archetype_name = st.session_state.selected_archetype_name
        if selected_archetype_name not in archetype_to_build:
            selected_archetype_name = archetype_names[0]
            st.session_state.selected_archetype_name = selected_archetype_name
        selected_build = archetype_to_build[selected_archetype_name]

        if PUBLIC_DEMO_MODE:
            presets = _demo_character_presets(st.session_state.coc_builds)
            preset_names = [str(preset.get('archetype', 'Investigator')) for preset in presets]
            if not preset_names:
                st.error('No playable character presets are available.')
                st.stop()
            if st.session_state.get('public_character_choice') not in preset_names:
                st.session_state['public_character_choice'] = preset_names[0]

            selected_preset_name = st.radio(
                'Choose investigator',
                options=preset_names,
                horizontal=True,
                key='public_character_choice',
            )
            selected_preset = next(
                (preset for preset in presets if str(preset.get('archetype', 'Investigator')) == selected_preset_name),
                presets[0],
            )

            if st.session_state.get('last_public_character_choice') != selected_preset_name:
                st.session_state['last_public_character_choice'] = selected_preset_name
                st.session_state['character_name_input'] = str(selected_preset.get('default_name', selected_preset_name))
                st.session_state['character_background_input'] = str(selected_preset.get('background', ''))
                st.session_state['character_stats_line'] = str(selected_preset.get('line', ''))
                st.session_state['occupation_alloc_text'] = '\n'.join(
                    str(item) for item in selected_preset.get('occupation_suggested', [])
                )
                st.session_state['interest_alloc_text'] = '\n'.join(
                    str(item) for item in selected_preset.get('interest_suggested', [])
                )

            _render_investigator_card(selected_preset)
            id_col1, id_col2 = st.columns([0.8, 1.4])
            with id_col1:
                name = st.text_input('Name', key='character_name_input')
            with id_col2:
                background = st.text_area('Background', key='character_background_input', height=108)

            with st.expander(_demo_text('advanced_sheet_label', 'Customize character sheet'), expanded=False):
                stats = st.text_input('Characteristics', key='character_stats_line')
                occ_default = '\n'.join(_ensure_default_skill_lines(list(selected_preset.get('occupation_suggested', []))))
                interest_default = '\n'.join(str(item) for item in selected_preset.get('interest_suggested', []))
                if not st.session_state.occupation_alloc_text:
                    st.session_state.occupation_alloc_text = occ_default
                if not st.session_state.interest_alloc_text:
                    st.session_state.interest_alloc_text = interest_default
                occupation_alloc = st.text_area(
                    'Occupation Skills',
                    key='occupation_alloc_text',
                    height=145,
                )
                interest_alloc = st.text_area(
                    'Personal Interest Skills',
                    key='interest_alloc_text',
                    height=115,
                )

            save_label = _demo_text('save_character_button', 'Enter Story')
            if st.button(save_label, use_container_width=True):
                if not name.strip():
                    st.error('Name is required.')
                    st.stop()
                if not background.strip():
                    st.error('Background is required.')
                    st.stop()

                profile = _profile_from_preset(
                    selected_preset,
                    name,
                    background,
                    st.session_state.character_stats_line,
                    st.session_state.occupation_alloc_text,
                    st.session_state.interest_alloc_text,
                )
                if profile is None:
                    st.error('Please provide a valid CoC characteristic line before saving.')
                    st.stop()
                db.save_player_profile(profile)
                db.update_system_state({'stage': 'session'})
                st.rerun()
            return

        _render_section_header('Step 1: Character Identity', 'Required')
        id_col1, id_col2 = st.columns([1, 1.2])
        with id_col1:
            name = st.text_input('Name', key='character_name_input')
        with id_col2:
            background = st.text_area('Background', key='character_background_input', height=120)

        _render_section_header('Step 2: Characteristic Build', 'Step 2')
        st.caption('STR/CON/DEX/APP/POW: 15-90. SIZ/INT/EDU: 40-90.')

        build_options = [f"{b['archetype']} | {b['line']}" for b in st.session_state.coc_builds]
        chosen_build_label = st.selectbox('Generated Builds (10)', options=build_options, key='build_pick_label')
        chosen_build_line = chosen_build_label.split(' | ', 1)[1]
        if st.button('Apply Build to Characteristics'):
            st.session_state.character_stats_line = chosen_build_line

        st.code('\n'.join([b['line'] for b in st.session_state.coc_builds]), language='text')

        stats = st.text_input('Characteristics', key='character_stats_line')
        parsed_stats = _parse_stats_line(stats)
        stats_valid = bool(parsed_stats and _validate_coc_stats(parsed_stats))
        if stats_valid:
            derived = _calc_derived(parsed_stats)
            d1, d2, d3, d4, d5 = st.columns(5)
            with d1:
                st.metric('HP', derived['HP'])
            with d2:
                st.metric('MP', derived['MP'])
            with d3:
                st.metric('SAN', derived['SAN'])
            with d4:
                st.metric('Occupation', derived['occupation_skill_points'])
            with d5:
                st.metric('Interest', derived['personal_interest_points'])
        else:
            st.warning(
                'Invalid characteristics. Use format: STR:65,CON:50,SIZ:60,DEX:80,APP:60,INT:75,POW:75,EDU:70'
            )

        _render_section_header('Step 3: Archetype', 'Step 3')
        selected_archetype_name = st.selectbox('Archetype', options=archetype_names, key='selected_archetype_name')
        selected_build = archetype_to_build[selected_archetype_name]

        _render_section_header('Step 4: Skills', 'Step 4')
        occ_default = '\n'.join(_ensure_default_skill_lines(list(selected_build['occupation_suggested'])))
        interest_default = '\n'.join(list(selected_build['interest_suggested']))
        if not st.session_state.occupation_alloc_text:
            st.session_state.occupation_alloc_text = occ_default
        if not st.session_state.interest_alloc_text:
            st.session_state.interest_alloc_text = interest_default
        if st.button('Use Suggested Skills for Selected Archetype'):
            st.session_state.occupation_alloc_text = occ_default
            st.session_state.interest_alloc_text = interest_default

        occupation_alloc = st.text_area(
            'Occupation Skills (one per line, e.g., Spot Hidden:60)',
            key='occupation_alloc_text',
            height=160,
        )
        interest_alloc = st.text_area(
            'Personal Interest Skills (one per line, e.g., Occult:40)',
            key='interest_alloc_text',
            height=130,
        )

        save_label = _demo_text('save_character_button', 'Enter Story') if PUBLIC_DEMO_MODE else 'Save Character'
        if st.button(save_label, use_container_width=PUBLIC_DEMO_MODE):
            if not name.strip():
                st.error('Name is required.')
                st.stop()
            if not background.strip():
                st.error('Background is required.')
                st.stop()

            parsed = _parse_stats_line(stats)
            if not parsed or not _validate_coc_stats(parsed):
                st.error('Please provide a valid CoC characteristic line before saving.')
                st.stop()

            derived = _calc_derived(parsed)
            profile = {
                'name': name.strip(),
                'background': background.strip(),
                'characteristics': parsed,
                'derived_attributes': {
                    'HP': derived['HP'],
                    'MP': derived['MP'],
                    'SAN': derived['SAN'],
                },
                'skill_points': {
                    'occupation': derived['occupation_skill_points'],
                    'personal_interest': derived['personal_interest_points'],
                },
                'selected_archetype': selected_archetype_name,
                'suggested_skill_allocations': {
                    'occupation': selected_build['occupation_suggested'],
                    'personal_interest': selected_build['interest_suggested'],
                },
                'chosen_skill_allocations': {
                    'occupation': [line.strip() for line in occupation_alloc.splitlines() if line.strip()],
                    'personal_interest': [line.strip() for line in interest_alloc.splitlines() if line.strip()],
                },
            }
            db.save_player_profile(profile)
            db.update_system_state({'stage': 'session'})
            st.rerun()

    else:
        player_turns = _player_turn_count(st.session_state.messages)
        turn_limit_reached = PUBLIC_DEMO_MODE and player_turns >= PUBLIC_DEMO_MAX_TURNS
        if PUBLIC_DEMO_MODE:
            _render_public_investigation_sidebar(db, state, player_turns)
            _render_session_banner(state, player_turns)

        for turn in st.session_state.messages:
            if PUBLIC_DEMO_MODE:
                if turn['user']:
                    _render_public_dialogue('user', turn['user'])
                _render_public_dialogue('keeper', turn['agent'], turn.get('dice'), turn.get('skill_check'))
            else:
                if turn['user']:
                    st.chat_message('user').write(turn['user'])
                st.chat_message('assistant').write(turn['agent'])
                if turn.get('dice'):
                    st.caption(f"Dice Roll Result: {turn['dice']}")
                if turn.get('skill_check'):
                    st.caption(f"Skill Check Result: {turn['skill_check']}")
            if debug_mode and turn.get('debug_prompts'):
                with st.expander('Debug Prompts', expanded=False):
                    for idx, item in enumerate(turn.get('debug_prompts', []), start=1):
                        label = f"{idx}. {item.get('name', 'prompt')}"
                        with st.expander(label, expanded=False):
                            st.code(item.get('prompt', ''), language='text')

        if turn_limit_reached:
            _render_demo_end_card(db, vector, state, player_turns)
            user_msg = None
        else:
            if PUBLIC_DEMO_MODE:
                _render_public_hint_controls(db, state)
            placeholder = _demo_text('chat_placeholder', 'What do you do next?') if PUBLIC_DEMO_MODE else 'Describe your action...'
            user_msg = st.chat_input(placeholder)
        if user_msg:
            st.session_state.pop('public_demo_hint_lines', None)
            if PUBLIC_DEMO_MODE:
                if len(user_msg) > PUBLIC_DEMO_MAX_INPUT_CHARS:
                    st.warning(f'Please keep each action under {PUBLIC_DEMO_MAX_INPUT_CHARS} characters.')
                    st.stop()
                last_turn_at = float(st.session_state.get('public_demo_last_turn_at', 0.0) or 0.0)
                cooldown_remaining = PUBLIC_DEMO_MIN_TURN_SECONDS - (time.monotonic() - last_turn_at)
                if cooldown_remaining > 0:
                    st.warning(f'Give the Keeper {cooldown_remaining:.0f} more seconds before the next action.')
                    st.stop()
                budget_ok, budget_message = _reserve_public_demo_turn()
                if not budget_ok:
                    st.warning(budget_message)
                    st.stop()
                st.session_state.public_demo_last_turn_at = time.monotonic()
            if PUBLIC_DEMO_MODE:
                _render_public_dialogue('user', user_msg)
                thinking_placeholder = st.empty()
            else:
                st.chat_message('user').write(user_msg)
                thinking_placeholder = st.chat_message('assistant').empty()
            thinking_texts = [
                'The Keeper is thinking...',
                'Weaving the next scene...',
                'Something is unfolding...',
            ]
            _render_loading_state(
                thinking_placeholder,
                thinking_texts[len(st.session_state.messages) % len(thinking_texts)],
            )
            try:
                result = agent.run_turn(user_msg)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception('Turn generation failed')
                result = {
                    'response': _demo_text('fallback_turn') if PUBLIC_DEMO_MODE else f'Turn generation failed: {exc}',
                    'retrieved_docs': [],
                    'dice_result': None,
                    'skill_check_result': None,
                    'debug_prompts': [],
                }
            thinking_placeholder.empty()
            st.session_state.last_retrieved = result.get('retrieved_docs', [])
            st.session_state.messages.append(
                {
                    'user': user_msg,
                    'agent': result.get('response', ''),
                    'dice': result.get('dice_result'),
                    'skill_check': result.get('skill_check_result'),
                    'debug_prompts': result.get('debug_prompts', []),
                }
            )
            st.rerun()
        if debug_mode and st.session_state.last_retrieved:
            with st.expander('Retrieved Knowledge', expanded=False):
                for doc in st.session_state.last_retrieved[:5]:
                    st.write(f"- {doc.get('metadata', {}).get('type')}: {doc.get('content')}")
