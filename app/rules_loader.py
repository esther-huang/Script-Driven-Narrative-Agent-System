from __future__ import annotations

import re
from pathlib import Path
from typing import Any


RULES_PATH = Path(__file__).resolve().parent.parent / 'database' / 'GameRules.md'


def load_game_rules_knowledge() -> list[dict[str, Any]]:
    if not RULES_PATH.exists():
        return []

    raw_text = RULES_PATH.read_text(encoding='utf-8', errors='replace').strip()
    if not raw_text:
        return []

    sections = _split_markdown_sections(raw_text)
    knowledge_items: list[dict[str, Any]] = []
    for idx, section in enumerate(sections, start=1):
        title = section['title'].strip() or f'D100 Investigation Rules {idx}'
        content = section['content'].strip()
        if not content:
            continue
        knowledge_items.append(
            {
                'knowledge_id': f'game_rule_{idx}',
                'knowledge_type': 'rule',
                'title': title,
                'content': content,
                'metadata': {
                    'source': 'database/GameRules.md',
                    'section_heading': title,
                    'category': 'rules',
                },
            }
        )
    return knowledge_items


def _split_markdown_sections(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    sections: list[dict[str, str]] = []
    current_title = 'D100 Investigation Rules'
    current_lines: list[str] = []

    for line in lines:
        if re.match(r'^\s{0,3}#{1,6}\s+', line):
            if current_lines:
                sections.append({'title': current_title, 'content': '\n'.join(current_lines).strip()})
                current_lines = []
            current_title = re.sub(r'^\s{0,3}#{1,6}\s+', '', line).strip() or current_title
            current_lines.append(line)
            continue
        current_lines.append(line)

    if current_lines:
        sections.append({'title': current_title, 'content': '\n'.join(current_lines).strip()})

    return [section for section in sections if section['content'].strip()]
