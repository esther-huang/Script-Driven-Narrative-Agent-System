from __future__ import annotations

import json
import json5
import random
import re
from typing import Any, TypedDict
import traceback
import logging

from langgraph.graph import END, StateGraph

from app.database import Database
from app.llm_client import call_llm
from app.rag import categorize_docs, generate_retrieval_queries
from app.vector_store import ChromaStore

logger = logging.getLogger(__name__)


def truncate_plot_raw_text(text: str) -> str:
    return text if len(text) <= 3000 else text[:3000]

RESPONSE_PROMPT_TEMPLATE = """# SYSTEM PROMPT

You are {agent_role}, running an interactive {game_system} narrative experience.

## Core Responsibilities
- Respond to the player by describing NPC reactions and changes in the scene, strictly based on the Current Plot Raw Text.
- Progress toward current Plot goals and ultimately the Scene goal.
- Maintain long-term narrative coherence.
- Respect world logic and past events.

## Style
- Tone: {tone_style}
- Perspective: {narrative_perspective}
- Length: {response_length}

## Constraints
- Use only information provided in Context.
- Do not fabricate missing knowledge.
- Avoid meta commentary.
- Treat Current Plot Raw Text as authoritative script context.
- Final response language: {output_language}
- Write the response entirely in {output_language}.

## Player Interaction Rules

- The player controls the PC. Do NOT speak for the PC, extend their dialogue, or describe their internal thoughts.
- After the player acts or speaks, advance the scene with NPC dialogue, NPC action, or environmental consequences.
- If all meaningful content in the Current Plot Raw Text has been explored, gently suggest moving to the next scene or plot in parentheses ():
  - For example: 
    - "There are no further clues in this location. You may consider investigating other areas."
    - "Your meeting with John has come to an end. If you have nothing else to do, you may rest and begin a new day."
- When a dice roll or player choice is required **by the script or situation**, present it clearly and naturally in parentheses (), without breaking immersion.
    - For skill checks, briefly describe the situation and then indicate the required check, for example:
      "As you walk past, an elderly lady studies your appearance with clear judgment in her eyes, as if she values status and presentation. (Make an APP or Credit Rating check.)"
    - For branching choices, present the options in the narrative, then prompt the player to choose, for Example:
      "You step out of the house. Based on what you know, you could:
      1. Ask around the neighborhood.
      2. Search the graveyard.
         You pause, considering your next move. (Choose one option.)"
    - Maintain an immersive, in-world (diegetic) tone. Present checks and choices as a natural part of the scene, not as system instructions.
  **Do NOT present choices or branches every turn**. Overusing explicit choices can break immersion and reduce the player’s sense of freedom. Only include them when they are required by script or or really necessary for progression.
- If the player's action significantly deviates from the main storyline, guide them back naturally through NPC dialogue, NPC actions, or environmental consequences.
- Do NOT ask rhetorical or leading questions about the PC’s beliefs, thoughts, or motivations.
- Do NOT ask questions to the player or suggest what they should do next.

## Game Mechanics

- If a dice result is provided, the narrative should reflect the skill check outcome, within the limits of the Current Plot Raw Text.

- Interpretation rules:
  - **Extreme Success**:
    - Provide the most precise and complete understanding of existing details
    - Highlight subtle, easily missed elements already present in the Current Plot Raw Text
    - Do NOT introduce new objects, structures, or hidden elements not grounded in the text
  - **Hard Success**:
    - Clearly reveal important details present in the Current Plot Raw Text
    - May include minor additional interpretation, but no new facts
  - **Regular Success**:
    - Reveal expected information directly supported by the Current Plot Raw Text
  - **Fail**:
    - Withhold key information or provide vague, incomplete observations
  - **Fumble (Worst Fail)**:
    - Introduce negative consequences, risks, or misleading interpretations, but do NOT fabricate new world facts

- Information constraints (STRICT):
  - The Current Plot Raw Text is the primary and authoritative source
  - Retrieved clues are supplementary and may contain information intended for other scenes. 
  - Do NOT introduce clue information unless it is clearly supported by the Current Plot Raw Text
  - If no additional relevant information exists in the Current Plot Raw Text:
    - Do NOT invent new details
    - Do NOT create new objects, locations, or hidden elements (e.g., secret compartments, traces, entities)
    - Instead, refine, confirm, or reinterpret existing visible details only

- Do not mention system processing, hidden prompts, or any tooling in the narrative.

## Story Ending Narrative

A story ending is defined as a point where:

- The central mystery or main conflict has been clearly resolved
- The investigator has reached a final understanding, decision, or outcome
- The investigation has effectively concluded (no further meaningful actions remain)

If the current Plot Raw Text represents such a final resolution:

Note: Do NOT assume the story has reached the ending just because the Plot Raw Text contains ending content.
You must verify that the narrative has actually progressed to that point.

Step 1 — Check progression (DO NOT skip):

Before ending the story, determine whether the narrative has actually reached this final point.

Only consider the story ready to end if ALL of the following are true:

- The core mystery or conflict has already been resolved in the recent conversation or memory
- The player has reached or triggered the final situation described in the Plot Raw Text
- The investigation has naturally concluded (i.e., no meaningful next actions remain)
- The current moment aligns with the ending and does not feel premature

Step 2 — Branch decision:

IF the above conditions are NOT satisfied:

- Continue the scene normally
- Progress toward resolving the core mystery or conflict
- Do NOT summarize or conclude the story prematurely

IF the above conditions ARE satisfied:

- You MUST end the story clearly and conclusively
- Do NOT introduce new scenes, plots, or choices
- Do NOT suggest further actions
- Do NOT continue the story beyond this point

Step 3 — Ending requirements:

When ending the story, your response should:

1. Clearly resolve the main mystery or conflict
2. Describe the final outcome for key characters
3. Show that the investigation has concluded
4. Provide a sense of closure (emotional or narrative)
5. Optionally include a brief epilogue

The ending should feel final, complete, and satisfying.

---

# INSTRUCTION

Generate the next narrative response.

Requirements:
- Respond to the player by describing NPC reactions and changes in the scene, strictly based on the Current Plot Raw Text. 
- Move story toward Plot Goal.
- Maintain consistency with Memory.
- Preserve immersion.

---

# USER INPUT

Player Input:
{user_input}

---

# SCRIPT STATE

Scene ID: {scene_id}  
Scene Name: {scene_name}
Plot ID: {plot_id}
Plot Name: {plot_name}

Scene Goal:
{current_scene_goal}

Scene Description:
{current_scene_description}

Plot Goal:
{current_plot_goal}

Current Plot Raw Text:
This is the most important reference. It is from the original script. Please read it carefully and follow the instructions provided.
{current_plot_raw_text}

Global Story Summary: 
This provides an overview of the entire story. You should progress the narrative according to this main storyline.
{script_summary}

---

# MEMORY

Previous Plot Summary:
{previous_plot_summary}

Current Scene Summary:
{current_scene_summary}

Long-Term Memory:
{long_term_memory}

Recent Conversation:
{recent_conversation}

---

# RETRIEVED KNOWLEDGE

NPC:
{npc_related_info}

Player:
{player_related_info}

Setting:
{setting}

Clue:
- Retrieved dynamically during the story
- Do NOT reveal spoilers beyond the current context
- Treat as supplementary information only
- Prioritize the Current Plot Raw Text over clues in most cases

{clue}

---

# DICE CHECK RESULT

Dice Result:
{dice_result}

Skill Check Result:
{skill_check_result}

---
# OTHER SCENE INFORMATION

If the current scene is sufficiently explored (i.e., no meaningful information remains in the Current Plot Raw Text), you may guide the player naturally toward the next scene. This transition should feel seamless and in-world (e.g., through narration, NPC actions, or environmental cues)

Next Scene Name:
{next_scene_name}

Next Scene Description:
{next_scene_description}

Unvisited Scene Name:
{unvisited_scene_name}

Visited Scene Name:
{visited_scene_name}

---
# OTHER PLOT INFORMATION

These represent the possible plot directions or next progressions within the current scene. Provide subtle hints when it helps guide progression.

Unvisited Plot Name:
{unvisited_plot_name}

"""

ROLL_CHECK_PROMPT_TEMPLATE = """# SYSTEM PROMPT

You are a d100 investigation rules assistant.
Decide whether the player's latest action requires a deterministic dice or skill check.

Return strict JSON only:
{{
  "need_check": true or false,
  "skill": "skill name or SAN or empty string",
  "reason": "brief English reason",
  "dice_type": "1d100 or empty string"
}}

Rules:
- Use English only.
- Trigger a skill check only when there is a clear and explicit intent:
    - The player directly attempts an action that involves uncertainty, risk, investigation, combat, or sanity pressure, or
    - The Keeper has previously introduced a situation that clearly invites a check, and the player responds to it.
- Do NOT trigger a skill check if the player has not expressed a clear action or intent.
  - Passive, vague, or observational actions without a specific goal should not automatically result in a check.
- A skill check should feel natural within the narrative:
  - It may be initiated by the player’s action, or
  - Introduced by the Keeper through the scene, but only becomes active when the player engages with it.
- Do NOT introduce checks abruptly without narrative or player-driven justification.
- If no meaningful uncertainty or risk exists, set `need_check` to false.
- For d100 investigation skill checks, use 1d100.
- Prefer skill names from the player skill list when possible.

Examples when checks are needed:
- Searching for hidden evidence -> Spot Hidden
- Reading strange documents -> Library Use
- Staying calm before horror -> SAN
- Forcing a stuck door -> STR
- Dodging an attack -> Dodge

Tool usage format reference:
TOOL_CALL: roll_dice
{{
  "dice_type": "1d100",
  "reason": "Spot Hidden check"
}}

Player Input:
{user_input}

Scene ID: {scene_id}
Scene Name: {scene_name}
Plot ID: {plot_id}
Plot Name: {plot_name}
Scene Goal:
{current_scene_goal}

Scene Description:
{current_scene_description}

Plot Goal:
{current_plot_goal}

Current Plot Raw Text:
{current_plot_raw_text}

Previous Plot Summary:
{previous_plot_summary}

Current Scene Summary:
{current_scene_summary}

Recent Conversation:
{recent_conversation}

Player:
{player_related_info}

Full Player Skill List:
{player_skill_list}
"""

BRANCH_TRANSITION_PROMPT_TEMPLATE = """Your task is to decide whether to:
- stay in the current plot, or
- move to another plot or scene

---

### Core Principle

- Scene transitions represent narrative progression. 
- A change in location usually indicates a scene transition, unless clearly part of the same scene.
- The decision must be grounded in the Current Plot Raw Text and Player Explicit Decision.

---

### Decision Rules

Switch (move to another plot or scene) if ANY of the following is true:

1. Explicit user intent

- The user clearly expresses intent to move to another place or activity
- This ALWAYS triggers a switch, regardless of the current scene or plot state
- If the user previously expressed such intent but the system did not switch, and the current scene does not match that intent, you MUST switch now.
- Example: "I leave and go to the library"

2. The current scene is sufficiently explored

A scene is considered sufficiently explored when:

- Most meaningful information in the Current Plot Raw Text has already been extracted  
  OR  
- The player has attempted most obvious actions

This judgment of sufficiently explored is based on the Current Plot Raw Text, NOT on newly generated narrative content.

Examples of "sufficiently explored"

- The Current Plot Raw Text no longer provides new actionable information
- The player has completed the main interactions (e.g., asking key questions, checking obvious locations)
- The next scene represents a natural progression of the story
- The transition does NOT require player choice
- The next scene may occur in the SAME location or nearby

---

### Conflict Resolution (CRITICAL)

- If there is ANY conflict between:
  - player intent or movement
  - sufficiently explored

→ ALWAYS follow player intent (SWITCH)

--

### Target Selection Rule

When selecting the next scene:

- Choose the scene that best matches the script progression and the player's intended location or activity.
- If multiple candidates exist:
  - Prefer unvisited scenes
  - If still tied, choose the **earlier** one in the list. Later scenes typically depend on earlier ones and should not be selected first.

---

### Default Behavior (IMPORTANT)

- If you are uncertain whether the scene is complete → SWITCH to the next scene
- If the user moves to a different location → SWITCH
- Only STAY if BOTH conditions are clearly true:
  - There is remaining meaningful information in the Current Plot Raw Text
  - The player is actively interacting with that remaining content

---

### Output (STRICT JSON)

{branch_decision_json}

---

### Notes

- Scene transitions are narrative (temporal or logical), not purely spatial
- Revisiting scenes/plots is allowed but uncommon

---

Inputs:

Recent Conversation (last 3 rounds, from previous to latest):
This is the MOST IMPORTANT signal of the player's intent.

- If the player clearly expresses a desire to move to another location or start a different activity → you MUST switch.
- Player intent ALWAYS takes priority over all other rules.
- Player location persists across turns:
  - If the player previously moved to a new location, you must assume they are still there unless explicitly stated otherwise.
  - If the player’s current location does NOT match the current scene → you MUST switch.
- Even if the system failed to switch previously, you MUST correct it now.

{global_recent_conversation}

Long-Term Memory:
Provides background context and previously discovered information.
Use it to maintain consistency, but DO NOT use it to override the player’s current intent.

{long_term_memory}

Current Plot Raw Text:
This is the MOST IMPORTANT reference for determining whether the current plot is complete.

- If the player has already explored the main actionable content in this plot → you should switch.
- If no new meaningful information or actions remain → you should switch.
- This judgment must be based ONLY on the raw plot content, NOT on newly generated narrative. 
  - If the narrative suggests continuing exploration, BUT the raw text contains no new actionable information → the plot is already fully explored → you should switch.

{current_plot_raw_text}

---

Global Story Summary:
This defines the main storyline. 
Base your scene and plot transition decisions on this summary to ensure the narrative progresses smoothly and in the intended direction. 
Do not switch to content that significantly deviates from the main progression.

{script_summary}

---
Current Scene:
{current_scene}

Unvisited Scenes:
{unvisited_scenes}

Visited Scenes:
{visited_scenes}

---
Current Plot:
{current_plot}

Unvisited Plots (within current scene):
{unvisited_plots}

Visited Plots (within current scene):
{visited_plots}
"""

LONG_TERM_MEMORY_UPDATE_PROMPT_TEMPLATE = """The conversation has been updated. Update the long-term memory by summarizing the current story state.

Goal:
Produce a concise summary that supports reasoning, branch decisions, and response generation, while preserving all information relevant to story progression. 

Instructions:
- Merge the previous long_term_memory with the recent conversation.
- Produce a concise summary, but preserve all information that is relevant to story progression and decision-making.
- Do not omit important details in order to shorten the summary.
- Explicitly track:
  1. User actions (what the player has done)
  2. Confirmed information (facts learned from the world or NPCs)
  3. Clues (potentially important hints or leads)
  4. User hypotheses (player assumptions, guesses, or interpretations)

Requirements:
- Maximum 4 sentences.
- Use clear and explicit language.
- Avoid vague phrases (e.g., "some exploration", "various actions").
- Preserve important entities (names, locations, objects).
- Do NOT introduce new information not present in the conversation.

Input:

Current long-term memory:
{current_long_term_memory}

Recent conversation (last 3 rounds):
{recent_conversation}

Output:
Updated long-term memory (<= 4 sentences, cumulative and state-focused).
"""

KP_OPENING_MARKER = '[KP_OPENING]'


class NarrativeState(TypedDict, total=False):
    scene_id: str
    scene_name: str
    plot_id: str
    plot_name: str
    player_profile: dict[str, Any]
    conversation_history: list[dict[str, Any]]
    global_conversation_history: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    latest_user_input: str
    prompt: str
    roll_check_prompt: str
    retrieval_queries: list[str]
    response: str
    dice_result: str | None
    skill_check_result: str | None
    need_check: bool
    check_skill: str
    check_reason: str
    dice_type: str
    scene_goal: str
    scene_description: str
    plot_goal: str
    current_plot_raw_text: str
    setting: str
    clue: str
    previous_plot_summary: str
    current_scene_summary: str
    long_term_memory: str
    script_summary: str
    visited_scenes: list[str]
    visited_plots: list[str]
    transition_switch: bool
    transition_target_plot_id: str
    previous_scene_id: str
    previous_plot_id: str
    previous_scene_name: str
    previous_plot_name: str
    previous_scene_goal: str
    previous_plot_goal: str
    output_language: str
    debug_prompts: list[dict[str, str]]


class NarrativeAgent:
    def __init__(self, db: Database, vector_store: ChromaStore) -> None:
        self.db = db
        self.vector_store = vector_store
        self.debug_mode = False
        self.latest_debug_prompts: list[dict[str, str]] = []
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(NarrativeState)
        workflow.add_node('build_prompt', self.build_prompt)
        workflow.add_node('retrieve_memory', self.retrieve_memory)
        workflow.add_node('decide_branch_transition', self.decide_branch_transition)
        workflow.add_node('generate_retrieval_queries', self.generate_retrieval_queries)
        workflow.add_node('vector_retrieve', self.vector_retrieve)
        workflow.add_node('construct_context', self.construct_context)
        workflow.add_node('check_whether_roll_dice', self.check_whether_roll_dice)
        workflow.add_node('roll_dice', self.roll_dice)
        workflow.add_node('generate_response', self.generate_response)
        workflow.add_node('write_memory', self.write_memory)
        workflow.add_node('finalize_turn_state', self.finalize_turn_state)

        workflow.set_entry_point('build_prompt')
        workflow.add_edge('build_prompt', 'retrieve_memory')
        workflow.add_edge('retrieve_memory', 'decide_branch_transition')
        workflow.add_edge('decide_branch_transition', 'generate_retrieval_queries')
        workflow.add_edge('generate_retrieval_queries', 'vector_retrieve')
        workflow.add_edge('vector_retrieve', 'construct_context')
        workflow.add_edge('construct_context', 'check_whether_roll_dice')
        workflow.add_conditional_edges(
            'check_whether_roll_dice',
            self._route_after_roll_check,
            {
                'roll_dice': 'roll_dice',
                'generate_response': 'generate_response',
            },
        )
        workflow.add_edge('roll_dice', 'generate_response')
        workflow.add_edge('generate_response', 'write_memory')
        workflow.add_edge('write_memory', 'finalize_turn_state')
        workflow.add_edge('finalize_turn_state', END)
        return workflow.compile()

    def run_turn(self, user_input: str) -> dict[str, Any]:
        self.latest_debug_prompts = []
        system_state = self.db.get_system_state()
        state: NarrativeState = {
            'scene_id': system_state.get('current_scene_id', ''),
            'plot_id': system_state.get('current_plot_id', ''),
            'output_language': system_state.get('output_language', 'English'),
            'player_profile': self.db.get_player_profile(),
            'latest_user_input': user_input,
            'conversation_history': [],
            'global_conversation_history': [],
            'retrieved_docs': [],
            'scene_name': '',
            'plot_name': '',
            'current_plot_raw_text': '',
            'setting': 'None',
            'clue': 'None',
            'long_term_memory': '',
            'script_summary': '',
            'dice_result': None,
            'skill_check_result': None,
            'need_check': False,
            'check_skill': '',
            'check_reason': '',
            'dice_type': '',
            'visited_scenes': [],
            'visited_plots': [],
            'transition_switch': False,
            'transition_target_plot_id': '',
            'debug_prompts': [],
        }
        result = self.graph.invoke(state)
        self.latest_debug_prompts = result.get('debug_prompts', [])
        return result

    def generate_initial_response(self) -> dict[str, Any]:
        self.latest_debug_prompts = []
        system_state = self.db.get_system_state()
        scene_id = str(system_state.get('current_scene_id', '') or '')
        plot_id = str(system_state.get('current_plot_id', '') or '')
        if not scene_id or not plot_id:
            return {
                'scene_id': scene_id,
                'plot_id': plot_id,
                'response': '',
                'retrieved_docs': [],
                'dice_result': None,
                'skill_check_result': None,
                'debug_prompts': [],
            }
        if self.db.has_global_opening(KP_OPENING_MARKER):
            row = self.db.conn.execute(
                "SELECT agent FROM memory WHERE user = ? ORDER BY id ASC LIMIT 1",
                (KP_OPENING_MARKER,),
            ).fetchone()
            visit_state: NarrativeState = {'visited_scenes': [], 'visited_plots': []}
            self._load_visited_state(visit_state)
            self._mark_visited(visit_state, scene_id, plot_id)
            self._save_visited_state(visit_state)
            self.latest_debug_prompts = []
            return {
                'scene_id': scene_id,
                'plot_id': plot_id,
                'response': str(row['agent']) if row else '',
                'retrieved_docs': [],
                'dice_result': None,
                'skill_check_result': None,
                'debug_prompts': [],
            }
        opening = self._generate_scene_opening(scene_id, plot_id)
        self.db.append_memory(scene_id, plot_id, KP_OPENING_MARKER, opening)
        visit_state: NarrativeState = {'visited_scenes': [], 'visited_plots': []}
        self._load_visited_state(visit_state)
        self._mark_visited(visit_state, scene_id, plot_id)
        self._save_visited_state(visit_state)
        return {
            'scene_id': scene_id,
            'plot_id': plot_id,
            'response': opening,
            'retrieved_docs': [],
            'dice_result': None,
            'skill_check_result': None,
            'debug_prompts': list(self.latest_debug_prompts),
        }

    def set_debug_mode(self, enabled: bool) -> None:
        self.debug_mode = bool(enabled)

    def _record_prompt(self, state: NarrativeState | None, name: str, prompt: str) -> None:
        if not self.debug_mode:
            return
        entry = {'name': name, 'prompt': prompt}
        if state is not None:
            state.setdefault('debug_prompts', []).append(entry)
        self.latest_debug_prompts.append(entry)

    def _llm_call(self, prompt: str, *, step_name: str) -> str:
        logger.info("LLM step=%s prompt_length=%s", step_name, len(prompt))
        return call_llm(prompt, step_name=step_name).strip()

    def _get_output_language(self, state: NarrativeState | None = None) -> str:
        if state and state.get('output_language'):
            return str(state['output_language'])
        return str(self.db.get_system_state().get('output_language', 'English'))

    def _route_after_roll_check(self, state: NarrativeState) -> str:
        return 'roll_dice' if state.get('need_check') else 'generate_response'

    def _format_player_skill_list(self, state: NarrativeState) -> str:
        profile = state.get('player_profile', {}) or {}
        lines: list[str] = []
        for group_name in ('occupation', 'personal_interest'):
            chosen = profile.get('chosen_skill_allocations', {}).get(group_name, [])
            if chosen:
                lines.append(f"{group_name.title()} Skills:")
                lines.extend(str(item) for item in chosen)
        characteristics = profile.get('characteristics', {})
        if characteristics:
            lines.append('Characteristics:')
            lines.extend(f"{k}:{v}" for k, v in characteristics.items())
        derived = profile.get('derived_attributes', {})
        if derived:
            lines.append('Derived Attributes:')
            lines.extend(f"{k}:{v}" for k, v in derived.items())
        return '\n'.join(lines) or 'No player skills available.'

    def _format_recent_conversation(self, history: list[dict[str, Any]] | None, rounds: int = 3) -> str:
        turns = (history or [])[-rounds:]
        if not turns:
            return 'None'
        lines: list[str] = []
        for turn in turns:
            user_text = str(turn.get('user', '')).strip() or '(no player input)'
            keeper_text = str(turn.get('agent', '')).strip() or '(no keeper response)'
            if user_text == KP_OPENING_MARKER:
                lines.append(f"Keeper: {keeper_text}")
                continue
            lines.append(f"Player: {user_text}")
            lines.append(f"Keeper: {keeper_text}")
        return '\n'.join(lines)

    def _parse_roll_check_response(self, text: str) -> dict[str, Any]:
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        payload = match.group(0) if match else text
        data = json5.loads(payload)
        if not isinstance(data, dict):
            return {}
        return data

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _scene_brief(self, scene: dict[str, Any] | None) -> dict[str, str]:
        scene = scene or {}
        return {
            'id': str(scene.get('scene_id', '')),
            'name': str(scene.get('scene_name', '')),
            'goal': str(scene.get('scene_goal', '')),
            'description': str(scene.get('scene_description', '')),
        }

    def _plot_brief(self, plot: dict[str, Any] | None) -> dict[str, str]:
        plot = plot or {}
        return {
            'id': str(plot.get('plot_id', '')),
            'name': str(plot.get('plot_name', '')),
            'goal': str(plot.get('plot_goal', '')),
        }

    def _format_scene_names(self, scenes: list[dict[str, Any]]) -> str:
        names = [
            str(scene.get('scene_name', '') or scene.get('scene_id', '')).strip()
            for scene in scenes
            if str(scene.get('scene_name', '') or scene.get('scene_id', '')).strip()
        ]
        return '\n'.join(f"- {name}" for name in names) or 'None'

    def _find_scene_and_plot(self, plot_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        for scene in self.db.list_scenes():
            for plot in scene.get('plots', []):
                if str(plot.get('plot_id', '')) == plot_id:
                    return scene, plot
        return None, None

    def _resolve_target_plot_id(self, target_plot_id: str) -> str:
        target_plot_id = str(target_plot_id or '').strip()
        if not target_plot_id:
            return ''
        target_scene, target_plot = self._find_scene_and_plot(target_plot_id)
        if target_scene and target_plot:
            return target_plot_id
        for scene in self.db.list_scenes():
            if str(scene.get('scene_id', '')) != target_plot_id:
                continue
            plots = scene.get('plots', [])
            if not plots:
                return ''
            return str(plots[0].get('plot_id', '') or '')
        return target_plot_id

    def _load_visited_state(self, state: NarrativeState) -> None:
        nav = self.db.get_system_state().get('navigation_state', {}) or {}
        visited_scenes = nav.get('visited_scenes', [])
        visited_plots = nav.get('visited_plots', [])
        state['visited_scenes'] = [str(item) for item in visited_scenes if str(item)]
        state['visited_plots'] = [str(item) for item in visited_plots if str(item)]
        state['long_term_memory'] = str(nav.get('long_term_memory', '') or '')

    def _save_visited_state(self, state: NarrativeState) -> None:
        navigation_state = self.db.get_system_state().get('navigation_state', {}) or {}
        navigation_state.update(
            {
                'visited_scenes': sorted(set(state.get('visited_scenes', []))),
                'visited_plots': sorted(set(state.get('visited_plots', []))),
                'long_term_memory': str(
                    state.get('long_term_memory', navigation_state.get('long_term_memory', '')) or ''
                ),
            }
        )
        self.db.update_system_state(
            {
                'navigation_state': navigation_state
            }
        )

    def _mark_visited(self, state: NarrativeState, scene_id: str | None = None, plot_id: str | None = None) -> None:
        scene_id = str(scene_id or state.get('scene_id', '') or '')
        plot_id = str(plot_id or state.get('plot_id', '') or '')
        if scene_id:
            state['visited_scenes'] = sorted(set(state.get('visited_scenes', []) + [scene_id]))
        if plot_id:
            state['visited_plots'] = sorted(set(state.get('visited_plots', []) + [plot_id]))

    def _hydrate_current_context(self, state: NarrativeState) -> None:
        scene = self.db.get_scene(state.get('scene_id', ''))
        state['scene_name'] = ''
        state['scene_goal'] = ''
        state['scene_description'] = ''
        state['current_scene_summary'] = ''
        state['plot_name'] = ''
        state['plot_goal'] = ''
        state['current_plot_raw_text'] = ''
        state['previous_plot_summary'] = ''
        state['script_summary'] = self.db.get_summary('script')
        if not scene:
            return
        state['scene_name'] = scene.get('scene_name', '')
        state['scene_goal'] = scene.get('scene_goal', '')
        state['scene_description'] = scene.get('scene_description', '')
        state['current_scene_summary'] = scene.get('scene_summary', '')
        for plot in scene.get('plots', []):
            if str(plot.get('plot_id', '')) == state.get('plot_id', ''):
                state['plot_name'] = plot.get('plot_name', '')
                state['plot_goal'] = plot.get('plot_goal', '')
                state['current_plot_raw_text'] = truncate_plot_raw_text(plot.get('raw_text', ''))
                state['previous_plot_summary'] = self.db.get_summary(
                    'plot',
                    scene_id=state.get('scene_id', ''),
                    plot_id=state.get('plot_id', ''),
                )
                break

    def _branch_prompt(self, state: NarrativeState) -> str:
        scenes = self.db.list_scenes()
        current_scene = self.db.get_scene(state.get('scene_id', '')) or {}
        current_plot = self.db.get_plot(state.get('plot_id', '')) or {}
        visited_scenes = set(state.get('visited_scenes', []))
        visited_plots = set(state.get('visited_plots', []))
        current_scene_plots = current_scene.get('plots', []) if current_scene else []
        branch_history = list(state.get('global_conversation_history', []))
        if state.get('latest_user_input'):
            branch_history.append({'user': state.get('latest_user_input', ''), 'agent': ''})
        return BRANCH_TRANSITION_PROMPT_TEMPLATE.format(
            branch_decision_json='{\n  "switch": true/false,\n  "target_plot_id": "scene_x_plot_y" or ""\n}',
            global_recent_conversation=self._format_recent_conversation(branch_history, rounds=3),
            script_summary=state.get('script_summary', '') or 'None',
            long_term_memory=state.get('long_term_memory', '') or 'None',
            current_plot_raw_text=state.get('current_plot_raw_text', '') or 'None',
            current_scene=self._json_dumps(self._scene_brief(current_scene)),
            unvisited_scenes=self._json_dumps([self._scene_brief(scene) for scene in scenes if str(scene.get('scene_id', '')) not in visited_scenes]),
            visited_scenes=self._json_dumps([self._scene_brief(scene) for scene in scenes if str(scene.get('scene_id', '')) in visited_scenes]),
            current_plot=self._json_dumps(self._plot_brief(current_plot)),
            unvisited_plots=self._json_dumps([self._plot_brief(plot) for plot in current_scene_plots if str(plot.get('plot_id', '')) not in visited_plots]),
            visited_plots=self._json_dumps([self._plot_brief(plot) for plot in current_scene_plots if str(plot.get('plot_id', '')) in visited_plots]),
        )

    def _parse_branch_decision(self, text: str) -> dict[str, Any]:
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        payload = match.group(0) if match else text
        data = json5.loads(payload)
        return data if isinstance(data, dict) else {}

    def build_prompt(self, state: NarrativeState) -> NarrativeState:
        return state

    def retrieve_memory(self, state: NarrativeState) -> NarrativeState:
        state['conversation_history'] = self.db.get_recent_turns(state['scene_id'], state['plot_id'], limit=12)
        state['global_conversation_history'] = self.db.get_global_recent_turns(limit=12)
        self._load_visited_state(state)
        self._hydrate_current_context(state)
        self._mark_visited(state)
        return state

    def decide_branch_transition(self, state: NarrativeState) -> NarrativeState:
        state['transition_switch'] = False
        state['transition_target_plot_id'] = ''
        prompt = self._branch_prompt(state)
        self._record_prompt(state, 'branch_transition_prompt', prompt)
        try:
            raw = self._llm_call(prompt, step_name='branch_transition_decision')
            decision = self._parse_branch_decision(raw)
            logger.info(f"PARSED: {decision}")
        except Exception as exc:
            logger.error("Branch transition decision failed error=%s", exc)
            decision = {}
        target_plot_id = self._resolve_target_plot_id(decision.get('target_plot_id', ''))
        logger.info(f"Resolved target_plot: {target_plot_id}")
        switch_value = decision.get('switch', False)
        requested_switch = switch_value.strip().lower() == 'true' if isinstance(switch_value, str) else bool(switch_value)
        should_switch = requested_switch and target_plot_id and target_plot_id != state.get('plot_id', '')
        target_scene, target_plot = self._find_scene_and_plot(target_plot_id)
        if not should_switch or not target_scene or not target_plot:
            return state

        state['transition_switch'] = True
        state['transition_target_plot_id'] = target_plot_id
        state['previous_scene_id'] = state.get('scene_id', '')
        state['previous_plot_id'] = state.get('plot_id', '')
        state['previous_scene_name'] = state.get('scene_name', '')
        state['previous_plot_name'] = state.get('plot_name', '')
        state['previous_scene_goal'] = state.get('scene_goal', '')
        state['previous_plot_goal'] = state.get('plot_goal', '')

        state['scene_id'] = str(target_scene.get('scene_id', ''))
        state['plot_id'] = str(target_plot.get('plot_id', ''))
        self.db.update_system_state(
            {
                'current_scene_id': state['scene_id'],
                'current_plot_id': state['plot_id'],
            }
        )
        state['conversation_history'] = self.db.get_recent_turns(state['scene_id'], state['plot_id'], limit=12)
        self._hydrate_current_context(state)
        self._mark_visited(state)
        self._save_visited_state(state)
        return state

    def generate_retrieval_queries(self, state: NarrativeState) -> NarrativeState:
        state['retrieval_queries'] = generate_retrieval_queries(
            state['latest_user_input'],
            state.get('plot_goal', ''),
            state.get('conversation_history', []),
        )
        return state

    def vector_retrieve(self, state: NarrativeState) -> NarrativeState:
        docs = []
        for q in state.get('retrieval_queries', []):
            docs.extend(self.vector_store.search(q, k=3))
        state['retrieved_docs'] = docs[:8]
        return state

    def construct_context(self, state: NarrativeState) -> NarrativeState:
        categorized = categorize_docs(state.get('retrieved_docs', []))
        state['setting'] = categorized['setting']
        state['clue'] = categorized['clue']
        player_skill_list = self._format_player_skill_list(state)
        recent_conversation = self._format_recent_conversation(state.get('global_conversation_history', []), rounds=2)
        state['roll_check_prompt'] = ROLL_CHECK_PROMPT_TEMPLATE.format(
            user_input=state['latest_user_input'],
            scene_id=state.get('scene_id', ''),
            scene_name=state.get('scene_name', '') or 'None',
            plot_id=state.get('plot_id', ''),
            plot_name=state.get('plot_name', '') or 'None',
            current_scene_goal=state.get('scene_goal', '') or 'None',
            current_scene_description=state.get('scene_description', '') or 'None',
            current_plot_goal=state.get('plot_goal', '') or 'None',
            current_plot_raw_text=state.get('current_plot_raw_text', '') or 'None',
            previous_plot_summary=state.get('previous_plot_summary', '') or 'None',
            current_scene_summary=state.get('current_scene_summary', '') or 'None',
            recent_conversation=recent_conversation,
            player_related_info=str(state.get('player_profile', {})),
            player_skill_list=player_skill_list,
        )
        self._record_prompt(state, 'roll_check_prompt', state['roll_check_prompt'])
        return state

    def check_whether_roll_dice(self, state: NarrativeState) -> NarrativeState:
        try:
            raw = self._llm_call(state['roll_check_prompt'], step_name='check_whether_roll_dice')
            parsed = self._parse_roll_check_response(raw)
            state['need_check'] = bool(parsed.get('need_check', False))
            state['check_skill'] = str(parsed.get('skill', '')).strip()
            state['check_reason'] = str(parsed.get('reason', '')).strip() or state['check_skill'] or 'skill check'
            state['dice_type'] = str(parsed.get('dice_type', '')).strip() or ('1d100' if state['need_check'] else '')
            logger.info(
                "Roll check decision need_check=%s skill=%s reason=%s dice_type=%s",
                state.get('need_check'),
                state.get('check_skill'),
                state.get('check_reason'),
                state.get('dice_type'),
            )
        except Exception as exc:
            logger.error("Roll check evaluation failed error=%s", exc)
            state['need_check'] = False
            state['check_skill'] = ''
            state['check_reason'] = ''
            state['dice_type'] = ''
        return state

    def roll_dice(self, state: NarrativeState) -> NarrativeState:
        dice_type = state.get('dice_type', '') or '1d100'
        dice_value = self._roll_dice_expr(dice_type)
        if dice_value:
            reason = state.get('check_reason', '') or 'skill check'
            state['dice_result'] = f"{dice_type}: {dice_value} (reason: {reason})"
            state['skill_check_result'] = self._build_skill_check_result(
                state,
                dice_type,
                state.get('check_skill', '') or state.get('check_reason', ''),
                dice_value,
            )
            logger.info(
                "Deterministic roll completed dice_result=%s skill_check_result=%s",
                state.get('dice_result'),
                state.get('skill_check_result'),
            )
        return state

    def generate_response(self, state: NarrativeState) -> NarrativeState:
        try:
            categorized = categorize_docs(state.get('retrieved_docs', []))
            recent_conversation = self._format_recent_conversation(state.get('global_conversation_history', []), rounds=3)
            scenes = self.db.list_scenes()
            visited_scene_ids = set(state.get('visited_scenes', []))
            visited_plot_ids = set(state.get('visited_plots', []))
            current_scene = self.db.get_scene(state.get('scene_id', '')) or {}
            current_scene_plots = current_scene.get('plots', []) if current_scene else []
            current_scene_index = next(
                (idx for idx, scene in enumerate(scenes) if str(scene.get('scene_id', '')) == state.get('scene_id', '')),
                -1,
            )
            next_scene = scenes[current_scene_index + 1] if 0 <= current_scene_index + 1 < len(scenes) else {}
            state['prompt'] = RESPONSE_PROMPT_TEMPLATE.format(
                agent_role='Narrative Agent',
                game_system='TRPG',
                tone_style='Immersive and grounded',
                narrative_perspective='Second person',
                response_length='Concise',
                output_language=self._get_output_language(state),
                user_input=state['latest_user_input'],
                scene_id=state.get('scene_id', ''),
                scene_name=state.get('scene_name', '') or 'None',
                plot_id=state.get('plot_id', ''),
                plot_name=state.get('plot_name', '') or 'None',
                current_scene_goal=state.get('scene_goal', '') or 'None',
                current_scene_description=state.get('scene_description', '') or 'None',
                next_scene_name=next_scene.get('scene_name', '') or 'None',
                next_scene_description=next_scene.get('scene_description', '') or 'None',
                unvisited_scene_name=self._format_scene_names(
                    [scene for scene in scenes if str(scene.get('scene_id', '')) not in visited_scene_ids]
                ),
                visited_scene_name=self._format_scene_names(
                    [scene for scene in scenes if str(scene.get('scene_id', '')) in visited_scene_ids]
                ),
                unvisited_plot_name=self._format_scene_names(
                    [
                        {
                            'scene_name': plot.get('plot_name', '') or plot.get('plot_id', ''),
                        }
                        for plot in current_scene_plots
                        if str(plot.get('plot_id', '')) not in visited_plot_ids
                    ]
                ),
                current_plot_goal=state.get('plot_goal', '') or 'None',
                current_plot_raw_text=state.get('current_plot_raw_text', '') or 'None',
                previous_plot_summary=state.get('previous_plot_summary', '') or 'None',
                current_scene_summary=state.get('current_scene_summary', '') or 'None',
                long_term_memory=state.get('long_term_memory', '') or 'None',
                script_summary=state.get('script_summary', '') or 'None',
                recent_conversation=recent_conversation,
                npc_related_info=categorized['npc_related_info'],
                player_related_info=str(state.get('player_profile', {})),
                setting=state.get('setting', 'None'),
                clue=state.get('clue', 'None'),
                dice_result=state.get('dice_result') or 'None',
                skill_check_result=state.get('skill_check_result') or 'None',
            )
            self._record_prompt(state, 'generate_response_prompt', state['prompt'])
            state['response'] = self._llm_call(state['prompt'], step_name='generate_response')
        except Exception as e:
            logger.error("LLM error in generate_response prompt_length=%s error=%s", len(state.get('prompt', '')), e)
            print("LLM error:", e)
            traceback.print_exc()
            state['response'] = self._fallback_response(state)
        return state

    def write_memory(self, state: NarrativeState) -> NarrativeState:
        self.db.append_memory(state['scene_id'], state['plot_id'], state['latest_user_input'], state['response'])
        return state

    def finalize_turn_state(self, state: NarrativeState) -> NarrativeState:
        self._mark_visited(state)
        if state.get('transition_switch'):
            if state.get('previous_plot_id'):
                self.db.save_summary(
                    'plot',
                    self._build_plot_summary(state, previous=True),
                    scene_id=state.get('previous_scene_id', ''),
                    plot_id=state.get('previous_plot_id', ''),
                )
            if state.get('previous_scene_id') and state.get('previous_scene_id') != state.get('scene_id'):
                scene_summary = self._build_scene_summary(state.get('previous_scene_id', ''), state=state)
                self.db.update_scene(state.get('previous_scene_id', ''), {'scene_summary': scene_summary})
                self.db.save_summary('scene', scene_summary, scene_id=state.get('previous_scene_id', ''))
        self._update_long_term_memory(state)
        self._save_visited_state(state)
        return state

    def _update_long_term_memory(self, state: NarrativeState) -> None:
        row = self.db.conn.execute(
            "SELECT COUNT(*) AS turn_count FROM memory WHERE user <> ?",
            (KP_OPENING_MARKER,),
        ).fetchone()
        turn_count = int(row['turn_count']) if row else 0
        if turn_count <= 0 or turn_count % 3 != 0:
            return

        recent_history = self.db.get_global_recent_turns(limit=3)
        prompt = LONG_TERM_MEMORY_UPDATE_PROMPT_TEMPLATE.format(
            current_long_term_memory=state.get('long_term_memory', '') or 'None',
            recent_conversation=self._format_recent_conversation(recent_history, rounds=3),
        )
        self._record_prompt(state, 'long_term_memory_update_prompt', prompt)
        try:
            updated_memory = self._llm_call(prompt, step_name='long_term_memory_update')
        except Exception as exc:
            logger.error("Long-term memory update failed error=%s", exc)
            return
        if updated_memory:
            state['long_term_memory'] = updated_memory

    def _roll_dice_expr(self, dice_expr: str) -> str | None:
        m = re.fullmatch(r'\s*(\d*)d(\d+)\s*', (dice_expr or '').lower())
        if not m:
            return None
        count = int(m.group(1) or '1')
        sides = int(m.group(2))
        count = max(1, min(count, 20))
        sides = max(2, min(sides, 1000))
        rolls = [random.randint(1, sides) for _ in range(count)]
        return f"{rolls} (sum={sum(rolls)})"

    def _normalize_skill_name(self, value: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', (value or '').strip().lower())

    def _extract_named_value(self, text: str) -> tuple[str, int] | None:
        if ':' not in text:
            return None
        name, raw_value = text.split(':', 1)
        try:
            return name.strip(), int(float(raw_value.strip()))
        except ValueError:
            return None

    def _resolve_skill_value(self, state: NarrativeState, reason: str) -> tuple[str, int] | None:
        profile = state.get('player_profile', {}) or {}
        candidates: dict[str, tuple[str, int]] = {}

        for bucket_name in ('occupation', 'personal_interest'):
            for entry in profile.get('chosen_skill_allocations', {}).get(bucket_name, []):
                parsed = self._extract_named_value(str(entry))
                if parsed:
                    skill_name, skill_value = parsed
                    candidates[self._normalize_skill_name(skill_name)] = (skill_name, skill_value)

        for attr_group in ('characteristics', 'derived_attributes'):
            for skill_name, skill_value in profile.get(attr_group, {}).items():
                try:
                    candidates[self._normalize_skill_name(str(skill_name))] = (str(skill_name), int(float(skill_value)))
                except (TypeError, ValueError):
                    continue

        reason_norm = self._normalize_skill_name(reason)
        if not reason_norm:
            return None
        if 'sanity' in reason.lower():
            san_entry = candidates.get('san')
            if san_entry:
                return ('SAN', san_entry[1])
        for key, value in candidates.items():
            if key and (key in reason_norm or reason_norm in key):
                return value
        return None

    def _extract_roll_total(self, dice_text: str) -> int | None:
        m = re.search(r'sum=(\d+)', dice_text)
        if m:
            return int(m.group(1))
        return None

    def _evaluate_skill_check(self, roll_total: int, skill_value: int) -> str:
        if roll_total >= 96:
            return 'Worst Fail'
        if roll_total <= max(1, skill_value // 5):
            return 'Extreme Success'
        if roll_total <= max(1, skill_value // 2):
            return 'Hard Success'
        if roll_total <= skill_value:
            return 'Regular Success'
        return 'Fail'

    def _build_skill_check_result(
        self,
        state: NarrativeState,
        dice_type: str,
        reason: str,
        dice_text: str,
    ) -> str | None:
        if self._normalize_skill_name(dice_type) != '1d100':
            return None
        roll_total = self._extract_roll_total(dice_text)
        if roll_total is None:
            return None
        resolved = self._resolve_skill_value(state, reason)
        if resolved is None:
            return None
        skill_name, skill_value = resolved
        outcome = self._evaluate_skill_check(roll_total, skill_value)
        return f"{skill_name} {skill_value}: {outcome}"

    def _fallback_response(self, state: NarrativeState) -> str:
        output_language = self._get_output_language(state)
        user_input = state['latest_user_input']
        dice_hint = re.search(r'(\d*d\d+)', user_input.lower())
        if dice_hint:
            rolled = self._roll_dice_expr(dice_hint.group(1))
            if rolled:
                state['dice_result'] = f"{dice_hint.group(1)}: {rolled}"
                state['skill_check_result'] = None
        if output_language == 'Chinese':
            base = f"你的行动是：{user_input}。"
            if state.get('plot_goal'):
                base += f"剧情正朝着以下目标推进：{state['plot_goal']}。"
            if state.get('clue') and state.get('clue') != 'None':
                base += f"相关线索：{state['clue']}。"
            if state.get('dice_result'):
                base += f"已应用骰子结果（{state['dice_result']}）。"
            base += '接下来会发生什么？'
            return base
        base = f"You act: {user_input}. "
        if state.get('plot_goal'):
            base += f"The story advances toward: {state['plot_goal']}. "
        if state.get('clue') and state.get('clue') != 'None':
            base += f"Relevant clue: {state['clue']}. "
        if state.get('dice_result'):
            base += f"Dice result applied ({state['dice_result']}). "
        if state.get('skill_check_result'):
            base += f"Skill check result applied ({state['skill_check_result']}). "
        base += 'What do you do next?'
        return base

    def _generate_scene_opening(self, scene_id: str, plot_id: str) -> str:
        scene = self.db.get_scene(scene_id) or {}
        plot = self.db.get_plot(plot_id) or {}
        output_language = self._get_output_language()
        prompt = f"""
You are a TRPG Keeper.

Write the opening narration for the beginning of the story.

Guidelines:
- Describe the immediate environment, situation, NPC dialogue, and NPC actions in a vivid and immersive way.
- Focus only on what is happening right now.
- Reveal information gradually. Do not provide too much information at once; encourage player exploration and role-play.
- Present the situation, then STOP and wait for the player to decide what to do next.
- Avoid meta commentary. Do not mention scene IDs, scene names, or any system-level information.
- Do NOT reveal future events or the full storyline.
- Do NOT decide the player character’s actions or thoughts.
- Do NOT ask hook questions.
- Write the opening entirely in {output_language}.


Scene ID: {scene_id}
Scene Name: {scene.get('scene_name', '')}
Scene Goal: {scene.get('scene_goal', '')}
Scene Description: {scene.get('scene_description', '')}
Plot Name: {plot.get('plot_name', '')}
Plot Goal: {plot.get('plot_goal', '')}
Current Plot Raw Text: {truncate_plot_raw_text(plot.get('raw_text', '')) or 'None'}
Player Profile: {self.db.get_player_profile()}
"""
        self._record_prompt(None, 'scene_opening_prompt', prompt)
        try:
            result = self._llm_call(prompt, step_name='scene_opening_generation')
            if result:
                return result
        except Exception:
            pass
        scene_label = scene.get('scene_name', '') or scene_id
        plot_label = plot.get('plot_name', '') or plot.get('plot_goal', 'advance the story')
        if output_language == 'Chinese':
            return (
                f"夜色笼罩着{scene_label}。你感到命运的下一条线索正在将你向前牵引。\n"
                f"你当前的直接目标是：{plot_label}。\n"
                "你从哪里开始？"
            )
        return (
            f"Night settles over {scene_label}. You sense the next thread of fate pulling you forward.\n"
            f"Your immediate objective: {plot_label}.\n"
            "Where do you begin?"
        )

    def _build_plot_summary(self, state: NarrativeState, *, previous: bool = False) -> str:
        tail = (state.get('global_conversation_history', []) if previous else state.get('conversation_history', []))[-8:]
        history_text = '\n'.join([f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}" for t in tail])
        scene_id = state.get('previous_scene_id', '') if previous else state.get('scene_id', '')
        plot_id = state.get('previous_plot_id', '') if previous else state.get('plot_id', '')
        plot_goal = state.get('previous_plot_goal', '') if previous else state.get('plot_goal', '')
        prompt = f"""
Summarize the plot before a player-driven branch switch in 3 bullet points.
Focus on:
1) what happened
2) key clues
3) character changes

Scene: {scene_id}
Plot: {plot_id}
Plot Goal: {plot_goal}
Latest Turn User: {state.get('latest_user_input', '')}
Latest Turn Agent: {state.get('response', '')}
Recent History:
{history_text}
"""
        self._record_prompt(state, 'plot_summary_prompt', prompt)
        try:
            summary = call_llm(prompt, step_name='plot_summary_generation').strip()
            if summary:
                return summary
        except Exception:
            pass
        return f"Plot {plot_id} was left after player-driven branch movement from: {plot_goal or 'the active plot'}."

    def _build_scene_summary(self, scene_id: str, state: NarrativeState | None = None) -> str:
        scene = self.db.get_scene(scene_id) or {}
        plots = scene.get('plots', [])
        plot_lines = '\n'.join(
            [
                f"- {p.get('plot_id')}: goal={p.get('plot_goal', '')}"
                for p in plots
            ]
        )
        prompt = f"""
Summarize a scene before a player-driven branch switch in 4 bullet points.
Include:
- core conflict
- emotional shift
- gained information
- narrative turning point

Scene: {scene_id}
Scene Goal: {scene.get('scene_goal', '')}
Plots:
{plot_lines}
"""
        self._record_prompt(state, 'scene_summary_prompt', prompt)
        try:
            summary = call_llm(prompt, step_name='scene_summary_generation').strip()
            if summary:
                return summary
        except Exception:
            pass
        return f"Scene {scene_id} was left after player-driven branch movement from: {scene.get('scene_goal', '')}"

