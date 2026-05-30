# Script-Driven Narrative Agent System

A local Streamlit application for script-driven narrative role-playing. It parses a Markdown script into scenes, plots, and knowledge entries, stores structured state in SQLite, retrieves supporting context from Chroma, and runs each turn through a LangGraph-based narrative pipeline.

## Project Overview

The current system is organized around a simple flow:

- upload a Markdown script
- parse it into structured scenes, plots, and knowledge
- review the parse and create a character in the UI
- run interactive turns against the current plot context

Design goals in the current codebase:

- keep generation grounded in the active plot raw text
- preserve long-term story state across turns
- support scene and plot transitions without losing context
- separate deterministic state storage from semantic retrieval


## Architecture

```text
+----------------------- Streamlit UI ------------------------+
| Upload Markdown -> Review Parse -> Create Character -> Chat |
+------------------------------+------------------------------+
                               |
                               v
+---------------------- Parse / Storage ----------------------+
| parser.py -> scenes / plots / knowledge / script summary    |
| database.py -> persist structured runtime data              |
| vector_store.py -> index knowledge in Chroma                |
+------------------------------+------------------------------+
                               |
                               v
+------------------- LangGraph Turn Pipeline -----------------+
| retrieve_memory -> decide_branch_transition                 |
| -> generate_retrieval_queries -> vector_retrieve            |
| -> construct_context -> check_whether_roll_dice             |
| -> (roll_dice or skip) -> generate_response                 |
| -> write_memory -> finalize_turn_state                      |
+------------------------------+------------------------------+
                               |
                +--------------+--------------+
                v                             v
         SQLite runtime state           Chroma retrieval
  scenes / plots / memory / summaries   semantic search over
  system_state / navigation_state       parsed knowledge
```

### Pipeline Summary

- **Ingestion**
  - `app/ui.py` accepts Markdown input.
  - `app/parser.py` uses heading structure plus LLM extraction to build scenes, plots, knowledge entries, and a script summary.
  - `app/database.py` stores the structured output.
  - `app/vector_store.py` indexes knowledge entries for retrieval.

- **Session setup**
  - The UI progresses through parse review and character creation.
  - `system_state` stores the active app stage and the current scene / plot pointer.
  - `NarrativeAgent.generate_initial_response()` creates the opening narration for the first plot.

- **Per-turn runtime**
  - `app/agent_graph.py` loads recent memory and current plot context.
  - A branch decision step may switch the active plot or scene before retrieval and response generation.
  - Retrieval queries are generated from player input, the current plot goal, and recent conversation, then sent to Chroma.
  - The runtime determines whether a dice roll is needed, rolls deterministically when required, and generates the next response.
  - The turn is written to memory, then summaries and navigation state are refreshed.

## Project Structure

```text
/app
    agent_graph.py
    database.py
    llm_client.py
    parser.py
    rag.py
    rules_loader.py
    ui.py
    vector_store.py

main.py
requirements.txt
README.md
```

## Core Components

### `app/database.py` (SQLite)
- Canonical structured store for the application.
- Main tables:
  - `scenes`
  - `plots`
  - `memory`
  - `summaries`
  - `system_state`
  - `knowledge_base`
- Persists:
  - current app stage
  - current scene / plot pointer
  - turn history
  - plot / scene summaries
  - output language
  - `navigation_state_json` for visited scenes, visited plots, and long-term memory

### `app/vector_store.py` (Chroma)
- Semantic retrieval layer.
- Uses a local sentence-transformer embedding model.
- Indexes parsed knowledge entries and returns top-k matches for prompt context.

### `app/parser.py`
- Script ingestion for Markdown input.
- Splits the source by heading structure into scene sections and plot spans.
- Extracts:
  - scene metadata
  - plot metadata
  - knowledge entries
  - a compact script summary
- Keeps each plot linked to its original raw Markdown text, which is the main grounding source during play.

### `app/agent_graph.py`
- LangGraph orchestration for the runtime turn pipeline.
- Main nodes:
  1. `build_prompt`
  2. `retrieve_memory`
  3. `decide_branch_transition`
  4. `generate_retrieval_queries`
  5. `vector_retrieve`
  6. `construct_context`
  7. `check_whether_roll_dice`
  8. `roll_dice`
  9. `generate_response`
  10. `write_memory`
  11. `finalize_turn_state`
- Handles branch switching, retrieval orchestration, deterministic dice resolution, response generation, and post-turn summary updates.

### `app/rag.py`
- Generates retrieval queries from player input, current plot goal, and recent conversation.
- Buckets retrieved documents into prompt sections such as NPC, setting, and clue.

### `app/llm_client.py`
- Centralizes LLM backend selection and invocation.
- Supports provider/model routing through `llm_backend.txt`, environment variables, and per-step overrides.

### `app/rules_loader.py`
- Utility for turning `database/GameRules.md` into knowledge chunks.
- Separate from the main UI parsing path.

### `app/ui.py` + `main.py`
Streamlit runtime.

Flow:
1. Upload Markdown script
2. Parse and review scene structure
3. Create character
4. Start narrative chat session
5. Play the story through the chat loop

The UI also includes:
- output language selection (`English` / `Chinese`)
- debug prompt inspection
- CoC-style assisted character creation
- visible dice roll and skill check results during play

## Parse Mechanism

The system parses uploaded Markdown scripts into structured runtime data before play begins. The parsing flow converts raw document text into scenes, plots, knowledge entries, and a script summary, then stores the result for later retrieval and turn execution.

### High-Level Workflow

- `app/ui.py` accepts an uploaded Markdown file.
- `read_uploaded_document()` in `app/parser.py` reads and normalizes the raw file content.
- `parse_script_bundle()` in `app/parser.py` runs the main parsing pipeline.
- The parser splits the Markdown by heading structure into scene sections and plot spans.
- Each scene section is sent through an LLM extraction prompt to produce:
  - scene metadata
  - plot metadata
  - knowledge entries
- The parser builds a compact global script summary from the extracted scene/plot structure.
- `app/database.py` stores the parsed scenes, plots, knowledge, and summaries in SQLite.
- `app/vector_store.py` indexes the parsed knowledge entries in Chroma for later retrieval.

### Key Parsing Steps

- **Input normalization**
  - decode Markdown text
  - remove decorative markup that should not affect parsing
- **Structural segmentation**
  - detect heading hierarchy
  - determine scene sections and plot-level spans from the Markdown structure
- **LLM extraction**
  - Read and summarize each section to identify key information
  * Transform scene-related sections into structured outputs, including structured scene / plot / knowledge output
- **Persistence**
  - write structured results to SQLite
  - write knowledge embeddings to Chroma
- **Initialization**
  - select the first playable scene and plot
  - store script summary and source metadata for the UI and runtime

### Key Files

- `app/parser.py`
  - owns the Markdown parsing pipeline and structured extraction
  - key functions: `read_uploaded_document()`, `parse_script_bundle()`
- `app/ui.py`
  - triggers parsing, resets old story data, and initializes the first playable position
- `app/database.py`
  - stores parsed scenes, plots, knowledge, summaries, and system state
- `app/vector_store.py`
  - indexes parsed knowledge entries for semantic retrieval during play

## Branch Mechanism

Branching is handled as a pre-response transition step inside `app/agent_graph.py`. Before retrieval and response generation, the runtime decides whether to remain in the current plot or switch to another plot or scene.

### High-Level Decision Logic

- The branch prompt is built from:
  - recent global conversation
  - current plot raw text
  - long-term memory
  - script summary
  - current and visited scenes / plots
- The LLM returns strict JSON:
  - `switch`
  - `target_plot_id`
- A transition is applied only when:
  - `switch` is true
  - the target resolves to a valid plot
  - the target is different from the current plot

### State Update Flow

- `_resolve_target_plot_id()` normalizes the returned target.
- `decide_branch_transition()` updates:
  - in-memory `scene_id` / `plot_id`
  - `system_state.current_scene_id`
  - `system_state.current_plot_id`
- The new scene / plot context is reloaded immediately, so the rest of the turn runs against the new target.
- `finalize_turn_state()` then:
  - saves a plot summary for the plot that was left
  - saves a scene summary when the switch leaves the previous scene
  - updates visited scenes / plots
  - stores refreshed long-term memory in `navigation_state_json`

### Key Files

- `app/agent_graph.py`
  - `_branch_prompt()` builds the decision prompt.
  - `decide_branch_transition()` makes and applies the decision.
  - `_resolve_target_plot_id()` validates the target.
  - `finalize_turn_state()` writes summaries and navigation updates.
- `app/database.py`
  - `system_state` stores the active scene and plot.
  - `navigation_state_json` stores visited scenes, visited plots, and long-term memory.
  - `save_summary()` persists plot and scene summaries used after transitions.
- `app/ui.py`
  - initializes the first playable scene / plot after parsing
  - calls `agent.run_turn()` during the chat session
- `app/parser.py`
  - defines the scene and plot IDs that branch transitions target

## Response Generation Mechanism

Responses are generated inside the same LangGraph turn pipeline after the current story position has been loaded, and after any branch transition has been applied. The goal is to build a grounded prompt from the active plot, recent memory, retrieved knowledge, and any dice outcome, then generate the next narrative turn.

### High-Level Workflow

- `retrieve_memory()` loads:
  - recent plot-local turns
  - recent global turns
  - current scene / plot metadata
  - saved summaries and long-term memory
- `generate_retrieval_queries()` creates retrieval queries from:
  - the latest player input
  - the current plot goal
  - recent conversation
- `vector_retrieve()` searches Chroma for supporting knowledge.
- `construct_context()` organizes retrieved results into prompt sections such as NPC, setting, and clue, and prepares the dice-check prompt.
- `check_whether_roll_dice()` decides whether the player action requires a deterministic skill check.
- `roll_dice()` runs the roll when needed and converts the result into a usable skill-check outcome.
- `generate_response()` builds the final narrative prompt and calls the LLM to produce the next response.
- `write_memory()` stores the player turn and generated response in SQLite.
- `finalize_turn_state()` updates summaries, visited-state tracking, and long-term memory for later turns.

### What Grounds the Response

- current plot raw text
- current scene and plot goals
- recent conversation and prior summaries
- retrieved semantic knowledge from Chroma
- player profile and skill values
- dice result and evaluated skill-check outcome, when a check occurs

### Key Files

- `app/agent_graph.py`
  - owns the turn pipeline and builds the final response prompt in `generate_response()`
  - decides and resolves dice checks before generation
- `app/rag.py`
  - generates retrieval queries and categorizes retrieved knowledge for prompt use
- `app/vector_store.py`
  - executes semantic search over indexed knowledge
- `app/database.py`
  - provides memory, summaries, system state, and persistence for each turn
- `app/llm_client.py`
  - performs the actual model call used for query generation, dice-check decisions, and final response generation

## SQLite vs Chroma Responsibilities

- **SQLite**: canonical structured runtime state.
  - scenes, plots, memory, summaries, and system state
  - current scene / plot pointer
  - output language
  - navigation state

- **Chroma**: semantic similarity only.
  - vector search over parsed knowledge entries
  - retrieval for context augmentation

This separation keeps deterministic state operations isolated from semantic search operations.

## Setup

### 1) Choose the LLM backend (one file, default is Qwen / NVIDIA)

In the project root, edit **`llm_backend.txt`**. The first non-empty line that is not a comment (`#`) must start with either:

- **`qwen`** — use the NVIDIA Build API with Qwen models (same as writing `nvidia`; this is the **default**).
- **`openai`** — use the OpenAI Chat Completions API instead.

Example:

```text
qwen
```

You can optionally specify a model on the same line:

```text
qwen qwen/qwen3.5-397b-a17b
```

```text
openai gpt-5_4-mini-2026-03-17
```

To switch to OpenAI, change that line to:

```text
openai
```

Reference copy: **`llm_backend.example.txt`** (you can duplicate it to `llm_backend.txt` if needed).

**Per-step backend and model (optional):** after the first line, add rows of the form `step_name = backend ...model...`. The `step_name` must match the `step_name` string used in code (for example `generate_response`, `check_whether_roll_dice`, `plot_summary_generation`). This lets different pipeline steps use different providers or model IDs without editing Python.

```text
qwen qwen/qwen2.5-7b-instruct

generate_response = openai gpt-5_4-mini-2026-03-17
plot_summary_generation = qwen qwen/qwen3.5-397b-a17b
```

Resolution order:

- **Provider:** explicit `call_llm(..., provider=...)` → `LLM_PROVIDER` env → per-step line → global first line → default `qwen`.
- **Model:** explicit `call_llm(..., model=...)` → `NVIDIA_MODEL` / `OPENAI_MODEL` env → per-step line (only if that line’s backend matches the resolved provider) → global first line (if its backend matches) → built-in default.

Common `step_name` values include: `generate_response`, `check_whether_roll_dice`, `scene_opening_generation`, `plot_summary_generation`, `scene_summary_generation`, `generate_retrieval_queries`, `player_alignment_classification`, `turn_state_extraction`, `plot_completion_evaluation`, and `generation` for generic calls.

**Override (optional):** for CI or one-off runs, set environment variable `LLM_PROVIDER` to `qwen`, `nvidia`, or `openai`. It takes precedence over per-step and global lines in `llm_backend.txt`.

### 2) API keys

**Qwen / NVIDIA** (when backend is `qwen` or `nvidia`):

- Get a key from [NVIDIA Build API keys](https://build.nvidia.com/settings/api-keys).
- Put it in project root **`api_key.txt`** (single line), or set **`NVIDIA_API_KEY`**, or use a combined file (see below).

**OpenAI** (when backend is `openai`):

- Set **`OPENAI_API_KEY`**, or put the key in **`openai_api_key.txt`** (one line, bare `sk-...`), or add a line to **`api_key.txt`**:

```text
OPENAI_API_KEY=sk-...
```

If more than one source is set, **only one is used** (no merge): **`OPENAI_API_KEY` env → `openai_api_key.txt` → `api_key.txt`** with `OPENAI_API_KEY=...`. Same key in two places does not cause errors; the first in that order wins.

**Both backends in one `api_key.txt` (optional):**

```text
NVIDIA_API_KEY=nvapi-...
OPENAI_API_KEY=sk-...
```

OpenAI-only tuning (optional):

```bash
export OPENAI_MODEL="gpt-5_4-mini-2026-03-17"
export OPENAI_MAX_TOKENS=2048
export OPENAI_TEMPERATURE=0.6
export OPENAI_TOP_P=0.95
```

**GPT-5.x on Chat Completions:** newer models expect **`max_completion_tokens`** instead of `max_tokens`. This project sends the correct field automatically when the model id looks like GPT-5 (for example `gpt-5.2`, `gpt-5.4`, or snapshots containing `gpt-5`). If you still see `unsupported parameter` errors, set `OPENAI_USE_MAX_COMPLETION_TOKENS=true` to force that behavior.

**Per-step keys must match code:** lines like `parser_extract = ...` only apply if the Python code passes `step_name="parser_extract"`. If the name does not match any call site, that row is ignored (see the list of known `step_name` values above).

No code changes are required to switch: edit **`llm_backend.txt`** and ensure the matching API key is configured.

### 3) Public demo deployment guardrails

For Streamlit Community Cloud or any public deployment, configure secrets/environment variables in the deployment UI. Do **not** commit `api_key.txt`, `openai_api_key.txt`, or `.streamlit/secrets.toml`.

Recommended public demo secrets:

```toml
OPENAI_API_KEY = "sk-..."
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-5_4-mini-2026-03-17"
PUBLIC_DEMO_MODE = "true"
PUBLIC_DEMO_MAX_TURNS = "18"
PUBLIC_DEMO_MAX_INPUT_CHARS = "1200"
PUBLIC_DEMO_MAX_UPLOAD_BYTES = "200000"
PUBLIC_DEMO_MIN_TURN_SECONDS = "6"
PUBLIC_DEMO_DAILY_TURN_BUDGET = "250"
PUBLIC_DEMO_SESSION_TTL_HOURS = "24"
PUBLIC_DEMO_STRICT_CONFIG = "true"

# Optional, recommended for multi-instance public deployment:
UPSTASH_REDIS_REST_URL = "https://..."
UPSTASH_REDIS_REST_TOKEN = "..."
```

Public demo safeguards:

- each browser session uses its own runtime directory under `.runtime/sessions/<session_id>`
- old public demo session directories are cleaned after `PUBLIC_DEMO_SESSION_TTL_HOURS`
- each public player action is capped by `PUBLIC_DEMO_MAX_INPUT_CHARS`
- Markdown uploads are capped by `PUBLIC_DEMO_MAX_UPLOAD_BYTES`
- repeated player actions are throttled by `PUBLIC_DEMO_MIN_TURN_SECONDS`
- a daily action budget is enforced by `PUBLIC_DEMO_DAILY_TURN_BUDGET`
- if `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` are set, the daily budget is shared through Upstash Redis; otherwise it falls back to an instance-local `.runtime/public_demo_usage.json` file
- when `PUBLIC_DEMO_STRICT_CONFIG=true`, missing LLM credentials or invalid core limits stop the public demo with a friendly unavailable message
- embedding model loading is lazy, so opening the public landing screen does not immediately load the local sentence-transformer model

Once the API key is set up, create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
streamlit run main.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Public Demo Deployment

For a public playable deployment checklist, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Notes

- This project is designed for local standalone execution.
- All state is persisted locally (`narrative.db` and `.chroma/`).

## Debug Scripts

A new `test/` directory has been added.
These scripts all use **mock inputs + print outputs**, allowing you to manually verify whether the results match expectations.

Directory contents:

* `test/test_llm_generate.py`: Tests the raw LLM API call (streaming behavior, thinking/reasoning/content parsing).
  
  ⚠️ **Before running this test, make sure API keys are configured as described in the Setup section** (`api_key.txt` / env vars).

  ⚠️ **All LLM invocation logic in the project must stay aligned with this file.**

  Any changes to request payload, headers, or streaming parsing should be validated here first.
* `test/test_init.py`: Tests importing `app/__init__.py`
* `test/test_database.py`: Tests `app/database.py` (database creation, insert, read, state updates)
* `test/test_parser.py`: Tests `app/parser.py` (Markdown parsing and scene/plot extraction)
* `test/test_rag.py`: Tests `app/rag.py` (query generation and knowledge classification)
* `test/test_rules_loader.py`: Tests `app/rules_loader.py` (loading `database/GameRules.md` into knowledge chunks)
* `test/test_state.py`: Legacy progression test harness kept under `test/`
* `test/test_vector_store.py`: Tests `app/vector_store.py` (insertion and retrieval)
* `test/test_agent_graph.py`: Tests `app/agent_graph.py` (complete single-turn workflow)
* `test/test_main.py`: Tests `main.py` import and entry-point availability
* `test/run_all.py`: Executes all test scripts sequentially

Run individually:

```bash
python test/test_llm_generate.py
python test/test_database.py
python test/test_parser.py
python test/test_agent_graph.py
```

Run all at once:

```bash
python test/run_all.py
```
