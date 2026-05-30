from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def _parse_json_object(value: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return dict(default or {})


class Database:
    def __init__(self, db_path: str = "narrative.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("PRAGMA foreign_keys = OFF")
        self._ensure_story_tables()
        self._ensure_support_tables()
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.commit()
        self._ensure_system_state_row()

    def _ensure_story_tables(self) -> None:
        expected_scenes = [
            "scene_id",
            "scene_index",
            "scene_name",
            "scene_goal",
            "scene_description",
            "scene_summary",
            "status",
        ]
        expected_plots = [
            "plot_id",
            "scene_id",
            "plot_index",
            "plot_name",
            "plot_goal",
            "raw_text",
            "status",
            "progress",
        ]
        expected_knowledge = [
            "knowledge_id",
            "knowledge_type",
            "title",
            "content",
        ]

        self._rebuild_table_if_needed("scenes", expected_scenes, self._create_scenes_table, self._copy_scenes_rows)
        self._rebuild_table_if_needed("plots", expected_plots, self._create_plots_table, self._copy_plots_rows)
        self._rebuild_table_if_needed(
            "knowledge_base",
            expected_knowledge,
            self._create_knowledge_table,
            self._copy_knowledge_rows,
        )

    def _ensure_support_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id TEXT NOT NULL,
                plot_id TEXT NOT NULL,
                user TEXT NOT NULL,
                agent TEXT NOT NULL,
                turn_state_json TEXT NOT NULL DEFAULT '{}',
                visit_id INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_type TEXT NOT NULL,
                scene_id TEXT,
                plot_id TEXT,
                content TEXT NOT NULL,
                UNIQUE(summary_type, scene_id, plot_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                stage TEXT NOT NULL,
                current_scene_id TEXT NOT NULL,
                current_plot_id TEXT NOT NULL,
                plot_progress REAL NOT NULL,
                scene_progress REAL NOT NULL,
                player_profile TEXT NOT NULL,
                current_scene_intro TEXT NOT NULL,
                output_language TEXT NOT NULL DEFAULT 'English',
                navigation_state_json TEXT NOT NULL DEFAULT '{}',
                current_visit_id INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._ensure_column("memory", "turn_state_json TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("memory", "visit_id INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("system_state", "output_language TEXT NOT NULL DEFAULT 'English'")
        self._ensure_column("system_state", "navigation_state_json TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("system_state", "current_visit_id INTEGER NOT NULL DEFAULT 0")

    def _ensure_system_state_row(self) -> None:
        if self.conn.execute("SELECT 1 FROM system_state WHERE id = 1").fetchone():
            return
        self.conn.execute(
            """
            INSERT INTO system_state (
                id,
                stage,
                current_scene_id,
                current_plot_id,
                plot_progress,
                scene_progress,
                player_profile,
                current_scene_intro,
                output_language,
                navigation_state_json,
                current_visit_id
            )
            VALUES (1, 'upload', '', '', 0.0, 0.0, '{}', '', 'English', '{}', 0)
            """
        )
        self.conn.commit()

    def _rebuild_table_if_needed(
        self,
        table: str,
        expected_columns: list[str],
        create_table: Callable[[], None],
        copy_rows: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        existing_columns = self._table_columns(table)
        if not existing_columns:
            create_table()
            return
        if existing_columns == expected_columns:
            return

        legacy_name = f"{table}_legacy"
        self.conn.execute(f"DROP TABLE IF EXISTS {legacy_name}")
        legacy_rows = [dict(row) for row in self.conn.execute(f"SELECT * FROM {table}").fetchall()]
        self.conn.execute(f"ALTER TABLE {table} RENAME TO {legacy_name}")
        create_table()
        copy_rows(legacy_rows)
        self.conn.execute(f"DROP TABLE IF EXISTS {legacy_name}")

    def _create_scenes_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE scenes (
                scene_id TEXT PRIMARY KEY,
                scene_index INTEGER NOT NULL,
                scene_name TEXT NOT NULL,
                scene_goal TEXT NOT NULL,
                scene_description TEXT NOT NULL DEFAULT '',
                scene_summary TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )

    def _create_plots_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE plots (
                plot_id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                plot_index INTEGER NOT NULL,
                plot_name TEXT NOT NULL,
                plot_goal TEXT NOT NULL,
                raw_text TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                progress REAL NOT NULL DEFAULT 0.0,
                FOREIGN KEY(scene_id) REFERENCES scenes(scene_id)
            )
            """
        )

    def _create_knowledge_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE knowledge_base (
                knowledge_id TEXT PRIMARY KEY,
                knowledge_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )

    def _copy_scenes_rows(self, rows: list[dict[str, Any]]) -> None:
        ordered_rows = sorted(rows, key=lambda row: self._natural_id_key(str(row.get("scene_id", ""))))
        for index, row in enumerate(ordered_rows, start=1):
            scene_goal = str(row.get("scene_goal", "")).strip() or f"Scene {index}"
            self.conn.execute(
                """
                INSERT INTO scenes (
                    scene_id,
                    scene_index,
                    scene_name,
                    scene_goal,
                    scene_description,
                    scene_summary,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("scene_id", f"scene_{index}")),
                    int(row.get("scene_index", index) or index),
                    str(row.get("scene_name", "")).strip() or scene_goal,
                    scene_goal,
                    str(row.get("scene_description", "")).strip(),
                    str(row.get("scene_summary", "")).strip(),
                    str(row.get("status", "pending")).strip() or "pending",
                ),
            )

    def _copy_plots_rows(self, rows: list[dict[str, Any]]) -> None:
        ordered_rows = sorted(rows, key=lambda row: self._natural_id_key(str(row.get("plot_id", ""))))
        for index, row in enumerate(ordered_rows, start=1):
            plot_goal = str(row.get("plot_goal", "")).strip() or f"Plot {index}"
            progress = row.get("progress", 0.0)
            try:
                progress_value = float(progress)
            except Exception:
                progress_value = 0.0
            self.conn.execute(
                """
                INSERT INTO plots (
                    plot_id,
                    scene_id,
                    plot_index,
                    plot_name,
                    plot_goal,
                    raw_text,
                    status,
                    progress
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("plot_id", f"plot_{index}")),
                    str(row.get("scene_id", "")),
                    int(row.get("plot_index", index) or index),
                    str(row.get("plot_name", "")).strip() or plot_goal,
                    plot_goal,
                    str(row.get("raw_text", "")).strip(),
                    str(row.get("status", "pending")).strip() or "pending",
                    progress_value,
                ),
            )

    def _copy_knowledge_rows(self, rows: list[dict[str, Any]]) -> None:
        for index, row in enumerate(rows, start=1):
            self.conn.execute(
                """
                INSERT INTO knowledge_base (
                    knowledge_id,
                    knowledge_type,
                    title,
                    content
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(row.get("knowledge_id", f"knowledge_{index}")),
                    str(row.get("knowledge_type", "other")).strip() or "other",
                    str(row.get("title", "")).strip() or f"Knowledge {index}",
                    str(row.get("content", "")).strip(),
                ),
            )

    def _table_columns(self, table: str) -> list[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [str(row["name"]) for row in rows]

    def _ensure_column(self, table: str, column_def: str) -> None:
        existing_columns = set(self._table_columns(table))
        column_name = column_def.split()[0]
        if column_name not in existing_columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

    def close(self) -> None:
        self.conn.close()

    def initial_story_snapshot_path(self) -> Path:
        db_file = Path(self.db_path)
        suffix = db_file.suffix or ".db"
        stem = db_file.stem if db_file.suffix else db_file.name
        return db_file.with_name(f"{stem}.initial_story_snapshot{suffix}")

    def has_initial_story_snapshot(self) -> bool:
        return self.initial_story_snapshot_path().exists()

    def delete_initial_story_snapshot(self) -> None:
        snapshot_path = self.initial_story_snapshot_path()
        if not snapshot_path.exists():
            return
        for _ in range(10):
            try:
                snapshot_path.unlink()
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(0.1)

    def save_initial_story_snapshot(self) -> str:
        snapshot_path = self.initial_story_snapshot_path()
        self.conn.commit()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(snapshot_path)) as snapshot_conn:
            self.conn.backup(snapshot_conn)
        return str(snapshot_path)

    def restore_initial_story_snapshot(self) -> str:
        snapshot_path = self.initial_story_snapshot_path()
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Initial story snapshot not found: {snapshot_path}")
        self.conn.commit()
        with sqlite3.connect(str(snapshot_path)) as snapshot_conn:
            snapshot_conn.backup(self.conn)
        self.conn.commit()
        return str(snapshot_path)

    def reset_story_data(self) -> None:
        # Delete child records before parents to satisfy FK constraints.
        with self.conn:
            self.conn.execute("DELETE FROM plots")
            self.conn.execute("DELETE FROM scenes")
            self.conn.execute("DELETE FROM memory")
            self.conn.execute("DELETE FROM summaries")
            self.conn.execute("DELETE FROM knowledge_base")
            self.conn.execute(
                """
                UPDATE system_state
                SET current_scene_id = '',
                    current_plot_id = '',
                    plot_progress = 0.0,
                    scene_progress = 0.0,
                    current_scene_intro = '',
                    navigation_state_json = '{}',
                    current_visit_id = 0
                WHERE id = 1
                """
            )

    def insert_scenes(self, scenes: list[dict[str, Any]]) -> None:
        for scene_index, scene in enumerate(scenes, start=1):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO scenes (
                    scene_id,
                    scene_index,
                    scene_name,
                    scene_goal,
                    scene_description,
                    scene_summary,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(scene["scene_id"]),
                    int(scene.get("scene_index", scene_index) or scene_index),
                    str(scene.get("scene_name", "")).strip() or str(scene.get("scene_goal", "")),
                    str(scene.get("scene_goal", "")).strip(),
                    str(scene.get("scene_description", "")).strip(),
                    str(scene.get("scene_summary", "")).strip(),
                    str(scene.get("status", "pending")).strip() or "pending",
                ),
            )

            for plot_index, plot in enumerate(scene.get("plots", []), start=1):
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO plots (
                        plot_id,
                        scene_id,
                        plot_index,
                        plot_name,
                        plot_goal,
                        raw_text,
                        status,
                        progress
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(plot["plot_id"]),
                        str(scene["scene_id"]),
                        int(plot.get("plot_index", plot_index) or plot_index),
                        str(plot.get("plot_name", "")).strip() or str(plot.get("plot_goal", "")),
                        str(plot.get("plot_goal", "")).strip(),
                        str(plot.get("raw_text", "")),
                        str(plot.get("status", "pending")).strip() or "pending",
                        float(plot.get("progress", 0.0) or 0.0),
                    ),
                )
        self.conn.commit()

    def insert_knowledge(self, knowledge_items: list[dict[str, Any]]) -> None:
        for index, item in enumerate(knowledge_items, start=1):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_base (
                    knowledge_id,
                    knowledge_type,
                    title,
                    content
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(item.get("knowledge_id", f"knowledge_{index}")),
                    str(item.get("knowledge_type", "other")).strip() or "other",
                    str(item.get("title", "")).strip() or f"Knowledge {index}",
                    str(item.get("content", "")).strip(),
                ),
            )
        self.conn.commit()

    def list_knowledge(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM knowledge_base ORDER BY knowledge_id").fetchall()]

    def get_knowledge_by_type(self, knowledge_type: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM knowledge_base WHERE knowledge_type = ? ORDER BY knowledge_id",
                (knowledge_type,),
            ).fetchall()
        ]

    def list_scenes(self) -> list[dict[str, Any]]:
        scenes = [dict(row) for row in self.conn.execute("SELECT * FROM scenes ORDER BY scene_index, scene_id").fetchall()]
        if not scenes:
            return []

        plot_rows = [dict(row) for row in self.conn.execute("SELECT * FROM plots ORDER BY scene_id, plot_index, plot_id").fetchall()]
        plots_by_scene: dict[str, list[dict[str, Any]]] = {}
        for plot in plot_rows:
            plots_by_scene.setdefault(str(plot.get("scene_id", "")), []).append(plot)

        hydrated: list[dict[str, Any]] = []
        for scene in scenes:
            scene["plots"] = [dict(plot) for plot in plots_by_scene.get(str(scene.get("scene_id", "")), [])]
            hydrated.append(scene)
        return hydrated

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        return next((scene for scene in self.list_scenes() if str(scene.get("scene_id", "")) == scene_id), None)

    def get_plot(self, plot_id: str) -> dict[str, Any] | None:
        for scene in self.list_scenes():
            for plot in scene.get("plots", []):
                if str(plot.get("plot_id", "")) == plot_id:
                    return plot
        return None

    def update_plot(self, plot_id: str, status: str | None = None, progress: float | None = None) -> None:
        if status is not None:
            self.conn.execute("UPDATE plots SET status = ? WHERE plot_id = ?", (status, plot_id))
        if progress is not None:
            self.conn.execute("UPDATE plots SET progress = ? WHERE plot_id = ?", (float(progress), plot_id))
        self.conn.commit()

    def update_scene(self, scene_id: str, updates: dict[str, Any]) -> None:
        allowed_columns = {"scene_index", "scene_name", "scene_goal", "scene_description", "scene_summary", "status"}
        for key, value in updates.items():
            if key not in allowed_columns:
                continue
            self.conn.execute(f"UPDATE scenes SET {key} = ? WHERE scene_id = ?", (value, scene_id))
        self.conn.commit()

    def append_memory(
        self,
        scene_id: str,
        plot_id: str,
        user: str,
        agent: str,
        visit_id: int = 0,
        *,
        turn_state: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO memory(scene_id, plot_id, user, agent, turn_state_json, visit_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                scene_id,
                plot_id,
                user,
                agent,
                json.dumps(turn_state or {}, ensure_ascii=False),
                int(visit_id),
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def get_recent_turns(self, scene_id: str, plot_id: str, limit: int = 12, *, visit_id: int | None = None) -> list[dict[str, Any]]:
        if visit_id is None:
            rows = self.conn.execute(
                "SELECT user, agent, turn_state_json, timestamp, visit_id FROM memory WHERE scene_id = ? AND plot_id = ? ORDER BY id DESC LIMIT ?",
                (scene_id, plot_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT user, agent, turn_state_json, timestamp, visit_id FROM memory WHERE scene_id = ? AND plot_id = ? AND visit_id = ? ORDER BY id DESC LIMIT ?",
                (scene_id, plot_id, int(visit_id), limit),
            ).fetchall()
        hydrated_rows: list[dict[str, Any]] = []
        for row in reversed(rows):
            record = dict(row)
            record["turn_state"] = _parse_json_object(record.get("turn_state_json", "{}"))
            hydrated_rows.append(record)
        return hydrated_rows

    def get_global_recent_turns(self, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT user, agent, turn_state_json, timestamp, visit_id FROM memory ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        hydrated_rows: list[dict[str, Any]] = []
        for row in reversed(rows):
            record = dict(row)
            record["turn_state"] = _parse_json_object(record.get("turn_state_json", "{}"))
            hydrated_rows.append(record)
        return hydrated_rows

    def has_global_opening(self, marker: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM memory WHERE user = ? LIMIT 1", (marker,)).fetchone()
        return row is not None

    def save_summary(self, summary_type: str, content: str, scene_id: str = "", plot_id: str = "") -> None:
        self.conn.execute(
            """
            INSERT INTO summaries(summary_type, scene_id, plot_id, content)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(summary_type, scene_id, plot_id) DO UPDATE SET content = excluded.content
            """,
            (summary_type, scene_id, plot_id, content),
        )
        self.conn.commit()

    def get_summary(self, summary_type: str, scene_id: str = "", plot_id: str = "") -> str:
        row = self.conn.execute(
            "SELECT content FROM summaries WHERE summary_type = ? AND scene_id = ? AND plot_id = ?",
            (summary_type, scene_id, plot_id),
        ).fetchone()
        return row["content"] if row else ""

    def get_system_state(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
        state = dict(row)
        state["player_profile"] = json.loads(state.get("player_profile", "{}") or "{}")
        state["navigation_state_json"] = state.get("navigation_state_json", "{}")
        state["navigation_state"] = _parse_json_object(state.get("navigation_state_json", "{}"))
        return state

    def update_system_state(self, updates: dict[str, Any]) -> None:
        current = self.get_system_state()
        normalized = dict(updates)
        if "player_profile" in normalized and isinstance(normalized["player_profile"], dict):
            normalized["player_profile"] = json.dumps(normalized["player_profile"], ensure_ascii=False)
        elif "player_profile" not in normalized:
            normalized["player_profile"] = json.dumps(current.get("player_profile", {}), ensure_ascii=False)

        if "navigation_state" in normalized and isinstance(normalized["navigation_state"], dict):
            normalized["navigation_state_json"] = json.dumps(normalized.pop("navigation_state"), ensure_ascii=False)
        elif "navigation_state_json" not in normalized:
            normalized["navigation_state_json"] = json.dumps(current.get("navigation_state", {}), ensure_ascii=False)

        valid_columns = set(self._table_columns("system_state"))
        for key, value in normalized.items():
            if key not in valid_columns:
                continue
            self.conn.execute(f"UPDATE system_state SET {key} = ? WHERE id = 1", (value,))
        self.conn.commit()

    def get_player_profile(self) -> dict[str, Any]:
        return self.get_system_state().get("player_profile", {})

    def save_player_profile(self, profile: dict[str, Any]) -> None:
        self.update_system_state({"player_profile": profile})

    @staticmethod
    def _natural_id_key(value: str) -> tuple[Any, ...]:
        parts = re.split(r"(\d+)", (value or "").lower())
        key: list[Any] = []
        for part in parts:
            if not part:
                continue
            key.append(int(part) if part.isdigit() else part)
        return tuple(key)
