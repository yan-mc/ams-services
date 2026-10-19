"""SQLite 持久化（stdlib sqlite3 + WAL）。

表：
- jobs        任务主体（状态机/进度/参数/错误/引擎/GPU 锁）
- job_events  SSE 事件（同时承担任务日志环的持久化）
- artifacts   产物索引（文件在磁盘，库只存元数据）

设计要点：
- 单连接 + RLock（Gateway 进程内并发量很低，M0 不需要连接池）；
- JSON 字段统一存 TEXT，读时解码；
- 服务重启后历史任务可查（M0.3 验收要求）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOB_STATUSES = {
    "queued",
    "running",
    "stopping",
    "succeeded",
    "failed",
    "cancelled",
}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

JOB_UPDATE_COLUMNS = {
    "status",
    "progress",
    "stage",
    "started_at",
    "finished_at",
    "error_json",
    "engine_json",
    "gpu_lock_json",
    "cancel_requested",
    "resumable",
}

SCHEMA_VERSION = 1


def utc_now() -> str:
    """当前 UTC 时间（ISO 8601，秒级）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:16]


def new_artifact_id() -> str:
    return "art_" + uuid.uuid4().hex[:16]


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class Database:
    """Gateway 的 SQLite 封装（线程安全，单文件）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()

    # ------------------------------------------------------------------ 基础
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    project_id TEXT,
                    agent_session_id TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL,
                    error_json TEXT,
                    engine_json TEXT NOT NULL DEFAULT '{}',
                    gpu_lock_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    resumable INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'file',
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    bytes INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT,
                    created_at TEXT NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_job ON artifacts(job_id);
                """
            )
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            except sqlite3.Error:
                pass
            self._conn.close()

    # ------------------------------------------------------------------ 行转换
    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> dict:
        job = dict(row)
        job["params"] = _loads(job.pop("params_json", None), {})
        job["error"] = _loads(job.pop("error_json", None), None)
        job["engine"] = _loads(job.pop("engine_json", None), {})
        job["gpu_lock"] = _loads(job.pop("gpu_lock_json", None), None)
        job["cancel_requested"] = bool(job.get("cancel_requested", 0))
        job["resumable"] = bool(job.get("resumable", 0))
        return job

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "job_id": row["job_id"],
            "kind": row["kind"],
            "ts": row["ts"],
            "data": _loads(row["data_json"], {}),
        }

    # ------------------------------------------------------------------ jobs
    def create_job(
        self,
        *,
        job_type: str,
        params: dict,
        project_id: str | None = None,
        agent_session_id: str | None = None,
        engine: dict | None = None,
        gpu_waiting: bool = False,
    ) -> dict:
        now = utc_now()
        job_id = new_job_id()
        gpu_lock = {"state": "waiting", "reason": "等待 GPU 工作线程"} if gpu_waiting else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, type, status, progress, params_json, project_id,
                    agent_session_id, created_at, updated_at, engine_json, gpu_lock_json
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    _dumps(params),
                    project_id,
                    agent_session_id,
                    now,
                    now,
                    _dumps(engine or {}),
                    _dumps(gpu_lock) if gpu_lock else None,
                ),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_from_row(row)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        project_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        clauses: list[str] = []
        values: list[Any] = []
        if status:
            clauses.append("status = ?")
            values.append(status)
        if job_type:
            clauses.append("type = ?")
            values.append(job_type)
        if project_id:
            clauses.append("project_id = ?")
            values.append(project_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            total = int(
                self._conn.execute("SELECT COUNT(*) FROM jobs" + where, tuple(values)).fetchone()[0]
            )
            rows = self._conn.execute(
                "SELECT * FROM jobs" + where + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                tuple(values) + (limit, offset),
            ).fetchall()
        return [self._job_from_row(row) for row in rows], total

    def update_job(self, job_id: str, **fields: Any) -> dict | None:
        unknown = set(fields) - JOB_UPDATE_COLUMNS
        if unknown:
            raise ValueError(f"不允许更新的列：{sorted(unknown)}")
        if not fields:
            return self.get_job(job_id)
        fields = dict(fields)
        for key in ("error_json", "engine_json", "gpu_lock_json"):
            if key in fields and not isinstance(fields[key], (str, type(None))):
                fields[key] = _dumps(fields[key])
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [job_id]
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
            self._conn.commit()
        return self.get_job(job_id)

    def try_start_job(self, job_id: str, *, gpu_lock: dict | None = None) -> dict | None:
        """原子领取任务：仅当状态仍为 queued 时置为 running，避免与取消竞争。"""
        now = utc_now()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?, updated_at = ?, progress = 0,
                    stage = 'starting', error_json = NULL, gpu_lock_json = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, _dumps(gpu_lock) if gpu_lock else None, job_id),
            )
            changed = int(cursor.rowcount or 0)
            self._conn.commit()
        return self.get_job(job_id) if changed else None

    def try_cancel_queued(self, job_id: str) -> dict | None:
        """原子取消排队中任务：仅当状态仍为 queued 时置为 cancelled。"""
        now = utc_now()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', finished_at = ?, updated_at = ?, stage = 'cancelled',
                    gpu_lock_json = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, job_id),
            )
            changed = int(cursor.rowcount or 0)
            self._conn.commit()
        return self.get_job(job_id) if changed else None

    def queued_job_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [row["id"] for row in rows]

    def recover_on_startup(self) -> dict:
        """服务重启恢复：running/stopping 视为失败（M0 不做断点续跑）；queued 重新入队。"""
        now = utc_now()
        error = {
            "code": "GATEWAY_RESTARTED",
            "message": "Gateway 重启导致任务中断；M0 不支持断点续跑，请重试",
        }
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', finished_at = ?, updated_at = ?, error_json = ?
                WHERE status IN ('running', 'stopping')
                """,
                (now, now, _dumps(error)),
            )
            interrupted = int(cursor.rowcount or 0)
            self._conn.commit()
            queued = self.queued_job_ids()
        return {"interrupted": interrupted, "requeued": len(queued), "queued_ids": queued}

    def job_counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        counts = {row["status"]: int(row["n"]) for row in rows}
        for status in JOB_STATUSES:
            counts.setdefault(status, 0)
        return counts

    # ------------------------------------------------------------------ events
    def add_event(
        self,
        job_id: str,
        kind: str,
        data: dict,
        *,
        max_events_per_job: int = 2000,
    ) -> dict:
        now = utc_now()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO job_events (job_id, kind, ts, data_json) VALUES (?, ?, ?, ?)",
                (job_id, kind, now, _dumps(data)),
            )
            event_id = int(cursor.lastrowid)
            # 日志环：每个任务最多保留 max_events_per_job 条，超出删最旧
            self._conn.execute(
                """
                DELETE FROM job_events WHERE job_id = ? AND id <= (
                    SELECT id FROM job_events WHERE job_id = ?
                    ORDER BY id DESC LIMIT 1 OFFSET ?
                )
                """,
                (job_id, job_id, max_events_per_job),
            )
            self._conn.commit()
        return {"id": event_id, "job_id": job_id, "kind": kind, "ts": now, "data": data}

    def get_events(self, job_id: str, *, after_id: int = 0, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (job_id, after_id, limit),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    # ------------------------------------------------------------------ artifacts
    def add_artifact(
        self,
        *,
        job_id: str,
        name: str,
        path: str,
        kind: str = "file",
        size: int = 0,
        sha256: str | None = None,
        meta: dict | None = None,
    ) -> dict:
        artifact_id = new_artifact_id()
        now = utc_now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO artifacts (id, job_id, kind, name, path, bytes, sha256, created_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, job_id, kind, name, path, int(size), sha256, now, _dumps(meta or {})),
            )
            self._conn.commit()
        return {
            "id": artifact_id,
            "job_id": job_id,
            "kind": kind,
            "name": name,
            "path": path,
            "bytes": int(size),
            "sha256": sha256,
            "created_at": now,
            "meta": meta or {},
        }

    def list_artifacts(self, job_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE job_id = ? ORDER BY created_at ASC, id ASC",
                (job_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["meta"] = _loads(item.pop("meta_json", None), {})
            result.append(item)
        return result

    def get_artifact(self, artifact_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["meta"] = _loads(item.pop("meta_json", None), {})
        return item
