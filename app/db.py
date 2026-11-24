"""SQLite 存档与合并维护。

表结构
------
branch_reports
    各馆按 (月份, 资源号, 馆号) 保存的现行 HLL 摘要（128 个寄存器）。
    换报即整行覆盖，撤回即删除。

register_counts
    按 (月份, 资源号, 寄存器位置, 取值) 维护该取值当前有多少个馆贡献，
    即每个位置的完整取值直方图。它是各馆现行摘要逐位置最大值的增量
    表示：换报/撤回时只需读取本馆旧摘要做差分，按主键增减对应取值的
    计数，再以 GROUP BY 位置取 MAX(value) 得到合并值，全程不遍历其他
    馆的摘要。最大值馆撤回或降值时，次大值可直接从直方图恢复。

merged_summaries
    按 (月份, 资源号) 保存的合并摘要与月度读者规模预估（estimate），
    与馆报在同一事务内一并生效；预估失败或落库失败即整体回滚，
    库内仍保留原摘要与原预估。旧行可能没有预估字段（estimate 为 NULL），
    读取时按寄存器即时补出，页面与下载 JSON 看到的始终是同一版数值。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from .errors import NotFoundError
from .hll import (
    REGISTER_COUNT,
    estimate_readers,
    validate_identifier,
    validate_month,
    validate_registers,
)

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS branch_reports (
    month      TEXT NOT NULL,
    resource   TEXT NOT NULL,
    branch     TEXT NOT NULL,
    registers  TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (month, resource, branch)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS register_counts (
    month     TEXT NOT NULL,
    resource  TEXT NOT NULL,
    position  INTEGER NOT NULL,
    value     INTEGER NOT NULL,
    cnt       INTEGER NOT NULL,
    PRIMARY KEY (month, resource, position, value)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS merged_summaries (
    month      TEXT NOT NULL,
    resource   TEXT NOT NULL,
    registers  TEXT NOT NULL,
    estimate   REAL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (month, resource)
) WITHOUT ROWID;
"""

_BUSY_RETRIES = 8
_BUSY_BACKOFF = 0.02  # 秒，指数退避：0.02, 0.04, ...


class Database:
    def __init__(self, path: str):
        self.path = path
        self._init_lock = threading.Lock()
        self._initialized = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,  # 事务完全由显式 BEGIN/COMMIT 控制
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            conn = self._connect()
            try:
                conn.executescript(SCHEMA)
                self._migrate(conn)
            finally:
                conn.close()
            self._initialized = True

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """旧库迁移：merged_summaries 可能还没有 estimate 列，按需补上。

        旧行的 estimate 保持 NULL，读取时按寄存器即时补出；
        下一次换报/撤回会在事务内写入新预估。
        """
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(merged_summaries)").fetchall()
        }
        if "estimate" not in columns:
            conn.execute("ALTER TABLE merged_summaries ADD COLUMN estimate REAL")

    # -- 事务工具 -------------------------------------------------------

    @staticmethod
    def _commit(conn: sqlite3.Connection) -> None:
        """单独抽出便于测试注入提交失败；正常路径即 SQLite 提交。"""
        conn.commit()

    def _transaction(self, conn: sqlite3.Connection) -> sqlite3.Cursor:
        """开启写事务，遇锁按退避重试；BEGIN IMMEDIATE 保证写操作串行化。"""
        last_error: Exception | None = None
        for attempt in range(_BUSY_RETRIES):
            try:
                return conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:  # database is locked / busy
                last_error = exc
                time.sleep(_BUSY_BACKOFF * (2**attempt))
        assert last_error is not None
        raise last_error

    # -- 内部：基于本馆旧摘要的差分更新 ----------------------------------

    @staticmethod
    def _apply_deltas(
        conn: sqlite3.Connection,
        month: str,
        resource: str,
        deltas: dict[int, dict[int, int]],
    ) -> None:
        """对逐位置取值直方图施加 {位置: {取值: 增减量}} 的差分。

        只按 (月份,资源,位置,取值) 主键 upsert/delete，不读取其他馆摘要。
        """
        for position, changes in deltas.items():
            for value, delta in changes.items():
                if value == 0:
                    continue  # 取值 0 不占用计数，全馆无贡献时该位置自然为 0
                row = conn.execute(
                    "SELECT cnt FROM register_counts "
                    "WHERE month = ? AND resource = ? AND position = ? AND value = ?",
                    (month, resource, position, value),
                ).fetchone()
                new_cnt = (row["cnt"] if row is not None else 0) + delta
                if new_cnt <= 0:
                    conn.execute(
                        "DELETE FROM register_counts "
                        "WHERE month = ? AND resource = ? AND position = ? AND value = ?",
                        (month, resource, position, value),
                    )
                elif row is None:
                    conn.execute(
                        "INSERT INTO register_counts "
                        "(month, resource, position, value, cnt) VALUES (?, ?, ?, ?, ?)",
                        (month, resource, position, value, new_cnt),
                    )
                else:
                    conn.execute(
                        "UPDATE register_counts SET cnt = ? "
                        "WHERE month = ? AND resource = ? AND position = ? AND value = ?",
                        (new_cnt, month, resource, position, value),
                    )

    @staticmethod
    def _refresh_merged(
        conn: sqlite3.Connection, month: str, resource: str
    ) -> tuple[list[int], float]:
        """从取值直方图聚合（逐位置 MAX），重算预估并落盘合并摘要。

        预估与摘要在同一事务内一并生效：预估计算或落库失败都会向上抛出，
        整次变更回滚到原摘要和原预估。
        """
        merged = [0] * REGISTER_COUNT
        rows = conn.execute(
            "SELECT position, MAX(value) AS value FROM register_counts "
            "WHERE month = ? AND resource = ? GROUP BY position",
            (month, resource),
        ).fetchall()
        for row in rows:
            merged[row["position"]] = row["value"]
        estimate = estimate_readers(merged)
        payload = json.dumps(merged, separators=(",", ":"))
        conn.execute(
            "INSERT INTO merged_summaries (month, resource, registers, estimate, updated_at) "
            "VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
            "ON CONFLICT (month, resource) DO UPDATE SET "
            "registers = excluded.registers, estimate = excluded.estimate, "
            "updated_at = excluded.updated_at",
            (month, resource, payload, estimate),
        )
        return merged, estimate

    # -- 对外操作 --------------------------------------------------------

    def upsert_report(
        self, month: str, resource: str, branch: str, registers: list[int]
    ) -> dict[str, Any]:
        """上传或换报：整行覆盖本馆摘要，并增量维护合并结果。"""
        month = validate_month(month)
        resource = validate_identifier("resource", resource)
        branch = validate_identifier("branch", branch)
        registers = validate_registers(registers)

        conn = self._connect()
        try:
            self._transaction(conn)
            try:
                old_row = conn.execute(
                    "SELECT registers FROM branch_reports "
                    "WHERE month = ? AND resource = ? AND branch = ?",
                    (month, resource, branch),
                ).fetchone()
                old_registers = (
                    json.loads(old_row["registers"]) if old_row is not None else None
                )

                deltas: dict[int, dict[int, int]] = {}
                for position, new_value in enumerate(registers):
                    old_value = (
                        old_registers[position] if old_registers is not None else 0
                    )
                    if old_value == new_value:
                        continue
                    change: dict[int, int] = {}
                    if old_value:
                        change[old_value] = -1
                    if new_value:
                        change[new_value] = change.get(new_value, 0) + 1
                    if change:
                        deltas[position] = change
                if deltas:
                    self._apply_deltas(conn, month, resource, deltas)

                conn.execute(
                    "INSERT INTO branch_reports (month, resource, branch, registers, updated_at) "
                    "VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
                    "ON CONFLICT (month, resource, branch) DO UPDATE SET "
                    "registers = excluded.registers, updated_at = excluded.updated_at",
                    (
                        month,
                        resource,
                        branch,
                        json.dumps(registers, separators=(",", ":")),
                    ),
                )
                merged, estimate = self._refresh_merged(conn, month, resource)
                self._commit(conn)
            except BaseException:
                conn.rollback()
                raise
        finally:
            conn.close()

        return {
            "month": month,
            "resource": resource,
            "branch": branch,
            "registers": registers,
            "merged": merged,
            "estimate": estimate,
            "replaced": old_registers is not None,
        }

    def withdraw_report(
        self, month: str, resource: str, branch: str
    ) -> dict[str, Any]:
        """撤回本馆摘要：删除本馆行并增量维护合并结果，不触碰其他馆摘要。"""
        month = validate_month(month)
        resource = validate_identifier("resource", resource)
        branch = validate_identifier("branch", branch)

        conn = self._connect()
        try:
            self._transaction(conn)
            try:
                old_row = conn.execute(
                    "SELECT registers FROM branch_reports "
                    "WHERE month = ? AND resource = ? AND branch = ?",
                    (month, resource, branch),
                ).fetchone()
                if old_row is None:
                    raise NotFoundError(
                        f"馆 {branch} 在 {month} 对资源 {resource} 没有已保存的报送，无法撤回"
                    )
                old_registers = json.loads(old_row["registers"])

                deltas: dict[int, dict[int, int]] = {}
                for position, old_value in enumerate(old_registers):
                    if old_value:
                        deltas[position] = {old_value: -1}
                if deltas:
                    self._apply_deltas(conn, month, resource, deltas)

                conn.execute(
                    "DELETE FROM branch_reports "
                    "WHERE month = ? AND resource = ? AND branch = ?",
                    (month, resource, branch),
                )
                merged, estimate = self._refresh_merged(conn, month, resource)
                self._commit(conn)
            except BaseException:
                conn.rollback()
                raise
        finally:
            conn.close()

        return {
            "month": month,
            "resource": resource,
            "branch": branch,
            "merged": merged,
            "estimate": estimate,
        }

    def get_merged(self, month: str, resource: str) -> dict[str, Any]:
        """读取已保存的合并摘要与预估（无任何报送记录时 404）。

        旧摘要可能没有预估字段（estimate 为 NULL），按寄存器即时补出，
        保证页面与下载 JSON 看到的预估口径一致。
        """
        month = validate_month(month)
        resource = validate_identifier("resource", resource)

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT registers, estimate, updated_at FROM merged_summaries "
                "WHERE month = ? AND resource = ?",
                (month, resource),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise NotFoundError(f"{month} 资源 {resource} 尚无任何报送，没有合并摘要")
        registers = json.loads(row["registers"])
        return {
            "month": month,
            "resource": resource,
            "registers": registers,
            "estimate": self._stored_or_fresh_estimate(registers, row["estimate"]),
            "updated_at": row["updated_at"],
        }

    def list_merged(self) -> list[dict[str, Any]]:
        """列出全部已保存的合并摘要预估，供联盟页面展示（按月份倒序、资源号排序）。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT month, resource, registers, estimate, updated_at "
                "FROM merged_summaries ORDER BY month DESC, resource",
            ).fetchall()
        finally:
            conn.close()
        items = []
        for row in rows:
            registers = json.loads(row["registers"])
            items.append(
                {
                    "month": row["month"],
                    "resource": row["resource"],
                    "nonzero_registers": sum(1 for value in registers if value),
                    "estimate": self._stored_or_fresh_estimate(
                        registers, row["estimate"]
                    ),
                    "updated_at": row["updated_at"],
                }
            )
        return items

    @staticmethod
    def _stored_or_fresh_estimate(registers: list[int], estimate: Any) -> float:
        """优先使用已落库的预估；旧摘要没有预估字段时按寄存器即时补出。"""
        if estimate is None:
            return estimate_readers(registers)
        return float(estimate)

    def get_report(
        self, month: str, resource: str, branch: str
    ) -> dict[str, Any]:
        """读取某馆的现行报送（供换报前确认，可选辅助接口）。"""
        month = validate_month(month)
        resource = validate_identifier("resource", resource)
        branch = validate_identifier("branch", branch)

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT registers, updated_at FROM branch_reports "
                "WHERE month = ? AND resource = ? AND branch = ?",
                (month, resource, branch),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise NotFoundError(
                f"馆 {branch} 在 {month} 对资源 {resource} 没有已保存的报送"
            )
        return {
            "month": month,
            "resource": resource,
            "branch": branch,
            "registers": json.loads(row["registers"]),
            "updated_at": row["updated_at"],
        }
