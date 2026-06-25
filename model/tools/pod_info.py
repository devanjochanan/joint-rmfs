import sqlite3

from src.rmfs.runtime_io.detail_db import (
    commit,
    connect,
    debug_log,
    execute,
    is_detail_db_enabled,
)

TS = None

def initialize_pod_info_table(timestamp: str, db_path="warehouse.db"):
    global TS
    TS = timestamp   
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        CREATE TABLE IF NOT EXISTS pod_info_{TS} (
            id TEXT PRIMARY KEY,
            x REAL,
            y REAL,
            is_idle INTEGER
        )
    """)

    commit(conn)
    conn.close()
    debug_log("pod_info table initialized.")

def clear_pod_info(db_path="warehouse.db"):
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"DELETE FROM pod_info_{TS}")

    commit(conn)
    conn.close()
    debug_log("All pod info rows have been cleared.")

def upsert_pod_location(pod_id: str, x: float, y: float, db_path="warehouse.db"):
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        INSERT INTO pod_info_{TS} (id, x, y)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET x=excluded.x, y=excluded.y
    """, (pod_id, x, y))

    commit(conn)
    conn.close()
    debug_log(f"Pod {pod_id} location set to ({x}, {y}).")

def upsert_pod_idle(pod_id: str, is_idle: bool, db_path="warehouse.db"):
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        INSERT INTO pod_info_{TS} (id, is_idle)
        VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE SET is_idle=excluded.is_idle
    """, (pod_id, int(is_idle)))

    commit(conn)
    conn.close()
    debug_log(f"Pod {pod_id} idle status set to {is_idle}.")

def get_pod_info(pod_id: str, db_path="warehouse.db"):
    if not is_detail_db_enabled():
        return None

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"SELECT x, y, is_idle FROM pod_info_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        debug_log(f"Pod {pod_id} info: location=({result[0]}, {result[1]}), is_idle={bool(result[2])}")
        return result
    else:
        debug_log(f"Pod {pod_id} not found.")
        return None

def get_pod_location(pod_id: str, db_path="warehouse.db"):
    if not is_detail_db_enabled():
        return None

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"SELECT x, y FROM pod_info_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        debug_log(f"Pod {pod_id} is at location ({result[0]}, {result[1]})")
        return result
    else:
        debug_log(f"Pod {pod_id} not found.")
        return None

def get_pod_idle(pod_id: str, db_path="warehouse.db"):
    if not is_detail_db_enabled():
        return None

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"SELECT is_idle FROM pod_info_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()
    conn.close()

    if result is not None:
        debug_log(f"Pod {pod_id} is_idle = {bool(result[0])}")
        return bool(result[0])
    else:
        debug_log(f"Pod {pod_id} not found.")
        return None
