import sqlite3

from src.rmfs.runtime_io.detail_db import (
    commit,
    connect,
    debug_log,
    execute,
    is_detail_db_enabled,
)

TS = None  # Same global timestamp

def initialize_pre_assign_table(timestamp: str, db_path="warehouse.db"):
    """
    Create the pre_assign table if it doesn't exist.
    """
    global TS
    TS = timestamp
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        CREATE TABLE IF NOT EXISTS pre_assign_{TS} (
            time REAL,
            current TEXT,
            order_id TEXT,
            score REAL,
            bestpicker TEXT,
            bestscore REAL
        )
    """)

    commit(conn)
    conn.close()
    debug_log("pre_assign table initialized.")

def clear_pre_assign_table(db_path="warehouse.db"):
    """
    Delete all rows from the pre_assign table.
    """
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"DELETE FROM pre_assign_{TS}")
    commit(conn)
    conn.close()
    debug_log("All pre_assign records have been cleared.")

def insert_pre_assign(
        time: float,
        current: str,
        order: str,
        score: float,
        bestpicker: str,
        bestscore: float,
        db_path: str = "warehouse.db"):
    """
    Insert a new pre_assign record.
    """
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        INSERT INTO pre_assign_{TS} 
        (time, current, order_id, score, bestpicker, bestscore)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (time, current, order, score, bestpicker, bestscore))

    commit(conn)
    conn.close()
    debug_log(f"Inserted pre_assign record: current={current}, order_id={order}, score={score}")
