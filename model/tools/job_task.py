import sqlite3

from src.rmfs.runtime_io.detail_db import (
    commit,
    connect,
    debug_log,
    execute,
    is_detail_db_enabled,
)

TS = None

def initialize_job_task_table(timestamp: str, db_path="warehouse.db"):
    """
    Create the job_task table if it doesn't exist.
    """
    global TS
    TS = timestamp
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        CREATE TABLE IF NOT EXISTS job_task_{TS} (
            pod_id INTEGER,
            order_id INTEGER,
            sku INTEGER,
            qty INTEGER,
            assigned_station TEXT,
            pod_assigned_time REAL,
            status TEXT,
            finish_time REAL,
            PRIMARY KEY (pod_id, order_id, sku, qty)
        )
    """)

    commit(conn)
    conn.close()
    debug_log("job_task table initialized.")

def clear_job_task_table(db_path="warehouse.db"):
    """
    Delete all rows from the job_task table.
    """
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"DELETE FROM job_task_{TS}")

    commit(conn)
    conn.close()
    debug_log("All job tasks have been cleared.")

def upsert_job_task(
        pod_id: int,
        order_id: int,
        sku: int,
        qty: int,
        assigned_station: str = None,
        pod_assigned_time: float = None,
        status: str = None,
        finish_time: float = None,
        db_path: str = "warehouse.db"):
    """
    Insert or update a job task based on (pod_id, order_id, sku, qty) as composite key.
    Only non-None fields will be updated on conflict.
    """
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    # Check if the entry exists
    execute(cursor, f"""
        SELECT * FROM job_task_{TS}
        WHERE pod_id = ? AND order_id = ? AND sku = ? AND qty = ?
    """, (pod_id, order_id, sku, qty))
    existing = cursor.fetchone()

    if existing:
        # Prepare dynamic update
        fields = []
        values = []

        if assigned_station is not None:
            fields.append("assigned_station = ?")
            values.append(assigned_station)
        if pod_assigned_time is not None:
            fields.append("pod_assigned_time = ?")
            values.append(pod_assigned_time)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if finish_time is not None:
            fields.append("finish_time = ?")
            values.append(finish_time)

        if fields:
            query = f"""
                UPDATE job_task_{TS} SET {', '.join(fields)}
                WHERE pod_id = ? AND order_id = ? AND sku = ? AND qty = ?
            """
            values.extend([pod_id, order_id, sku, qty])
            execute(cursor, query, tuple(values))
    else:
        # Insert new record
        execute(cursor, f"""
            INSERT INTO job_task_{TS} (pod_id, order_id, sku, qty, assigned_station, pod_assigned_time, status, finish_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (pod_id, order_id, sku, qty, assigned_station, pod_assigned_time, status, finish_time))

    commit(conn)
    conn.close()
    debug_log(f"Job task ({pod_id}, {order_id}, {sku}, {qty}) upserted.")

def update_job_task(
        pod_id: int,
        order_id: int,
        sku: int,
        qty: int,
        assigned_station: str = None,
        pod_assigned_time: float = None,
        status: str = None,
        finish_time: float = None,
        db_path: str = "warehouse.db"):
    """
    Insert or update a job task based on (pod_id, order_id, sku, qty) as composite key.
    Only non-None fields will be updated on conflict.
    """
    if not is_detail_db_enabled():
        return

    conn = connect(db_path)
    cursor = conn.cursor()

    # Prepare dynamic update
    fields = []
    values = []

    if assigned_station is not None:
        fields.append("assigned_station = ?")
        values.append(assigned_station)
    if pod_assigned_time is not None:
        fields.append("pod_assigned_time = ?")
        values.append(pod_assigned_time)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if finish_time is not None:
        fields.append("finish_time = ?")
        values.append(finish_time)

    if fields:
        query = f"""
            UPDATE job_task_{TS} SET {', '.join(fields)}
            WHERE pod_id = ? AND order_id = ? AND sku = ? AND qty = ?
        """
        values.extend([pod_id, order_id, sku, qty])
        execute(cursor, query, tuple(values))
    
    commit(conn)
    conn.close()
    debug_log(f"Job task ({pod_id}, {order_id}, {sku}, {qty}) updated.")

def get_job_task(pod_id: int = None, order_id: str = None, db_path: str = "warehouse.db"):
    """
    Retrieve job task history.
    - If pod_id and/or order_id is provided, filters accordingly.
    - If neither is provided, returns all job tasks.
    Returns a list of dictionaries.
    """
    if not is_detail_db_enabled():
        return []

    conn = connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = f"SELECT * FROM job_task_{TS}"
    conditions = []
    values = []

    if pod_id is not None:
        conditions.append("pod_id = ?")
        values.append(pod_id)
    if order_id is not None:
        conditions.append("order_id = ?")
        values.append(order_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY pod_assigned_time"

    cursor.execute(query, tuple(values))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
