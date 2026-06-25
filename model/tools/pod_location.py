from src.rmfs.runtime_io.detail_db import (
    commit,
    connect,
    debug_log,
    execute,
    get_cached_pod_location,
    is_detail_db_enabled,
    record_pod_location,
    reset_pod_location_cache,
    resolve_db_path,
)

TS = None
DEFAULT_DB_PATH = "warehouse.db"


def configure_default_db_path(db_path):
    global DEFAULT_DB_PATH
    DEFAULT_DB_PATH = db_path


def _effective_db_path(db_path):
    return DEFAULT_DB_PATH if db_path == "warehouse.db" else db_path


def _connect(db_path):
    return connect(_effective_db_path(db_path))

def clear_pod_locations(db_path="warehouse.db"):
    """
    Delete all rows from the pod_location table.
    """
    effective_db_path = _effective_db_path(db_path)
    reset_pod_location_cache(effective_db_path)
    if not is_detail_db_enabled():
        return

    conn = _connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"DELETE FROM pod_location_{TS}")

    commit(conn)
    conn.close()
    debug_log("All pod locations have been cleared.")

def initialize_pod_location_table(timestamp: str, db_path="warehouse.db"):
    global TS
    TS = timestamp   
    effective_db_path = _effective_db_path(db_path)
    reset_pod_location_cache(effective_db_path)
    if not is_detail_db_enabled():
        return
    
    conn = _connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        CREATE TABLE IF NOT EXISTS pod_location_{TS} (
            id TEXT PRIMARY KEY,
            x INTEGER,
            y INTEGER
        )
    """)

    commit(conn)
    conn.close()
    debug_log("pod_location table initialized.")

def upsert_pod_location(pod_id: str, x: int, y: int, db_path="warehouse.db"):
    """
    Insert or update a pod's (x, y) location.
    """
    effective_db_path = _effective_db_path(db_path)
    record_pod_location(pod_id, x, y, effective_db_path)
    if not is_detail_db_enabled():
        return

    conn = _connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"""
        INSERT INTO pod_location_{TS} (id, x, y)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET x=excluded.x, y=excluded.y
    """, (pod_id, x, y))

    commit(conn)
    conn.close()
    debug_log(f"Pod {pod_id} location set to ({x}, {y}).")

def get_pod_location(pod_id: str, db_path="warehouse.db"):
    """
    Retrieve (x, y) location of a pod by ID.
    Returns a tuple (x, y) or None if not found.
    """
    effective_db_path = _effective_db_path(db_path)
    if not is_detail_db_enabled():
        return get_cached_pod_location(pod_id, effective_db_path)

    conn = _connect(db_path)
    cursor = conn.cursor()

    execute(cursor, f"SELECT x, y FROM pod_location_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()

    conn.close()

    if result:
        record_pod_location(pod_id, result[0], result[1], effective_db_path)
        debug_log(f"Pod {pod_id} is at location {result}")
        return result
    else:
        debug_log(f"Pod {pod_id} not found.")
        return None

# Example usage
if __name__ == "__main__":
    initialize_pod_location_table()
    upsert_pod_location("POD_001", 10, 5)
    upsert_pod_location("POD_002", 3, 7)

    get_pod_location("POD_001")
    get_pod_location("POD_003")  # Not found case
