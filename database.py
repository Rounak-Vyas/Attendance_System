import sqlite3
import numpy as np
from datetime import datetime, timedelta

DB_PATH = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            user_code TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            user_code TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_user(name: str, user_code: str, embedding: np.ndarray):
    emb = embedding.astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO users (name, user_code, embedding, created_at) VALUES (?, ?, ?, ?)",
        (name, user_code, emb.tobytes(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()

def delete_user(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_all_users(include_embeddings: bool = True):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, user_code, embedding, created_at FROM users ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    users = []
    for uid, name, code, blob, created_at in rows:
        item = {"id": uid, "name": name, "user_code": code, "created_at": created_at}
        if include_embeddings:
            item["embedding"] = np.frombuffer(blob, dtype=np.float32)
        users.append(item)
    return users

def get_recent_logs(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, name, user_code, confidence, timestamp FROM attendance_logs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "name": r[1], "user_code": r[2], "confidence": r[3], "timestamp": r[4]}
        for r in rows
    ]

def log_attendance(user_id: int, name: str, user_code: str, confidence: float, cooldown_sec: int = 60) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp FROM attendance_logs WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
    row = c.fetchone()

    now = datetime.now()
    if row:
        last_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        if (now - last_time) < timedelta(seconds=cooldown_sec):
            conn.close()
            return False

    c.execute(
        "INSERT INTO attendance_logs (user_id, name, user_code, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, name, user_code, round(confidence, 3), now.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
    return True