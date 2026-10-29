"""
SQLite Database Layer for Prompt Injection Prevention and Detection Framework.
Maintains relational tables for requests, multi-layered detection scores,
prevention actions, and AI generation audit logs.
"""
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from config import DATABASE_PATH


class DatabaseManager:
    """Manages SQLite connections and security audit logging."""

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = str(db_path)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database tables and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Requests table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                original_prompt TEXT NOT NULL,
                normalized_prompt TEXT NOT NULL,
                length INTEGER NOT NULL
            );
            """)

            # Detections table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                rule_score REAL NOT NULL,
                ml_score REAL NOT NULL,
                llm_score REAL NOT NULL,
                final_risk_score REAL NOT NULL,
                classification TEXT NOT NULL,
                attack_category TEXT NOT NULL,
                reason TEXT,
                latency_ms REAL NOT NULL,
                FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
            );
            """)

            # Actions / Prevention table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                decision TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                reason TEXT,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
            );
            """)

            # Model Generation / Responses table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                response_text TEXT,
                generation_latency_ms REAL,
                total_latency_ms REAL,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
            );
            """)

            # Create Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_req ON detections(request_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_req ON actions(request_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_responses_req ON responses(request_id);")
            
            conn.commit()

    def log_request(self, session_id: str, original_prompt: str, normalized_prompt: str, length: int) -> int:
        """Logs a new incoming prompt and returns its primary request ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO requests (session_id, timestamp, original_prompt, normalized_prompt, length)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, now, original_prompt, normalized_prompt, length),
            )
            conn.commit()
            return cursor.lastrowid

    def log_detection(
        self,
        request_id: int,
        rule_score: float,
        ml_score: float,
        llm_score: float,
        final_risk_score: float,
        classification: str,
        attack_category: str,
        reason: str,
        latency_ms: float,
    ) -> int:
        """Logs multi-layered detection metrics for a specific request."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO detections (
                    request_id, rule_score, ml_score, llm_score, final_risk_score,
                    classification, attack_category, reason, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    round(rule_score, 4),
                    round(ml_score, 4),
                    round(llm_score, 4),
                    round(final_risk_score, 4),
                    classification,
                    attack_category,
                    reason,
                    round(latency_ms, 2),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_action(self, request_id: int, decision: str, action_taken: str, reason: str) -> int:
        """Logs the security prevention decision (ALLOWED, BLOCKED, WARNED)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO actions (request_id, decision, action_taken, reason, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, decision, action_taken, reason, now),
            )
            conn.commit()
            return cursor.lastrowid

    def log_response(
        self,
        request_id: int,
        model_name: str,
        response_text: str,
        generation_latency_ms: float,
        total_latency_ms: float,
    ) -> int:
        """Logs the final AI model generation response and timing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO responses (
                    request_id, model_name, response_text, generation_latency_ms, total_latency_ms, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    model_name,
                    response_text,
                    round(generation_latency_ms, 2),
                    round(total_latency_ms, 2),
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves a consolidated view of recent requests, detections, decisions, and responses."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
            SELECT 
                r.id AS request_id,
                r.session_id,
                r.timestamp,
                r.original_prompt,
                d.rule_score,
                d.ml_score,
                d.llm_score,
                d.final_risk_score,
                d.classification,
                d.attack_category,
                d.reason AS detection_reason,
                d.latency_ms AS detection_latency_ms,
                a.decision,
                a.action_taken,
                res.model_name,
                res.response_text,
                res.generation_latency_ms,
                res.total_latency_ms
            FROM requests r
            LEFT JOIN detections d ON r.id = d.request_id
            LEFT JOIN actions a ON r.id = a.request_id
            LEFT JOIN responses res ON r.id = res.request_id
            ORDER BY r.id DESC
            LIMIT ?
            """
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Calculates security metrics and aggregate analytics from SQLite records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM requests")
            total_requests = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM actions WHERE decision = 'SAFE' OR action_taken = 'ALLOWED'")
            safe_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM actions WHERE decision = 'MALICIOUS' OR action_taken = 'BLOCKED'")
            blocked_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM actions WHERE decision = 'UNCERTAIN' OR action_taken = 'WARNED'")
            uncertain_count = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(latency_ms) FROM detections")
            avg_detection_latency = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT AVG(total_latency_ms) FROM responses WHERE total_latency_ms IS NOT NULL")
            avg_total_latency = cursor.fetchone()[0] or 0.0

            cursor.execute("""
                SELECT attack_category, COUNT(*) as count 
                FROM detections 
                GROUP BY attack_category 
                ORDER BY count DESC
            """)
            category_distribution = {row["attack_category"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_requests": total_requests,
                "safe_count": safe_count,
                "blocked_count": blocked_count,
                "uncertain_count": uncertain_count,
                "avg_detection_latency_ms": round(avg_detection_latency, 2),
                "avg_total_latency_ms": round(avg_total_latency, 2),
                "category_distribution": category_distribution,
            }

    def clear_logs(self):
        """Clears all log records for testing or resetting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM responses")
            cursor.execute("DELETE FROM actions")
            cursor.execute("DELETE FROM detections")
            cursor.execute("DELETE FROM requests")
            conn.commit()


# Singleton instance
db = DatabaseManager()
