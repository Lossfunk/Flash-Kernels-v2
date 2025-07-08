import sqlite3
import json
import datetime
from pathlib import Path
from utils.logging_utils import get_logger # Assuming you have this

logger = get_logger("SQLiteObserver")

DB_NAME = "kernel_pipeline_observations.sqlite"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / DB_NAME # Or a more configurable path

class SQLiteObserver:
    def __init__(self, db_path: str = None):
        if db_path is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(DB_PATH)
        else:
            self.db_path = db_path
        
        self._conn = None
        self._ensure_db_connection()
        self._create_table()

    def _ensure_db_connection(self):
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self.db_path)
                self._conn.row_factory = sqlite3.Row # Access columns by name
                logger.info(f"Connected to SQLite database at {self.db_path}")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to SQLite database at {self.db_path}: {e}")
                raise

    def _create_table(self):
        self._ensure_db_connection()
        try:
            with self._conn:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS Observation (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        session_id TEXT, -- To group observations from a single pipeline run.run()
                        op_hash TEXT NOT NULL,
                        attempt_type TEXT NOT NULL, -- e.g., "synthesis", "compilation", "correctness_tuning", "parameter_reasoning"
                        synthesis_attempt_number INTEGER,
                        sub_attempt_number INTEGER, -- e.g., correctness_attempt_number or compile_reasoning_attempt
                        
                        kernel_code TEXT,
                        kernel_path TEXT, -- path to the .triton file if saved
                        ptx_path TEXT,    -- path to the .ptx file if compiled

                        status TEXT NOT NULL, -- "success", "compile_error", "correctness_error", "runtime_error", "timeout", "config_error"
                        
                        error_type TEXT, -- More specific error type, e.g., "missing_arg_X", "shape_mismatch"
                        error_message TEXT,
                        error_details TEXT, -- JSON blob for structured error info

                        compilation_log TEXT,
                        
                        input_specs TEXT, -- JSON, from op_spec
                        execution_params TEXT, -- JSON, grid config, kernel args
                        
                        latency_ms REAL,
                        speedup REAL,
                        
                        metadata TEXT -- JSON, for hints, reasoner explanations, etc.
                    )
                """)
                logger.info("Ensured 'Observation' table exists.")
        except sqlite3.Error as e:
            logger.error(f"Error creating 'Observation' table: {e}")
            # If table creation fails, we probably can't proceed.
            if self._conn:
                self._conn.close()
                self._conn = None
            raise

    def record_observation(self, data: dict):
        self._ensure_db_connection()
        if not self._conn:
            logger.error("Cannot record observation, no database connection.")
            return

        # Basic validation and default setting
        if not data.get("op_hash") or not data.get("attempt_type") or not data.get("status"):
            logger.error(f"Core fields (op_hash, attempt_type, status) are missing in observation data: {data}")
            return
            
        data.setdefault("timestamp", datetime.datetime.now().isoformat())
        
        # Ensure JSON fields are stringified; fall back to str() for non-serializable objects
        for field in ["error_details", "input_specs", "execution_params", "metadata"]:
            if field in data and data[field] is not None and not isinstance(data[field], str):
                try:
                    data[field] = json.dumps(data[field], default=str)
                except (TypeError, ValueError):
                    # As an extra safeguard, stringify the entire field
                    data[field] = str(data[field])
        
        # Filter out keys not in the table schema to prevent errors
        allowed_keys = {
            "timestamp", "session_id", "op_hash", "attempt_type", 
            "synthesis_attempt_number", "sub_attempt_number",
            "kernel_code", "kernel_path", "ptx_path", "status", "error_type", 
            "error_message", "error_details", "compilation_log", 
            "input_specs", "execution_params", "latency_ms", "speedup", "metadata"
        }
        
        filtered_data = {k: v for k, v in data.items() if k in allowed_keys}
        
        columns = ', '.join(filtered_data.keys())
        placeholders = ', '.join(['?'] * len(filtered_data))
        sql = f"INSERT INTO Observation ({columns}) VALUES ({placeholders})"
        
        try:
            with self._conn:
                self._conn.execute(sql, list(filtered_data.values()))
            logger.debug(f"Recorded observation for op_hash {data.get('op_hash')}, type {data.get('attempt_type')}")
        except sqlite3.Error as e:
            logger.error(f"Error recording observation: {e} | Data: {filtered_data}")
            # Potentially re-raise or handle more gracefully

    def get_error_signature_history(self, op_hash: str, attempt_type: str, limit: int = 10) -> list[str]:
        """
        Retrieves a recent history of error signatures for a given op_hash and attempt_type.
        This is a placeholder for how Learners might query data later.
        For now, it could be used by KernelPipelineAgent for its early termination logic.
        """
        self._ensure_db_connection()
        if not self._conn:
            logger.error("Cannot get error history, no database connection.")
            return []
        
        try:
            with self._conn:
                cursor = self._conn.execute(
                    """
                    SELECT error_type FROM Observation 
                    WHERE op_hash = ? AND attempt_type = ? AND status LIKE '%error%'
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (op_hash, attempt_type, limit)
                )
                # Return just the error_type string, or a generated signature if error_type is too generic.
                # The current _extract_error_signature in KernelPipelineAgent produces more specific signatures.
                # We'll need to decide where that signature generation lives. For now, let's assume error_type is specific enough
                # or we'll adapt.
                return [row["error_type"] for row in cursor.fetchall() if row["error_type"]]
        except sqlite3.Error as e:
            logger.error(f"Error fetching error signature history: {e}")
            return []

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info(f"Closed connection to SQLite database at {self.db_path}")

# Example usage (optional, for testing)
if __name__ == '__main__':
    observer = SQLiteObserver(db_path=":memory:") # Use in-memory DB for test
    
    test_data_compile_fail = {
        "session_id": "test_session_001",
        "op_hash": "op123",
        "attempt_type": "compilation",
        "synthesis_attempt_number": 1,
        "kernel_code": "some kernel code...",
        "status": "compile_error",
        "error_type": "SyntaxError",
        "error_message": "Invalid syntax near X",
        "compilation_log": "Error line 10: ...",
        "input_specs": {"shape": [128, 128], "dtype": "float32"}
    }
    observer.record_observation(test_data_compile_fail)

    test_data_correctness_ok = {
        "session_id": "test_session_001",
        "op_hash": "op123",
        "attempt_type": "correctness_tuning",
        "synthesis_attempt_number": 1,
        "sub_attempt_number": 1, # First correctness attempt for this synthesis
        "kernel_path": "/path/to/kernel.triton",
        "ptx_path": "/path/to/kernel.ptx",
        "status": "success",
        "execution_params": {"GRID": "(128,)", "BLOCK": "(32,)"},
        "latency_ms": 1.23,
        "speedup": 10.5,
        "input_specs": {"shape": [128, 128], "dtype": "float32"}
    }
    observer.record_observation(test_data_correctness_ok)

    history = observer.get_error_signature_history("op123", "compilation")
    print("Error history for op123 compilation:", history)
    
    observer.close()
    
    # Test with persistent DB
    persistent_observer = SQLiteObserver() # Uses default DB_PATH
    persistent_observer.record_observation(test_data_compile_fail)
    persistent_observer.close()
    print(f"Check for {DB_PATH}") 