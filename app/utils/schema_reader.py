# app/utils/schema_reader.py

import os
import time
import pyodbc
import pandas as pd
import warnings
from functools import lru_cache
from typing import Dict, Set, Tuple, Optional
from app.db.sql_connection import get_connection_string

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable*", category=UserWarning)

# Cache TTL in seconds (1 hour by default)
SCHEMA_CACHE_TTL = 3600

# Global cache storage
_schema_cache: Dict[str, Tuple[float, dict, str, set]] = {}

def _create_db_connection():
    """Create and return a live pyodbc connection using .env variables."""
    try:
        server = os.getenv("SERVER")
        database = os.getenv("DATABASE")
        username = os.getenv("UID")
        password = os.getenv("PWD")

        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};DATABASE={database};UID={username};PWD={password}"
        )
        return conn
    except Exception as e:
        raise RuntimeError(f"❌ Database connection failed: {e}")


def _is_cache_valid() -> bool:
    """Check if cached schema is still valid."""
    if not _schema_cache:
        return False
    cache_time = _schema_cache.get('timestamp', 0)
    return time.time() - cache_time < SCHEMA_CACHE_TTL


def get_schema_and_sample_data():
    """
    Returns cached or fresh:
    - structured_schema: dict -> {table_name: [column1, column2, ...]}
    - schema_text: str -> Flattened for GPT input (table(column1, column2))
    - sample_data: set -> Unique values from top rows of tables
    """
    # Check cache first
    if _is_cache_valid():
        return (
            _schema_cache["schema"],
            _schema_cache["text"],
            _schema_cache["data"]
        )

    conn = _create_db_connection()
    cursor = conn.cursor()

    # === Fetch table/column schema ===
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA', 'sys')
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    rows = cursor.fetchall()

    structured_schema = {}
    for table, column in rows:
        structured_schema.setdefault(table.upper(), []).append(column)

    # === Create flattened schema text for GPT ===
    schema_text = "\n".join([
        f"{table}({', '.join(columns)})"
        for table, columns in structured_schema.items()
    ])

    # === Collect sample data ===
    all_sample_data = []
    for table in structured_schema:
        try:
            df = pd.read_sql(f"SELECT TOP 5 * FROM {table}", conn)
            sample_values = df.astype(str).values.flatten().tolist()
            all_sample_data.extend([val.lower() for val in sample_values if isinstance(val, str)])
        except Exception:
            continue  # Skip tables that can’t be queried

    conn.close()

    # === Cache results ===
    sample_data = set(all_sample_data)
    _schema_cache.update({
        'timestamp': time.time(),
        'schema': structured_schema,
        'text': schema_text,
        'data': sample_data
    })

    return structured_schema, schema_text, sample_data


def get_db_schema():
    """Returns database schema as table_name(column1, column2, ...)."""
    conn = _create_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA', 'sys')
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    rows = cursor.fetchall()

    schema = {}
    for table, column in rows:
        schema.setdefault(table.upper(), []).append(column)

    conn.close()

    # Flatten for output
    return "\n".join([f"{tbl}({', '.join(cols)})" for tbl, cols in schema.items()])
