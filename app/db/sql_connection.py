import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("app.db.sql_connection")

def get_connection_string() -> str:
    """Build MSSQL connection string using pyodbc driver and .env variables."""
    server = os.getenv("SERVER")
    database = os.getenv("DATABASE")
    username = os.getenv("UID")
    password = os.getenv("PWD")

    if not all([server, database, username, password]):
        raise ValueError("❌ Missing one or more SQL connection variables in .env")

    conn_str = (
        f"mssql+pyodbc://{username}:{password}@{server}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
    )
    return conn_str


def execute_sql_query(query: str):
    """Execute a SQL query safely and return results as DataFrame or error dict."""
    try:
        conn_str = get_connection_string()
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            logger.info("✅ SQL query executed successfully.")
            return df
    except Exception as e:
        logger.exception("❌ SQL execution failed")
        return {"error": str(e)}
