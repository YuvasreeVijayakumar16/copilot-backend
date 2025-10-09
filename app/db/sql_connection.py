# app/db/sql_connection.py

import pyodbc
import pandas as pd
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable*", category=UserWarning)

load_dotenv()  # Load values from .env file if present

def get_db_connection():
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={os.getenv('SERVER')};"
        f"DATABASE={os.getenv('DATABASE')};"
        f"UID={os.getenv('UID')};"
        f"PWD={os.getenv('PWD')};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    return conn

def execute_sql_query(query):
    conn = get_db_connection()
    try:
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()
