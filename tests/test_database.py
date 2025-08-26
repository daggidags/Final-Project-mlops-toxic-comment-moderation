import os
import psycopg2
import pytest

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "toxicity-db.cdowqssegxo6.us-east-1.rds.amazonaws.com"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "finalproject"),
    "dbname": os.getenv("DB_NAME", "postgres"),
}

def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError:
        pytest.skip("Skipping DB tests: Database not reachable from local environment")

def test_database_connection():
    """Check if we can connect to the DB."""
    conn = get_connection()
    assert conn is not None
    conn.close()

def test_prediction_logs_table_exists():
    """Check if the prediction_logs table exists."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'prediction_logs'
        );
    """)
    exists = cur.fetchone()[0]
    cur.close()
    conn.close()
    assert exists is True, "prediction_logs table should exist"
