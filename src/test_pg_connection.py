import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

def get_connection_uri():
    pg_user = os.getenv('PGUSER')
    pg_host = os.getenv('PGHOST')
    pg_database = os.getenv('PGDATABASE')
    token_path = os.path.join(os.path.dirname(__file__), 'aad_token.txt')
    if not (pg_user and pg_host and pg_database):
        raise Exception('Missing PGUSER, PGHOST, or PGDATABASE environment variable')
    if not os.path.exists(token_path):
        raise Exception(f"AAD token file not found: {token_path}")
    with open(token_path, 'r') as f:
        token = f.read().strip().replace('\n', '')
    from urllib.parse import quote
    encoded_user = quote(pg_user)
    return f"postgresql://{encoded_user}:{token}@{pg_host}:5432/{pg_database}?sslmode=require"

def test_connection():
    uri = get_connection_uri()
    print(f"Testing connection URI: {uri[:60]}... (token hidden)")
    try:
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                print(cur.fetchone()[0])
    except Exception as e:
        print('Connection failed:', e)

if __name__ == '__main__':
    test_connection()
