
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.path.join(os.path.dirname(__file__), "merged_people.csv")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS people (
    id SERIAL PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    city TEXT,
    skills TEXT,
    experience_years NUMERIC,
    ctc NUMERIC,
    status TEXT,
    rate_per_hour_normalized NUMERIC,
    verified BOOLEAN,
    projects_completed NUMERIC,
    sources TEXT,
    source_count INTEGER,
    quality_score TEXT,
    applied_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

ADD_APPLIED_DATE_SQL = """
ALTER TABLE people ADD COLUMN IF NOT EXISTS applied_date DATE;
"""

ADD_EMAIL_UNIQUE_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'people_email_unique'
    ) THEN
        ALTER TABLE people ADD CONSTRAINT people_email_unique UNIQUE (email);
    END IF;
END $$;
"""

CREATE_REVIEW_QUEUE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS review_queue (
    id SERIAL PRIMARY KEY,
    person_a_name TEXT, person_a_email TEXT, person_a_phone TEXT,
    person_a_city TEXT, person_a_sources TEXT,
    person_b_name TEXT, person_b_email TEXT, person_b_phone TEXT,
    person_b_city TEXT, person_b_sources TEXT,
    name_similarity_pct NUMERIC,
    reason TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
"""
CREATE_STAGING_TABLE_SQL = """
CREATE TEMP TABLE staging_people (
    LIKE people INCLUDING DEFAULTS
) ON COMMIT DROP;
ALTER TABLE staging_people DROP COLUMN id;
"""

INSERT_NEW_ONLY_SQL = """
INSERT INTO people ({cols})
SELECT {cols}
FROM staging_people s
WHERE NOT EXISTS (
    SELECT 1 FROM people p
    WHERE (s.email IS NOT NULL AND p.email = s.email)
       OR (s.phone IS NOT NULL AND p.phone = s.phone)
);
"""

COUNT_SKIPPED_SQL = """
SELECT COUNT(*) FROM staging_people s
WHERE EXISTS (
    SELECT 1 FROM people p
    WHERE (s.email IS NOT NULL AND p.email = s.email)
       OR (s.phone IS NOT NULL AND p.phone = s.phone)
);
"""

CHECK_EXISTING_SQL = """
SELECT id, name, email, phone FROM people
WHERE (email IS NOT NULL AND email = %(email)s)
   OR (phone IS NOT NULL AND phone = %(phone)s)
LIMIT 1;
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_people_email ON people(email);
CREATE INDEX IF NOT EXISTS idx_people_phone ON people(phone);
"""

INSERT_COLUMNS = [
    "name", "email", "phone", "city", "skills", "experience_years", "ctc",
    "status", "rate_per_hour_normalized", "verified", "projects_completed",
    "sources", "source_count", "quality_score", "applied_date",
]


def get_connection():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)
    # fallback: individual vars, if someone prefers that style
    return psycopg2.connect(
        host=os.environ["SUPABASE_HOST"],
        port=os.environ.get("SUPABASE_PORT", "5432"),
        dbname=os.environ.get("SUPABASE_DB", "postgres"),
        user=os.environ.get("SUPABASE_USER", "postgres"),
        password=os.environ["SUPABASE_PASSWORD"],
    )


def load_people_scalable(cur, df: pd.DataFrame) -> tuple[int, int]:
    """
    Bulk duplicate-check + insert using a staging table and one set-based
    anti-join, instead of a per-row SELECT+INSERT. Round-trip count stays
    constant regardless of how many rows are in df.
    """
    cur.execute(CREATE_STAGING_TABLE_SQL)

    rows = [tuple(r) for r in df[INSERT_COLUMNS].itertuples(index=False)]
    cols_sql = ", ".join(INSERT_COLUMNS)
    execute_values(
        cur, f"INSERT INTO staging_people ({cols_sql}) VALUES %s", rows
    )

    cur.execute(COUNT_SKIPPED_SQL)
    skipped = cur.fetchone()[0]

    cur.execute(INSERT_NEW_ONLY_SQL.format(cols=cols_sql))
    inserted = cur.rowcount

    return inserted, skipped


def main():
    df = pd.read_csv(CSV_PATH, dtype={"phone": str})  # keep phone as text, never numeric
    applied_date = df.get("applied_date", pd.Series(None, index=df.index))
    df["applied_date"] = pd.to_datetime(applied_date, errors="coerce").dt.date
    df = df.astype(object).where(pd.notna(df), None)  # NaN -> SQL NULL

    print(f"Loaded {len(df)} merged records from {CSV_PATH}")

    review_path = os.path.join(os.path.dirname(__file__), "review_queue.csv")
    review_df = None
    if os.path.exists(review_path):
        review_df = pd.read_csv(
            review_path,
            dtype={"person_a_phone": str, "person_b_phone": str},
        )
        review_df = review_df.astype(object).where(pd.notna(review_df), None)

    conn = get_connection()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            print("Creating tables (if not exist)...")
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(ADD_APPLIED_DATE_SQL)
            cur.execute(CREATE_REVIEW_QUEUE_TABLE_SQL)
            cur.execute(ADD_EMAIL_UNIQUE_SQL)

            print("Bulk loading via staging table + set-based dedup "
                  "(scales to millions of rows, not just this batch)...")
            inserted, skipped_dupes = load_people_scalable(cur, df)

            print(f"Inserted {inserted} new people, skipped {skipped_dupes} already-existing "
                  f"(matched by email or phone).")

            if review_df is not None and len(review_df):
                print(f"Refreshing review_queue with {len(review_df)} ambiguous pairs "
                      f"(derived data - safe to replace on every run)...")
                cur.execute("DELETE FROM review_queue;")
                review_cols = [
                    "person_a_name", "person_a_email", "person_a_phone", "person_a_city",
                    "person_a_sources", "person_b_name", "person_b_email", "person_b_phone",
                    "person_b_city", "person_b_sources", "name_similarity_pct", "reason",
                ]
                placeholders = ", ".join(f"%({c})s" for c in review_cols)
                for row in review_df[review_cols].itertuples(index=False):
                    row = dict(zip(review_cols, row))
                    cur.execute(
                        f"INSERT INTO review_queue ({', '.join(review_cols)}) VALUES ({placeholders})",
                        row,
                    )

            print("Creating indexes for fast duplicate-lookup (used by Task 2 n8n flow)...")
            cur.execute(CREATE_INDEXES_SQL)

        conn.commit()
        print("Done. Data committed to Supabase.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
