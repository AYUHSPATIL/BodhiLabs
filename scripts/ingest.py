import json
import logging
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data" / "preprocessed" / "dynamic_preprocessed_data"

DF1_PATH  = DATA_DIR / "df1_all_users_mcq_attempts.csv"
DF2_PATH  = DATA_DIR / "df2_all_users_ordering_steps.csv"
QMAP_PATH = DATA_DIR / "question_map_all_modules.json"

# ── MySQL Config —
DB_USER     = "root"
DB_PASSWORD = "bodhi"
DB_HOST     = "localhost"
DB_PORT     = "3306"
DB_NAME     = "medlearn"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── DDL ────────────────────────────────────────────────────────────────────────
TABLES = {
    "mcq_attempts": """
        CREATE TABLE IF NOT EXISTS mcq_attempts (
            id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
            user_id             BIGINT          NOT NULL,
            course_id           VARCHAR(50)     NOT NULL,
            question_id         BIGINT          NOT NULL,
            selected_option_id  BIGINT,
            correct_option_id   BIGINT,
            is_correct          BOOLEAN         NOT NULL,
            marks_awarded       INT             NOT NULL,
            max_marks           INT             NOT NULL,
            quiz_id             BIGINT          NOT NULL,
            quiz_name           VARCHAR(255),
            attempt_id          BIGINT          NOT NULL,
            response_timestamp  DATETIME,
            competency_id       VARCHAR(100),
            is_mappable         BOOLEAN         NOT NULL DEFAULT FALSE,
            INDEX idx_user_comp   (user_id, competency_id),
            INDEX idx_user_course (user_id, course_id),
            INDEX idx_mappable    (is_mappable),
            INDEX idx_competency  (competency_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "ordering_steps": """
        CREATE TABLE IF NOT EXISTS ordering_steps (
            id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
            user_id             BIGINT          NOT NULL,
            course_id           VARCHAR(50)     NOT NULL,
            question_id         BIGINT          NOT NULL,
            option_id           BIGINT          NOT NULL,
            attempt_id          BIGINT          NOT NULL,
            quiz_id             BIGINT          NOT NULL,
            quiz_name           VARCHAR(255),
            response_timestamp  DATETIME,
            user_position       INT             NOT NULL,
            correct_position    INT             NOT NULL,
            is_correct          BOOLEAN         NOT NULL,
            position_delta      INT             NOT NULL,
            marks_awarded       INT             NOT NULL,
            max_marks           INT             NOT NULL,
            competency_id       TEXT,
            is_mappable         BOOLEAN         NOT NULL DEFAULT FALSE,
            INDEX idx_user_course (user_id, course_id),
            INDEX idx_user_qid    (user_id, question_id),
            INDEX idx_mappable    (is_mappable)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "question_map": """
        CREATE TABLE IF NOT EXISTS question_map (
            question_id     VARCHAR(50)  NOT NULL,
            course_id       VARCHAR(50)  NOT NULL,
            competency_id   VARCHAR(255),
            PRIMARY KEY (question_id),
            INDEX idx_course      (course_id),
            INDEX idx_competency  (competency_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
}


def build_qmap_df(path):
    with open(path, encoding='utf-8') as f:
        qmap = json.load(f)
    rows = []
    for course_id, questions in qmap.items():
        for question_id, comp in questions.items():
            rows.append({
                "question_id":   question_id,
                "course_id":     course_id,
                "competency_id": comp.get("competency_id"),
            })
    return pd.DataFrame(rows)


def truncate_and_insert(df, table, engine, chunksize=1000):
    """Truncate table then insert fresh — safe for reruns."""
    with engine.connect() as conn:
        conn.execute(text(f"SET FOREIGN_KEY_CHECKS=0;"))
        conn.execute(text(f"TRUNCATE TABLE {table};"))
        conn.execute(text(f"SET FOREIGN_KEY_CHECKS=1;"))
        conn.commit()
    df.to_sql(table, con=engine, if_exists="append", index=False, chunksize=chunksize)


def main():
    engine = create_engine(DATABASE_URL, echo=False)

    # Create tables if not exist
    with engine.connect() as conn:
        for ddl in TABLES.values():
            conn.execute(text(ddl))
        conn.commit()
    logger.info("Tables ready.")

    # Load data
    df1    = pd.read_csv(DF1_PATH, parse_dates=["response_timestamp"])
    df2    = pd.read_csv(DF2_PATH, parse_dates=["response_timestamp"])
    df_map = build_qmap_df(QMAP_PATH)

    # Ingest
    logger.info("Ingesting... (this may take a moment)")
    truncate_and_insert(df1,    "mcq_attempts",   engine)
    truncate_and_insert(df2,    "ordering_steps", engine)
    truncate_and_insert(df_map, "question_map",   engine)

    # Verify
    with engine.connect() as conn:
        for table in TABLES:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            logger.info(f"  {table}: {count} rows")

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()