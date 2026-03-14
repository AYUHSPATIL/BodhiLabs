import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── DB Config ──────────────────────────────────────────────────────────────────
DB_USER     = "root"
DB_PASSWORD = "bodhi"
DB_HOST     = "localhost"
DB_PORT     = "3306"
DB_NAME     = "medlearn"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── DDL ────────────────────────────────────────────────────────────────────────
CREATE_COMP_SCORES = """
CREATE TABLE IF NOT EXISTS competency_scores (
    id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id                 BIGINT       NOT NULL,
    competency_id           VARCHAR(100) NOT NULL,

    -- MCQ based features
    total_attempts          INT          DEFAULT 0,
    correct_attempts        INT          DEFAULT 0,
    wrong_attempts          INT          DEFAULT 0,
    accuracy_rate           FLOAT        DEFAULT 0.0,  -- correct / total
    avg_marks_pct           FLOAT        DEFAULT 0.0,  -- avg(marks_awarded / max_marks)
    attempt_trend           FLOAT        DEFAULT 0.0,  -- avg(last 2 correct) - avg(first 2 correct)

    -- Checklist based features (ordering_steps)
    checklist_steps_correct INT          DEFAULT 0,
    checklist_total_steps   INT          DEFAULT 0,
    checklist_accuracy      FLOAT        DEFAULT NULL, -- correct_steps / total_steps
    avg_position_delta      FLOAT        DEFAULT NULL, -- AVG(position_delta), 0 = perfect

    UNIQUE KEY uq_user_comp (user_id, competency_id),
    INDEX idx_user          (user_id),
    INDEX idx_comp          (competency_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_attempt_trend(series: pd.Series) -> float:
    """avg(last 2 attempts correct) - avg(first 2 attempts correct)
    Positive = improving, Negative = declining, 0 = no change."""
    if len(series) < 2:
        return 0.0
    first2 = series.iloc[:2].astype(float).mean()
    last2  = series.iloc[-2:].astype(float).mean()
    return round(float(last2 - first2), 4)


# ── Feature builders ───────────────────────────────────────────────────────────

def build_mcq_features(df1: pd.DataFrame) -> pd.DataFrame:
    """Per-user per-competency features from mcq_attempts."""
    df = df1[df1["is_mappable"] & df1["competency_id"].notna()].copy()
    df = df.sort_values(["user_id", "competency_id", "response_timestamp"])

    rows = []
    for (user_id, comp_id), grp in df.groupby(["user_id", "competency_id"]):
        total     = len(grp)
        correct   = int(grp["is_correct"].sum())
        marks_pct = (grp["marks_awarded"] / grp["max_marks"].replace(0, np.nan)).mean()

        rows.append({
            "user_id":          user_id,
            "competency_id":    comp_id,
            "total_attempts":   total,
            "correct_attempts": correct,
            "wrong_attempts":   total - correct,
            "accuracy_rate":    round(correct / total, 4),
            "avg_marks_pct":    round(float(marks_pct), 4),
            "attempt_trend":    compute_attempt_trend(grp["is_correct"]),
        })

    return pd.DataFrame(rows)


def build_checklist_features(df2: pd.DataFrame) -> pd.DataFrame:
    """Per-user per-competency features from ordering_steps.
    Explode comma-separated competency_id so each comp gets its own score row."""
    df = df2[df2["is_mappable"] & df2["competency_id"].notna()].copy()

    df["competency_id"] = df["competency_id"].str.split(",")
    df = df.explode("competency_id")
    df["competency_id"] = df["competency_id"].str.strip()

    rows = []
    for (user_id, comp_id), grp in df.groupby(["user_id", "competency_id"]):
        total_steps   = len(grp)
        correct_steps = int(grp["is_correct"].sum())

        rows.append({
            "user_id":                 user_id,
            "competency_id":           comp_id,
            "checklist_steps_correct": correct_steps,
            "checklist_total_steps":   total_steps,
            "checklist_accuracy":      round(correct_steps / total_steps, 4) if total_steps > 0 else None,
            "avg_position_delta":      round(float(grp["position_delta"].mean()), 4),
        })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    engine = create_engine(DATABASE_URL, echo=False)

    with engine.connect() as conn:
        conn.execute(text(CREATE_COMP_SCORES))
        conn.commit()
    logger.info("Table competency_scores ready.")

    logger.info("Loading data from DB...")
    df1 = pd.read_sql("SELECT * FROM mcq_attempts",   con=engine)
    df2 = pd.read_sql("SELECT * FROM ordering_steps", con=engine)
    logger.info(f"  mcq_attempts: {len(df1)} | ordering_steps: {len(df2)}")

    logger.info("Computing features...")
    mcq_feat = build_mcq_features(df1)
    ckl_feat = build_checklist_features(df2)

    # Outer merge — users with only MCQ or only checklist data both get a row
    combined = mcq_feat.merge(ckl_feat, on=["user_id", "competency_id"], how="outer")
    logger.info(f"  competency_scores: {len(combined)} rows")

    # Rerun safe — truncate then insert
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE competency_scores;"))
        conn.commit()

    combined.to_sql("competency_scores", con=engine, if_exists="append", index=False, chunksize=1000)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM competency_scores")).scalar()
    logger.info(f"  Saved: {count} rows")
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()