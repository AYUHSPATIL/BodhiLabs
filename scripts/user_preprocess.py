import json
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR       = Path(__file__).parent.parent
RAW_MODULE_DIR = BASE_DIR / "data" / "raw" / "static_data"
RAW_USER_DIR   = BASE_DIR / "data" / "raw" / "dynamic_data"
OUTPUT_DIR     = BASE_DIR / "data" / "preprocessed" / "dynamic_preprocessed_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def normalize_questions(data):
    """Handle both list and dict formats for questions field."""
    questions = data.get("questions", [])
    if isinstance(questions, dict):
        return list(questions.values())
    return questions

def empty_to_none(val):
    """Convert empty string → None so MySQL stores proper NULL."""
    if isinstance(val, str) and val.strip() == "":
        return None
    return val

def parse_timestamp(ts_str):
    """Strip IST suffix and parse to proper datetime."""
    if not ts_str:
        return None
    return pd.to_datetime(ts_str.replace(" IST", "").strip(), errors='coerce')


# ── Question map builder ───────────────────────────────────────────────────────

def build_question_map(module_data):
    """
    Builds question_id → competency_id mapping from one module JSON.

    MCQ questions       → single competency_id  e.g. "UCK001"
    Checklist questions → ALL skill competency_ids from module
                          as comma-separated string e.g. "UCS002,UCS003,UCS006"

    No competency_type or competency_area stored here — those are
    text fields that already live in ChromaDB. DFs only carry IDs.
    """
    q_map     = {}
    module    = module_data[0]
    course_id = module["key_module_field"]["course_id"]

    # MCQ → one competency_id per question
    for comp_block in module["question_type_mcq"]:
        comp  = comp_block["competency"]
        q_ids = [q.strip().strip("'") for q in comp_block["question"]["question_ids"].split(",")]
        for qid in q_ids:
            if qid:
                q_map[qid] = {
                    "competency_id": empty_to_none(comp["competency_id"]),
                }

    # Collect ALL competency_ids from this module for checklist mapping
    all_skill_comp_ids = [
        c["competency"]["competency_id"]
        for c in module["question_type_mcq"]
        if c["competency"]["competency_id"]
    ]
    checklist_comp_str = ",".join(all_skill_comp_ids) if all_skill_comp_ids else None

    # Checklist → comma-separated all skill comp_ids
    for checklist in module.get("question_type_checklist", []):
        qid = str(checklist["question"]["question_id"])
        q_map[qid] = {
            "competency_id": checklist_comp_str,
        }
        if not checklist_comp_str:
            logger.warning(f"  No skill competencies found for checklist qid={qid} in course={course_id}")

    return course_id, q_map


# ── DataFrame builders ─────────────────────────────────────────────────────────

def build_df1(user_data, question_maps, qid_to_course):
    """
    DF1: one row per multichoice attempt.

    Columns: user_id, course_id, question_id, selected_option_id,
             correct_option_id, is_correct, marks_awarded, max_marks,
             quiz_id, quiz_name, attempt_id, response_timestamp,
             competency_id, is_mappable
    """
    rows      = []
    questions = normalize_questions(user_data)

    for q in questions:
        if q["question_type"] != "multichoice":
            continue
        qid       = str(q["question_id"])
        course_id = qid_to_course.get(qid, "unknown")
        comp      = question_maps.get(course_id, {}).get(qid, {})

        for attempt in q["attempts"]:
            selected = attempt["selected_options"][0] if attempt["selected_options"] else {}
            rows.append({
                "user_id":            user_data["user"]["username"],
                "course_id":          course_id,
                "question_id":        qid,
                "selected_option_id": selected.get("option_id"),
                "correct_option_id":  next((o["option_id"] for o in q["options"] if o["is_correct"]), None),
                "is_correct":         selected.get("is_correct", False),
                "marks_awarded":      attempt["marks_awarded"],
                "max_marks":          q["max_marks"],
                "quiz_id":            attempt["quiz_id"],
                "quiz_name":          attempt["quiz_name"],
                "attempt_id":         attempt["attempt_id"],
                "response_timestamp": parse_timestamp(attempt["response_timestamp"]),
                "competency_id":      empty_to_none(comp.get("competency_id")),
                "is_mappable":        course_id != "unknown",
            })
    return pd.DataFrame(rows)


def build_df2(user_data, question_maps, qid_to_course):
    """
    DF2: one row per step per ordering/checklist attempt.

    competency_id = comma-separated all skill comp_ids for that module
                    e.g. "UCS002,UCS003,UCS006"

    Columns: user_id, course_id, question_id, option_id, attempt_id,
             quiz_id, quiz_name, response_timestamp, user_position,
             correct_position, is_correct, position_delta,
             marks_awarded, max_marks, competency_id, is_mappable
    """
    rows      = []
    questions = normalize_questions(user_data)

    for q in questions:
        if q["question_type"] != "ordering":
            continue
        qid       = str(q["question_id"])
        course_id = qid_to_course.get(qid, "unknown")
        comp      = question_maps.get(course_id, {}).get(qid, {})

        for attempt in q["attempts"]:
            for step in attempt.get("selected_sequence", []):
                rows.append({
                    "user_id":            user_data["user"]["username"],
                    "course_id":          course_id,
                    "question_id":        qid,
                    "option_id":          step["option_id"],
                    "attempt_id":         attempt["attempt_id"],
                    "quiz_id":            attempt["quiz_id"],
                    "quiz_name":          attempt["quiz_name"],
                    "response_timestamp": parse_timestamp(attempt["response_timestamp"]),
                    "user_position":      step["user_position"],
                    "correct_position":   step["correct_position"],
                    "is_correct":         step["is_correct"],
                    "position_delta":     abs(step["user_position"] - step["correct_position"]),
                    "marks_awarded":      attempt["marks_awarded"],
                    "max_marks":          q["max_marks"],
                    "competency_id":      empty_to_none(comp.get("competency_id")),
                    "is_mappable":        course_id != "unknown",
                })
    return pd.DataFrame(rows)


# ── Pre-ingestion validation ───────────────────────────────────────────────────

def validate_df(df, name):
    """Sanity checks before saving — logs warnings, does not raise."""
    logger.info(f"\n  [{name}] Validation:")
    logger.info(f"    Shape           : {df.shape}")
    logger.info(f"    Unmappable rows : {(~df['is_mappable']).sum()}")
    logger.info(f"    Null comp_id    : {df['competency_id'].isnull().sum()}")
    empty_str = (df['competency_id'] == '').sum()
    if empty_str:
        logger.warning(f"    ⚠ Empty string comp_ids still present: {empty_str}")
    bad_ts = df['response_timestamp'].isnull().sum()
    if bad_ts:
        logger.warning(f"    ⚠ Unparseable timestamps : {bad_ts}")
    else:
        logger.info(f"    Timestamps      : all parsed OK")
    # log a sample to verify comma-separated format in DF2
    if name == "DF2 Ordering" and not df.empty:
        sample = df[df['competency_id'].notna()]['competency_id'].iloc[0]
        logger.info(f"    Sample comp_id  : {sample}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("Building question maps from all module JSONs...")
    question_maps = {}
    qid_to_course = {}

    for module_path in RAW_MODULE_DIR.glob("*.json"):
        try:
            module_data = load_json(module_path)
            course_id, q_map = build_question_map(module_data)
            question_maps[course_id] = q_map
            for qid in q_map:
                qid_to_course[qid] = course_id
            logger.info(f"  Loaded: {module_path.name} → course_id={course_id} | {len(q_map)} questions mapped")
        except Exception as e:
            logger.warning(f"  Could not build map for {module_path.name}: {e}")

    logger.info(f"\nLoaded question maps for {len(question_maps)} modules\n")

    with open(OUTPUT_DIR / "question_map_all_modules.json", 'w', encoding='utf-8') as f:
        json.dump(question_maps, f, indent=2, ensure_ascii=False)

    user_files = sorted(RAW_USER_DIR.glob("user_data_*.json"))
    if not user_files:
        logger.error(f"No user data files found in {RAW_USER_DIR}")
        return

    logger.info(f"Found {len(user_files)} user file(s) to process\n")

    all_df1      = []
    all_df2      = []
    all_unmapped = set()
    success, failed = 0, []

    for user_path in user_files:
        try:
            user_data = load_json(user_path)
            username  = user_data["user"]["username"]

            df1 = build_df1(user_data, question_maps, qid_to_course)
            df2 = build_df2(user_data, question_maps, qid_to_course)

            all_df1.append(df1)
            all_df2.append(df2)

            if not df1.empty:
                all_unmapped.update(df1[df1["course_id"] == "unknown"]["question_id"].unique())
            if not df2.empty:
                all_unmapped.update(df2[df2["course_id"] == "unknown"]["question_id"].unique())

            logger.info(f"  {username} | mcq: {len(df1)} rows | ordering: {len(df2)} rows")
            success += 1

        except Exception as e:
            logger.error(f"  Failed on {user_path.name}: {e}", exc_info=True)
            failed.append(user_path.name)

    if all_df1:
        df1_combined = pd.concat(all_df1, ignore_index=True)
        validate_df(df1_combined, "DF1 MCQ")
        df1_combined.to_csv(OUTPUT_DIR / "df1_all_users_mcq_attempts.csv", index=False)
        logger.info(f"\n  Saved DF1 | shape: {df1_combined.shape}")

    if all_df2:
        df2_combined = pd.concat(all_df2, ignore_index=True)
        validate_df(df2_combined, "DF2 Ordering")
        df2_combined.to_csv(OUTPUT_DIR / "df2_all_users_ordering_steps.csv", index=False)
        logger.info(f"  Saved DF2 | shape: {df2_combined.shape}")

    if all_unmapped:
        logger.warning(f"\n  Unmapped question IDs: {sorted(all_unmapped)}")
        with open(OUTPUT_DIR / "unmapped_question_ids.json", 'w') as f:
            json.dump(sorted(all_unmapped), f, indent=2)
    else:
        logger.info(f"\n  All question IDs mapped successfully")

    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Success: {success} | Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed files: {failed}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()