import json
import re
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
RAW_MODULE_DIR  = BASE_DIR / "data" / "raw" / "static_data"
RAW_USER_DIR    = BASE_DIR / "data" / "raw" / "dynamic_data"
OUTPUT_DIR      = BASE_DIR / "data" / "preprocessed" / "dynamic_preprocessed_data"

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


# ── Question map builder ───────────────────────────────────────────────────────

def build_question_map(module_data):
    """Builds question_id → competency mapping from one module JSON."""
    q_map     = {}
    module    = module_data[0]
    course_id = module["key_module_field"]["course_id"]

    for comp_block in module["question_type_mcq"]:
        comp  = comp_block["competency"]
        q_ids = [q.strip().strip("'") for q in comp_block["question"]["question_ids"].split(",")]
        for qid in q_ids:
            if qid:
                q_map[qid] = {
                    "competency_id":   comp["competency_id"],
                    "competency_type": comp["competency_type"],
                    "competency_area": comp["module_compentency_area"],
                }

    # Map checklist questions to last skill competency
    skill_comps = [c["competency"] for c in module["question_type_mcq"]
                   if c["competency"]["competency_type"] == "Skill"]
    for checklist in module.get("question_type_checklist", []):
        qid = str(checklist["question"]["question_id"])
        if skill_comps:
            q_map[qid] = {
                "competency_id":   skill_comps[-1]["competency_id"],
                "competency_type": skill_comps[-1]["competency_type"],
                "competency_area": skill_comps[-1]["module_compentency_area"],
            }

    return course_id, q_map


# ── DataFrame builders ─────────────────────────────────────────────────────────

def build_df1(user_data, question_maps, qid_to_course):
    """DF1: one row per multichoice attempt — IDs and scores only."""
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
                "response_timestamp": attempt["response_timestamp"],
                "competency_id":      comp.get("competency_id", ""),
                "competency_type":    comp.get("competency_type", ""),
                "competency_area":    comp.get("competency_area", ""),
            })
    return pd.DataFrame(rows)


def build_df2(user_data, question_maps, qid_to_course):
    """DF2: one row per step per ordering attempt — IDs and scores only."""
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
                    "user_id":          user_data["user"]["username"],
                    "course_id":        course_id,
                    "question_id":      qid,
                    "option_id":        step["option_id"],
                    "attempt_id":       attempt["attempt_id"],
                    "quiz_id":          attempt["quiz_id"],
                    "quiz_name":        attempt["quiz_name"],
                    "response_timestamp": attempt["response_timestamp"],
                    "user_position":    step["user_position"],
                    "correct_position": step["correct_position"],
                    "is_correct":       step["is_correct"],
                    "position_delta":   abs(step["user_position"] - step["correct_position"]),
                    "marks_awarded":    attempt["marks_awarded"],
                    "max_marks":        q["max_marks"],
                    "competency_id":    comp.get("competency_id", ""),
                    "competency_type":  comp.get("competency_type", ""),
                    "competency_area":  comp.get("competency_area", ""),
                })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load all module question maps
    logger.info("Building question maps from all module JSONs...")
    question_maps  = {}   # course_id → q_map
    qid_to_course  = {}   # question_id → course_id

    for module_path in RAW_MODULE_DIR.glob("*.json"):
        try:
            module_data = load_json(module_path)
            course_id, q_map = build_question_map(module_data)
            question_maps[course_id] = q_map
            for qid in q_map:
                qid_to_course[qid] = course_id
        except Exception as e:
            logger.warning(f"  Could not build map for {module_path.name}: {e}")

    logger.info(f"Loaded question maps for {len(question_maps)} modules\n")

    # Save combined question map
    with open(OUTPUT_DIR / "question_map_all_modules.json", 'w', encoding='utf-8') as f:
        json.dump(question_maps, f, indent=2, ensure_ascii=False)

    # Process all user JSONs
    user_files = sorted(RAW_USER_DIR.glob("user_data_*.json"))
    if not user_files:
        logger.error(f"No user data files found in {RAW_USER_DIR}")
        return

    logger.info(f"Found {len(user_files)} user file(s) to process\n")

    all_df1        = []
    all_df2        = []
    all_unmapped   = set()
    success, failed = 0, []

    for user_path in user_files:
        try:
            user_data = load_json(user_path)
            username  = user_data["user"]["username"]

            df1 = build_df1(user_data, question_maps, qid_to_course)
            df2 = build_df2(user_data, question_maps, qid_to_course)

            all_df1.append(df1)
            all_df2.append(df2)

            # Collect unmapped question IDs for this user
            if not df1.empty:
                unmapped = set(df1[df1["course_id"] == "unknown"]["question_id"].unique())
                all_unmapped.update(unmapped)

            logger.info(f" {username} | mcq attempts: {len(df1)} | checklist steps: {len(df2)}")
            success += 1

        except Exception as e:
            logger.error(f" Failed on {user_path.name}: {e}", exc_info=True)
            failed.append(user_path.name)

    # Save outputs — always overwrite (upsert on rerun)
    if all_df1:
        df1_combined = pd.concat(all_df1, ignore_index=True)
        df1_combined.to_csv(OUTPUT_DIR / "df1_all_users_mcq_attempts.csv", index=False)
        logger.info(f"\nSaved DF1 | shape: {df1_combined.shape}")

    if all_df2:
        df2_combined = pd.concat(all_df2, ignore_index=True)
        df2_combined.to_csv(OUTPUT_DIR / "df2_all_users_ordering_steps.csv", index=False)
        logger.info(f"Saved DF2 | shape: {df2_combined.shape}")

    # Unmapped QID report
    if all_unmapped:
        logger.warning(f"\n Unmapped question IDs (attempted but not in any module JSON):")
        logger.warning(f"  {sorted(all_unmapped)}")
        with open(OUTPUT_DIR / "unmapped_question_ids.json", 'w') as f:
            json.dump(sorted(all_unmapped), f, indent=2)
        logger.info(f"  Saved to unmapped_question_ids.json")
    else:
        logger.info(f"\n All attempted question IDs mapped successfully")

    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Success: {success} | Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed: {failed}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()