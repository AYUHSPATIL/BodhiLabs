import json
import pandas as pd
from pathlib import Path

# relative to script location → works on any machine after cloning
BASE_DIR   = Path(__file__).parent.parent   # Project/
RAW_DIR    = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # create if not exists


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_question_map(module_data):
    """Builds question_id → competency mapping from module JSON."""
    q_map = {}
    module    = module_data[0]
    course_id = module["key_module_field"]["course_id"]

    for comp_block in module["question_type_mcq"]:
        comp  = comp_block["competency"]
        q_ids = [q.strip().strip("'") for q in comp_block["question"]["question_ids"].split(",")]
        for qid in q_ids:
            q_map[qid] = {
                "competency_id":   comp["competency_id"],
                "competency_type": comp["competency_type"],
                "competency_area": comp["module_compentency_area"],
            }

    # map checklist questions to last skill competency
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


def build_df1(user_data, course_id):
    """DF1: one row per multichoice attempt."""
    rows = []
    for q in user_data["questions"]:
        if q["question_type"] != "multichoice":
            continue
        for attempt in q["attempts"]:
            rows.append({
                "user_id":            user_data["user"]["username"],
                "course_id":          course_id,
                "question_id":        str(q["question_id"]),
                "selected_option_id": attempt["selected_options"][0]["option_id"],
                "correct_option_id":  next((o["option_id"] for o in q["options"] if o["is_correct"]), None),
                "is_correct":         attempt["selected_options"][0]["is_correct"],
                "marks_awarded":      attempt["marks_awarded"],
                "max_marks":          q["max_marks"],
                "quiz_id":            attempt["quiz_id"],
                "quiz_name":          attempt["quiz_name"],
                "attempt_id":         attempt["attempt_id"],
                "response_timestamp": attempt["response_timestamp"],
            })
    return pd.DataFrame(rows)


def build_df2(user_data, course_id):
    """DF2: one row per step per ordering attempt."""
    rows = []
    for q in user_data["questions"]:
        if q["question_type"] != "ordering":
            continue
        for attempt in q["attempts"]:
            for step in attempt["selected_sequence"]:
                rows.append({
                    "user_id":            user_data["user"]["username"],
                    "course_id":          course_id,
                    "question_id":        str(q["question_id"]),
                    "option_id":          step["option_id"],
                    "attempt_id":         attempt["attempt_id"],
                    "quiz_id":            attempt["quiz_id"],
                    "quiz_name":          attempt["quiz_name"],
                    "response_timestamp": attempt["response_timestamp"],
                    "user_position":      step["user_position"],
                    "correct_position":   step["correct_position"],
                    "is_correct":         step["is_correct"],
                    "position_delta":     abs(step["user_position"] - step["correct_position"]),
                    "marks_awarded":      attempt["marks_awarded"],
                    "max_marks":          q["max_marks"],
                })
    return pd.DataFrame(rows)


def save_question_map(course_id, q_map):
    output = {"course_id": course_id, "question_map": q_map}
    with open(OUTPUT_DIR / "question_map.json", "w") as f:
        json.dump(output, f, indent=2)


def validate(df1, df2, q_map):
    print(f"DF1 shape : {df1.shape}")
    print(f"DF2 shape : {df2.shape}")
    print(f"Question map size : {len(q_map)}")

    print("\nDF1 attempts by quiz:")
    print(df1.groupby("quiz_name")["attempt_id"].count())

    unmapped = df1[~df1["question_id"].isin(q_map.keys())]["question_id"].unique()
    if len(unmapped):
        print(f"\n Unmapped question_ids: {unmapped}")
    else:
        print("\n All questions mapped")


if __name__ == "__main__":
    user_data   = load_json(RAW_DIR / "user_attempt_data.json")
    module_data = load_json(RAW_DIR / "urinary_catheterization_with_correctOptionText.json")

    course_id, q_map = build_question_map(module_data)

    df1 = build_df1(user_data, course_id)
    df2 = build_df2(user_data, course_id)

    df1.to_csv(OUTPUT_DIR / "df1_user_question_attempts.csv", index=False)
    df2.to_csv(OUTPUT_DIR / "df2_user_ordering_step_details.csv", index=False)
    save_question_map(course_id, q_map)

    validate(df1, df2, q_map)