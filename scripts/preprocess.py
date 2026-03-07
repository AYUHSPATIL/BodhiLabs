import json
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
INPUT_DIR  = BASE_DIR / "data" / "raw" / "static_data"
OUTPUT_DIR = BASE_DIR / "data" / "preprocessed" / "static_preprocessed_data"

MAX_CHARS = 2048  # 512 tokens * 4 chars/token


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_html(text):
    return re.sub(r'<[^>]+>', '', text)

def parse_quoted_list(text):
    if not text:
        return []
    # Strip outer whitespace, then split on ', ' between quoted items
    text = text.strip()
    items = re.split(r"',\s*'", text)
    # Strip leading/trailing quote from first and last item
    if items:
        items[0] = items[0].lstrip("'")
        items[-1] = items[-1].rstrip("'")
    return [item for item in items if item]

def truncate_if_needed(text, max_chars=MAX_CHARS):
    if len(text) <= max_chars:
        return text
    logger.warning(f"Text truncated: {len(text)} → {max_chars} chars")
    return text[:max_chars]

def safe_get(lst, index, fallback=""):
    """Only used for parallel lists that may genuinely be mismatched."""
    try:
        return lst[index]
    except IndexError:
        logger.warning(f"Index {index} out of range for list of length {len(lst)} — using fallback")
        return fallback


# ── Per-module processor ───────────────────────────────────────────────────────

def process_module(module):
    module_fields = module['key_module_field']
    course_id     = module_fields['course_id']
    module_name   = module_fields['competency_name']
    logger.info(f"  Module: {module_name} (Course ID: {course_id})")

    competencies, mcqs, checklists = [], [], []

    # ── Competencies + MCQs ───────────────────────────────────────────────────
    for comp_group in module['question_type_mcq']:
        comp      = comp_group['competency']
        questions = comp_group['question']

        question_ids        = parse_quoted_list(questions['question_ids'])
        question_texts      = parse_quoted_list(questions['question_texts'])
        correct_options     = parse_quoted_list(questions['correct_options_texts'])
        learning_objectives = parse_quoted_list(questions['question_id_competency_definition'])
        all_options         = parse_quoted_list(questions['options_texts'])

        # Warn if parallel lists are mismatched
        expected = len(question_ids)
        for name, lst in [
            ('question_texts',      question_texts),
            ('correct_options',     correct_options),
            ('learning_objectives', learning_objectives),
            ('all_options',         all_options),
        ]:
            if len(lst) != expected:
                logger.warning(
                    f"  [{comp['competency_id']}] '{name}' length {len(lst)} "
                    f"!= question_ids length {expected}. Will use fallback for missing entries."
                )

        # ── Competency record ─────────────────────────────────────────────────
        if not question_ids:
            logger.info(f"  [{comp['competency_id']}] No question IDs found — competency saved, no MCQs generated.")

        objectives_text = "\n".join(f"- {obj}" for obj in learning_objectives[:10])

        embed_text = truncate_if_needed(
            f"Competency: {comp['module_compentency_area']}\n"
            f"Type: {comp['competency_type']}\n\n"
            f"Definition: {clean_html(comp['module_competency_definition'])}\n\n"
            f"This competency assesses:\n{objectives_text}\n\n"
            f"Activities: {comp['activity_names'].replace(chr(39), '')}"
        )

        competencies.append({
            'id': comp['competency_id'],
            'embedding_text': embed_text,
            'metadata': {
                'doc_type':                     'competency',
                'competency_id':                comp['competency_id'],
                'competency_type':              comp['competency_type'],
                'module_compentency_area':      comp['module_compentency_area'],
                'module_competency_definition': clean_html(comp['module_competency_definition']),
                'activity_names':               comp['activity_names'].replace("'", ""),
                'question_ids':                 json.dumps(question_ids),
                'question_count':               len(question_ids),
                'course_id':                    course_id,
                'competency_name':              module_name,
                'module_domain':                module_fields['module_domain'],
            }
        })

        # ── MCQ records ───────────────────────────────────────────────────────
        for i, qid in enumerate(question_ids):
            q_text  = safe_get(question_texts,      i, fallback="[missing question text]")
            correct = safe_get(correct_options,      i, fallback="[missing correct option]")
            obj     = safe_get(learning_objectives,  i, fallback="[missing learning objective]")
            opts    = safe_get(all_options,          i, fallback="[missing options]")

            embed_text = truncate_if_needed(
                f"Question: {q_text}\n"
                f"Correct Answer: {correct}\n"
                f"Learning Objective: {obj}\n"
                f"All Options: {opts}\n"
                f"Competency: {comp['module_compentency_area']} ({comp['competency_type']})"
            )

            mcqs.append({
                'id': qid,
                'embedding_text': embed_text,
                'metadata': {
                    'doc_type':                          'mcq',
                    'question_id':                       qid,
                    'question_text':                     q_text,
                    'options_text':                      opts,
                    'correct_option_text':               correct,
                    'question_id_competency_definition': obj,
                    'competency_id':                     comp['competency_id'],
                    'competency_type':                   comp['competency_type'],
                    'module_compentency_area':           comp['module_compentency_area'],
                    'course_id':                         course_id,
                    'competency_name':                   module_name,
                    'module_domain':                     module_fields['module_domain'],
                }
            })

    # ── Checklists ────────────────────────────────────────────────────────────
    for checklist in module.get('question_type_checklist', []):
        question   = checklist['question']
        steps      = checklist['option']
        steps_text = "\n".join(f"{s['option_sequence']}. {s['option_text']}" for s in steps)

        embed_text = truncate_if_needed(
            f"Procedural Checklist: {question['question_text']}\n"
            f"Competency: Procedural Skills (Skill)\n"
            f"Correct sequence ({len(steps)} steps):\n"
            f"{steps_text}"
        )

        checklists.append({
            'id': str(question['question_id']),
            'embedding_text': embed_text,
            'metadata': {
                'doc_type':        'checklist',
                'question_id':     str(question['question_id']),
                'question_text':   question['question_text'],
                'total_steps':     len(steps),
                'steps':           json.dumps([
                    {'step_number': s['option_sequence'], 'step_text': s['option_text']}
                    for s in steps
                ]),
                'competency_type': 'Skill',
                'course_id':       course_id,
                'competency_name': module_name,
                'module_domain':   module_fields['module_domain'],
            }
        })

    return {
        'module_info':   module_fields,
        'competencies':  competencies,
        'mcqs':          mcqs,
        'checklists':    checklists,
        'total_vectors': len(competencies) + len(mcqs) + len(checklists),
    }


# ── Main: loop over all JSON files ────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(json_files)} JSON file(s) to process")
    grand_total = 0

    for json_file in json_files:
        logger.info(f"\nProcessing file: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raw_data = [raw_data]

        for module in raw_data:
            try:
                processed = process_module(module)
                course_id = module['key_module_field']['course_id']
                out_path  = OUTPUT_DIR / f"processed_module_{course_id}.json"

                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(processed, f, indent=2, ensure_ascii=False)

                grand_total += processed['total_vectors']
                logger.info(
                    f" Saved {out_path.name} | "
                    f"vectors: {processed['total_vectors']} "
                    f"(comp:{len(processed['competencies'])} "
                    f"mcq:{len(processed['mcqs'])} "
                    f"cl:{len(processed['checklists'])})"
                )

            except Exception as e:
                logger.error(f" Failed on module in {json_file.name}: {e}", exc_info=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"All done! Grand total vectors across all modules: {grand_total}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()