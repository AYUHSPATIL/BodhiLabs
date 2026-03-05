import json
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
INPUT_JSON = BASE_DIR / "data" / "raw" / "urinary_catheterization_with_correctOptionText.json"
OUTPUT_DIR = BASE_DIR / "data" / "preprocessed"

MAX_CHARS = 2048  # 512 tokens * 4 chars/token


def clean_html(text):
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)

def parse_quoted_list(text):
    """Parse 'item1', 'item2' into ['item1', 'item2']."""
    return re.findall(r"'([^']*)'", text)

def truncate_if_needed(text, max_chars=MAX_CHARS):
    """Truncate text if too long."""
    if len(text) <= max_chars:
        return text
    logger.warning(f"Text truncated: {len(text)} → {max_chars} chars")
    return text[:max_chars]


logger.info(f"Loading raw data: {INPUT_JSON}")

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

module_fields = raw_data[0]['key_module_field']
course_id     = module_fields['course_id']
module_name   = module_fields['competency_name']

logger.info(f"Module: {module_name} (Course ID: {course_id})")


# ── Process Competencies ───────────────────────────────────────────────────────
logger.info("Processing competencies...")
competencies = []

for comp_group in raw_data[0]['question_type_mcq']:
    comp      = comp_group['competency']
    questions = comp_group['question']

    question_ids       = parse_quoted_list(questions['question_ids'])
    learning_objectives = parse_quoted_list(questions['question_id_competency_definition'])

    objectives_text = "\n".join(f"- {obj}" for obj in learning_objectives[:10])

    # clean embedding text — no indentation leakage
    embed_text = (
        f"Competency: {comp['module_compentency_area']}\n"
        f"Type: {comp['competency_type']}\n\n"
        f"Definition: {clean_html(comp['module_competency_definition'])}\n\n"
        f"This competency assesses:\n{objectives_text}\n\n"
        f"Activities: {comp['activity_names'].replace(chr(39), '')}"
    )

    embed_text = truncate_if_needed(embed_text)

    metadata = {
        'doc_type':                    'competency',
        'competency_id':               comp['competency_id'],
        'competency_type':             comp['competency_type'],
        'module_compentency_area':     comp['module_compentency_area'],
        'module_competency_definition': clean_html(comp['module_competency_definition']),
        'activity_names':              comp['activity_names'].replace("'", ""),
        'question_ids':                json.dumps(question_ids),
        'question_count':              len(question_ids),
        'course_id':                   course_id,
        'competency_name':             module_name,
        'module_domain':               module_fields['module_domain'],
    }

    competencies.append({'id': comp['competency_id'], 'embedding_text': embed_text, 'metadata': metadata})

logger.info(f"{len(competencies)} competencies processed")


# ── Process MCQ Questions ──────────────────────────────────────────────────────
logger.info("Processing MCQ questions...")
mcqs = []

for comp_group in raw_data[0]['question_type_mcq']:
    comp      = comp_group['competency']
    questions = comp_group['question']

    question_ids        = parse_quoted_list(questions['question_ids'])
    question_texts      = parse_quoted_list(questions['question_texts'])
    correct_options     = parse_quoted_list(questions['correct_options_texts'])
    learning_objectives = parse_quoted_list(questions['question_id_competency_definition'])
    all_options         = parse_quoted_list(questions['options_texts'])

    for i, qid in enumerate(question_ids):
        # clean embedding text — no indentation leakage
        embed_text = (
            f"Question: {question_texts[i]}\n"
            f"Correct Answer: {correct_options[i]}\n"
            f"Learning Objective: {learning_objectives[i]}\n"
            f"All Options: {all_options[i]}\n"
            f"Competency: {comp['module_compentency_area']} ({comp['competency_type']})"
        )

        embed_text = truncate_if_needed(embed_text)

        metadata = {
            'doc_type':                        'mcq',
            'question_id':                     qid,
            'question_text':                   question_texts[i],
            'options_text':                    all_options[i],
            'correct_option_text':             correct_options[i],
            'question_id_competency_definition': learning_objectives[i],
            'competency_id':                   comp['competency_id'],
            'competency_type':                 comp['competency_type'],
            'module_compentency_area':         comp['module_compentency_area'],
            'course_id':                       course_id,
            'competency_name':                 module_name,
            'module_domain':                   module_fields['module_domain'],
        }

        mcqs.append({'id': qid, 'embedding_text': embed_text, 'metadata': metadata})

logger.info(f"{len(mcqs)} MCQ questions processed")


# ── Process Checklists ─────────────────────────────────────────────────────────
logger.info("Processing checklists...")
checklists = []

if 'question_type_checklist' in raw_data[0]:
    for checklist in raw_data[0]['question_type_checklist']:
        question   = checklist['question']
        steps      = checklist['option']
        steps_text = "\n".join(f"{s['option_sequence']}. {s['option_text']}" for s in steps)

        # clean embedding text — no indentation leakage
        embed_text = (
            f"Procedural Checklist: {question['question_text']}\n"
            f"Competency: Procedural Skills (Skill)\n"
            f"Correct sequence ({len(steps)} steps):\n"
            f"{steps_text}"
        )

        embed_text = truncate_if_needed(embed_text)

        steps_metadata = [
            {'step_number': s['option_sequence'], 'step_text': s['option_text']}
            for s in steps
        ]

        metadata = {
            'doc_type':       'checklist',
            'question_id':    str(question['question_id']),
            'question_text':  question['question_text'],
            'total_steps':    len(steps),
            'steps':          json.dumps(steps_metadata),
            'competency_type': 'Skill',
            'course_id':      course_id,
            'competency_name': module_name,
            'module_domain':  module_fields['module_domain'],
        }

        checklists.append({'id': question['question_id'], 'embedding_text': embed_text, 'metadata': metadata})

logger.info(f"{len(checklists)} checklists processed")


# ── Save Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / f"processed_module_{course_id}.json"

processed_data = {
    'module_info':    module_fields,
    'competencies':   competencies,
    'mcqs':           mcqs,
    'checklists':     checklists,
    'total_vectors':  len(competencies) + len(mcqs) + len(checklists),
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

logger.info(f"\n{'='*60}")
logger.info(f"Processing complete!")
logger.info(f"Total vectors : {processed_data['total_vectors']}")
logger.info(f"  Competencies: {len(competencies)}")
logger.info(f"  MCQs        : {len(mcqs)}")
logger.info(f"  Checklists  : {len(checklists)}")
logger.info(f"Saved to      : {output_file}")
logger.info(f"{'='*60}")