import json
import re
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# Input: Raw module JSON file
INPUT_JSON = r"D:\BodhiLabs\Project\data\raw\urinary_catheterization_with_correctOptionText.json"

# Output: Will be auto-named as processed_module_{course_id}.json
OUTPUT_DIR = "D:\BodhiLabs\Project\data\preprocessed"

# Token limits for MedEmbed-small-v0.1
MAX_CHARS = 2048  # 512 tokens * 4 chars/token


# Helper Functions
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
    logger.warning(f" Text truncated: {len(text)} → {max_chars} chars")
    return text[:max_chars]


# Load Raw Data
logger.info(f"Loading raw data: {INPUT_JSON}")

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

module_fields = raw_data[0]['key_module_field']
course_id = module_fields['course_id']
module_name = module_fields['competency_name']

logger.info(f"Module: {module_name} (Course ID: {course_id})")


# Process Competencies
logger.info("Processing competencies...")

competencies = []

for comp_group in raw_data[0]['question_type_mcq']:
    comp = comp_group['competency']
    questions = comp_group['question']
    
    # Parse question data
    question_ids = parse_quoted_list(questions['question_ids'])
    learning_objectives = parse_quoted_list(questions['question_id_competency_definition'])
    
    # Build embedding text
    embed_text = f"""Competency: {comp['module_compentency_area']}
                    Type: {comp['competency_type']}

                    Definition: {clean_html(comp['module_competency_definition'])}

                    This competency assesses:
                    {chr(10).join(f"- {obj}" for obj in learning_objectives[:10])}

                    Activities: {comp['activity_names'].replace("'", "")}"""
    
    embed_text = truncate_if_needed(embed_text)
    
    # Build metadata
    metadata = {
        'doc_type': 'competency',
        'competency_id': comp['competency_id'],
        'competency_type': comp['competency_type'],
        'module_compentency_area': comp['module_compentency_area'],
        'module_competency_definition': clean_html(comp['module_competency_definition']),
        'activity_names': comp['activity_names'].replace("'", ""),
        'question_ids': json.dumps(question_ids),
        'question_count': len(question_ids),
        'course_id': course_id,
        'competency_name': module_name,
        'module_domain': module_fields['module_domain']
    }
    
    competencies.append({
        'id': comp['competency_id'],
        'embedding_text': embed_text,
        'metadata': metadata
    })

logger.info(f" {len(competencies)} competencies processed")


# Process MCQ Questions
logger.info("Processing MCQ questions...")

mcqs = []

for comp_group in raw_data[0]['question_type_mcq']:
    comp = comp_group['competency']
    questions = comp_group['question']
    
    # Parse parallel arrays
    question_ids = parse_quoted_list(questions['question_ids'])
    question_texts = parse_quoted_list(questions['question_texts'])
    correct_options = parse_quoted_list(questions['correct_options_texts'])
    learning_objectives = parse_quoted_list(questions['question_id_competency_definition'])
    all_options = parse_quoted_list(questions['options_texts'])
    
    # Process each question
    for i, qid in enumerate(question_ids):
        embed_text = f"""Question: {question_texts[i]}
                        Correct Answer: {correct_options[i]}
                        Learning Objective: {learning_objectives[i]}
                        All Options: {all_options[i]}
                        Competency: {comp['module_compentency_area']} ({comp['competency_type']})"""
        
        embed_text = truncate_if_needed(embed_text)
        
        metadata = {
            'doc_type': 'mcq',
            'question_id': qid,
            'question_text': question_texts[i],
            'options_text': all_options[i],
            'correct_option_text': correct_options[i],
            'question_id_competency_definition': learning_objectives[i],
            'competency_id': comp['competency_id'],
            'competency_type': comp['competency_type'],
            'module_compentency_area': comp['module_compentency_area'],
            'course_id': course_id,
            'competency_name': module_name,
            'module_domain': module_fields['module_domain']
        }
        
        mcqs.append({
            'id': qid,
            'embedding_text': embed_text,
            'metadata': metadata
        })

logger.info(f" {len(mcqs)} MCQ questions processed")


# Process Checklists
logger.info("Processing checklists...")

checklists = []

if 'question_type_checklist' in raw_data[0]:
    for checklist in raw_data[0]['question_type_checklist']:
        question = checklist['question']
        steps = checklist['option']
        
        # Build steps text
        steps_text = "\n".join(
            f"{step['option_sequence']}. {step['option_text']}" 
            for step in steps
        )
        
        # Build embedding text
        embed_text = f"""Procedural Checklist: {question['question_text']}
                        Competency: Procedural Skills (Skill)
                        Correct sequence ({len(steps)} steps):
                        {steps_text}"""
        
        # Compress if too long
        if len(embed_text) > MAX_CHARS:
            logger.warning(f" Checklist too long, compressing...")
            embed_text = f"""Procedural Checklist: {question['question_text']}
                            Steps:
                            {steps_text}"""
        
        embed_text = truncate_if_needed(embed_text)
        
        # Build steps metadata
        steps_metadata = [
            {'step_number': step['option_sequence'], 'step_text': step['option_text']}
            for step in steps
        ]
        
        metadata = {
            'doc_type': 'checklist',
            'question_id': str(question['question_id']),
            'question_text': question['question_text'],
            'total_steps': len(steps),
            'steps': json.dumps(steps_metadata),
            'competency_type': 'Skill',
            'course_id': course_id,
            'competency_name': module_name,
            'module_domain': module_fields['module_domain']
        }
        
        checklists.append({
            'id': question['question_id'],
            'embedding_text': embed_text,
            'metadata': metadata
        })

logger.info(f" {len(checklists)} checklists processed")


# Save Processed Data
output_file = f"{OUTPUT_DIR}/processed_module_{course_id}.json"

processed_data = {
    'module_info': module_fields,
    'competencies': competencies,
    'mcqs': mcqs,
    'checklists': checklists,
    'total_vectors': len(competencies) + len(mcqs) + len(checklists)
}

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

logger.info(f"\n{'='*60}")
logger.info(f" Processing complete!")
logger.info(f" Total vectors: {processed_data['total_vectors']}")
logger.info(f"  - Competencies: {len(competencies)}")
logger.info(f"  - MCQs: {len(mcqs)}")
logger.info(f"  - Checklists: {len(checklists)}")
logger.info(f"\n  Saved to: {output_file}")
logger.info(f"{'='*60}")
