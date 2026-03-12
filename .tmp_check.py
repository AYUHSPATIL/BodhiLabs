import json
from pathlib import Path

BASE_DIR = Path('d:/BodhiLabs/Project')
RAW_MODULE_DIR = BASE_DIR / 'data' / 'raw' / 'static_data'

qid_to_course = {}
for module_path in RAW_MODULE_DIR.glob('*.json'):
    module_data = json.load(open(module_path, encoding='utf-8'))
    module = module_data[0]
    course_id = module['key_module_field']['course_id']
    for comp_block in module['question_type_mcq']:
        q_ids = [q.strip().strip("'") for q in comp_block['question']['question_ids'].split(',')]
        for qid in q_ids:
            if qid:
                qid_to_course[qid] = course_id
    for checklist in module.get('question_type_checklist', []):
        qid = str(checklist['question']['question_id'])
        qid_to_course[qid] = course_id

for qid in ['1901','1903','1907','1921','1922','1932','1937','1958']:
    print(qid, qid_to_course.get(qid))

print('347 count', sum(1 for v in qid_to_course.values() if v=='347'))
