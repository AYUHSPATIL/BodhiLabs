import json
import logging
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed" / "static_preprocessed_data"
CHROMA_DIR      = str(BASE_DIR / "vectorstore" / "chroma")

EMBED_MODEL = "abhinand/MedEmbed-small-v0.1"

# ── Load embedding model once (shared across all modules) ─────────────────────
logger.info(f"Loading embedding model: {EMBED_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logger.info("Embedding model loaded")


# ── Load documents from one preprocessed JSON ─────────────────────────────────
def load_module_documents(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    module_info     = data['module_info']
    collection_name = (
        f"module_{module_info['course_id']}_"
        f"{module_info['competency_name'].lower().replace(' ', '_')}"
    )
    collection_name = re.sub(r'[^a-zA-Z0-9._-]', '', collection_name)

    documents, ids = [], []

    for comp in data['competencies']:
        documents.append(Document(page_content=comp['embedding_text'], metadata=comp['metadata']))
        ids.append(str(comp['id']))

    for mcq in data['mcqs']:
        documents.append(Document(page_content=mcq['embedding_text'], metadata=mcq['metadata']))
        ids.append(str(mcq['id']))

    for checklist in data['checklists']:
        documents.append(Document(page_content=checklist['embedding_text'], metadata=checklist['metadata']))
        ids.append(str(checklist['id']))

    return documents, ids, collection_name


# ── Main: loop over all preprocessed JSONs ────────────────────────────────────
def main():
    pre_files = sorted(PREPROCESSED_DIR.glob("processed_module_*.json"))
    if not pre_files:
        logger.error(f"No preprocessed files found in {PREPROCESSED_DIR}")
        return

    logger.info(f"Found {len(pre_files)} preprocessed file(s) to vectorize\n")
    grand_total = 0

    for pre_path in pre_files:
        logger.info(f"Processing: {pre_path.name}")
        try:
            documents, ids, collection_name = load_module_documents(pre_path)

            logger.info(f"  Collection : {collection_name}")
            logger.info(f"  Documents  : {len(documents)}")

            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR
            )

            existing_count = vectorstore._collection.count()
            if existing_count > 0:
                logger.warning(f"  Collection already exists with {existing_count} vectors — upserting")

            vectorstore.add_documents(documents=documents, ids=ids)

            final_count = vectorstore._collection.count()
            grand_total += final_count
            logger.info(f" Final vector count: {final_count}")

        except Exception as e:
            logger.error(f" Failed on {pre_path.name}: {e}", exc_info=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"All done! Total vectors across all collections: {grand_total}")
    logger.info(f"Saved to: {CHROMA_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()