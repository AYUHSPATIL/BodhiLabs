import json
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Input
PROCESSED_JSON = "D:\BodhiLabs\Project\data\preprocessed\processed_module_341.json"

# Output
CHROMA_DIR = "D:/BodhiLabs/Project/vectorstore/chroma"

# Embedding model (fine-tuned on medical data)
EMBED_MODEL = "abhinand/MedEmbed-small-v0.1"

# Initialize Embedding Model
logger.info(f"Loading embedding model: {EMBED_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logger.info(" Embedding model loaded")


# Load Preprocessed Documents
def load_module_documents(json_path: str):
    """Load documents from preprocessed JSON."""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    module_info = data['module_info']
    collection_name = f"module_{module_info['course_id']}_{module_info['competency_name'].lower().replace(' ', '_')}"
    
    documents = []
    ids = []
    
    # Process competencies
    for comp in data['competencies']:
        doc = Document(
            page_content=comp['embedding_text'],
            metadata=comp['metadata']
        )
        documents.append(doc)
        ids.append(str(comp['id']))
    
    # Process MCQs
    for mcq in data['mcqs']:
        doc = Document(
            page_content=mcq['embedding_text'],
            metadata=mcq['metadata']
        )
        documents.append(doc)
        ids.append(str(mcq['id']))
    
    # Process checklists
    for checklist in data['checklists']:
        doc = Document(
            page_content=checklist['embedding_text'],
            metadata=checklist['metadata']
        )
        documents.append(doc)
        ids.append(str(checklist['id']))
    
    logger.info(f"Loaded {len(documents)} documents")
    logger.info(f"- Competencies: {len(data['competencies'])}")
    logger.info(f"- MCQs: {len(data['mcqs'])}")
    logger.info(f"- Checklists: {len(data['checklists'])}")
    
    return documents, ids, collection_name


# Build Vector Store
logger.info("Loading preprocessed documents")
documents, ids, collection_name = load_module_documents(PROCESSED_JSON)

logger.info(f"Building/Updating Chroma vectorstore: {collection_name}")
logger.info(f"Total vectors: {len(documents)}")

# Get or create collection 
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# Check if collection already has data
existing_count = vectorstore._collection.count()
if existing_count > 0:
    logger.warning(f" Collection already exists with {existing_count} vectors")
    logger.warning(f" Upserting {len(documents)} vectors (will replace duplicates by ID)")

# Upsert documents (replaces if ID exists, adds if new)
vectorstore.add_documents(documents=documents, ids=ids)

logger.info(" Vectorstore successfully built/updated")
logger.info(f" Final vector count: {vectorstore._collection.count()}")
logger.info(f" Saved to: {CHROMA_DIR}/{collection_name}")


# logger.info("\nTesting vectorstore with sample query...")

# test_query = "catheter balloon inflation procedure"
# results = vectorstore.similarity_search(
#     test_query,
#     k=3,
#     filter={"doc_type": "mcq"}
# )

# logger.info(f"Query: '{test_query}'")
# logger.info(f"Found {len(results)} results:\n")

# for i, result in enumerate(results, 1):
#     print(f"{i}. Question ID: {result.metadata.get('question_id')}")
#     print(f"   Type: {result.metadata.get('doc_type')}")
#     print(f"   Preview: {result.page_content[:100]}...")
#     print()

logger.info(" All done!")
