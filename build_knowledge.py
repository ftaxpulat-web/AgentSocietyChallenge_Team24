import os
import json
from tqdm import tqdm
from team24_agent.llm import GeminiLLM
from langchain_chroma import Chroma
from langchain.docstore.document import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
DB_PATH = os.path.join(BASE_DIR, "global_chroma_db")

def get_item_text(item):
    """
    Helper to normalize item text across Yelp (name), Amazon (title), Goodreads (title).
    """
    # 1. Get Title/Name
    title = item.get('name') or item.get('title') or "Unknown Item"
    
    # 2. Get Category (Handle list vs string)
    cats = item.get('categories', '')
    if isinstance(cats, list):
        # Amazon often has [['Books', 'Fiction'], ['Books', 'Fantasy']]
        # Flatten distinct values
        flat_cats = set()
        for c in cats:
            if isinstance(c, list):
                flat_cats.update(c)
            else:
                flat_cats.add(c)
        cat_str = ", ".join(list(flat_cats)[:5]) # Limit to top 5 tags
    else:
        cat_str = str(cats)
        
    return f"{title} ({cat_str})"

def build_database():
    print(">>> Initializing Embedding Model...")
    llm = GeminiLLM(model="gemini-2.5-flash")
    embeddings = llm.get_embedding_model()

    print(">>> Loading Target Users...")
    task_dir = os.path.join(BASE_DIR, "example", "track1", "amazon", "tasks")
    target_users = set()
    for f in os.listdir(task_dir):
        if f.endswith(".json"):
            with open(os.path.join(task_dir, f)) as file:
                data = json.load(file)
                target_users.add(data['user_id'])
    
    print(">>> Loading Item Metadata...")
    item_map = {}
    item_path = os.path.join(DATA_DIR, "item.json")
    with open(item_path) as f:
        for line in tqdm(f, desc="Indexing Items"):
            item = json.loads(line)
            # Pre-compute the clean text representation
            item_map[item['item_id']] = get_item_text(item)

    print(">>> Building Vector Knowledge Base...")
    review_path = os.path.join(DATA_DIR, "review.json")
    docs = []
    batch_size = 1000
    
    with open(review_path) as f:
        for line in tqdm(f, desc="Processing Reviews"):
            review = json.loads(line)
            uid = review['user_id']
            
            if uid in target_users:
                # Retrieve the clean Item text we computed earlier
                item_desc = item_map.get(review['item_id'], "Unknown Item")
                
                # RICH EMBEDDING CONTENT
                content = f"Item: {item_desc}. Review: {review['text']}"
                
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "user_id": uid,
                        "item_id": review['item_id'],
                        "stars": review['stars'],
                        "text": review['text'],
                        "item_desc": item_desc # Save the clean name for logging
                    }
                ))
                
                if len(docs) >= batch_size:
                    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_PATH)
                    docs = []

    if docs:
        Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_PATH)

    print(f">>> Success! DB built at {DB_PATH}")

if __name__ == "__main__":
    build_database()