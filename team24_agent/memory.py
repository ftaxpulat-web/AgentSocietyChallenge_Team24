# team24_agent/memory.py
import os
import shutil
import logging
import uuid
from typing import List, Dict
from langchain_chroma import Chroma
from langchain.docstore.document import Document

class GlobalReviewRAG:
    """
    Manages a persistent Vector Database for all users in the dataset.
    Built ONCE by build_knowledge.py.
    """
    def __init__(self, embedding_model, db_path="./global_chroma_db"):
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.vector_store = None
        
        # Initialize immediately if it exists
        if os.path.exists(self.db_path):
            self.vector_store = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embedding_model
            )

    def retrieve(self, user_id: str, query_item_text: str, k: int = 5):
        """
        Retrieve reviews BY this user, SIMILAR to the query item.
        """
        if not self.vector_store:
            return []

        # ChromaDB Filter: ONLY look at reviews written by this specific user
        results = self.vector_store.similarity_search(
            query=query_item_text,
            k=k,
            filter={"user_id": user_id} 
        )
        
        return [doc.metadata for doc in results]


class ReviewRAG:
    """
    Episodic (Temporary) Memory for a single simulation task.
    Creates a tiny Vector DB on the fly, uses it, and deletes it.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        # Unique path for this specific task to avoid collision
        self.db_path = os.path.join('./db_temp', str(uuid.uuid4()))
        self.vector_store = None

    def index_user_history(self, user_reviews: List[Dict], interaction_tool):
        """
        Enrich and index user reviews.
        """
        docs = []
        for r in user_reviews:
            # OPTIMIZATION: We fetch the item metadata for this review
            past_item = interaction_tool.get_item(item_id=r['item_id'])
            if not past_item: continue

            # Construct a rich semantic string
            # "Category: Electronics. Name: Sony Headphones. Review: Great bass..."
            content = f"Category: {past_item.get('categories', 'Unknown')}. Name: {past_item.get('name', 'Unknown')}. Review: {r.get('text', '')}"
            
            docs.append(Document(
                page_content=content,
                metadata={
                    "stars": r.get('stars', 0),
                    "text": r.get('text', ''),
                    "item_name": past_item.get('name', '')
                }
            ))

        if docs:
            self.vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                persist_directory=self.db_path
            )

    def retrieve(self, target_item: Dict, k: int = 5) -> List[Dict]:
        """
        Retrieve reviews similar to the target item.
        """
        if not self.vector_store:
            return []

        # Query using the Target Item's metadata
        query = f"Category: {target_item.get('categories', 'Unknown')}. Name: {target_item.get('name', 'Unknown')}"
        
        results = self.vector_store.similarity_search(query, k=k)
        
        # Unpack the metadata back into a list of dicts
        retrieved_reviews = []
        for doc in results:
            retrieved_reviews.append({
                "text": doc.metadata['text'],
                "stars": doc.metadata['stars'],
                "item_name": doc.metadata['item_name']
            })
        return retrieved_reviews

    def cleanup(self):
        """Delete the temporary database."""
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)