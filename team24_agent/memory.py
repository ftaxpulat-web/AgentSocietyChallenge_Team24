# team24_agent/memory.py
import os
import shutil
import logging
import uuid
from typing import List, Dict
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from websocietysimulator.agent.modules.memory_modules import MemoryBase

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

class HybridRAGMemory(MemoryBase):
    """
    Adapter that combines:
    1. GlobalRAG: To find the User's past taste (Long-term memory).
    2. Live Tool: To find the Item's current reputation (Short-term context).
    """
    def __init__(self, llm, db_path="./global_chroma_db", interaction_tool=None):
        # Initialize parent with dummy type since we override logic
        super().__init__(memory_type='hybrid_rag', llm=llm)
        
        # 1. Load the Pre-computed User History DB
        self.rag_engine = GlobalReviewRAG(llm.get_embedding_model(), db_path)
        
        # 2. Tool Access (injected later via set_tool)
        self.interaction_tool = interaction_tool 

    def set_tool(self, tool):
        self.interaction_tool = tool

    def retriveMemory(self, query_scenario: str) -> str:
        """
        Expects query_scenario in format: "USER_ID|ITEM_ID|QUERY_TEXT"
        Returns a formatted string containing both context sets.
        """
        try:
            if "|" not in query_scenario: return ""
            user_id, item_id, query_text = query_scenario.split("|", 2)
            
            # --- PART A: User's Relevant History (From Global DB) ---
            # "How did I rate similar items in the past?"
            user_history_text = "No relevant history found."
            past_reviews = self.rag_engine.retrieve(user_id, query_text, k=3)
            
            if past_reviews:
                blocks = []
                for r in past_reviews:
                    # r is metadata dict: {'item_desc':..., 'stars':..., 'text':...}
                    blocks.append(f"- [{r.get('item_desc','Product')}] ({r.get('stars')} stars): \"{r.get('text','')[:200]}...\"")
                user_history_text = "\n".join(blocks)

            # --- PART B: Item's Community Consensus (From Live Tool) ---
            # "What are other people saying about THIS item?"
            item_consensus_text = "No community reviews available."
            if self.interaction_tool:
                # Fetch recent reviews for the target item
                item_reviews = self.interaction_tool.get_reviews(item_id=item_id)
                # Take top 3 most useful/recent
                if item_reviews:
                    blocks = []
                    for r in item_reviews[:3]:
                        blocks.append(f"- [Community Member] ({r.get('stars')} stars): \"{r.get('text','')[:200]}...\"")
                    item_consensus_text = "\n".join(blocks)

            # --- FORMAT OUTPUT ---
            return f"""
                [User's Past Reviews on Similar Items]
                (Use this to gauge the user's specific taste and rating strictness)
                {user_history_text}

                [Community Reviews of Target Item]
                (Use this to identify the item's actual pros/cons)
                {item_consensus_text}
            """
        except Exception as e:
            logging.error(f"Hybrid Memory Error: {e}")
            return ""

    def addMemory(self, current_situation: str):
        # We use pre-computed DB, so we don't add memory during inference
        pass