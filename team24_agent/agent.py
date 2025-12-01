import logging
import json 
from typing import Dict, Any, Optional

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase

from .prompts import (
    PERSONA_PROMPT, 
    PLANNING_PROMPT, 
    REFLECTION_PROMPT, 
    FINAL_WRITING_PROMPT
)
from .memory import GlobalReviewRAG, ReviewRAG 
from .config import ExperimentConfig

logger = logging.getLogger("team24_agent")

class MySimulationAgent(SimulationAgent):
    """
    Refactored Agent supporting:
    1. Hybrid RAG (Recent History + Relevant History)
    2. Global Pre-computed Knowledge Base
    3. Reflection (Self-Correction)
    """

    def __init__(self, llm: LLMBase, config: ExperimentConfig = None):
        super().__init__(llm=llm)
        self.config = config or ExperimentConfig()
        
        # Initialize Memory
        self.rag_engine = None
        if self.config.memory_type == "global_rag":
            logger.info(f"Connecting to Global Vector DB at {self.config.global_db_path}")
            self.rag_engine = GlobalReviewRAG(
                embedding_model=self.llm.get_embedding_model(),
                db_path=self.config.global_db_path
            )
        
        logger.info(f"Agent Initialized. Reflection: {self.config.use_reflection}")

    def _call_llm(self, prompt: str, temp: float = 0.4) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self.llm(messages=messages, temperature=temp, max_tokens=4096)
        return result.strip() if isinstance(result, str) else ""

    def _get_clean_item_text(self, item):
        """Helper to create the search query from item metadata"""
        title = item.get('name') or item.get('title') or "Unknown Item"
        cats = item.get('categories', '')
        if isinstance(cats, list):
            flat_cats = set()
            for c in cats:
                if isinstance(c, list): flat_cats.update(c)
                else: flat_cats.add(c)
            cat_str = ", ".join(list(flat_cats)[:5])
        else:
            cat_str = str(cats)
        return f"{title} ({cat_str})"

    # ---------- Stage 1: Persona (Hybrid RAG) ----------
    def _stage1_persona(self, user_profile: str, recent_reviews: list, relevant_reviews: list) -> str:
        print("  -> Generating Persona (Hybrid)...") 

        # Helper to format reviews nicely
        def format_reviews(revs):
            if not revs: return "None."
            block = ""
            for i, r in enumerate(revs):
                # Handle both dicts (recent) and Chroma metadata (relevant)
                item = r.get('item_name') or r.get('item_desc', 'Product')
                text = r.get('text', '')
                stars = r.get('stars', 'N/A')
                block += f"- [{item}] ({stars} stars): {text}\n"
            return block

        # Populate the prompt with BOTH sets
        prompt = PERSONA_PROMPT.format(
            user_profile=user_profile, 
            recent_reviews=format_reviews(recent_reviews),
            relevant_reviews=format_reviews(relevant_reviews)
        )
        return self._call_llm(prompt)

    # ---------- Stage 2: Planning & Reflection ----------
    def _stage2_plan(self, persona: str, business_info: str, similar_reviews: str) -> str:
        print("  -> Planning Rating & Outline...") 
        prompt = PLANNING_PROMPT.format(
            persona=persona,
            business=business_info,
            similar_reviews=similar_reviews
        )
        plan = self._call_llm(prompt)

        if self.config.use_reflection:
            print("  -> Reflecting/Critiquing Plan...") 
            critic_prompt = REFLECTION_PROMPT.format(plan=plan)
            refined_plan = self._call_llm(critic_prompt, temp=0.1)
            return refined_plan
        
        return plan

    def _parse_rating(self, plan: str) -> float:
        if not plan: return 3.0
        try:
            lines = plan.splitlines()
            rating_line = next((line for line in lines if "rating:" in line), None)
            if rating_line:
                val_part = rating_line.split(":", 1)[1].strip().split()[0]
                val = float(val_part)
                if 1.0 <= val <= 5.0: return val
        except Exception:
            pass
        return 3.0 

    # ---------- Stage 3: Writing ----------
    def _stage3_write(self, persona: str, business: str, rating: float, plan: str) -> str:
        print("  -> Drafting Final Review...") 
        prompt = FINAL_WRITING_PROMPT.format(
            persona=persona,
            business=business,
            rating=rating,
            plan=plan
        )
        return self._call_llm(prompt)[:2048]

    # ---------- Main Workflow ----------
    def workflow(self):
        try:
            print(f"\n[Task Start] User: {self.task['user_id']} | Item: {self.task['item_id']}") 
            
            user_obj = self.interaction_tool.get_user(user_id=self.task['user_id'])
            item_obj = self.interaction_tool.get_item(item_id=self.task['item_id'])
            
            # 1. SET A: Recent History (For Voice)
            # The interaction_tool returns list; usually index 0 is most recent or oldest
            # We take a slice to get a snapshot of 'general' writing style
            all_user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            recent_reviews_set = all_user_reviews[:3]

            # 2. SET B: Relevant History (For Stance)
            relevant_reviews_set = []
            rag_details = [] 
            
            if self.config.memory_type == "global_rag" and self.rag_engine:
                print("  -> Retrieving Relevant Context...")
                query_text = f"Item: {self._get_clean_item_text(item_obj)}"
                
                # Fetch top 3 relevant reviews
                relevant_reviews_set = self.rag_engine.retrieve(
                    user_id=self.task['user_id'],
                    query_item_text=query_text,
                    k=3 
                )
                
                # Logging details
                for r in relevant_reviews_set:
                    rag_details.append({
                        "user_check": r.get('user_id'),
                        "item": r.get('item_desc', 'Unknown'),
                        "stars": r.get('stars'),
                        "snippet": r.get('text', '')[:100] + "..."
                    })

            # 3. Pipeline Execution
            persona = self._stage1_persona(str(user_obj), recent_reviews_set, relevant_reviews_set)
            
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            similar_reviews_text = "\n".join([r.get('text','') for r in reviews_item[:3]])
            
            plan = self._stage2_plan(persona, str(item_obj), similar_reviews_text)
            rating = self._parse_rating(plan)
            review = self._stage3_write(persona, str(item_obj), rating, plan)
            
            return {
                "stars": float(rating),
                "review": review,
                "rag_context": json.dumps(rag_details, indent=2), 
                "persona": persona
            }
            
        except Exception as e:
            print(f"Error in workflow: {e}")
            import traceback
            traceback.print_exc()
            return {"stars": 0.0, "review": ""}