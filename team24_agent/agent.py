import logging
import re
from typing import Any, Dict

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase

# Import standard modules
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningIO, ReasoningCOT, ReasoningCOTSC, ReasoningTOT,
    ReasoningDILU, ReasoningSelfRefine, ReasoningStepBack,
)
from websocietysimulator.agent.modules.memory_modules import (
    MemoryDILU, MemoryGenerative, MemoryTP, MemoryVoyager
)

from team24_agent.memory import HybridRAGMemory

from .config import ExperimentConfig
from .prompts import TASK_DESCRIPTION

logger = logging.getLogger("team24_modular_agent")

REASONING_REGISTRY = {
    "io": ReasoningIO,
    "cot": ReasoningCOT,
    "cotsc": ReasoningCOTSC,
    "tot": ReasoningTOT,
    "dilu": ReasoningDILU,
    "self_refine": ReasoningSelfRefine,
    "step_back": ReasoningStepBack,
}

MEMORY_REGISTRY = {
    "none": None,
    "dilu": MemoryDILU,
    "generative": MemoryGenerative,
    "tp": MemoryTP,
    "voyager": MemoryVoyager,
    "hybrid_rag": HybridRAGMemory,
}

class Team24Agent(SimulationAgent):
    def __init__(self, llm: LLMBase, config: ExperimentConfig):
        super().__init__(llm=llm)
        self.config = config

        # 1) Memory Initialization
        memory_cls = MEMORY_REGISTRY.get(config.memory_type)
        self.memory = memory_cls(llm=self.llm) if memory_cls is not None else None

        # 2) Reasoning Initialization
        reasoning_cls = REASONING_REGISTRY.get(config.reasoning_type, ReasoningIO)
        self.reasoning = reasoning_cls(
            profile_type_prompt="",   
            memory=self.memory,
            llm=self.llm,
        )

    # --- HELPER: Clean User Profile ---
    def _format_user_profile(self, user_data: dict) -> str:
        if not user_data: return "Unknown User"
        name = user_data.get('name', 'Anonymous')
        rev_count = user_data.get('review_count', 0)
        avg_stars = user_data.get('average_stars', 'N/A')
        since = str(user_data.get('yelping_since', 'Unknown')).split()[0]
        elite = user_data.get('elite')
        elite_str = f"| Elite Status: {elite}" if elite and elite != 'None' else ""
        return f"Name: {name}\nStats: {rev_count} reviews, Avg Rating {avg_stars}, Member since {since} {elite_str}"

    # --- HELPER: Clean Item Profile ---
    def _format_item_profile(self, item_data: dict) -> str:
        """
        Flattens Yelp 'attributes' and includes Community Rating.
        """
        if not item_data: return "Unknown Business"

        name = item_data.get('name', 'Unknown')
        categories = item_data.get('categories', '')
        
        # --- NEW: Grab the Community Rating ---
        community_rating = item_data.get('stars', 'N/A')
        
        # Unpack Attributes
        attrs = item_data.get('attributes', {})
        details = []
        
        if attrs and isinstance(attrs, dict):
            for k, v in attrs.items():
                val_str = str(v).replace("u'", "").replace("'", "").replace("{", "").replace("}", "")
                details.append(f"- {k}: {val_str}")

        # Limit to top 10 attributes
        details_block = "\n".join(details[:10]) 

        # Return string with Community Rating included
        return f"""
            Business: {name}
            Community Rating: {community_rating} / 5.0
            Categories: {categories}
            Attributes:
            {details_block}
        """

    def workflow(self) -> Dict[str, Any]:
        try:
            user_id = self.task["user_id"]
            item_id = self.task["item_id"]

            # --- Fetch & Format Data ---
            user_obj = self.interaction_tool.get_user(user_id=user_id)
            item_obj = self.interaction_tool.get_item(item_id=item_id)
            
            clean_user = self._format_user_profile(user_obj)
            clean_item = self._format_item_profile(item_obj)

            # --- HYBRID RAG MEMORY ---
            rag_context_str = "(No memory context available)"
            
            if self.memory:
                # 1. Inject Tool (Crucial for fetching Item Consensus)
                if hasattr(self.memory, 'set_tool'):
                    self.memory.set_tool(self.interaction_tool)
                
                # 2. Construct Query for Global DB
                # Helper to get "Title (Category)" for semantic search
                item_title = item_obj.get('name') or item_obj.get('title') or "Unknown"
                cats = item_obj.get('categories', '')
                # Handle list of lists if necessary, or just stringify
                cat_str = str(cats)
                
                # Packed format: USER_ID | ITEM_ID | QUERY_TEXT
                query_payload = f"{user_id}|{item_id}|Item: {item_title} ({cat_str})"
                
                # 3. Retrieve
                # This returns the big formatted string from HybridRAGMemory
                rag_context_str = self.memory(query_payload)

            # --- Build Prompt ---
            task_desc_str = TASK_DESCRIPTION.format(
                user=clean_user,
                business=clean_item,
                rag_context=rag_context_str, # Injects both history and community opinions
            )

            # --- Reasoning (CoT) ---
            result = self.reasoning(task_desc_str)

            # --- Parse Output (Same as before) ---
            stars = 0.0
            review_text = ""
            try:
                lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
                
                stars_line = next((ln for ln in lines if "stars:" in ln.lower()), None)
                if stars_line:
                    match = re.search(r"stars:\s*([0-9.]+)", stars_line, re.IGNORECASE)
                    if match: stars = float(match.group(1))

                review_line_index = next((i for i, ln in enumerate(lines) if "review:" in ln.lower()), None)
                if review_line_index is not None:
                    first_line_content = lines[review_line_index].split(":", 1)[1].strip()
                    rest_of_content = " ".join(lines[review_line_index+1:])
                    review_text = f"{first_line_content} {rest_of_content}".strip()
                
                if not review_text and stars_line:
                    review_text = result.replace(stars_line, "").strip()

            except Exception as parse_e:
                logger.warning(f"Failed to parse output: {parse_e}")

            if stars <= 0 or stars > 5: stars = 3.0
            if not review_text: review_text = "Review generation failed."

            return {"stars": float(stars), "review": review_text[:512]}

        except Exception as e:
            logger.exception(f"Error in workflow: {e}")
            return {"stars": 0.0, "review": ""}