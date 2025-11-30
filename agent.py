from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json
import os
import shutil
import uuid
import argparse
import logging
import datetime
import numpy as np
from typing import List, Dict, Any, Optional

# --- Imports from your custom modules ---
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase, ReasoningSelfRefine
from websocietysimulator.agent.modules.memory_modules import MemoryDILU

# --- Google GenAI Imports ---
from google import genai
from dotenv import load_dotenv

# --- Vector DB Imports ---
from langchain_chroma import Chroma
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ==========================================
#           HELPER CLASSES
# ==========================================

class GeminiEmbeddingModel:
    """Wrapper for Gemini embeddings, compatible with LangChain/Chroma."""
    def __init__(self, client: genai.Client, model: str = "text-embedding-004"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (list of strings)."""
        vectors: List[List[float]] = []
        # Batching (max 100) is recommended for production, but loop is safe for small N
        for t in texts:
            if not t:
                vectors.append([])
                continue
            try:
                # Task type 'RETRIEVAL_DOCUMENT' is optimal for indexing
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=t,
                    config={'task_type': 'RETRIEVAL_DOCUMENT'}
                )
                vectors.append(resp.embeddings[0].values)
            except Exception as e:
                print(f"Embedding error: {e}")
                vectors.append([]) 
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        try:
            # Task type 'RETRIEVAL_QUERY' is optimal for search queries
            resp = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={'task_type': 'RETRIEVAL_QUERY'}
            )
            return resp.embeddings[0].values
        except Exception as e:
            print(f"Embedding query error: {e}")
            return []

class GeminiLLM(LLMBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash", 
        embedding_model: str = "text-embedding-004",
    ):
        super().__init__(model=model)

        load_dotenv()
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set and no api_key passed to GeminiLLM")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.embedding_model_name = embedding_model
        self._embedding_model = GeminiEmbeddingModel(self.client, model=embedding_model)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> str:
        model_name = model or self.model_name
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            gemini_role = "model" if role == "assistant" else role
            contents.append({
                "role": gemini_role,
                "parts": [{"text": m.get("content", "")}],
            })

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            # Robust parsing for safety blocks or empty responses
            try:
                if response.text:
                    return response.text
            except Exception:
                pass

            if response.candidates:
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and 
                    candidate.content and 
                    hasattr(candidate.content, 'parts') and 
                    candidate.content.parts):
                    return "".join(getattr(p, "text", "") for p in candidate.content.parts)
            
            print(f"Warning: Gemini returned empty response. Finish reason: {getattr(response.candidates[0], 'finish_reason', 'Unknown') if response.candidates else 'No candidates'}")
            return ""

        except Exception as e:
            print("Error calling Gemini:", repr(e))
            return ""

    def get_embedding_model(self):
        return self._embedding_model

class DummyMemory:
    def __init__(self): self.store = []
    def __call__(self, text=""): return ""

# ==========================================
#           NEW RAG MEMORY MODULE
# ==========================================

class ReviewRAG:
    """
    Episodic Memory for a single simulation task.
    Indexes a user's past reviews to find those most relevant to the current item.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        # Unique path for this specific task/agent instance to avoid collision
        self.db_path = os.path.join('./db_temp', str(uuid.uuid4()))
        self.vector_store = None

    def index_user_history(self, user_reviews: List[Dict], interaction_tool):
        """
        Enrich and index user reviews.
        We fetch the ITEM details for each review to make the embedding semantic.
        """
        docs = []
        for r in user_reviews:
            # OPTIMIZATION: We fetch the item metadata for this review
            # This allows us to match "Electronics" query to "Electronics" history
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
        # "Category: Electronics. Name: Bose Headphones"
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

# ==========================================
#           THE AGENT
# ==========================================

class MySimulationAgent(SimulationAgent):
    def __init__(self, llm: LLMBase, config: dict = None):
        super().__init__(llm=llm)
        self.config = config or {}
        
        # 1. Configure RAG Memory (Review Retrieval)
        self.memory_type = self.config.get("memory_type", "none")
        self.rag = None # Initialized per task in workflow if enabled

        # 2. Configure Reflection
        self.use_reflection = self.config.get("use_reflection", False)
        if self.use_reflection:
            self.refiner = ReasoningSelfRefine(profile_type_prompt='', memory=DummyMemory(), llm=self.llm)

    def _call_llm(self, prompt: str, temp: float = 0.4) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self.llm(messages=messages, temperature=temp, max_tokens=4096)
        return result.strip() if isinstance(result, str) else ""

    # ---------- Stage 1: Persona (Enhanced with RAG) ----------
    def _stage1_persona(self, user, relevant_reviews: list) -> str:
        
        # Format the selected reviews
        reviews_text_block = ""
        if not relevant_reviews:
            reviews_text_block = "The user has no relevant prior reviews."
        else:
            for i, r in enumerate(relevant_reviews):
                # We include the Item Name if RAG retrieved it, otherwise generic
                item_label = r.get('item_name', 'Product')
                reviews_text_block += f"- Review {i+1} on [{item_label}] (stars: {r.get('stars', 'N/A')}): {r.get('text', '')}\n\n"

        prompt = f"""
You are analyzing a Yelp user's historical behavior.
User profile: {user}

Here are the user's past reviews *for products similar to the one currently being viewed*:
{reviews_text_block}

Task:
1. Summarize typical preferences (especially for this category of items).
2. Summarize tone and writing style.
3. Summarize rating behavior.

Output 2-4 sentences in plain English.
"""
        return self._call_llm(prompt)

    # ---------- Stage 2: Rating & Plan ----------
    def _stage2_rating_plan(self, persona: str, user, business, similar_reviews_text: str) -> str:
        prompt = f"""
You are simulating a real human Yelp user writing a new review.
User persona: {persona}
Business: {business}
Relevant reviews from others:
{similar_reviews_text}

Tasks:
1. Decide rating (1.0-5.0).
2. Explain why.
3. Outline 3-5 specific points.

Format:
rating: [rating]
explanation: [text]
outline:
- [point 1]
- [point 2]
"""
        plan = self._call_llm(prompt)

        if self.use_reflection:
            refine_prompt = f"""
Reflect on the following review plan. Does the rating match the user persona? Are the points specific enough?
Original Plan:
{plan}
If it is good, output the Original Plan exactly. If it needs improvement, output the improved plan maintaining the same format.
"""
            return self._call_llm(refine_prompt, temp=0.1)
        return plan

    def _parse_rating_from_plan(self, plan: str) -> float:
        if not plan: return 3.0
        try:
            lines = plan.splitlines()
            rating_line = next((line for line in lines if "rating:" in line), None)
            if rating_line:
                val = float(rating_line.split(":", 1)[1].strip())
                if val in {1.0, 2.0, 3.0, 4.0, 5.0}: return val
        except:
            pass
        return 3.0 

    # ---------- Stage 3: Final Review ----------
    def _stage3_final_review(self, persona: str, business, rating: float, plan: str) -> str:
        prompt = f"""
User persona: {persona}
Business: {business}
Rating: {rating}
Plan:
{plan}

Write the FINAL review (2-4 sentences). Match tone/style. No extra labels.
"""
        return self._call_llm(prompt)[:2048]

    # ---------- Main Workflow ----------
    def workflow(self):
        # Initialize RAG for this task if enabled
        rag_memory = None
        try:
            # 1. Info Retrieval
            user_obj = self.interaction_tool.get_user(user_id=self.task['user_id'])
            item_obj = self.interaction_tool.get_item(item_id=self.task['item_id'])
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            # 2. RAG Logic (The "Memory" Module)
            selected_reviews = []
            if self.memory_type == "dilu":
                # Create ephemeral index for this user
                rag_memory = ReviewRAG(self.llm.get_embedding_model())
                # Index user's history enriched with item data
                history_cap = 30
                rag_memory.index_user_history(reviews_user[:history_cap], self.interaction_tool)
                # Retrieve top 5 reviews relevant to the CURRENT item
                selected_reviews = rag_memory.retrieve(target_item=item_obj, k=5)
            else:
                # Baseline: Just take the last 5
                selected_reviews = reviews_user[:5]

            # 3. Execution Pipeline
            persona = self._stage1_persona(str(user_obj), selected_reviews)
            
            # Simple retrieval of "similar" reviews from OTHER users (Baseline context)
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            similar_reviews_text = "\n".join([r.get('text','') for r in reviews_item[:3]])
            
            plan = self._stage2_rating_plan(persona, str(user_obj), str(item_obj), similar_reviews_text)
            rating = self._parse_rating_from_plan(plan)
            review = self._stage3_final_review(persona, str(item_obj), rating, plan)
            
            return {"stars": float(rating), "review": review}
            
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {"stars": 0.0, "review": ""}
        finally:
            # Cleanup temp database
            if rag_memory:
                rag_memory.cleanup()

# ==========================================
#           EXPERIMENT RUNNER
# ==========================================

if __name__ == "__main__":
    import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment Name")
    parser.add_argument("--memory", type=str, default="none", choices=["none", "dilu"], help="Memory type")
    parser.add_argument("--reflection", action="store_true", help="Enable self-reflection")
    parser.add_argument("--tasks", type=int, default=10, help="Task count")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="LLM Model")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset") # use the full dataset
    TASK_SET = "amazon"
    TASK_DIR = os.path.join(BASE_DIR, "example", "track1", TASK_SET, "tasks")
    GT_DIR = os.path.join(BASE_DIR, "example", "track1", TASK_SET, "groundtruth")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", args.exp_name, timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = {
        "exp_name": args.exp_name,
        "timestamp": timestamp,
        "memory_type": args.memory,
        "use_reflection": args.reflection,
        "model": args.model,
        "task_count": args.tasks
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n>>> STARTING EXPERIMENT: {args.exp_name} ({timestamp})")
    
    try:
        simulator = Simulator(data_dir=DATA_DIR, device="auto", cache=True)
        simulator.set_task_and_groundtruth(task_dir=TASK_DIR, groundtruth_dir=GT_DIR)

        class ConfiguredAgent(MySimulationAgent):
            def __init__(self, llm):
                super().__init__(llm, config=config)

        simulator.set_agent(ConfiguredAgent)
        simulator.set_llm(GeminiLLM(model=args.model))
        
        print(f"Running {args.tasks} tasks...")
        outputs = simulator.run_simulation(number_of_tasks=args.tasks, enable_threading=False)
        
        print("Evaluating...")
        evaluation_results = simulator.evaluate()       
        
        # Calculate RMSE
        ground_truth_subset = simulator.groundtruth_data[:len(outputs)]
        stars_pred = []
        stars_real = []
        detailed_logs = []
        
        for i, (agent_res, gt_res) in enumerate(zip(outputs, ground_truth_subset)):
            if not agent_res or 'output' not in agent_res: continue

            pred_star = agent_res['output'].get('stars', 0.0)
            real_star = gt_res.get('stars', 0.0)
            stars_pred.append(pred_star)
            stars_real.append(real_star)
            
            detailed_logs.append({
                "task_id": i,
                "user_id": agent_res['task']['user_id'],
                "item_id": agent_res['task']['item_id'],
                "ground_truth_stars": real_star,
                "predicted_stars": pred_star,
                "ground_truth_review": gt_res.get('review', ''),
                "predicted_review": agent_res['output'].get('review', '')
            })

        if stars_pred:
            mse = np.mean((np.array(stars_pred) - np.array(stars_real)) ** 2)
            rmse = np.sqrt(mse)
            evaluation_results['metrics']['rmse'] = float(rmse)
            print(f">>> Calculated RMSE: {rmse:.4f}")

        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(evaluation_results, f, indent=4)
        with open(os.path.join(OUTPUT_DIR, "detailed_logs.json"), "w") as f:
            json.dump(detailed_logs, f, indent=4)

        print(f"\n>>> SUCCESS! Output: {OUTPUT_DIR}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()