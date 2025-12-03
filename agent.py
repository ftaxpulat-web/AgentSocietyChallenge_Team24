from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
import logging
logging.basicConfig(level=logging.INFO)

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None', 
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        return reasoning_result


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent with a 3-stage pipeline."""
    USE_MEMORY = False          # ablation toggle
    USE_REFLECTION = True
    MAX_ITEM_MEM = 25           # limit how many item reviews you embed/store
    _shared_memory = None       # shared across tasks in one run

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        
        use_mem = bool(getattr(self.__class__, "USE_MEMORY", False))

        if use_mem:
            # One shared memory instance across all tasks in this run
            if self.__class__._shared_memory is None:
                self.__class__._shared_memory = MemoryDILU(llm=self.llm)
            self.memory = self.__class__._shared_memory
        else:
            self.memory = DummyMemory()


    # ---------- Helper: safe LLM call with one prompt string ----------
    def _call_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self.llm(
            messages=messages,
            temperature=0.4,   # a bit of creativity, but not too wild
            max_tokens=512
        )
        if not isinstance(result, str):
            print("LLM returned non-string:", repr(result))
            return ""
        return result.strip()

    # ---------- Stage 1: Persona & style inference ----------
    def _stage1_persona(self, user, user_reviews: list) -> str:
        """
        Build a textual persona summary for the user based on their past reviews.
        """
        # Use up to K user reviews to keep prompt manageable
        K = 5
        selected_reviews = user_reviews[:K]
        reviews_text_block = "\n\n".join(
            f"- Review {i+1} (stars: {r.get('stars', 'N/A')}): {r.get('text', '')}"
            for i, r in enumerate(selected_reviews)
        )
        if not reviews_text_block:
            reviews_text_block = "The user has no prior reviews."

        prompt = f"""
You are analyzing a Yelp user's historical behavior.

User profile:
{user}

Here are some of the user's past reviews and ratings:
{reviews_text_block}

Your task:
1. Summarize this user's typical preferences (e.g., what they care about most).
2. Summarize their typical tone and writing style (e.g., funny, blunt, detailed).
3. Summarize their usual rating behavior (e.g., usually gives 4-5 stars, harsh with 1-2 stars, etc.).

Output your answer as 2-4 sentences in plain English, addressing all three points.
"""
        persona = self._call_llm(prompt)
        # Optionally store persona in memory for reuse
        self.memory(f"review: persona_summary: {persona}")
        return persona

    # ---------- Stage 2: Rating & content planning ----------
    def _stage2_rating_plan(self, persona: str, user, business, similar_reviews_text: str) -> str:
        """
        Decide on rating and outline main points to mention in the review.
        Returns the raw LLM plan output (we'll parse rating later).
        """
        prompt = f"""
You are simulating a real human Yelp user writing a new review.

User persona (based on their historical behavior):
{persona}

Business to review:
{business}

Some relevant reviews about this business from other users:
{similar_reviews_text}

Your tasks:
1. Decide what star rating this user would likely give this business, as one of {{1.0, 2.0, 3.0, 4.0, 5.0}}.
2. Explain briefly why, in terms of how this business meets or fails their preferences (from the persona).
3. Outline 3-5 specific points that the user would mention in their review (bullet points).

You MUST output in exactly the following format:

rating: [one of 1.0, 2.0, 3.0, 4.0, 5.0]
explanation: [1-2 sentences explaining the rating]
outline:
- [point 1]
- [point 2]
- [point 3]
[optional more bullet points]

Do not include any other sections or headings.
"""
        plan = self._call_llm(prompt)
        return plan

    def _parse_rating_from_plan(self, plan: str) -> float:
        """
        Extract numeric rating from the Stage 2 plan.
        """
        if not plan:
            return 0.0
        lines = plan.splitlines()
        rating_lines = [line for line in lines if "rating:" in line]
        if not rating_lines:
            print("Could not find rating line in plan:\n", plan)
            return 0.0
        rating_line = rating_lines[0]
        try:
            rating_str = rating_line.split(":", 1)[1].strip()
            rating_val = float(rating_str)
            if rating_val not in {1.0, 2.0, 3.0, 4.0, 5.0}:
                print("Parsed rating not in allowed set:", rating_val)
                return 0.0
            return rating_val
        except Exception as e:
            print("Error parsing rating from line:", rating_line, "error:", e)
            return 0.0

    # ---------- Stage 3: Final review generation ----------
    def _stage3_final_review(self, persona: str, business, rating: float, plan: str) -> str:
        """
        Turn rating + persona + outline into final 2-4 sentence review text.
        """
        prompt = f"""
You are simulating a real human Yelp user writing a review.

User persona:
{persona}

Business:
{business}

Planned rating: {rating}
Review plan (reasoning and bullet points):
{plan}

Write the FINAL review text that this user would post on Yelp, following these rules:
- 2-4 sentences.
- Match the user's tone and style from the persona.
- Focus on specific details about the business (not generic comments).
- Be consistent with the planned rating and reasoning.

You MUST output ONLY the review text, with no extra labels or explanation.
"""
        review_text = self._call_llm(prompt)
        if len(review_text) > 512:
            review_text = review_text[:512]
        return review_text

    # ---------- Main workflow ----------
    def workflow(self):
        """
        Simulate user behavior with 3-stage pipeline:
        1) Persona inference  2) Rating plan  3) Final review
        """
        try:
            # Basic retrieval
            user_obj = self.interaction_tool.get_user(user_id=self.task['user_id'])
            business_obj = self.interaction_tool.get_item(item_id=self.task['item_id'])

            user = str(user_obj)
            business = str(business_obj)

            # Collect item reviews (for memory and "similar reviews")
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                self.memory(f'review: {review.get("text", "")}')

            # Collect user reviews (for persona)
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            # Stage 1: Persona
            persona = self._stage1_persona(user=user, user_reviews=reviews_user)

            # ---- Memory-based selection of item reviews ----
            use_mem = bool(getattr(self.__class__, "USE_MEMORY", False))

            if use_mem:
                MAX_ITEM_MEM = getattr(self.__class__, "MAX_ITEM_MEM", 25)

                for r in reviews_item[:MAX_ITEM_MEM]:
                    txt = (r.get("text") or "").strip()
                    if not txt:
                        continue
                    stars = r.get("stars", "N/A")
                    self.memory(f"review: item_review item_id={self.task['item_id']} stars={stars} text={txt}")

                retrieved_item_review = self.memory(
                    f"Find the most relevant item_review for this persona.\nPersona: {persona}\nBusiness: {business}\nitem_id={self.task['item_id']}"
                )

                if retrieved_item_review and retrieved_item_review.strip():
                    similar_reviews_text = retrieved_item_review.strip()
                else:
                    # fallback
                    K_sim = 3
                    selected_item_reviews = reviews_item[:K_sim]
                    similar_reviews_text = "\n\n".join(
                        f"- Review {i+1} (stars: {r.get('stars', 'N/A')}): {r.get('text', '')}"
                        for i, r in enumerate(selected_item_reviews)
                    ) or "There are no prior reviews available for this business."
            else:
                # Memory OFF = always use the original baseline-style selection
                K_sim = 3
                selected_item_reviews = reviews_item[:K_sim]
                similar_reviews_text = "\n\n".join(
                    f"- Review {i+1} (stars: {r.get('stars', 'N/A')}): {r.get('text', '')}"
                    for i, r in enumerate(selected_item_reviews)
                ) or "There are no prior reviews available for this business."

            # Stage 2: Rating + plan
            plan = self._stage2_rating_plan(
                persona=persona,
                user=user,
                business=business,
                similar_reviews_text=similar_reviews_text,
            )

            use_reflection = bool(getattr(self.__class__, "USE_REFLECTION", True))

            if use_reflection:
                refined_plan = self._stage2b_reflect_plan(
                    persona=persona,
                    user=user,
                    business=business,
                    plan=plan,
                )

                # Prefer rating from refined plan; fall back to raw plan
                rating = self._parse_rating_from_plan(refined_plan)
                if rating == 0.0:
                    rating = self._parse_rating_from_plan(plan)

                plan_for_review = refined_plan if refined_plan else plan
            else:
                # No reflection: use the raw plan directly
                rating = self._parse_rating_from_plan(plan)
                plan_for_review = plan

            if rating == 0.0:
                rating = 3.0  # last resort neutral

            # Stage 3: Final review text
            final_review = self._stage3_final_review(
                persona=persona,
                business=business,
                rating=rating,
                plan=plan
            )

            return {
                "stars": float(rating),
                "review": final_review
            }

        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 0.0,
                "review": ""
            }

class DummyLLM(LLMBase):
    def __init__(self, model: str = "dummy"):
        super().__init__(model=model)

    def __call__(self, messages, model=None, temperature=0.0, max_tokens=500, stop_strs=None, n=1):
        # Always return something in the expected format
        return "stars: 5.0\nreview: This is a dummy review for testing."

    def get_embedding_model(self):
        return None
class DummyMemory:
    """Minimal no-op memory used to avoid heavy embedding calls during debugging."""

    def __init__(self):
        self.store = []

    def __call__(self, current_situation: str = ""):
        # Just record the text; don't do any embedding or retrieval.
        if current_situation:
            self.store.append(current_situation)
        # Return empty or some trivial string; MemoryDILU callers usually expect a string.
        return ""
    
from typing import List, Dict, Any, Optional
import os

from google import genai
from websocietysimulator.llm import LLMBase


from typing import List

class GeminiEmbeddingModel:
    """Wrapper for Gemini embeddings, compatible with MemoryDILU expectations."""
    def __init__(self, client: genai.Client, model: str = "text-embedding-004"):
        self.client = client
        self.model = model

    # Optional: keep your original single-text helper
    def embed(self, text: str) -> List[float]:
        if not text:
            return []
        resp = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return resp.embeddings[0].values

    # --- NEW: what MemoryDILU expects ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (list of strings)."""
        vectors: List[List[float]] = []
        for t in texts:
            if not t:
                vectors.append([])
                continue
            resp = self.client.models.embed_content(
                model=self.model,
                contents=t,
            )
            vectors.append(resp.embeddings[0].values)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        # You can delegate to embed_documents for consistency
        return self.embed_documents([text])[0]


from dotenv import load_dotenv

class GeminiLLM(LLMBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",   # or "gemini-2.5-flash"
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
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> str:
        """Call Gemini and always return a string (never None)."""
        model_name = model or self.model_name

        # Map messages -> Gemini contents
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            gemini_role = "model" if role == "assistant" else role
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": m.get("content", "")}],
                }
            )

        try:
            # Minimal call: rely on model defaults for temperature / max_tokens etc.
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
            )

            text = getattr(response, "text", None)
            if text is None:
                # Fallback: manually join parts if needed
                if getattr(response, "candidates", None):
                    parts = response.candidates[0].content.parts
                    text = "".join(getattr(p, "text", "") for p in parts)
                else:
                    text = ""

            return text

        except Exception as e:
            print("Error calling Gemini:", repr(e))
            # Return empty string so workflow() can handle it gracefully
            return ""

    def get_embedding_model(self):
        return self._embedding_model

if __name__ == "__main__":
    
    # Set the data
    task_set = "amazon" # "goodreads" or "yelp"
    try:
        use_memory = os.environ.get("USE_MEMORY", "0") == "1"
        use_reflection = os.environ.get("USE_REFLECTION", "1") == "1"

        MySimulationAgent.USE_MEMORY = use_memory
        MySimulationAgent.USE_REFLECTION = use_reflection

        print(f"[Config] USE_MEMORY={use_memory} USE_REFLECTION={use_reflection}")

        simulator = Simulator(data_dir="../big_data", device="gpu", cache=False)
        simulator.set_task_and_groundtruth(task_dir=f"./example/track1/{task_set}/tasks", groundtruth_dir=f"./example/track1/{task_set}/groundtruth")

        # Set the agent and LLM
        simulator.set_agent(MySimulationAgent)
        simulator.set_llm(GeminiLLM(model="gemini-2.5-flash"))
        print("Running simulation...")

        # Run the simulation
        # If you don't set the number of tasks, the simulator will run all tasks.
        outputs = simulator.run_simulation(number_of_tasks=25, enable_threading=False, max_workers=1)
        print("Simulation finished, evaluating...")
        
        # Evaluate the agent
        evaluation_results = simulator.evaluate()       
        with open(f'./results/evaluation_results_track1_{task_set}_mem{int(use_memory)}_refl{int(use_reflection)}.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        # Get evaluation history
        evaluation_history = simulator.get_evaluation_history()
        print("Evaluation results:")
    except Exception as e:
        print("ERROR in main:", repr(e))