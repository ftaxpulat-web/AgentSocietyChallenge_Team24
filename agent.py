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
    """Participant's implementation of SimulationAgent."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            if reviews_user:
                review_similar = self.memory(f'{reviews_user[0]["text"]}')
            else:
                review_similar = ""  # or ask MemoryDILU with some default / skip this part

            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            You MUST output in exactly the following format, with nothing before or after:

            stars: [one of 1.0, 2.0, 3.0, 4.0, 5.0]
            review: [your review in 2–4 sentences]

            Do not output any other text.
            '''
            result = self.reasoning(task_description)

            if not isinstance(result, str) or not result.strip():
                print("LLM returned empty or non-string result:", repr(result))
                return {
                    "stars": 0.0,
                    "review": "",
                }

            lines = result.splitlines()
            stars_lines = [line for line in lines if "stars:" in line]
            review_lines = [line for line in lines if "review:" in line]

            if not stars_lines or not review_lines:
                print("Parse error, raw LLM output:\n", result)
                return {
                    "stars": 0.0,
                    "review": "",
                }

            stars_line = stars_lines[0]
            review_line = review_lines[0]

            try:
                stars = float(stars_line.split(":", 1)[1].strip())
            except Exception as e:
                print("Error parsing stars from line:", stars_line, "error:", e)
                stars = 0.0

            review_text = review_line.split(":", 1)[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]

            return {
                "stars": stars,
                "review": review_text,
            }

        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 0,
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
    
from typing import List, Dict, Any, Optional
import os

from google import genai
from websocietysimulator.llm import LLMBase


class GeminiEmbeddingModel:
    """Tiny wrapper for Gemini embeddings, used by MemoryDILU if needed."""
    def __init__(self, client: genai.Client, model: str = "text-embedding-004"):
        self.client = client
        self.model = model

    def embed(self, text: str) -> List[float]:
        # You can adjust this depending on how MemoryDILU expects to be called
        resp = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        # SDK returns something like resp.embeddings[0].values
        return resp.embeddings[0].values

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
    task_set = "yelp" # "goodreads" or "yelp"
    try:
        simulator = Simulator(data_dir="./tiny_data", device="gpu", cache=False)
        simulator.set_task_and_groundtruth(task_dir=f"./example/track1/{task_set}/tasks", groundtruth_dir=f"./example/track1/{task_set}/groundtruth")

        # Set the agent and LLM
        simulator.set_agent(MySimulationAgent)
        simulator.set_llm(GeminiLLM(model="gemini-2.5-flash"))
        print("Running simulation...")

        # Run the simulation
        # If you don't set the number of tasks, the simulator will run all tasks.
        outputs = simulator.run_simulation(number_of_tasks=5, enable_threading=False, max_workers=1)
        print("Simulation finished, evaluating...")
        
        # Evaluate the agent
        evaluation_results = simulator.evaluate()       
        with open(f'./results/evaluation_results_track1_{task_set}.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        # Get evaluation history
        evaluation_history = simulator.get_evaluation_history()
        print("Evaluation results:")
    except Exception as e:
        print("ERROR in main:", repr(e))