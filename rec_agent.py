import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # plan = self.planning(task_type='Recommendation Task',
        #                      task_description="Please make a plan to query user information, you can choose to query user, item, and review information",
        #                      feedback='',
        #                      few_shot='')
        # print(f"The plan is :{plan}")
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user = ''
        item_list = []
        history_review = ''
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in sub_task['description']:
                keys_to_extract = [
                    'item_id', 'name', 'stars', 'review_count', 'attributes',
                    'title', 'average_rating', 'rating_number', 'description',
                    'ratings_count', 'title_without_series',
                ]

                for item_id in self.task['candidate_list']:
                    item = self.interaction_tool.get_item(item_id=item_id)

                    # Handle missing items gracefully
                    if item is None:
                        print(f"[WARN] get_item returned None for item_id={item_id}, skipping.")
                        continue

                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                    item_list.append(filtered_item)
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        task_description = f'''
        You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        Please rank the more interested items more front in your rank list.
        The information of the above 20 candidate items is as follows: {item_list}.

        Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        '''
        result = self.reasoning(task_description)

        try:
            # print('Meta Output:',result)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
            print('Processed Output:',eval(result))
            # time.sleep(4)
            return eval(result)
        except:
            print('format error')
            return ['']

from typing import List, Dict, Any, Optional
import ast

class DummyEmbeddingModel:
    """
    Minimal embedding model stub so that parts of the framework
    that expect embed_documents/embed_query don't crash.
    It just returns constant vectors.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if texts is None:
            return []
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0]
class DummyLLM(LLMBase):
    """
    Simple dummy LLM for the recommendation agent.
    It ignores true semantics and just returns a ranked list
    of the candidate items extracted from the prompt.
    """

    def __init__(self, model: str = "dummy-rec"):
        super().__init__(model=model)
        self._embedding_model = DummyEmbeddingModel()

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> str:
        # Take the last user message content (the task_description)
        if not messages:
            return "['']"

        last_msg = messages[-1].get("content", "")
        if not isinstance(last_msg, str):
            return "['']"

        # Try to extract the candidate_list from the prompt.
        # The prompt contains a line like:
        # "Now you need to rank the following 20 items: [...] according to their match degree..."
        candidate_list = None
        try:
            # Only look at the part before the "The information of the above 20 candidate items" section
            before_info = last_msg.split(
                "The information of the above 20 candidate items", 1
            )[0]
            start = before_info.find("[")
            end = before_info.find("]", start)
            if start != -1 and end != -1:
                list_str = before_info[start : end + 1]
                candidate_list = ast.literal_eval(list_str)
        except Exception as e:
            print("DummyLLM: error parsing candidate_list:", e)
            candidate_list = None

        if not candidate_list or not isinstance(candidate_list, list):
            # Fallback: return a trivial list so the agent doesn't crash
            return "['']"

        # Dummy ranking: reverse the list (you could also just keep it as-is)
        ranked = list(reversed(candidate_list))

        # The agent expects something like: "['item id1', 'item id2', ...]"
        return repr(ranked)

    def get_embedding_model(self):
        # Return a stub embedding model instead of None,
        # so evaluation/memory code won't crash.
        return self._embedding_model

"""
if __name__ == "__main__":
 
    task_set = "amazon" # "goodreads" or "yelp"
    try:
        # Initialize Simulator
        simulator = Simulator(data_dir="./tiny_data", device="auto", cache=False)

        # Load scenarios
        simulator.set_task_and_groundtruth(task_dir=f"./example/track2/{task_set}/tasks", groundtruth_dir=f"./example/track2/{task_set}/groundtruth")

        # Set your custom agent
        simulator.set_agent(MyRecommendationAgent)

        # Set LLM client
        simulator.set_llm(DummyLLM())

        # Run evaluation
        # If you don't set the number of tasks, the simulator will run all tasks.
        agent_outputs = simulator.run_simulation(number_of_tasks=5, enable_threading=False, max_workers=1)

        # Evaluate the agent
        evaluation_results = simulator.evaluate()
        with open(f'./results/evaluation_results_track2_{task_set}.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        print(f"The evaluation_results is :{evaluation_results}")
    except Exception as e:
        print("ERROR in main:", repr(e))"""

if __name__ == "__main__":
    task_set = "amazon"  # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="./tiny_data", device="auto", cache=False)

    # Load scenarios
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks",
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth",
    )

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    simulator.set_llm(DummyLLM())

    # --- STEP 1: test run_simulation only ---
    print("About to run simulation...")
    agent_outputs = simulator.run_simulation(
        number_of_tasks=5, enable_threading=False, max_workers=1
    )
    print("Simulation finished. Sample output for first task:", agent_outputs[:1])

    # --- STEP 2: test evaluate separately ---
    print("About to evaluate...")
    evaluation_results = simulator.evaluate()
    print("Evaluation finished:", evaluation_results)

    # Save results
    import os
    os.makedirs("./results", exist_ok=True)
    with open(f'./results/evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
